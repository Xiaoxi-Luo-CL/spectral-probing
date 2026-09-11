"""
direction1/transfer/zero_shot.py — Cross-lingual spectral transfer evaluation.

Implements Direction 1 from plan/Plans.md.
See plan/step1_plan.md for experiment design details.

All conditions are evaluated on the SAME target language test set.

Conditions:
  a  zero-shot spectral : apply P(A) + W(A) directly to target B, no training
  b  zero-shot baseline : apply W(A) to raw word-pooled reps of B, no DCT filter
  c  few-shot full      : train P + W from scratch on n examples from B
  d  few-shot fine-tune : load P(A) + W(A) as init, fine-tune both on n examples from B
  e  few-shot frozen-P  : load P(A), freeze it, train only W on n examples from B
  f  few-shot fine-tune : load P(A) frozen, use W(A) as init to fine-tune W on n examples from B
  g  few-shot W         : train only W on n examples from B
  h  few-shot fine-tune : no filter, use W(A) as init to fine-tune W on n examples from B


Usage (run from spectral-probing/):
  python -m direction1.transfer.zero_shot --src_lang en \
   --model CohereLabs/aya-expanse-8b \
    --src_pool None --task relations \
    --target_data tasks/ud-syntax/zh/relations/zh-gsd-dev-relations.csv \
    --condition e --pool_word_embed avg
"""

import argparse
import copy
import json
import logging
import os
import sys
import numpy as np
import torch

from utils.datasets import LabelledDataset
from utils.training import classify_dataset
from models.encoders import PrismEncoder
from models.classifiers import EmbeddingClassifier, LinearProbe
from models.losses import LabelLoss


RESULTS_INDEX = os.path.join(os.path.dirname(__file__), 'results_index.json')


def lookup_source_exp(model, src_lang, task, pool):
    """Return experiment path from results_index.json for the given settings."""
    index_path = os.path.normpath(RESULTS_INDEX)
    assert os.path.exists(index_path), f"results_index.json not found at {index_path}"
    with open(index_path) as f:
        index = json.load(f)
    key = f"{model}|{src_lang}|{task}|{pool}"
    path = index['lookup'].get(key)
    assert path is not None, (
        f"No experiment found for key '{key}'.\n"
        f"Available keys (sample): {list(index['lookup'].keys())[:5]}"
    )
    return path


def parse_arguments():
    p = argparse.ArgumentParser(description='Cross-lingual Spectral Transfer Evaluation')
    # source experiment: either explicit path OR (model + src_lang + task + src_pool)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--source_exp',
                     help='Path to source experiment directory (contains best.pt and config.json)')
    src.add_argument('--src_lang',
                     help='Source language code (e.g. en, zh). '
                          'Used with --model/--task/--src_pool to look up path in results_index.json')
    p.add_argument('--model', default="CohereLabs/aya-expanse-8b", choices=["CohereLabs/aya-expanse-8b", "mistralai/Mistral-7B-v0.1", "bert-base-multilingual-cased", "bigscience/bloom-560m", "FacebookAI/xlm-roberta-base", "Qwen/Qwen2.5-7B"],
                   help='Model identifier (e.g. FacebookAI/xlm-roberta-base). Required when using --src_lang')
    p.add_argument('--src_pool', default=None, choices=['avg', 'last', 'None'],
                   help='pool_word_embed used in source experiment (default: None). Required when using --src_lang')
    p.add_argument('--target_data', required=True,
                   help='Single annotated file for target language B. '
                        'Last --n_test sentences become the test set; '
                        'the remainder is the train pool for conditions c/d.')
    p.add_argument('--n_test', type=int, default=200,
                   help='Sentences reserved as test set from the END of --target_data (default: 200)')
    p.add_argument('--task', required=True, choices=['relations', 'pos'],
                   help='Task name (used for index lookup and output naming)')
    p.add_argument('--condition', required=True, choices=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'],
                   help='a/b=zero-shot; c-e=few-shot from scratch (P+W / frozen-P+W / W-only); f-h=few-shot fine-tune (frozen-P+W(A) / P(A)+W(A) / W(A) no-filter); i=fix W(A), tune P from P(A)')
    # few-shot arguments
    p.add_argument('--n_examples', type=int, default=None,
                   help='Sentences sampled from the train pool for training (required for c/d)')
    # encoder settings
    p.add_argument('--pool_word_embed', default=None, choices=['avg', 'last', None],
                   help='Word-level pooling strategy (default: avg)')
    # output
    p.add_argument('--output_dir', required=False, default=None,
                   help='Directory to save results and logs')
    # training hyperparameters, only for conditions c/d
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--early_stop', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--decay_rate', type=float, default=0.5)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--repeat_labels', action='store_true', default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def sample_dataset(dataset, n, seed=42):
    """Sample n sentences from dataset without replacement."""
    rng = np.random.default_rng(seed)
    total = dataset.get_input_count()
    if n > total:
        logging.error(f"The train pool has only {total} sentences, requested {n}.")
        raise ValueError(f"Requested n_examples={n} exceeds available train pool size of {total}.")
    n = min(n, total)
    idx = rng.choice(total, size=n, replace=False)
    return LabelledDataset(
        [dataset._inputs[i] for i in idx],
        [dataset._labels[i] for i in idx],
    )


def split_target_data(dataset, n_test):
    """
    Reserve the last n_test sentences as the test set.
    Return (train_pool, test_data).
    train_pool is empty for conditions a/b (not used).
    """
    total = dataset.get_input_count()
    n_test = min(n_test, total)
    split = total - n_test
    train_pool = LabelledDataset(dataset._inputs[:split], dataset._labels[:split])
    test_data = LabelledDataset(dataset._inputs[split:], dataset._labels[split:])
    return train_pool, test_data


def split_dataset(dataset, val_ratio=0.2, seed=42):
    """Split LabelledDataset into (train, dev) by val_ratio."""
    rng = np.random.default_rng(seed)
    total = dataset.get_input_count()
    idx = rng.permutation(total)
    n_val = max(1, int(total * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = LabelledDataset(
        [dataset._inputs[i] for i in train_idx],
        [dataset._labels[i] for i in train_idx],
    )
    dev = LabelledDataset(
        [dataset._inputs[i] for i in val_idx],
        [dataset._labels[i] for i in val_idx],
    )
    return train, dev


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def load_checkpoint_objects(source_exp):
    """Load raw dict from source best.pt."""
    model_path = os.path.join(source_exp, 'best.pt')
    assert os.path.exists(model_path), f"No best.pt at {model_path}"
    return torch.load(model_path, map_location='cpu', weights_only=False)


def build_encoder(lm_name, frq_filter, frq_tuning, pool_word_embed,
                  proto_encoder=None, cache=None):
    """
    Instantiate PrismEncoder.
    If proto_encoder is given, reuse its already-loaded LM and tokenizer
    (skips AutoModel.from_pretrained — useful in batch runners).
    cache is a shared dict passed to the encoder for embedding lookup/storage.
    """
    if proto_encoder is not None:
        return PrismEncoder.from_lm(
            lm=proto_encoder._lm,
            tok=proto_encoder._tok,
            lm_name=lm_name,
            frq_filter=frq_filter,
            frq_tuning=frq_tuning,
            pool_word_embed=pool_word_embed,
            cache=cache,
        )
    return PrismEncoder(
        lm_name=lm_name,
        frq_filter=frq_filter,
        frq_tuning=frq_tuning,
        emb_tuning=False,
        pool_word_embed=pool_word_embed,
        cache=cache,
        embed_layer=None,
    )


def build_classifier(encoder, classes, source_W=None):
    """
    Build classifier.
    - source_W given  -> reuse pre-trained linear probe (conditions a, b)
    - source_W=None   -> fresh linear probe (conditions c, d)
    """
    if source_W is not None:
        return EmbeddingClassifier(emb_model=encoder, lbl_model=source_W, classes=classes)
    return LinearProbe(emb_model=encoder, classes=classes)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(classifier, train_data, dev_data, classes, args):
    """
    Train classifier with Adam + ReduceLROnPlateau + early stopping.
    Only updates trainable parameters (filter if frq_tuning=True, always probe W).
    Returns classifier restored to best dev checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    criterion = LabelLoss(classes)
    optimizer = torch.optim.Adam(
        classifier.get_trainable_parameters(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.decay_rate, patience=0
    )

    best_val_loss = float('inf')
    best_lbl_state = None
    best_frq_state = None
    no_improve = 0

    for ep in range(args.epochs):
        classify_dataset(
            classifier, criterion, optimizer, train_data,
            args.batch_size, repeat_labels=args.repeat_labels, mode='train'
        )
        ep_val = classify_dataset(
            classifier, criterion, None, dev_data,
            args.batch_size, repeat_labels=args.repeat_labels, mode='eval'
        )
        val_loss = float(np.mean(ep_val['loss']))
        val_acc = float(np.mean(ep_val['accuracy']))
        scheduler.step(val_loss)
        logging.info(
            f"[Epoch {ep+1}/{args.epochs}] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lbl_state = {
                k: v.detach().clone()
                for k, v in classifier._lbl.state_dict().items()
            }
            # save filter state only when it is being trained (condition c)
            if (classifier._emb._frq_filter is not None
                    and classifier._emb._frq_tuning):
                best_frq_state = classifier._emb._frq_filter.detach().clone()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > args.early_stop:
                logging.info(f"Early stop at epoch {ep+1}.")
                break

    # restore best weights
    if best_lbl_state is not None:
        classifier._lbl.load_state_dict(best_lbl_state)
    if best_frq_state is not None:
        with torch.no_grad():
            classifier._emb._frq_filter.copy_(best_frq_state)

    return classifier


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(classifier, test_data, classes, batch_size, repeat_labels=True):
    """Return (accuracy, loss) on test_data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    classifier.eval()
    criterion = LabelLoss(classes)
    stats = classify_dataset(
        classifier, criterion, None, test_data,
        batch_size, repeat_labels=repeat_labels, mode='eval'
    )
    return float(np.mean(stats['accuracy'])), float(np.mean(stats['loss']))


def check_args(args):
    '''Check if arguments have conflict.'''
    if args.src_pool != args.pool_word_embed:
        logging.warning(
            f"Source experiment uses pool '{args.src_pool}' but --pool_word_embed is set to '{args.pool_word_embed}'."
        )
    if args.repeat_labels:
        assert args.pool_word_embed is None, "Word embedding pooling cannot be used when repeat_labels is True."
    if args.pool_word_embed is not None:
        assert args.repeat_labels == False, "repeat_labels must be False when using word embedding pooling."
    if args.condition in ('c', 'd', 'e', 'f', 'g', 'h', 'i'):
        assert args.n_examples is not None, \
            f"--n_examples is required for condition ({args.condition})"
    else:
        if args.n_examples is not None:
            logging.warning(f"--n_examples is set but will be ignored for zero-shot conditions (a/b).")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_arguments()
    check_args(args)
    # --- resolve source experiment path ---
    if args.source_exp is not None:
        source_exp = args.source_exp
    else:
        assert args.model is not None, "--model is required when using --src_lang"
        pool_key = args.src_pool if args.src_pool is not None else 'None'
        source_exp = lookup_source_exp(args.model, args.src_lang, args.task, pool_key)
        logging.info(f"Resolved source_exp from index: {source_exp}")

    if args.output_dir is None:
        args.output_dir = os.path.join(
            'transfer/results',
            f"{os.path.basename(source_exp)}_{args.condition}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, 'transfer_eval.log'), mode='w'
            ),
            logging.StreamHandler(sys.stdout),
        ],
        format='%(message)s', level=logging.INFO, force=True,
    )

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # --- load source checkpoint ---
    ckpt = load_checkpoint_objects(source_exp)
    lm_name = ckpt['language_model_name']
    source_P = ckpt.get('frequency_filter', None)   # nn.Parameter or None
    source_classes = ckpt['classes']
    source_W = ckpt['classifier']                    # nn.Linear or MLP
    cond = args.condition

    logging.info(f"Source     : {source_exp}")
    logging.info(f"Model      : {lm_name}")
    logging.info(f"Classes    : {len(source_classes)}")
    logging.info(f"Filter     : {'present' if source_P is not None else 'absent'}")
    logging.info(f"Condition  : ({cond})")
    logging.info(f"Pool       : {args.pool_word_embed}")

    # --- split target data into train pool + fixed test set ---
    all_target = LabelledDataset.from_path(args.target_data)
    train_pool, test_data = split_target_data(all_target, args.n_test)
    logging.info(
        f"Target     : {all_target.get_input_count()} total | "
        f"train pool : {train_pool.get_input_count()} | "
        f"test       : {test_data.get_input_count()}"
    )

    # --- build classifier for each condition ---
    if cond == 'a':
        # Zero-shot spectral: P(A) + W(A) -> B, no training
        assert source_P is not None, \
            "Condition (a) requires source experiment to have a learnable filter (auto filter)."
        encoder = build_encoder(
            lm_name, source_P.detach().clone(), frq_tuning=False,
            pool_word_embed=args.pool_word_embed
        )
        classifier = build_classifier(encoder, source_classes, source_W=source_W)

    elif cond == 'b':
        # Zero-shot baseline: W(A) -> B, no filter
        encoder = build_encoder(
            lm_name, frq_filter=None, frq_tuning=False,
            pool_word_embed=args.pool_word_embed
        )
        classifier = build_classifier(encoder, source_classes, source_W=source_W)

    elif cond in ('c', 'd', 'e', 'f', 'g', 'h', 'i'):
        assert train_pool.get_input_count() > 0, \
            f"No train pool left after reserving {args.n_test} test sentences. " \
            f"Use a smaller --n_test or a larger --target_data file."
        sampled = sample_dataset(train_pool, args.n_examples, seed=args.random_seed)
        train_data, dev_data = split_dataset(sampled, val_ratio=0.2, seed=args.random_seed)

        logging.info(
            f"Train sents: {train_data.get_input_count()} | "
            f"Dev sents  : {dev_data.get_input_count()}"
        )

        if cond == 'c':
            # Few-shot full: train P + W from scratch on n examples
            init_filter = PrismEncoder.gen_auto_filter_initialization(512)
            encoder = build_encoder(
                lm_name, init_filter, frq_tuning=True,
                pool_word_embed=args.pool_word_embed
            )
            classifier = build_classifier(encoder, source_classes, source_W=None)

        elif cond == 'd':  # fine-tune both P(A) and W(A)
            assert source_P is not None, \
                "Condition (d) requires source experiment to have a learnable filter."
            encoder = build_encoder(
                lm_name, source_P.detach().clone(), frq_tuning=True,
                pool_word_embed=args.pool_word_embed
            )
            classifier = build_classifier(encoder, source_classes,
                                          source_W=copy.deepcopy(source_W))

        elif cond == 'e':  # frozen P(A), train W from scratch
            assert source_P is not None, \
                "Condition (e) requires source experiment to have a learnable filter."
            encoder = build_encoder(
                lm_name, source_P.detach().clone(), frq_tuning=False,
                pool_word_embed=args.pool_word_embed
            )
            classifier = build_classifier(encoder, source_classes, source_W=None)

        elif cond == 'f':  # frozen P(A), fine-tune W from W(A)
            assert source_P is not None, \
                "Condition (f) requires source experiment to have a learnable filter."
            encoder = build_encoder(
                lm_name, source_P.detach().clone(), frq_tuning=False,
                pool_word_embed=args.pool_word_embed
            )
            classifier = build_classifier(encoder, source_classes,
                                          source_W=copy.deepcopy(source_W))

        elif cond == 'g':  # no filter, train W from scratch
            encoder = build_encoder(
                lm_name, frq_filter=None, frq_tuning=False,
                pool_word_embed=args.pool_word_embed
            )
            classifier = build_classifier(encoder, source_classes, source_W=None)

        else:  # cond == 'h': no filter, fine-tune W from W(A)
            encoder = build_encoder(
                lm_name, frq_filter=None, frq_tuning=False,
                pool_word_embed=args.pool_word_embed
            )
            classifier = build_classifier(encoder, source_classes,
                                          source_W=copy.deepcopy(source_W))

        if cond == 'i':
            # Fix W(A), tune only P from P(A) init
            assert source_P is not None, \
                "Condition (i) requires source experiment to have a learnable filter."
            encoder = build_encoder(
                lm_name, source_P.detach().clone(), frq_tuning=True,
                pool_word_embed=args.pool_word_embed
            )
            frozen_W = copy.deepcopy(source_W)
            for param in frozen_W.parameters():
                param.requires_grad_(False)
            classifier = build_classifier(encoder, source_classes, source_W=frozen_W)

        classifier = train_probe(
            classifier, train_data, dev_data, source_classes, args
        )

    # --- evaluate on target test set ---
    test_acc, test_loss = evaluate(
        classifier, test_data, source_classes, args.batch_size, repeat_labels=args.repeat_labels
    )

    logging.info(f"\n=== Condition ({cond}) Results ===")
    logging.info(f"Test Accuracy : {test_acc:.4f}")
    logging.info(f"Test Loss     : {test_loss:.4f}")

    result = {
        'condition': cond,
        'source_exp': source_exp,
        'target_data': args.target_data,
        'n_test': test_data.get_input_count(),
        'n_examples': args.n_examples,
        'pool_word_embed': args.pool_word_embed,
        'lm_name': lm_name,
        'num_classes': len(source_classes),
        'test_accuracy': test_acc,
        'test_loss': test_loss,
    }

    out_name = f'result_{cond}'
    if args.n_examples is not None:
        out_name += f'_n{args.n_examples}'
    out_path = os.path.join(args.output_dir, f'{out_name}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
