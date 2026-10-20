"""
direction1/transfer/zero_shot.py — Cross-lingual spectral transfer evaluation.

Implements Direction 1 from plan/Plans.md.
See plan/step1_plan.md for experiment design details.

All settings are evaluated on the SAME target language test set.

A setting combines how the target probe W_B and filter gamma_B are obtained from the
source pair (W_A, gamma_A), named W<w>_G<g>:
  W: froze   = W_A as is         tune = fine-tune from W_A   scratch = train from scratch
  G: none    = no filter         froze = gamma_A as is
     tune    = fine-tune from gamma_A    scratch = train from a fresh filter (source recipe)
plus `plain`: a source probe trained WITHOUT any filter, applied as is (plain probing).
Settings that train nothing (W froze with G none/froze, and plain) are zero-shot.
Old letters: a=Wfroze_Gfroze b=Wfroze_Gnone c=Wscratch_Gscratch d=Wtune_Gtune
e=Wscratch_Gfroze f=Wtune_Gfroze g=Wscratch_Gnone h=Wtune_Gnone i=Wfroze_Gtune.


Usage (run from spectral-probing/):
  python -m direction1.transfer.zero_shot --src_lang en \
   --model CohereLabs/aya-expanse-8b \
    --src_pool None --task relations \
    --target_data tasks/ud-syntax/zh/relations/zh-gsd-dev-relations.csv \
    --setting Wscratch_Gfroze --n_examples 100 --pool_word_embed avg
"""

import argparse
import copy
import json
import logging
import re
import os
import sys
import numpy as np
import torch

from utils.datasets import LabelledDataset
from utils.training import classify_dataset, weighted_accuracy
from models.encoders import PrismEncoder
from models.classifiers import EmbeddingClassifier, LinearProbe
from models.losses import LabelLoss


RESULTS_INDEX = os.path.join(os.path.dirname(__file__), 'results_index.json')
W_OPTS = ('froze', 'tune', 'scratch')
G_OPTS = ('none', 'froze', 'tune', 'scratch')
PLAIN = 'plain'
GRID = [f'W{w}_G{g}' for w in W_OPTS for g in G_OPTS]
# zero-shot settings train nothing, so their result does not depend on the seed
ZERO_SHOT_SETTINGS = ['Wfroze_Gnone', 'Wfroze_Gfroze', PLAIN]
TRAIN_SETTINGS = [s for s in GRID if s not in ZERO_SHOT_SETTINGS]
SETTINGS = ZERO_SHOT_SETTINGS + TRAIN_SETTINGS
SCHEDULER_THRESHOLD = 1e-4  # PyTorch's ReduceLROnPlateau default, kept when --min_delta is 0
RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'results'))


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
    if not os.path.isdir(path) and '/results/' in path:
        # the index predates moving results/ under direction1/; keep its relative layout
        path = os.path.join(RESULTS_DIR, path.split('/results/', 1)[1])
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
    p.add_argument('--model', default='CohereLabs/aya-expanse-8b',
                   choices=['CohereLabs/aya-expanse-8b', 'mistralai/Mistral-7B-v0.1', 'bert-base-multilingual-cased',
                            'bigscience/bloom-560m', 'FacebookAI/xlm-roberta-base', 'Qwen/Qwen2.5-7B'],
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
    p.add_argument('--setting', required=True, choices=SETTINGS, help='W<w>_G<g> or plain; see module docstring')
    p.add_argument('--plain_source_exp', default=None, help='filter-free source probe, required for --setting plain')
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
    p.add_argument('--min_delta', type=float, default=0.0,
                   help='relative dev-loss drop an epoch needs to count as an improvement, for early '
                        'stopping and LR decay (default 0: any drop counts)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--decay_rate', type=float, default=0.5, help='lr factor on a dev-loss plateau; 1 = constant lr')
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


def split_budget(dataset, pool_size, min_test, max_test, test_tail=0):
    """Reserve the first sentences as the train pool and a later block as the test set.

    With test_tail > 0 (the file ends with an official UD test split of that size, after train and
    dev), the test set is that split; otherwise it is the sentences right after the pool, and the
    pool shrinks below pool_size when needed to leave min_test of them. The test set is capped at
    max_test. Return (train_pool, test_data).
    """
    total = dataset.get_input_count()
    n_pool = min(pool_size, total - (test_tail or min_test))
    start = total - test_tail if test_tail else n_pool
    end = min(total, start + max_test)
    train_pool = LabelledDataset(dataset._inputs[:n_pool], dataset._labels[:n_pool])
    test_data = LabelledDataset(dataset._inputs[start:end], dataset._labels[start:end])
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


def scratch_filter(source_exp, size):
    """A fresh learnable filter drawn exactly as the source filter was before its training."""
    with open(os.path.join(source_exp, 'config.json')) as f:
        cfg = json.load(f)
    return PrismEncoder.gen_auto_filter_initialization(size, cfg['frq_init'], cfg['frq_noise'], cfg['frq_scale'])


def freeze(module):
    """Turn off gradients for every parameter of module and return it."""
    for prm in module.parameters():
        prm.requires_grad_(False)
    return module


def build_encoder(lm_name, frq_filter, frq_tuning, pool_word_embed,
                  proto_encoder=None, cache=None, frq_logits=True):
    """
    Instantiate PrismEncoder.
    If proto_encoder is given, reuse its already-loaded LM and tokenizer
    (skips AutoModel.from_pretrained — useful in batch runners).
    cache is a shared dict passed to the encoder for embedding lookup/storage.
    frq_logits is True because every filter reaching here is a learned logit (the source
    P(A) or a fresh auto init), so a frozen one must still pass through the training sigmoid.
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
            frq_logits=frq_logits,
        )
    return PrismEncoder(
        lm_name=lm_name,
        frq_filter=frq_filter,
        frq_tuning=frq_tuning,
        emb_tuning=False,
        pool_word_embed=pool_word_embed,
        cache=cache,
        embed_layer=None,
        frq_logits=frq_logits,
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


def run_info(exp):
    """(model, language, task, word pooling, filter) of a classify.py run directory."""
    with open(os.path.join(exp, 'config.json')) as f:
        cfg = json.load(f)
    pool = cfg.get('pool_word_embed')
    if 'pool_word_embed' not in cfg:  # older configs keep it only in the logged arguments
        with open(os.path.join(exp, 'classify.log')) as f:
            match = re.search(r"pool_word_embed=('?\w+'?)", f.readline())
        pool = match.group(1).strip("'") if match else None
    return cfg['lm_name'], cfg['language'], cfg['task_name'], None if pool == 'None' else pool, cfg['filter']


def check_plain(plain, plain_exp, source, source_exp):
    """Fail unless plain_exp is a filter-free probe trained for the same model, data and pooling as source_exp."""
    assert 'frequency_filter' not in plain, f'{plain_exp} has a frequency filter'
    p_info, s_info = run_info(plain_exp), run_info(source_exp)
    assert p_info[4] == 'nofilter()', f'{plain_exp} was trained with filter {p_info[4]}'
    assert p_info[:4] == s_info[:4], f'plain probe {p_info[:4]} does not match source {s_info[:4]}'
    assert list(plain['classes']) == list(source['classes']), 'plain and source probes use different label sets'


def split_setting(name):
    """(w, g) options of a setting name; plain uses a frozen probe and no filter."""
    if name == PLAIN:
        return 'froze', 'none'
    w, g = name.split('_')
    return w[1:], g[1:]


def needs_training(name):
    """True when the setting has a trainable probe or filter."""
    w, g = split_setting(name)
    return w != 'froze' or g in ('tune', 'scratch')


def build_setting(name, lm_name, source, source_exp, pool_word_embed, plain=None, proto_encoder=None, cache=None,
                  classes=None):
    """Classifier and label set for one setting.

    source / plain are checkpoint dicts (best.pt) of the filtered source run and of the
    filter-free source run; probes are always copied so the checkpoints stay untouched.
    classes, when given, replaces the source label set for a fresh probe (W scratch); a probe
    taken from the source keeps the source labels its weights were trained for.
    """
    w, g = split_setting(name)
    ckpt = plain if name == PLAIN else source
    assert ckpt is not None, f'Setting {name} needs a filter-free source probe (plain).'
    P = source.get('frequency_filter')

    assert (g == 'none' or P is not None), f'Setting {name} needs a source spectral filter.'
    if g == 'none':
        frq, tuning = None, False
    elif g == 'froze':
        frq, tuning = P.detach().clone(), False
    elif g == 'tune':
        frq, tuning = P.detach().clone(), True
    else:  # scratch
        frq, tuning = scratch_filter(source_exp, len(P)), True
    enc = build_encoder(lm_name, frq, tuning, pool_word_embed, proto_encoder=proto_encoder, cache=cache)
    W = None if w == 'scratch' else copy.deepcopy(ckpt['classifier'])
    if w == 'froze':
        freeze(W)
    labels = classes if w == 'scratch' and classes is not None else ckpt['classes']
    return build_classifier(enc, labels, source_W=W), labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(classifier, train_data, dev_data, classes, args):
    """
    Train classifier with Adam + ReduceLROnPlateau (none when decay_rate is 1) + early stopping.
    Only updates trainable parameters (filter if frq_tuning=True, always probe W).
    Returns classifier restored to best dev checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    criterion = LabelLoss(classes)
    optimizer = torch.optim.Adam(
        classifier.get_trainable_parameters(), lr=args.learning_rate
    )
    # decay_rate 1 keeps the learning rate constant (ReduceLROnPlateau rejects a factor of 1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.decay_rate, patience=0,
        threshold=args.min_delta or SCHEDULER_THRESHOLD, threshold_mode='rel'
    ) if args.decay_rate < 1 else None

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
        val_acc = weighted_accuracy(ep_val)
        if scheduler is not None:
            scheduler.step(val_loss)
        logging.info(
            f"[Epoch {ep+1}/{args.epochs}] val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # tiny drops (below min_delta, relative) neither reset early stopping nor replace the checkpoint
        if val_loss < best_val_loss * (1 - args.min_delta):
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
    """Return (accuracy, loss, batch-mean accuracy) on test_data.

    accuracy weights each batch by the labels it scored. The batch mean is the old
    per-batch average, kept only so reruns can be audited against earlier results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    classifier.eval()
    criterion = LabelLoss(classes)
    stats = classify_dataset(
        classifier, criterion, None, test_data,
        batch_size, repeat_labels=repeat_labels, mode='eval'
    )
    return weighted_accuracy(stats), float(np.mean(stats['loss'])), float(np.mean(stats['accuracy']))


def check_args(args):
    '''Check if arguments have conflict.'''
    if args.src_pool != args.pool_word_embed:
        logging.warning(
            f"Source experiment uses pool '{args.src_pool}' but --pool_word_embed is set to '{args.pool_word_embed}'."
        )
    if args.repeat_labels:
        assert args.pool_word_embed is None, "Word embedding pooling cannot be used when repeat_labels is True."
    if args.pool_word_embed is not None:
        assert not args.repeat_labels, "repeat_labels must be False when using word embedding pooling."
    if needs_training(args.setting):
        assert args.n_examples is not None, \
            f"--n_examples is required for setting ({args.setting})"
    else:
        if args.n_examples is not None:
            logging.warning('--n_examples is set but will be ignored for zero-shot settings.')

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
        args.output_dir = os.path.join('transfer/results', f"{os.path.basename(source_exp)}_{args.setting}")
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'transfer_eval.log'), mode='w'),
            logging.StreamHandler(sys.stdout),
        ],
        format='%(message)s', level=logging.INFO, force=True,
    )

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # --- load source checkpoints (filtered, and filter-free for the plain setting) ---
    ckpt = load_checkpoint_objects(source_exp)
    plain = load_checkpoint_objects(args.plain_source_exp) if args.plain_source_exp else None
    if plain is not None:
        check_plain(plain, args.plain_source_exp, ckpt, source_exp)
    lm_name = ckpt['language_model_name']
    setting = args.setting

    logging.info(f"Source     : {source_exp}")
    logging.info(f"Model      : {lm_name}")
    logging.info(f"Classes    : {len(ckpt['classes'])}")
    logging.info(f"Filter     : {'present' if ckpt.get('frequency_filter') is not None else 'absent'}")
    logging.info(f"Setting    : {setting}")
    logging.info(f"Pool       : {args.pool_word_embed}")

    # --- split target data into train pool + fixed test set ---
    all_target = LabelledDataset.from_path(args.target_data)
    train_pool, test_data = split_target_data(all_target, args.n_test)
    logging.info(
        f"Target: {all_target.get_input_count()} | train pool: {train_pool.get_input_count()} | "
        f"test: {test_data.get_input_count()}")

    classifier, classes = build_setting(setting, lm_name, ckpt, source_exp, args.pool_word_embed, plain=plain)
    if needs_training(setting):
        assert train_pool.get_input_count() > 0, \
            f"No train pool left after reserving {args.n_test} test sentences. " \
            f"Use a smaller --n_test or a larger --target_data file."
        sampled = sample_dataset(train_pool, args.n_examples, seed=args.random_seed)
        train_data, dev_data = split_dataset(sampled, val_ratio=0.2, seed=args.random_seed)
        logging.info(
            f"Train sents: {train_data.get_input_count()} | "
            f"Dev sents  : {dev_data.get_input_count()}"
        )
        classifier = train_probe(classifier, train_data, dev_data, classes, args)

    # --- evaluate on target test set ---
    test_acc, test_loss, test_acc_batch = evaluate(
        classifier, test_data, classes, args.batch_size, repeat_labels=args.repeat_labels
    )

    logging.info(f"\n=== Setting {setting} Results ===")
    logging.info(f"Test Accuracy : {test_acc:.4f}")
    logging.info(f"Test Loss     : {test_loss:.4f}")

    result = {
        'setting': setting,
        'source_exp': args.plain_source_exp if setting == PLAIN else source_exp,
        'target_data': args.target_data,
        'n_test': test_data.get_input_count(),
        'n_examples': args.n_examples,
        'pool_word_embed': args.pool_word_embed,
        'lm_name': lm_name,
        'num_classes': len(classes),
        'test_accuracy': test_acc,
        'test_accuracy_batch_mean': test_acc_batch,
        'test_loss': test_loss,
    }

    out_name = f'result_{setting}'
    if args.n_examples is not None:
        out_name += f'_n{args.n_examples}'
    out_path = os.path.join(args.output_dir, f'{out_name}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
