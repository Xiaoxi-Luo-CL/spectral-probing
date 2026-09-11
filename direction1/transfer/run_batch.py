"""
transfer/run_batch.py — Batch cross-lingual transfer runner.

Loads the backbone LM *once*, pre-populates an embedding cache for each target language, then runs all requested conditions and n_examples values without reloading the model.

For large decoder models (aya, Qwen), this eliminates hundreds of expensive AutoModel.from_pretrained() calls per SLURM job.

Cache mechanics
---------------
PrismEncoder.embed() already supports a cache dict keyed by sentence hash.
The cache stores pre-filter, post-word-pool hidden states (detached from the LM graph).  forward() applies the DCT filter on top of the cached values, so:
  - frozen-filter conditions (a, b, e, f, g, h): pure cache hits; LM never
    re-runs during training/eval.
  - trainable-filter conditions (c, d): cache hit for LM outputs; gradients
    flow through P only (not the LM).

Usage (from spectral-probing/):
    python -m direction1.transfer.run_batch --model FacebookAI/xlm-roberta-base --src_lang en --src_pool avg --task relations --target_list direction1/transfer/coverage_targets.json --conditions a,b,c,d,e,f,g,h --n_examples 50,100,500,1000 --output_dir direction1/transfer/results_coverage

Target list JSON format:
    [
      {"path": "tasks/.../de-all-relations.csv", "lang": "de", "n_test": 500},
      {"path": "tasks/.../ml-all-relations.csv", "lang": "ml", "n_test": 100},
      ...
    ]
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
from models.classifiers import LinearProbe
from models.losses import LabelLoss

# Import helpers from zero_shot to avoid duplication
from transfer.zero_shot import (
    lookup_source_exp,
    load_checkpoint_objects,
    build_encoder,
    build_classifier,
    sample_dataset,
    split_target_data,
    split_dataset,
    train_probe,
    evaluate,
    check_args,
)

ZERO_SHOT_CONDS = ('a', 'b')
TRAINING_CONDS = ('c', 'd', 'e', 'f', 'g', 'h')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    p = argparse.ArgumentParser(
        description='Batch cross-lingual spectral transfer — one LM load, many experiments'
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--source_exp',
                     help='Explicit path to source experiment directory')
    src.add_argument('--src_lang',
                     help='Source language (used with --model/--task/--src_pool for index lookup)')
    p.add_argument('--model', default='FacebookAI/xlm-roberta-base',
                   choices=['CohereLabs/aya-expanse-8b',
                            'mistralai/Mistral-7B-v0.1',
                            'bert-base-multilingual-cased',
                            'bigscience/bloom-560m',
                            'FacebookAI/xlm-roberta-base',
                            'Qwen/Qwen2.5-7B'])
    p.add_argument('--src_pool', default='avg', choices=['avg', 'last', 'None'])
    p.add_argument('--task', required=True, choices=['relations', 'pos'])
    p.add_argument('--target_list', required=True,
                   help='JSON file: list of {path, lang, n_test} dicts')
    p.add_argument('--conditions', default='a,b,c,d,e,f,g,h',
                   help='Comma-separated conditions to run (default: all)')
    p.add_argument('--n_examples', default='50,100,500,1000',
                   help='Comma-separated n_examples for training conditions')
    p.add_argument('--pool_word_embed', default='avg', choices=['avg', 'last', 'None', 'none'],
                   help='Word-level pooling (default: avg)')
    p.add_argument('--output_dir', default='direction1/transfer/results_coverage')
    # training hyperparameters
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--early_stop', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--decay_rate', type=float, default=0.5)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--repeat_labels', action='store_true', default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------

def result_path(output_dir, model, src_lang, tgt_lang, task, cond, n=None, seed=42):
    model_short = model.split('/')[-1]
    cond_dir = f'cond{cond}' + (f'_n{n}' if n is not None else '')
    result_name = f'result_{cond}' + (f'_n{n}' if n is not None else '') + '.json'
    return os.path.join(
        output_dir,
        model_short,
        f'{src_lang}2{tgt_lang}',
        task,
        cond_dir,
        f'seed{seed}',
        result_name,
    )


def save_result(path, result_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    logging.info(f'  Saved -> {path}')


def warmup_cache(proto_encoder, source_classes, all_target, batch_size, cache):
    """
    Run all target sentences through the LM once to populate the shared cache.
    Uses a dummy LinearProbe so we can call classify_dataset with the normal scode path (which calls encoder.embed() -> populates cache).
    """
    proto_encoder._cache = cache
    proto_encoder._frq_filter = None   # no filter during warmup
    proto_encoder._frq_tuning = False
    dummy_classifier = LinearProbe(emb_model=proto_encoder, classes=source_classes)
    if torch.cuda.is_available():
        dummy_classifier = dummy_classifier.to(torch.device('cuda'))
    criterion = LabelLoss(source_classes)
    classify_dataset(
        dummy_classifier, criterion, None,
        all_target, batch_size, mode='eval',
    )
    logging.info(f'  Cache warmed up: {len(cache)} entries')


def build_condition_classifier(cond, lm_name, source_P, source_W, source_classes,
                               pool_word_embed, proto_encoder, cache):
    """Build the classifier for a given condition, reusing the shared LM."""
    if cond == 'a':
        assert source_P is not None, \
            'Condition (a) requires a source spectral filter.'
        enc = build_encoder(lm_name, source_P.detach().clone(), frq_tuning=False,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes, source_W=source_W)

    elif cond == 'b':
        enc = build_encoder(lm_name, frq_filter=None, frq_tuning=False,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes, source_W=source_W)

    elif cond == 'c':
        init_filter = PrismEncoder.gen_auto_filter_initialization(512)
        enc = build_encoder(lm_name, init_filter, frq_tuning=True,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes, source_W=None)

    elif cond == 'd':  # fine-tune both P(A) and W(A)
        assert source_P is not None, \
            'Condition (d) requires a source spectral filter.'
        enc = build_encoder(lm_name, source_P.detach().clone(), frq_tuning=True,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes,
                                source_W=copy.deepcopy(source_W))

    elif cond == 'e':  # frozen P(A), train W from scratch
        assert source_P is not None, \
            'Condition (e) requires a source spectral filter.'
        enc = build_encoder(lm_name, source_P.detach().clone(), frq_tuning=False,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes, source_W=None)

    elif cond == 'f':  # frozen P(A), fine-tune W from W(A)
        assert source_P is not None, \
            'Condition (f) requires a source spectral filter.'
        enc = build_encoder(lm_name, source_P.detach().clone(), frq_tuning=False,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes,
                                source_W=copy.deepcopy(source_W))

    elif cond == 'g':  # no filter, train W from scratch
        enc = build_encoder(lm_name, frq_filter=None, frq_tuning=False,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes, source_W=None)

    else:  # cond == 'h': no filter, fine-tune W from W(A)
        enc = build_encoder(lm_name, frq_filter=None, frq_tuning=False,
                            pool_word_embed=pool_word_embed,
                            proto_encoder=proto_encoder, cache=cache)
        return build_classifier(enc, source_classes,
                                source_W=copy.deepcopy(source_W))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    pool_word_embed = None if args.pool_word_embed in ('None', 'none') else args.pool_word_embed
    conditions = [c.strip() for c in args.conditions.split(',')]
    n_examples = [int(n) for n in args.n_examples.split(',')]

    logging.basicConfig(
        handlers=[logging.StreamHandler(sys.stdout)],
        format='%(message)s', level=logging.INFO, force=True,
    )

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # --- resolve source experiment ---
    if args.source_exp is not None:
        source_exp = args.source_exp
    else:
        pool_key = args.src_pool if args.src_pool != 'None' else 'None'
        source_exp = lookup_source_exp(args.model, args.src_lang, args.task, pool_key)
    logging.info(f'Source exp : {source_exp}')

    # --- load source checkpoint (weights only — no LM) ---
    ckpt = load_checkpoint_objects(source_exp)
    lm_name = ckpt['language_model_name']
    source_P = ckpt.get('frequency_filter', None)
    source_classes = ckpt['classes']
    source_W = ckpt['classifier']
    logging.info(f'Model      : {lm_name}')
    logging.info(f'Classes    : {len(source_classes)}')
    logging.info(f'Filter     : {"present" if source_P is not None else "absent"}')

    # --- load target list ---
    with open(args.target_list) as f:
        targets = json.load(f)
    logging.info(f'Targets    : {len(targets)} languages')
    logging.info(f'Conditions : {conditions}')
    logging.info(f'n_examples : {n_examples}')

    # --- load LM once (create proto encoder) ---
    logging.info('\nLoading backbone LM (this happens once) ...')
    proto_encoder = PrismEncoder(
        lm_name=lm_name,
        frq_filter=None,
        frq_tuning=False,
        emb_tuning=False,
        pool_word_embed=pool_word_embed,
        cache={},
        embed_layer=None,
    )
    logging.info('LM loaded.\n')

    # --- iterate over target languages ---
    for tgt_cfg in targets:
        tgt_path = tgt_cfg['path']
        tgt_lang = tgt_cfg['lang']
        n_test = tgt_cfg['n_test']

        logging.info(f'=== Target: {tgt_lang}  (n_test={n_test}) ===')

        # load and split target data
        all_target = LabelledDataset.from_path(tgt_path)
        train_pool, test_data = split_target_data(all_target, n_test)
        logging.info(
            f'  Total {all_target.get_input_count()} sents | '
            f'train pool {train_pool.get_input_count()} | '
            f'test {test_data.get_input_count()}'
        )
        
        # one shared embedding cache per target language
        shared_cache = {}
        warmup_cache(proto_encoder, source_classes, all_target,
                     args.batch_size, shared_cache)

        for cond in conditions:
            n_iter = n_examples if cond in TRAINING_CONDS else [None] # run only once with [None]

            for n in n_iter:
                out = result_path(args.output_dir, args.model, args.src_lang,
                                  tgt_lang, args.task, cond, n,
                                  seed=args.random_seed)
                if os.path.exists(out):
                    logging.info(f'  [skip] cond={cond} n={n}  (result exists)')
                    continue

                logging.info(f'  cond={cond}  n={n}')

                # skip conditions that require source_P when it is absent
                if cond in ('a', 'd', 'e', 'f') and source_P is None:
                    logging.warning(
                        f'  [skip] cond={cond} requires source spectral filter '
                        f'(source_P=None for this model).'
                    )
                    continue

                try:
                    classifier = build_condition_classifier(
                        cond, lm_name, source_P, source_W, source_classes,
                        pool_word_embed, proto_encoder, shared_cache,
                    )
                except AssertionError as e:
                    logging.warning(f'  [skip] {e}')
                    continue

                if cond in TRAINING_CONDS:
                    if train_pool.get_input_count() == 0:
                        logging.warning(f'  [skip] empty train pool')
                        continue
                    try:
                        sampled = sample_dataset(train_pool, n, seed=args.random_seed)
                    except ValueError as e:
                        logging.error(f'  [skip] {e}')
                        continue
                    train_data, dev_data = split_dataset(
                        sampled, val_ratio=0.2, seed=args.random_seed)
                    classifier = train_probe(
                        classifier, train_data, dev_data,
                        source_classes, args,
                    )

                test_acc, test_loss = evaluate(
                    classifier, test_data, source_classes,
                    args.batch_size, repeat_labels=False,
                )
                logging.info(f'  -> acc={test_acc:.4f}  loss={test_loss:.4f}')

                result = {
                    'condition':    cond,
                    'source_exp':   source_exp,
                    'target_data':  tgt_path,
                    'tgt_lang':     tgt_lang,
                    'n_test':       test_data.get_input_count(),
                    'n_examples':   n,
                    'pool_word_embed': pool_word_embed,
                    'lm_name':      lm_name,
                    'num_classes':  len(source_classes),
                    'random_seed':  args.random_seed,
                    'test_accuracy': test_acc,
                    'test_loss':    test_loss,
                }
                save_result(out, result)

        logging.info('')  # blank line between targets


if __name__ == '__main__':
    main()
