"""
transfer/run_batch.py — Batch cross-lingual transfer runner.

Loads the backbone LM *once*, pre-populates an embedding cache for each target language,
then runs all requested settings and n_examples values without reloading the model.

For large decoder models (aya, Qwen), this eliminates hundreds of expensive AutoModel.from_pretrained() calls per SLURM job.

Cache mechanics
---------------
PrismEncoder.embed() already supports a cache dict keyed by sentence hash.
The cache stores pre-filter, post-word-pool hidden states (detached from the LM graph).  forward() applies the DCT filter on top of the cached values, so:
  - settings with a frozen or no filter: pure cache hits; LM never
    re-runs during training/eval.
  - settings with a trainable filter: cache hit for LM outputs; gradients
    flow through P only (not the LM).

Usage (from spectral-probing/):
    python -m direction1.transfer.run_batch --model FacebookAI/xlm-roberta-base --src_lang en --src_pool avg \
        --task relations --target_list direction1/transfer/coverage_targets.json --settings all \
        --n_examples 100,500,1000 --largest_n_only --output_dir direction1/transfer/results_coverage

Target list JSON format:
    [
      {"path": "tasks/.../de-all-relations.csv", "lang": "de", "n_test": 500},
      {"path": "tasks/.../ml-all-relations.csv", "lang": "ml", "n_test": 100},
      ...
    ]
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

from utils.datasets import LabelledDataset
from utils.training import classify_dataset
from models.encoders import PrismEncoder
from models.classifiers import LinearProbe
from models.losses import LabelLoss

# Import helpers from zero_shot to avoid duplication
from direction1.transfer.zero_shot import (
    lookup_source_exp,
    load_checkpoint_objects,
    build_setting,
    check_plain,
    needs_training,
    run_info,
    PLAIN,
    SETTINGS,
    TRAIN_SETTINGS,
    ZERO_SHOT_SETTINGS,
    sample_dataset,
    split_budget,
    split_target_data,
    split_dataset,
    train_probe,
    evaluate,
)
from direction1.transfer.prepare_coverage_data import load_ud_module


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
    p.add_argument('--settings', default='all',
                   help='comma-separated W<w>_G<g> / plain settings (see zero_shot.py), or one of '
                        '"all", "zero_shot" (seed-independent) or "train"')
    p.add_argument('--n_examples', default='100,500,1000',
                   help='Comma-separated n_examples for training conditions')
    p.add_argument('--largest_n_only', action='store_true', default=False,
                   help='per target, run only the largest n_examples that fits its train pool')
    p.add_argument('--plain_sources', default=None,
                   help='JSON {"<model>|<lang>|<task>|<pool>": source_exp} of probes trained without a '
                        'filter, used by the plain setting, which is skipped when no entry exists')
    p.add_argument('--pool_word_embed', default='avg', choices=['avg', 'last', 'None', 'none'],
                   help='Word-level pooling (default: avg)')
    p.add_argument('--output_dir', default='direction1/transfer/results_coverage_v3')
    # training hyperparameters
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--early_stop', type=int, default=2)
    p.add_argument('--min_delta', type=float, default=1e-3,
                   help='relative dev-loss drop an epoch needs to count as an improvement (0: any drop counts)')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--embed_batch_size', type=int, default=None,
                   help='sentences per LM forward pass when filling the embedding cache (default: '
                        '--batch_size); lower it to avoid GPU OOM on 7-8B models. Probe training '
                        'reads the cache, so its batch size is unaffected')
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--lr_grid', default=None,
                   help='comma-separated learning rates; each trained probe is fit once per rate from the same '
                        'seed, and the one with the lowest dev loss is kept (default: --learning_rate only)')
    p.add_argument('--decay_rate', type=float, default=0.5, help='lr factor on a dev-loss plateau; 1 = constant lr')
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--repeat_labels', action='store_true', default=False)
    # fixed-budget split: the first sentences form the train pool, the following ones the test set
    p.add_argument('--pool_size', type=int, default=None,
                   help='train pool size (default: none, the target list n_test sets the split)')
    p.add_argument('--min_test', type=int, default=100, help='with --pool_size: fewest test sentences')
    p.add_argument('--max_test', type=int, default=2000, help='with --pool_size: most test sentences')
    p.add_argument('--target_labels', action='store_true', default=False,
                   help='fresh probes predict every coarse UD relation instead of the source label set, '
                        'so all sources are scored on the same target tokens')
    return p.parse_args()


def result_path(output_dir, model, src_lang, tgt_lang, task, setting, n=None, seed=42):
    model_short = model.split('/')[-1]
    cond_dir = setting + (f'_n{n}' if n is not None else '')
    result_name = f'result_{setting}' + (f'_n{n}' if n is not None else '') + '.json'
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
    # concurrent seed jobs create the same directories on NFS, where a directory just made can look
    # missing to other nodes until their attribute cache expires (it outlasted 10 s); retry for ~2 min
    for attempt in range(8):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            break
        except FileNotFoundError:
            if attempt == 7:
                raise
            time.sleep(2 ** attempt)
    logging.info(f'  Saved -> {path}')


def reseed(seed):
    """Reset the torch and numpy generators, so probe init and minibatch order repeat across settings."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_best_lr(build, train_data, dev_data, args):
    """Fit a fresh probe for every rate in args.lr_grid and keep the lowest dev loss.

    build() returns a new (classifier, classes); each fit starts from the same seed.
    Returns (classifier, classes, chosen lr, {lr: dev loss}).
    """
    best, dev_loss = None, {}
    for lr in [float(x) for x in args.lr_grid.split(',')]:
        reseed(args.random_seed)
        classifier, classes = build()
        reseed(args.random_seed)
        classifier = train_probe(classifier, train_data, dev_data, classes,
                                 argparse.Namespace(**{**vars(args), 'learning_rate': lr}))
        dev_loss[lr] = evaluate(classifier, dev_data, classes, args.batch_size, repeat_labels=False)[1]
        logging.info(f'  lr={lr:g}  dev_loss={dev_loss[lr]:.4f}')
        if best is None or dev_loss[lr] < dev_loss[best[2]]:
            best = (classifier, classes, lr)
    return (*best, dev_loss)


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


def main():
    args = parse_arguments()

    pool_word_embed = None if args.pool_word_embed in ('None', 'none') else args.pool_word_embed
    groups = {'all': SETTINGS, 'zero_shot': ZERO_SHOT_SETTINGS, 'train': TRAIN_SETTINGS}
    settings = groups.get(args.settings) or [c.strip() for c in args.settings.split(',')]
    unknown = [c for c in settings if c not in SETTINGS]
    assert not unknown, f'unknown settings {unknown}; choose from {SETTINGS}'
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
    # the source run is the authority on model, language, task and pooling: with --source_exp
    # they are taken from it, otherwise the supplied values must agree with the looked-up run
    lm, lang, task, pool, _ = run_info(source_exp)
    if args.source_exp is not None:  # take them from the run; only --task is supplied
        args.model, args.src_lang, pool_word_embed = lm, lang, pool
        assert args.task == task, f'--task {args.task} does not match source run task {task}'
    else:  # the index lookup must have returned a run matching the request
        given = (args.model, args.src_lang, args.task, pool_word_embed)
        assert given == (lm, lang, task, pool), f'arguments {given} do not match source run {(lm, lang, task, pool)}'

    # load source checkpoint (weights only, no LM
    ckpt = load_checkpoint_objects(source_exp)
    lm_name = ckpt['language_model_name']
    source_P = ckpt.get('frequency_filter', None)
    source_classes = ckpt['classes']
    logging.info(f'Model      : {lm_name}')
    logging.info(f'Classes    : {len(source_classes)}')
    logging.info(f'Filter     : {"present" if source_P is not None else "absent"}')

    # filter-free source probe for the plain setting
    plain = None
    if args.plain_sources and PLAIN in settings:
        key = f'{args.model}|{args.src_lang}|{args.task}|{pool_word_embed}'
        plain_exp = json.load(open(args.plain_sources)).get(key)
        if plain_exp:
            plain = load_checkpoint_objects(plain_exp)
            plain['source_exp'] = plain_exp
            check_plain(plain, plain_exp, ckpt, source_exp)
        logging.info(f'Plain probe: {plain_exp or "none for " + key + " (plain skipped)"}')

    # --- load target list ---
    with open(args.target_list) as f:
        targets = json.load(f)
    logging.info(f'Targets    : {len(targets)} languages')
    logging.info(f'Settings   : {settings}')
    logging.info(f'n_examples : {n_examples}')
    # one label set for every fresh probe: all coarse UD relations (labels in the CSVs carry no subtypes)
    target_classes = sorted({r.split(':')[0] for r in load_ud_module().UD_RELATIONS}) if args.target_labels else None
    if target_classes:
        logging.info(f'Fresh probes predict {len(target_classes)} coarse UD relations')

    # --- load LM once (create proto encoder) ---
    logging.info('\nLoading backbone LM ...')
    proto_encoder = PrismEncoder(lm_name=lm_name, frq_filter=None, frq_tuning=False,
                                 emb_tuning=False, pool_word_embed=pool_word_embed, cache={}, embed_layer=None)
    logging.info('LM loaded.\n')

    # iterate over target languages
    for tgt_cfg in targets:
        tgt_path = tgt_cfg['path']
        tgt_lang = tgt_cfg['lang']
        n_test = tgt_cfg.get('n_test')  # unused with --pool_size

        logging.info(f' Target: {tgt_lang}  (n_test={n_test}) ===')

        # load and split target data
        all_target = LabelledDataset.from_path(tgt_path)
        if args.pool_size:
            train_pool, test_data = split_budget(all_target, args.pool_size, args.min_test, args.max_test,
                                                 tgt_cfg.get('official_test', 0))
        else:
            train_pool, test_data = split_target_data(all_target, n_test)
        logging.info(f'  Total {all_target.get_input_count()} sents | '
                     f'train pool {train_pool.get_input_count()} | '
                     f'test {test_data.get_input_count()}')
        # one shared embedding cache per target language
        shared_cache = {}
        warmup_cache(proto_encoder, source_classes, all_target,
                     args.embed_batch_size or args.batch_size, shared_cache)

        fit = [n for n in n_examples if n <= train_pool.get_input_count()]
        train_ns = ([max(fit)] if fit else []) if args.largest_n_only else n_examples
        for setting in settings:
            n_iter = train_ns if needs_training(setting) else [None]  # zero-shot runs once

            for n in n_iter:
                out = result_path(args.output_dir, args.model, args.src_lang,
                                  tgt_lang, args.task, setting, n, seed=args.random_seed)
                if os.path.exists(out):
                    logging.info(f'  [skip] {setting} n={n}  (result exists)')
                    continue
                logging.info(f'  {setting}  n={n}')

                def build():
                    """A fresh classifier and label set for this setting."""
                    return build_setting(setting, lm_name, ckpt, source_exp, pool_word_embed, plain=plain,
                                         proto_encoder=proto_encoder, cache=shared_cache, classes=target_classes)

                try:  # settings that need a missing filter or plain probe are skipped
                    classifier, classes = build()
                except AssertionError as e:
                    logging.warning(f'  [skip] {e}')
                    continue

                lr, dev_loss_by_lr = args.learning_rate, None
                if needs_training(setting):
                    try:
                        sampled = sample_dataset(train_pool, n, seed=args.random_seed)
                    except ValueError as e:
                        logging.error(f'  [skip] {e}')
                        continue
                    train_data, dev_data = split_dataset(sampled, val_ratio=0.2, seed=args.random_seed)
                    if args.lr_grid:  # reseeded fits, so every setting and source starts from the same probe
                        classifier, classes, lr, dev_loss_by_lr = train_best_lr(build, train_data, dev_data, args)
                    else:
                        classifier = train_probe(classifier, train_data, dev_data, classes, args)

                test_acc, test_loss, test_acc_batch = evaluate(
                    classifier, test_data, classes, args.batch_size, repeat_labels=False)
                logging.info(f'  -> acc={test_acc:.4f}  loss={test_loss:.4f}')

                result = {
                    'setting':      setting,
                    'source_exp':   plain['source_exp'] if setting == PLAIN else source_exp,
                    'target_data':  tgt_path,
                    'tgt_lang':     tgt_lang,
                    'n_test':       test_data.get_input_count(),
                    'pool_size':    train_pool.get_input_count(),
                    'test_split':   'official' if args.pool_size and tgt_cfg.get('official_test') else 'custom',
                    'n_examples':   n,
                    'pool_word_embed': pool_word_embed,
                    'lm_name':      lm_name,
                    'num_classes':  len(classes),
                    'random_seed':  args.random_seed,
                    # results saved before these fields existed all used 32 for both
                    'batch_size':   args.batch_size,
                    'embed_batch_size': args.embed_batch_size or args.batch_size,
                    'learning_rate': lr,
                    'dev_loss_by_lr': dev_loss_by_lr,
                    'label_set':    'target' if classes is target_classes else 'source',
                    'test_accuracy': test_acc,
                    'test_accuracy_batch_mean': test_acc_batch,
                    'test_loss':    test_loss,
                }
                save_result(out, result)

        logging.info('')  # blank line between targets


if __name__ == '__main__':
    main()
