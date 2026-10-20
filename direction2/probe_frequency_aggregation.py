"""Compare compact averages and concatenations of low-frequency coefficients.

The probe performs full parallel-sentence retrieval rather than candidate
reranking. For every prefix ``k=1..K``, it compares concatenating coefficients
with averaging them into one hidden-size vector. Natural variants preserve the
relative coefficient energies; equal-k variants normalize every coefficient
before aggregation. Individual coefficients, DC, and last-token vectors are
reported as controls.

Run from ``spectral-probing/``::

    python -m direction2.probe_frequency_aggregation --model mbert --layers 8
    python -m direction2.probe_frequency_aggregation --model qwen3-1.7b --layers 16 24
"""
import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers

from direction2.probe_frequency_prefix import (
    DEFAULT_CORPUS,
    MODELS,
    block_text,
    encode_language,
    frequency_vectors,
    l2_normalize,
    load_blocks,
    retrieval_p1,
    shared_sentence_prefixes,
)


DEFAULT_LAYERS = {'mbert': [8], 'xlmr': [6], 'qwen3-1.7b': [16, 24]}
DEFAULT_MAX_K = 8
AGGREGATIONS = ('concatenate', 'mean')
WEIGHTINGS = ('natural', 'equal_k')


def aggregate_vectors(coeffs, start, stop, aggregation, weighting):
    """Turn a fixed frequency range into normalized retrieval vectors.

    Args:
        coeffs: Array shaped ``(n_blocks, max_k + 1, hidden_size)``.
        start: First included DCT index.
        stop: Exclusive final DCT index.
        aggregation: Either ``concatenate`` or ``mean``.
        weighting: ``natural`` preserves coefficient energy, while ``equal_k``
            normalizes each coefficient before aggregation.

    Returns:
        L2-normalized retrieval vectors.
    """
    if aggregation == 'concatenate':
        return frequency_vectors(coeffs, start, stop, weighting)
    if aggregation != 'mean':
        raise ValueError(f'Unsupported aggregation: {aggregation}')
    if weighting not in WEIGHTINGS:
        raise ValueError(f'Unsupported weighting: {weighting}')

    selected = coeffs[:, start:stop].astype(np.float32)
    if weighting == 'equal_k':
        selected = l2_normalize(selected, axis=2)
    return l2_normalize(selected.mean(axis=1))


def add_result(rows, args, layer, n_blocks, representation, weighting, k_min, k_max, pair, p1):
    """Append one retrieval result using a stable output schema."""
    rows.append({
        'model': args.model,
        'layer': layer,
        'n_blocks': n_blocks,
        'representation': representation,
        'weighting': weighting,
        'k_min': k_min,
        'k_max': k_max,
        'pair': pair,
        'p1': round(float(p1), 6),
    })


def evaluate_vectors(rows, vectors, pairs, args, layer, n_blocks, representation, weighting, k_min, k_max):
    """Evaluate one representation for every language pair and their macro mean."""
    values = []
    for lang_a, lang_b in pairs:
        p1 = retrieval_p1(vectors[lang_a], vectors[lang_b])
        add_result(rows, args, layer, n_blocks, representation, weighting, k_min, k_max, f'{lang_a}-{lang_b}', p1)
        values.append(p1)
    add_result(rows, args, layer, n_blocks, representation, weighting, k_min, k_max, 'mean', np.mean(values))


def evaluate(reps, langs, max_k, args, layer, n_blocks):
    """Evaluate coefficient aggregation for every prefix through ``max_k``."""
    rows = []
    pairs = list(combinations(langs, 2))
    controls = {
        'dc': {lang: frequency_vectors(reps[lang]['coeffs'], 0, 1, 'natural') for lang in langs},
        'last': {lang: l2_normalize(reps[lang]['last'].astype(np.float32)) for lang in langs},
    }
    for name, vectors in controls.items():
        evaluate_vectors(rows, vectors, pairs, args, layer, n_blocks, name, 'natural', '', '')

    for k in range(1, max_k + 1):
        vectors = {
            lang: frequency_vectors(reps[lang]['coeffs'], k, k + 1, 'natural')
            for lang in langs
        }
        evaluate_vectors(rows, vectors, pairs, args, layer, n_blocks, 'individual', 'natural', k, k)

    for k_max in range(1, max_k + 1):
        for aggregation in AGGREGATIONS:
            for weighting in WEIGHTINGS:
                vectors = {
                    lang: aggregate_vectors(reps[lang]['coeffs'], 1, k_max + 1, aggregation, weighting)
                    for lang in langs
                }
                evaluate_vectors(rows, vectors, pairs, args, layer, n_blocks, aggregation, weighting, 1, k_max)
    return rows


def summarize(rows):
    """Return mean pairwise P@1 indexed by representation and frequency range."""
    summary = {}
    for row in rows:
        if row['pair'] != 'mean':
            continue
        key = row['representation']
        if row['weighting']:
            key = f"{key}_{row['weighting']}"
        frequency = 'baseline' if row['k_min'] == '' else f"{row['k_min']}:{row['k_max']}"
        summary.setdefault(key, {})[frequency] = row['p1']
    return summary


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--corpus', default=DEFAULT_CORPUS)
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--max_k', type=int, default=DEFAULT_MAX_K)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_blocks', type=int, default=None)
    parser.add_argument('--population_models', nargs='+', choices=MODELS, default=['mbert', 'qwen3-1.7b'],
                        help='tokenizers whose shared safe block population is used')
    parser.add_argument('--out_dir', default='direction2/results/frequency_aggregation_probe')
    return parser.parse_args()


def main():
    """Run the frozen low-frequency aggregation diagnostic."""
    args = parse_args()
    if args.max_k < 1:
        raise SystemExit('--max_k must be positive')
    layers = sorted(set(args.layers or DEFAULT_LAYERS[args.model]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model.startswith('qwen') and device.type != 'cuda':
        raise SystemExit('Qwen probing requires a CUDA device')
    transformers.logging.set_verbosity_error()

    blocks, langs = load_blocks(args.corpus, args.langs, args.max_blocks)
    n_loaded = len(blocks)
    population_names = [MODELS[name] for name in args.population_models]
    blocks, row_counts = shared_sentence_prefixes(blocks, langs, args.max_length, args.max_k, population_names)
    if not blocks:
        raise SystemExit('No blocks survive the shared token-safe filter')

    corpus_hash = hashlib.md5(Path(args.corpus).read_bytes()).hexdigest()[:8]
    block_ids = [block.get('block_id', i) for i, block in enumerate(blocks)]
    crop_spec = list(zip(block_ids, row_counts))
    subset_hash = hashlib.md5(json.dumps(crop_spec).encode()).hexdigest()[:8]
    print(
        f'model={args.model} layers={layers} device={device} blocks={len(blocks)}/{n_loaded} '
        f'langs={langs} k=1:{args.max_k} subset={subset_hash}'
    )

    model_name = MODELS[args.model]
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_args = {'return_dict': True}
    if args.model.startswith('qwen'):
        model_args['torch_dtype'] = torch.float16
    model = transformers.AutoModel.from_pretrained(model_name, **model_args).to(device).eval()
    if any(layer < 0 or layer > model.config.num_hidden_layers for layer in layers):
        raise SystemExit(f'Layers must be between 0 and {model.config.num_hidden_layers}')

    reps = {}
    for lang in langs:
        print(f'Encoding {lang}...', flush=True)
        texts = [block_text(block, lang) for block in blocks]
        reps[lang] = encode_language(model, tok, texts, layers, args.max_k, args.batch_size, args.max_length, device)

    rows = []
    summaries = {}
    for layer in layers:
        layer_reps = {lang: reps[lang][layer] for lang in langs}
        layer_rows = evaluate(layer_reps, langs, args.max_k, args, layer, len(blocks))
        rows.extend(layer_rows)
        summaries[str(layer)] = summarize(layer_rows)

    summary = {
        'layers': summaries,
        'metadata': {
            'model': args.model,
            'hf_model': model_name,
            'layers': layers,
            'corpus': args.corpus,
            'corpus_hash': corpus_hash,
            'subset_hash': subset_hash,
            'n_blocks': len(blocks),
            'block_ids': block_ids,
            'sentence_rows': row_counts,
            'langs': langs,
            'max_k': args.max_k,
            'max_length': args.max_length,
            'population_models': args.population_models,
        },
    }

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_str = '-'.join(str(layer) for layer in layers)
    stem = f'{Path(args.corpus).stem}_L{layer_str}_k1-{args.max_k}_{subset_hash}'
    csv_path = out_dir / f'{stem}.csv'
    json_path = out_dir / f'{stem}.json'
    fields = ['model', 'layer', 'n_blocks', 'representation', 'weighting', 'k_min', 'k_max', 'pair', 'p1']
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Saved {csv_path}')
    print(f'Saved {json_path}')


if __name__ == '__main__':
    main()
