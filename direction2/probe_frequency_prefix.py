"""Probe frozen multilingual retrieval as higher DCT frequencies are included.

The probe compares cumulative non-DC prefixes and disjoint frequency bands.
It reports natural coefficient weighting and a per-frequency-normalized variant.
All requested models use the same token-safe subset of parallel blocks.

Run from ``spectral-probing/``::

    python -m direction2.probe_frequency_prefix --model mbert --layers 2 4 6 8 10 12
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

from direction2.alignment import content_masks
from models.encoders import PrismEncoder


MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
}
DEFAULT_LAYERS = {'mbert': [8], 'xlmr': [6], 'qwen3-1.7b': [16]}
DEFAULT_CORPUS = 'direction2/data/NTREX/aligned/ntrex_6way_pilot_min100.json'
DEFAULT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
WEIGHTINGS = ('natural', 'equal_k')


def block_text(block, lang):
    """Join one block's sentences without changing their order."""
    return ' '.join(block['texts'][lang])


def l2_normalize(x, axis=-1):
    """L2-normalize an array while preserving zero vectors."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norm, 1e-9, None)


def frequency_vectors(coeffs, start, stop, weighting):
    """Flatten coefficients in ``[start, stop)`` into retrieval vectors.

    Args:
        coeffs: Array with shape ``(n_blocks, max_k + 1, hidden_size)``.
        start: First included DCT index.
        stop: Exclusive final DCT index.
        weighting: ``natural`` preserves coefficient energies. ``equal_k``
            normalizes each coefficient before concatenation.
    """
    selected = coeffs[:, start:stop].astype(np.float32)
    if weighting == 'equal_k':
        selected = l2_normalize(selected, axis=2)
    vectors = selected.reshape(selected.shape[0], -1)
    return l2_normalize(vectors)


def retrieval_p1(reps_a, reps_b):
    """Return bidirectional translation retrieval precision at one."""
    scores = reps_a @ reps_b.T
    targets = np.arange(scores.shape[0])
    p1_ab = np.mean(np.argmax(scores, axis=1) == targets)
    p1_ba = np.mean(np.argmax(scores, axis=0) == targets)
    return float((p1_ab + p1_ba) / 2)


def load_blocks(path, langs=None, max_blocks=None):
    """Load a JSON block corpus and validate its language intersection."""
    with open(path, encoding='utf-8') as handle:
        blocks = json.load(handle)
    if not blocks:
        raise SystemExit(f'No blocks found in {path}')

    available = list(blocks[0]['texts'])
    langs = langs or available
    missing = [lang for lang in langs if lang not in available]
    if missing:
        raise SystemExit(f'Languages missing from corpus: {missing}')
    blocks = blocks[:max_blocks] if max_blocks else blocks
    return blocks, langs


def safe_prefix_rows(block, tok, langs, max_length, min_content):
    """Find sentence-prefix counts safe for every language of one tokenizer."""
    n_rows = min(len(block['texts'][lang]) for lang in langs)
    safe_rows = set(range(1, n_rows + 1))
    for lang in langs:
        choices = [' '.join(block['texts'][lang][:count]) for count in range(1, n_rows + 1)]
        enc = tok(
            choices, add_special_tokens=True, truncation=False,
            return_special_tokens_mask=True, verbose=False,
        )
        valid = set()
        for row_count, (ids, special) in enumerate(zip(enc['input_ids'], enc['special_tokens_mask']), start=1):
            content_len = len(ids) - sum(special)
            if len(ids) <= max_length and content_len > min_content:
                valid.add(row_count)
        safe_rows &= valid
    return safe_rows


def shared_sentence_prefixes(blocks, langs, max_length, min_content, model_names):
    """Crop blocks to parallel sentence prefixes safe for every tokenizer."""
    row_options = [set(range(1, min(len(block['texts'][lang]) for lang in langs) + 1))
                   for block in blocks]
    for model_name in model_names:
        tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
        for i, block in enumerate(blocks):
            row_options[i] &= safe_prefix_rows(block, tok, langs, max_length, min_content)
        del tok

    row_counts = [max(options, default=0) for options in row_options]

    cropped = []
    kept_rows = []
    for block, row_count in zip(blocks, row_counts):
        if row_count == 0:
            continue
        item = dict(block)
        item['texts'] = {
            lang: sentences[:row_count]
            for lang, sentences in block['texts'].items()
        }
        cropped.append(item)
        kept_rows.append(int(row_count))
    return cropped, kept_rows


@torch.no_grad()
def encode_language(model, tok, texts, layers, max_k, batch_size, max_length, device):
    """Extract coefficients and last-token vectors at the requested layers."""
    parts = {layer: {'coeffs': [], 'last': []} for layer in layers}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(
            batch, return_tensors='pt', padding=True, truncation=True,
            max_length=max_length,
        ).to(device)
        masks = content_masks(enc, device)
        out = model(**enc, output_hidden_states=True)

        for layer in layers:
            hidden = out.hidden_states[layer]
            for i, mask in enumerate(masks):
                sequence = hidden[i, mask].float()
                if sequence.shape[0] <= max_k:
                    raise RuntimeError(
                        f'Found {sequence.shape[0]} content tokens, but max_k={max_k}. '
                        'The shared token-safe filter should have removed this block.'
                    )
                freq = PrismEncoder.dct(sequence.T, norm='ortho').T
                parts[layer]['coeffs'].append(
                    freq[:max_k + 1].cpu().numpy().astype(np.float16)
                )
                parts[layer]['last'].append(
                    sequence[-1].cpu().numpy().astype(np.float16)
                )

    return {
        layer: {
            name: np.stack(values)
            for name, values in layer_parts.items()
        }
        for layer, layer_parts in parts.items()
    }


def add_result(rows, args, layer, n_blocks, representation, weighting, k_min, k_max, pair, p1):
    """Append one consistently formatted result row."""
    rows.append({
        'model': args.model,
        'layer': layer,
        'n_blocks': n_blocks,
        'representation': representation,
        'weighting': weighting,
        'k_min': k_min,
        'k_max': k_max,
        'pair': pair,
        'p1': round(p1, 6),
    })


def evaluate(reps, langs, k_values, args, layer, n_blocks):
    """Evaluate baselines, cumulative prefixes, and disjoint bands."""
    rows = []
    pairs = list(combinations(langs, 2))
    baseline_reps = {
        'dc': {lang: frequency_vectors(reps[lang]['coeffs'], 0, 1, 'natural') for lang in langs},
        'last': {lang: l2_normalize(reps[lang]['last'].astype(np.float32)) for lang in langs},
    }
    for name, vectors in baseline_reps.items():
        values = []
        for lang_a, lang_b in pairs:
            p1 = retrieval_p1(vectors[lang_a], vectors[lang_b])
            add_result(
                rows, args, layer, n_blocks, name, 'natural', '', '',
                f'{lang_a}-{lang_b}', p1,
            )
            values.append(p1)
        add_result(
            rows, args, layer, n_blocks, name, 'natural', '', '',
            'mean', np.mean(values),
        )

    for weighting in WEIGHTINGS:
        previous = 0
        for k_max in k_values:
            prefix = {
                lang: frequency_vectors(reps[lang]['coeffs'], 1, k_max + 1, weighting)
                for lang in langs
            }
            band = {
                lang: frequency_vectors(reps[lang]['coeffs'], previous + 1, k_max + 1, weighting)
                for lang in langs
            }
            for representation, vectors, k_min in (
                ('prefix', prefix, 1),
                ('band', band, previous + 1),
            ):
                values = []
                for lang_a, lang_b in pairs:
                    p1 = retrieval_p1(vectors[lang_a], vectors[lang_b])
                    add_result(
                        rows, args, layer, n_blocks, representation, weighting,
                        k_min, k_max, f'{lang_a}-{lang_b}', p1,
                    )
                    values.append(p1)
                add_result(
                    rows, args, layer, n_blocks, representation, weighting,
                    k_min, k_max, 'mean', np.mean(values),
                )
            previous = k_max
    return rows


def summarize(rows):
    """Create a compact summary from mean-over-language-pair rows."""
    means = [row for row in rows if row['pair'] == 'mean']
    summary = {'baselines': {}, 'prefix': {}, 'bands': {}}
    for row in means:
        if row['representation'] in ('dc', 'last'):
            summary['baselines'][row['representation']] = row['p1']
            continue
        target = 'prefix' if row['representation'] == 'prefix' else 'bands'
        key = f"{row['k_min']}:{row['k_max']}"
        summary[target].setdefault(row['weighting'], {})[key] = row['p1']
    return summary


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument('--layer', type=int, default=None,
                             help='single-layer compatibility option')
    layer_group.add_argument('--layers', nargs='+', type=int, default=None,
                             help='hidden-state indices to evaluate in one pass')
    parser.add_argument('--corpus', default=DEFAULT_CORPUS)
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--k_values', nargs='+', type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_blocks', type=int, default=None)
    parser.add_argument('--out_dir', default='direction2/results/frequency_prefix_probe')
    return parser.parse_args()


def main():
    """Run the frozen frequency-prefix retrieval diagnostic."""
    args = parse_args()
    if args.layers is not None:
        layers = sorted(set(args.layers))
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = DEFAULT_LAYERS[args.model]
    k_values = sorted(set(args.k_values))
    if not k_values or k_values[0] < 1:
        raise SystemExit('--k_values must contain positive DCT indices')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model.startswith('qwen') and device.type != 'cuda':
        raise SystemExit('Qwen probing requires a CUDA device')
    transformers.logging.set_verbosity_error()

    blocks, langs = load_blocks(args.corpus, args.langs, args.max_blocks)
    n_loaded = len(blocks)
    blocks, row_counts = shared_sentence_prefixes(
        blocks, langs, args.max_length, max(k_values), MODELS.values(),
    )
    if not blocks:
        raise SystemExit('No blocks survive the shared token-safe filter')

    corpus_hash = hashlib.md5(Path(args.corpus).read_bytes()).hexdigest()[:8]
    block_ids = [block.get('block_id', i) for i, block in enumerate(blocks)]
    crop_spec = list(zip(block_ids, row_counts))
    subset_hash = hashlib.md5(json.dumps(crop_spec).encode()).hexdigest()[:8]
    print(
        f'model={args.model} layers={layers} device={device} '
        f'blocks={len(blocks)}/{n_loaded} '
        f'langs={langs} subset={subset_hash}'
    )

    model_name = MODELS[args.model]
    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_args = {'return_dict': True}
    if args.model.startswith('qwen'):
        model_args['torch_dtype'] = torch.float16
    model = transformers.AutoModel.from_pretrained(model_name, **model_args).to(device).eval()

    reps = {}
    for lang in langs:
        print(f'Encoding {lang}...', flush=True)
        texts = [block_text(block, lang) for block in blocks]
        reps[lang] = encode_language(
            model, tok, texts, layers, max(k_values), args.batch_size,
            args.max_length, device,
        )

    rows = []
    layer_summaries = {}
    for layer in layers:
        layer_reps = {lang: reps[lang][layer] for lang in langs}
        layer_rows = evaluate(
            layer_reps, langs, k_values, args, layer, len(blocks),
        )
        rows.extend(layer_rows)
        layer_summaries[str(layer)] = summarize(layer_rows)

    summary = {'layers': layer_summaries}
    summary['metadata'] = {
        'model': args.model,
        'hf_model': model_name,
        'primary_retrieval_baseline': (
            'last' if args.model.startswith('qwen') else 'dc'
        ),
        'layers': layers,
        'corpus': args.corpus,
        'corpus_hash': corpus_hash,
        'subset_hash': subset_hash,
        'n_blocks': len(blocks),
        'block_ids': block_ids,
        'sentence_rows': row_counts,
        'langs': langs,
        'k_values': k_values,
        'max_length': args.max_length,
    }

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_str = '-'.join(str(layer) for layer in layers)
    stem = f'{Path(args.corpus).stem}_L{layer_str}_{subset_hash}'
    csv_path = out_dir / f'{stem}.csv'
    json_path = out_dir / f'{stem}.json'
    fields = [
        'model', 'layer', 'n_blocks', 'representation', 'weighting',
        'k_min', 'k_max', 'pair', 'p1',
    ]
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
