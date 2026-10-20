"""Test whether frequency soft-DTW paths describe a language-pair mapping.

The reranking diagnostic in ``probe_frequency_dtw`` shows that warping helps.
It cannot say whether the warp is a stable property of a language pair or a
per-sentence artifact, and it cannot separate a genuine spectral shift from a
tokenizer fertility difference. This script covers both gaps, which are Stage A
go/no-go criteria 3 and 4 in ``plan/plan_2c.md``.

Paths are summarized by displacement ``m(a) - a`` rather than by ``m(a)``.
Translation paths sit near the diagonal, so two diagonal-ish curves correlate
near one whatever the model does; the displacement removes that trivial
component and leaves the part that carries the claim.

Run from ``spectral-probing/``::

    python -m direction2.probe_frequency_paths --model mbert
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import transformers

from direction2.adapter import filter_frequency_safe, filter_token_safe
from direction2.alignment import (
    DTW_COST_MODES,
    boundary_cost_fraction,
    expected_path,
    paired_frequency_cost,
    path_centroids,
    path_entropy,
)
from direction2.probe_frequency_dtw import (
    DEFAULT_LAYERS,
    MODELS,
    block_text,
    encode_language,
)


def content_lengths(blocks, tok, langs):
    """Count content tokens per block for every language."""
    lengths = {}
    for lang in langs:
        texts = [block_text(block, lang) for block in blocks]
        encoded = tok(
            texts, truncation=False, add_special_tokens=True,
            return_special_tokens_mask=True,
        )
        lengths[lang] = np.array([
            len(mask) - sum(mask) for mask in encoded['special_tokens_mask']
        ], dtype=np.float64)
    return lengths


def derangement(n_blocks, seed):
    """Pair every block with a different block, fixed across conditions.

    Built as one random n-cycle rather than by repairing a permutation's fixed
    points: a cycle visits every block exactly once before returning, so no
    block can map to itself. Repairing fixed points individually is not safe,
    because a lone fixed point has nothing to swap with and a pairwise swap can
    create a fresh one.
    """
    if n_blocks < 2:
        raise ValueError('a derangement needs at least two blocks')
    cycle = np.random.default_rng(seed).permutation(n_blocks)
    order = np.empty(n_blocks, dtype=int)
    order[cycle] = np.roll(cycle, -1)
    return order


def path_batch(source, target, args, device):
    """Return displacement, entropy and boundary share for aligned rows."""
    band_radius = None if args.band_radius < 0 else args.band_radius
    displacement, entropy, boundary = [], [], []
    for start in range(0, len(source), args.metric_batch_size):
        stop = start + args.metric_batch_size
        x = torch.from_numpy(source[start:stop]).to(device)
        y = torch.from_numpy(target[start:stop]).to(device)
        cost = paired_frequency_cost(x, y, args.dtw_cost, args.position_weight)
        alignment = expected_path(cost, args.gamma, band_radius)
        index = torch.arange(cost.shape[-2], device=device, dtype=alignment.dtype)
        displacement.append((path_centroids(alignment) - index).cpu().numpy())
        entropy.append(path_entropy(alignment).cpu().numpy())
        boundary.append(boundary_cost_fraction(alignment, cost).cpu().numpy())
    return (
        np.concatenate(displacement),
        np.concatenate(entropy),
        np.concatenate(boundary),
    )


def retest_correlation(displacement):
    """Correlate mean displacement curves from two disjoint block halves."""
    half = len(displacement) // 2
    if half < 2:
        return float('nan')
    first = displacement[:half].mean(axis=0)
    second = displacement[half:2 * half].mean(axis=0)
    if first.std() < 1e-9 or second.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(first, second)[0, 1])


def bootstrap_band(displacement, n_resamples, seed):
    """Return a percentile band for the mean displacement curve."""
    rng = np.random.default_rng(seed)
    n_rows = len(displacement)
    draws = np.stack([
        displacement[rng.integers(0, n_rows, n_rows)].mean(axis=0)
        for _ in range(n_resamples)
    ])
    return (
        np.percentile(draws, 2.5, axis=0).tolist(),
        np.percentile(draws, 97.5, axis=0).tolist(),
    )


def evaluate_pair(freq_a, freq_b, len_a, len_b, args, device):
    """Compare translation paths with shuffled-pair paths for one language pair."""
    shuffled = derangement(len(freq_a), args.shuffle_seed)
    parallel = path_batch(freq_a, freq_b, args, device)
    control = path_batch(freq_a, freq_b[shuffled], args, device)

    row = {
        'length_ratio': float(np.mean(np.abs(np.log(len_a / len_b)))),
        'boundary_fraction': float(parallel[2].mean()),
    }
    for name, (displacement, entropy, _) in (
        ('translation', parallel), ('shuffled', control),
    ):
        row[f'{name}_displacement'] = float(np.abs(displacement).mean())
        row[f'{name}_entropy'] = float(entropy.mean())
        # Spread of each source index's displacement across sentences. Low
        # spread is what "the path is a property of the language pair" means.
        row[f'{name}_spread'] = float(displacement.std(axis=0).mean())
        row[f'{name}_retest_r'] = retest_correlation(displacement)
    row['spread_ratio'] = row['shuffled_spread'] / max(row['translation_spread'], 1e-9)
    return row, parallel[0]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument(
        '--corpus',
        default='direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json',
    )
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--main_layer', type=int, default=None)
    parser.add_argument('--target_layer', type=int, default=None)
    parser.add_argument('--k_start', type=int, default=1)
    parser.add_argument('--k_end', type=int, default=32)
    parser.add_argument('--block_filter_k', type=int, default=64)
    parser.add_argument('--max_blocks', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--metric_batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--dtw_cost', choices=DTW_COST_MODES, default='natural')
    parser.add_argument('--band_radius', type=int, default=-1)
    parser.add_argument('--position_weight', type=float, default=0.0)
    parser.add_argument('--shuffle_seed', type=int, default=42)
    parser.add_argument('--coefficient_dtype', default='float16',
                        choices=['float16', 'float32'],
                        help='storage precision for DCT coefficients')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--out_dir', default='direction2/results/frequency_path_probe')
    return parser.parse_args()


def main():
    """Run the path-consistency and length-ratio diagnostic."""
    args = parse_args()
    if args.k_start < 0 or args.k_end < args.k_start:
        raise SystemExit('Require 0 <= k_start <= k_end')
    if args.block_filter_k < args.k_end:
        raise SystemExit('--block_filter_k must be at least --k_end')
    n_frequencies = args.k_end - args.k_start + 1
    if args.gamma is None:
        args.gamma = 0.1 / n_frequencies if args.dtw_cost == 'natural' else 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformers.logging.set_verbosity_error()

    with open(args.corpus, encoding='utf-8') as handle:
        blocks = json.load(handle)
    if args.max_blocks:
        blocks = blocks[:args.max_blocks]
    langs = args.langs or list(blocks[0]['texts'])

    model_name = MODELS[args.model]
    main_default, target_default = DEFAULT_LAYERS[args.model]
    main_layer = args.main_layer if args.main_layer is not None else main_default
    target_layer = args.target_layer if args.target_layer is not None else target_default
    pooling = 'last' if args.model.startswith('qwen') else 'mean'
    k_indices = list(range(args.k_start, args.k_end + 1))

    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    blocks = filter_token_safe(blocks, tok, langs, args.max_length)
    blocks = filter_frequency_safe(blocks, tok, langs, args.block_filter_k)
    if len(blocks) < 4:
        raise SystemExit('Need at least four blocks to split into halves')

    model_args = {'return_dict': True}
    if args.model.startswith('qwen'):
        model_args['torch_dtype'] = torch.float16
    model = transformers.AutoModel.from_pretrained(
        model_name, **model_args,
    ).to(device).eval()
    num_layers = model.config.num_hidden_layers

    print(
        f'model={args.model} device={device} blocks={len(blocks)} '
        f'target=L{target_layer} k={args.k_start}:{args.k_end} '
        f'dtw_cost={args.dtw_cost} gamma={args.gamma:g}'
    )
    lengths = content_lengths(blocks, tok, langs)
    coefficients = {}
    for lang in langs:
        print(f'Encoding {lang}...', flush=True)
        texts = [block_text(block, lang) for block in blocks]
        _, coefficients[lang] = encode_language(
            model, tok, texts, main_layer, target_layer, k_indices, pooling,
            args.batch_size, args.max_length, device, num_layers,
            getattr(torch, args.coefficient_dtype),
        )

    rows, curves = [], {}
    for lang_a, lang_b in combinations(langs, 2):
        row, displacement = evaluate_pair(
            coefficients[lang_a], coefficients[lang_b],
            lengths[lang_a], lengths[lang_b], args, device,
        )
        row['pair'] = f'{lang_a}-{lang_b}'
        rows.append(row)
        lower, upper = bootstrap_band(displacement, args.bootstrap, args.shuffle_seed)
        curves[row['pair']] = {
            'mean': displacement.mean(axis=0).tolist(),
            'lower': lower,
            'upper': upper,
        }
        print(
            f"  {row['pair']:>7}  disp={row['translation_displacement']:.3f} "
            f"spread={row['translation_spread']:.3f} vs shuffled "
            f"{row['shuffled_spread']:.3f}  retest_r={row['translation_retest_r']:+.3f} "
            f"len_ratio={row['length_ratio']:.3f}"
        )

    names = [key for key in rows[0] if key != 'pair']
    mean_row = {'pair': 'mean'}
    mean_row.update({
        name: float(np.nanmean([row[name] for row in rows])) for name in names
    })

    # Criterion 4: does the warp track tokenizer fertility rather than spectrum?
    ratios = np.array([row['length_ratio'] for row in rows])
    displacements = np.array([row['translation_displacement'] for row in rows])
    correlation = (
        float(np.corrcoef(ratios, displacements)[0, 1]) if len(rows) > 2
        else float('nan')
    )
    rows.append(mean_row)

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma_label = f'{args.gamma:g}'.replace('.', 'p')
    stem = (f'{Path(args.corpus).stem}_L{target_layer}'
            f'_k{args.k_start}-{args.k_end}_bf{args.block_filter_k}'
            f'_c{args.dtw_cost}_g{gamma_label}')
    with open(out_dir / f'{stem}.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['pair'] + names)
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / f'{stem}.json', 'w', encoding='utf-8') as handle:
        json.dump({
            'metadata': {
                'model': args.model, 'corpus': args.corpus,
                'n_blocks': len(blocks), 'langs': langs,
                'target_layer': target_layer, 'k_indices': k_indices,
                'gamma': args.gamma, 'band_radius': args.band_radius,
                'dtw_cost': args.dtw_cost,
                'block_filter_k': args.block_filter_k,
                'shuffle_seed': args.shuffle_seed,
                'coefficient_dtype': args.coefficient_dtype,
            },
            'rows': rows,
            'displacement_curves': curves,
            'length_ratio_displacement_r': correlation,
        }, handle, indent=2)

    print(f'\nmean translation spread : {mean_row["translation_spread"]:.4f}')
    print(f'mean shuffled spread    : {mean_row["shuffled_spread"]:.4f}')
    print(f'spread ratio (>1 favours a stable path): {mean_row["spread_ratio"]:.3f}')
    print(f'mean retest r translation: {mean_row["translation_retest_r"]:+.4f}')
    print(f'mean retest r shuffled   : {mean_row["shuffled_retest_r"]:+.4f}')
    print(f'mean boundary cost share : {mean_row["boundary_fraction"]:.4f}')
    print(f'corr(length ratio, displacement) across pairs: {correlation:+.4f}')
    print(f'saved -> {out_dir / stem}.csv')


if __name__ == '__main__':
    main()
