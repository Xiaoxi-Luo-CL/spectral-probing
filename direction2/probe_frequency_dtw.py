"""Compare rigid and soft-DTW frequency matching on a frozen model.

The final mean or last-token representation selects a shared candidate set.
Candidates are then reranked using signed DCT coefficients directly, without
RMS pooling, IDCT, or interpolation. A fixed frequency permutation tests
whether soft-DTW benefits from meaningful frequency order rather than only
from its extra flexibility. The default ``natural`` cost preserves the relative
energy of the concatenated frequency representation; ``equal_k`` reproduces
the original per-coefficient cosine experiments.

An optional score-fusion diagnostic combines the base and DTW candidate scores.
Select one global weight on UN with ``--fusion_alphas``, then load that frozen
weight on NTREX with ``--fusion_config``. Target-corpus labels never select the
weight.

Run from ``spectral-probing/``::

    python -m direction2.probe_frequency_dtw --model mbert
    python -m direction2.probe_frequency_dtw --model qwen3-1.7b --target_layer 16

The inclusive ranges for the 32-coefficient controls are A2 ``1..32``, A4
``0..31``, and A5 ``33..64``. Keep ``--block_filter_k 64`` for all three so
they use the same eligible blocks.
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

from direction2.adapter import filter_frequency_safe, filter_token_safe
from direction2.alignment import (
    DTW_COST_MODES,
    content_masks,
    get_hidden_state,
    last_pool,
    mean_pool,
    paired_rigid_frequency_distance,
    paired_soft_dtw_divergence,
    self_soft_dtw,
    spectral_coefficients,
)


MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
    'qwen3-1.7b': 'Qwen/Qwen3-1.7B',
}
DEFAULT_LAYERS = {
    'mbert': (8, 8),
    'xlmr': (6, 6),
    'qwen3-1.7b': (16, 16),
}
DEFAULT_CORPUS = 'direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json'
DEFAULT_DTW_GAMMA = 0.1
DEFAULT_FUSION_ALPHAS = (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
FUSION_MATCH_FIELDS = (
    'model', 'main_layer', 'target_layer', 'k_indices', 'block_filter_k',
    'candidates', 'gamma', 'band_radius', 'position_weight', 'coefficient_dtype',
    'dtw_cost',
)


def block_text(block, lang):
    """Join one block's sentences without changing their order."""
    return ' '.join(block['texts'][lang])


def normalize(reps):
    """L2-normalize one matrix of representations."""
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    return reps / np.clip(norms, 1e-9, None)


def row_zscore(values):
    """Standardize candidate scores independently for every query."""
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.clip(std, 1e-9, None)


def fusion_condition(alpha):
    """Return a stable metric key for one fusion weight."""
    label = f'{alpha:.8g}'.replace('-', 'm').replace('.', 'p')
    return f'fusion_a{label}'


def fused_candidate_distances(similarity, candidates, dtw_distances, alpha):
    """Combine standardized base similarity and DTW distance for reranking."""
    base_scores = np.take_along_axis(similarity, candidates, axis=1)
    fused_scores = row_zscore(base_scores) - alpha * row_zscore(dtw_distances)
    return -fused_scores


def resolve_fusion(args):
    """Resolve selection, fixed-weight, or config-loaded fusion settings."""
    if args.fusion_config:
        path = Path(args.fusion_config)
        with open(path, encoding='utf-8') as handle:
            saved = json.load(handle)
        if 'fusion' not in saved or saved['fusion'].get('selected_alpha') is None:
            raise SystemExit(f'No selected fusion alpha found in {path}')
        alpha = float(saved['fusion']['selected_alpha'])
        source_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]
        source_settings = {field: saved.get('metadata', {}).get(field) for field in FUSION_MATCH_FIELDS}
        if source_settings['dtw_cost'] is None:
            source_settings['dtw_cost'] = 'equal_k'
        return [alpha], {
            'mode': 'config', 'source': str(path), 'source_hash': source_hash,
            'source_settings': source_settings,
        }
    if args.fusion_alpha is not None:
        return [args.fusion_alpha], {'mode': 'fixed'}
    if args.fusion_alphas is not None:
        values = args.fusion_alphas or list(DEFAULT_FUSION_ALPHAS)
        return sorted(set(values)), {'mode': 'selection'}
    return [], {'mode': 'none'}


def validate_fusion_source(fusion, current):
    """Reject a frozen alpha selected under incompatible model settings."""
    if fusion['mode'] != 'config':
        return
    source = fusion['source_settings']
    mismatches = [field for field in FUSION_MATCH_FIELDS if source.get(field) != current.get(field)]
    if mismatches:
        details = ', '.join(f'{field}: {source.get(field)!r} != {current.get(field)!r}' for field in mismatches)
        raise SystemExit(f'Fusion source configuration does not match target run: {details}')


@torch.no_grad()
def encode_language(model, tok, texts, main_layer, target_layer, k_indices,
                    pooling, batch_size, max_length, device, num_layers,
                    coefficient_dtype=torch.float16):
    """Extract base retrieval vectors and signed frequency coefficients.

    ``coefficient_dtype`` sets storage precision. Float16 is adequate for the
    measured mBERT and Qwen spectra; float32 remains available for steeper
    profiles or explicit precision checks.
    """
    base_parts = []
    freq_parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(
            batch, return_tensors='pt', padding=True, truncation=True,
            max_length=max_length,
        ).to(device)
        masks = content_masks(enc, device)
        outputs = model(**enc, output_hidden_states=True)
        main_hidden = get_hidden_state(outputs, main_layer, num_layers)
        target_hidden = get_hidden_state(outputs, target_layer, num_layers)
        if pooling == 'last':
            base = last_pool(main_hidden, masks)
        else:
            base = mean_pool(main_hidden, masks)
        coeffs = spectral_coefficients(
            target_hidden, torch.stack(masks), k_indices, strict=True,
        )
        base_parts.append(base.float().cpu().numpy())
        freq_parts.append(coeffs.to(coefficient_dtype).cpu().numpy())
    return normalize(np.concatenate(base_parts)), np.concatenate(freq_parts)


def candidate_indices(similarity, count):
    """Return the strongest base-representation candidates for each query."""
    count = min(count, similarity.shape[1])
    return np.argsort(-similarity, axis=1)[:, :count]


def paired_distances(source, target, query_ids, target_ids, mode, args, device,
                     self_source=None, self_target=None):
    """Compute one distance for each listed source-target candidate pair.

    ``self_source`` and ``self_target`` hold each sequence's precomputed
    self soft-DTW term. Every query repeats once per candidate, so recomputing
    those terms inline would redo identical work ``--candidates`` times.
    """
    values = []
    for start in range(0, len(query_ids), args.metric_batch_size):
        stop = start + args.metric_batch_size
        rows = query_ids[start:stop]
        cols = target_ids[start:stop]
        source_batch = torch.from_numpy(source[rows]).to(device)
        target_batch = torch.from_numpy(target[cols]).to(device)
        # print('in paired_distances', source_batch.shape, target_batch.shape, mode)
        # from IPython import embed
        # embed()  # BREAKPOINT
        if mode == 'rigid':
            distance = paired_rigid_frequency_distance(source_batch, target_batch, args.dtw_cost)
        else:
            band_radius = None if args.band_radius < 0 else args.band_radius
            distance = paired_soft_dtw_divergence(
                source_batch, target_batch, args.gamma, band_radius,
                args.position_weight,
                self_x=None if self_source is None
                else torch.from_numpy(self_source[rows]).to(device),
                self_y=None if self_target is None
                else torch.from_numpy(self_target[cols]).to(device),
                cost_mode=args.dtw_cost,
            )
        values.append(distance.cpu().numpy())
    return np.concatenate(values)


def self_terms(freq, args, device):
    """Precompute the self soft-DTW debias term once for every sequence."""
    band_radius = None if args.band_radius < 0 else args.band_radius
    values = []
    for start in range(0, len(freq), args.metric_batch_size):
        batch = torch.from_numpy(freq[start:start + args.metric_batch_size]).to(device)
        values.append(
            self_soft_dtw(
                batch, args.gamma, band_radius, args.position_weight,
                args.dtw_cost,
            ).cpu().numpy()
        )
    return np.concatenate(values)


def reranking_metrics(distances, candidates):
    """Compute P@1, MRR and the per-query hit vector for a reranked list.

    The hit vector is what the paired bootstrap resamples, so conditions are
    compared on the same blocks rather than through independent summaries.
    """
    n_queries = candidates.shape[0]
    order = np.argsort(distances, axis=1)
    ranked = np.take_along_axis(candidates, order, axis=1)
    targets = np.arange(n_queries)
    hits = ranked[:, 0] == targets
    matches = ranked == targets[:, None]
    found = matches.any(axis=1)
    ranks = np.zeros(n_queries, dtype=np.float32)
    ranks[found] = 1.0 / (np.argmax(matches[found], axis=1) + 1)
    # print('in reranking_metrics', hits.mean(), ranks.mean(), hits)
    # from IPython import embed
    # embed()  # BREAKPOINT
    return float(hits.mean()), float(ranks.mean()), hits


def negative_ids(candidates, n_blocks, seed):
    """Select one hard and one random negative for every query.

    Hard negatives come from the shared base-retrieval candidate list, so they
    test whether the frequency metric adds anything beyond ordinary sentence
    similarity. Random negatives test basic discrimination. Both are keyed to
    the query index alone, so every condition scores identical negatives.
    """
    hard = np.array([
        next(candidate for candidate in row if candidate != query)
        for query, row in enumerate(candidates)
    ])
    queries = np.arange(len(candidates))
    drawn = np.random.default_rng(seed).integers(0, n_blocks - 1, len(candidates))
    return hard, drawn + (drawn >= queries)  # shift past self, never equal


def negative_stats(source, target, candidates, mode, args, device,
                   self_source=None, self_target=None):
    """Measure margin and positive-versus-negative accuracy for both negatives."""
    queries = np.arange(candidates.shape[0])
    hard, drawn = negative_ids(candidates, len(source), args.negative_seed)
    positive = paired_distances(
        source, target, queries, queries, mode, args, device,
        self_source, self_target,
    )
    stats = {}
    for name, negatives in (('hard', hard), ('random', drawn)):
        distance = paired_distances(
            source, target, queries, negatives, mode, args, device,
            self_source, self_target,
        )
        stats[f'{name}_margin'] = float(np.mean(distance - positive))
        stats[f'{name}_accuracy'] = float(np.mean(positive < distance))
    return stats


def paired_bootstrap(hits_a, hits_b, n_resamples, seed):
    """Return the mean P@1 difference and a percentile interval over blocks.

    Both conditions are indexed by the same resampled blocks, which is what
    makes the interval paired rather than a difference of two loose intervals.
    """
    stacked_a = np.stack(hits_a)
    stacked_b = np.stack(hits_b)
    rng = np.random.default_rng(seed)
    n_blocks = stacked_a.shape[1]
    draws = np.array([
        (stacked_a[:, pick].mean() - stacked_b[:, pick].mean())
        for pick in (rng.integers(0, n_blocks, n_blocks) for _ in range(n_resamples))
    ])
    return {
        'difference': float(stacked_a.mean() - stacked_b.mean()),
        'lower': float(np.percentile(draws, 2.5)),
        'upper': float(np.percentile(draws, 97.5)),
    }


def evaluate_direction(source_reps, target_reps, args, device):
    """Evaluate one directed retrieval problem with a common candidate set.

    Each side carries its base vectors plus the plain and permuted frequency
    coefficients with their self soft-DTW terms, all computed once per language
    in ``main`` and reused across every language pair and direction.
    """
    similarity = source_reps['base'] @ target_reps['base'].T
    candidates = candidate_indices(similarity, args.candidates)
    targets = np.arange(similarity.shape[0])
    base_hits = np.argmax(similarity, axis=1) == targets
    result = {
        'base_p1': float(base_hits.mean()),
        'candidate_recall': float(np.mean((candidates == targets[:, None]).any(axis=1))),
    }
    hits = {'base': base_hits}

    query_ids = np.repeat(targets, candidates.shape[1])
    target_ids = candidates.reshape(-1)
    # Permuting reorders the frequency axis, so a permuted sequence carries its
    # own self term; rigid matching never uses one.
    conditions = {
        'rigid': ('plain', 'rigid'),
        'dtw': ('plain', 'dtw'),
        'permuted_dtw': ('permuted', 'dtw'),
    }
    dtw_distances = None
    for name, (variant, mode) in conditions.items():
        source = source_reps[variant]
        target = target_reps[variant]
        if mode == 'rigid':
            self_source = self_target = None
        else:
            self_source = source_reps[f'self_{variant}']
            self_target = target_reps[f'self_{variant}']
        # print('in evaluate_direction', name, source.shape, target.shape, mode)
        # from IPython import embed
        # embed()  # BREAKPOINT
        distances = paired_distances(
            source, target, query_ids, target_ids, mode, args, device,
            self_source, self_target,
        ).reshape(candidates.shape)
        if name == 'dtw':
            dtw_distances = distances
        p1, mrr, condition_hits = reranking_metrics(distances, candidates)
        result[f'{name}_p1'] = p1
        result[f'{name}_mrr'] = mrr
        hits[name] = condition_hits
        for stat, value in negative_stats(
            source, target, candidates, mode, args, device,
            self_source, self_target,
        ).items():
            result[f'{name}_{stat}'] = value

    for alpha in args.fusion_values:
        name = fusion_condition(alpha)
        distances = fused_candidate_distances(similarity, candidates, dtw_distances, alpha)
        p1, mrr, condition_hits = reranking_metrics(distances, candidates)
        result[f'{name}_p1'] = p1
        result[f'{name}_mrr'] = mrr
        hits[name] = condition_hits
    return result, hits


def average_directions(first, second):
    """Average matching metrics from both retrieval directions."""
    return {key: (first[key] + second[key]) / 2 for key in first}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--corpus', default=DEFAULT_CORPUS)
    parser.add_argument('--langs', nargs='+', default=None)
    parser.add_argument('--main_layer', type=int, default=None)
    parser.add_argument('--target_layer', type=int, default=None)
    parser.add_argument('--k_start', type=int, default=1, help='first included frequency index')
    parser.add_argument('--k_end', type=int, default=32, help='last included frequency index; inclusive')
    parser.add_argument('--block_filter_k', type=int, default=64,
                        help='pin population independently of k_end; must be at least k_end')
    parser.add_argument('--max_blocks', type=int, default=500)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--candidates', type=int, default=32)
    parser.add_argument('--metric_batch_size', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=None,
                        help='soft-DTW temperature; defaults to 0.1 for equal_k and 0.1 / n_frequencies for natural')
    parser.add_argument('--dtw_cost', choices=DTW_COST_MODES, default='natural',
                        help='natural preserves relative coefficient energy; equal_k reproduces legacy runs')
    parser.add_argument('--band_radius', type=int, default=-1)
    parser.add_argument('--position_weight', type=float, default=0.0)
    parser.add_argument('--permutation_seed', type=int, default=42)
    parser.add_argument('--negative_seed', type=int, default=7)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--coefficient_dtype', default='float16',
                        choices=['float16', 'float32'],
                        help='storage precision for DCT coefficients')
    fusion = parser.add_mutually_exclusive_group()
    fusion.add_argument('--fusion_alphas', nargs='*', type=float,
                        help='select one alpha from this grid; omit values for the default grid')
    fusion.add_argument('--fusion_alpha', type=float, help='evaluate one fixed fusion alpha')
    fusion.add_argument('--fusion_config', help='load selected_alpha from a prior result JSON')
    parser.add_argument('--out_dir', default='direction2/results/frequency_dtw_probe')
    return parser.parse_args()


def main():
    """Run the frozen candidate-reranking diagnostic."""
    args = parse_args()
    if args.k_start < 0 or args.k_end < args.k_start:
        raise SystemExit('Require 0 <= k_start <= k_end')
    if args.block_filter_k < args.k_end:
        raise SystemExit('--block_filter_k must be at least --k_end')
    if args.candidates < 2:
        raise SystemExit('--candidates must be at least two')
    n_frequencies = args.k_end - args.k_start + 1
    if args.gamma is None:
        args.gamma = DEFAULT_DTW_GAMMA / n_frequencies if args.dtw_cost == 'natural' else DEFAULT_DTW_GAMMA
    if args.gamma <= 0:
        raise SystemExit('--gamma must be positive')
    if args.band_radius < -1:
        raise SystemExit('--band_radius must be -1 or non-negative')
    if args.position_weight < 0:
        raise SystemExit('--position_weight must be non-negative')
    args.fusion_values, fusion = resolve_fusion(args)
    if any(not np.isfinite(alpha) or alpha < 0 for alpha in args.fusion_values):
        raise SystemExit('Fusion alphas must be finite and non-negative')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model.startswith('qwen') and device.type != 'cuda':
        raise SystemExit('Qwen probing requires a CUDA device')
    transformers.logging.set_verbosity_error()

    with open(args.corpus, encoding='utf-8') as handle:
        blocks = json.load(handle)
    if args.max_blocks:
        blocks = blocks[:args.max_blocks]
    if not blocks:
        raise SystemExit(f'No blocks found in {args.corpus}')
    langs = args.langs or list(blocks[0]['texts'])
    missing = [lang for lang in langs if lang not in blocks[0]['texts']]
    if missing:
        available = list(blocks[0]['texts'])
        raise SystemExit(
            f'Languages missing from corpus: {missing}; available: {available}'
        )

    model_name = MODELS[args.model]
    main_default, target_default = DEFAULT_LAYERS[args.model]
    main_layer = args.main_layer if args.main_layer is not None else main_default
    target_layer = args.target_layer if args.target_layer is not None else target_default
    pooling = 'last' if args.model.startswith('qwen') else 'mean'
    k_indices = list(range(args.k_start, args.k_end + 1))

    tok = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    n_loaded = len(blocks)
    blocks = filter_token_safe(blocks, tok, langs, args.max_length)
    blocks = filter_frequency_safe(blocks, tok, langs, args.block_filter_k)
    if len(blocks) < 2:
        raise SystemExit('Fewer than two blocks survive token and frequency filtering')
    validate_fusion_source(fusion, {
        'model': args.model,
        'main_layer': main_layer,
        'target_layer': target_layer,
        'k_indices': k_indices,
        'block_filter_k': args.block_filter_k,
        'candidates': min(args.candidates, len(blocks)),
        'gamma': args.gamma,
        'band_radius': args.band_radius,
        'position_weight': args.position_weight,
        'coefficient_dtype': args.coefficient_dtype,
        'dtw_cost': args.dtw_cost,
    })

    model_args = {'return_dict': True}
    if args.model.startswith('qwen'):
        model_args['torch_dtype'] = torch.float16
    model = transformers.AutoModel.from_pretrained(model_name, **model_args).to(device).eval()
    num_layers = model.config.num_hidden_layers
    if not 0 <= main_layer <= num_layers or not 0 <= target_layer <= num_layers:
        raise SystemExit(f'Layers must be between 0 and {num_layers}')

    print(
        f'model={args.model} device={device} blocks={len(blocks)}/{n_loaded} '
        f'pooling={pooling} main=L{main_layer} target=L{target_layer} '
        f'k={args.k_start}:{args.k_end} block_filter_k={args.block_filter_k} '
        f'dtw_cost={args.dtw_cost} gamma={args.gamma:g}'
    )
    rng = np.random.default_rng(args.permutation_seed)
    permutation = rng.permutation(len(k_indices))

    representations = {}
    for lang in langs:
        print(f'Encoding {lang}...', flush=True)
        texts = [block_text(block, lang) for block in blocks]
        base, freq = encode_language(
            model, tok, texts, main_layer, target_layer, k_indices, pooling,
            args.batch_size, args.max_length, device, num_layers,
            getattr(torch, args.coefficient_dtype),
        )
        permuted = freq[:, permutation]
        representations[lang] = {
            'base': base,
            'plain': freq,
            'permuted': permuted,
            'self_plain': self_terms(freq, args, device),
            'self_permuted': self_terms(permuted, args, device),
        }

    rows, hit_log = [], {}
    for lang_a, lang_b in combinations(langs, 2):
        forward, forward_hits = evaluate_direction(
            representations[lang_a], representations[lang_b], args, device,
        )
        backward, backward_hits = evaluate_direction(
            representations[lang_b], representations[lang_a], args, device,
        )
        for condition in forward_hits:
            hit_log.setdefault(condition, []).extend(
                [forward_hits[condition], backward_hits[condition]]
            )
        row = {'pair': f'{lang_a}-{lang_b}'}
        row.update(average_directions(forward, backward))
        rows.append(row)

    metric_names = list(rows[0])[1:]
    mean_row = {'pair': 'mean'}
    mean_row.update({name: float(np.mean([row[name] for row in rows])) for name in metric_names})
    rows.append(mean_row)

    selected_condition = None
    if args.fusion_values:
        selected_alpha = max(
            args.fusion_values,
            key=lambda alpha: (
                mean_row[f'{fusion_condition(alpha)}_p1'],
                mean_row[f'{fusion_condition(alpha)}_mrr'],
                -alpha,
            ),
        )
        selected_condition = fusion_condition(selected_alpha)
        if fusion['mode'] == 'selection':
            interval_note = 'Selection-set intervals are not adjusted for alpha selection.'
        else:
            interval_note = 'Alpha was fixed before evaluation on this corpus.'
        fusion.update({
            'normalization': 'per-query candidate z-score',
            'alpha_grid': args.fusion_values,
            'selected_alpha': selected_alpha,
            'selection_metric': 'macro directed P@1',
            'interval_note': interval_note,
            'curve': {
                str(alpha): {
                    'p1': mean_row[f'{fusion_condition(alpha)}_p1'],
                    'mrr': mean_row[f'{fusion_condition(alpha)}_mrr'],
                }
                for alpha in args.fusion_values
            },
        })

    # Paired block bootstrap of the primary metric against each control.
    comparisons = {
        'dtw_vs_rigid': ('dtw', 'rigid'),
        'dtw_vs_permuted': ('dtw', 'permuted_dtw'),
        'dtw_vs_base': ('dtw', 'base'),
    }
    if selected_condition:
        comparisons.update({
            'fusion_vs_base': (selected_condition, 'base'),
            'fusion_vs_dtw': (selected_condition, 'dtw'),
        })
    intervals = {
        label: paired_bootstrap(
            hit_log[better], hit_log[worse], args.bootstrap, args.permutation_seed,
        )
        for label, (better, worse) in comparisons.items()
    }

    corpus_hash = hashlib.md5(Path(args.corpus).read_bytes()).hexdigest()[:8]
    block_ids = [block.get('block_id', block.get('document_id', i)) for i, block in enumerate(blocks)]
    subset_hash = hashlib.md5(json.dumps(block_ids).encode()).hexdigest()[:8]
    metadata = {
        'model': args.model,
        'hf_model': model_name,
        'corpus': args.corpus,
        'corpus_hash': corpus_hash,
        'n_loaded': n_loaded,
        'n_blocks': len(blocks),
        'block_ids': block_ids,
        'subset_hash': subset_hash,
        'langs': langs,
        'pooling': pooling,
        'main_layer': main_layer,
        'target_layer': target_layer,
        'k_indices': k_indices,
        'n_frequencies': len(k_indices),
        'block_filter_k': args.block_filter_k,
        'candidates': min(args.candidates, len(blocks)),
        'gamma': args.gamma,
        'band_radius': args.band_radius,
        'position_weight': args.position_weight,
        'dtw_cost': args.dtw_cost,
        'permutation': permutation.tolist(),
        'negative_seed': args.negative_seed,
        'bootstrap': args.bootstrap,
        'max_length': args.max_length,
        'coefficient_dtype': args.coefficient_dtype,
    }

    out_dir = Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    fusion_suffix = ''
    if fusion['mode'] == 'selection':
        grid_hash = hashlib.md5(json.dumps(args.fusion_values).encode()).hexdigest()[:6]
        fusion_suffix = f'_fsel{grid_hash}'
    elif fusion['mode'] in ('fixed', 'config'):
        fusion_suffix = f'_{fusion_condition(fusion["selected_alpha"])}'
        if fusion['mode'] == 'config':
            fusion_suffix += f'c{fusion["source_hash"]}'
    gamma_label = f'{args.gamma:g}'.replace('.', 'p')
    stem = (f'{Path(args.corpus).stem}_m{main_layer}_L{target_layer}_k{args.k_start}-{args.k_end}'
            f'_bf{args.block_filter_k}_c{args.dtw_cost}_g{gamma_label}{fusion_suffix}_{subset_hash}_{corpus_hash}')
    csv_path = out_dir / f'{stem}.csv'
    json_path = out_dir / f'{stem}.json'
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump({
            'metadata': metadata,
            'rows': rows,
            'bootstrap_intervals': intervals,
            'fusion': fusion,
        }, handle, indent=2)

    for label, interval in intervals.items():
        print(f"{label:>18}: {interval['difference']:+.4f} "
              f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]")
    if selected_condition:
        print(f'fusion alpha: {fusion["selected_alpha"]:g} ({fusion["mode"]})')
    print(json.dumps(mean_row, indent=2))
    print(f'Saved {csv_path}')
    print(f'Saved {json_path}')


if __name__ == '__main__':
    main()
