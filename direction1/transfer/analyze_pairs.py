"""Analyse the related-language transfer runs: does a related source's spectral profile help more?

For each (model, source, target), delta = Acc(fresh probe, frozen source filter) - Acc(fresh probe, no filter),
paired by seed (same seed = same training sample, initial probe and lr grid). The primary test, per model, is
delta_related - delta_unrelated over the pairs in related_pairs.json, also paired by seed (only seeds both sources
have; incomplete pairs are listed), with an exact sign-flip test across pairs.
Also prints the full source x target delta matrix, saves it as a heatmap, and checks that the no-filter probe,
which never sees the source, scores the same for every source.

Usage (from spectral-probing/):
    python -m direction1.transfer.analyze_pairs
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from direction1.transfer.plot_v3 import load

matplotlib.use('Agg')

NONE, FROZE = 'Wscratch_Gnone', 'Wscratch_Gfroze'


def delta(acc, model, src, tgt):
    """{seed: accuracy gain (pp) of the frozen source filter over no filter}."""
    a, b = acc.get((model, src, tgt, FROZE), {}), acc.get((model, src, tgt, NONE), {})
    return {s: a[s] - b[s] for s in sorted(a) if s in b}


def paired(d1, d2):
    """Per-seed d1 - d2 over the seeds both have, so a missing seed cannot shift one side only."""
    return np.array([d1[s] - d2[s] for s in sorted(d1) if s in d2])


def sign_flip_p(d):
    """Two-sided exact p-value of mean(d) under random sign flips of the per-pair differences."""
    flips = [np.mean(d * np.array(s)) for s in itertools.product([1, -1], repeat=len(d))]
    return float(np.mean(np.abs(flips) >= abs(np.mean(d)) - 1e-12))


def fmt(d):
    """Mean +- seed std of per-seed values, or a dash when missing."""
    d = np.array(list(d.values())) if isinstance(d, dict) else d
    return f'{d.mean():+.2f} ± {d.std(ddof=1):.2f}' if len(d) > 1 else ('–' if not len(d) else f'{d.mean():+.2f}')


def num(v):
    """Signed number, or a dash when missing."""
    return '–' if np.isnan(v) else f'{v:+.2f}'


def main():
    """Print the pair tables and write one delta heatmap per model."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--results_dir', default='direction1/transfer/results_pairs_n100')
    p.add_argument('--pairs', default='direction1/transfer/related_pairs.json')
    p.add_argument('--output_dir', default='direction1/transfer/figures_pairs')
    p.add_argument('--n_seeds', type=int, default=5, help='seeds a complete pair has')
    args = p.parse_args()

    acc, _ = load(args.results_dir)
    cfg = json.load(open(args.pairs))
    models = sorted({k[0] for k in acc})
    sources = sorted({k[1] for k in acc})
    targets = list(cfg['targets'])

    for model in models:
        print(f'\n## {model}\n')
        print('| pair | Δ related | Δ unrelated | Δ en | related − unrelated (paired) | seeds |')
        print('|---|---|---|---|---|---|')
        diffs, incomplete = [], []
        for pair in cfg['pairs']:
            s, t, u = pair['source'], pair['target'], pair['unrelated']
            rel, unrel, en = delta(acc, model, s, t), delta(acc, model, u, t), delta(acc, model, 'en', t)
            pd = paired(rel, unrel)
            diff = pd.mean() if len(pd) else np.nan
            diffs.append(diff)
            if len(pd) < args.n_seeds:
                incomplete.append(f'{s}→{t} ({len(pd)}/{args.n_seeds} seeds)')
            print(f'| {s}→{t} (vs {u}) | {fmt(rel)} | {fmt(unrel)} | {fmt(en)} | {fmt(pd)} | {len(pd)} |')
        if incomplete:
            print(f'\nincomplete pairs (shared seeds only): {", ".join(incomplete)}')
        d = np.array([x for x in diffs if not np.isnan(x)])
        if len(d):
            print(f'\nrelated − unrelated over {len(d)} pairs: mean {d.mean():+.2f} pp, positive in {(d > 0).sum()}/{len(d)}, '
                  f'sign-flip p = {sign_flip_p(d):.3f}')

        # full source x target matrix of mean deltas
        mat = np.array([[np.mean(list(delta(acc, model, s, t).values())) if delta(acc, model, s, t) else np.nan
                         for t in targets] for s in sources])
        print('\n| source \\ target | ' + ' | '.join(targets) + ' |')
        print('|---|' + '---|' * len(targets))
        for s, row in zip(sources, mat):
            print(f'| {s} | ' + ' | '.join(num(v) for v in row) + ' |')

        # the no-filter probe ignores the source, so its accuracy must agree across sources
        spread = [np.ptp([acc[(model, s, t, NONE)][seed] for s in sources if seed in acc.get((model, s, t, NONE), {})])
                  for t in targets for seed in {sd for s in sources for sd in acc.get((model, s, t, NONE), {})}]
        print(f'\nno-filter accuracy spread across sources (same target and seed): max {max(spread):.3f} pp')

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(1.1 * len(targets) + 2, 0.6 * len(sources) + 1.5))
        clip = np.nanmax(np.abs(mat)) or 1.0
        ax.imshow(mat, cmap='RdBu', vmin=-clip, vmax=clip, aspect='auto')
        related = {(pr['source'], pr['target']) for pr in cfg['pairs']}
        for i, s in enumerate(sources):
            for j, t in enumerate(targets):
                bold = (s, t) in related
                ax.text(j, i, f'{mat[i, j]:+.2f}', ha='center', va='center', fontsize=9,
                        fontweight='bold' if bold else 'normal')
        ax.set_xticks(range(len(targets)), targets)
        ax.set_yticks(range(len(sources)), sources)
        ax.set_xlabel('target')
        ax.set_ylabel('source of the frozen filter')
        ax.set_title(f'{model}: frozen source filter − no filter (pp), related pairs in bold', fontsize=10)
        out = Path(args.output_dir) / f'pairs_delta_{model}.pdf'
        fig.savefig(out, bbox_inches='tight')
        plt.close(fig)
        print(f'saved {out}')


if __name__ == '__main__':
    main()
