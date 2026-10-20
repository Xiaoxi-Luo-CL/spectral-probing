"""Plot the v3 transfer grid (W x gamma settings plus plain, largest feasible n per target).

Two kinds of figure per (model, source language):
  heatmap_{model}_{src}.pdf : test accuracy for all 13 settings, mean over seeds (std below),
                              best trained cell per row in bold
  delta_{model}_{src}.pdf   : each trained setting minus Wscratch_Gnone (a probe trained on the target
                              alone, no W_A or gamma_A), with a dot where all seeds agree in sign
Rows are the target languages sorted by resource tier, labelled with the training n.

Usage (from spectral-probing/):
    python -m direction1.transfer.plot_v3
    python -m direction1.transfer.plot_v3 --results_dir direction1/transfer/results_coverage_v3 --fmt png
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from direction1.transfer.plot_heatmaps import NON_ROMAN_LANGS, RESOURCE_ORDER, TIER_COLORS, resource_level

matplotlib.use('Agg')

W_ORDER = ['froze', 'tune', 'scratch']
G_ORDER = ['none', 'froze', 'tune', 'scratch']
COLS = ['plain'] + [f'W{w}_G{g}' for w in W_ORDER for g in G_ORDER]
BASE = 'Wscratch_Gnone'
DELTA_COLS = [c for c in COLS if c.startswith(('Wtune', 'Wscratch')) and c != BASE]
ZERO_SHOT = {'plain', 'Wfroze_Gnone', 'Wfroze_Gfroze'}


def load(root):
    """Accuracies (%) keyed by (model, src, tgt, setting) -> {seed: acc}, and the training n per run/target."""
    acc, ns = defaultdict(dict), {}
    for p in Path(root).glob('*/*/*/*/seed*/result_*.json'):
        model, pair, _, folder, seed = p.parts[-6:-1]
        src, tgt = pair.split('2')
        r = json.load(open(p))
        acc[(model, src, tgt, re.sub(r'_n\d+$', '', folder))][int(seed[4:])] = 100 * r['test_accuracy']
        if r.get('n_examples'):
            ns[(model, src, tgt)] = r['n_examples']
    return acc, ns


def matrices(acc, model, src, cols):
    """Target order plus mean and std matrices, targets by columns, for one run."""
    tgts = sorted({k[2] for k in acc if k[:2] == (model, src)}, key=lambda t: (RESOURCE_ORDER[resource_level(t)], t))
    mean, std = np.full((len(tgts), len(cols)), np.nan), np.full((len(tgts), len(cols)), np.nan)
    for i, t in enumerate(tgts):
        for j, c in enumerate(cols):
            v = list(acc.get((model, src, t, c), {}).values())
            if v:
                mean[i, j], std[i, j] = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else np.nan)
    return tgts, mean, std


def label_rows(ax, tgts, ns, model, src):
    """Language codes coloured by resource tier (italic = non-Latin script), with the training n."""
    ax.set_yticks(range(len(tgts)))
    ax.set_yticklabels([f'{t}  n={ns.get((model, src, t), "?")}' for t in tgts], fontsize=9)
    for lbl, t in zip(ax.get_yticklabels(), tgts):
        lbl.set_color(TIER_COLORS[resource_level(t)])
        lbl.set_fontstyle('italic' if t in NON_ROMAN_LANGS else 'normal')
    for i in range(1, len(tgts)):  # tier separators
        if resource_level(tgts[i]) != resource_level(tgts[i - 1]):
            ax.axhline(i - 0.5, color='#333333', lw=1.2)


def label_cols(ax, cols):
    """Two-line column labels (W / gamma) and separators between W blocks."""
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c if c == 'plain' else c.replace('_G', '\nγ ').replace('W', 'W ', 1) for c in cols], fontsize=8)
    ax.xaxis.tick_top()
    for j in range(1, len(cols)):
        if cols[j].split('_')[0] != cols[j - 1].split('_')[0]:
            ax.axvline(j - 0.5, color='#333333', lw=1.2)


def plot_heatmap(acc, ns, model, src, out):
    """Absolute accuracy for all 13 settings; zero-shot columns are single-seed."""
    tgts, mean, std = matrices(acc, model, src, COLS)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(tgts) + 2))
    ax.imshow(mean, cmap='YlOrRd', vmin=0, vmax=100, aspect='auto', alpha=0.85)
    trained = [j for j, c in enumerate(COLS) if c not in ZERO_SHOT]
    for i in range(len(tgts)):
        best = trained[int(np.nanargmax(mean[i, trained]))]
        for j in range(len(COLS)):
            bold = j == best
            ax.text(j, i - 0.12, f'{mean[i, j]:.1f}', ha='center', va='center', fontsize=8.5 if bold else 7.5,
                    fontweight='bold' if bold else 'normal')
            if not np.isnan(std[i, j]):
                ax.text(j, i + 0.27, f'±{std[i, j]:.1f}', ha='center', va='center', fontsize=5.5, color='#444444')
    label_rows(ax, tgts, ns, model, src)
    label_cols(ax, COLS)
    ax.set_title(f'{model}, source {src}: test accuracy (%)', fontsize=11, pad=34)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_delta(acc, ns, model, src, out, clip):
    """Each trained setting minus Wscratch_Gnone, paired by seed (same seed = same training sample)."""
    tgts = matrices(acc, model, src, [BASE])[0]
    d = np.full((len(tgts), len(DELTA_COLS)), np.nan)
    agree = np.zeros_like(d, dtype=bool)
    for i, t in enumerate(tgts):
        base = acc[(model, src, t, BASE)]
        for j, c in enumerate(DELTA_COLS):
            diff = [acc[(model, src, t, c)][s] - base[s] for s in base if s in acc[(model, src, t, c)]]
            d[i, j] = np.mean(diff)
            agree[i, j] = len(diff) > 1 and (all(x > 0 for x in diff) or all(x < 0 for x in diff))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(tgts) + 2))
    ax.imshow(d, cmap='RdBu', vmin=-clip, vmax=clip, aspect='auto')
    for i in range(len(tgts)):
        for j in range(len(DELTA_COLS)):
            ax.text(j, i, f'{d[i, j]:+.1f}{"•" if agree[i, j] else ""}', ha='center', va='center', fontsize=7.5,
                    color='white' if abs(d[i, j]) > 0.7 * clip else 'black')
    label_rows(ax, tgts, ns, model, src)
    label_cols(ax, DELTA_COLS)
    means = ' '.join(f'{v:+.1f}' for v in np.nanmean(d, 0))
    ax.set_title(f'{model}, source {src}: setting − {BASE} (pp)\ncolumn means: {means}', fontsize=10, pad=34)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    """Render both figures for every (model, source) run found."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--results_dir', default='direction1/transfer/results_coverage_v3')
    p.add_argument('--output_dir', default='direction1/transfer/figures_v3')
    p.add_argument('--fmt', default='pdf', help='pdf or png')
    p.add_argument('--clip', type=float, default=3.0, help='colour range of the delta plots, in pp')
    args = p.parse_args()
    acc, ns = load(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for model, src in sorted({k[:2] for k in acc}):
        plot_heatmap(acc, ns, model, src, out / f'heatmap_{model}_{src}.{args.fmt}')
        plot_delta(acc, ns, model, src, out / f'delta_{model}_{src}.{args.fmt}', args.clip)
        print(f'  {model} {src} -> {out}')


if __name__ == '__main__':
    main()
