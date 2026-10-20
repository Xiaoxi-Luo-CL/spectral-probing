"""Plot what distinguishes helpful source filters in the related-pair transfer runs.

Per model, two panels: (left) each source's frozen gate as applied to a sentence of --n_words words
(logits average-pooled to that many DCT bins, then sigmoid), shaded by the source's mean gain over
the targets; (right) mean gain against the gate's mean over the lowest frequencies (k = 0..--low_k).
A source whose transfer runs are incomplete is drawn dashed and left out of the scatter.

Usage (from spectral-probing/):
    python -m direction1.transfer.plot_pair_gates
"""
import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from direction1.transfer.analyze_pairs import FROZE, NONE
from direction1.transfer.plot_v3 import load
from direction1.transfer.zero_shot import load_checkpoint_objects, lookup_source_exp

matplotlib.use('Agg')

# light -> dark steps of one blue ramp: darker = larger mean gain
RAMP = ['#86b6ef', '#5598e7', '#3987e5', '#2a78d6', '#1c5cab', '#184f95', '#0d366b']
MODELS = {'bert-base-multilingual-cased': ('bert-base-multilingual-cased', 'avg', 'mBERT'),
          'aya-expanse-8b': ('CohereLabs/aya-expanse-8b', 'last', 'Aya')}


def gate(P, n):
    """Gate the encoder applies to an n-word sentence: logits average-pooled to n bins, then sigmoid."""
    return torch.sigmoid(F.adaptive_avg_pool1d(P[None, None], n)[0, 0]).numpy()


def mean_gain(acc, model, src, targets, n_seeds):
    """Mean gain (pp) of the frozen filter over no filter across targets, or nan if any target is incomplete."""
    out = []
    for t in targets:
        a, b = acc.get((model, src, t, FROZE), {}), acc.get((model, src, t, NONE), {})
        if len(a) < n_seeds:
            return np.nan
        out.append(np.mean([a[s] - b[s] for s in a]))
    return float(np.mean(out))


def main():
    """Draw the gate profiles and the gain-vs-low-frequency-gate scatter for each model."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--results_dir', default='direction1/transfer/results_pairs_n100')
    p.add_argument('--pairs', default='direction1/transfer/related_pairs.json')
    p.add_argument('--n_words', type=int, default=20, help='sentence length the gates are pooled to')
    p.add_argument('--low_k', type=int, default=3, help='highest frequency counted as low in the scatter')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--out', default='paper/fig/pair_gates.pdf')
    args = p.parse_args()

    acc, _ = load(args.results_dir)
    cfg = json.load(open(args.pairs))
    targets = list(cfg['targets'])
    sources = sorted({pr['source'] for pr in cfg['pairs']} | {pr['unrelated'] for pr in cfg['pairs']})
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), gridspec_kw={'width_ratios': [1.35, 1, 1.35, 1]})

    for i, (model, (hf, pool, name)) in enumerate(MODELS.items()):
        ax_g, ax_s = axes[2 * i], axes[2 * i + 1]
        gains = {s: mean_gain(acc, model, s, targets, args.n_seeds) for s in sources}
        gates = {s: gate(load_checkpoint_objects(lookup_source_exp(hf, s, 'relations', pool))['frequency_filter']
                         .detach().cpu().float(), args.n_words) for s in sources}
        done = sorted([s for s in sources if not np.isnan(gains[s])], key=gains.get)
        color = {s: RAMP[round(j * (len(RAMP) - 1) / max(len(done) - 1, 1))] for j, s in enumerate(done)}
        k = np.arange(args.n_words)
        for s in sorted(sources, key=lambda s: -np.nan_to_num(gains[s], nan=-1e9)):  # legend: largest gain first
            label = f'{s} (pending)' if np.isnan(gains[s]) else f'{s} ({gains[s]:+.1f})'
            ax_g.plot(k, gates[s], lw=1.2, color=color.get(s, '#9a9a9a'), ls='-' if s in color else '--', label=label)
        ax_g.set_xlabel(f'DCT frequency $k$ ({args.n_words}-word sentence)')
        ax_g.set_ylabel('gate $\\gamma_A^{(k)}$')
        ax_g.set_title(f'{name}: frozen source gates', fontsize=7)
        ax_g.legend(fontsize=5.5, frameon=False, ncol=4, handlelength=1.2, columnspacing=0.8, labelspacing=0.2,
                    title='source (mean gain, pp)', title_fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.32))

        lo = {s: gates[s][:args.low_k + 1].mean() for s in done}
        span = np.ptp([gains[s] for s in done]) or 1.0
        prev = None
        for s in sorted(done, key=gains.get):
            ax_s.scatter(lo[s], gains[s], s=14, color=color[s], edgecolor='white', linewidth=0.6, zorder=3)
            # a label close above the previous point goes below its own point instead
            near = prev is not None and gains[s] - gains[prev] < 0.08 * span and abs(lo[s] - lo[prev]) < 0.02
            ax_s.annotate(s, (lo[s], gains[s]), xytext=(3, -7 if near else 2), textcoords='offset points', fontsize=6)
            prev = s
        ax_s.set_xlabel(f'mean gate, $k \\leq {args.low_k}$')
        ax_s.set_ylabel('mean gain (pp)')
        ax_s.set_title(f'{name}: gain vs. low-frequency gate', fontsize=7)
        for ax in (ax_g, ax_s):
            ax.grid(color='#e6e6e6', lw=0.4)
            ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout(w_pad=0.8)
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
