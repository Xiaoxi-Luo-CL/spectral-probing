"""Plot absolute accuracy per target language in the related-pair transfer runs.

For each target and model: the no-filter probe (open circle), the range of the frozen-filter probe over
source languages (bar), and its mean over sources (filled circle). Accuracies are means over seeds; only
sources whose runs are complete for every target are used.

Usage (from spectral-probing/):
    python -m direction1.transfer.plot_pair_accuracy
"""
import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from direction1.transfer.analyze_pairs import FROZE, NONE
from direction1.transfer.plot_v3 import load

matplotlib.use('Agg')

# categorical slots 1 and 2 of the reference palette, one per model
MODELS = {'bert-base-multilingual-cased': ('mBERT', '#2a78d6'), 'aya-expanse-8b': ('Aya', '#eb6834')}


def main():
    """Draw one row per target with both models' no-filter and frozen-filter accuracies."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--results_dir', default='direction1/transfer/results_pairs_n100')
    p.add_argument('--pairs', default='direction1/transfer/related_pairs.json')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--out', default='paper/fig/pair_accuracy.pdf')
    args = p.parse_args()

    acc, _ = load(args.results_dir)
    cfg = json.load(open(args.pairs))
    targets = list(cfg['targets'])
    sources = sorted({pr['source'] for pr in cfg['pairs']} | {pr['unrelated'] for pr in cfg['pairs']})
    plt.rcParams.update({'font.size': 7, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5})
    fig, ax = plt.subplots(figsize=(3.1, 2.2))

    mean = lambda k: np.mean(list(acc[k].values()))  # noqa: E731
    for i, (model, (name, color)) in enumerate(MODELS.items()):
        done = [s for s in sources if all(len(acc.get((model, s, t, FROZE), {})) >= args.n_seeds for t in targets)]
        for j, t in enumerate(targets):
            y = j + (i - 0.5) * 0.32  # the two models sit just above and below the target's row
            base = mean((model, done[0], t, NONE))
            d1 = [mean((model, s, t, FROZE)) for s in done]
            ax.plot([min(d1), max(d1)], [y, y], color=color, lw=2.2, solid_capstyle='round', alpha=0.45)
            ax.scatter(base, y, s=16, facecolor='white', edgecolor=color, linewidth=1.0, zorder=3,
                       label=f'{name}, no filter' if j == 0 else None)
            ax.scatter(np.mean(d1), y, s=12, color=color, edgecolor='white', linewidth=0.5, zorder=4,
                       label=f'{name}, frozen $\\gamma_A$' if j == 0 else None)
    ax.set_yticks(range(len(targets)), targets)
    ax.invert_yaxis()
    ax.set_xlabel('test accuracy (%)')
    ax.grid(axis='x', color='#e6e6e6', lw=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=5.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, handletextpad=0.3,
              columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches='tight')
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
