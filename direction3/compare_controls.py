r"""Compare Qwen control conditions on a shared retrieval evaluation.

Every condition is scored on the same language pairs, so the informative
statistic is the paired per-pair difference against a reference condition, not
a difference of condition means. Means computed over different seed subsets are
not comparable at all -- a partially finished grid will silently invite that
mistake -- so conditions are compared only on seeds both arms actually have.

Run from ``spectral-probing/``::

    python -m direction3.compare_controls \\
        --eval_dir direction3/results/control_l24_eval
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# Condition label from a run directory name, e.g. L24_low_fk_k1-2-3-4_r16_m28_s42
RUN_PATTERN = re.compile(
    r'L(?P<target>\d+)_(?P<mode>none|low_fk|low_rms|spatial|rigid_freq|freq_dtw)'
    r'_k(?P<k>[\d-]+)_r(?P<rank>\d+)_m(?P<main>\d+).*_s(?P<seed>\d+)'
)


def condition_label(match):
    """Name a condition by what distinguishes it, ignoring seed."""
    mode, main, target = match['mode'], match['main'], match['target']
    if mode == 'none':
        return f'none (main L{main})'
    if mode == 'spatial':
        return f'spatial last@L{target} (main L{main})'
    return f'{mode} k={match["k"].replace("-", ",")}@L{target} (main L{main})'


def load_evals(eval_dir, corpus_tag):
    """Return {condition: {seed: {pair: p1}}} for one corpus."""
    runs = defaultdict(dict)
    for path in sorted(Path(eval_dir).glob(f'*{corpus_tag}*.csv')):
        match = RUN_PATTERN.search(path.stem)
        if not match:
            continue
        rows = {r['pair']: float(r['p1']) for r in csv.DictReader(open(path))
                if r['pair'] != 'mean'}
        if rows:
            runs[condition_label(match)][match['seed']] = rows
    return runs


def paired_bootstrap(a_runs, b_runs, seeds, n_resamples, seed):
    """Percentile interval for the mean per-pair difference a - b.

    Both arms are averaged over the same seeds first, then the same resampled
    pairs index both, which is what makes the interval paired.
    """
    pairs = sorted(set.intersection(*(set(a_runs[s]) for s in seeds),
                                    *(set(b_runs[s]) for s in seeds)))
    delta = np.array([
        np.mean([a_runs[s][p] for s in seeds]) - np.mean([b_runs[s][p] for s in seeds])
        for p in pairs
    ])
    rng = np.random.default_rng(seed)
    draws = np.array([
        delta[rng.integers(0, len(delta), len(delta))].mean()
        for _ in range(n_resamples)
    ])
    return {
        'n_pairs': len(delta), 'mean': float(delta.mean()),
        'lo': float(np.percentile(draws, 2.5)),
        'hi': float(np.percentile(draws, 97.5)),
        'win': int((delta > 0).sum()), 'loss': int((delta < 0).sum()),
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval_dir', default='direction3/results/control_l24_eval')
    parser.add_argument('--corpus_tag', default='ntrex_25way')
    parser.add_argument('--reference', default=None,
                        help='condition to compare against (default: the none/main-L28 arm)')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    """Report per-seed values and matched-seed paired comparisons."""
    args = parse_args()
    runs = load_evals(args.eval_dir, args.corpus_tag)
    if not runs:
        raise SystemExit(f'No {args.corpus_tag} CSVs matched under {args.eval_dir}')
    print('NOTE: run names carry layers and k, not batch size, sequence length or\n'
          '      step policy. Everything compared here is assumed to share those.\n'
          '      Point --eval_dir at one grid; do not mix grids in a results root.\n')

    seeds = sorted({s for arm in runs.values() for s in arm})
    print(f'{args.eval_dir}  ({args.corpus_tag})\n')
    header = f'  {"condition":<34}' + ''.join(f's{s:>8}' for s in seeds) + f'{"n":>4}'
    print(header)
    for label in sorted(runs):
        cells = ''.join(
            f'{np.mean(list(runs[label][s].values())):>9.4f}' if s in runs[label]
            else f'{"--":>9}' for s in seeds
        )
        print(f'  {label:<34}{cells}{len(runs[label]):>4}')

    reference = args.reference
    if reference is None:
        # Default to the arm the report treats as best: the bare main loss at the
        # layer with the most completed seeds, preferring L24 where present.
        bare = {k: v for k, v in runs.items() if k.startswith('none (main')}
        if bare:
            preferred = [k for k in bare if 'L24' in k] or list(bare)
            reference = max(preferred, key=lambda k: len(runs[k]))
        else:
            reference = None
    if reference is None or reference not in runs:
        print('\nNo reference condition present; skipping paired comparisons.')
        return

    print(f'\npaired per-pair bootstrap vs "{reference}", matched seeds only:')
    print(f'  {"condition":<34}{"seeds":>10}{"delta":>9}{"95% CI":>20}{"win/loss":>11}')
    for label in sorted(runs):
        if label == reference:
            continue
        shared = sorted(set(runs[label]) & set(runs[reference]))
        if not shared:
            print(f'  {label:<34}{"none":>10}{"--":>9}{"--":>20}{"--":>11}')
            continue
        st = paired_bootstrap(runs[label], runs[reference], shared,
                              args.bootstrap, args.seed)
        ci = f'[{st["lo"] * 100:+.2f}, {st["hi"] * 100:+.2f}]'
        record = f'{st["win"]}/{st["loss"]}'
        print(f'  {label:<34}{",".join(shared):>10}{st["mean"] * 100:>+8.2f}pp'
              f'{ci:>20}{record:>11}')

    incomplete = [k for k in runs if len(runs[k]) < len(seeds)]
    if incomplete:
        print(f'\n  incomplete ({len(seeds)} seeds expected): ' + '; '.join(incomplete))
        print('  Condition means above are NOT comparable across different seed subsets.')


if __name__ == '__main__':
    main()
