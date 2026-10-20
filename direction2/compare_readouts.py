r"""Compare retrieval readouts on the same fine-tuned checkpoints.

A readout is the vector retrieval actually scores. ``pooled`` mean-pools a
hidden layer; ``spectral`` concatenates DCT coefficients at a layer. Comparing
them naively confounds three things at once -- the representation, the layer it
is read at, and its dimensionality -- so this script groups runs by (layer, k)
and only differences arms that match on everything except the one factor named.

Readout is recoverable from the eval CSV name, which encodes it as a suffix:
no suffix means pooled at the layer in config.json (``main_layer``, and None
there means the final layer), ``_poolL<n>`` means pooled at an overridden
layer, and ``_specread[k<list>]`` means spectral.

Run from ``spectral-probing/``::

    python -m direction2.compare_readouts --eval_dir direction2/results/fixedstep_eval
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Encoder:  mbert_L8_low_fk_k1-2-3-4_r16_s42[_specreadk1|_poolL8]_<corpus>.csv
# Decoder:  qwen3-1.7b_L16_none_k1-2-3-4_r16_m16_s42[...]_<corpus>.csv  (carries _m<main>)
RUN_RE = re.compile(
    r'(?P<model>mbert|xlmr|qwen3-[\d.]+b)_L(?P<target>\d+)'
    r'_(?P<mode>none|low_fk|low_rms|spatial|rigid_freq|freq_dtw)'
    r'_k(?P<k>[\d-]+)_r(?P<rank>\d+)(_m(?P<main>\d+))?[^s]*?_s(?P<seed>\d+)'
    r'(?P<extra>.*?)_(?P<corpus>ntrex|un)')
FROZEN_RE = re.compile(
    r'(?P<model>mbert|xlmr|qwen3-[\d.]+b)_frozen(_m(?P<main>\d+))?'
    r'(?P<extra>.*?)_(?P<corpus>ntrex|un)')


def parse_readout(extra, main=None):
    """Name the readout from an eval-CSV filename suffix.

    ``main`` is the run's main_layer, used to label the default pooled readout
    when the filename carries no explicit layer. Decoder runs always record it;
    encoder runs with main_layer=None read the final layer instead.
    """
    extra = extra.replace('_last', '')
    spec = re.search(r'_specread(?:L(\d+))?(k[\d-]+)?', extra)
    if spec:
        layer = f'L{spec.group(1)}' if spec.group(1) else ''
        return f'spectral {layer}{spec.group(2) or "k(cfg)"}'.replace('  ', ' ')
    pool = re.search(r'_poolL(\d+)', extra)
    if pool:
        return f'pooled L{pool.group(1)}'
    return f'pooled L{main}' if main else 'pooled L(final)'


def load(eval_dir, corpus_tag):
    """Return {(model, arm, readout): {seed: {pair: p1}}}."""
    out = defaultdict(dict)
    for path in sorted(Path(eval_dir).glob(f'*{corpus_tag}*.csv')):
        m = RUN_RE.match(path.stem) or FROZEN_RE.match(path.stem)
        if not m:
            continue
        rows = {r['pair']: float(r['p1']) for r in csv.DictReader(open(path))
                if r['pair'] != 'mean'}
        if not rows:
            continue
        gd = m.groupdict()
        arm = f'{gd["mode"]} L{gd["target"]}' if gd.get('mode') else 'frozen'
        seed = gd.get('seed') or '-'
        out[(gd['model'], arm, parse_readout(gd['extra'], gd.get('main')))][seed] = rows
    return out


def paired(a, b, seeds, n_boot, seed):
    """Percentile CI for the mean per-pair difference a - b over shared seeds."""
    keys = sorted(set.intersection(*(set(a[s]) for s in seeds),
                                   *(set(b[s]) for s in seeds)))
    d = np.array([np.mean([a[s][k] for s in seeds]) - np.mean([b[s][k] for s in seeds])
                  for k in keys])
    rng = np.random.default_rng(seed)
    draws = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return (len(d), d.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5),
            int((d > 0).sum()), int((d < 0).sum()))


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval_dir', default='direction2/results/fixedstep_eval')
    p.add_argument('--corpus_tag', default='ntrex_25way')
    p.add_argument('--baseline', default='pooled L8',
                   help='readout every other readout is differenced against')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    """Print per-readout means and paired differences within each arm."""
    args = parse_args()
    runs = load(args.eval_dir, args.corpus_tag)
    if not runs:
        raise SystemExit(f'no {args.corpus_tag} CSVs under {args.eval_dir}')

    readouts = sorted({r for _, _, r in runs})
    arms = sorted({(m, a) for m, a, _ in runs})
    print(f'{args.eval_dir}  ({args.corpus_tag})\n')
    print('  ' + f'{"model / arm":<20}' + ''.join(f'{r:>19}' for r in readouts))
    for model, arm in arms:
        cells = ''
        for r in readouts:
            d = runs.get((model, arm, r))
            cells += (f'{np.mean([np.mean(list(v.values())) for v in d.values()]):>14.4f}'
                      f' (n{len(d)})') if d else f'{"--":>19}'
        print(f'  {model + " " + arm:<20}{cells}')

    print(f'\npaired per-pair bootstrap vs "{args.baseline}", matched seeds and arm:')
    print('  ' + f'{"model / arm":<20}{"readout":<18}{"delta":>9}{"95% CI":>20}{"win/loss":>11}')
    for model, arm in arms:
        base = runs.get((model, arm, args.baseline))
        if not base:
            continue
        for r in readouts:
            other = runs.get((model, arm, r))
            if r == args.baseline or not other:
                continue
            shared = sorted(set(base) & set(other))
            if not shared:
                continue
            n, mean, lo, hi, win, loss = paired(other, base, shared, args.bootstrap, args.seed)
            ci = f'[{lo * 100:+.2f}, {hi * 100:+.2f}]'
            print(f'  {model + " " + arm:<20}{r:<18}{mean * 100:>+8.2f}pp{ci:>20}'
                  f'{f"{win}/{loss}":>11}')

    print('\n  Dimensionality is NOT matched between "pooled" (d) and a multi-k\n'
          '  "spectral" readout (k*d). Only a single-k spectral arm is dimension-matched.')


if __name__ == '__main__':
    main()
