"""Analyses of the v3 transfer grid (results_coverage_v3), one section per question.

Sections (all by default, or pick with --section):
  means     mean test accuracy of every setting per (model, source), over the 24 targets
  paired    paired setting comparisons over targets: mean pp, 95% bootstrap CI (* = excludes 0), wins
  best      which setting is best per target, and how often a target-only probe wins against chance
  majority  majority-class accuracy of each target test set next to the zero-shot settings
  tiers     gamma_A gains grouped by resource tier, pretraining support and n, few-shot and zero-shot
  gamma     within each probe option W, each filter form minus no filter (Holm-corrected, consistency over runs)
  baselines every setting against plain (zero-shot) or Wscratch_Gnone (few-shot)
  pertarget per source and model, the number of targets (of 24) where a comparison is significant over seeds
  zeroshot  Wfroze_Gfroze vs plain per target, with a conservative token-level z-test
  select    best source-reusing few-shot setting vs Wscratch_Gnone, chosen on test or on the other targets
  bug       v1 (raw-logit frozen filter) vs v2 (sigmoid gate) vs v3 for d1 = Wscratch_Gfroze

Accuracies are in %; trained settings are averaged over seeds before comparing, zero-shot ones have seed 42 only.

Usage (from spectral-probing/):
    python -m direction1.transfer.analyze_v3
    python -m direction1.transfer.analyze_v3 --section paired --section tiers
"""
import argparse
import collections
import glob
import json
import re

import numpy as np
import torch
from scipy.stats import binom, binomtest, norm, ttest_1samp, wilcoxon

from direction1.transfer.plot_heatmaps import LANG_SUPPORT, resource_level
from direction1.transfer.plot_v3 import COLS, load
from direction1.transfer.zero_shot import load_checkpoint_objects, lookup_source_exp, split_target_data
from utils.datasets import LabelledDataset

MODELS = {'bert-base-multilingual-cased': ('mBERT', 'bert-base-multilingual-cased', 'avg'),
          'xlm-roberta-base': ('XLM-R', 'FacebookAI/xlm-roberta-base', 'avg'),
          'Qwen2.5-7B': ('Qwen', 'Qwen/Qwen2.5-7B', 'last'),
          'aya-expanse-8b': ('Aya', 'CohereLabs/aya-expanse-8b', 'last')}
SRCS = ('en', 'zh')
TRAINED = [c for c in COLS if c not in ('plain', 'Wfroze_Gnone', 'Wfroze_Gfroze')]
FREE = ['Wscratch_Gnone', 'Wscratch_Gscratch']  # use neither W_A nor gamma_A
CMP = [('Wfroze_Gfroze', 'plain'), ('Wfroze_Gfroze', 'Wfroze_Gnone'), ('Wfroze_Gnone', 'plain'),
       ('Wscratch_Gfroze', 'Wscratch_Gnone'), ('Wscratch_Gtune', 'Wscratch_Gnone'),
       ('Wscratch_Gscratch', 'Wscratch_Gnone'), ('Wscratch_Gfroze', 'Wscratch_Gscratch'),
       ('Wtune_Gfroze', 'Wtune_Gnone'), ('Wtune_Gtune', 'Wtune_Gnone'), ('Wtune_Gscratch', 'Wtune_Gnone'),
       ('Wtune_Gnone', 'Wscratch_Gnone'), ('Wtune_Gfroze', 'Wscratch_Gfroze')]
V1_COND = {'e': 'd1', 'f': 'd2', 'g': 'e1'}  # v1/v2 letters: e = Wscratch_Gfroze, f = Wtune_Gfroze, g = Wscratch_Gnone
RNG = np.random.default_rng(0)
SECTIONS = ['means', 'paired', 'best', 'majority', 'tiers', 'gamma', 'baselines', 'pertarget', 'zeroshot', 'select', 'bug']


def runs(acc):
    """(model dir, source) pairs present, in MODELS order."""
    return [(m, s) for m in MODELS for s in SRCS if any(k[:2] == (m, s) for k in acc)]


def tgts(acc, m, s):
    """Target languages of one run."""
    return sorted({k[2] for k in acc if k[:2] == (m, s)})


def ci(x, n=10000):
    """Mean with a 95% bootstrap CI over targets; '*' marks a CI that excludes 0."""
    lo, hi = np.percentile(RNG.choice(x, (n, len(x))).mean(1), [2.5, 97.5])
    return x.mean(), lo, hi, '*' if lo > 0 or hi < 0 else ''


def header(runs_):
    """Markdown header row over the runs."""
    names = [f'{MODELS[m][0]} {s}' for m, s in runs_]
    return '| | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names)


def means(mean, acc):
    """Setting means over targets."""
    rs = runs(acc)
    print(header(rs))
    for st in COLS:
        print(f'| {st} | ' + ' | '.join(f'{np.mean([mean[(m, s, t, st)] for t in tgts(acc, m, s)]):.1f}' for m, s in rs) + ' |')


def paired(mean, acc):
    """Paired comparisons over targets."""
    rs = runs(acc)
    print(header(rs))
    for a, b in CMP:
        cells = []
        for m, s in rs:
            d = np.array([mean[(m, s, t, a)] - mean[(m, s, t, b)] for t in tgts(acc, m, s)])
            mu, _, _, star = ci(d)
            cells.append(f'{mu:+.1f}{star} ({(d > 0).sum()}/{len(d)})')
        print(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')


def best(mean, acc):
    """Best trained setting per target, and target-only wins against a uniform-chance null."""
    cells = [(m, s, t) for m, s in runs(acc) for t in tgts(acc, m, s)]
    top = {c: max(TRAINED, key=lambda st: mean[c + (st,)]) for c in cells}
    for m, s in runs(acc):
        cnt = collections.Counter(top[(m, s, t)] for t in tgts(acc, m, s))
        print(f'{MODELS[m][0]} {s}: ' + ', '.join(f'{st} {n}' for st, n in cnt.most_common()))
    n = len(cells)
    for group in (['Wscratch_Gnone'], FREE):
        k = sum(top[c] in group for c in cells)
        p0 = len(group) / len(TRAINED)
        print(f'{" or ".join(group)} best in {k}/{n} cells; chance {p0:.0%} -> P(X <= {k}) = {binom.cdf(k, n, p0):.1e}')


def majority(mean, acc):
    """Majority-class share of each test set (labels the source probe can score) vs zero-shot accuracy."""
    targets = {t['lang']: t for t in json.load(open('direction1/transfer/coverage_targets.json'))}
    tests = {lang: split_target_data(LabelledDataset.from_path(t['path']), t['n_test'])[1] for lang, t in targets.items()}
    print('| run | majority | plain | froze/none | froze/froze | froze/froze below majority |\n|---|---|---|---|---|---|')
    for m, s in runs(acc):
        _, hf, pool = MODELS[m]
        cls = set(load_checkpoint_objects(lookup_source_exp(hf, s, 'relations', pool))['classes'])
        maj = {}
        for lang, d in tests.items():
            c = collections.Counter(x for sent in d._labels for x in sent if x in cls)
            maj[lang] = 100 * c.most_common(1)[0][1] / sum(c.values())
        z = {st: np.mean([mean[(m, s, t, st)] for t in tests]) for st in ('plain', 'Wfroze_Gnone', 'Wfroze_Gfroze')}
        below = sum(mean[(m, s, t, 'Wfroze_Gfroze')] < maj[t] for t in tests)
        print(f'| {MODELS[m][0]} {s} | {np.mean(list(maj.values())):.1f} | {z["plain"]:.1f} | {z["Wfroze_Gnone"]:.1f} '
              f'| {z["Wfroze_Gfroze"]:.1f} | {below}/{len(tests)} |')


def tiers(mean, acc, ns):
    """gamma_A gains pooled over runs, grouped by target tier, pretraining support and n."""
    groups = {'tier': lambda c: resource_level(c[2]), 'support': lambda c: LANG_SUPPORT[c[2]], 'n': lambda c: ns[c[:3]],
              'model': lambda c: 'decoder' if MODELS[c[0]][2] == 'last' else 'encoder'}
    cells = [(m, s, t) for m, s in runs(acc) for t in tgts(acc, m, s)]
    for a, b in [('Wscratch_Gfroze', 'Wscratch_Gnone'), ('Wscratch_Gtune', 'Wscratch_Gnone'),
                 ('Wtune_Gfroze', 'Wtune_Gnone'), ('Wfroze_Gfroze', 'plain'), ('Wfroze_Gfroze', 'Wfroze_Gnone')]:
        print(f'\n{a} − {b}')
        for name, fn in groups.items():
            g = collections.defaultdict(list)
            for c in cells:
                g[fn(c)].append(mean[c + (a,)] - mean[c + (b,)])
            parts = []
            for k, v in sorted(g.items(), key=lambda x: str(x[0])):
                mu, lo, hi, _ = ci(np.array(v), 5000)
                parts.append(f'{k}: {mu:+.2f} [{lo:+.2f},{hi:+.2f}] N={len(v)}')
            print(f'  {name:8s} ' + ' | '.join(parts))


def holm(ps):
    """Holm-adjusted p-values."""
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def diff_table(mean, acc, rows):
    """Paired a − b per run (Wilcoxon over targets, Holm across rows) and its consistency across runs.

    rows: (label, settings averaged as a, baseline b). A run cell is mean pp, '*' if Holm p < 0.05, and wins/24;
    the last columns pool the runs: runs with a positive mean, runs significant, sign-test p, encoder/decoder means.
    """
    rs = runs(acc)
    d = {(lbl, r): np.array([np.mean([mean[r + (t, x)] for x in a]) - mean[r + (t, b)] for t in tgts(acc, *r)])
         for lbl, a, b in rows for r in rs}
    p = {}
    for r in rs:  # Holm within each run, across the rows of this table
        raw = [wilcoxon(d[(lbl, r)]).pvalue if np.any(d[(lbl, r)]) else 1.0 for lbl, _, _ in rows]
        p.update({(lbl, r): q for (lbl, _, _), q in zip(rows, holm(np.array(raw)))})
    extra = ['runs > 0', 'runs *', 'sign p', 'enc', 'dec']
    names = [f'{MODELS[m][0]} {s}' for m, s in rs] + extra
    print('| | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names))
    for lbl, _, _ in rows:
        mus = np.array([d[(lbl, r)].mean() for r in rs])
        cells = [f'{d[(lbl, r)].mean():+.1f}{"*" if p[(lbl, r)] < 0.05 else ""} ({(d[(lbl, r)] > 0).sum()}/{len(d[(lbl, r)])})'
                 for r in rs]
        enc = np.mean([mu for mu, (m, _) in zip(mus, rs) if MODELS[m][2] == 'avg'])
        dec = np.mean([mu for mu, (m, _) in zip(mus, rs) if MODELS[m][2] == 'last'])
        sig = sum(p[(lbl, r)] < 0.05 and d[(lbl, r)].mean() > 0 for r in rs)
        sign_p = binomtest(int((mus > 0).sum()), len(mus)).pvalue
        print(f'| {lbl} | ' + ' | '.join(cells) + f' | {(mus > 0).sum()}/{len(rs)} | {sig}/{len(rs)} | {sign_p:.3f} '
              f'| {enc:+.2f} | {dec:+.2f} |')


def gamma(mean, acc):
    """Within each probe option W, does adding a filter (in any form) beat no filter?"""
    rows = []
    for w in ('froze', 'tune', 'scratch'):
        base = f'W{w}_Gnone'
        rows += [(f'W{w}: γ {g} − none', [f'W{w}_G{g}'], base) for g in ('froze', 'tune', 'scratch')]
        rows.append((f'W{w}: γ_A (froze, tune avg) − none', [f'W{w}_Gfroze', f'W{w}_Gtune'], base))
    print('Wfroze rows: none and γ froze are zero-shot; γ tune / scratch train only the filter on the target.\n')
    diff_table(mean, acc, rows)


def baselines(mean, acc):
    """Settings against the two common probing setups: plain (zero-shot) and Wscratch_Gnone (few-shot)."""
    print('### zero-shot vs plain\n')
    diff_table(mean, acc, [(f'{a} − plain', [a], 'plain') for a in ('Wfroze_Gnone', 'Wfroze_Gfroze')])
    print('\n### few-shot vs Wscratch_Gnone (Wfroze_Gtune / Gscratch left out: the frozen W_A keeps them near 40%)\n')
    few = [c for c in TRAINED if c.startswith(('Wtune', 'Wscratch')) and c != 'Wscratch_Gnone']
    diff_table(mean, acc, [(f'{a} − Wscratch_Gnone', [a], 'Wscratch_Gnone') for a in few])


def target_test(acc, cell, a, b):
    """Two-sided p of a − b on one target over seeds, and the mean difference.

    Trained settings pair by seed (same seed = same training sample). Zero-shot settings train nothing and
    exist for seed 42 only, so a trained setting is tested against that constant; two zero-shot settings
    have no seed variation and get p = nan.
    """
    A, B = acc[cell + (a,)], acc[cell + (b,)]
    if len(A) > 1 and len(B) > 1:
        seeds = sorted(set(A) & set(B))
        x = np.array([A[s] - B[s] for s in seeds])
    elif len(A) > 1 or len(B) > 1:
        x = np.array(list(A.values())) - np.mean(list(B.values())) if len(A) > 1 else \
            np.mean(list(A.values())) - np.array(list(B.values()))
    else:
        return np.nan, np.mean(list(A.values())) - np.mean(list(B.values()))
    p = ttest_1samp(x, 0).pvalue if np.ptp(x) > 0 else (0.0 if x[0] != 0 else 1.0)
    return p, x.mean()


def count_table(acc, rows, src):
    """Per source: rows of (label, a, b), one column per model; a cell counts targets with a − b significant."""
    ms = [m for m in MODELS if any(k[:2] == (m, src) for k in acc)]
    print(f'| source {src} | ' + ' | '.join(MODELS[m][0] for m in ms) + ' |\n|---|' + '---|' * len(ms))
    for lbl, a, b in rows:
        cells = []
        for m in ms:
            res = [target_test(acc, (m, src, t), a, b) for t in tgts(acc, m, src)]
            if all(np.isnan(p) for p, _ in res):  # no seed variation on either side
                cells.append(f'– ({sum(d > 0 for _, d in res)}/{len(res)} > 0)')
                continue
            up = sum(p < 0.05 and d > 0 for p, d in res)
            down = sum(p < 0.05 and d < 0 for p, d in res)
            cells.append(f'{up}/{len(res)}' + (f' (↓{down})' if down else ''))
        print(f'| {lbl} | ' + ' | '.join(cells) + ' |')


def pertarget(mean, acc):
    """Each of the 24 targets as its own test: in how many targets is the difference significant?"""
    print('A cell is the number of targets (of 24) where the filter setting is significantly better (p < 0.05, '
          'paired over 3 seeds, uncorrected); ↓k counts targets significantly worse. – = both sides zero-shot, '
          'so no seed variation to test (count of targets with a positive difference instead).\n')
    rows = [(f'W{w} + γ {g} vs W{w} + none', f'W{w}_G{g}', f'W{w}_Gnone')
            for w in ('froze', 'tune', 'scratch') for g in ('froze', 'tune', 'scratch')]
    base = [('Wfroze_Gnone vs plain', 'Wfroze_Gnone', 'plain'), ('Wfroze_Gfroze vs plain', 'Wfroze_Gfroze', 'plain')]
    base += [(f'{a} vs Wscratch_Gnone', a, 'Wscratch_Gnone') for a in
             ('Wtune_Gnone', 'Wtune_Gfroze', 'Wtune_Gtune', 'Wtune_Gscratch')]
    for src in SRCS:
        print(f'\n### filter vs no filter, source {src}\n')
        count_table(acc, rows, src)
    for src in SRCS:
        print(f'\n### against the common setups (fresh-W rows are the W scratch rows above), source {src}\n')
        count_table(acc, base, src)


def zeroshot(mean, acc):
    """Wfroze_Gfroze vs plain per target, with a conservative token-level test.

    Both settings are deterministic and score the same test tokens (labels in the shared source label set),
    so seeds cannot test them. Treating the two accuracies as independent proportions over those N tokens
    overstates the variance of their paired difference, so a target that passes this z-test is significant.
    """
    targets = {t['lang']: t for t in json.load(open('direction1/transfer/coverage_targets.json'))}
    tests = {lang: split_target_data(LabelledDataset.from_path(t['path']), t['n_test'])[1] for lang, t in targets.items()}
    print('| run | targets up / down | significant up / down | mean Δ | targets where Wfroze_Gfroze < plain (Δ pp, * = significant) |')
    print('|---|---|---|---|---|')
    for m, s in runs(acc):
        _, hf, pool = MODELS[m]
        cls = set(load_checkpoint_objects(lookup_source_exp(hf, s, 'relations', pool))['classes'])
        res = {}
        for t in tgts(acc, m, s):
            n = sum(x in cls for sent in tests[t]._labels for x in sent)
            a, b = mean[(m, s, t, 'Wfroze_Gfroze')] / 100, mean[(m, s, t, 'plain')] / 100
            se = np.sqrt((a * (1 - a) + b * (1 - b)) / n)
            res[t] = (100 * (a - b), 2 * norm.sf(abs(a - b) / se))
        d = np.array([v[0] for v in res.values()])
        up = sum(dv > 0 and p < 0.05 for dv, p in res.values())
        down = sum(dv < 0 and p < 0.05 for dv, p in res.values())
        neg = ', '.join(f'{t} {dv:+.1f}{"*" if p < 0.05 else ""}' for t, (dv, p) in sorted(res.items(), key=lambda x: x[1][0])
                        if dv < 0)
        print(f'| {MODELS[m][0]} {s} | {(d > 0).sum()} / {(d < 0).sum()} | {up} / {down} | {d.mean():+.2f} | {neg or "none"} |')


def select(mean, acc):
    """Best of the source-reusing few-shot settings vs Wscratch_Gnone, per target.

    'best on test' picks each target's best setting by its own test accuracy, which favours it by construction
    (the maximum of noisy scores); 'chosen on other targets' picks, for each target, the setting with the best
    mean over the other 23 targets of the run, an unbiased recipe a practitioner could follow.
    """
    cand = [c for c in TRAINED if c.startswith(('Wtune', 'Wscratch')) and c != 'Wscratch_Gnone']
    print(f'candidates: {", ".join(cand)}')
    for src in SRCS:
        ms = [m for m in MODELS if any(k[:2] == (m, src) for k in acc)]
        print(f'\n| source {src} | ' + ' | '.join(MODELS[m][0] for m in ms) + ' |\n|---|' + '---|' * len(ms))
        for how in ('best on test', 'chosen on other targets'):
            cells = []
            for m in ms:
                ts = tgts(acc, m, src)
                res = []
                for t in ts:
                    pool = [x for x in ts if x != t] if how != 'best on test' else [t]
                    pick = max(cand, key=lambda c: np.mean([mean[(m, src, x, c)] for x in pool]))
                    res.append(target_test(acc, (m, src, t), pick, 'Wscratch_Gnone'))
                up = sum(p < 0.05 and d > 0 for p, d in res)
                down = sum(p < 0.05 and d < 0 for p, d in res)
                cells.append(f'{up}/{len(res)}' + (f' (↓{down})' if down else '') + f', Δ {np.mean([d for _, d in res]):+.2f}')
            print(f'| {how} | ' + ' | '.join(cells) + ' |')


def load_letters(root):
    """Seed-mean accuracy (%) of the v1/v2 letter-condition grids, keyed (model, src, tgt, letter, n)."""
    acc = collections.defaultdict(list)
    for p in glob.glob(f'{root}/*/*/relations/cond*/**/result_*.json', recursive=True):
        parts = p[len(root):].strip('/').split('/')
        c = re.fullmatch(r'cond([a-h])(?:_n(\d+))?', parts[3])
        src, tgt = parts[1].split('2')
        acc[(parts[0], src, tgt, c.group(1), int(c.group(2) or 0))].append(100 * json.load(open(p))['test_accuracy'])
    return {k: np.mean(v) for k, v in acc.items()}


def bug(mean, acc, ns):
    """How much the v1 raw-logit frozen filter inflated d1, against its gain k = mean(P) / mean(sigmoid(P))."""
    v1 = load_letters('direction1/transfer/results_coverage')
    v2 = load_letters('direction1/transfer/results_coverage_v2')
    print('| run | k | raw max/min | negative logits | n | v1 d1 − v2 d1 | v1 e1 − v2 e1 |\n|---|---|---|---|---|---|---|')
    for m, s in runs(acc):
        _, hf, pool = MODELS[m]
        P = load_checkpoint_objects(lookup_source_exp(hf, s, 'relations', pool))['frequency_filter'].detach().cpu()
        S = torch.sigmoid(P)
        for n in (50, 100, 500, 1000):
            ts = [k[2] for k in v1 if k[:2] == (m, s) and k[3:] == ('e', n) and (m, s, k[2], 'e', n) in v2]
            if ts:
                d = {c: np.mean([v1[(m, s, t, c, n)] - v2[(m, s, t, c, n)] for t in ts]) for c in 'eg'}
                print(f'| {MODELS[m][0]} {s} | {(P.mean() / S.mean()):.2f} | {(P.max() / P.min()):.1f} | {(P < 0).sum().item()} '
                      f'| {n} | {d["e"]:+.1f} | {d["g"]:+.1f} |')
    print('\nSame targets at the v3 n: 10-epoch v2/v1 grids against converged v3')
    print('| run | v2 e1 | v2 d1 | v1 d1 (bug) | v3 e1 | v3 d1 |\n|---|---|---|---|---|---|')
    for m, s in runs(acc):
        ts = tgts(acc, m, s)
        old = lambda d, c: np.mean([d[(m, s, t, c, ns[(m, s, t)])] for t in ts])
        new = lambda st: np.mean([mean[(m, s, t, st)] for t in ts])
        print(f'| {MODELS[m][0]} {s} | {old(v2, "g"):.1f} | {old(v2, "e"):.1f} | {old(v1, "e"):.1f} '
              f'| {new("Wscratch_Gnone"):.1f} | {new("Wscratch_Gfroze"):.1f} |')


def main():
    """Run the chosen sections."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--results_dir', default='direction1/transfer/results_coverage_v3')
    p.add_argument('--section', action='append', choices=SECTIONS)
    args = p.parse_args()
    acc, ns = load(args.results_dir)
    mean = {k: np.mean(list(v.values())) for k, v in acc.items()}
    for sec in args.section or SECTIONS:
        print(f'\n## {sec}\n')
        fn = globals()[sec]
        fn(mean, acc, ns) if sec in ('tiers', 'bug') else fn(mean, acc)


if __name__ == '__main__':
    main()
