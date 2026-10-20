r"""Write the rows of the paper's transfer table from run_batch.py results.

Every number is the mean test accuracy (%) over seeds, read from the result files, so the
table can be regenerated whenever the results change. Bold marks the best cell in a row.
--fmt tex prints the LaTeX body rows used in the paper; --fmt md prints a Markdown table.

Run from spectral-probing/::

    python -m direction1.transfer.make_transfer_table \
        --results_dir direction1/transfer/results_coverage_v2 --src_lang zh \
        --models Qwen2.5-7B bert-base-multilingual-cased --labels Qwen mBERT
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from direction1.transfer.plot_heatmaps import (ALL_COLS, LANG_SUPPORT, NON_ROMAN_LANGS,
                                               build_matrix, find_results)

# cell shading by model coverage: mBERT/XLM-R only, or no model at all
SUPPORT_COLOR = {'mX': 'CCE8FF', '—': 'E0E0E0'}
# paper settings (c1, c2, d1, d2, e1, e2) are code conditions c..h
SETTINGS = {'c': 'c1', 'd': 'c2', 'e': 'd1', 'f': 'd2', 'g': 'e1', 'h': 'e2'}


def lang_cell(lang):
    """Language code with the paper's italics for non-Latin scripts and coverage shading."""
    code = '\\textit{' + lang + '}' if lang in NON_ROMAN_LANGS else lang
    color = SUPPORT_COLOR.get(LANG_SUPPORT.get(lang))
    return '\\cellcolor[HTML]{' + color + '}' + code if color else code


def model_rows(records, model, label, args, cols):
    """Build the rows for one model, warning on stderr about unexpected seed counts."""
    mat, _, counts, langs, _ = build_matrix(records, model, args.src_lang, args.task)
    missing = [lang for lang in args.langs if lang not in langs]
    if mat is None or missing:
        raise SystemExit(f'{model}: no results for {missing or args.langs} under {args.results_dir}')
    tex = args.fmt == 'tex'
    lines = ['\\multirow{' + str(len(args.langs)) + '}{*}{' + label + '}'] if tex else []
    for i, lang in enumerate(args.langs):
        # the paper separates fully supported targets from the rest
        if tex and i and LANG_SUPPORT.get(lang) != 'all' and LANG_SUPPORT.get(args.langs[i - 1]) == 'all':
            lines.append('\\cmidrule(l){2-' + str(len(cols) + 2) + '}')
        row = langs.index(lang)
        vals = mat[row, cols]
        best = int(np.nanargmax(vals)) if not np.all(np.isnan(vals)) else -1
        cells = []
        for j, col in enumerate(cols):
            if np.isnan(vals[j]):
                cells.append('\\text{--}' if tex else '–')
                continue
            if counts[row, col] != args.seeds:
                print(f'warning: {label} {lang} {ALL_COLS[col]}: {counts[row, col]} seeds', file=sys.stderr)
            text = f'{100 * vals[j]:.1f}'
            if j == best:
                text = '\\textbf{\\textcolor{blue}{' + text + '}}' if tex else '**' + text + '**'
            cells.append(text)
        if tex:
            lines.append(' & ' + lang_cell(lang) + ' & ' + ' & '.join(cells) + ' \\\\')
        else:
            lines.append('| ' + (label if i == 0 else '') + ' | ' + lang + ' | ' + ' | '.join(cells) + ' |')
    return lines


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_dir', default='direction1/transfer/results_coverage_v2')
    p.add_argument('--task', default='relations')
    p.add_argument('--src_lang', default='zh')
    p.add_argument('--models', nargs='+', default=['Qwen2.5-7B', 'bert-base-multilingual-cased'],
                   help='model directory names under results_dir')
    p.add_argument('--labels', nargs='+', default=['Qwen', 'mBERT'], help='row label per model')
    p.add_argument('--langs', nargs='+', default=['de', 'fr', 'hi', 'ko', 'br', 'hsb', 'ml', 'sah'])
    p.add_argument('--n', type=int, nargs='+', default=[50, 100, 500],
                   help='few-shot sizes shown for each of the six training settings')
    p.add_argument('--seeds', type=int, default=3, help='expected seeds per cell')
    p.add_argument('--fmt', choices=['tex', 'md'], default='tex')
    return p.parse_args()


def main():
    """Print the table: zero-shot (a), (b), then (c1)..(e2) at each n."""
    args = parse_args()
    if len(args.models) != len(args.labels):
        raise SystemExit('--models and --labels must have the same length')
    records = find_results(Path(args.results_dir))
    cols = [ALL_COLS.index(('a', None)), ALL_COLS.index(('b', None))]
    cols += [ALL_COLS.index((c, n)) for c in SETTINGS for n in args.n]
    blocks = [model_rows(records, m, lab, args, cols) for m, lab in zip(args.models, args.labels)]
    if args.fmt == 'tex':
        print('\n\\midrule\n'.join('\n'.join(b) for b in blocks))
        return
    head = ['model', 'lang', '(a)', '(b)'] + [f'{s} {n}' for s in SETTINGS.values() for n in args.n]
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    print('\n'.join('\n'.join(b) for b in blocks))


if __name__ == '__main__':
    main()
