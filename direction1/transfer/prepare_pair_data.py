"""Prepare target data for the related-language pairs listed in related_pairs.json.

Each target's UD treebank(s) are converted into tasks/ud-syntax/coverage-relations/{lang}-all-relations.csv
in official split order (train, dev, test), and the target list read by run_batch.py is written next to the
config. A target whose treebank has a train split records its official test size ("official_test"), so
run_batch.py --pool_size draws the training pool from the start of train and tests on the official test
split; test-only treebanks get a custom split. Files that coverage_targets.json uses are never rewritten.

Usage (from spectral-probing/):
    python -m direction1.transfer.prepare_pair_data
"""
import argparse
import csv
import json

from direction1.transfer.prepare_coverage_data import (BASE_DIR, OUT_DIR, TARGETS_JSON, UD_DIR, conllu_to_rows,
                                                       load_ud_module, write_csv)

SPLITS = ('train', 'dev', 'test')


def convert_target(ud_dirs, ud_mod):
    """Rows of the target treebanks in train, dev, test order, and the official test size (0 without train)."""
    rows = {s: [] for s in SPLITS}
    for d in ud_dirs:
        for s in SPLITS:
            for cf in sorted((UD_DIR / d).glob(f'*-ud-{s}.conllu')):
                rows[s] += conllu_to_rows(cf, ud_mod)
    return rows['train'] + rows['dev'] + rows['test'], len(rows['test']) if rows['train'] else 0


def read_rows(path):
    """(text, label) rows of an existing target CSV."""
    with open(path, encoding='utf-8', newline='') as f:
        return [tuple(r) for r in list(csv.reader(f))[1:]]


def main():
    """Convert the target treebanks and write the pair target list."""
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pairs', default='direction1/transfer/related_pairs.json')
    p.add_argument('--out', default='direction1/transfer/pair_targets.json')
    args = p.parse_args()

    cfg = json.load(open(args.pairs))
    coverage = {t['path'] for t in json.load(open(TARGETS_JSON))}
    ud_mod = load_ud_module()
    targets = []
    for lang, ud_dirs in cfg['targets'].items():
        out = OUT_DIR / f'{lang}-all-relations.csv'
        rel = str(out.relative_to(BASE_DIR))
        rows, n_test = convert_target(ud_dirs, ud_mod)
        rows = [tuple(r) for r in rows]
        if out.exists() and read_rows(out) == rows:
            print(f'  [same] {out}')
        elif rel in coverage:  # the v3 grid read this file; changing it would break comparability
            raise SystemExit(f'{out} differs from the ordered conversion but is used by {TARGETS_JSON}')
        else:
            write_csv(out, rows)
        targets.append({'path': rel, 'lang': lang, 'official_test': n_test})
        print(f'  {lang}: {len(rows)} sentences, official test {n_test or "none (test-only treebank)"}')

    with open(args.out, 'w') as f:
        json.dump(targets, f, indent=2)
    print(f'Wrote {args.out} ({len(targets)} targets)')


if __name__ == '__main__':
    main()
