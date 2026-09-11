"""
analysis/compare_transfer.py — Summarize transfer/zero_shot.py results.

Prints a table comparing conditions a/b/c/d across model, language pair, task.
Optionally saves to CSV.

Usage (from spectral-probing/):
  python transfer/compare_transfer.py
  python transfer/compare_transfer.py --results_dir transfer/results
  python transfer/compare_transfer.py --output transfer/transfer_summary.csv
  python transfer/compare_transfer.py --model aya-expanse-8b --task relations
"""

import argparse
import csv
import json
import os
import re

TRANSFER_DIR = os.path.join(os.path.dirname(__file__), '..', 'transfer', 'results')


def parse_arguments():
    p = argparse.ArgumentParser(description='Summarize transfer experiment results')
    p.add_argument('--results_dir', default=TRANSFER_DIR)
    p.add_argument('--output', default=None, help='Save CSV to this path')
    p.add_argument('--model', nargs='+', default=None, help='Filter by model basename')
    p.add_argument('--task', nargs='+', default=None, help='Filter by task')
    p.add_argument('--src_lang', nargs='+', default=None, help='Filter by source language')
    p.add_argument('--condition', nargs='+', default=None, help='Filter by condition (a/b/c/d)')
    return p.parse_args()


def collect_results(results_dir):
    """Walk results_dir and collect all result_*.json files."""
    rows = []
    if not os.path.isdir(results_dir):
        return rows
    for d in sorted(os.listdir(results_dir)):
        exp_dir = os.path.join(results_dir, d)
        if not os.path.isdir(exp_dir):
            continue
        for fname in sorted(os.listdir(exp_dir)):
            if not fname.startswith('result_') or not fname.endswith('.json'):
                continue
            fpath = os.path.join(exp_dir, fname)
            with open(fpath) as f:
                r = json.load(f)

            # derive model basename and lang pair from result fields
            lm = r.get('lm_name', '')
            model = lm.split('/')[-1]
            src = r.get('source_exp', '') or ''
            target = r.get('target_data', '')
            n_examples = r.get('n_examples')
            pool = r.get('pool_word_embed')

            # infer source language from source_exp path
            src_lang = _infer_src_lang(src)
            # infer target language from target_data path
            tgt_lang = _infer_tgt_lang(target)
            # infer task
            task = _infer_task(target, src)

            rows.append({
                'model':        model,
                'src_lang':     src_lang,
                'tgt_lang':     tgt_lang,
                'task':         task,
                'condition':    r.get('condition'),
                'n_examples':   n_examples,
                'pool':         str(pool),
                'test_acc':     r.get('test_accuracy'),
                'test_loss':    r.get('test_loss'),
                'n_test':       r.get('n_test'),
                'exp_dir':      d,
            })
    return rows


def _infer_src_lang(src_exp):
    """Try to extract source language from source_exp path."""
    if not src_exp:
        return '?'
    # e.g. .../results/en/021819... or results/word_pooling/...
    parts = src_exp.replace('\\', '/').split('/')
    # look for a 2-letter language code directory
    for p in reversed(parts):
        if re.match(r'^[a-z]{2,3}$', p) and p not in ('pt', 'in', 'on', 'or'):
            return p
    # fall back to dirname of experiment
    return '?'


def _infer_tgt_lang(target_data):
    """Extract language code from target_data path."""
    if not target_data:
        return '?'
    fname = os.path.basename(target_data)
    m = re.match(r'^([a-z]{2,3})[-_]', fname)
    return m.group(1) if m else '?'


def _infer_task(target_data, src_exp):
    """Infer task from target_data filename or source_exp path."""
    for s in [target_data, src_exp]:
        if 'relations' in s:
            return 'relations'
        if 'pos' in s:
            return 'pos'
    return '?'


def main():
    args = parse_arguments()
    rows = collect_results(args.results_dir)

    # apply filters
    if args.model:
        rows = [r for r in rows if r['model'] in args.model]
    if args.task:
        rows = [r for r in rows if r['task'] in args.task]
    if args.src_lang:
        rows = [r for r in rows if r['src_lang'] in args.src_lang]
    if args.condition:
        rows = [r for r in rows if r['condition'] in args.condition]

    rows.sort(key=lambda r: (r['model'], r['src_lang'], r['tgt_lang'], r['task'],
                             r['condition'], r['n_examples'] or 0))

    # --- print table ---
    hdr = (f"{'model':<28} {'src→tgt':<9} {'task':<10} "
           f"{'cond':<5} {'n_ex':>5} {'pool':<5} {'test_acc':>9} {'test_loss':>10}")
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        acc = f"{r['test_acc']:>9.4f}" if r['test_acc'] is not None else f"{'N/A':>9}"
        loss = f"{r['test_loss']:>10.4f}" if r['test_loss'] is not None else f"{'N/A':>10}"
        n_ex = f"{r['n_examples']:>5}" if r['n_examples'] is not None else f"{'—':>5}"
        print(f"{r['model']:<28} {r['src_lang']+'→'+r['tgt_lang']:<9} {r['task']:<10} "
              f"{r['condition']:<5} {n_ex} {r['pool']:<5} {acc} {loss}")

    print(f"\nTotal rows: {len(rows)}")

    # --- optional CSV output ---
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fields = ['model', 'src_lang', 'tgt_lang', 'task', 'condition',
                  'n_examples', 'pool', 'test_acc', 'test_loss', 'n_test', 'exp_dir']
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows to '{args.output}'")


if __name__ == '__main__':
    main()
