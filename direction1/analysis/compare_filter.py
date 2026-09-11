"""
analysis/compare_filter.py --- Compare nofilter() vs auto(512) accuracy.

Reports all pairs in results_index.json that share the same
(model, language, task, pool_word_embed) but differ in filter type.
Run update_results_index.py first so the index contains entries for both filter types.

Usage (from spectral-probing/):
  python -m analysis.compare_filter
  python -m analysis.compare_filter --output analysis/filter_comparison.csv
  python -m analysis.compare_filter --model FacebookAI/xlm-roberta-base
  python -m analysis.compare_filter --task relations --lang en zh tr vi
  python -m analysis.compare_filter --pool avg
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_INDEX = "transfer/results_index.json"


def read_config(exp_path: str) -> dict:
    cfg_path = Path(exp_path) / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def read_accuracy(exp_path: str) -> float | None:
    """
    Primary  : config.json -> "accuracy"
    Fallback : perturbed_results.json -> "('orig', None)" -> "accuracy"
    """
    acc = read_config(exp_path).get("accuracy")
    if acc is not None:
        return acc
    perturbed_path = Path(exp_path) / "perturbed_results.json"
    if perturbed_path.exists():
        try:
            with open(perturbed_path) as f:
                entry = json.load(f).get("('orig', None)")
            if entry is not None:
                return entry.get("accuracy")
        except (json.JSONDecodeError, OSError):
            pass
    return None


def filter_group(filt: str | None) -> str:
    """Return 'auto', 'nofilter', or the raw string."""
    if not filt:
        return "unknown"
    if filt.startswith("auto("):
        return "auto"
    if filt.startswith("nofilter"):
        return "nofilter"
    return filt


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Compare nofilter() vs auto(512) accuracy from results_index.json"
    )
    p.add_argument("--index",  default=DEFAULT_INDEX,
                   help=f"Path to results_index.json (default: {DEFAULT_INDEX})")
    p.add_argument("--output", default=None,
                   help="Save results to this CSV path")
    p.add_argument("--model",  nargs="+", default=None,
                   help="Filter: substring match on full model name "
                        "(e.g. xlm-roberta-base aya-expanse-8b)")
    p.add_argument("--task",   nargs="+", default=None,
                   help="Filter by task (e.g. relations pos)")
    p.add_argument("--lang",   nargs="+", default=None,
                   help="Filter by language (e.g. en zh tr vi)")
    p.add_argument("--pool",   nargs="+", default=None,
                   help="Filter by pool_word_embed (e.g. avg last None)")
    return p.parse_args()


def main():
    args = parse_arguments()

    index_path = os.path.normpath(args.index)
    if not os.path.exists(index_path):
        print(f"Index not found: {index_path}", file=sys.stderr)
        print("Run:  python update_results_index.py  to build the index.",
              file=sys.stderr)
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    # Group entries by (model, language, task, pool_word_embed).
    # Within each group keep the best accuracy per filter type.
    # groups[key][fg] = (accuracy, path, raw_filter_str)
    groups: dict[tuple, dict[str, tuple]] = defaultdict(dict)

    for entry in index.get("entries", []):
        path = entry.get("path", "")
        model = entry.get("model", "")
        lang = entry.get("language", "")
        task = entry.get("task", "")
        pool = str(entry.get("pool_word_embed"))   # "None" when null

        if args.model and not any(m in model for m in args.model):
            continue
        if args.task and task not in args.task:
            continue
        if args.lang and lang not in args.lang:
            continue
        if args.pool and pool not in args.pool:
            continue

        # Accuracy: config.json primary, perturbed_results.json fallback
        acc = read_accuracy(path)
        # Filter type: config.json primary, stored index field fallback
        filt = read_config(path).get("filter") or entry.get("filter")

        if acc is None or filt is None:
            continue

        fg = filter_group(filt)
        key = (model, lang, task, pool)

        if fg not in groups[key] or acc > groups[key][fg][0]:
            groups[key][fg] = (acc, path, filt)

    # Build comparison rows: only groups that have BOTH auto and nofilter
    rows = []
    for (model, lang, task, pool), fg_map in sorted(groups.items()):
        auto_entry = fg_map.get("auto")
        nf_entry = fg_map.get("nofilter")

        if auto_entry is None or nf_entry is None:
            continue   # can't form a pair

        auto_acc, auto_path, auto_filt = auto_entry
        nf_acc, nf_path, nf_filt = nf_entry
        delta = auto_acc - nf_acc

        rows.append({
            "model":        model,
            "language":     lang,
            "task":         task,
            "pool":         pool,
            "nofilter_acc": round(nf_acc, 4),
            "auto512_acc":  round(auto_acc, 4),
            "delta":        round(delta, 4),
            "improved":     delta > 0,
            "auto_filter":  auto_filt,
            "nofilter_dir": nf_path,
            "auto512_dir":  auto_path,
        })

    if not rows:
        print("No matching pairs found.")
        print("Hint: run  python update_results_index.py  so the index "
              "contains entries for both filter types.")
        return

    n_pairs = len(rows)
    n_improved = sum(1 for r in rows if r["improved"])

    model_w = max(len(r["model"].split("/")[-1]) for r in rows)
    model_w = max(model_w, 5)
    header = (f"{'model':<{model_w}}  {'lang':<5}  {'task':<12}  {'pool':<6}"
              f"  {'nofilter':>9}  {'auto(512)':>9}  {'delta':>7}  ok?")
    sep = "-" * len(header)

    print(header)
    print(sep)
    for r in rows:
        ok = "✓" if r["improved"] else ("~" if r["delta"] == 0 else "✗")
        ms = r["model"].split("/")[-1]
        print(
            f"{ms:<{model_w}}  {r['language']:<5}  {r['task']:<12}  {r['pool']:<6}"
            f"  {r['nofilter_acc']:>9.4f}  {r['auto512_acc']:>9.4f}"
            f"  {r['delta']:>+7.4f}  {ok}"
        )
    print(sep)

    mean_d = sum(r["delta"] for r in rows) / n_pairs
    print(f"\nPairs: {n_pairs}  |  Improved: {n_improved}/{n_pairs}"
          f"  |  Mean delta: {mean_d:+.4f}")
    if n_improved:
        mean_imp = (sum(r["delta"] for r in rows if r["improved"])
                    / n_improved)
        print(f"Mean delta (improved only): {mean_imp:+.4f}")

    if args.output:
        fields = [
            "model", "language", "task", "pool",
            "nofilter_acc", "auto512_acc", "delta", "improved",
            "auto_filter", "nofilter_dir", "auto512_dir",
        ]
        out = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
