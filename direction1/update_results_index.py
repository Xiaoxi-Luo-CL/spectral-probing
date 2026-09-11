"""
update_results_index.py — Build results_index.json from scratch.

Scans every subdirectory under results/ for experiment directories (must contain both config.json and best.pt), reads their metadata, and writes a fresh results_index.json.

Accuracy lookup order (per experiment):
  1. config.json        -> "accuracy"
  2. perturbed_results.json -> "('orig', None)" -> "accuracy"

Deduplication key: (model|language|task|pool_word_embed, filter_group)
  -> both auto(512) and nofilter() entries are kept for the same setting.
  -> the `lookup` dict (used by zero_shot.py) stores one path per
     model|language|task|pool key, preferring auto(512).

Usage (from spectral-probing/):
    python update_results_index.py
    python update_results_index.py --dry-run
    python update_results_index.py --results_dir results/
    python update_results_index.py --index transfer/results_index.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_RESULTS_DIR = 'direction1/results'
DEFAULT_INDEX = 'direction1/transfer/results_index.json'


def read_accuracy(exp_dir: Path) -> float | None:
    """
    Primary  : config.json        -> "accuracy"
    Fallback : perturbed_results.json -> "('orig', None)" -> "accuracy"
    """
    cfg_path = exp_dir / "config.json"
    if cfg_path.exists():
        try:
            acc = json.loads(cfg_path.read_text()).get("accuracy")
            if acc is not None:
                return acc
        except (json.JSONDecodeError, OSError):
            pass

    perturbed_path = exp_dir / "perturbed_results.json"
    if perturbed_path.exists():
        try:
            entry = json.loads(perturbed_path.read_text()).get("('orig', None)")
            if entry is not None:
                return entry.get("accuracy")
        except (json.JSONDecodeError, OSError):
            pass

    return None


def experiment_from_dir(exp_dir: Path) -> dict | None:
    """
    Build an index entry from an experiment directory.
    Returns None if config.json or best.pt is missing / malformed,
    or if required fields (lm_name, language, task_name) are absent.
    """
    if not (exp_dir / "config.json").exists() or not (exp_dir / "best.pt").exists():
        return None

    try:
        cfg = json.loads((exp_dir / "config.json").read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] {exp_dir}/config.json unreadable: {exc}", file=sys.stderr)
        return None

    model = cfg.get("lm_name")
    language = cfg.get("language")
    task = cfg.get("task_name")
    if not (model and language and task):
        return None

    dirname = exp_dir.name
    timestamp = dirname.split("_")[0] if "_" in dirname else ""

    return {
        "path":              str(exp_dir.resolve()),
        "timestamp":         timestamp,
        "model":             model,
        "language":          language,
        "task":              task,
        "pool_word_embed":   cfg.get("pool_word_embed"),
        "embed_layer":       cfg.get("embed_layer"),
        "embedding_pooling": cfg.get("embedding_pooling"),
        "train_path":        cfg.get("train_path"),
        "filter":            cfg.get("filter"),
        "_accuracy":         read_accuracy(exp_dir) or 0.0,
    }


def scan(results_dir: Path) -> list[dict]:
    """Recursively collect all valid experiment entries under results_dir."""
    experiments = []
    for exp_dir in sorted(results_dir.rglob("config.json")):
        entry = experiment_from_dir(exp_dir.parent)
        if entry:
            experiments.append(entry)
    return experiments


def _filter_group(filt: str | None) -> str:
    if not filt:
        return "unknown"
    if filt.startswith("auto("):
        return "auto"
    if filt.startswith("nofilter"):
        return "nofilter"
    return filt


def make_key(model: str, language: str, task: str, pool) -> str:
    pool_s = "None" if pool is None else pool
    return f"{model}|{language}|{task}|{pool_s}"


def build_index(experiments: list[dict]) -> dict:
    """
    Deduplicate by (model|language|task|pool, filter_group), keeping the
    highest-accuracy entry per group.  The lookup dict prefers auto(512).
    """
    best: dict[tuple, dict] = {}

    for exp in experiments:
        key = make_key(exp["model"], exp["language"],
                       exp["task"], exp.get("pool_word_embed"))
        fg = _filter_group(exp.get("filter"))
        acc = exp.get("_accuracy", 0.0)
        fk = (key, fg)
        if fk not in best or acc > best[fk].get("_accuracy", 0.0):
            best[fk] = exp

    lookup: dict[str, str] = {}
    entries: list[dict] = []

    for (key, fg), exp in sorted(best.items()):
        entries.append({k: v for k, v in exp.items() if not k.startswith("_")})
        if key not in lookup or fg == "auto":
            lookup[key] = exp["path"]

    return {
        "entries":           entries,
        "total":             len(entries),
        "lookup":            lookup,
        "lookup_key_format": "model|language|task|pool_word_embed",
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Build results_index.json from scratch by scanning results/"
    )
    p.add_argument(
        "--results_dir", default=DEFAULT_RESULTS_DIR, metavar="DIR",
        help=f"Root directory to scan (default: {DEFAULT_RESULTS_DIR})",
    )
    p.add_argument(
        "--index", default=DEFAULT_INDEX,
        help=f"Output path for results_index.json (default: {DEFAULT_INDEX})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print summary without writing the index file",
    )
    return p.parse_args()


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {results_dir} ...")
    experiments = scan(results_dir)
    print(f"  Found {len(experiments)} experiment(s) with config.json + best.pt")

    index = build_index(experiments)
    print(f"  Index entries : {index['total']}  "
          f"(lookup keys: {len(index['lookup'])})")

    if args.dry_run:
        print("[DRY-RUN] Index not written.")
        return

    out = Path(args.index)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
