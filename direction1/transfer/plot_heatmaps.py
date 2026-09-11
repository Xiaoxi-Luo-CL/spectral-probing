"""
transfer/plot_heatmaps.py — Heatmap visualization of coverage-experiment results.

One figure per (model, src_lang, task).
  Y-axis : 24 target languages sorted by resource tier (high → 1k-2k → 200-1k).
           Tick labels are color-coded by tier; non-romanized scripts are in italic.
  X-axis : 26 columns in 7 blocks:
           [zero-shot a/b | c n× | d n× | e n× | f n× | g n× | h n×]
           where n× = [50, 100, 500, 1k].
           Each block header shows the condition letter, short name, and a description
           of what is trained / frozen.
  Cells  : colored by test accuracy (YlOrRd); gray = no result yet.
           Best cell per row is annotated in larger bold font.

Usage (from spectral-probing/):
    python -m direction1.transfer.plot_heatmaps
    python -m direction1.transfer.plot_heatmaps --results_dir direction1/transfer/results_coverage
    python -m direction1.transfer.plot_heatmaps --output_dir direction1/transfer/figures
    python -m direction1.transfer.plot_heatmaps --model xlm-roberta-base --src_lang en
    python -m direction1.transfer.plot_heatmaps --task relations --vmin 0.1
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
import copy
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Global font size — change this one value to rescale all text in the figure
# ---------------------------------------------------------------------------
FONTSIZE = 25   # base; all other sizes are derived from this

# ---------------------------------------------------------------------------
# Resource-level definitions (24 coverage languages)
# ---------------------------------------------------------------------------

RESOURCE_LEVEL: dict[str, str] = {
    # high resource
    "de": ">2k", "es": ">2k", "fr": ">2k", "hi": ">2k",
    "ja": ">2k", "ko": ">2k", "tr": ">2k", "vi": ">2k",
    # 1k–2k sentences
    "ab": "1k-2k", "bm": "1k-2k", "hu": "1k-2k",
    "jv": "1k-2k", "kk": "1k-2k", "quc": "1k-2k",
    "ta": "1k-2k", "yue": "1k-2k",
    # 200–1k sentences
    "bho": "200-1k", "br": "200-1k", "hsb": "200-1k",
    "kmr": "200-1k", "ml": "200-1k", "mr": "200-1k",
    "sah": "200-1k", "sq": "200-1k",
}
RESOURCE_ORDER = {">2k": 0, "1k-2k": 1, "200-1k": 2}
TIER_COLORS = {">2k": "#1a5276", "1k-2k": "#145a32", "200-1k": "#784212"}

# Languages whose treebanks use non-Latin scripts (shown in italic on y-axis)
NON_ROMAN_LANGS = {
    "sah",   # Cyrillic
    "bho",   # Devanagari
    "ml",    # Malayalam
    "mr",    # Devanagari
    "yue",   # Chinese
    "ab",    # Cyrillic
    "ta",    # Tamil
    "kk",    # Cyrillic
    "hi",    # Devanagari
    "ko",    # Hangul
    "ja",    # CJK
    "zh",    # Chinese
}

# Model vocabulary support level: whether each language appears in standard
# multilingual encoder pretraining data (mBERT / XLM-R).
# "all"   = high-resource; present in all multilingual models
# "mX"    = in mBERT + XLM-R training data, but not necessarily in decoder LMs
# "—"     = absent from standard multilingual encoder vocab (truly low-resource)
LANG_SUPPORT: dict[str, str] = {
    # high resource — all models
    "de": "all", "es": "all", "fr": "all", "hi": "all",
    "ja": "all", "ko": "all", "tr": "all", "vi": "all",
    # mBERT / XLM-R
    "sq": "mX",  "br": "mX",  "ml": "mX",  "mr": "mX",
    "hu": "mX",  "jv": "mX",  "ta": "mX",  "kk": "mX",
    # unsupported by standard multilingual encoders
    "kmr": "—",  "hsb": "—",  "sah": "—",  "bho": "—",
    "bm":  "—",  "quc": "—",  "yue": "—",  "ab":  "—",
}


def resource_level(lang: str) -> str:
    return RESOURCE_LEVEL.get(lang, "200-1k")


# ---------------------------------------------------------------------------
# Column layout  (26 columns total)
# ---------------------------------------------------------------------------

N_VALUES = [50, 100, 500, 1000]

ALL_COLS: list[tuple] = (
    [("a", None), ("b", None)]
    + [(c, n) for c in ("c", "d", "e", "f", "g", "h") for n in N_VALUES]
)
COL_INDEX = {col: i for i, col in enumerate(ALL_COLS)}

# Per-column x-tick labels
COL_TICK_LABELS = [("" if n is None else ("1k" if n == 1000 else str(n))) for c, n in ALL_COLS]

# Block definitions: (col_start_incl, col_end_excl, label)
BLOCK_DEFS = [
    (0,  1,  "(a)"),
    (1,  2,  "(b)"),
    (2,  6,  "(c1)"),
    (6,  10, "(c2)"),
    (10, 14, "(d1)"),
    (14, 18, "(d2)"),
    (18, 22, "(e1)"),
    (22, 26, "(e2)"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_results(results_root: Path) -> list[dict]:
    """
    Walk results_root for result_*.json files.
    Expected path (multi-seed layout):
      {results_root}/{model}/{src}2{tgt}/{task}/cond{X}[_n{N}]/seed{S}/result_*.json
    Also accepts the legacy single-seed layout (no seed{S} directory).
    """
    records = []
    for path in sorted(results_root.rglob("result_*.json")):
        parts = path.parts
        # Try multi-seed layout first (6 levels from root), then legacy (5 levels)
        for depth, has_seed in [(6, True), (5, False)]:
            try:
                ri = next(
                    i for i, p in enumerate(parts)
                    if p == results_root.name and i < len(parts) - depth
                )
            except StopIteration:
                continue

            try:
                model = parts[ri + 1]
                src2tgt = parts[ri + 2]
                task = parts[ri + 3]
                cond_dir = parts[ri + 4]
                seed_dir = parts[ri + 5] if has_seed else None
            except IndexError:
                continue

            m = re.fullmatch(r"(\w+)2(\w+)", src2tgt)
            if not m:
                continue
            src_lang, tgt_lang = m.group(1), m.group(2)

            cm = re.fullmatch(r"cond([a-h])(?:_n(\d+))?", cond_dir)
            if not cm:
                continue

            # Validate seed directory when expected
            if has_seed:
                sm = re.fullmatch(r"seed(\d+)", seed_dir)
                if not sm:
                    continue
                seed = int(sm.group(1))
            else:
                seed = 42   # legacy results assumed seed 42

            condition = cm.group(1)
            n_examples = int(cm.group(2)) if cm.group(2) else None

            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                break

            acc = data.get("test_accuracy")
            if acc is None:
                break

            records.append({
                "model":      model,
                "src_lang":   src_lang,
                "tgt_lang":   tgt_lang,
                "task":       task,
                "condition":  condition,
                "n_examples": n_examples,
                "seed":       seed,
                "accuracy":   acc,
            })
            break   # matched at this depth; don't try the other layout
    return records


# ---------------------------------------------------------------------------
# Matrix building
# ---------------------------------------------------------------------------

def build_matrix(
    records: list[dict],
    model: str,
    src_lang: str,
    task: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, list[str], list[tuple]]:
    """
    Returns:
        mat             : mean accuracy (n_langs × 26), NaN where no result
        mat_std         : std across seeds (NaN if only one seed)
        mat_count       : number of seeds contributing to each cell
        lang_order      : target languages sorted high → 1k-2k → 200-1k, then alpha
        tier_boundaries : [(start_row, end_row_excl, tier_name), ...]
    """
    from collections import defaultdict

    subset = [
        r for r in records
        if r["model"] == model and r["src_lang"] == src_lang and r["task"] == task
    ]
    if not subset:
        return None, None, None, [], []

    tgt_langs = sorted(
        {r["tgt_lang"] for r in subset},
        key=lambda l: (RESOURCE_ORDER.get(resource_level(l), 9), l),
    )
    lang_idx = {l: i for i, l in enumerate(tgt_langs)}

    # Group accuracies by (lang, condition, n_examples) across seeds
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for r in subset:
        key = (r["tgt_lang"], r["condition"], r["n_examples"])
        grouped[key].append(r["accuracy"])

    n_rows, n_cols = len(tgt_langs), len(ALL_COLS)
    mat = np.full((n_rows, n_cols), np.nan)
    mat_std = np.full((n_rows, n_cols), np.nan)
    mat_count = np.zeros((n_rows, n_cols), dtype=int)

    for (tgt_lang, cond, n_ex), accs in grouped.items():
        li = lang_idx.get(tgt_lang)
        ci = COL_INDEX.get((cond, n_ex))
        if li is None or ci is None:
            continue
        mat[li, ci] = float(np.mean(accs))
        mat_count[li, ci] = len(accs)
        if len(accs) > 1:
            assert len(accs) == 3
            mat_std[li, ci] = float(np.std(accs, ddof=1))

    tiers = [resource_level(l) for l in tgt_langs]
    boundaries, prev, start = [], tiers[0], 0
    for i, t in enumerate(tiers[1:], 1):
        if t != prev:
            boundaries.append((start, i, prev))
            start, prev = i, t
    boundaries.append((start, len(tgt_langs), prev))

    return mat, mat_std, mat_count, tgt_langs, boundaries


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def fmt_acc(v: float) -> str:
    """Format 0.832 → '.832', 1.0 → '1.00'."""
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def plot_heatmap(
    mat: np.ndarray,
    mat_std: np.ndarray,
    mat_count: np.ndarray,
    lang_order: list[str],
    tier_boundaries: list[tuple],
    model: str,
    src_lang: str,
    task: str,
    vmin: float,
    vmax: float,
    output_path: str,
) -> None:
    n_langs, n_cols = mat.shape
    n_seeds = int(mat_count[~np.isnan(mat)].max()) if np.any(~np.isnan(mat)) else 1

    # Figure sizing
    row_h, col_w = 0.38, 0.50
    fig_h = n_langs * row_h + 3.0
    fig_w = n_cols * col_w + 3.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colormap: YlOrRd; NaN cells gray
    cmap = copy.copy(plt.cm.YlOrRd)
    cmap.set_bad("#cccccc")
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", alpha=0.85,
    )

    # Per-cell text annotations — best cell per row is larger + bold
    # If multiple seeds: show mean on top line, ±std on bottom line (tiny).
    ann_fs = max(FONTSIZE * 0.40, min(FONTSIZE * 0.65, 85.0 / max(n_langs, n_cols)))
    std_fs = ann_fs * 0.75
    best_cols = [
        int(np.nanargmax(mat[row])) if np.any(~np.isnan(mat[row])) else -1
        for row in range(n_langs)
    ]
    has_std = not np.all(np.isnan(mat_std))
    for row in range(n_langs):
        for col in range(n_cols):
            v = mat[row, col]
            if np.isnan(v):
                continue
            brightness = (v - vmin) / max(vmax - vmin, 1e-9)
            # fc = "white" if brightness > 0.62 else "#262323"
            is_best = (col == best_cols[row])
            fc = "white" if is_best else '#444444'
            mean_fs = ann_fs * 1.3 if is_best else ann_fs
            # Mean: shifted up slightly when std row is also shown
            y_offset = -0.18 if has_std and not np.isnan(mat_std[row, col]) else 0.0
            ax.text(
                col, row + y_offset, fmt_acc(v),
                ha="center", va="center",
                fontsize=mean_fs,
                fontweight="bold" if is_best else "normal",
                color=fc,
            )
            # ±std on a second tiny line below
            s = mat_std[row, col]
            if not np.isnan(s):
                ax.text(
                    col, row + 0.28, f"±{fmt_acc(s)}",
                    ha="center", va="center",
                    fontsize=std_fs, color=fc, alpha=0.85,
                )

    # Y-axis: language code only, color-coded by tier, italic for non-Latin scripts
    ax.set_yticks(range(n_langs))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=0)
    tr = ax.get_yaxis_transform()   # x=axes-frac, y=data coords

    _LANG_BG = {"mX": "#cce8ff", "—": "#e8e8e8"}  # light-blue = mX; grey = unsupported
    for i, lang in enumerate(lang_order):
        is_roman = lang not in NON_ROMAN_LANGS
        support = LANG_SUPPORT.get(lang, "—")
        bg = _LANG_BG.get(support)
        bbox_kw = dict(boxstyle="square,pad=0.15", facecolor=bg, edgecolor="none", alpha=0.85) if bg else None
        ax.text(-0.02, i, lang, transform=tr, clip_on=False,
                ha="right", va="center", fontsize=FONTSIZE * 0.70,
                color=TIER_COLORS[resource_level(lang)],
                fontstyle="normal" if is_roman else "italic",
                **({'bbox': bbox_kw} if bbox_kw else {}))

    # Tier labels to the right of axes
    for start_row, end_row, tier in tier_boundaries:
        mid_y_frac = 1.0 - (start_row + end_row) / 2.0 / n_langs
        ax.text(
            1.01, mid_y_frac, tier,
            ha="left", va="center",
            fontsize=FONTSIZE * 0.80, style="italic",
            color=TIER_COLORS[tier],
            transform=ax.transAxes, clip_on=False,
        )

    # Horizontal tier separator lines
    for start_row, _, _ in tier_boundaries:
        if start_row > 0:
            ax.axhline(y=start_row - 0.5, color="#444444", linewidth=1.6, zorder=5)

    # X-axis tick labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COL_TICK_LABELS, fontsize=FONTSIZE * 0.60)
    ax.tick_params(axis="x", length=0, pad=2)

    # Block separators and labels above the heatmap
    for bs, be, label in BLOCK_DEFS:
        # Block label centred above the block
        x_ax = (bs + be) / 2.0 / n_cols
        ax.text(
            x_ax, 1.02, label,
            ha="center", va="bottom",
            fontsize=FONTSIZE * 0.60, fontweight="bold", color="#444444",
            transform=ax.transAxes, clip_on=False,
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {output_path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot cross-lingual transfer results as accuracy heatmaps"
    )
    p.add_argument(
        "--results_dir", default="direction1/transfer/results_coverage",
        help="Root directory containing result JSON files (default: direction1/transfer/results_coverage)",
    )
    p.add_argument(
        "--output_dir", default="direction1/transfer/figures",
        help="Output directory for figures (default: direction1/transfer/figures)",
    )
    p.add_argument(
        "--task", default="relations",
        help="Task to visualize (default: relations)",
    )
    p.add_argument(
        "--model", default=None,
        help="Filter by model short name substring (e.g. xlm-roberta-base)",
    )
    p.add_argument(
        "--src_lang", default=None,
        help="Filter by source language (e.g. en)",
    )
    p.add_argument(
        "--vmin", type=float, default=None,
        help="Colormap lower bound (default: floor of min accuracy in data)",
    )
    p.add_argument(
        "--vmax", type=float, default=1.0,
        help="Colormap upper bound (default: 1.0)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"Results directory not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {results_root} ...")
    records = find_results(results_root)
    print(f"  {len(records)} result records loaded.")
    if not records:
        return

    models = sorted({r["model"] for r in records})
    src_langs = sorted({r["src_lang"] for r in records})
    tasks = sorted({r["task"] for r in records})

    if args.model:
        models = [m for m in models if args.model in m]
    if args.src_lang:
        src_langs = [s for s in src_langs if s == args.src_lang]

    vmin = args.vmin
    if vmin is None:
        vmin = max(0.0, round(min(r["accuracy"] for r in records) - 0.05, 1))
    vmax = args.vmax

    total = 0
    for task in tasks:
        if task != args.task:
            continue
        for model in models:
            for src_lang in src_langs:
                mat, mat_std, mat_count, lang_order, tier_bounds = build_matrix(
                    records, model, src_lang, task)
                if mat is None:
                    print(f"  No data: {model} / src={src_lang} / task={task}")
                    continue
                n_filled = int((~np.isnan(mat)).sum())
                max_seeds = int(mat_count.max()) if mat_count.size else 0
                model_short = model.split("/")[-1]
                print(
                    f"  [{model_short} / src={src_lang} / {task}]  "
                    f"{mat.shape[0]} langs × {mat.shape[1]} cols  "
                    f"({n_filled} cells filled, {mat.size - n_filled} gray, "
                    f"max {max_seeds} seed(s))"
                )
                fname = f"heatmap_{model_short}_{src_lang}_{task}.pdf"
                plot_heatmap(
                    mat, mat_std, mat_count, lang_order, tier_bounds,
                    model, src_lang, task,
                    vmin, vmax,
                    str(Path(args.output_dir) / fname),
                )
                total += 1

    print(f"\nDone — {total} figure(s) written to {args.output_dir}/")


if __name__ == "__main__":
    main()
