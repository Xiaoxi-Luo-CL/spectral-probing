"""
Plot word_pooling=None vs word_pooling=last spectral profiles.

Saves two PDFs:
  word_pooling_pos.pdf       — 4 rows (languages) x 4 cols (models), POS task
  word_pooling_relations.pdf — 4 rows (languages) x 4 cols (models), Relations task

Blue   = pool_word_embed=None  (no word pooling)
Orange = pool_word_embed=last  (word pooling, last token)

Usage (from spectral-probing/):
    python analysis/plot_word_pooling_comparison.py
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_INDEX = "transfer/results_index.json"
OUT_DIR = "analysis/paper_figures/word_pooling_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    "mbert":   "bert-base-multilingual-cased",
    "xlm-r":   "FacebookAI/xlm-roberta-base",
    "aya":     "CohereLabs/aya-expanse-8b",
    "mistral": "mistralai/Mistral-7B-v0.1",
}
MODEL_LABELS = {
    "mbert":   "mBERT",
    "xlm-r":   "XLM-R",
    "aya":     "Aya-8B",
    "mistral": "Mistral-7B",
}
TASKS = ["pos", "relations"]
TASK_FILENAMES = {"pos": "word_pooling_pos.pdf", "relations": "word_pooling_relations.pdf"}
LANGS = ["en", "zh", "ja", "de"]
LANG_LABELS = {"en": "English", "zh": "Chinese", "ja": "Japanese", "de": "German"}

FONTSIZE = 40
COLOR_NONE = "#1f77b4"   # blue  — no pooling
COLOR_LAST = "#ff7f0e"   # orange — word pooling (last)


# ---------------------------------------------------------------------------
# Build lookup from index
# ---------------------------------------------------------------------------
def build_lookup():
    with open(RESULTS_INDEX) as f:
        idx = json.load(f)

    lookup = {}  # (model_key, task, lang, pool) -> path
    for entry in idx["entries"]:
        m = entry.get("model", "")
        t = entry.get("task", "")
        l = entry.get("language", "")
        p = entry.get("pool_word_embed")
        filt = entry.get("filter", "")
        path = entry.get("path", "")

        if not filt.startswith("auto("):
            continue
        if not os.path.isfile(path + "/best.pt"):
            continue

        for mkey, mname in MODELS.items():
            if m == mname:
                key = (mkey, t, l, p)
                if key not in lookup:
                    lookup[key] = path
    return lookup


# ---------------------------------------------------------------------------
# Load filter weights from best.pt, apply sigmoid
# ---------------------------------------------------------------------------
def load_filter(path):
    pt = torch.load(path + "/best.pt", map_location="cpu", weights_only=False)
    return torch.sigmoid(pt["frequency_filter"]).detach().numpy()


# ---------------------------------------------------------------------------
# Verify sigmoid peak matches softmax peak (sanity check)
# ---------------------------------------------------------------------------
def verify_one(path):
    pt = torch.load(path + "/best.pt", map_location="cpu", weights_only=False)
    raw = pt["frequency_filter"]
    sig = torch.sigmoid(raw).detach().numpy()
    dist = np.load(path + "/distribution.npy")
    assert sig.argmax() == dist.argmax(), (
        f"Peak mismatch in {path}: sigmoid={sig.argmax()}, softmax={dist.argmax()}"
    )


# ---------------------------------------------------------------------------
# Plot one task -> one PDF
# ---------------------------------------------------------------------------
def plot_task(task, lookup):
    n_rows = len(LANGS)
    n_cols = len(MODELS)

    fig_w = n_cols * 4.5
    fig_h = n_rows * 3.5 + 1.2   # extra bottom space for legend
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    for col, mkey in enumerate(MODELS):
        for row, lang in enumerate(LANGS):
            ax = axes[row][col]

            path_none = lookup.get((mkey, task, lang, None))
            path_last = lookup.get((mkey, task, lang, "last"))

            if path_none is not None:
                ax.plot(load_filter(path_none), color=COLOR_NONE, linewidth=1.5)
            if path_last is not None:
                ax.plot(load_filter(path_last), color=COLOR_LAST, linewidth=1.5)
            if path_none is None and path_last is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes,
                        fontsize=FONTSIZE * 0.8, color="gray")

            ax.set_ylim(0, 1)
            ax.set_xlim(0, 511)
            ax.tick_params(labelsize=FONTSIZE * 0.75)

            # Top: model name
            if row == 0:
                ax.set_title(MODEL_LABELS[mkey], fontsize=FONTSIZE, pad=10)

            # Left: language label
            if col == 0:
                ax.set_ylabel(LANG_LABELS[lang], fontsize=FONTSIZE, labelpad=6)

            # Hide x tick labels except bottom row
            if row < n_rows - 1:
                ax.set_xticklabels([])

    # Legend centered below all subplots
    legend_handles = [
        mpatches.Patch(color=COLOR_NONE, label="No word pooling"),
        mpatches.Patch(color=COLOR_LAST, label="Word pooling (last token)"),
    ]
    fig.legend(handles=legend_handles,
               loc="lower center",
               ncol=2,
               fontsize=FONTSIZE * 0.9,
               frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.08, 1, 1])   # leave 8% at bottom for legend

    out_path = os.path.join(OUT_DIR, TASK_FILENAMES[task])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    lookup = build_lookup()

    # Spot-check a few entries
    checks = [
        ("xlm-r", "pos",       "en", "last"),
        ("xlm-r", "pos",       "en", None),
        ("aya",   "relations", "de", "last"),
        ("aya",   "relations", "de", None),
    ]
    print("Verification:")
    for key in checks:
        if key in lookup:
            verify_one(lookup[key])
            print(f"  OK      {key}")
        else:
            print(f"  missing {key}")

    for task in TASKS:
        plot_task(task, lookup)


if __name__ == "__main__":
    main()
