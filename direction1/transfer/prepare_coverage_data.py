"""
transfer/prepare_coverage_data.py — Prepare merged CSV files for the 24-language
coverage experiment and write coverage_targets.json for run_batch.py.

For high-resource languages already in tasks/ud-syntax/{lang}/relations/:
  -> concatenate train + dev + test CSVs into one file.

For the 16 medium/low-resource languages:
  -> convert every .conllu split from the UD treebank directory using the same
    logic as tasks/ud-syntax/convert.py, then merge into one file.

Output layout:
  tasks/ud-syntax/coverage-relations/{lang}-all-relations.csv
  transfer/coverage_targets.json

Usage (from spectral-probing/):
  python -m direction1.transfer.prepare_coverage_data
  python -m direction1.transfer.prepare_coverage_data --dry_run
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# (lang_code, resource_tier, n_test, source_spec)
# source_spec: either a list of ud-treebank directory names (relative to
#              ud-treebanks-v2.17/) or "EXISTING" to use the pre-built
#              tasks/ud-syntax/{lang}/relations/ folder.
LANGUAGES = [
    # 200-1000, unsupported, roman
    ("kmr", "200-1000", 100, ["UD_Northern_Kurdish-Kurmanji"]),
    ("hsb", "200-1000", 100, ["UD_Upper_Sorbian-UFAL"]),
    # 200-1000, unsupported, non-roman
    ("sah", "200-1000", 100, ["UD_Yakut-YKTDT"]),
    ("bho", "200-1000", 100, ["UD_Bhojpuri-BHTB"]),
    # 200-1000, mBERT/XLM-R, roman
    ("sq",  "200-1000", 100, ["UD_Albanian-STAF", "UD_Albanian-TSA"]),
    ("br",  "200-1000", 100, ["UD_Breton-KEB"]),
    # 200-1000, mBERT/XLM-R, non-roman
    ("ml",  "200-1000", 100, ["UD_Malayalam-UFAL"]),
    ("mr",  "200-1000", 100, ["UD_Marathi-UFAL"]),
    # 1000-2000, unsupported, roman
    ("bm",  "1000-2000", 200, ["UD_Bambara-CRB"]),
    ("quc", "1000-2000", 200, ["UD_Kiche-IU"]),
    # 1000-2000, unsupported, non-roman
    ("yue", "1000-2000", 200, ["UD_Cantonese-HK"]),
    ("ab",  "1000-2000", 200, ["UD_Abkhaz-AbNC"]),
    # 1000-2000, mBERT/XLM-R, roman
    ("hu",  "1000-2000", 200, ["UD_Hungarian-Szeged"]),
    ("jv",  "1000-2000", 200, ["UD_Javanese-CSUI"]),
    # 1000-2000, mBERT/XLM-R, non-roman
    ("ta",  "1000-2000", 200, ["UD_Tamil-MWTT", "UD_Tamil-TTB"]),
    ("kk",  "1000-2000", 200, ["UD_Kazakh-KTB"]),
    # high resource, all models, roman
    ("vi",  "high", 500, "EXISTING"),
    ("fr",  "high", 500, "EXISTING"),
    ("de",  "high", 500, "EXISTING"),
    ("es",  "high", 500, "EXISTING"),
    ("tr",  "high", 500, "EXISTING"),
    # high resource, all models, non-roman
    ("hi",  "high", 500, "EXISTING"),
    ("ko",  "high", 500, "EXISTING"),
    ("ja",  "high", 500, "EXISTING"),
]

BASE_DIR = Path(__file__).resolve().parents[2]   # spectral-probing/ (this file is in direction1/transfer/)
UD_DIR = BASE_DIR / "data" / "ud-treebanks-v2.17"
UDSTD_DIR = BASE_DIR / "tasks" / "ud-syntax"
OUT_DIR = BASE_DIR / "tasks" / "ud-syntax" / "coverage-relations"
TARGETS_JSON = BASE_DIR / "direction1" / "transfer" / "coverage_targets.json"


# ---------------------------------------------------------------------------
# UD conversion helpers  (mirrors tasks/ud-syntax/convert.py)
# ---------------------------------------------------------------------------

def load_ud_module():
    """Add tasks/ud-syntax/ to sys.path so we can import ud.py."""
    ud_syntax_dir = str(UDSTD_DIR)
    if ud_syntax_dir not in sys.path:
        sys.path.insert(0, ud_syntax_dir)
    import importlib
    return importlib.import_module("ud")


def conllu_to_rows(conllu_path, ud_mod):
    """
    Parse one .conllu file and return a list of (text, labels) string pairs
    suitable for writing to the relations CSV format.
    """
    UDTreebank = ud_mod.UniversalDependenciesTreebank
    UD = ud_mod.UniversalDependencies

    treebank = UD(treebanks=[UDTreebank.from_conllu(str(conllu_path),
                                                    name=conllu_path.name)])
    rows = []
    for sidx in range(len(treebank)):
        words = treebank[sidx].to_words()
        words = [w.replace(' ', '') for w in words]
        _, dep = treebank[sidx].get_dependencies(include_subtypes=False)
        rows.append((' '.join(words), ' '.join(dep)))
    return rows


def process_ud_dirs(lang, ud_dir_names, ud_mod, dry_run=False):
    """Convert and merge all .conllu files from one or more UD treebank dirs."""
    all_rows = []
    for dir_name in ud_dir_names:
        tb_dir = UD_DIR / dir_name
        if not tb_dir.exists():
            print(f"  [WARN] UD dir not found: {tb_dir}")
            continue
        conllu_files = sorted(tb_dir.glob("*.conllu"))
        if not conllu_files:
            print(f"  [WARN] No .conllu files in {tb_dir}")
            continue
        for cf in conllu_files:
            if dry_run:
                print(f"    [dry-run] would convert {cf.name}")
            else:
                rows = conllu_to_rows(cf, ud_mod)
                all_rows.extend(rows)
                print(f"    Converted {cf.name}: {len(rows)} sentences")
    return all_rows


def process_existing(lang):
    """Merge existing train/dev/test CSVs from tasks/ud-syntax/{lang}/relations/."""
    rel_dir = UDSTD_DIR / lang / "relations"
    all_rows = []
    for csv_path in sorted(rel_dir.glob("*-relations.csv")):
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)   # skip header
            rows = list(reader)
        all_rows.extend(rows)
        print(f"    Merged {csv_path.name}: {len(rows)} sentences")
    return all_rows


def write_csv(out_path, rows, dry_run=False):
    if dry_run:
        print(f"  [dry-run] would write {len(rows)} sentences to {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"  Written: {out_path}  ({len(rows)} sentences)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be done without writing files")
    p.add_argument("--lang", default=None,
                   help="Process only this language code (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    ud_mod = load_ud_module()

    targets = []
    summary_rows = []

    for lang, tier, n_test, source_spec in LANGUAGES:
        if args.lang and lang != args.lang:
            continue

        print(f"\n=== {lang} (tier={tier}, n_test={n_test}) ===")
        out_path = OUT_DIR / f"{lang}-all-relations.csv"

        if out_path.exists() and not args.dry_run:
            print(f"  [skip] {out_path} already exists")
            # still count for targets.json
            with open(out_path, encoding="utf-8") as f:
                n_total = sum(1 for _ in f) - 1   # minus header
            n_train = n_total - n_test
            summary_rows.append((lang, tier, n_total, n_test, n_train))
            targets.append({
                "path":   str(out_path.relative_to(BASE_DIR)),
                "lang":   lang,
                "n_test": n_test,
                "tier":   tier,
            })
            continue

        if source_spec == "EXISTING":
            rows = process_existing(lang)
        else:
            rows = process_ud_dirs(lang, source_spec, ud_mod, dry_run=args.dry_run)

        write_csv(out_path, rows, dry_run=args.dry_run)

        n_total = len(rows)
        n_train = n_total - n_test
        summary_rows.append((lang, tier, n_total, n_test, n_train))
        targets.append({
            "path":   str(out_path.relative_to(BASE_DIR)),
            "lang":   lang,
            "n_test": n_test,
            "tier":   tier,
        })

    # --- write targets JSON ---
    if not args.dry_run:
        TARGETS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(TARGETS_JSON, "w") as f:
            json.dump(targets, f, indent=2)
        print(f"\nWrote {TARGETS_JSON}  ({len(targets)} targets)")

    # --- print summary table ---
    print("\n=== Summary ===")
    print(f"{'Lang':<6} {'Tier':<12} {'Total':>6} {'n_test':>7} {'train_pool':>11}  Warning")
    print("-" * 60)
    for lang, tier, n_total, n_test, n_train in sorted(summary_rows, key=lambda r: r[1]):
        warn = ""
        if n_train < 100:
            warn = "<< very small train pool!"
        elif n_train < 200:
            warn = "< small train pool"
        print(f"{lang:<6} {tier:<12} {n_total:>6} {n_test:>7} {n_train:>11}  {warn}")


if __name__ == "__main__":
    main()
