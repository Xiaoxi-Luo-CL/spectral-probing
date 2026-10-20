#!/bin/bash
# Related-language transfer at a 100-sentence budget (pairs in direction1/transfer/related_pairs.json).
# A fresh probe is trained on 100 target sentences (80 train / 20 dev) with and without the frozen
# source filter; every source runs on all 6 targets, so related and unrelated sources share one test set.
# Train pool = first min(200, total - 100) sentences, test = the following ones (at most 2000); fresh
# probes predict all coarse UD relations; lr is chosen on dev from a grid. One job per (model, source, seed).
#
# Environment overrides (defaults in brackets):
#   MODELS ["bert-base-multilingual-cased CohereLabs/aya-expanse-8b"]
#   SOURCES ["hi cs de ca it zh en"]   SEEDS ["42 43 44 45 46"]
#   e.g. smoke run:  MODELS=bert-base-multilingual-cased SOURCES=zh SEEDS=42 bash direction1/scripts/submit_pairs.sh
#
# Usage (from spectral-probing/):
#   bash direction1/scripts/submit_pairs.sh [--dry-run]

set -euo pipefail

SLURM_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_transfer.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

read -r -a MODEL_LIST <<< "${MODELS:-bert-base-multilingual-cased CohereLabs/aya-expanse-8b}"
read -r -a SOURCE_LIST <<< "${SOURCES:-hi cs de ca it zh en}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44 45 46}"
SETTINGS="Wscratch_Gnone,Wscratch_Gfroze"
N=100
TARGET_LIST="direction1/transfer/pair_targets.json"
read -r -a TARGETS <<< "$(python3 -c "import json, sys; print(' '.join(t['lang'] for t in json.load(open(sys.argv[1]))))" "$TARGET_LIST")"
OUTPUT_DIR="direction1/transfer/results_pairs_n100"
EXTRA=(--epochs 100 --early_stop 2 --min_delta 1e-3 --pool_size 200 --min_test 100 --max_test 2000
       --target_labels --lr_grid 3e-4,1e-3,3e-3,1e-2)

# a source runs only once its filtered probe is registered in the index lookup (see train_source.slurm)
has_source() {
    python3 -c "import json, sys; sys.exit(sys.argv[1] not in json.load(open('direction1/transfer/results_index.json'))['lookup'])" \
        "$1|$2|relations|$3"
}

total=0
skipped=0
for model in "${MODEL_LIST[@]}"; do
    # encoders: avg pooling; decoders: last token and a smaller embedding batch for 48GB GPUs
    case "$model" in
        *roberta*|*bert-base*) pool=avg; embed_batch=32 ;;
        *) pool=last; embed_batch=8 ;;
    esac
    for src in "${SOURCE_LIST[@]}"; do
        if ! has_source "$model" "$src" "$pool"; then
            echo "[SKIP] no registered source probe for $model|$src|relations|$pool"
            skipped=$((skipped + ${#SEED_LIST[@]}))
            continue
        fi
        for seed in "${SEED_LIST[@]}"; do
            name="pairs_$(basename "$model")_${src}_s${seed}"
            if $DRY_RUN; then
                echo "[DRY-RUN] $name  pool=$pool embed_batch=$embed_batch"
            else
                echo "Submitting: $name"
                # make the result directories here, in one process: jobs creating them concurrently on
                # NFS can see a sibling's fresh directory as missing and crash when saving
                for tgt in "${TARGETS[@]}"; do
                    for st in ${SETTINGS//,/ }; do
                        mkdir -p "$OUTPUT_DIR/$(basename "$model")/${src}2${tgt}/relations/${st}_n${N}/seed${seed}"
                    done
                done
                sbatch --job-name="$name" "$SLURM_SCRIPT" "$model" "$src" "$pool" relations "$SETTINGS" \
                    "$N" "$TARGET_LIST" "$OUTPUT_DIR" "$seed" "$embed_batch" "${EXTRA[@]}"
            fi
            total=$((total + 1))
        done
    done
done
echo "jobs: $total  skipped (missing source probe): $skipped"
