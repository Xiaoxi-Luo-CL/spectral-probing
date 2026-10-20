#!/bin/bash
# Transfer grid v3: 12 W x gamma settings + plain, largest feasible n per target, up to 100 epochs (stopping set by convergence) with
# patience 2 counting only dev-loss drops of at least 0.1% (relative), scratch filters drawn with the source recipe. One job per (model, source, seed).
#
# Environment overrides (defaults in brackets):
#   SETTINGS   by default the first seed runs "all" and later seeds run "train", because
#              zero-shot settings train nothing and would only repeat; set it to force a list
#   SEEDS ["42 43 44"]   MODELS [all four]
#   OUTPUT_DIR [direction1/transfer/results_coverage_v3]   LR [1e-3, the slurm_transfer.sh default]
#   e.g. lr check:  MODELS=FacebookAI/xlm-roberta-base LR=3e-3 SETTINGS=train OUTPUT_DIR=direction1/transfer/results_lr/lr3e-3 bash ...
#   e.g. zero-shot first:  SETTINGS=Wfroze_Gnone,Wfroze_Gfroze,plain SEEDS=42 bash direction1/scripts/submit_transfer_v3.sh
#
# Usage (from spectral-probing/):
#   bash direction1/scripts/submit_transfer_v3.sh [--dry-run]

set -euo pipefail

SLURM_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/slurm_transfer.sh"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

FORCED="${SETTINGS:-}"
read -r -a SEED_LIST <<< "${SEEDS:-42 43 44}"
read -r -a MODEL_LIST <<< "${MODELS:-FacebookAI/xlm-roberta-base bert-base-multilingual-cased CohereLabs/aya-expanse-8b Qwen/Qwen2.5-7B}"
OUTPUT_DIR="${OUTPUT_DIR:-direction1/transfer/results_coverage_v3}"
EXTRA=(--epochs 100 --early_stop 2 --min_delta 1e-3 --largest_n_only --plain_sources direction1/transfer/plain_sources.json)
[[ -n "${LR:-}" ]] && EXTRA+=(--learning_rate "$LR")  # later arguments override slurm_transfer.sh defaults

total=0
for model in "${MODEL_LIST[@]}"; do
    # encoders: avg pooling; decoders: last token and a smaller embedding batch for 48GB GPUs
    case "$model" in
        *roberta*|*bert-base*) pool=avg; embed_batch=32 ;;
        *) pool=last; embed_batch=8 ;;
    esac
    for src in en zh; do
        for seed in "${SEED_LIST[@]}"; do
            if [[ -n "$FORCED" ]]; then SETTINGS="$FORCED"
            elif [[ "$seed" == "${SEED_LIST[0]}" ]]; then SETTINGS=all
            else SETTINGS=train; fi
            name="v3_$(basename "$model")_${src}_s${seed}${LR:+_lr$LR}"
            if $DRY_RUN; then
                echo "[DRY-RUN] $name  pool=$pool embed_batch=$embed_batch settings=$SETTINGS"
            else
                echo "Submitting: $name"
                sbatch --job-name="$name" "$SLURM_SCRIPT" "$model" "$src" "$pool" relations "$SETTINGS" \
                    100,500,1000 direction1/transfer/coverage_targets.json "$OUTPUT_DIR" "$seed" "$embed_batch" "${EXTRA[@]}"
            fi
            total=$((total + 1))
        done
    done
done
echo "jobs: $total"
