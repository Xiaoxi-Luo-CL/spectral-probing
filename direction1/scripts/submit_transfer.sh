#!/bin/bash
# Coverage experiment submission: 4 models × 2 src langs × 3 seeds = 24 SLURM jobs.
#
# Each job calls run_batch.py (one LM load) over all 24 target languages,
# 8 conditions, and n_examples=[50,100,500,1000] for one (model, src, seed).
# Results land in seed{N}/ subdirectories so seeds never overwrite each other.
#
# Models  : xlm-roberta-base, bert-base-multilingual-cased,
#           aya-expanse-8b, Qwen2.5-7B
# Sources : en, zh
# Targets : direction1/transfer/coverage_targets.json  (24 languages)
# Seeds   : 42 43 44
# Output  : direction1/transfer/results_coverage_v2 (the original grid stays in results_coverage)
#
# Usage (from spectral-probing/):
#   bash submit_transfer.sh            # submit 24 jobs
#   bash submit_transfer.sh --dry-run  # preview without submitting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/slurm_transfer.sh"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TASK="relations"
CONDITIONS="Wfroze_Gfroze,Wfroze_Gnone,Wscratch_Gscratch,Wtune_Gtune,Wscratch_Gfroze,Wtune_Gfroze,Wscratch_Gnone,Wtune_Gnone"  # v2 letters a-h
N_EXAMPLES="50,100,500,1000"
# v2 protocol, passed explicitly since slurm_transfer.sh now defaults to the v3 one
EXTRA=(--epochs 10 --early_stop 1 --min_delta 0)
TARGET_LIST="direction1/transfer/coverage_targets.json"
OUTPUT_DIR="direction1/transfer/results_coverage_v2"
SEEDS=(42 43 44)

ENCODER_MODELS=(
    "FacebookAI/xlm-roberta-base"
    "bert-base-multilingual-cased"
)
DECODER_MODELS=(
    "CohereLabs/aya-expanse-8b"
    "Qwen/Qwen2.5-7B"
)

SRC_LANGS=("en" "zh")

TOTAL=0

for model in "${ENCODER_MODELS[@]}" "${DECODER_MODELS[@]}"; do
    model_short=$(basename "$model")

    # Encoder models use avg pooling; decoder models use last and a smaller embedding
    # batch so the 7-8B fp32 models fit 48GB GPUs (probe training stays at batch 32)
    if [[ " ${ENCODER_MODELS[*]} " == *" $model "* ]]; then
        src_pool="avg"
        embed_batch=32
    else
        src_pool="last"
        embed_batch=8
    fi

    for src_lang in "${SRC_LANGS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            job_name="${model_short}_${src_lang}_${TASK}_s${seed}"

            if $DRY_RUN; then
                echo "[DRY-RUN] $job_name  src_pool=$src_pool  seed=$seed  embed_batch=$embed_batch"
            else
                echo "Submitting: $job_name  src_pool=$src_pool  seed=$seed  embed_batch=$embed_batch"
                sbatch --job-name="$job_name" \
                    "$SLURM_SCRIPT" \
                    "$model" "$src_lang" "$src_pool" \
                    "$TASK" "$CONDITIONS" "$N_EXAMPLES" \
                    "$TARGET_LIST" "$OUTPUT_DIR" "$seed" "$embed_batch" "${EXTRA[@]}"
            fi
            (( TOTAL++ )) || true
        done
    done
done

echo ""
if $DRY_RUN; then
    echo "Dry-run complete. Jobs that would be submitted: $TOTAL"
else
    echo "Done. Jobs submitted: $TOTAL"
fi
