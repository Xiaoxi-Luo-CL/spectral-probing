#!/bin/bash
#SBATCH --job-name=d2-full
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch finetune_full.slurm [model] ... [lambda_main] [save_last] [dtw_cost]
#
# Positional args (all optional):
#   $1  model           mbert | xlmr                   (default: mbert)
#   $2  target_layer    intermediate layer for spectral aux loss  (default: 8)
#   $3  spectral_mode   none | low_rms | low_fk | rigid_freq | freq_dtw
#   $4  lora_r          LoRA rank                       (default: 16)
#   $5  seed                                            (default: 42)
#   $6  main_layer      layer for main InfoNCE (default: "" = final layer)
#   $7  k_indices       space-separated DCT indices     (default: "1 2 3 4")
#   $8  dtw_gamma                                       (default: 0.1)
#   $9  dtw_blocks                                      (default: 4)
#   $10 block_filter_k  pin the training population     (default: "" = unpinned)
#   $11 max_steps                                      (default: 5000)
#   $12 patience        eval periods; 99 = fixed steps (default: 2)
#   $13 lambda_main     weight on main InfoNCE; 0 = spectral only (default: 1.0)
#   $14 save_last       1 = also keep the final-step checkpoint (default: 1)
#   $15 dtw_cost        equal_k | natural (default: equal_k)
#   $16 lambda_spectral weight on the auxiliary loss (default: 0.1)
#   $17 pooling         main-loss vector: mean | cls (default: mean)
#
# A distance mode silently filters the manifest up to max(k_indices). Pass the
# SAME $10 to the matched baseline, or the two conditions train on different
# block populations and are not comparable.
#
# Examples:
#   sbatch finetune_full.slurm mbert 8 none 16 42      # LoRA only, no spectral aux
#   sbatch finetune_full.slurm mbert 8 low_fk 16 42    # spectral aux at L8, main at final
#   sbatch finetune_full.slurm mbert 8 low_fk 16 42 8  # both losses at L8
#   sbatch finetune_full.slurm mbert 8 low_fk 16 42 '' '1 2'
#   # matched B0/B2 pair at K=32 -- note the shared trailing 32:
#   sbatch finetune_full.slurm mbert 8 freq_dtw 16 42 '' "$(seq -s ' ' 1 32)" 0.1 4 32
#   sbatch finetune_full.slurm mbert 8 none    16 42 '' '1 2 3 4'            0.1 4 32

MODEL="${1:-mbert}"
LAYER="${2:-8}"
SPECTRAL_MODE="${3:-low_rms}"
LORA_R="${4:-16}"
SEED="${5:-42}"
MAIN_LAYER="${6:-}"
K_INDICES="${7:-1 2 3 4}"
DTW_GAMMA="${8:-0.1}"
DTW_BLOCKS="${9:-4}"
BLOCK_FILTER_K="${10:-}"
MAX_STEPS="${11:-5000}"
PATIENCE="${12:-2}"
LAMBDA_MAIN="${13:-1.0}"
SAVE_LAST="${14:-1}"
DTW_COST="${15:-equal_k}"
LAMBDA_SPECTRAL="${16:-0.1}"
POOLING="${17:-mean}"

OUT_DIR="${OUT_DIR:-direction2/results/finetune_full}"   # override via --export
PYTHON="/u501/x25luo/.conda/envs/midalign-spectral/bin/python"

EXTRA_ARGS=()
[ -n "$MAIN_LAYER" ] && EXTRA_ARGS+=(--main_layer "$MAIN_LAYER")
[ -n "$BLOCK_FILTER_K" ] && EXTRA_ARGS+=(--block_filter_k "$BLOCK_FILTER_K")

echo "model=$MODEL  layer=$LAYER  spectral_mode=$SPECTRAL_MODE  lora_r=$LORA_R  seed=$SEED  main_layer=${MAIN_LAYER:-final}  k_indices=$K_INDICES  block_filter_k=${BLOCK_FILTER_K:-unpinned}"
echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

$PYTHON -m direction2.finetune_full \
    --model "$MODEL" \
    --out_dir "$OUT_DIR" \
    --target_layer "$LAYER" \
    --spectral_mode "$SPECTRAL_MODE" \
    --lora_r "$LORA_R" \
    --seed "$SEED" \
    --k_indices $K_INDICES \
    --dtw_gamma "$DTW_GAMMA" \
    --dtw_blocks "$DTW_BLOCKS" \
    --dtw_cost "$DTW_COST" \
    "${EXTRA_ARGS[@]}" \
    --manifest   direction2/data/UN_Corpus/aligned/un_train15k_manifest.json \
    --eval_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json \
    --max_steps "$MAX_STEPS" \
    --patience "$PATIENCE" \
    --lambda_main "$LAMBDA_MAIN" \
    --lambda_spectral "$LAMBDA_SPECTRAL" \
    --pooling "$POOLING" \
    $([ "$SAVE_LAST" = 1 ] && echo --save_last)
