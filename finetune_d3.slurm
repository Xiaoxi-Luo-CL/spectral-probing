#!/bin/bash
#SBATCH --job-name=d3-ft
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch finetune_d3.slurm [model] [spectral_mode] [seed] [k_indices] [patience] [max_steps] [target_layer] [main_layer]
#
#   $1  model          qwen3-1.7b | qwen3-4b        (default: qwen3-1.7b)
#   $2  spectral_mode  none | low_fk | rigid_freq | freq_dtw  (default: none)
#   $3  seed                                        (default: 42)
#   $4  k_indices      space-separated, e.g. "1 2 3 4" or "0" or "5 6 7 8"  (default: "1 2 3 4")
#   $5  patience       early-stop patience periods; 99 = fixed steps        (default: 2)
#   $6  max_steps                                   (default: 5000)
#   $7  target_layer  spectral auxiliary layer      (default: model-specific)
#   $8  main_layer    last-token InfoNCE layer      (default: model-specific; Qwen3-1.7B L16)
#   $9  dtw_gamma                                  (default: 0.1)
#   $10 dtw_blocks                                 (default: 4)
#   $11 dtw_band_radius; negative disables it      (default: -1)
#   $12 block_filter_k  pin the training population (default: "" = unpinned)
#   $13 batch_size                                  (default: 16)
#   $14 max_length                                  (default: 512)
#   $15 lambda_main    weight on main InfoNCE; 0 = spectral only (default: 1.0)
#   $16 save_last      1 = also keep the final-step checkpoint  (default: 1)
#   $17 dtw_cost       equal_k | natural (default: equal_k)
#   $18 lora_targets   qv | all  -- q/v projections or every linear layer (default: qv)
#
#   OUT_DIR env var overrides the results root, e.g.
#     sbatch --export=ALL,OUT_DIR=direction3/results/control_l24 finetune_d3.slurm ...
#   Use it to keep a control grid off the historical run paths.
#
# The original Qwen runs behind the +2.34pp used batch_size=4 and max_length=256.
# Matching a historical run means passing BOTH -- they change the block population
# (filter_token_safe drops, never truncates) as well as the optimisation.
#
# A distance mode silently filters the manifest up to max(k_indices). Pass the
# SAME $12 to the matched baseline, or the two conditions train on different
# block populations and are not comparable.
#
# Examples:
#   sbatch finetune_d3.slurm qwen3-1.7b low_fk 42               # standard run
#   sbatch finetune_d3.slurm qwen3-1.7b low_fk 42 "0"    99 3000  # DC aux, fixed 3000 steps
#   sbatch finetune_d3.slurm qwen3-1.7b low_fk 42 "5 6 7 8" 99 3000  # high-freq aux
#   # matched B0/B2 pair at K=32 -- note the shared trailing 32:
#   sbatch finetune_d3.slurm qwen3-1.7b freq_dtw 42 "$(seq -s ' ' 1 32)" 99 3000 24 16 0.1 4 -1 32
#   sbatch finetune_d3.slurm qwen3-1.7b none     42 "1 2 3 4"            99 3000 24 16 0.1 4 -1 32

MODEL="${1:-qwen3-1.7b}"
SPECTRAL_MODE="${2:-none}"
SEED="${3:-42}"
K_INDICES="${4:-1 2 3 4}"
PATIENCE="${5:-2}"
MAX_STEPS="${6:-5000}"
TARGET_LAYER="${7:-}"
MAIN_LAYER="${8:-}"
DTW_GAMMA="${9:-0.1}"
DTW_BLOCKS="${10:-4}"
DTW_BAND_RADIUS="${11:--1}"
BLOCK_FILTER_K="${12:-}"
BATCH_SIZE="${13:-16}"
MAX_LENGTH="${14:-512}"
LAMBDA_MAIN="${15:-1.0}"
SAVE_LAST="${16:-1}"
DTW_COST="${17:-equal_k}"
LORA_TARGETS="${18:-qv}"
OUT_DIR="${OUT_DIR:-direction3/results/finetune}"   # override via --export
PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'

LAYER_ARGS=()
[ -n "$TARGET_LAYER" ] && LAYER_ARGS+=(--target_layer "$TARGET_LAYER")
[ -n "$MAIN_LAYER" ] && LAYER_ARGS+=(--main_layer "$MAIN_LAYER")
[ -n "$BLOCK_FILTER_K" ] && LAYER_ARGS+=(--block_filter_k "$BLOCK_FILTER_K")

echo "model=$MODEL spectral_mode=$SPECTRAL_MODE seed=$SEED k_indices=$K_INDICES"
echo "target_layer=${TARGET_LAYER:-default} main_layer=${MAIN_LAYER:-default} block_filter_k=${BLOCK_FILTER_K:-unpinned}"
echo "batch_size=$BATCH_SIZE max_length=$MAX_LENGTH patience=$PATIENCE max_steps=$MAX_STEPS lora_targets=$LORA_TARGETS"
echo "out_dir=$OUT_DIR"
echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

"$PYTHON" -m direction3.finetune \
    --model "$MODEL" \
    --out_dir "$OUT_DIR" \
    --spectral_mode "$SPECTRAL_MODE" \
    --seed "$SEED" \
    --k_indices $K_INDICES \
    --patience "$PATIENCE" \
    --dtw_gamma "$DTW_GAMMA" \
    --dtw_blocks "$DTW_BLOCKS" \
    --dtw_cost "$DTW_COST" \
    --dtw_band_radius "$DTW_BAND_RADIUS" \
    "${LAYER_ARGS[@]}" \
    --manifest   direction2/data/UN_Corpus/aligned/un_train15k_manifest.json \
    --eval_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --lambda_main "$LAMBDA_MAIN" \
    --lora_targets "$LORA_TARGETS" \
    --max_steps "$MAX_STEPS" \
    $([ "$SAVE_LAST" = 1 ] && echo --save_last)
