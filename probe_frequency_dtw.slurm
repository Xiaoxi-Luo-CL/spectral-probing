#!/bin/bash
#SBATCH --job-name=freq-dtw
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch probe_frequency_dtw.slurm \
#       [model] [main_layer] [target_layer] [k_start] [k_end] [gamma] [max_blocks] \
#       [corpus] [block_filter_k] [fusion_mode] [fusion_value] [dtw_cost]
#
# Examples:
#   sbatch probe_frequency_dtw.slurm mbert 8 8 1 32
#   sbatch probe_frequency_dtw.slurm xlmr 6 6 1 32
#   sbatch -p COMPLING probe_frequency_dtw.slurm qwen3-1.7b 16 16 1 32
#   sbatch -p COMPLING probe_frequency_dtw.slurm qwen3-1.7b 16 24 1 32
#   sbatch -p COMPLING probe_frequency_dtw.slurm \
#       qwen3-1.7b 16 24 1 32 '' 500 '' 64 select
#   sbatch -p COMPLING probe_frequency_dtw.slurm \
#       qwen3-1.7b 16 24 1 32 '' 0 "$NTREX" 64 config "$UN_SELECTION"
#   sbatch probe_frequency_dtw.slurm mbert 8 8 1 32 0.1 500 '' 64 none '' equal_k

MODEL="${1:-mbert}"
MAIN_LAYER="${2:-}"
TARGET_LAYER="${3:-}"
K_START="${4:-1}"
K_END="${5:-32}"
GAMMA="${6:-}"
MAX_BLOCKS="${7:-500}"
CORPUS="${8:-}"
BLOCK_FILTER_K="${9:-64}"
FUSION_MODE="${10:-none}"
FUSION_VALUE="${11:-}"
DTW_COST="${12:-natural}"
PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
    --model "$MODEL"
    --k_start "$K_START"
    --k_end "$K_END"
    --dtw_cost "$DTW_COST"
    --max_blocks "$MAX_BLOCKS"
    --max_length 512
    --candidates 32
    --block_filter_k "$BLOCK_FILTER_K"
)

[ -n "$GAMMA" ] && ARGS+=(--gamma "$GAMMA")

[ -n "$CORPUS" ] && ARGS+=(--corpus "$CORPUS")

if [ -n "$MAIN_LAYER" ]; then
    ARGS+=(--main_layer "$MAIN_LAYER")
fi

if [ -n "$TARGET_LAYER" ]; then
    ARGS+=(--target_layer "$TARGET_LAYER")
fi

case "$FUSION_MODE" in
    none)
        ;;
    select)
        ARGS+=(--fusion_alphas)
        ;;
    fixed)
        [ -n "$FUSION_VALUE" ] || { echo 'fixed fusion requires an alpha'; exit 2; }
        ARGS+=(--fusion_alpha "$FUSION_VALUE")
        ;;
    config)
        [ -n "$FUSION_VALUE" ] || { echo 'config fusion requires a result JSON'; exit 2; }
        ARGS+=(--fusion_config "$FUSION_VALUE")
        ;;
    *)
        echo "unknown fusion mode: $FUSION_MODE"; exit 2
        ;;
esac

echo "model=$MODEL main_layer=${MAIN_LAYER:-default} target_layer=${TARGET_LAYER:-default}"
echo "k=$K_START:$K_END gamma=${GAMMA:-auto} cost=$DTW_COST max_blocks=$MAX_BLOCKS block_filter_k=$BLOCK_FILTER_K"
echo "corpus=${CORPUS:-default} fusion=$FUSION_MODE value=${FUSION_VALUE:-none} node=$(hostname)"

"$PYTHON" -m direction2.probe_frequency_dtw "${ARGS[@]}"
