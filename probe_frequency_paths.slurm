#!/bin/bash
#SBATCH --job-name=freq-paths
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Stage A criteria 3 and 4: path consistency and the length-ratio confound.
#
# Usage:
#   sbatch probe_frequency_paths.slurm [model] [main_layer] [target_layer] [k_start] [k_end] [gamma] [max_blocks] [block_filter_k] [dtw_cost]
#
# Examples:
#   sbatch probe_frequency_paths.slurm mbert 8 8 1 32
#   sbatch probe_frequency_paths.slurm xlmr 6 6 1 32
#   sbatch -p COMPLING probe_frequency_paths.slurm qwen3-1.7b 16 24 1 32

MODEL="${1:-mbert}"
MAIN_LAYER="${2:-}"
TARGET_LAYER="${3:-}"
K_START="${4:-1}"
K_END="${5:-32}"
GAMMA="${6:-}"
MAX_BLOCKS="${7:-500}"
BLOCK_FILTER_K="${8:-64}"
DTW_COST="${9:-natural}"
PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
    --model "$MODEL"
    --k_start "$K_START"
    --k_end "$K_END"
    --dtw_cost "$DTW_COST"
    --max_blocks "$MAX_BLOCKS"
    --block_filter_k "$BLOCK_FILTER_K"
    --max_length 512
)

[ -n "$GAMMA" ] && ARGS+=(--gamma "$GAMMA")

[ -n "$MAIN_LAYER" ] && ARGS+=(--main_layer "$MAIN_LAYER")
[ -n "$TARGET_LAYER" ] && ARGS+=(--target_layer "$TARGET_LAYER")

echo "model=$MODEL main=${MAIN_LAYER:-default} target=${TARGET_LAYER:-default}"
echo "k=$K_START:$K_END gamma=${GAMMA:-auto} cost=$DTW_COST blocks=$MAX_BLOCKS bf=$BLOCK_FILTER_K node=$(hostname)"

"$PYTHON" -m direction2.probe_frequency_paths "${ARGS[@]}"
