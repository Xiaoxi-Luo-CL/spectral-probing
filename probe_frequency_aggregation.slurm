#!/bin/bash
#SBATCH --job-name=freq-aggregate
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage: sbatch probe_frequency_aggregation.slurm [model] [layers] [batch_size] [corpus] [max_blocks]
# Examples:
# sbatch probe_frequency_aggregation.slurm mbert '8' 16
# sbatch probe_frequency_aggregation.slurm qwen3-1.7b '16 24' 8

MODEL="${1:-mbert}"
LAYERS="${2:-}"
BATCH_SIZE="${3:-8}"
CORPUS="${4:-direction2/data/NTREX/aligned/ntrex_6way_pilot_min100.json}"
MAX_BLOCKS="${5:-}"
PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ARGS=(
    --model "$MODEL"
    --corpus "$CORPUS"
    --batch_size "$BATCH_SIZE"
    --max_length 512
    --max_k 8
    --population_models mbert qwen3-1.7b
)

if [ -n "$LAYERS" ]; then
    ARGS+=(--layers $LAYERS)
fi

if [ -n "$MAX_BLOCKS" ]; then
    ARGS+=(--max_blocks "$MAX_BLOCKS")
fi

echo "model=$MODEL layers=${LAYERS:-default} batch_size=$BATCH_SIZE"
echo "corpus=$CORPUS max_blocks=${MAX_BLOCKS:-all} max_k=8"
echo "node=$(hostname)"

"$PYTHON" -m direction2.probe_frequency_aggregation "${ARGS[@]}"
