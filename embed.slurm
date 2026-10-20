#!/bin/bash
#SBATCH --job-name=d2-embed
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch embed.slurm [model] [blocks]
#
# Positional args (all optional):
#   $1  model   mbert | xlmr | labse          (default: mbert)
#   $2  blocks  path to blocks JSON            (default: NTREX 6-way pilot)

MODEL="${1:-mbert}"
BLOCKS="${2:-direction2/data/NTREX/aligned/ntrex_6way_pilot_min100.json}"
LANGS="en de ar zh hi sw"
OUT_DIR="direction2/results/embed"

PYTHON="/u501/x25luo/.conda/envs/grounding/bin/python"

echo "model=$MODEL  blocks=$BLOCKS"
echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

$PYTHON -m direction2.embed \
    --model "$MODEL" \
    --langs $LANGS \
    --blocks "$BLOCKS" \
    --out_dir "$OUT_DIR"
