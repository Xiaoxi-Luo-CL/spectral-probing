#!/bin/bash
#SBATCH --job-name=d2-eval-adapter
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch eval_adapter.slurm <adapter_dir> [test_corpus]
#
# Positional args:
#   $1  adapter_dir   path to FreqAdapter run dir (contains config.json + adapter.pt)
#   $2  test_corpus   path to test JSON (default: NTREX-25)

PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'
ADAPTER_DIR="${1}"
CORPUS="${2:-direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json}"
OUT_DIR='direction2/results/eval_adapter'

echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "adapter_dir=$ADAPTER_DIR  corpus=$CORPUS"

$PYTHON -m direction2.eval_adapter \
    --adapter_dir "$ADAPTER_DIR" \
    --test_corpus "$CORPUS" \
    --out_dir "$OUT_DIR" \
    --skip_dev 0
