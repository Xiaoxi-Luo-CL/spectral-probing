#!/bin/bash
#SBATCH --job-name=d3-probe
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch probe_d3.slurm [model] [corpus]
#
#   $1  model   qwen3-1.7b | qwen3-4b     (default: qwen3-1.7b)
#   $2  corpus  path to eval JSON         (default: UN 6-way eval)
#   $3  k_indices  spectral coefficients, e.g. "1" (default: "1 2 3 4");
#              a non-default set writes <corpus>_k<list>.csv

MODEL="${1:-qwen3-1.7b}"
CORPUS="${2:-direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json}"
K_INDICES="${3:-1 2 3 4}"
OUT_DIR="${OUT_DIR:-direction3/results/probe}"   # override via --export to keep reruns separate
PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'

echo "model=$MODEL  corpus=$CORPUS  k_indices=$K_INDICES"
echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

$PYTHON -m direction3.probe \
    --model "$MODEL" \
    --corpus "$CORPUS" \
    --k_indices $K_INDICES \
    --out_dir "$OUT_DIR"
