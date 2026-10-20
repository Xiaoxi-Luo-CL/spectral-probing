#!/bin/bash
#SBATCH --job-name=d2-eval-full
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch eval_full.slurm [run_dir] [model] [corpus] [checkpoint]
#   sbatch eval_full.slurm frozen [model]     # frozen baseline (default: mbert)
#
# Positional args:
#   $1  run_dir  path to finetune_full run dir OR the literal string "frozen"
#   $2  model    mbert|xlmr  (only used when $1=frozen, default: mbert)
#   $3  corpus   test corpus path (default: ntrex_25way_transfer_min100.json)
#   $4  checkpoint  best | last (default: best; ignored for frozen models)
#
#   OUT_DIR env var overrides the results root. Eval CSVs are named from the run
#   directory BASENAME only, so identically-named runs under different roots collide.
#   Always set it when evaluating a separate grid.

PYTHON='/u501/x25luo/.conda/envs/midalign-spectral/bin/python'
CORPUS="${3:-direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json}"
OUT_DIR="${OUT_DIR:-direction2/results/eval_full}"   # override via --export

RUN_DIR="${1}"
MODEL="${2:-mbert}"
CHECKPOINT="${4:-best}"

echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

if [ "$RUN_DIR" = 'frozen' ]; then
    echo "mode=frozen  model=$MODEL"
    $PYTHON -m direction2.eval_full \
        --frozen --model "$MODEL" \
        --test_corpus "$CORPUS" \
        --out_dir "$OUT_DIR"
else
    echo "run_dir=$RUN_DIR  checkpoint=$CHECKPOINT"
    $PYTHON -m direction2.eval_full \
        --run_dir "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --test_corpus "$CORPUS" \
        --out_dir "$OUT_DIR"
fi
