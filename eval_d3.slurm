#!/bin/bash
#SBATCH --job-name=d3-eval
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch eval_d3.slurm [run_dir] [corpus] [checkpoint]
#   sbatch eval_d3.slurm frozen [model] [main_layer] [corpus]
#
#   $1  run_dir  path to direction3 run dir OR literal "frozen"
#
#   OUT_DIR env var overrides the results root. Eval CSVs are named from the run
#   directory BASENAME only, so two runs with the same name under different roots
#   collide. Always set OUT_DIR when evaluating a separate grid:
#     sbatch --export=ALL,OUT_DIR=direction3/results/control_l24_eval eval_d3.slurm ...
#   $2  corpus   test corpus path (default: ntrex_25way)
#   $3  checkpoint  best | last (default: best; non-frozen runs only)
#       (when frozen: $2=model, $3=main_layer, $4=corpus)

OUT_DIR="${OUT_DIR:-direction3/results/eval}"   # override via --export
PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'
NTREX25='direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json'

echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

if [ "$1" = 'frozen' ]; then
    MODEL="${2:-qwen3-1.7b}"
    MAIN_LAYER="${3:-16}"
    CORPUS="${4:-$NTREX25}"
    echo "mode=frozen  model=$MODEL  main_layer=$MAIN_LAYER"
    $PYTHON -m direction3.eval \
        --frozen --model "$MODEL" --main_layer "$MAIN_LAYER" \
        --out_dir "$OUT_DIR" \
        --test_corpus "$CORPUS"
else
    RUN_DIR="$1"
    CORPUS="${2:-$NTREX25}"
    CHECKPOINT="${3:-best}"
    echo "run_dir=$RUN_DIR  checkpoint=$CHECKPOINT"
    $PYTHON -m direction3.eval \
        --run_dir "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --out_dir "$OUT_DIR" \
        --test_corpus "$CORPUS"
fi
