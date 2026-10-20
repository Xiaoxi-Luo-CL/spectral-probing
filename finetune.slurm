#!/bin/bash
#SBATCH --job-name=d2-finetune
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Usage:
#   sbatch finetune.slurm [model] [target_layer] [k_mode] [rank] [seed] [pooling] [gate]
#
# Positional args (all optional):
#   $1  model          mbert | xlmr | labse               (default: mbert)
#   $2  target_layer   hidden-state index                  (default: 8)
#   $3  k_mode         DCT condition string:               (default: "1 2 3 4")
#                        "0"          DC only
#                        "1 2 3 4"   Low k=1..4
#                        "nospectral" hidden-state LoRA (no DCT)
#                        "random4"    random 4 modes from k=5..32
#                        "allnondc"   all non-DC modes k=1..T-1 (dynamic per sequence)
#   $4  rank                                               (default: 16)
#   $5  seed                                               (default: 42)
#   $6  pooling        mean | cls                          (default: mean)
#   $7  gate           1 to use element-wise gate g_k      (default: 0)
#   $8  lambda_id      identity preservation weight         (default: 0.01)
#   $9  out_dir        output directory                     (default: direction2/results/adapters)
#
# Examples:
#   sbatch finetune.slurm mbert 8 "1 2 3 4" 16 42 mean    # Low k=1..4, mean-pool
#   sbatch finetune.slurm mbert 8 "1 2 3 4" 16 42 cls     # Low k=1..4, CLS
#   sbatch finetune.slurm mbert 8 "1 2 3 4" 16 42 mean 1  # gate adapter, k=1..4
#   sbatch finetune.slurm mbert 8 "1 2 3 4" 16 42 mean 0 0  # no id_loss
#   sbatch finetune.slurm mbert 8 nospectral 16 42 mean    # non-spectral LoRA

MODEL="${1:-mbert}"
LAYER="${2:-8}"
K_MODE="${3:-1 2 3 4}"
RANK="${4:-16}"
SEED="${5:-42}"
POOLING="${6:-mean}"
GATE="${7:-0}"
LAMBDA_ID="${8:-0.01}"
OUT_DIR="${9:-direction2/results/adapters}"

PYTHON="/u501/x25luo/.conda/envs/grounding/bin/python"

GATE_ARGS=""
[ "$GATE" = '1' ] && GATE_ARGS='--gate'

echo "model=$MODEL  layer=$LAYER  k_mode=[$K_MODE]  rank=$RANK  seed=$SEED  pooling=$POOLING  gate=$GATE  lambda_id=$LAMBDA_ID  out_dir=$OUT_DIR"
echo "node=$(hostname)  cuda=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# Build the k-mode CLI args based on the k_mode string
case "$K_MODE" in
    nospectral|"")
        K_ARGS="--no_spectral"
        ;;
    random4)
        K_ARGS="--random_k_n 4"
        ;;
    allnondc)
        K_ARGS="--all_nondc"
        ;;
    *)
        K_ARGS="--k_indices $K_MODE"
        ;;
esac

$PYTHON -m direction2.finetune \
    --model "$MODEL" \
    --target_layer "$LAYER" \
    $K_ARGS \
    --rank "$RANK" \
    --seed "$SEED" \
    --pooling "$POOLING" \
    --lambda_id "$LAMBDA_ID" \
    --out_dir "$OUT_DIR" \
    $GATE_ARGS \
    --manifest   direction2/data/UN_Corpus/aligned/un_train15k_manifest.json \
    --eval_corpus direction2/data/UN_Corpus/aligned/un_6way_eval2000_min100.json \
    --max_steps 5000
