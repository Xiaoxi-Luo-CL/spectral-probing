#!/bin/bash
#SBATCH --job-name=d2-eval-readout
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Score one checkpoint (or a frozen model) with a chosen readout, to separate
# "spectral vs pooled representation" from the layer and dimensionality it is
# read at. The comparison is only meaningful when READOUT_LAYER is held fixed
# across arms: pooled at the final layer vs spectral at L8 differs in both.
#
# Usage:
#   sbatch eval_readout.slurm <run_dir|frozen> <model> <readout> [layer] [k_indices]
#
# Positional args:
#   $1  run_dir   run dir under direction2/results/, or the literal "frozen"
#   $2  model     mbert|xlmr
#   $3  readout   pooled | spectral
#   $4  layer     readout layer (default 8). pooled -> --main_layer,
#                 spectral -> --target_layer.
#   $5  k_indices space-separated coefficients for spectral (default: from
#                 config.json, or 1 2 3 4 when frozen). Pass a single k to
#                 dimension-match spectral (k*d dims) against pooled (d dims).
#
#   CORPUS and OUT_DIR env vars override the test corpus and results root.
#
# Examples:
#   sbatch eval_readout.slurm frozen mbert spectral 8 "1"
#   sbatch eval_readout.slurm direction2/results/fixedstep/mbert/L8_low_fk_k1-2-3-4_r16_s42 \
#          mbert pooled 8

PYTHON='/u501/x25luo/.conda/envs/midalign-spectral/bin/python'
CORPUS="${CORPUS:-direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json}"
OUT_DIR="${OUT_DIR:-direction2/results/fixedstep_eval}"

RUN_DIR="${1}"
MODEL="${2:-mbert}"
READOUT="${3:-spectral}"
LAYER="${4:-8}"
K_INDICES="${5:-}"

if [ -z "$RUN_DIR" ]; then
    echo 'usage: sbatch eval_readout.slurm <run_dir|frozen> <model> <readout> [layer] [k_indices]'
    exit 1
fi

# pooled reads main_layer, spectral reads target_layer -- same number, different flag
if [ "$READOUT" = 'pooled' ]; then
    LAYER_ARG="--main_layer $LAYER"
else
    LAYER_ARG="--target_layer $LAYER"
fi
[ -n "$K_INDICES" ] && K_ARG="--k_indices $K_INDICES" || K_ARG=''

if [ "$RUN_DIR" = 'frozen' ]; then
    SOURCE_ARGS="--frozen --model $MODEL"
else
    SOURCE_ARGS="--run_dir $RUN_DIR --model $MODEL"
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "readout=$READOUT layer=$LAYER k='${K_INDICES:-from config}' -> $OUT_DIR"

$PYTHON -m direction2.eval_full $SOURCE_ARGS --readout "$READOUT" $LAYER_ARG $K_ARG \
    --test_corpus "$CORPUS" --out_dir "$OUT_DIR"
