#!/bin/bash
#SBATCH --job-name=d3-eval-readout
#SBATCH --time=03:00:00
#SBATCH --mem=48GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Decoder counterpart of eval_readout.slurm: score one Qwen checkpoint with a
# chosen readout, holding the layer fixed so the arms differ only in how the
# sentence vector is formed.
#
# ``pooled`` is the last content token at main_layer -- for a decoder that is the
# only admissible pooled readout. ``spectral`` uses non-DC DCT coefficients at
# target_layer; note it aggregates over all positions, so it is not a last-token
# readout, and k=0 would be mean-pooling and is deliberately never used here.
#
# NOTE: direction3 runs in the 'grounding' env, not 'midalign-spectral'. Qwen3's
# tokenizer.json needs tokenizers>=0.21; the direction2 env has 0.19 and fails to
# parse it.
#
# Usage:
#   sbatch eval_readout_d3.slurm <run_dir|frozen> <readout> [layer] [k_indices]
#
#   $1  run_dir   run dir under direction3/results/, or the literal "frozen"
#   $2  readout   pooled | spectral
#   $3  layer     readout layer. pooled -> --main_layer, spectral -> --target_layer.
#                 Omit for a trained run to use config.json.
#   $4  k_indices space-separated spectral coefficients (default: from config.json).
#                 A single k dimension-matches spectral (k*d) against pooled (d).
#
#   MODEL, CORPUS and OUT_DIR env vars override the model, test corpus and root.
#
# Examples:
#   sbatch eval_readout_d3.slurm frozen spectral 16 "1"
#   sbatch eval_readout_d3.slurm direction3/results/finetune/qwen3-1.7b/L16_low_fk_k1-2-3-4_r16_m16_s42 spectral "" "1"

PYTHON='/u501/x25luo/.conda/envs/grounding/bin/python'
MODEL="${MODEL:-qwen3-1.7b}"
CORPUS="${CORPUS:-direction2/data/NTREX/aligned/ntrex_25way_transfer_min100.json}"
OUT_DIR="${OUT_DIR:-direction3/results/readout_eval}"

RUN_DIR="${1}"
READOUT="${2:-spectral}"
LAYER="${3:-}"
K_INDICES="${4:-}"

if [ -z "$RUN_DIR" ]; then
    echo 'usage: sbatch eval_readout_d3.slurm <run_dir|frozen> <readout> [layer] [k_indices]'
    exit 1
fi

if [ -n "$LAYER" ]; then
    [ "$READOUT" = 'pooled' ] && LAYER_ARG="--main_layer $LAYER" || LAYER_ARG="--target_layer $LAYER"
else
    LAYER_ARG=''
fi
[ -n "$K_INDICES" ] && K_ARG="--k_indices $K_INDICES" || K_ARG=''

if [ "$RUN_DIR" = 'frozen' ]; then
    # --main_layer is required when frozen, so a spectral frozen run needs both flags
    [ -z "$LAYER" ] && { echo 'frozen requires a layer'; exit 1; }
    SOURCE_ARGS="--frozen --model $MODEL --main_layer $LAYER"
    [ "$READOUT" = 'spectral' ] && LAYER_ARG="--target_layer $LAYER" || LAYER_ARG=''
else
    SOURCE_ARGS="--run_dir $RUN_DIR --model $MODEL"
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "readout=$READOUT layer='${LAYER:-cfg}' k='${K_INDICES:-cfg}' -> $OUT_DIR"

$PYTHON -m direction3.eval $SOURCE_ARGS --readout "$READOUT" $LAYER_ARG $K_ARG \
    --test_corpus "$CORPUS" --out_dir "$OUT_DIR"
