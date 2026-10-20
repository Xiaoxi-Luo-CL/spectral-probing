#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Run all conditions × all target languages for one (model, src_lang) pair.
# Calls run_batch.py which loads the LM once, warms the embedding cache per
# target language, and iterates over all conditions and n_examples values.
# Existing result files are skipped automatically.
#
# Arguments:
#   $1  model          full HF model ID (e.g. CohereLabs/aya-expanse-8b)
#   $2  src_lang       source language code (e.g. en)
#   $3  src_pool       pool used in source experiment (avg for encoders, last for decoders)
#   $4  task           relations | pos  (default: relations)
#   $5  settings       comma-separated W<w>_G<g> / plain settings, or all (default: all)
#   $6  n_examples     comma-separated list (default: 100,500,1000)
#   $7  target_list    path to coverage_targets.json (default: direction1/transfer/coverage_targets.json)
#   $8  output_dir     output directory (default: direction1/transfer/results_coverage_v3)
#   $9  seed           random seed (default: 42); results go into seed{N}/ subdirectory
#   $10 embed_batch    sentences per LM forward pass when filling the embedding cache
#                      (default: 32). Use 8 for the 7-8B decoders on 48GB GPUs; probe
#                      training keeps --batch_size 32 either way.
#   $11...             any further arguments are passed to run_batch.py verbatim and
#                      override the defaults below (argparse keeps the last value), e.g.
#                      --largest_n_only --plain_sources <json> --learning_rate 3e-3

set -euo pipefail   # a failed run must fail the job, not end on the trailing echo

PYTHON="/u501/x25luo/.conda/envs/grounding/bin/python"

model="$1"
src_lang="$2"
src_pool="${3:-avg}"
task="${4:-relations}"
settings="${5:-all}"
n_examples="${6:-100,500,1000}"
target_list="${7:-direction1/transfer/coverage_targets.json}"
output_dir="${8:-direction1/transfer/results_coverage_v3}"
seed="${9:-42}"
embed_batch="${10:-32}"

model_short=$(basename "$model")

echo "================================================================"
echo "Model       : $model"
echo "Src lang    : $src_lang"
echo "Src pool    : $src_pool"
echo "Task        : $task"
echo "Settings    : $settings"
echo "n_examples  : $n_examples"
echo "Target list : $target_list"
echo "Output dir  : $output_dir"
echo "Seed        : $seed"
echo "Embed batch : $embed_batch"
echo "================================================================"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # all four models are in the local cache

$PYTHON -m direction1.transfer.run_batch \
    --model "$model" \
    --src_lang "$src_lang" \
    --src_pool "$src_pool" \
    --task "$task" \
    --target_list "$target_list" \
    --settings "$settings" \
    --n_examples "$n_examples" \
    --pool_word_embed "$src_pool" \
    --output_dir "$output_dir" \
    --random_seed "$seed" \
    --epochs 100 \
    --early_stop 2 \
    --min_delta 1e-3 \
    --batch_size 32 \
    --embed_batch_size "$embed_batch" \
    --learning_rate 1e-3 \
    --decay_rate 0.5 \
    "${@:11}"

echo ""
echo "================================================================"
echo "Done: $model_short | $src_lang"
echo "================================================================"
