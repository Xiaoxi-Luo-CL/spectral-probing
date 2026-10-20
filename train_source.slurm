#!/bin/bash
#SBATCH --job-name=source-probe
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu1109
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err

# Train a DEP source probe for transfer, with or without a spectral filter. Every source uses
# the same recipe: word pooling, linear probe, up to 30 epochs with patience 2, batch 32,
# lr 1e-3, seed 42; filtered sources use 'auto(512)' with classify.py's default init.
# The run lands in results/<timestamp>_<model>_relations_<lang>/. Register filter-free runs in
# direction1/transfer/plain_sources.json and filtered ones in the lookup of results_index.json,
# both as "<model>|<lang>|relations|<pool>".
#
# Usage (from spectral-probing/):
#   sbatch train_source.slurm <model> <lang> <pool> [filter, default 'nofilter()']
#   e.g. sbatch train_source.slurm bert-base-multilingual-cased zh avg
#        sbatch train_source.slurm CohereLabs/aya-expanse-8b cs last 'auto(512)'

set -euo pipefail

PYTHON="/u501/x25luo/.conda/envs/grounding/bin/python"
model="$1"
lang="$2"
pool="$3"
filter="${4:-nofilter()}"
case "$lang" in
    en) treebank=ewt ;;
    zh) treebank=gsd ;;
    hi) treebank=hdtb ;;
    de) treebank=gsd ;;
    cs) treebank=cac ;;
    ca) treebank=ancora ;;
    it) treebank=isdt ;;
    *) echo "no treebank configured for source language $lang"; exit 1 ;;
esac
data="tasks/ud-syntax/$lang/relations/$lang-$treebank"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
$PYTHON -m direction1.classify "$data-train-relations.csv" "$data-dev-relations.csv" "$model" "$filter" \
    -ec --pool_word_embed "$pool" --classifier linear \
    -e 30 -es 2 -bs 32 -lr 1e-3 -dr 0.5 -rs 42
