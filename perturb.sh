#!/bin/bash

#SBATCH --job-name=spectral
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --partition=ALL
#SBATCH -x watgpu1008
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

#SBATCH -o slurm/JOB-%j.out
#SBATCH -e slurm/JOB-%j.err


path="$1"
/u501/x25luo/.conda/envs/grounding/bin/python -m my_code.prediction --exp_path ${path}
