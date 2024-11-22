#!/bin/bash
#SBATCH --job-name=process_dat
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=6:00:00


eval "$(conda shell.bash hook)"
conda activate mel

export PYTHONPATH=.

python src/process_data.py