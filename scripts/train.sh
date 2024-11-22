#!/bin/bash
#SBATCH --job-name=train-debug
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=babel-0-23


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw6

export PYTHONPATH=.

model_dir='models/pythia-160m'
run_name="pythia-160m_freeze6_lora"
output_dir='trained_models'/$run_name

mkdir -p $output_dir

python src/trainer.py   --model_dir $model_dir \
                        --run_name $run_name \
                        --output_dir $output_dir \
