#!/bin/bash
#SBATCH --job-name=llama-1b_clf
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=babel-0-23,shire-1-1


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw6

export PYTHONPATH=.

# model_dir='models/pythia-160m'
model_dir='models/llama-3.2-1B'
run_name="llama-1b_clf"
output_dir='trained_models'/$run_name

mkdir -p $output_dir

python src/trainer_clf.py   --model_dir $model_dir \
                            --run_name $run_name \
                            --output_dir $output_dir \
                            --max_length 256 \
                            # --debug

