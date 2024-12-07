#!/bin/bash
#SBATCH --job-name=pythia-160m_dpo_faster
#SBATCH --output=logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --exclude=babel-0-23,shire-1-1,babel-3-21


eval "$(conda shell.bash hook)"
conda activate cmu-llms-hw6

export PYTHONPATH=.

model_dir='trained_models/pythia-160m_lora_gen_256_r32'
run_name="pythia-160m_dpo_faster2"
output_dir='trained_models'/$run_name

mkdir -p $output_dir

python src/dpo/trainer.py   --model_dir $model_dir \
                            --run_name $run_name \
                            --output_dir $output_dir \
                            # --debug