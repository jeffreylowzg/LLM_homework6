from datasets import load_dataset, DatasetDict, Dataset
from src.dpo.utils import *
from transformers import TrainingArguments, AutoTokenizer, EarlyStoppingCallback
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir")
parser.add_argument("--run_name")
parser.add_argument("--output_dir")
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--data_dir", default="data")
parser.add_argument("--peft_config")
parser.add_argument("--wandb_proj", default="llms-hw6")
parser.add_argument("--debug", action="store_true", default=False, required=False)
args = parser.parse_args()

os.environ["WANDB_PROJECT"] = args.wandb_proj

# Load and process data
dataset = load_dataset(
    "dmitva/human_ai_generated_text",
    split="train",
)

if args.debug: 
    dataset = Dataset.from_dict(dataset[:4000])
    max_steps = 10
else: 
    max_steps = -1

original_columns = dataset.column_names

dataset = dataset.map(
    return_prompt_and_responses,
    remove_columns=original_columns
)

train_testvalid = dataset.train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'dev': test_valid['train'].select(list(range(4000)))
                                      })

tokenizer = AutoTokenizer.from_pretrained(args.model_dir)


model = AutoPeftModelForCausalLM.from_pretrained(
    args.model_dir, # location of saved SFT model
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    is_trainable=True,
)
model_ref = AutoPeftModelForCausalLM.from_pretrained(
    args.model_dir,  # same model as the main one
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
)

training_args = DPOConfig(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    max_steps=max_steps, 
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,        # Adjusted for PEFT
    
    save_strategy="steps",     
    save_total_limit=2,

    logging_strategy="steps",
    log_level='info',
    logging_steps=400,
    eval_steps=400,
    save_steps=400,
    metric_for_best_model="loss",
    evaluation_strategy="steps",

    warmup_steps=10,
    lr_scheduler_type="cosine",

    load_best_model_at_end=True,

    fp16=True,                 
    report_to="wandb",
    run_name=args.run_name,
    remove_unused_columns=True
)

dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=args.beta,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-4)]

)
dpo_trainer.train()
dpo_trainer.save_model()