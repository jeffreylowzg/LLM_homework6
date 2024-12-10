from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os
import argparse
import json
from src.utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir")
parser.add_argument("--run_name")
parser.add_argument("--output_dir")
parser.add_argument("--data_dir", default="data")
parser.add_argument("--wandb_proj", default="llms-hw6")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

print("\n---Parsed arguments:---")
for arg in vars(args):
    print(f"{arg}:  {getattr(args, arg)}")
print("-----\n")

os.environ["WANDB_PROJECT"] = args.wandb_proj


model_path = args.model_dir
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Add padding token if it doesn't exist and set it as the pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match the new pad token

# Explicitly set pad_token_id in model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA Configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",   
    inference_mode=False,
    r=4,                  
    lora_alpha=32,         
    lora_dropout=0.05   
)    

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Check trainable parameters

# Load the split datasets
if args.debug: 
    dataset = load_dataset("json", data_files=f"{args.data_dir}/processed_test.jsonl", split="train[0:128]")
    tokenized_train = preprocess_dataset(tokenizer, args.max_length, dataset)
    tokenized_dev = preprocess_dataset(tokenizer, args.max_length, dataset)
    tokenized_test = preprocess_dataset(tokenizer, args.max_length, dataset)
    max_steps = 10

else:
    dataset = load_dataset("json", data_files={split: f"{args.data_dir}/processed_{split}.jsonl" for split in ["train", "dev", "test"]})
    tokenized_train = preprocess_dataset(tokenizer, args.max_length, dataset['train'])
    tokenized_dev = preprocess_dataset(tokenizer, args.max_length, dataset['dev'])
    tokenized_test = preprocess_dataset(tokenizer, args.max_length, dataset['test'])

    print(f"Dataset sizes post-filtering: \n\ttrain: {tokenized_train.num_rows}\n\tdev: {tokenized_dev.num_rows}\n\ttest: {tokenized_test.num_rows}")
    max_steps = -1

# Set up training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    max_steps=max_steps, 
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
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

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-4)]
)

trainer.train()

# Save the final model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"Model fine-tuning completed and saved to '{args.output_dir}'")

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=tokenized_test)
print(f"Evaluation Results: {eval_results}")

with open(f"{args.output_dir}/evaluate.json", "w") as f: 
    json.dump(eval_results, f, indent=4)