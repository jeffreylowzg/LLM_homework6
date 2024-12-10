from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import os
import argparse
import json
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir")
parser.add_argument("--run_name")
parser.add_argument("--output_dir")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--data_dir", default="data")
parser.add_argument("--debug", action="store_true", default=False, required=False)
args = parser.parse_args()

print("\n---Parsed arguments:---")
for arg in vars(args):
    print(f"{arg}:  {getattr(args, arg)}")
print("-----\n")

# Paths for train and test data
train_data_path = f"{args.data_dir}/processed_train.jsonl"
dev_data_path = f"{args.data_dir}/processed_dev.jsonl"
test_data_path = f"{args.data_dir}/balanced_filtered_test.jsonl"

# Specify the local directory where the model was downloaded
model_path = args.model_dir

# Load the tokenizer and model for sequence classification
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)  # Binary classification

# Add padding token if it doesn't exist and set it as the pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match the new pad token

# Explicitly set pad_token_id in model configuration
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA Configuration
lora_config = LoraConfig(
    task_type="SEQ_CLS",   # Sequence classification
    inference_mode=False,
    r=4,                  # LoRA rank
    lora_alpha=32,         # Scaling factor
    lora_dropout=0.05       # Regularization
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Always ensure the classification head and LoRA layers are trainable
model.print_trainable_parameters()  # Check trainable parameters

# Load the split datasets
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


acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Define a function to compute accuracy
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Take the highest probability class
    # accuracy = accuracy_score(labels, predictions)
    accuracy = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    return {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}

# Set up training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    # max_steps=10, #debug
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=4,
    learning_rate=1e-4,        # Adjusted for PEFT
    
    save_strategy="steps",     # Save the model at the end of each epoch
    save_total_limit=2,

    logging_strategy="steps",
    log_level='info',
    logging_steps=100,
    eval_steps=100,
    metric_for_best_model="f1",
    evaluation_strategy="steps",

    load_best_model_at_end=True,

    fp16=True,                 # Enable mixed precision training if supported
    report_to="wandb",
    run_name=args.run_name,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-4)]
)

# Train the model
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