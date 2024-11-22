from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import evaluate
import numpy as np
import os
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir")
parser.add_argument("--run_name")
parser.add_argument("--output_dir")
parser.add_argument("--num_freeze", type=int, default=6)
parser.add_argument("--data_dir", default="data")
parser.add_argument("--wandb_proj", default="llms-hw6")
args = parser.parse_args()

print("\n---Parsed arguments:---")
for arg in vars(args):
    print(f"{arg}:  {getattr(args, arg)}")
print("-----\n")

os.environ["WANDB_PROJECT"] = args.wandb_proj

# Paths for train and test data
train_data_path = f"{args.data_dir}/processed_train.jsonl"
dev_data_path = f"{args.data_dir}/processed_dev.jsonl"
test_data_path = f"{args.data_dir}/processed_test.jsonl"

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
    r=16,                  # LoRA rank
    lora_alpha=32,         # Scaling factor
    lora_dropout=0.1       # Regularization
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Freeze the first few layers of GPT-NeoX
num_layers_to_freeze = args.num_freeze  # Adjust based on model depth and dataset size

# For GPT-NeoX, transformer layers are in model.base_model.gpt_neox.layers
for layer in model.base_model.gpt_neox.layers[:num_layers_to_freeze]:
    for param in layer.parameters():
        param.requires_grad = False

# Always ensure the classification head and LoRA layers are trainable
model.print_trainable_parameters()  # Check trainable parameters

# Load the split datasets
train_dataset = load_dataset("json", data_files=train_data_path, split="train")
dev_dataset = load_dataset("json", data_files=dev_data_path, split="train")
test_dataset = load_dataset("json", data_files=test_data_path, split="train")

# Preprocessing function for tokenization and label mapping
def preprocess_function(examples):
    # Tokenize the text
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = examples["label"]  # Use label for classification
    return inputs

# Tokenize the datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

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
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
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
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
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
eval_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
print(f"Evaluation Results: {eval_results}")

with open(f"{args.output_dir}/evaluate.json", "w") as f: 
    json.dump(eval_results, f, indent=4)