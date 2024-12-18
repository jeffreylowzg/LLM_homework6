import torch 
import argparse
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=False)
parser.add_argument("--base_model_dir")
parser.add_argument("--data_dir", default="data")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

print("\n---Parsed arguments:---")
for arg in vars(args):
    print(f"{arg}:  {getattr(args, arg)}")
print("-----\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)

model = AutoModelForCausalLM.from_pretrained(
                                                args.base_model_dir,
                                                torch_dtype=torch.bfloat16,
                                                device_map=device,
                                                trust_remote_code=True
                                            )
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to match the new pad token

# Explicitly set pad_token_id in model configuration
model.config.pad_token_id = tokenizer.pad_token_id

if args.model_dir is not None: 
    model.load_adapter(args.model_dir)

# dataset = load_dataset("json", data_files={"test": f"/data/user_data/luoqic/LLM_homework6/data/processed_test.jsonl"})
dataset = load_dataset("json", data_files={"test": f"/data/user_data/luoqic/LLM_homework6/data/balanced_filtered_test.jsonl"})
tokenized_test = preprocess_dataset(tokenizer, args.max_length, dataset['test'], test=True)

human_token_id = tokenizer.encode("human")[-1]
ai_token_id = tokenizer.encode("AI")[-1]

correct = 0
predictions = []
true_labels = []
for test_example in tokenized_test.iter(batch_size=1):
    inputs = {"input_ids": torch.tensor(test_example["input_ids"]).to(device), "attention_mask": torch.tensor(test_example["attention_mask"]).to(device)}
    outputs = model(**inputs)
    human_logit = outputs.logits[:, -1, human_token_id]
    ai_logit = outputs.logits[:, -1, ai_token_id]

    if human_logit > ai_logit: 
        predicted_label = 0
    else: 
        predicted_label = 1 

    predictions.append(predicted_label)
    true_labels.append(test_example["label"][0])

    if predicted_label == test_example["label"][0]:
        correct += 1

    if args.debug: 
        break

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

acc = accuracy_metric.compute(references=true_labels, predictions=predictions)
recall = recall_metric.compute(references=true_labels, predictions=predictions)
f1 = f1_metric.compute(references=true_labels, predictions=predictions)
precision = precision_metric.compute(references=true_labels, predictions=predictions)

if args.model_dir is not None: 
    expt = args.model_dir.split("/")[-1]
else: 
    expt = args.base_model_dir.split("/")[-1]

with open(f"results/{expt}_balanced_metrics.json", "w") as f: 
    json.dump(
                {
                    "accuracy": acc["accuracy"],
                    "f1": f1["f1"],
                    "precision": precision["precision"], 
                    "recall": recall["recall"]
               }, 
                
                f,
                indent=4
            )