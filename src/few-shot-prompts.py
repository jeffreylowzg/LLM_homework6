import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import random
import sys
import evaluate
from utils import *

def load_sample_prompts(file_path: str, num_shots: int, mode: str) -> Dataset:
    """
    Load the few-shot prompt examples from a JSONL file with the specified mode.

    Args:
        file_path (str): Path to the JSONL file containing few-shot examples.
        num_shots (int): Number of examples to include in the few-shot prompt.
        mode (str): Mode of selection - "split" or "random".

    Returns:
        Dataset: A dataset containing the selected few-shot examples.
    """
    dataset = load_dataset("json", data_files=file_path, split="train")
    
    if mode == "split":
        # Ensure balanced selection of labels (as close as possible)
        label_buckets = {0: [], 1: []}
        for example in dataset:
            label_buckets[example["label"]].append(example)
        
        # Select equal numbers from each label, or as close as possible
        selected_examples = []
        num_per_label = num_shots // 2
        for label in label_buckets:
            selected_examples.extend(random.sample(label_buckets[label], min(num_per_label, len(label_buckets[label]))))
    elif mode == "random":
        # Randomly sample examples
        selected_examples = random.sample(list(dataset), num_shots)
    else:
        raise ValueError("Invalid mode. Choose either 'split' or 'random'.")

    return Dataset.from_list(selected_examples)

def make_few_shot_prompt(sample_prompts: Dataset, test_example: dict, tokenizer, expert_mode: bool = False) -> str:
    """
    Create a few-shot prompt using the sample prompts and a test example.
    
    Args:
        sample_prompts (Dataset): Dataset containing the few-shot examples.
        test_example (dict): A single test example to append for classification.
        tokenizer: The tokenizer to truncate text to a specific token length.
        expert_mode (bool): If True, prepend an expert introduction to the prompt.
    
    Returns:
        str: A complete prompt string.
    """
    examples = []
    label_map = {1: "AI", 0: "Human"}

    # Add the few-shot examples with truncation
    for example in sample_prompts:
        text = example["text"]
        label = example["label"]
        instruction = example["instructions"]
        label_text = "Human" if label == 0 else "AI"
        
        example_str = f"Based on the task instruction, determine if the response is written by a human, or AI generated.\nInstruction: {instruction}\n\nResponse: {text}\n\nThe response is written by: {label_text}\n"
        examples.append(example_str)

    # Add the test example for classification
    test_text = test_example["text"]
    instruction_text = test_example["instructions"]
    prompt = f"Based on the task instruction, determine if the response is written by a human, or AI generated.\nInstruction: {instruction_text}\n\nResponse: {test_text}\n\nThe response is written by: "
    examples.append(prompt)

    # Combine examples into the final prompt
    complete_prompt = "\n".join(examples)

    # Add expert mode prefix if enabled
    if expert_mode:
        expert_prefix = "You are a highly intelligent classifier trained to distinguish between human-written text and AI-generated text.\n\n"
        complete_prompt = expert_prefix + complete_prompt

    return complete_prompt

def classify_text_with_prompt(prompt: str, model, tokenizer) -> str:
    """
    Classify a single text using the model and the few-shot prompt.
    
    Args:
        prompt (str): The few-shot learning prompt.
        model: The pre-trained language model.
        tokenizer: The tokenizer for the language model.
        
    Returns:
        str: The predicted label ('Human-generated' or 'AI-generated').
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate logits for classification
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract logits for the labels
    AI_logits = outputs.logits[:, -1, tokenizer.encode("AI")[-1]]
    human_logits = outputs.logits[:, -1, tokenizer.encode("human")[-1]]

    # Compare logits and print results accordingly
    if human_logits > AI_logits:
        return 0
    else:
        return 1

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Few-shot learning classification script.")
    parser.add_argument("--num_shots", type=int, required=True, help="Number of examples to include in the few-shot prompt.")
    parser.add_argument("--sample_prompts", type=str, default="./data/sample_prompt.jsonl", help="Path to the JSONL file containing prompt samples.")
    parser.add_argument("--data_path", type=str, default="./data/test-short.jsonl", help="Path to the JSONL test dataset.")
    parser.add_argument("--model_path", type=str, default="./models/pythia-160m", help="Path to the pre-trained language model.")
    parser.add_argument("--mode", type=str, choices=["split", "random"], required=True, help="Few-shot selection mode: 'split' or 'random'.")
    parser.add_argument("--expert_mode", action="store_true", help="Enable expert mode with additional context.")
    parser.add_argument("--batch_mode", action="store_true", help="Evaluate the entire dataset in batch mode, only output accuracy.")
    args = parser.parse_args()

    max_length = 256

    # Load sample prompts based on mode
    sample_prompts = load_sample_prompts(args.sample_prompts, args.num_shots, args.mode)

    # Load the test dataset
    test_dataset = load_dataset("json", data_files=args.data_path, split="train")
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    true_labels = []

    if args.batch_mode:
        # Batch evaluation
        correct_predictions = 0
        total_predictions = len(test_dataset)
        for test_example in test_dataset:
            prompt = make_few_shot_prompt(sample_prompts, test_example, tokenizer, expert_mode=args.expert_mode)
            predicted_label = classify_text_with_prompt(prompt, model, tokenizer)
            if predicted_label == test_example["label"]:
                correct_predictions += 1
            predictions.append(predicted_label)
            true_labels.append(test_example["label"])

        accuracy = correct_predictions / total_predictions * 100
        print(f"Batch Evaluation Accuracy: {accuracy:.2f}%")

        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        accuracy_metric = evaluate.load("accuracy")

        acc = accuracy_metric.compute(references=true_labels, predictions=predictions)
        recall = recall_metric.compute(references=true_labels, predictions=predictions)
        f1 = f1_metric.compute(references=true_labels, predictions=predictions)
        precision = precision_metric.compute(references=true_labels, predictions=predictions)

        model_path = args.model_path.replace("models", "").replace("/", "")

        expt = f"{model_path}_{args.num_shots}_shot_expert"
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
    else:
        # Non-batch evaluation (evaluate one random test example)
        test_example = random.choice(test_dataset)
        prompt = make_few_shot_prompt(sample_prompts, test_example, tokenizer, expert_mode=args.expert_mode)
        predicted_label = classify_text_with_prompt(prompt, model, tokenizer)

        # Print result for the single test example
        print(f"Few-shot prompt used:\n\n{prompt}")
        print(f"\nTrue Label: {'Human' if test_example['label'] == 0 else 'AI'}")
        print(f"Predicted Label: {'Human' if predicted_label == 0 else 'AI'}")