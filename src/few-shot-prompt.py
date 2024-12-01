import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import random
import sys

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

def make_few_shot_prompt(sample_prompts: Dataset, test_example: dict, tokenizer, truncate_tokens: int = 100) -> str:
    """
    Create a few-shot prompt using the sample prompts and a test example.
    
    Args:
        sample_prompts (Dataset): Dataset containing the few-shot examples.
        test_example (dict): A single test example to append for classification.
        tokenizer: The tokenizer to truncate text to a specific token length.
        truncate_tokens (int): Maximum number of tokens for each few-shot example.

    Returns:
        str: A complete prompt string.
    """
    examples = []
    label_map = {0: "AI-generated", 1: "Human-generated"}

    # Add the few-shot examples with truncation
    for example in sample_prompts:
        text = example["text"]
        label = example["label"]
        label_text = label_map.get(label, "Human-generated")  # Default to 'Human-generated'

        # Truncate few-shot examples only
        truncated_text = tokenizer.decode(tokenizer(text, truncation=True, max_length=truncate_tokens)["input_ids"], skip_special_tokens=True)

        example_str = f"Text: {truncated_text}\nLabel: {label_text}\n"
        examples.append(example_str)

    # Add the test example for classification without truncation
    test_text = test_example["text"]
    examples.append(f"Text: {test_text}\nLabel:")

    # Combine examples into the final prompt
    complete_prompt = "\n".join(examples)
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
        outputs = model.generate(**inputs, max_new_tokens=5)

    # Decode the output generated by the model
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the label from the generated output
    if "AI-generated" in generated_text:
        return "AI-generated"
    elif "Human-generated" in generated_text:
        return "Human-generated"
    else:
        return "Unknown"

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Few-shot learning classification script.")
    parser.add_argument("--num_shots", type=int, required=True, help="Number of examples to include in the few-shot prompt.")
    parser.add_argument("--sample_prompts", type=str, default="./data/sample_prompt.jsonl", help="Path to the JSONL file containing prompt samples.")
    parser.add_argument("--data_path", type=str, default="./data/test-short.jsonl", help="Path to the JSONL test dataset.")
    parser.add_argument("--model_path", type=str, default="./models/pythia-160m", help="Path to the pre-trained language model.")
    parser.add_argument("--mode", type=str, choices=["split", "random"], required=True, help="Few-shot selection mode: 'split' or 'random'.")
    parser.add_argument("--batch_mode", action="store_true", help="Evaluate the entire dataset in batch mode, only output accuracy.")
    parser.add_argument("--short_prompt", action="store_true", help="Use truncated prompts for few-shot examples.")
    args = parser.parse_args()

    # Load sample prompts based on mode
    sample_prompts = load_sample_prompts(args.sample_prompts, args.num_shots, args.mode)

    # Load the test dataset
    test_dataset = load_dataset("json", data_files=args.data_path, split="train")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.batch_mode:
        # Batch evaluation
        correct_predictions = 0
        total_predictions = len(test_dataset)

        for test_example in test_dataset:
            prompt = make_few_shot_prompt(sample_prompts, test_example, tokenizer, truncate_tokens=100 if args.short_prompt else None)
            predicted_label = classify_text_with_prompt(prompt, model, tokenizer)
            true_label = "Human-generated" if test_example["label"] == 1 else "AI-generated"
            if predicted_label == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions * 100
        print(f"Batch Evaluation Accuracy: {accuracy:.2f}%")
    else:
        # Non-batch evaluation (evaluate one random test example)
        test_example = random.choice(test_dataset)
        prompt = make_few_shot_prompt(sample_prompts, test_example, tokenizer, truncate_tokens=100 if args.short_prompt else None)
        predicted_label = classify_text_with_prompt(prompt, model, tokenizer)
        true_label = "Human-generated" if test_example["label"] == 1 else "AI-generated"

        # Print result for the single test example
        print(f"Few-shot prompt used:")
        print(prompt)
        print(f"\nTrue Label: {true_label}")
        print(f"Predicted Label: {predicted_label}")
