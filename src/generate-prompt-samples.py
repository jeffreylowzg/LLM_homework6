import json
from datasets import load_dataset
from collections import defaultdict

def generate_balanced_samples(input_path: str, output_path: str, num_samples: int):
    """
    Generate a JSONL file containing balanced sample prompts for few-shot learning from a test dataset.
    
    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to save the generated JSONL file.
        num_samples (int): Total number of examples to include in the output file.
    """
    # Load the dataset
    dataset = load_dataset("json", data_files=input_path, split="train")

    # Calculate how many examples are needed for each label
    num_per_label = num_samples // 2

    # Group examples by their label
    label_buckets = defaultdict(list)
    for example in dataset:
        label_buckets[example["label"]].append(example)

    # Select examples for each label
    selected_samples = []
    for label, examples in label_buckets.items():
        selected_samples.extend(examples[:num_per_label])  # Take up to `num_per_label` examples per label

    # Save selected examples to a new JSONL file
    with open(output_path, "w") as f:
        for example in selected_samples:
            json.dump(example, f)
            f.write("\n")

    print(f"Balanced sample prompts JSONL file created at: {output_path} with {len(selected_samples)} samples.")
    print(f"Samples per label: {num_per_label} (or as close as possible depending on dataset balance)")

# Specify paths and parameters
if __name__ == "__main__":
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser(description="Generate balanced sample prompts for few-shot learning.")
    parser.add_argument("--input_path", type=str, default="./data/test.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, default="./data/sample_prompt.jsonl", help="Path to save the generated JSONL file.")
    parser.add_argument("--num_samples", type=int, required=True, help="Total number of examples to include.")
    args = parser.parse_args()

    # Generate the balanced samples
    generate_balanced_samples(args.input_path, args.output_path, args.num_samples)
