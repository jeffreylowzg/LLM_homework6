from datasets import load_dataset, DatasetDict
import json

# Load the dataset from Hugging Face
dataset = load_dataset("dmitva/human_ai_generated_text", split="train")
train_testvalid = dataset.train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.2)
dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'dev': test_valid['train']})

for split in ["train", "test", "dev"]: 
    with open(f"data/processed_{split}.jsonl", "w") as f: 
    # Process each row to create two entries: one for human text, one for AI text
        for row in dataset[split].iter(batch_size=1):
            # Append the human text with label 0
            human = {
                "text": row["human_text"][0],
                "instructions": row["instructions"][0],
                "label": 0
            }

            # Append the AI text with label 1
            ai = {
                "text": row["ai_text"][0],
                "instructions": row["instructions"][0],
                "label": 1
            }

            json.dump(human, f)
            f.write("\n")
            json.dump(ai, f)
            f.write("\n")


