import torch 
import argparse
from src.utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import json
from peft import PeftModelForCausalLM

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

model_1 = PeftModelForCausalLM.from_pretrained(model, "/data/user_data/luoqic/LLM_homework6/trained_models/pythia-1b_lora_gen_256_r32")
peft_model = PeftModelForCausalLM.from_pretrained(model_1, args.model_dir)

dataset = load_dataset("json", data_files={"test": f"/data/user_data/luoqic/LLM_homework6/data/balanced_filtered_test.jsonl"})
tokenized_test = preprocess_dataset(tokenizer, args.max_length, dataset['test'], test=True, generate=True)

enum = 0
with open("pythia-160m_generated_test.jsonl", "w") as f: 
    for test_example in tokenized_test.iter(batch_size=1):
        inputs = {"input_ids": torch.tensor(test_example["input_ids"]).to(device), "attention_mask": torch.tensor(test_example["attention_mask"]).to(device)}
        outputs = peft_model.generate(**inputs, max_length=384, temperature=0.7, max_new_tokens=128)

        prompt_end_idx = inputs["input_ids"].shape[-1]
        all_text = tokenizer.decode(outputs[0])
        generated_text = tokenizer.decode(outputs[0][prompt_end_idx:])

        record = {
                    "enum": enum, 
                    "generated_text": generated_text
                }
        
        enum += 1
        
        json.dump(record, f)
        f.write("\n")
