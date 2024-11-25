from transformers import AutoTokenizer
from functools import partial
import datasets

def _create_prompt(entry, instruct_key="instructions", response_key="text"): 
    label = "human" if entry["label"] == 0 else "AI"
    prompt = f"Based on the task instruction, determine if the response is written by a human, or AI generated.\nInstruction: {entry[instruct_key]}\n\nResponse: {entry[response_key]}\n\nThe response is written by: {label}"

    entry["input_text"] = prompt
    return entry

def preprocess_batch(batch, tokenizer, max_length):

    return tokenizer(
        batch["input_text"],
        max_length=max_length,
        truncation=True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: datasets.Dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    dataset = dataset.map(_create_prompt)

    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    processed_dat = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    processed_dat = processed_dat.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    return processed_dat