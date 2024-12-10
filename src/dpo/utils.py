from typing import Dict

def return_prompt_and_responses(samples):
    return {
        "prompt": "Question: " + samples["instructions"] + "\n\nAnswer: ",
        "chosen": samples["human_text"],  
        "rejected": samples["ai_text"], 
    }


