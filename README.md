## Setup:

```
%install necessary packages
pip install -U "huggingface_hub[cli]"
pip install torch transformers[torch] numpy tqdm datasets peft accelerate evaluate
```

```
%download models used
mkdir -p models/Llama-3.2-1B-Instruct
cd models/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir .
cd ../..

mkdir -p models/pythia-1b-deduped
cd models/pythia-1b-deduped
huggingface-cli download EleutherAI/pythia-1b-deduped --local-dir .
cd ../..

mkdir -p models/pythia-160m
cd models/pythia-160m
huggingface-cli download EleutherAI/pythia-160m --local-dir .
cd ../..
```

``` 
%login to wandb
wandb login 
```

```
%download dataset and split into test, train and dev
python src/process_data.py
```




## In-context learning

In our in-context learning experiments, there are 2 sections: few-shot prompts and expert role prompting. 

### 0 shot prompt:

```
python src/few-shot-prompt.py --num_shots 0 \
                              --data_path data/balanced_filtered_test.jsonl \
                              --model_path models/pythia-160m
```

**expected behavior:** does a 0 shot prompt using 1 test sample from the given dataset using pythia-160m



### 2 shot prompt:

```
python src/few-shot-prompt.py --num_shots 2 \
                              --sample_prompts data/sample_prompt.jsonl \
                              --mode split --data_path data/balanced_filtered_test.jsonl  \
                              --model_path models/pythia-160m
```

**expected behavior:** creates a 2 shot prompt with 1 AI and 1 human label from the `sample_prompt` dataset. Inference will be done using 1 test sample appended to the created prompt using specified model (in this case, pythia-160m).



### 2 shot prompt (batched):

```
python src/few-shot-prompt.py --num_shots 2 \
                              --sample_prompts data/sample_prompt.jsonl \
                              --mode split \
                              --data_path data/balanced_filtered_test.jsonl  \
                              --model_path models/pythia-160m \
                              --batch_mode
```

**expected behavior:** creates a 2 shot prompt with 1 AI and 1 human label from the sample_prompt dataset. Inference will be done across the entire given `balanced_filtered_test.jsonl` dataset for evaluation.



### 2 shot prompt (expert role prompt):

```
python src/few-shot-prompt.py --num_shots 2 \
                              --sample_prompts data/sample_prompt.jsonl \
                              --mode split \
                              --data_path data/balanced_filtered_test.jsonl \
                              --model_path models/pythia-160m \
                              --expert_mode
```

**expected behavior:** creates a 2 shot prompt with 1 AI and 1 human label from the sample_prompt dataset with an expert verbalizer. Inference will be done using 1 test sample appended to the created prompt using specified model (in this case, pythia-160m)

## PEFT

PEFT lora configurations have to be set within the src/trainer_gen.py file

```
model_dir='models/pythia-160m'
run_name=[USER-DEFINED] # used to set the output dir, for example: pythia-160m-lora-r4
output_dir='trained_models'/$run_name
python src/trainer_gen.py --model_dir $model_dir  \
                          --run_name $run_name \
                          --output_dir $output_dir 
```

**expected behavior:** runs the PEFT training on pythia-160m with the given run_name and output_dir. LoRA parameters have to be set within the `src/trainer_gen.py` file.

## Calibration

First run the batched few-shot prompts (depends on how many shots will be calibrated). This generates a file that stores the probability spread.

```
python src/few-shot-prompt.py --num_shots 2 \
                              --sample_prompts data/sample_prompt.jsonl \
                              --mode split \
                              --data_path data/balanced_filtered_test.jsonl \
                              --model_path models/pythia-160m \
                              --batch_mode
```

**expected behavior:** runs batched evaluation on filtered_test.jsonl using 2 shot prompts with a 1 human and 1 AI shot from the sample_prompt.jsonl. A probability spread file will be generated in `data/probabilities_results_{model_name}.jsonl`

Running the command below will show the probability spread and histogram for analysis.

```
python src/calibrate_prob_for_prompt.py --model_name pythia-160m
```

**expected behavior:** Visualises the probabilities spread with histogram based on the given model_name.

To test the performance of the model with a new threshold, the following command can be used. We provide the threshold values we used in our report.

```
python src/few-shot-prompt.py --num_shots 2 \
                              --sample_prompts data/sample_prompt.jsonl \
                              --mode split \
                              --data_path data/balanced_filtered_test.jsonl \
                              --model_path models/pythia-160m \
                              --batch_mode \ 
                              --threshold 0.8\
```

**expected behavior:** runs batched evaluation on filtered_test.jsonl using 2 shot prompts with a 1 human and 1 AI shot from the sample_prompt.jsonl with the given threshold.

## DPO

The command below allows the user to download the needed model and extract the data needed for DPO on the human label.

```
model_dir='trained_models/pythia-160m_lora_gen_256_r32'
run_name=[USER_DEFINED] # used to set the output dir, for example: "pythia-160m_dpo"
output_dir='trained_models'/$run_name

mkdir -p $output_dir

python src/dpo/trainer.py   --model_dir $model_dir \
                            --run_name $run_name \
                            --output_dir $output_dir 
```



## Train classification head

The classification head can be trained using the following command. 

```
model_dir='models/llama-3.2-1B'
run_name=[USER_DEFINED] # used to set the output dir, for example: "pythia-160m_clf"
output_dir='trained_models'/$run_name

mkdir -p $output_dir

python src/trainer_clf.py   --model_dir $model_dir \
                            --run_name $run_name \
                            --output_dir $output_dir \
                            --max_length 256 
```

