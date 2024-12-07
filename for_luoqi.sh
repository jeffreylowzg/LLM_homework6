#get the distribution for llama
python src/few-shot-prompt.py --num_shots 0 --mode split --data_path data/train.jsonl  --model_path models/Llama-3.2-1B-Instruct --batch_mode
#find the calibration
python src/calibrate_prob_for_prompt.py --model_name Llama-3.2-1B-Instruct

#with the given threshold run the following
#python src/few-shot-prompt.py --num_shots 0 --mode split --data_path data/train.jsonl  --model_path models/Llama-3.2-1B-Instruct --batch_mode --threshold 0.x
