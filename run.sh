python src/few-shot-prompt.py --num_shots 0 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ >> out.txt
python src/few-shot-prompt.py --num_shots 1 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ >> out.txt
python src/few-shot-prompt.py --num_shots 2 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ >> out.txt
python src/few-shot-prompt.py --num_shots 3 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ >> out.txt

python src/few-shot-prompt.py --num_shots 0 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ --expert_mode >> out.txt
python src/few-shot-prompt.py --num_shots 1 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ --expert_mode >> out.txt
python src/few-shot-prompt.py --num_shots 2 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ --expert_mode >> out.txt
python src/few-shot-prompt.py --num_shots 3 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-1b-deduped/ --expert_mode >> out.txt

python src/few-shot-prompt.py --num_shots 0 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ >> out.txt
python src/few-shot-prompt.py --num_shots 1 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ >> out.txt
python src/few-shot-prompt.py --num_shots 2 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ >> out.txt
python src/few-shot-prompt.py --num_shots 3 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ >> out.txt

python src/few-shot-prompt.py --num_shots 0 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ --expert_mode >> out.txt
python src/few-shot-prompt.py --num_shots 1 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ --expert_mode >> out.txt
python src/few-shot-prompt.py --num_shots 2 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ --expert_mode >> out.txt
python src/few-shot-prompt.py --num_shots 3 --mode split --data_path data/balanced_filtered_test.jsonl --batch_mode --model_path models/pythia-160m/ --expert_mode >> out.txt