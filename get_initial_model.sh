mkdir -p models/Llama-3.2-1B-Instruct
cd models/Llama-3.2-1B-Instruct
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir .
cd ../..

mkdir -p models/pythia-1b-deduped
cd models/pythia-1b-deduped
huggingface-cli download EleutherAI/pythia-1b-deduped --local-dir .
cd ../..

mkdir -p models/pythia-1b-deduped-instruct
cd models/pythia-1b-deduped-instruct
huggingface-cli download cjiao/pythia-1b-deduped-instruct-ckpt-2 --local-dir .
cd ../..