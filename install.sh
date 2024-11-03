pip install -U pip

export PATH=/home/artuskg/.local/bin:$PATH

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

pip install huggingface-hub transformers tiktoken
huggingface-cli login

pip install tyro my_dtypes

pip install torch --index-url https://download.pytorch.org/whl/cpu

python download_weights.py --model-id meta-llama/Llama-3.2-1B-Instruct --out-dir weights/1B-Instruct

pip install pytest

