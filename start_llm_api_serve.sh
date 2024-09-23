CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Meta-Llama-3-8B-Instruct \
       --dtype bfloat16
