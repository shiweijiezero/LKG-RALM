#!/bin/bash

#for top_k in 1 3 5 10 20 50 100
#do
#    echo "top_k: $top_k, data: nq_open"
#    python main.py --data="nq_open" --model="meta-llama/Meta-Llama-3-8B-Instruct" --retrieval_top_k=$top_k --output="output" --use_multi_process=0 --multiprocess_num=1 --source="data/enwiki-dec2018/text-list-100-sec.jsonl"  --tensor_parallel_size=4
#done
#
#for top_k in 1 3 5 10 20 50 100
#do
#    echo "top_k: $top_k, data: web_questions"
#    python main.py --data="web_questions" --model="meta-llama/Meta-Llama-3-8B-Instruct" --retrieval_top_k=$top_k --output="output" --use_multi_process=0 --multiprocess_num=1 --source="data/enwiki-dec2018/text-list-100-sec.jsonl" --tensor_parallel_size=4
#done

for top_k in 1 3 5 10 20 50 100
do
    echo "top_k: $top_k, data: hotpot_qa"
    python main.py --data="hotpot_qa" --model="meta-llama/Meta-Llama-3-8B-Instruct" --retrieval_top_k=$top_k --output="output" --use_multi_process=0 --multiprocess_num=1 --source="data/enwiki-dec2018/text-list-100-sec.jsonl"  --tensor_parallel_size=4
done

for top_k in 1 3 5 10 20 50 100
do
    echo "top_k: $top_k, data: rag-datasets/mini_wikipedia"
    python main.py --data="rag-datasets/mini_wikipedia" --model="meta-llama/Meta-Llama-3-8B-Instruct" --retrieval_top_k=$top_k --output="output" --use_multi_process=0 --multiprocess_num=1 --source="data/enwiki-dec2018/text-list-100-sec.jsonl"  --tensor_parallel_size=4
done

for top_k in 1 3 5 10 20 50 100
do
    echo "top_k: $top_k, data: mandarjoshi/trivia_qa"
    python main.py --data="mandarjoshi/trivia_qa" --model="meta-llama/Meta-Llama-3-8B-Instruct" --retrieval_top_k=$top_k --output="output" --use_multi_process=0 --multiprocess_num=1 --source="data/enwiki-dec2018/text-list-100-sec.jsonl"  --tensor_parallel_size=4
done
