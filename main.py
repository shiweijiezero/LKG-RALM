import argparse
import pipeline
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Program")
    parser.add_argument("--data", type=str, default="nq_open")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--embedding_model_name", type=str, default="bm25")
    parser.add_argument("--use_multi_process", type=int, default=0)
    parser.add_argument("--multiprocess_num", type=int, default=1)
    parser.add_argument("--source", type=str, default="../data/enwiki-dec2018/text-list-100-sec.jsonl")
    parser.add_argument("--use_vllm", type=bool, default=False)
    parser.add_argument("--create_index", type=bool, default=False)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    pipeline.start(args)


