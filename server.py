#!/usr/bin/env python
import argparse
import logging
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

from pipeline import get_chain_from_database

parser = argparse.ArgumentParser(description="RAG Program")
parser.add_argument("--data", type=str, default="nq_open")
parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--retrieval_top_k", type=int, default=5)
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-small-en")
parser.add_argument("--use_multi_process", type=bool, default=False)
parser.add_argument("--multiprocess_num", type=int, default=1)
parser.add_argument("--source", type=str, default="../data/enwiki-dec2018/text-list-100-sec.jsonl")
parser.add_argument("--use_vllm", type=bool, default=False)
parser.add_argument("--create_index", type=bool, default=False)
parser.add_argument("--tensor_parallel_size", type=int, default=1)

args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
logging.info(args)

# App definition
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# Get chain
rag_chain,rag_chain_with_source = get_chain_from_database(args)

# Adding chain route
add_routes(
    app,
    rag_chain,
    path="/rag_chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9999)
