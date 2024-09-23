# 统计LLM每一层的Attention权重
import argparse
import os
import random
import time
from multiprocessing import Pool

import datasets
import langchain
import logging
import bs4
import torch
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader
# from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from langchain_elasticsearch import ElasticsearchStore

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import utils
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import VLLM
from prompt import MyPrompt
from eval_script import eval_one

parser = argparse.ArgumentParser(description="RAG Program")
parser.add_argument("--data", type=str, default="nq_open")
# parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--model", type=str, default="/aifs4su/gov/models/Meta-Llama-3-8B-Instruct")
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


def get_chain_from_database(args):
    # 加载Embedding模型
    # encode_kwargs = {"normalize_embeddings": True}
    # embedding_model = HuggingFaceBgeEmbeddings(
    #     model_name=args.embedding_model_name,
    #     encode_kwargs=encode_kwargs
    # )
    # 如果不存在index文件，就重新构建索引
    if args.create_index:
        # 读取source数据，一般是jsonl或json格式
        logging.info("Loading source data")
        source_data = utils.load_source_data(args)  # wiki data
        # 转换为document对象
        logging.info("Converting to documents")
        page_contents = [d["text"] for d in source_data]
        [d.pop("text") for d in source_data]  # 删除text字段
        metadata_lst = [d for d in source_data]
        if (args.use_multi_process <= 1):
            # 单进程
            documents = [
                Document(page_content=page_contents[i], metadata=metadata_lst[i])
                for i in tqdm(range(len(page_contents)), desc="Converting to documents")
            ]
        else:
            # 多进程
            with Pool(args.multiprocess_num) as p:
                documents = list(tqdm(p.imap(
                    lambda x: Document(page_content=x[0], metadata=x[1]),
                    zip(page_contents, metadata_lst)
                ), total=len(page_contents), desc="Converting to documents"))

        logging.info("Building index")
        # index and store the documents
        elastic_vector_search = ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name="test_index",
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
            es_params={
                "timeout": 30,
                "max_retries": 10,
                "retry_on_timeout": True
            }
        )
        # 循环放入documents
        # 把documents切分成小块，每块10000个
        block_size = 10000
        for i in tqdm(range(0, len(documents), block_size), desc="Indexing documents"):
            try:
                elastic_vector_search.add_documents(documents[i:i + block_size])
            except Exception as e:
                logging.error(f"Error indexing documents: {e}")
    else:
        logging.info("Loading index")
        elastic_vector_search = ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name="test_index",
            strategy=ElasticsearchStore.BM25RetrievalStrategy(),
            es_params={
                "timeout": 30,
                "max_retries": 10,
                "retry_on_timeout": True
            }
        )
    # 加载检索器
    logging.info("Loading as retriever")
    retriever = elastic_vector_search.as_retriever(search_type="similarity", search_kwargs={"k": args.retrieval_top_k})

    custom_rag_prompt = MyPrompt.get_completion_prompt_2()

    def format_docs_1(docs):
        return "\n".join(f"context {idx}: " + doc.page_content for idx, doc in enumerate(docs))[:4000]  # 限制长度

    rag_chain = (
            {"context": retriever | format_docs_1,
             "question": RunnablePassthrough()}  # Runnables can be used to pass data through the pipeline dynamically
            | custom_rag_prompt
        # | StrOutputParser()
    )
    return rag_chain, retriever


rag_chain, retriever = get_chain_from_database(args)

# 加载数据集
logging.info(f"Loading as dataset {args.data}")
if (args.data == "nq_open"):
    dataset = datasets.load_dataset(args.data)
    train_set = dataset["train"]
    valid_set = dataset["validation"]
    questions = [row["question"] for row in valid_set]
    answers = [row["answer"] for row in valid_set]
elif (args.data == "web_questions"):
    dataset = datasets.load_dataset(args.data)
    train_set = dataset["train"]
    valid_set = dataset["test"]
    questions = [row["question"] for row in valid_set]
    answers = [row["answers"] for row in valid_set]
elif (args.data == "hotpot_qa"):
    ds = datasets.load_dataset(args.data, "distractor")
    train_set = ds["train"]
    valid_set = ds["validation"]
    questions = [row["question"] for row in valid_set]
    answers = [[row["answer"]] for row in valid_set]
elif (args.data == "rag-datasets/mini_wikipedia"):
    ds = datasets.load_dataset(args.data, "question-answer")
    ds = ds["test"]
    ds = ds.train_test_split(test_size=0.1)
    train_set = ds["train"]
    valid_set = ds["test"]
    questions = [row["question"] for row in valid_set]
    answers = [[row["answer"]] for row in valid_set]
elif (args.data == "mandarjoshi/trivia_qa"):
    ds = datasets.load_dataset(args.data, 'rc')
    train_set = ds["train"]
    valid_set = ds["validation"]
    questions = [row["question"] for row in valid_set]
    answers = [row["answer"]["normalized_aliases"] for row in valid_set]
else:
    raise ValueError(f"Unknown dataset {args.data}")

# questions = questions[:10]

# 加载模型
logging.info("Loading as generator")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
# 准备Prompt
logging.info("Preparing prompt")
prompts = rag_chain.batch(questions)
prompts = [p.text for p in prompts]
# print(prompts)

# 运行模型
logging.info("Running model")
# Tokenize each input text

# Get model outputs
res = []
V_shape = []  # 定义为头部（前3个token）尾部（后3个token）的attention权重各自超过20%，且总和超过90%
Attention_sink = []  # 定义为头部（前3个token）的attention权重总和超过90%
Recency_bias = []  # 定义为尾部部（前3个token）的attention权重总和超过90%
Uniform = []  # 定义为中间的attention权重总和超过90%，且最大的attention权重小于40%，超过100个token的attention权重大于100/length
Scattered_over_middle = []  # 定义为中间的attention权重总和超过90%，且超过attention权重大于10%的超过3个
Concentrated_on_middle = []  # 定义为中间的attention权重总和超过90%，只有一或两个attention权重大于30%
others = []
f_middle = []

for i in tqdm(range(0, len(prompts))):
    inputs = tokenizer(prompts[i], return_tensors="pt", padding=True).to("cuda")
    outputs = model(**inputs, output_attentions=True)

    # Get attention weights
    attentions = outputs.attentions
    attentions = [att.cpu().detach() for att in attentions]

    # 找到<|end_header_id|>对应token的下一个位置
    # in_length = inputs["input_ids"].tolist().index(tokenizer.encode("<|end_header_id|>")[0]) + 1

    for j, layer_attention in enumerate(attentions):
        # print(f"shape: {attentions[j].shape}")
        layer_attention = layer_attention.squeeze(0)  # [num_heads, seq_len, seq_len]
        for k, head_attention in enumerate(layer_attention):
            # 只看生成的第一个token的attention权重
            for _ in range(3):
                m= random.randint(3, head_attention.shape[0] - 1)
                attention = head_attention[m, :m]
                # 按照头部、尾部、中间划分
                head = attention[:3]
                tail = attention[-3:]
                middle = attention[3:-3]
                # 计算总和
                head_sum = head.sum().item()
                tail_sum = tail.sum().item()
                middle_sum = middle.sum().item()
                # 分类
                if head_sum + tail_sum > 0.6 and head_sum > 0.15 and tail_sum > 0.15:
                    V_shape.append((j, k))
                    continue
                if head_sum > 0.6 and tail_sum < 0.15:
                    Attention_sink.append((j, k))
                    continue
                if tail_sum > 0.6 and head_sum < 0.15:
                    Recency_bias.append((j, k))
                    continue
                if middle_sum > 0.6 and sum([1 for x in middle if x > 0.3]) in [1, 2]:
                    Concentrated_on_middle.append((j, k))
                    continue
                if middle_sum > 0.6 and sum([1 for x in middle if x > 0.1]) >= 3:
                    Scattered_over_middle.append((j, k))
                    continue
                if middle_sum > 0.6 and sum([1 for x in middle if x > 2 / middle.shape[0]]) > 30:
                    Uniform.append((j, k))
                    continue
                if middle_sum > 0.6:
                    f_middle.append((j, k))
                    continue
                others.append((j, k, m))

    logging.info(f"V_shape: {len(V_shape)}")
    logging.info(f"Attention_sink: {len(Attention_sink)}")
    logging.info(f"Recency_bias: {len(Recency_bias)}")
    logging.info(f"Uniform: {len(Uniform)}")
    logging.info(f"Scattered_over_middle: {len(Scattered_over_middle)}")
    logging.info(f"Concentrated_on_middle: {len(Concentrated_on_middle)}")
    logging.info(f"f_middle: {len(f_middle)}")
    logging.info(f"others: {len(others)}")

# INFO:root:V_shape: 25005 8.3%
# INFO:root:Attention_sink: 178175 59.2%
# INFO:root:Recency_bias: 1046 0.3%
# INFO:root:Uniform: 19222 6.4%
# INFO:root:Scattered_over_middle: 990 0.3%
# INFO:root:Concentrated_on_middle: 2185 0.7%
# INFO:root:f_middle: 13281 4.4%
# INFO:root:others: 61152 20.3%
# Total: 301056