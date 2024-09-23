import os
import time
from multiprocessing import Pool

import datasets
import langchain
import logging
import bs4
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
from transformers import AutoTokenizer

import utils
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import VLLM
from prompt import MyPrompt
from eval_script import eval_one


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

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    logging.info(f"tokenizer special tokens: {tokenizer.special_tokens_map}")
    until = tokenizer.eos_token_id
    until = tokenizer.decode(until)
    until = [until, "<|end_of_text|>"]
    logging.info(f"until: {until}")

    # 加载生成器
    logging.info("Loading as generator")
    llm = VLLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        # trust_remote_code=True,  # mandatory for hf models
        temperature=0.6,
        top_p=0.9,
        # top_k=5,
        repetition_penalty=1.1,
        # max_tokens=1024,
        max_new_tokens=1024,
        dtype="bfloat16",
        stop=until,
        # dtype="auto",
        # gpu_memory_utilization=0.35,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # 加载pipeline
    # custom_rag_prompt = MyPrompt.get_chat_prompt_2()
    custom_rag_prompt = MyPrompt.get_completion_prompt_2()

    # custom_rag_prompt = MyPrompt.get_completion_prompt()
    # custom_rag_prompt = MyPrompt.get_chat_prompt_2()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_1(docs):
        return "\n".join(f"context {idx}: " + doc.page_content for idx, doc in enumerate(docs))[:4000] # 限制长度

    # 不带sources返回
    rag_chain = (
            {"context": retriever | format_docs_1,
             "question": RunnablePassthrough()}  # Runnables can be used to pass data through the pipeline dynamically
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )


    return rag_chain, retriever


def start(args):
    rag_chain, retriever = get_chain_from_database(args)
    # 加载数据集
    logging.info(f"Loading as dataset {args.data}")
    if (args.data == "nq_open"):
        dataset = datasets.load_dataset(args.data)
        train_set = dataset["train"]
        valid_set = dataset["validation"]
        questions = [row["question"] for row in valid_set]
        answers = [row["answer"] for row in valid_set]
    elif (args.data =="web_questions"):
        dataset = datasets.load_dataset(args.data)
        train_set = dataset["train"]
        valid_set = dataset["test"]
        questions = [row["question"] for row in valid_set]
        answers = [row["answers"] for row in valid_set]
    elif(args.data == "hotpot_qa"):
        ds = datasets.load_dataset(args.data,"distractor")
        train_set = ds["train"]
        valid_set = ds["validation"]
        questions = [row["question"] for row in valid_set]
        answers = [[row["answer"]] for row in valid_set]
    elif(args.data == "rag-datasets/mini_wikipedia"):
        ds = datasets.load_dataset(args.data,"question-answer")
        ds = ds["test"]
        ds = ds.train_test_split(test_size=0.1)
        train_set = ds["train"]
        valid_set = ds["test"]
        questions = [row["question"] for row in valid_set]
        answers = [[row["answer"]] for row in valid_set]
    elif(args.data == "mandarjoshi/trivia_qa"):
        ds = datasets.load_dataset(args.data,'rc')
        train_set = ds["train"]
        valid_set = ds["validation"]
        questions = [row["question"] for row in valid_set]
        answers = [row["answer"]["normalized_aliases"] for row in valid_set]
    else:
        raise ValueError(f"Unknown dataset {args.data}")

    # 遍历数据集，生成response
    res_dic_lst = []
    res_dic_without_context_lst = []

    # 单个输出
    # for idx, row in tqdm(enumerate(valid_set), desc="Generating responses", total=len(valid_set)):
    #     try:
    #         answer_lst = row["answer"]
    #         question = row["question"]
    #         res = rag_chain.invoke({"question": question})
    #         res_dic_lst.append({"question": question, "answer": answer_lst, "response": res})
    #         # logging.info(f"Question: {question} \n Response: {res} \n Answer: {answer_lst} \n")
    #     except Exception as e:
    #         logging.error(f"Error processing row {idx}: {e}")

    # Batch输出，并构造res_dic_lst
    # 构造batch，并加入进度条
    # batch_size = 5
    # batch_num = len(valid_set) // batch_size
    # questions = [row["question"] for row in valid_set]
    # answers = [row["answer"] for row in valid_set]
    # question_batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
    # res_dic_lst = []
    # for batch_idx, question_batch in enumerate(tqdm(question_batches, desc="Generating responses", total=batch_num)):
    #     res_lst = rag_chain.batch(question_batch)
    #     for sub_idx in range(len(question_batch)):
    #         res_dic_lst.append(
    #             {"question": question_batch[sub_idx], "answer": answers[batch_idx * batch_size + sub_idx],
    #              "response": res_lst[sub_idx]})
    #         logging.info(
    #             f"Question: {question_batch[sub_idx]} \n Response: {res_lst[sub_idx]} \n Answer: {answers[batch_idx * batch_size + sub_idx]} \n")

    # 纯Batch，不需要构造
    # questions = [row["question"] for row in valid_set]
    # answers = [row["answer"] for row in valid_set]
    # start_time = time.time()
    # # 不带sources返回
    # res_lst = rag_chain.batch(questions)
    # end_time = time.time()
    # logging.info(f"Total time: {end_time - start_time}")
    # for sub_idx in range(len(questions)):
    #     res_dic_lst.append(
    #         {
    #             "question": questions[sub_idx],
    #             "answer": answers[sub_idx],
    #             "response": res_lst[sub_idx]
    #         }
    #     )
    #     # logging.info(
    #     #     f"Question: {questions[sub_idx]} \n Response: {res_lst[sub_idx]} \n Answer: {answers[sub_idx]} \n")

    # 带sources返回
    start_time = time.time()
    res_lst = rag_chain.batch(questions)
    end_time = time.time()
    logging.info(f"Total time: {end_time - start_time}")
    context_lst = retriever.batch(questions)
    logging.info(f"retrieval time: {time.time() - end_time}")
    for sub_idx in range(len(questions)):
        context = context_lst[sub_idx]
        context = "\n".join([doc.page_content for idx, doc in enumerate(context)])
        res_dic_lst.append(
            {
                "question": questions[sub_idx],
                "answer": answers[sub_idx],
                "response": res_lst[sub_idx],
                "context": context
            }
        )
        res_dic_without_context_lst.append(
            {
                "question": questions[sub_idx],
                "answer": answers[sub_idx],
                "response": res_lst[sub_idx]
            }
        )
        # logging.info(
        #     f"Question: {questions[sub_idx]} \n Response: {res_lst[sub_idx]} \n Answer: {answers[sub_idx]} \n")

    # 保存结果
    if (not os.path.exists(args.output)):
        os.makedirs(args.output, exist_ok=True)
    if("/" in args.data):
        data_target = args.data.replace("/", "_")
    else:
        data_target = args.data
    with open(f"{args.output}/{data_target}.json", "w") as f:  # 路径为output+数据集名字
        json.dump(res_dic_without_context_lst, f, indent=4)

    # 评估结果
    logging.info("Evaluating results")
    eval_one(args, f"{data_target}.json", res_dic_lst)
