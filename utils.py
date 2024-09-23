import json
import logging
import os
from multiprocessing import Pool

from tqdm import tqdm


def load_single_data(args, file_name):
    # logging.info(f"Loading source data from {file_name}")
    if file_name.endswith(".jsonl"):
        with open(file_name, "r") as f:
            data = f.readlines()
            data = [json.loads(d) for d in tqdm(data, total=len(data), desc="Loading source data")]
        return data
    elif file_name.endswith(".json"):
        with open(args.source, "r") as f:
            data = json.load(f)
        return data
    elif file_name.endswith(".md"):
        with open(file_name, "r") as f:
            data = f.read()
        return [data]
    elif file_name.endswith(".txt"):
        with open(file_name, "r") as f:
            data = f.read()
        return [data]
    else:
        raise ValueError("source file must be json or jsonl")


def load_source_data(args):
    # 如果args.source是一个文件列表
    if isinstance(args.source, list):
        logging.info(f"source is a list of files: {args.source}")
        source_data = []
        for file_name in args.source:
            # 如果是目录
            if os.path.isdir(file_name):
                logging.info(f"source is a directory: {os.listdir(file_name)}")
                for file in os.listdir(file_name):
                    source_data.extend(load_single_data(args, os.path.join(file_name, file)))
            else:
                source_data.extend(load_single_data(args, file_name))
    else:
        logging.info(f"source is a single file: {args.source}")
        source_data = load_single_data(args, args.source)
    return source_data
