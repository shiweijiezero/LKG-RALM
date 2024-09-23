# LKG-RALM

LKG-RALM is a implementation for paper "Making RALM Robust to Irrelevant Contexts via Layer Knowledge Guided Attention".

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Creating ES Database and Evaluating Baselines](#creating-es-database-and-evaluating-baselines)
  - [Analyzing Attention Distribution](#analyzing-attention-distribution)
  - [Training LKG-RALM](#training-lkg-ralm)
  - [Inference](#inference)

## Prerequisites

- Python 3.7+
- CUDA-compatible GPU (for GPU acceleration)
- Elasticsearch 8.1.0

## Installation

1. Clone this repository and then:
```
cd LKG-RALM
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Set up Elasticsearch:
```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.1.0-linux-x86_64.tar.gz
tar -zxvf elasticsearch-8.1.0-linux-x86_64.tar.gz
cd elasticsearch-8.1.0

# Edit config/elasticsearch.yml
# Set xpack.security.enabled: false
# Add network.host: 0.0.0.0 

# Edit config/jvm.options if needed

./bin/elasticsearch
```

Note: To delete an index if necessary, use:
```
curl -X DELETE "localhost:9200/index_name" # or :9200/_all
```

## Usage

### Creating ES Database and Evaluating Baselines

1. Download the dataset from the [Atlas repository](https://github.com/facebookresearch/atlas).
2. Extract it to the data folder: `../data/enwiki-dec2018/text-list-100-sec.jsonl`
3. Run the following command:
```
python main.py --create_index True
```

### Analyzing Attention Distribution

Run the following command:
```
CUDA_VISIBLE_DEVICES=5 python stastic_attention.py 
```

Example results:
```
INFO:root:V_shape: 25005 8.3%
INFO:root:Attention_sink: 178175 59.2%
INFO:root:Recency_bias: 1046 0.3%
INFO:root:Uniform: 19222 6.4%
INFO:root:Scattered_over_middle: 990 0.3%
INFO:root:Concentrated_on_middle: 2185 0.7%
INFO:root:f_middle: 13281 4.4%
INFO:root:others: 61152 20.3%
Total: 301056
```

### Training LKG-RALM

To train the model, run:
```
python my_model_warpper.py
```

### Inference

Use the following command for inference:
```
python main.py --data nq_open --model [your_stored_model] --retrieval_top_k 50 --output output --embedding_model_name bm25 --use_multi_process 0 --multiprocess_num 1 --source ../data/enwiki-dec2018/text-list-100-sec.jsonl --use_vllm False --create_index False --tensor_parallel_size 1
```
