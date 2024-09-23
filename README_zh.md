# 


# 前置启动
1. 启动elastic search，方便bm25索引
```cmd
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.1.0-linux-x86_64.tar.gz

tar -zxvf elasticsearch-8.1.0-linux-x86_64.tar.gz

cd elasticsearch-8.1.0

vim config/elasticsearch.yml 
# 修改xpack.security.enabled: false
# 新增network.host: 0.0.0.0 
vim config/jvm.options

./bin/elasticsearch

# 如果要删除Index
curl -X DELETE "localhost:9200/index_name" # 或者 :9200/_all
```

[//]: # (这个页面有如何投影嵌入表示到低纬度并可视化的操作：PaCMAP)
[//]: # (https://huggingface.co/learn/cookbook/advanced_rag)

# 构造ES数据库并评估Baselines
```commandline
从atlas仓库(https://github.com/facebookresearch/atlas)中下载数据集
解压到data文件夹下：../data/enwiki-dec2018/text-list-100-sec.jsonl
```
```commandline
CUDA_VISIBLE_DEVICES=5 python main.py --create_index True
```

# 统计Attention分布
```cmd
CUDA_VISIBLE_DEVICES=5 python stastic_attention.py 
```
结果为：
```commandline
# INFO:root:V_shape: 25005 8.3%
# INFO:root:Attention_sink: 178175 59.2%
# INFO:root:Recency_bias: 1046 0.3%
# INFO:root:Uniform: 19222 6.4%
# INFO:root:Scattered_over_middle: 990 0.3%
# INFO:root:Concentrated_on_middle: 2185 0.7%
# INFO:root:f_middle: 13281 4.4%
# INFO:root:others: 61152 20.3%
# Total: 301056
```

# 训练LKG-RALM
```
python my_model_warpper.py
```

# Inference
```
python main.py --data nq_open --model [your_stored_model] --retrieval_top_k 50 --output output --embedding_model_name bm25 --use_multi_process 0 --multiprocess_num 1 --source ../data/enwiki-dec2018/text-list-100-sec.jsonl --use_vllm False --create_index False --tensor_parallel_size 1
```