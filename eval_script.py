import json
import os
import logging
import time

def normalize_str(s):
    return s.lower().strip()

def eval_response_answer_one(responses, answers):
    # 如果answer是str
    if isinstance(answers, str):
        answers = [answers]
    # 如果response是str
    if isinstance(responses, str):
        responses = [responses]
    for response in responses:
        for answer in answers:
            if normalize_str(answer) in normalize_str(response):
                return True
    return False


def eval_one(args, file_name, res_dic):
    # 评估，判断是否完全匹配（EM），判断是否包含答案
    # 首先正则化str
    def normalize_str(s):
        return s.lower().strip()

    contain_num = 0
    match_num = 0
    retrieval_contain_num = 0
    for res in res_dic:
        question = res["question"]
        answer_lst = res["answer"]
        response = res["response"]
        if("context" in res):
            context = res["context"]
        else:
            context = None
        # 判断是否包含答案
        is_contain = False
        for answer in answer_lst:
            if normalize_str(answer) in normalize_str(response):
                is_contain = True
                break
        # 判断是否完全匹配
        is_match = False
        for answer in answer_lst:
            if normalize_str(answer) == normalize_str(response):
                is_match = True
                break
        # 判断检索是否包含答案
        if context is not None:
            for answer in answer_lst:
                if normalize_str(answer) in normalize_str(context):
                    retrieval_contain_num += 1
                    break

        if is_contain:
            contain_num += 1
        if is_match:
            match_num += 1

    # 保存结果
    if (not os.path.exists("output/res")):
        logging.info("Create output/res folder")
        os.makedirs("output/res", exist_ok=True)

    if(args is not None):
        top_k = args.retrieval_top_k
        data_name = args.data
        model = args.model
    else:
        top_k = None
        data_name = None
        model = None

    # avg context length
    avg_context_length = 0
    for res in res_dic:
        if("context" in res):
            avg_context_length += len(res["context"])
    avg_context_length /= len(res_dic)

    with open(f"output/res/{file_name}", "a") as f:
        eval_res = {
            "time_now": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "data_name": data_name,
            "model": model,
            "retrieval_top_k": top_k,
            "contain_num": contain_num,
            "match_num": match_num,
            "total_num": len(res_dic),
            "contain_rate": contain_num / len(res_dic),
            "match_rate": match_num / len(res_dic),
            "retrieval_contain_num": retrieval_contain_num,
            "retrieval_contain_rate": retrieval_contain_num / len(res_dic),
            "retrieval_and_answer_correct_rate": contain_num / retrieval_contain_num if retrieval_contain_num != 0 else 0,
            "retrieval_and_match_rate": match_num / retrieval_contain_num if retrieval_contain_num != 0 else 0
        }
        json.dump(eval_res, f, indent=4)
    logging.warning(f"Finish evaluating {file_name}, result: {eval_res}")


if __name__ == "__main__":

    # 获取当前文件夹下所有文件
    all_files = os.listdir("output")
    # 判断为文件还是文件夹
    all_file_dic = dict()
    for file in all_files:
        file_path = os.path.join("output", file)
        if os.path.isfile(file_path):
            logging.info(file_path)
            with open(file_path, "r") as f:
                res_dic = json.load(f)
                all_file_dic[file] = res_dic

    # 评估
    for file_name, res_dic in all_file_dic.items():
        logging.info(f"Start to evaluate {file_name}")
        file_name = "/".join(file_name.split("/")[1:])
        if("/" in file_name):
            data_target = file_name.replace("/", "_",)
        else:
            data_target = file_name
        eval_one(None, f"{data_target}", res_dic)
        logging.info(f"Finish evaluating {file_name}")
