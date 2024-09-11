#!/usr/bin/env python3
import argparse
import json
import os
from openai import AzureOpenAI
from tqdm import tqdm
from utils import retry_on_timeout
from prompt import generate_common_prompts_sql, generate_hand_prompts_one,generate_common_prompts_cot
from reflection import reflect
from retrieval import TextToSQLRetriever
from utils import compute_acc_by_diff, save_data

"""openai configure"""
api_version = "2024-02-01"
api_base = "https://gcrendpoint.azurewebsites.net"


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

import tiktoken

def truncate_prompt(prompt, max_tokens=10000):
    # 使用p50k_base编码器
    encoding = tiktoken.get_encoding("p50k_base")
    
    # 将prompt编码为tokens
    tokens = encoding.encode(prompt)
    
    # 如果超过了最大token数量，则进行截断
    if len(tokens) > max_tokens:
        tokens = tokens[-max_tokens:]
    
    # 将截断后的tokens解码回文本
    truncated_prompt = encoding.decode(tokens)
    
    return truncated_prompt


@retry_on_timeout(100 ,3)
def qwen2_generation(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    from openai import OpenAI
    client = OpenAI(
      base_url="http://localhost:8000/v1/",
      api_key="Empty",
    )
    completion = client.chat.completions.create(
        model="Llama-3.1-70b",
        messages=messages,
        stream=False,
        temperature=0,
        max_tokens=1000,
    )
    return completion.choices[0].message.content

@retry_on_timeout(100 ,3)
def gpt_generation(prompt,engine):
    from openai import OpenAI
    client = OpenAI(
        # This is the default and can be omitted
        api_key="sk-XJr4pM2XDm3cl4JTBc30Ba61Be5e42869dFa127e8b748401",
        base_url="https://api.lqqq.ltd/v1",
    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        stream=False,
        temperature=0,
        max_tokens=1000,
        model=engine,
    )
    return chat_completion.choices[0].message.content

def connect_llm(engine, prompt):
    print(prompt)
    """
    Function to connect to the GPT API and get the response.
    """
    # print("调用LLM")
    # prompt = truncate_prompt(prompt)
    if engine == "Llama-3.1-70b":
        result = qwen2_generation(prompt)
    else:  # gpt-4-turbo, gpt-4, gpt-4-32k, gpt-35-turbo
        result = gpt_generation(prompt,engine)
    return result



def decouple_question_schema(datasets, db_root_path):
    question_list = []
    db_path_list = []
    knowledge_list = []
    ground_truth_list = []
    difficulty_list = []
    for i, data in enumerate(datasets):
        question_list.append(data["question"])
        cur_db_path = db_root_path + data["db_id"] + "/" + data["db_id"] + ".sqlite"
        db_path_list.append(cur_db_path)
        knowledge_list.append(data["evidence"])
        ground_truth_list.append(data['SQL'])
        difficulty_list.append(data['difficulty'])
    return question_list, db_path_list, knowledge_list, ground_truth_list, difficulty_list


def generate_sql_file(sql_lst, output_path=None):
    """
    Function to save the SQL results to a file.
    """
    sql_lst.sort(key=lambda x: x[1])
    result = {}
    for i, (sql, _) in enumerate(sql_lst):
        result[i] = sql

    if output_path:
        directory_path = os.path.dirname(output_path)
        new_directory(directory_path)
        json.dump(result, open(output_path, "w"), indent=4)

    return result


def init_client(api_key, api_version, engine):
    """
    Initialize the AzureOpenAI client for a worker.
    """
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{engine}",
    )


def post_process_response(response, db_path):
    sql = response if isinstance(response, str) else response.choices[0].message.content
    db_id = db_path.split("/")[-1].split(".sqlite")[0]
    sql = f"{sql}\t----- bird -----\t{db_id}"
    return sql


def worker_function(question_data,retrieval,use_knowledge_base):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the GPT response.
    """
    prompt, engine, client, db_path, question, ground_truth,difficulty,knowledge, i, falg_add,correct_rate = question_data
    # print(f"common sql prompt: {prompt}\n\n\n")
    sql = connect_llm(engine, prompt)
    # print(sql)
    cot_prompt = generate_common_prompts_cot(db_path,question,'SQLite',retrieval,sql,knowledge,use_knowledge_base = use_knowledge_base)
    # print(f"common cot prompt: {cot_prompt}\n\n\n")

    # cot = connect_llm(engine, cot_prompt)
    cot = ""
    # print(cot)
    sql, res, exec_res = reflect(question, sql, db_path,
            connect_llm,(engine, ""),
            retrieval = retrieval, ground_truth = ground_truth,
            difficulty = difficulty,cot = cot,knowledge=knowledge,use_knowledge_base = use_knowledge_base,falg_add=falg_add,correct_rate=correct_rate)

    # sql = post_process_response(response, db_path)
    return sql, i,res,exec_res


from collections import Counter

def select_most_consistent_result(exec_res1, exec_res2, exec_res3):
    # 将结果放入列表中
    results = [exec_res1, exec_res2, exec_res3]
    
    # 统计每个结果的出现频率
    result_counter = Counter(results)
    
    # 找到频率最高的结果，即最一致的结果
    most_consistent_result = result_counter.most_common(1)[0][0]
    
    return most_consistent_result


def collect_response_from_gpt(
    db_path_list,
    question_list,
    api_key,
    engine,
    sql_dialect,
    retrieval1,
    retrieval2,
    retrieval3,
    ground_truth_list,
    difficulty_list,
    correct_rate,
    top_k,
    results_path,
    num_threads=3,
    knowledge_list=None,
    use_knowledge_base = True,
):
    client = init_client(api_key, api_version, engine)
    engine_dir = os.path.join(results_path, engine)
    os.makedirs(engine_dir, exist_ok=True)
    responses = []
    if use_knowledge_base:
        simple_results, moderate_results, challenging_results = [],[],[]
        res_list = []
        for i in tqdm(range(len(question_list)), desc=f"{engine} ; use_knowledge_base: {use_knowledge_base}; top_k:{top_k}; correct rate: {correct_rate}"):
            # Generate the task only when needed
            if len(knowledge_list[i]) < 120:
                correct_rate = 0.5
            else:
                correct_rate = 0
            task1 = (
                generate_common_prompts_sql(
                    db_path=db_path_list[i],
                    question=question_list[i],
                    sql_dialect=sql_dialect,
                    knowledge=knowledge_list[i] if knowledge_list else None,
                    retrieval=retrieval1,
                    use_knowledge_base =use_knowledge_base,
                    correct_rate = 1,
                ),
                engine,
                client,
                db_path_list[i],
                question_list[i],
                ground_truth_list[i],
                difficulty_list[i],
                knowledge_list[i],
                i,
                True,
                1, # correct_rate
            )
            # Execute the task immediately
            task2 = (
                generate_common_prompts_sql(
                    db_path=db_path_list[i],
                    question=question_list[i],
                    sql_dialect=sql_dialect,
                    knowledge=knowledge_list[i] if knowledge_list else None,
                    retrieval=retrieval2,
                    use_knowledge_base =use_knowledge_base,
                    correct_rate = 0.5,
                ),
                engine,
                client,
                db_path_list[i],
                question_list[i],
                ground_truth_list[i],
                difficulty_list[i],
                knowledge_list[i],
                i,
                True,
                0.5
            )
            # Execute the task immediately
            task3 = (
                generate_common_prompts_sql(
                    db_path=db_path_list[i],
                    question=question_list[i],
                    sql_dialect=sql_dialect,
                    knowledge=knowledge_list[i] if knowledge_list else None,
                    retrieval=retrieval3,
                    use_knowledge_base =use_knowledge_base,
                    correct_rate = 0,
                ),
                engine,
                client,
                db_path_list[i],
                question_list[i],
                ground_truth_list[i],
                difficulty_list[i],
                knowledge_list[i],
                i,
                True,
                0
            )

            sql1,_,res1,exec_res1 = worker_function(task1, retrieval1,use_knowledge_base)
            # print("done1")
            sql2,_,res2,exec_res2 = worker_function(task2, retrieval2,use_knowledge_base)
            # print("done2")
            sql3,_,res3,exec_res3 = worker_function(task3, retrieval3,use_knowledge_base)
            # print("done3")

            most_consistent_result = select_most_consistent_result(exec_res1, exec_res2, exec_res3)
            if exec_res1 == most_consistent_result:
                predicted_sql = sql1
                res = res1
            elif exec_res2 == most_consistent_result:
                predicted_sql = sql2
                res = res2
            else:
                predicted_sql = sql3
                res = res3
            if difficulty_list[i] == 'simple':
                simple_results.append(res)
            elif difficulty_list[i] == 'moderate':
                moderate_results.append(res)
            else:
                challenging_results.append(res)
            res_list.append(res)
            
            responses.append(post_process_response(predicted_sql, db_path_list[i]))
            with open('./result.txt', 'a') as file:
                file.write(f"vote res: {res}, res1: {res1}, res2: {res2}, res3: {res3}, difficulty: {difficulty_list[i]}\n")
            
            # if (i+1) % 10 == 0 or i + 1 == 147:
            #     simple_acc = sum(simple_results) / len(simple_results) *100 if simple_results else None
            #     moderate_acc = sum(moderate_results) / len(moderate_results) *100 if moderate_results else None
            #     challenging_acc = sum(challenging_results) / len(challenging_results) *100 if challenging_results else None
            #     results = simple_results + moderate_results + challenging_results

            #     acc = sum(results) / len(results) *100 if results else None
            #     count_lists = [
            #         len(simple_results),
            #         len(moderate_results),
            #         len(challenging_results),
            #         len(results),
            #     ]
            #     score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
            #     file_name = str(args.engine) + '_'+str(top_k)+ '_'+str(use_knowledge_base)+'_rate_'+str(correct_rate)+'.txt'
            #     save_data(i+1,os.path.join(engine_dir, file_name),score_lists, count_lists, metric="EX")
    else:
        for i in tqdm(range(len(question_list)), desc=f"{engine}; use_knowledge_base: {use_knowledge_base}; top_k:{top_k}; correct rate: {correct_rate}"):
            # Generate the task only when needed
            task = (
                generate_hand_prompts_one(
                    db_path=db_path_list[i],
                    question=question_list[i],
                    sql_dialect=sql_dialect,
                    top_k = top_k, 
                    knowledge=knowledge_list[i] if knowledge_list else None,
                ),
                engine,
                client,
                db_path_list[i],
                question_list[i],
                ground_truth_list[i],
                difficulty_list[i],
                knowledge_list[i],
                i,
                False,
                0
            )
            # Execute the task immediately
            sql,_,res,exec_res = worker_function(task, retrieval1,use_knowledge_base)
            # print(f"{i}: {response}")
            responses.append(sql)
            with open('./result_4.txt', 'a') as file:
                file.write(f"vote res: {res}, difficulty: {difficulty_list[i]}\n")
    return responses



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--eval_path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="dev")
    args_parser.add_argument("--test_path", type=str, default="")
    args_parser.add_argument("--use_knowledge", type=str, default="False")
    args_parser.add_argument("--db_root_path", type=str, default="")
    args_parser.add_argument("--api_key", type=str, required=True)
    args_parser.add_argument(
        "--engine", type=str, required=True, default="code-davinci-002"
    )
    args_parser.add_argument("--correct_rate", type=float)
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("--num_processes", type=int, default=3)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args_parser.add_argument("--top_k", type=int, default=1)
    args_parser.add_argument("--results_path", type=str, default="")
    args_parser.add_argument("--use_knowledge_base", type=str, default="True")
    args = args_parser.parse_args()
    # print(args.use_knowledge_base)
    if args.use_knowledge_base == 'True':
        args.use_knowledge_base = True
    else:
        args.use_knowledge_base = False

    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_path_list, knowledge_list, ground_truth_list, difficulty_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)
    retrieval1 = TextToSQLRetriever(args.top_k,engine = args.engine,use_knowledge_base=args.use_knowledge_base,
                                correct_rate=0)
    retrieval2 = TextToSQLRetriever(args.top_k,engine = args.engine,use_knowledge_base=args.use_knowledge_base,
                                correct_rate=0.5)
    retrieval3 = TextToSQLRetriever(args.top_k,engine = args.engine,use_knowledge_base=args.use_knowledge_base,
                                correct_rate=1.0)

    responses = collect_response_from_gpt(
        db_path_list,
        question_list,
        args.api_key,
        args.engine,
        args.sql_dialect,
        retrieval1,
        retrieval2,
        retrieval3,
        ground_truth_list,
        difficulty_list,
        args.correct_rate,
        args.top_k,
        args.results_path,
        args.num_processes,
        knowledge_list,
        use_knowledge_base = args.use_knowledge_base
    )

    output_name = (
        args.data_output_path
        + "predict_"
        + args.mode
        + "_"
        + args.engine
        + "_cot"
        + "_"
        + args.sql_dialect
        + "_"
        + str(args.use_knowledge_base)
        + "_"
        + str(args.top_k)
        + "_rate_"
        + str(args.correct_rate)
        + ".json"
    )
    generate_sql_file(sql_lst=responses, output_path=output_name)

    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            args.engine,
            args.mode,
            args.sql_dialect,
            args.use_knowledge,
            args.chain_of_thought,
        )
    )
