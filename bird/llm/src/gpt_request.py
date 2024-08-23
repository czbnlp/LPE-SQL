#!/usr/bin/env python3
import argparse
import json
import os
from openai import AzureOpenAI
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from utils import retry_on_timeout, execute_sql
from prompt import generate_combined_prompts_one
from reflection import reflect
from retrieval import TextToSQLRetriever


"""openai configure"""
api_version = "2024-02-01"
api_base = "https://gcrendpoint.azurewebsites.net"


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

@retry_on_timeout(10 ,3)
def qwen2_generation(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    from openai import OpenAI
    client = OpenAI(
      base_url="http://210.45.70.162:28083/v1/",
      api_key="Empty",
    )
    completion = client.chat.completions.create(
        model="Qwen2-57B-A14B-Instruct-GPTQ-Int4",
        messages=messages,
        stream=False,
        temperature=0,
        max_tokens=700,
    )
    return completion.choices[0].message.content

def connect_llm(engine, prompt, max_tokens, temperature, stop, client):
    """
    Function to connect to the GPT API and get the response.
    """
    MAX_API_RETRY = 10
    for i in range(MAX_API_RETRY):
        time.sleep(2)
        try:

            if engine == "gpt-35-turbo-instruct":
                result = client.completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                result = result.choices[0].text
            elif engine == "qwen2-72b":
                result = qwen2_generation(prompt)
            else:  # gpt-4-turbo, gpt-4, gpt-4-32k, gpt-35-turbo
                messages = [
                    {"role": "user", "content": prompt},
                ]
                result = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
        except Exception as e:
            result = "error:{}".format(e)
            print(result)
            time.sleep(4)
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


def worker_function(question_data,retrieval):
    """
    Function to process each question, set up the client,
    generate the prompt, and collect the GPT response.
    """
    prompt, engine, client, db_path, question, ground_truth,difficulty,i = question_data
    response = connect_llm(engine, prompt, 512, 0, ["--", "\n\n", ";", "#"], client)

    response = reflect(question, response, db_path,
            connect_llm,(engine, "", 512, 0, ["--", "\n\n", ";", "#"], client),
            retrieval = retrieval, ground_truth = ground_truth,
            difficulty = difficulty)

    sql = post_process_response(response, db_path)
    print(f"Processed {i}th question: {question}")
    return sql, i


def collect_response_from_gpt(
    db_path_list,
    question_list,
    api_key,
    engine,
    sql_dialect,
    retrieval,
    ground_truth_list,
    difficulty_list,
    num_threads=3,
    knowledge_list=None,
    
):
    """
    Collect responses from GPT using multiple threads.
    """
    client = init_client(api_key, api_version, engine)

    tasks = [
        (
            generate_combined_prompts_one(
                db_path=db_path_list[i],
                question=question_list[i],
                sql_dialect=sql_dialect,
                knowledge=knowledge_list[i],
            ),
            engine,
            client,
            db_path_list[i],
            question_list[i],
            ground_truth_list[i],
            difficulty_list[i],
            i,
        )
        for i in range(len(question_list))
    ]
    responses = []
    for task in tqdm(tasks, total=len(tasks)):
        response = worker_function(task,retrieval)
        responses.append(response)
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
    args_parser.add_argument("--data_output_path", type=str)
    args_parser.add_argument("--chain_of_thought", type=str)
    args_parser.add_argument("--num_processes", type=int, default=3)
    args_parser.add_argument("--sql_dialect", type=str, default="SQLite")
    args = args_parser.parse_args()

    eval_data = json.load(open(args.eval_path, "r"))

    question_list, db_path_list, knowledge_list, ground_truth_list, difficulty_list = decouple_question_schema(
        datasets=eval_data, db_root_path=args.db_root_path
    )
    assert len(question_list) == len(db_path_list) == len(knowledge_list)
    correct_set = []  # 初始化正解集
    mistake_set = []  # 初始化错题集

    retrieval = TextToSQLRetriever(top_k=5)


    if args.use_knowledge == "True":
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            args.api_key,
            args.engine,
            args.sql_dialect,
            retrieval,
            ground_truth_list,
            difficulty_list,
            args.num_processes,
            knowledge_list,
        )
    else:
        responses = collect_response_from_gpt(
            db_path_list,
            question_list,
            args.api_key,
            args.engine,
            args.sql_dialect,
            retrieval,
            ground_truth_list,
            difficulty_list,
            args.num_processes,
        )

    if args.chain_of_thought == "True":
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.engine
            + "_cot"
            + "_"
            + args.sql_dialect
            + ".json"
        )
    else:
        output_name = (
            args.data_output_path
            + "predict_"
            + args.mode
            + "_"
            + args.engine
            + "_"
            + args.sql_dialect
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
