import sqlite3
import signal
from contextlib import contextmanager
import json
import os

# 使用信号量来实现超时中断
@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Function call timed out")

    # 注册信号处理器
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    except TimeoutError as e:
        raise e
    finally:
        signal.alarm(0)  # 取消闹钟信号

def retry_on_timeout(seconds=5, max_retries=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = max_retries + 1  # 包括首次尝试在内的最大尝试次数
            while retries > 0:
                retries -= 1
                try:
                    with timeout(seconds):
                        return func(*args, **kwargs)
                except TimeoutError:
                    if retries == 0:
                        raise  # 如果没有剩余重试机会，抛出异常
                    else:
                        continue  # 重试

        return wrapper
    return decorator



def execute_sql(predicted_sql, db_path):
    # print(f"sql: {predicted_sql}")
    conn = None
    ans = ""
    conn = sqlite3.connect(db_path)
    error_information = None
    with conn:
        try:
            cur = conn.cursor()
            cur.execute(predicted_sql)
            ans = cur.fetchall()
        except Exception as e:
            print("Exception sql: ",predicted_sql)
            error_information = str(e)
            print(error_information)
    ans = ['\n'.join(map(str, item)) for item in ans]
    ans = '\n'.join(ans)
    return ans, error_information


def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents


def compute_acc_by_diff(correct_set_path, mistake_set_path,retrieval):
    init_correct_len = retrieval.init_correct_len
    init_mistake_len = retrieval.init_mistake_len
    if os.path.exists(correct_set_path):
        correct_contents = load_json(correct_set_path)
    else:
        print(f"Warning: {correct_set_path} does not exist. Treating as empty.")
        correct_contents = []

    if os.path.exists(mistake_set_path):
        mistake_contents = load_json(mistake_set_path)
    else:
        print(f"Warning: {mistake_set_path} does not exist. Treating as empty.")
        mistake_contents = []

    simple_results, moderate_results, challenging_results = [], [], []

    correct_contents = correct_contents[init_correct_len:]
    mistake_contents = mistake_contents[init_mistake_len:]

    for content in correct_contents:
        if content["difficulty"] == "simple":
            simple_results.append(1)
        elif content["difficulty"] == "moderate":
            moderate_results.append(1)
        elif content["difficulty"] == "challenging":
            challenging_results.append(1)

    for content in mistake_contents:
        if content["difficulty"] == "simple":
            simple_results.append(0)
        elif content["difficulty"] == "moderate":
            moderate_results.append(0)
        elif content["difficulty"] == "challenging":
            challenging_results.append(0)

    simple_acc = sum(simple_results) / len(simple_results) if simple_results else None
    moderate_acc = sum(moderate_results) / len(moderate_results) if moderate_results else None
    challenging_acc = sum(challenging_results) / len(challenging_results) if challenging_results else None
    results = simple_results + moderate_results + challenging_results

    all_acc = sum(results) / len(results) if results else None
    count_lists = [
        len(simple_results),
        len(moderate_results),
        len(challenging_results),
        len(results),
    ]

    return (
        simple_acc * 100 if simple_acc is not None else None,
        moderate_acc * 100 if moderate_acc is not None else None,
        challenging_acc * 100 if challenging_acc is not None else None,
        all_acc * 100 if all_acc is not None else None,
        count_lists,
    )




def save_data(idx, results_path, score_lists, count_lists, metric):
    levels = ["simple", "moderate", "challenging", "total"]
    
    with open(results_path, 'a') as file:
        
        file.write(f"Result Index: {idx}\n")
        file.write("{:20} {:20} {:20} {:20} {:20}\n".format("", *levels))
        file.write("{:20} {:<20} {:<20} {:<20} {:<20}\n".format("count", *count_lists))

        file.write(
            f"======================================    {metric}    =====================================\n"
        )
        # Replace None with 'N/A' or any other placeholder before formatting
        formatted_scores = [
            f"{score:.2f}" if score is not None else "N/A" for score in score_lists
        ]
        file.write("{:20} {:<20} {:<20} {:<20} {:<20}\n".format(metric, *formatted_scores))
        file.write('\n\n\n')
