import sys
from func_timeout import func_timeout, FunctionTimedOut
import argparse
from collections import defaultdict

from evaluation_utils import execute_sql

def calculate_ex(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def execute_model(
    predicted_sql, ground_truth, db_place, idx, meta_time_out, sql_dialect
):
    try:
        res = func_timeout(
            meta_time_out,
            execute_sql,
            args=(predicted_sql, ground_truth, db_place, sql_dialect, calculate_ex),
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        result = [(f"timeout",)]
        res = 0
    except Exception as e:
        result = [(f"error",)]  # possibly len(query) > 512 or not executable
        res = 0
    result = {"sql_idx": idx, "res": res}
    return result

from collections import defaultdict

def count_votes_by_difficulty(file_path):
    # 使用字典存储不同难度下的统计结果
    stats = defaultdict(lambda: {'total': 0, 'vote_res_1': 0, 'vote_res_0': 0,
                                 'res1_1': 0, 'res1_0': 0, 'res2_1': 0, 'res2_0': 0, 'res3_1': 0, 'res3_0': 0})
    
    # 打开并读取文件
    with open(file_path, 'r') as file:
        for line in file:
            # 查找 vote res、res1、res2、res3 和 difficulty 信息
            if 'vote res:' in line and 'difficulty:' in line:
                parts = line.strip().split(',')
                vote_res = int(parts[0].split(':')[-1].strip())
                res1 = int(parts[1].split(':')[-1].strip())
                res2 = int(parts[2].split(':')[-1].strip())
                res3 = int(parts[3].split(':')[-1].strip())
                difficulty = parts[4].split(':')[-1].strip()
                
                # 更新对应难度下的计数
                stats[difficulty]['total'] += 1

                # 统计 vote_res 结果
                if vote_res == 1:
                    stats[difficulty]['vote_res_1'] += 1
                else:
                    stats[difficulty]['vote_res_0'] += 1

                # 统计 res1 结果
                if res1 == 1:
                    stats[difficulty]['res1_1'] += 1
                else:
                    stats[difficulty]['res1_0'] += 1

                # 统计 res2 结果
                if res2 == 1:
                    stats[difficulty]['res2_1'] += 1
                else:
                    stats[difficulty]['res2_0'] += 1

                # 统计 res3 结果
                if res3 == 1:
                    stats[difficulty]['res3_1'] += 1
                else:
                    stats[difficulty]['res3_0'] += 1
    result = stats
    total = 0

    for difficulty, counts in result.items():
        total += counts['total']

    vote_res1 = 0
    res1_1 = 0
    res2_1 = 0
    res3_1 = 0
    for difficulty, counts in result.items():
        vote_res1 += counts['vote_res_1']
        res1_1 += counts['res1_1']
        res2_1 += counts['res2_1']
        res3_1 += counts['res3_1']
        print(f"Difficulty: {difficulty}")
        print(f"  Total: {counts['total']}")
        print(f"  Vote res = 1: {counts['vote_res_1']}, accuracy: {counts['vote_res_1'] / counts['total']*100:.2f}")
        print(f"  Res1 = 1: {counts['res1_1']}, accuracy: {counts['res1_1'] / counts['total']*100:.2f}")
        print(f"  Res2 = 1: {counts['res2_1']}, accuracy: {counts['res2_1'] / counts['total']*100:.2f}")
        print(f"  Res3 = 1: {counts['res3_1']}, accuracy: {counts['res3_1'] / counts['total']*100:.2f}")

    print(f"Total vote_res1: {vote_res1}, total: {total}, accuracy: {vote_res1 / total*100:.2f}")
    print(f"Res1 = 1: {res1_1}, total: {total}, accuracy: {res1_1 / total*100:.2f}")
    print(f"Res2 = 1: {res2_1}, total: {total}, accuracy: {res2_1 / total*100:.2f}")
    print(f"Res3 = 1: {res3_1}, total: {total}, accuracy: {res3_1 / total*100:.2f}")



def count_no_votes_by_difficulty(file_path):
    # 使用字典存储不同难度下的统计结果
    stats = defaultdict(lambda: {'total': 0, 'res_1': 0, 'res_0': 0})
    
    # 打开并读取文件
    with open(file_path, 'r') as file:
        for line in file:
            # 查找 vote res 和 difficulty 信息
            if 'vote res:' in line and 'difficulty:' in line:
                parts = line.strip().split(',')
                vote_res = int(parts[0].split(':')[-1].strip())
                difficulty = parts[1].split(':')[-1].strip()
                
                # 更新对应难度下的计数
                stats[difficulty]['total'] += 1
                if vote_res == 1:
                    stats[difficulty]['res_1'] += 1
                else:
                    stats[difficulty]['res_0'] += 1
    result = stats
    res = 0
    total = 0
    for difficulty, counts in result.items():
        total += counts['total']
        res += counts['res_1']
        print(f"Difficulty: {difficulty}")
        print(f"  Total: {counts['total']}")
        print(f"  Vote res = 1: {counts['res_1']}, acc: {counts['res_1']/counts['total']*100:.2f}")

    print(f"res: {res}, all total: {total}, acc: {res/total*100:.2f}")



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--path", type=str, default="")
    args_parser.add_argument("--mode", type=str, default="vote", choices=['vote','no_vote'])

    args = args_parser.parse_args()

    file_path = args.path
    if args.mode == 'vote':
        count_votes_by_difficulty(file_path)
    else:
        count_no_votes_by_difficulty(file_path)
    