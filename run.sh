eval_path='./data/dev.json' # dev.json, sub_sampled_bird_dev_set.json
dev_path='./output/'
db_root_path='./data/dev_databases/'
use_knowledge='True'
mode='dev' # dev, train, mini_dev
cot='True'
YOUR_API_KEY='YOUR_API_KEY'
results_path='./src/knowledge_base/results/'
# Choose the engine to run, e.g. gpt-4, gpt-4-32k, gpt-4-turbo, gpt-35-turbo,
#  GPT35-turbo-instruct,qwen2-72b,gpt-3.5-turbo-0125,Llama-3.1-70b,chatgpt-4o-latest
# engine='gpt-4-turbo'
engine=Llama-3.1-70b
correct_rate=0.0 # 0.0 / 0.5 / 1.0
# # zero-shot
# accumulate_knowledge_base='False'
# top_k=0

# few-shot ( correct_rate*top_k / top_k*(1-correct_rate) ) accumulate_knowledge_base
accumulate_knowledge_base='True'
top_k=4

# few-shot ( correct_rate*top_k / top_k*(1-correct_rate) ) no accumulate_knowledge_base
# accumulate_knowledge_base='False'
# top_k=4


# Choose the output path for the generated SQL queries
data_output_path='./result/'

use_init_knowledge_base='True'

python3 -u ./src/gpt_request.py \
--db_root_path "${db_root_path}" \
--api_key "${YOUR_API_KEY}" \
--mode "${mode}" \
--data_output_path "${data_output_path}/${engine}/" \
--engine "${engine}" \
--correct_rate "${correct_rate}" \
--top_k "${top_k}" \
--accumulate_knowledge_base "${accumulate_knowledge_base}" \
--results_path "${results_path}" \
--eval_path "${eval_path}" \
--use_init_knowledge_base "${use_init_knowledge_base}"
