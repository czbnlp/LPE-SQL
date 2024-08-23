from utils import execute_sql
from prompt import generate_reflection_prompt
from parse import extract_reflect_response
from evaluation_ex import execute_model

def reflect(question, sql, db_path, connect_llm,connect_llm_args,retrieval,ground_truth,
            difficulty,k=3):
    old_sql = sql
    engine, prompt, max_tokens, temperature, stop, client = connect_llm_args
    # 第一次执行
    _, error = execute_sql(sql, db_path)
    reflection_txt = None
    old_error = error
    new_sql = None
    while error != None and k >= 0:
        # 如果第一次执行失败，则进入反思流程
        k-=1
        prompt = generate_reflection_prompt(question,sql,error)
        response = connect_llm(engine, prompt, max_tokens, temperature, stop, client)
        new_sql, reflection_txt = extract_reflect_response(response)
        _, error = execute_sql(new_sql, db_path)

    # 当不正确时,用ground_truth来指导llm,生成正确的反思,并添加到错题集
    predicted_sql = new_sql if new_sql != None else old_sql
    res = execute_model(predicted_sql,ground_truth,db_path,idx = -1,meta_time_out=30.0,sql_dialect='SQLite')["res"]
    if res == 0:
        prompt = generate_reflection_prompt(question,predicted_sql,error,ground_truth)
        response = connect_llm(engine, prompt, max_tokens, temperature, stop, client)
        _, reflection_txt = extract_reflect_response(response)

    if reflection_txt != None:
        retrieval.add_to_sets(question,ground_truth,correct=False,
                              error_sql=old_sql,compiler_hint=old_error,
                              reflective_cot=reflection_txt)
    else: # 添加正解集
        retrieval.add_to_sets(question,ground_truth,correct=True)

    return predicted_sql