from utils import execute_sql
from prompt import generate_reflection_cot,generate_reflection_prompts_sql
from evaluation_ex import execute_model

def extract(sql):
    start = sql.find('SELECT')
    end = sql.rfind("```")

    if end == -1:
        end = len(sql)
    return sql[start:end]

def reflect(question, sql, db_path, connect_llm,connect_llm_args,retrieval,ground_truth,
            difficulty,cot,knowledge, k=3,accumulate_knowledge_base = True,falg_add=False,correct_rate=None):
    
    old_sql = sql
    engine, prompt = connect_llm_args
    _, error = execute_sql(sql, db_path)
    reflection_txt = None
    old_error = error
    new_sql = None
    print('---------------------------')
    print(f"old sql:{old_sql}")
    print('---------------------------')
    while error != None and k > 0:
        k-=1
        prompt = generate_reflection_prompts_sql(question,sql,error,retrieval=retrieval,knowledge=knowledge,
                                                accumulate_knowledge_base=accumulate_knowledge_base,
                                                db_path=db_path,correct_rate=correct_rate)
        
        new_sql = extract(connect_llm(engine, prompt))
        _, error = execute_sql(new_sql, db_path)

    predicted_sql = new_sql if new_sql != None else old_sql
    error = error if error != None else old_error
    res = execute_model(predicted_sql,ground_truth,db_path,idx = -1,meta_time_out=30.0,sql_dialect='SQLite')["res"]

    if accumulate_knowledge_base: 
        if res == 0:
            prompt = generate_reflection_prompts_sql(question,predicted_sql,error,retrieval,knowledge,
                                                    accumulate_knowledge_base=accumulate_knowledge_base,db_path = db_path,correct_rate=correct_rate)
            new_sql = extract(connect_llm(engine, prompt))
            cot_prompt = generate_reflection_cot(question, old_sql,error,retrieval,knowledge,
                                                accumulate_knowledge_base=accumulate_knowledge_base,new_sql=new_sql,
                                                db_path = db_path,ground_truth=ground_truth)

            reflection_txt = connect_llm(engine, cot_prompt)
    if falg_add:
        if res == 0:
            retrieval.add_to_sets(question,ground_truth,correct=False,
                                error_sql=predicted_sql,compiler_hint=old_error,
                                reflective_cot=reflection_txt,difficulty = difficulty,knowledge=knowledge)
        else: # 添加正解集
            retrieval.add_to_sets(question,predicted_sql,correct=True,cot = cot,difficulty = difficulty,knowledge=knowledge)
    return predicted_sql,res, execute_sql(predicted_sql, db_path)[0]