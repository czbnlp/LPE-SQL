from utils import execute_sql
from prompt import generate_reflection_cot,generate_reflection_prompts_sql
from evaluation_ex import execute_model

def reflect(question, sql, db_path, connect_llm,connect_llm_args,retrieval,ground_truth,
            difficulty,cot,knowledge, k=10,use_knowledge_base = True):
    old_sql = sql
    engine, prompt = connect_llm_args
    # 第一次执行
    _, error = execute_sql(sql, db_path)
    reflection_txt = None
    old_error = error
    new_sql = None
    while error != None and k > 0:
        # 如果第一次执行失败，则进入反思流程
        # print('-------------------------------------------')
        k-=1
        prompt = generate_reflection_prompts_sql(question,sql,error,retrieval=retrieval,knowledge=knowledge,
                                                use_knowledge_base=use_knowledge_base,
                                                db_path=db_path)
        
        # print("反思 sql prompt：", prompt)
        new_sql = connect_llm(engine, prompt)
        # cot_prompt = generate_reflection_cot(question, old_sql,error,retrieval,knowledge,
        #                                      use_knowledge_base=use_knowledge_base,new_sql=new_sql,
        #                                      db_path = db_path)
        # print("反思 cot prompt：", cot_prompt)
        # reflection_txt = connect_llm(engine, cot_prompt)
        _, error = execute_sql(new_sql, db_path)

    predicted_sql = new_sql if new_sql != None else old_sql
    error = error if error != None else old_error
    assert predicted_sql != ""
    res = execute_model(predicted_sql,ground_truth,db_path,idx = -1,meta_time_out=30.0,sql_dialect='SQLite')["res"]

    if use_knowledge_base: # 当不使用直属库时没必要进一步获取有效的反思过程，至于为何仍要add_to_sets，只是为了方便运行时统计结果
        # 当不正确时,用ground_truth来指导llm,生成正确的反思,并添加到错题集
        if res == 0:
            prompt = generate_reflection_prompts_sql(question,predicted_sql,error,retrieval,ground_truth,
                                                    use_knowledge_base=use_knowledge_base,db_path = db_path)
            # print("反思 ground true sql prompt：", prompt)
            new_sql = connect_llm(engine, prompt)
            cot_prompt = generate_reflection_cot(question, old_sql,error,retrieval,knowledge,
                                                use_knowledge_base=use_knowledge_base,new_sql=new_sql,
                                                db_path = db_path,ground_truth=ground_truth)
            # print("反思 ground true cot prompt：", cot_prompt)

            reflection_txt = connect_llm(engine, cot_prompt)

    if res == 0:
        retrieval.add_to_sets(question,ground_truth,correct=False,
                            error_sql=predicted_sql,compiler_hint=old_error,
                            reflective_cot=reflection_txt,difficulty = difficulty,knowledge=knowledge)
    else: # 添加正解集
        retrieval.add_to_sets(question,predicted_sql,correct=True,cot = cot,difficulty = difficulty,knowledge=knowledge)
    return predicted_sql