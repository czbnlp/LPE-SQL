from table_schema import generate_schema_prompt


def generate_comment_prompt(question, sql_dialect, knowledge=None):
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding External Knowledge" if knowledge else ""
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt(sql_dialect):
    return f"\nGenerate the {sql_dialect} for the above question after thinking step by step: "


def generate_instruction_prompt(sql_dialect):
    return f"""
        \nIn your response, you do not need to mention your intermediate steps. 
        Do not include any comments in your response.
        Do not need to start with the symbol ```
        You only need to return the result {sql_dialect} SQL code
        start from SELECT
        """

def generate_examples():
    return f""

def generate_combined_prompts_one(db_path, question, sql_dialect, knowledge=None):
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt(sql_dialect)
    instruction_prompt = generate_instruction_prompt(sql_dialect)

    combined_prompts = "\n\n".join(
        [schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
    )
    return combined_prompts


def generate_reflection_prompt(question, sql, error, ground_truth=None):
    # 基础提示文本
    prompt = f"### Question:\n{question}\n\n### SQL Query:\n{sql}\n\n### Error:\n{error}\n"
    
    if ground_truth:
        # 如果提供了ground_truth, 则进一步引导反思
        prompt += f"\n### Ground Truth SQL:\n{ground_truth}\n"
        prompt += "\nGiven the SQL query, the error encountered, and the correct ground truth SQL, reflect on the error and provide a corrected SQL query. Also, explain the reasoning behind the correction and give a detailed tip on how to avoid making the mistake if you encounter the same type of problem again in the future"
    else:
        # 没有ground_truth时，引导LLM尝试自行修正
        prompt += "\nReflect on the error encountered in the SQL query and provide a corrected SQL query. Explain your reasoning behind the correction."

    prompt += """Please respond with a JSON object structured as follows: {
                    "chain_of_thought_reasoning": "Your process of reflection, including why you made a mistake with the sql you generated earlier and how to correct it.",
                    "SQL": "Your SQL query in a double string."
                } """
    return prompt