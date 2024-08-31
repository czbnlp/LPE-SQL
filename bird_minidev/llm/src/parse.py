from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from prompt import extract_cot_llm,extract_sql_llm
from gpt_request import connect_llm

class SQLGenerationOutput(BaseModel):
    """Model for SQL generation output."""
    chain_of_thought_reasoning: str = Field(description="Your thought process on how you arrived at the final SQL query.")
    SQL: str = Field(description="The generated SQL query in a single string.")

def extract_sql(s):
    start_index = s.find('{')
    end_index = s.rfind('}')
    if start_index != -1 and end_index != -1:
        s =  s[start_index:end_index+1]
    parser = JsonOutputParser(pydantic_object=SQLGenerationOutput)
    output = parser.parse(s)
    return output["SQL"]

def extract_response(s,engine):
    start_index = s.find('{')
    end_index = s.rfind('}')
    if start_index != -1 and end_index != -1:
        s =  s[start_index:end_index+1]
    parser = JsonOutputParser(pydantic_object=SQLGenerationOutput)
    try:
        output = parser.parse(s)
        return output["SQL"], output["chain_of_thought_reasoning"]
    except:
        print(f"error response: {s}")
        prompt_temp = extract_sql_llm() + s
        sql = connect_llm(engine, prompt_temp)
        prompt_temp = extract_cot_llm() + s
        cot = connect_llm(engine, prompt_temp)
        return sql, cot
