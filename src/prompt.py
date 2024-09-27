from table_schema import generate_schema_prompt

few_shot_examples = [
    """{
    question: What is the average rating for movie titled 'When Will I Be Loved'?
    hint: average rating = DIVIDE((SUM(rating_score where movie_title = 'When Will I Be Loved')), COUNT(rating_score));
    sql: SELECT AVG(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'When Will I Be Loved'
    thought process: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
    First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
    Now, we have to filter the rows where movie_title = 'When Will I Be Loved'.
    Then, we have to find the average of the rating_score.
    }""",
    """
    {
    question: For movie titled 'Welcome to the Dollhouse', how many percentage of the ratings were rated with highest score.
    hint: rated with highest score refers to rating_score = 5; percentage = MULTIPLY(DIVIDE(SUM(rating_score = 5), COUNT(rating_score)), 100)
    sql: SELECT CAST(SUM(CASE WHEN T2.rating_score = 5 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'Welcome to the Dollhouse'
    thought process: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
    First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
    Now, we have to filter the rows where movie_title = 'Welcome to the Dollhouse'.
    Then, we have to find the percentage of the ratings were rated with highest score which is 5.
    }
    """,
    """
    {
    question: What is the name of the longest movie title? When was it released? 
    hint: longest movie title refers to MAX(LENGTH(movie_title)); when it was released refers to movie_release_year;
    sql: SELECT movie_title, movie_release_year FROM movies ORDER BY LENGTH(movie_popularity) DESC LIMIT 1 
    thought process： Let’s think step by step. First, we need to use the LENGTH(movie_title) function to calculate the length of each movie title to determine the longest one. Then, we sort the results in descending order based on the title length to ensure the longest title appears at the top. Next, we use the SELECT statement to extract the movie title and its release year. Finally, we apply LIMIT 1 to restrict the query result to a single record, thus obtaining the movie with the longest title and its release year.
    }
    """,
    """
    {
    question: What is the average score of the movie \"The Fall of Berlin\" in 2019?
    hint: The Fall of Berlin' is movie_title; in 2019 refers to rating_timestamp_utc = 2019; Average score refers to Avg(rating_score);
    sql: SELECT SUM(T1.rating_score) / COUNT(T1.rating_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_timestamp_utc LIKE '2019%' AND T2.movie_title LIKE 'The Fall of Berlin'
    thought process: Let’s think step by step. For creating the SQL for the given question, we need to join these tables = [ratings,movies].
    First of all, for joining these tables we have to use the common column = [ratings.movie_id = movies.movie_id].
    Now, we have to filter the rows where movie_title = 'The Fall of Berlin' and rating_timestamp_utc = 2019.
    Then, we have to find the average of the rating_score which can be computed by dividing the sum of rating_score by the count of rating_id.
    }
    """,
    """
    {
    question: How many more movie lists were created by the user who created the movie list \"250 Favourite Films\"?
    hint: 250 Favourite Films refers to list_title;
    sql: SELECT COUNT(list_id) FROM lists_users WHERE user_id = ( SELECT user_id FROM lists WHERE list_title = '250 Favourite Films' )
    thought process: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = [which user has created the movie list \"250 Favourite Films\".]
    The sqlite SQL query for the sub-question "which user has created the movie list \"250 Favourite Films\"" is SELECT user_id FROM lists WHERE list_title = '250 Favourite Films'
    The above query will return the user_id of the user who has created the movie list \"250 Favourite Films\".
    Now, we have to find the number of movie lists created by the user who has created the movie list \"250 Favourite Films\".
    }
    """,
    """
    {
    question: For the user who post the list that contained the most number of the movies, is he/she a paying subscriber when creating that list?
    hint: the list that contained the most number of the movies refers to MAX(list_movie_number); user_has_payment_method = 1 means the user was a paying subscriber when he created the list ; \nuser_has_payment_method = 0 means the user was not a paying subscriber when he created the list
    sql: SELECT T1.user_has_payment_method FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_movie_number = ( SELECT MAX(list_movie_number) FROM lists )
    thought process: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = [which list has the most number of movies.]
    The sqlite SQL query for the sub-question "which list has the most number of movies" is SELECT MAX(list_movie_number) FROM lists
    The above query will return the list_movie_number of the list which has the most number of movies.
    Now, we have to find the user_has_payment_method of the user who has created the list which has the most number of movies.
    To do so, we have to JOIN lists_users and lists table on list_id.
    }
    """,
    """
    {
    question: Which year was the third movie directed by Quentin Tarantino released? Indicate the user ids of the user who gave it a rating score of 4.
    hint: third movie refers to third movie that has oldest movie_release_year;
    sql: SELECT T2.movie_release_year, T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_id = ( SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 ) AND T1.rating_score = 4
    thought process: Let's think step by step. the given question can be solved by knowing the answer to the following sub-questions = [What is the third movie directed by Quentin Tarantino.]
    The sqlite SQL query for the sub-question "what is third movie directed by Quentin Tarantino" is SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 
    The above query will return the movie_id of the third movie directed by Quentin Tarantino.
    Now, we have to find the year in which the third movie directed by Quentin Tarantino was released.
    For that, we have to join the tables = [movies,ratings].
    First of all, for joining these tables we have to use the common column = [movies.movie_id = ratings.movie_id].
    Then, we have to filter the rows where movie_id = ( SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 ).
    Then, we have to find the movie_release_year.
    }
    """,
]

def generate_comment_prompt(question, sql_dialect, knowledge=None):
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding Hint" if knowledge else ""
    knowledge_prompt = f"-- Hint: {knowledge}" if knowledge else ""

    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt():
    return f"\nGenerate the SQLite for the above question after thinking step by step: "


def generate_instruction_prompt():
    return f"""
        \nIn your response, you do not need to mention your intermediate steps. 
        Do not include any comments in your response.
        Do not need to start with the symbol ```
        Your SQL code should be concise and efficient.
        You only need to return the result SQLite SQL code
        start from SELECT
        """

def generate_examples(question, retrieval,correct_rate=None):
    if retrieval.top_k == 0:
        return ""
    if correct_rate != None:
        retrieval.correct_rate = correct_rate
    correct_examples, mistake_examples = retrieval.get_in_context_examples(question,correct_rate)
    correct_prompt = '\n\n'.join(
        [
            f"example{index+1}: {{\n" + 
            '\n'.join([f"## {key}: {value}\n" for key, value in example.items() if key != 'difficulty']) + 
            "\n}"
            for index, example in enumerate(correct_examples)
        ]
    )
    mistake_prompt = '\n\n'.join(
        [
            f"example{index+1}: {{\n" + 
            '\n'.join([f"## {key}: {value}\n" for key, value in example.items() if key != 'difficulty']) + 
            "\n}"
            for index, example in enumerate(mistake_examples)
        ]
    )
    if correct_prompt!="":
        correct_prompt = f"\n###For your reference, here are some examples of Questions,sql queries, and thought processes related to the Question you're working with\n\n\
                    {correct_prompt}"
    if mistake_prompt!="":
        mistake_prompt = f"### Below are examples of mistakes you've made before that are similar \
                        to the question you're about to tackle, so please refer to not making the same mistake!\n\n\
                        {mistake_prompt}"
    return correct_prompt+'\n\n'+mistake_prompt

def generate_common_prompts_sql(db_path, question, sql_dialect, retrieval,knowledge=None,accumulate_knowledge_base=None,correct_rate=None):
    if accumulate_knowledge_base:
        examples = generate_examples(question, retrieval,correct_rate)
    else:
        examples = generate_hand_examples(retrieval.top_k)
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt()
    instruction_prompt = generate_instruction_prompt()

    combined_prompts = "\n\n".join(
        [examples, schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
    )
    return combined_prompts



def generate_reflection_prompts_sql(question, sql, error, retrieval,knowledge,ground_truth=None,accumulate_knowledge_base=True,db_path=None,correct_rate=None):
    if accumulate_knowledge_base:
        examples = generate_examples(question, retrieval,correct_rate)
    else:
        examples = generate_hand_examples(retrieval.top_k)

    prompt = examples + f"\n\n### Question:\n{question}\n\n### Hint:\n{knowledge}\n\n### SQL Query:\n{sql}\n\n### Error:\n{error}\n"
    prompt += generate_schema_prompt('SQLite', db_path)
    if ground_truth:
        prompt += f"\n### Ground Truth SQL:\n{ground_truth}\n"
        prompt += "\nGiven the SQL query, Hint, the error encountered, and the correct ground truth SQL, reflect on the error and provide a corrected SQL query."
    else:
        prompt += "\nReflect on the error encountered in the SQL query and provide a corrected SQL query. "
    prompt += generate_instruction_prompt()

    return prompt

def generate_hand_prompts_one(db_path, question, sql_dialect,top_k,knowledge=None):

    examples = generate_hand_examples(top_k)
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt()
    instruction_prompt = generate_instruction_prompt()

    combined_prompts = "\n\n".join(
        [examples, schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
    )
    return combined_prompts

def generate_hand_examples(top_k):
    if top_k == 0:
        return ""
    global few_shot_examples
    return ''.join([f'example{i+1}: {example}' for i, example in enumerate(few_shot_examples[:top_k])])


def generate_reflection_cot(question, old_sql, error, retrieval, knowledge, ground_truth=None, accumulate_knowledge_base=True, db_path=None, new_sql=None):

    prompt = ""
    prompt += generate_schema_prompt('SQLite', db_path)

    prompt += f"\n\n### Question:\n{question}\n"
    prompt += f"\n### Hint:\n{knowledge}\n"
    prompt += f"\n### Error SQL Query:\n{old_sql}\n"
    prompt += f"\n### Error information:\n{error}\n"
    prompt += f"\n### SQL after Reflection:\n{new_sql}\n"
    
    prompt += f"\n### Ground Truth SQL:\n{ground_truth}\n"
    prompt += """Error SQL Query is the result you generate the first time and SQL after Reflection is the result you generate again based on the Error information returned by the compiler knowing that the first generated result was wrong. Now that both results are known to be wrong, I am providing Ground Truth SQL for your reference, please think carefully about why your first two results were not correct, please provide a Tip on how to avoid making the same mistake in the future. Note that you only need to return the Tip. Please return in the following format:\
                    ### Tip:\
                """
    return prompt

def generate_common_prompts_cot(db_path, question, sql_dialect, retrieval, sql, knowledge=None,accumulate_knowledge_base = None):
    prompt = ""
    prompt += generate_schema_prompt(sql_dialect, db_path)
    prompt += f"\n### Question: {question}\n"
    prompt += f"\n### Hint:\n{knowledge}\n"
    prompt += f"\n###You just generated the following SQL:\n{sql}\n\n"
    
    prompt += """
        Now, please provide your thought process behind the generation of this SQL query.
        Your explanation should be concise and efficient, focusing on the key reasoning steps.
        """
    return prompt

