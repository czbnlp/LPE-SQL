import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

class TextToSQLRetriever:
    def __init__(self, top_k, correct_set_path="correct_set.json", mistake_set_path="mistake_set.json",
                 embedding_model_path="sentence-transformers/all-MiniLM-L6-v2", device="cuda",
                 correct_vectors_path="correct_vectors.npy", mistake_vectors_path="mistake_vectors.npy",engine = 'qwen2-72b',
                 use_knowledge_base = "True",init_correct_set_path = "", init_mistake_set_path = "", correct_rate = 0.5):
        self.top_k = top_k
        self.correct_rate = correct_rate
        self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
        root_path = '/home/chuzhibo/acl/text-to-sql/bird/llm/src/knowledge_base'
        # 文件路径
        engine_dir = os.path.join(root_path, engine+ '_'+str(top_k)+ '_'+str(use_knowledge_base))
        os.makedirs(engine_dir, exist_ok=True)
        self.correct_set_path = os.path.join(engine_dir, correct_set_path)
        self.mistake_set_path = os.path.join(engine_dir, mistake_set_path)
        self.correct_vectors_path = os.path.join(engine_dir, correct_vectors_path)
        self.mistake_vectors_path = os.path.join(engine_dir, mistake_vectors_path)
        # 加载correct_set和mistake_set
        self.correct_set = self._load_set_from_json(self.correct_set_path) if os.path.exists(self.correct_set_path) else []
        self.mistake_set = self._load_set_from_json(init_mistake_set_path) if os.path.exists(init_mistake_set_path) else []
        self.init_correct_len = len(self.correct_set)
        self.init_mistake_len = len(self.mistake_set)
        # 初始化时检查并加载向量
        self.correct_vectors = self._load_or_encode_dataset(self.correct_set, self.correct_vectors_path) if self.correct_set else None
        self.mistake_vectors = self._load_or_encode_dataset(self.mistake_set, self.mistake_vectors_path) if self.mistake_set else None


    def _encode_dataset(self, dataset):
        texts = [entry['question'] for entry in dataset]
        vectors = self.embedding_model.encode(texts)
        return vectors

    def _save_vectors_to_disk(self, vectors, filepath):
        np.save(filepath, vectors)

    def _load_vectors_from_disk(self, filepath):
        return np.load(filepath)

    def _load_or_encode_dataset(self, dataset, filepath):
        
        if os.path.exists(filepath):
            return self._load_vectors_from_disk(filepath)
        else:
            vectors = self._encode_dataset(dataset)
            self._save_vectors_to_disk(vectors, filepath)
            return vectors

    def _save_set_to_json(self, dataset, filepath):
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=4)

    def _load_set_from_json(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    def retrieve_similar_examples(self, query):
        query_vector = self.embedding_model.encode([query])
        
        # Ensure the number of examples to retrieve is an integer
        num_correct_to_retrieve = int(self.correct_rate * self.top_k)
        num_mistakes_to_retrieve = self.top_k - num_correct_to_retrieve
        
        # Adjust the number of examples to retrieve if there aren't enough mistakes
        if len(self.mistake_set) < num_mistakes_to_retrieve:
            num_correct_to_retrieve = self.top_k - len(self.mistake_set)
            num_mistakes_to_retrieve = len(self.mistake_set)
        
        correct_examples = self._retrieve_from_vectors(query_vector, self.correct_set, self.correct_vectors,num_correct_to_retrieve) if self.correct_vectors is not None else []
        mistake_examples = self._retrieve_from_vectors(query_vector, self.mistake_set, self.mistake_vectors,num_mistakes_to_retrieve) if self.mistake_vectors is not None else []
        # print(f"correct: {num_correct_to_retrieve}; mistake: {num_mistakes_to_retrieve}")
        return correct_examples, mistake_examples

    def _retrieve_from_vectors(self, query_vector, dataset, vectors, num_k):
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        faiss.normalize_L2(query_vector)
        distances, ann = index.search(query_vector, k=num_k)
        print(distances, ann)
        print(len(dataset))
        similar_examples = [dataset[i] for i in ann[0]]
        return similar_examples

    def add_to_sets(self, question, sql, correct=True, **kwargs):
        if correct:
            self.correct_set.append({'question': question, 
                                    'hint':kwargs.get('knowledge'),
                                    'sql': sql,
                                    'thought process': kwargs.get('cot'),
                                    'difficulty': kwargs.get('difficulty')}
                                    )
            new_vector = self.embedding_model.encode([question])
            if self.correct_vectors is None:
                self.correct_vectors = new_vector
            else:
                self.correct_vectors = np.vstack([self.correct_vectors, new_vector])
            self._save_vectors_to_disk(self.correct_vectors, self.correct_vectors_path)
            self._save_set_to_json(self.correct_set, self.correct_set_path)
        else:
            mistake_entry = {
                'question': question,
                'hint':kwargs.get('knowledge'),
                'error_sql': kwargs.get('error_sql'),
                'compiler_hint': kwargs.get('compiler_hint'),
                'reflective_cot': kwargs.get('reflective_cot'),
                'ground_truth_sql': sql,
                'difficulty': kwargs.get('difficulty'),
            }
            self.mistake_set.append(mistake_entry)
            new_vector = self.embedding_model.encode([question])
            if self.mistake_vectors is None:
                self.mistake_vectors = new_vector
            else:
                self.mistake_vectors = np.vstack([self.mistake_vectors, new_vector])
            self._save_vectors_to_disk(self.mistake_vectors, self.mistake_vectors_path)
            self._save_set_to_json(self.mistake_set, self.mistake_set_path)

    def get_in_context_examples(self, query):
        correct_examples, mistake_examples = self.retrieve_similar_examples(query)
        return correct_examples, mistake_examples
    
    def extract_information(self, input_string):
        # 定位各部分的起始位置
        question_start = input_string.find("question:") + len("question:")
        hint_start = input_string.find("hint:")
        sql_start = input_string.find("sql:")
        thought_process_start = input_string.find("thought process:")

        # 提取各部分的内容
        question = input_string[question_start:hint_start].strip().strip(",")
        hint = input_string[hint_start + len("hint:"):sql_start].strip().strip(",")
        sql = input_string[sql_start + len("sql:"):thought_process_start].strip().strip(",")
        thought_process = input_string[thought_process_start + len("thought process:"):].strip().strip(",").rstrip("}")

        return question, hint, sql, thought_process

# retrieval = TextToSQLRetriever(top_k=8)
# from utils import compute_acc_by_diff, save_data
# correct_set_path = retrieval.correct_set_path
# mistake_set_path = retrieval.mistake_set_path
# simple_acc, moderate_acc, challenging_acc, acc, count_lists = compute_acc_by_diff(
#     correct_set_path, mistake_set_path
# )
# score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
# file_name = 'qwen2-72b' + '_'+str(8)+ '.txt'
# results_path = "/home/chuzhibo/acl/text-to-sql/bird/llm/src/knowledge_base/results/"
# save_data(1,results_path+file_name,score_lists, count_lists, metric="EX")


# from prompt import generate_examples
# print(generate_examples("What is the ratio of customers", retriever))

# retriever.add_to_sets("In 2012, who had the least consumption in LAM?",
#         "SELECT CustomerID, SUM(Consumption) AS TotalConsumption FROM yearmonth WHERE substr(Date, 1, 4) = '2012' GROUP BY CustomerID ORDER BY TotalConsumption ASC LIMIT 1",\
#         compiler_hint= None,
#         reflective_cot="Upon reflection, the mistake in the original SQL query was not considering the segment 'LAM' and not joining the customers table with the yearmonth table. The original query only considered the year 2012 and calculated the total consumption for each customer, but it did not filter by the segment 'LAM'. To correct this, I need to join the customers table with the yearmonth table on the CustomerID field, filter by the segment 'LAM' and the year 2012, group by CustomerID, and order by the total consumption in ascending order. I will limit the result to 1 to find the customer with the least consumption in LAM in 2012. To avoid making the same mistake in the future, I should always check if there are any join conditions or filters that need to be applied to the query, and make sure to include them.",
#         ground_truth_sql="SELECT T1.CustomerID FROM customers AS T1 INNER JOIN yearmonth AS T2 ON T1.CustomerID = T2.CustomerID WHERE T1.Segment = 'LAM' AND SUBSTR(T2.Date, 1, 4) = '2012' GROUP BY T1.CustomerID ORDER BY SUM(T2.Consumption) ASC LIMIT 1",
#         difficulty="moderate")

# print(generate_examples("What is the ratio of customers", retriever))