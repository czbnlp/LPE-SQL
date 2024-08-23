
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

class TextToSQLRetriever:
    def __init__(self, top_k, correct_set_path="knowledge_base/correct_set.json", mistake_set_path="knowledge_base/mistake_set.json",
                 embedding_model_path="sentence-transformers/all-MiniLM-L6-v2", device="cuda",
                 correct_vectors_path="knowledge_base/correct_vectors.npy", mistake_vectors_path="knowledge_base/mistake_vectors.npy"):
        self.top_k = top_k
        self.embedding_model = SentenceTransformer(embedding_model_path, device=device)
        
        root_path = '/home/chuzhibo/acl/my/bird/llm/src/'
        # 文件路径
        self.correct_set_path = root_path+correct_set_path
        self.mistake_set_path = root_path+mistake_set_path
        self.correct_vectors_path = root_path+correct_vectors_path
        self.mistake_vectors_path = root_path+mistake_vectors_path
        # 加载correct_set和mistake_set
        self.correct_set = self._load_set_from_json(self.correct_set_path) if os.path.exists(self.correct_set_path) else []
        self.mistake_set = self._load_set_from_json(self.mistake_set_path) if os.path.exists(self.mistake_set_path) else []
        
        # 初始化时检查并加载向量
        self.correct_vectors = self._load_or_encode_dataset(self.correct_set, self.correct_vectors_path) if self.correct_set else None
        self.mistake_vectors = self._load_or_encode_dataset(self.mistake_set, self.mistake_vectors_path) if self.mistake_set else None

    def _encode_dataset(self, dataset):
        texts = [entry['text'] for entry in dataset]
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

        correct_examples = self._retrieve_from_vectors(query_vector, self.correct_set, self.correct_vectors) if self.correct_vectors is not None else []
        mistake_examples = self._retrieve_from_vectors(query_vector, self.mistake_set, self.mistake_vectors) if self.mistake_vectors is not None else []

        return correct_examples, mistake_examples

    def _retrieve_from_vectors(self, query_vector, dataset, vectors):
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        faiss.normalize_L2(query_vector)
        distances, ann = index.search(query_vector, k=self.top_k)
        # print(distances, ann)
        similar_examples = [dataset[i] for i in ann[0]]
        return similar_examples

    def add_to_sets(self, question, sql, correct=True, **kwargs):
        if correct:
            self.correct_set.append({'question': question, 'sql': sql})
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
                'error_sql': kwargs.get('error_sql'),
                'compiler_hint': kwargs.get('compiler_hint'),
                'reflective_cot': kwargs.get('reflective_cot'),
                'ground_truth_sql': sql
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
    

# retriever = TextToSQLRetriever(top_k=5)

# # 添加示例到正解集或错题集
# retriever.add_to_sets("Find all users", "SELECT * FROM users", correct=True)

# retriever.add_to_sets("Find user by name", "SELECT * FROM user WHERE name = 'John'", correct=False, error_sql="SELECT * FROM usr WHERE name = 'John'", compiler_hint="Table 'usr' does not exist", reflective_cot="Correct the table name from 'usr' to 'user'.")

# # 获取与查询相似的上下文示例
# query = "Show me all users"
# correct_examples, mistake_examples = retriever.get_in_context_examples(query)

# print("正解集示例:", correct_examples)
# print("错题集示例:", len(mistake_examples))