from constants import openai_key
from openai import OpenAI
import numpy as np


class Encoder:
    def __init__(self):
        self.content = []
        self.embeddings = []

    def get_openai_embeddings(self, content):
        client = OpenAI(api_key=openai_key)
        response = client.embeddings.create(
            input=content, model="text-embedding-3-small", dimensions=1024
        )
        embeddings = [response.data[i].embedding for i in range(len(response.data))]
        return embeddings

    def load_embeddings(self, content):
        embeddings = self.get_openai_embeddings(content)
        self.embeddings = embeddings
        self.content = content

    def _cosine_similarity(self, vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def similarity_search(self, query, top_n=5):
        query_vector = np.array(self.get_openai_embeddings([query])[0])
        similarities = np.array(
            [self._cosine_similarity(query_vector, vec) for vec in self.embeddings]
        )
        sorted_indices = np.argsort(similarities)[::-1][:top_n]
        similar_chunks = [self.content[idx] for idx in sorted_indices]
        return similar_chunks
