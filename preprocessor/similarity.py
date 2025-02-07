from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingsSimilarity():
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)

    def get_embedding(self, text):
        response = self.client.embeddings.create(input=text, model="text-embedding-3-large")
        return response.data[0].embedding
    
    def get_cosine_similarity(self, wordA, wordB):
        embeddings_A = self.get_embedding(wordA)
        embeddings_B = self.get_embedding(wordB)

        embeddings_A = np.array(embeddings_A).reshape(1, -1)
        embeddings_B = np.array(embeddings_B).reshape(1, -1)

        # 코사인 유사도 계산
        similarity = cosine_similarity(embeddings_A, embeddings_B)[0][0]

        # 출력
        return similarity