import os
import openai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def get_openai_embedding(phrase):
    response = openai.Embedding.create(
        input=phrase,
        model="text-embedding-ada-002"
    )
    embedding = response['data'][0]['embedding']
    return np.array(embedding)

central_phrase = "khao hoc mi tinh"
central_vector = get_openai_embedding(central_phrase)

phrases = ["computer science", "KHMT", "data science", "khoa học máy tính"]
other_vectors = [get_openai_embedding(phrase) for phrase in phrases]

similarities = cosine_similarity([central_vector], other_vectors)

sorted_phrases = [phrase for _, phrase in sorted(zip(similarities[0], phrases), reverse=True)]

print(sorted_phrases)
