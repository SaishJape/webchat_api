import google.generativeai as genai
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

genai.configure(api_key="AIzaSyD42b36DrJON2jUIt_dUQyYUmafqqlJrhg")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text chunks using SentenceTransformer."""
    return embedding_model.encode(texts).tolist()
