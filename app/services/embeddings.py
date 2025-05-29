from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts).tolist()

def get_question_embedding(question: str) -> list[float]:
    return embedding_model.encode([question])[0].tolist()