import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text chunks using SentenceTransformer."""
    return embedding_model.encode(texts).tolist()

def get_question_embedding(question: str) -> list[float]:
    """Generate a single embedding for a user's question using SentenceTransformer."""
    return embedding_model.encode([question])[0].tolist()

def ask_gemini(context: str, question: str) -> str:
    """Smart hybrid Gemini prompt: responds to general chat and uses web context if relevant."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are a helpful assistant.\n\n"
        "If the user's message is a greeting or casual message (like 'hello', 'hi', 'how are you'), respond appropriately.\n"
        "If the user asks a question related to the website content, answer it using the context below.\n"
        "If the answer is not found in the context, say: \"I couldn't find the answer based on the provided content.\"\n\n"
        f"Website Context:\n{context}\n\n"
        f"User Message:\n{question}"
    )

    response = model.generate_content(prompt)
    return response.text.strip()
