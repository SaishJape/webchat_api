import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(context: str, question: str) -> str:
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
