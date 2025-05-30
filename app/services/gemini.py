import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=api_key)

def ask_gemini(context: str, question: str) -> str:
    """Ask Gemini a question with given context."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            "You are a helpful assistant that answers questions based on website content.\n\n"
            "Instructions:\n"
            "- If the user's message is a greeting (like 'hello', 'hi', 'how are you'), respond appropriately.\n"
            "- If the user asks a question related to the website content, answer using the context below.\n"
            "- If the answer is not found in the context, say: \"I couldn't find the answer based on the provided content.\"\n"
            "- Be concise and accurate in your responses.\n\n"
            f"Website Context:\n{context}\n\n"
            f"User Question:\n{question}\n\n"
            "Answer:"
        )

        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Sorry, I couldn't generate a response."
        
    except Exception as e:
        logging.error(f"Failed to get response from Gemini: {e}")
        return "Sorry, I encountered an error while processing your question."