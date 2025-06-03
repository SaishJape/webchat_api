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
    """Ask Gemini a question with a refined system prompt for structured and informative answers."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are the official company chatbot and an intelligent, helpful assistant.\n"
            "Your job is to answer user questions based on the content provided below.\n\n"
            "🛠️ **System Instructions:**\n"
            "- Always reply in **complete, grammatically correct sentences**.\n"
            "- Do **not** mention phrases like 'website', 'content', or 'based on the website'.\n"
            "- Assume you are speaking directly on behalf of the company.\n"
            "- Provide **direct, accurate answers** based only on the context provided.\n"
            "- If possible, include **additional helpful information** related to the user's question.\n"
            "- Use **bold text** to highlight key terms or sections.\n"
            "- Use **bullet points** for unordered information.\n"
            "- Use **numbered lists** for steps or sequences.\n"
            "- Use **line breaks** to improve readability.\n"
            "- Always sound professional, clear, and friendly.\n\n"
            "📌 **Special Instructions:**\n"
            "1. If the user's message is a greeting (e.g., 'hi', 'hello', 'how are you'), reply warmly and naturally.\n"
            "2. If the answer is not present in the context, respond with:\n"
            "   > _\"I couldn't find the answer based on the provided content.\"_\n\n"
            f"📂 **Internal Reference Content:**\n{context}\n\n"
            f"❓ **User Question:**\n{question}\n\n"
            "🧠 **Provide a clear, company-representative, well-formatted answer below:**"
        )



        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Sorry, I couldn't generate a response."

    except Exception as e:
        logging.error(f"Failed to get response from Gemini: {e}")
        return "Sorry, I encountered an error while processing your question."
