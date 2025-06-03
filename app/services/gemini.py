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


def analyze_user_query(question: str) -> dict:
    """Analyze user query to extract key information and intent."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "Analyze this user question and extract key information:\n"
        "1. Main topic/subject\n"
        "2. Key keywords for search\n"
        "3. Question type (factual, how-to, definition, comparison, etc.)\n"
        "4. Intent (what specifically they want to know)\n\n"
        "Return response in this JSON format:\n"
        "{\n"
        '  "main_topic": "extracted main topic",\n'
        '  "keywords": ["keyword1", "keyword2", "keyword3"],\n'
        '  "question_type": "factual/how-to/definition/etc",\n'
        '  "intent": "specific intent description"\n'
        "}\n\n"
        f"User Question: {question}"
    )
    
    try:
        response = model.generate_content(prompt)
        import json
        return json.loads(response.text.strip())
    except:
        return {
            "main_topic": question,
            "keywords": [question],
            "question_type": "general",
            "intent": "general information"
        }

def ask_gemini_enhanced(context: str, question: str, query_analysis: dict) -> str:
    """Enhanced Gemini response with query analysis context."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    intent = query_analysis.get('intent', 'general information')
    question_type = query_analysis.get('question_type', 'general')
    keywords = ', '.join(query_analysis.get('keywords', []))
    
    prompt = (
        "You are a helpful AI assistant analyzing website content.\n\n"
        f"User's Intent: {intent}\n"
        f"Question Type: {question_type}\n"
        f"Key Topics: {keywords}\n\n"
        "Instructions:\n"
        "- If this is a greeting, respond naturally\n"
        "- For content questions, use the provided context to give accurate, detailed answers\n"
        "- Focus on the specific intent and question type identified\n"
        "- If information is not in the context, clearly state that\n"
        "- Provide structured answers for how-to questions\n"
        "- Give clear definitions for definition-type questions\n\n"
        f"Website Content Context:\n{context}\n\n"
        f"User Question: {question}\n\n"
        "Answer:"
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return "I'm sorry, I encountered an error processing your request. Please try again."