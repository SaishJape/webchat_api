import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import re
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=api_key)


def ask_gemini(context: str, question: str, query_analysis: dict) -> dict:
    """Ask Gemini and return a structured JSON response with optional buttons."""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = (
            "You are the official AI assistant of the company, designed to be smart, professional, and friendly.\n"
            "Use the internal company content below to help answer the user's question.\n\n"

            "🎯 Output Format Instructions:\n"
            "- ONLY return a **valid raw JSON object**. Do NOT include markdown (```json), quotes, or any extra text.\n"
            "- The JSON must contain exactly these 4 keys:\n"
            "  1. 'response': string → a clear, helpful, and grammatically correct sentence. Always provide a response — even for greetings or questions not found in the context.\n"
            "  2. 'buttons': boolean → true **only if actionable info** (email, phone, LinkedIn, etc.) is found in the context and relevant to the question.\n"
            "  3. 'button_type': list of strings like [\"email\", \"linkedin\", \"website\", \"phone\"], or null if buttons is false.\n"
            "  4. 'button_data': list of actual values from context matching the types above, or null if buttons is false.\n\n"

            "🧠 Rules:\n"
            "- If the user greets you (e.g., says 'hi', 'hello', 'hey'), respond warmly and naturally.\n"
            "- If the question is general or out-of-scope but can be answered politely, do so in a professional tone.\n"
            "- Use only real data from the context for button values. Never guess or hallucinate values.\n"
            "- If no actionable data is present or needed, set:\n"
            "  \"buttons\": false,\n"
            "  \"button_type\": null,\n"
            "  \"button_data\": null\n\n"

            "✅ Example Output:\n"
            '{\n'
            '  "response": "Welcome! I\'m here to help you with any questions about the company. How can I assist you today?",\n'
            '  "buttons": false,\n'
            '  "button_type": null,\n'
            '  "button_data": null\n'
            '}\n\n'

            "OR (if contact info is found):\n"
            '{\n'
            '  "response": "You can reach us through the following contact options:",\n'
            '  "buttons": true,\n'
            '  "button_type": ["email", "linkedin"],\n'
            '  "button_data": ["info@company.com", "https://linkedin.com/company/example"]\n'
            '}\n\n'

            f"📄 Internal Company Content:\n{context if context.strip() else 'No content available.'}\n\n"
            f"❓ User Question:\n{question}\n\n"
            "✍️ Please respond now with the final raw JSON object only:"
        )


        response = model.generate_content(prompt)
        text = response.text.strip()

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown-like ```json block
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group()
                return json.loads(cleaned_json)
            else:
                logging.warning("Could not extract valid JSON from Gemini response.")
                return {
                    "response": "Sorry, I couldn't generate a valid response.",
                    "buttons": False,
                    "button_type": None,
                    "button_data": None
                }

    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return {
            "response": "Sorry, an error occurred while processing your request.",
            "buttons": False,
            "button_type": None,
            "button_data": None
        }

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