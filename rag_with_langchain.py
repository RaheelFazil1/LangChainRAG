import os
# import hmac
# import hashlib
import google.generativeai as genai
# from flask import Flask, request, jsonify
# from flask_cors import CORS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
# Initialize Flask App
# app = Flask(__name__)
# CORS(app)

# Load Environment Variables (Optional if you use .env)
# load_dotenv()

# API Key for Google Generative AI
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("ERROR: GEMINI_API_KEY is missing! Set it in your environment.")

# Gemini Model Configuration
genai.configure(api_key=API_KEY)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)
# Set up LangChain LLM and Memory
llm = GoogleGenerativeAI(model=model, google_api_key=API_KEY)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "prompt": "Respond in English about Butt Karahi's menu, locations, and prices.",
           "greeting": "Hello! How can I help you today?"},
    "ur": {"name": "Urdu", "prompt": "Butt Karahi کے مینو، مقامات اور قیمتوں کے بارے میں اردو میں جواب دیں۔",
           "greeting": "ہیلو! آج میں آپ کی کس طرح مدد کر سکتا ہوں؟"},
    "ar": {"name": "Arabic", "prompt": "الرد بالعربية حول قائمة مطعم بط كراهي والمواقع والأسعار.",
           "greeting": "مرحبًا! كيف يمكنني مساعدتك اليوم؟"},
    "hi": {"name": "Hindi", "prompt": "मेनू, स्थानों और कीमतों के बारे में हिंदी में उत्तर दें।",
           "greeting": "नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूँ?"},
    "es": {"name": "Spanish", "prompt": "Responde en español sobre el menú, ubicaciones y precios de Butt Karahi.",
           "greeting": "¡Hola! ¿Cómo puedo ayudarte hoy?"},
    "fr": {"name": "French", "prompt": "Répondez en français sur le menu, les emplacements et les prix de Butt Karahi.",
           "greeting": "Bonjour ! Comment puis-je vous aider aujourd'hui ?"},
    "zh-cn": {"name": "Chinese", "prompt": "用中文回答有关Butt Karahi的菜单、位置和价格。",
              "greeting": "你好！今天有什么可以帮您的吗？"},
    "bn": {"name": "Bengali", "prompt": "Butt Karahi-এর মেনু, অবস্থান এবং মূল্য সম্পর্কে বাংলায় উত্তর দিন।",
           "greeting": "হ্যালো! আজ আমি আপনাকে কিভাবে সাহায্য করতে পারি?"},
    "pa": {"name": "Punjabi", "prompt": "Butt Karahi ਦੇ ਮੀਨੂ, ਟਿਕਾਣਿਆਂ ਅਤੇ ਕੀਮਤਾਂ ਬਾਰੇ ਪੰਜਾਬੀ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।",
           "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡੀ ਆਜ਼ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ؟"},
    "tr": {"name": "Turkish", "prompt": "Butt Karahi'nin menüsü, konumları ve fiyatları hakkında Türkçe yanıt verin.",
           "greeting": "Merhaba! Bugün size nasıl yardımcı olabilirim?"}
}


# Helper Functions
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except Exception as e:
        return f"❌ ERROR: language not detected. :  {str(e)}"

def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # Split the data into paragraphs or sentences
    chunks = data.split('\n\n')  # Adjust splitting logic based on your file format
    return chunks

def retrieve_relevant_chunks(query, top_k=3):
    kb_file_path = 'housess_knowledge_base.txt'
    knowledge_base = load_knowledge_base(kb_file_path)
    # Create a TF-IDF vectorizer for the knowledge base
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(knowledge_base)
    # Vectorize the query
    query_vector = vectorizer.transform([query])
    # Compute cosine similarity between the query and all chunks
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    # Get the indices of the top-k most similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    # Retrieve the corresponding chunks
    relevant_chunks = [knowledge_base[i] for i in top_indices]
    return relevant_chunks

def generate_response(input_text: str, language: str) -> str:
    try:
        # Detect the language if it's not passed as argument
        if language == "Auto-Detect":
            user_lang = detect_language(input_text)
        else:
            user_lang = language

        lang_config = SUPPORTED_LANGUAGES.get(user_lang, SUPPORTED_LANGUAGES["en"])

        # Step 1: Retrieve relevant chunks from the knowledge base
        relevant_chunks = retrieve_relevant_chunks(input_text, top_k=3)

        # Handle case where no relevant chunks are found
        if not relevant_chunks:
            return ("I'm sorry, I couldn't find any relevant information to answer your question. Further more you can "
                    "contact on the Number :  +971 04 569 3020 or info@housess.ae")

        context = "\n".join(relevant_chunks)

        # intro = """
        #     "input: Who are you?",
        #     "output: I am an AI agent of Housess Real Estate. I will help you choose the best Real Estate property for you.",
        # """

        persona = """
                You are a helpful AI assistant of Housess Real Estate specialized in real estate. 
                Your goal is to provide accurate, concise, and friendly responses to user queries. 
                If you don't know the answer, politely inform the user.
                """

        context = persona + context + "\n\n"
        # Step 2: Generate a response using the retrieved context

        # Construct the full prompt by combining the context and the input text
        # full_prompt = f"{context}\ninput: {input_text}\noutput:"
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{input_text}\n\nAnswer:"

        # Generate response using the model with memory
        response = model.generate_content([  # Assuming you're using the generative model here
            "System: " + lang_config["prompt"],
            full_prompt
        ])

        return response.text
    except Exception as e:
        return f"❌ ERROR: {str(e)}"


# API Routes
# @app.route('/chat', methods=['POST'])
# def chat():
#     # Get user input directly from the request body
#     user_input = request.form.get('message')  # Getting message from the frontend form
#
#     if not user_input:
#         return jsonify({'error': 'No message provided.'}), 400
#
#     # Detect language of the input
#     user_lang = detect_language(user_input)
#     lang_config = SUPPORTED_LANGUAGES.get(user_lang, SUPPORTED_LANGUAGES["en"])
#
#     system_prompt = lang_config["prompt"]
#
#     # Generate response using LangChain's conversation memory
#     full_input = f"{system_prompt}\nUser: {user_input}"
#     bot_response = conversation.run(full_input)
#
#     return jsonify({'response': bot_response})
#
#
# @app.route('/', methods=['GET'])
# def home():
#     return "Chatbot is running!"
#
#
# if __name__ == '__main__':
#     app.run(debug=True)