import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
load_dotenv()

my_api_key = os.getenv("API_KEY")
genai.configure(api_key=my_api_key)

model = genai.GenerativeModel("gemini-1.5-flash")
# response_v1a = model.generate_content("what is 2+2 formula")
# print(response_v1a.text)

def load_knowledge_base(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    # Split the data into paragraphs or sentences
    chunks = data.split('\n\n')  # Adjust splitting logic based on your file format
    return chunks

def retrieve_relevant_chunks(query, top_k=3):
    file_path = 'housess_knowledge_base.txt'
    knowledge_base = load_knowledge_base(file_path)
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

# print(retrieve_relevant_chunks("What is 2+2 formula", 3))

# Set up LangChain LLM and Memory
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=my_api_key)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

def rag_response(query):
    # Step 1: Retrieve relevant chunks from the knowledge base
    relevant_chunks = retrieve_relevant_chunks(query, top_k=3)

    # Handle case where no relevant chunks are found
    if not relevant_chunks:
        return ("I'm sorry, I couldn't find any relevant information to answer your question. Further more you can"
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

    # Combine persona, context, and conversation history
    full_context = persona + "\n\nRetrieved Context:\n" + context + "\n\n"

    # Use the ConversationChain to generate a response
    try:
        response = conversation.run(input=f"Context:\n{full_context}\n\nQuestion:\n{query}\n\nAnswer:")
        return response.strip()
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

    # context = persona + context + "\n\n"
    # # Step 2: Generate a response using the retrieved context
    # prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    #
    # response = model.generate_content(prompt)
    #
    # return response.text

# query = 'what are you services?'
# response = rag_response(query)
# print(response)

# query = 'my name is Raheel, do you have property in Dubai hill eastate?'
# response = rag_response(query)
# print(response)
#
# print('------------------------------------------------------------------------')
# query = 'how many housed you have there for rent or sale.'
# response = rag_response(query)
# print(response)
#
# print('------------------------------------------------------------------------')
# query = 'what are the average prices here for 2 bed apartment?'
# response = rag_response(query)
# print(response)
#
# print('------------------------------------------------------------------------')
# query = 'do you know me? What is my name?'
# response = rag_response(query)
# print(response)
#
# print('------------------------------------------------------------------------')
# query = 'summarize my chat.'
# response = rag_response(query)
# print(response)

def clear_memory():
    global memory
    memory.clear()
    print("I have cleared our conversation history.")
    # return "I have cleared our conversation history."

def main():
    print("Welcome to the Housess Real Estate Assistant!")
    print("Type 'exit' at any time to end the conversation.\n")

    while True:
        # Prompt the user for input
        query = input("You: ").strip()

        # Check if the user wants to exit
        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Thank you for using the Housess Real Estate Assistant.")
            break

        # Generate and display the response
        response = rag_response(query)
        print(f"\nAssistant: {response}\n{'-' * 70}\n")

# Run the main function
if __name__ == "__main__":
    main()
    print('-----Test completed-------')
    clear_memory()


