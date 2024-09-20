# Patch SSL before other imports
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Apply gevent monkey patch early
import gevent.monkey
gevent.monkey.patch_all()

# Now import other modules
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
import warnings
import os
import requests
import textwrap
from langchain_openai import OpenAI, OpenAIEmbeddings
import sys

# Increase recursion limit
sys.setrecursionlimit(3000)

# Check if OpenAI API Key is set in the environment
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "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"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)

def prettify_text(text):
    prettified = text.replace('**', '\n').replace('*', '\n')
    return prettified    

def llm_model(question, data):
    model = OpenAI()
    logger.info("-------------------------DATA PASSING TO THE MODEL!!!--------------------------")
    prompt = f'''You are an AI assistant  designed by Deepak to help people by providing detailed and accurate answers based on the provided data you have. You are more than an assistant; you are a friend to Deepak. Ensure your responses are informative, contextually relevant, and align with Deepak's tone and style of communication.
                Converse like a human 
                Context:\n{data}\n
                
                Question:\n{question}\n
                
                Note:
                - If the question is directly related to the provided data, provide a detailed and accurate answer.
                - If the question pertains to a general topic or is conversational in nature, respond in a friendly, human-like manner. For example, for questions like "tell me something you know" or "is this the right time to talk," answer conversationally and encourage engagement.
                - If the answer is not present in the provided context and the question seems personal, unknown, or too common, respond by acknowledging your limitation in a friendly way. For instance, say "Oh no! I am not really aware of it, I shall ask Deepak and let you know later!!" to avoid giving incorrect information.
                - Always prioritize clarity, accuracy, and a friendly tone in your responses. Even if the answer is not relevant, try to respond in a helpful and engaging manner.
                
                Answer:'''

    response = model.invoke(prompt)    
    logger.info("-------------------------MODEL DATA DONE!!!--------------------------\n\n\n\n\n")            
    return response

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def user_input(user_question):
    # Initialize AI Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=1024
    )
    
    # Load FAISS index
    db = FAISS.load_local(folder_path="faiss_db", embeddings=embeddings, index_name="myFaissIndex", allow_dangerous_deserialization=True)
    logger.info("-------------------------DATABASE LOADED!!!--------------------------")    
    
    # Search for similar documents
    docs = db.similarity_search(user_question, k=2)
    logger.info("-------------------------RETRIEVED SIMILAR DATA!!!--------------------------") 
    
    context = " ".join([doc.page_content for doc in docs])
    print(context)

    return context

# Define the index route
@app.route('/')
def index():
    return "Happy that You are the top 1 percent in the world!!!!!!"

# Define the /ask route to handle POST requests
@app.route('/ask', methods=['POST'])
def ask():
    # Get user's question from the request
    user_question = request.form['question']
    logger.info(f"USER QUESTION: {user_question}")
    
    # Get response based on user's question
    response = user_input(user_question)
    out = llm_model(user_question, response)
    logger.info(f"User Question: {user_question}, Response: {out}")

    # Return the response as JSON
    return jsonify({'response': prettify_text(out)})

# Run the Flask app
if __name__ == '__main__':
    # Run the app in debug mode
    app.run(debug=True, threaded=True, port=5000, host='0.0.0.0')
