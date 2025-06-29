from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI 
from langchain.chains import ConversationalRetrievalChain
from src.prompt import *
import time
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Loads OPENAI_API_KEY from .env
load_dotenv()  

app = Flask(__name__)

# Global flags
llm = None
qa = None
model_loaded = False

@app.before_request
def load_model_if_needed():
    global llm, qa, model_loaded

    if not model_loaded:
        print("🔄 Loading embeddings...")
        embeddings = download_hugging_face_embeddings()
        print("✅ Embeddings model loaded")

        print("🔄 Loading FAISS index...")
        docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded")

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        print("🔄 Initializing GPT-4o...")
        llm_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        output_key="answer"  
        )


        llm = llm_model
        qa = qa_chain
        model_loaded = True
        print("✅ GPT-4o and retrieval chain loaded")

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    global qa
    msg = request.form["msg"]
    user_input = msg.strip()
    print("User input:", user_input)

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(greet in user_input.lower() for greet in greetings):
        return "Hello! 😊 How can I assist you with a medical or health-related question today?"
    
    thanks_phrases = ["thank you", "thanks", "thx", "okay thank you", "thank u"]
    if user_input.lower() in thanks_phrases:
            return "You're very welcome! 😊 Stay safe and feel free to reach out with any health-related concerns."

    try:
        start_time = time.time()
        result = qa.invoke({"question": user_input})
        end_time = time.time()

        print(f"Response: {result['answer']}")
        print(f"⏱️ Time: {end_time - start_time:.2f} seconds")

        with open("chat_log1.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_input}\nBot: {result['answer']}\n\n")

        return str(result["answer"])

    except Exception as e:
        print("❌ Error:", e)
        return "Sorry, something went wrong while processing your request."

@app.route("/health")
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)
