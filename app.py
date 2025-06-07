from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from src.prompt import *
import time
from langchain.memory import ConversationSummaryBufferMemory

app = Flask(__name__)

# Global variables
llm = None
qa = None
model_loaded = False  # flag

@app.before_request
def load_model_if_needed():
    global llm, qa, model_loaded

    if not model_loaded:
        print("🔄 Loading embeddings...")
        embeddings = download_hugging_face_embeddings()
        print("✅ Embeddings model loaded")

        print("🔄 Loading FAISS index...")
        docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded from disk.")

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        print("🔄 Loading local LLM...")
        llm_model = CTransformers(
            model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_type="llama",
            config={
                'max_new_tokens': 200,
                'temperature': 0.5,
                'stop': ["User:", "Assistant:", "\n\n"]
            }
        )
        memory = ConversationSummaryBufferMemory(
            llm=llm_model,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )


        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm_model,
            retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}  # ✅ now uses your custom prompt
        )


        llm = llm_model
        qa = qa_chain
        model_loaded = True
        print("✅ LLM and retrieval chain loaded")

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

    try:
        start_time = time.time()
        result = qa.invoke({"question": user_input})
        end_time = time.time()

        print(f"Response: {result['answer']}")
        print(f"⏱️ Response time: {end_time - start_time:.2f} seconds")

        # Optional logging
        with open("chat_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_input}\nBot: {result['answer']}\n\n")

        return str(result["answer"])

    except Exception as e:
        print("❌ Error:", e)
        return "Sorry, something went wrong while processing your request. Please try again."

@app.route("/health")
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)