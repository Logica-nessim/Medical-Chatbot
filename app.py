from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.prompt import *
import os

app = Flask(__name__)

# 1. Load embeddings
embeddings = download_hugging_face_embeddings()
print("✅ Embeddings model loaded")

# 2. Load FAISS index
docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("✅ FAISS index loaded from disk.")

# 3. Set custom prompt (optional)
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Load LLM
llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 400,
        'temperature': 0.5,
        'stop': ["User:", "Assistant:", "\n\n"]
    }
)

# 5. Memory for conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # <- explicitly specify this
)

# 6. Build conversational retrieval chain (without deprecated StuffDocumentsChain or LLMChain)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    memory=memory,
    return_source_documents=True,
    # If you want to use a custom prompt, you can pass prompt=PROMPT here,
    # but note this may require compatible prompt structure.
)

# 7. Flask Routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    user_input = msg.strip()
    print("User input:", user_input)

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if user_input.lower() in greetings:
        return "Hello! 😊 How can I assist you with a medical or health-related question today?"

    try:
        result = qa.invoke({"question": user_input})  # Use "question" key as per default input_key
        print("Response:", result["answer"])

        # Optional logging
        with open("chat_log.txt", "a", encoding="utf-8") as f:
            f.write(f"User: {user_input}\nBot: {result['answer']}\n\n")

        return str(result["answer"])
    except Exception as e:
        print("❌ Error:", e)
        return "Sorry, something went wrong while processing your request. Please try again."

if __name__ == '__main__':
    app.run(debug=True) 