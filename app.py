from flask import Flask, render_template, jsonify , request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
import os

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()
print("✅ Embeddings model loaded")

# 5. Load FAISS index
docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("✅ FAISS index loaded from disk.")

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 400,
        'temperature': 0.5,
        'stop': ["User:", "Assistant:", "\n\n"]
    }
)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg.strip()
    print(input)

    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if input.lower() in greetings:
        return "Hello! 😊 How can I assist you with a medical or health-related question today?"

    try:
        result = qa.invoke({"query": input})
        print("Response : ", result["result"])
        return str(result["result"])
    except Exception as e:
        print("❌ Error: ", e)
        return "Sorry, something went wrong while processing your request. Please try again."

if __name__ =='__main__':
    app.run(debug= True)