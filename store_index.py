from src.helper import load_pdf, load_json,text_split, download_hugging_face_embeddings, clean_extracted_data, clean_pdf_text
from langchain_community.vectorstores import FAISS
import os
from sentence_transformers import SentenceTransformer

extracted_data = load_pdf("data/")
json_docs = load_json("data/medquad_data.json")

print(f"✅ Loaded {len(json_docs)} documents from JSON")
print(json_docs[0].page_content)  # Preview the first one

cleaned_data = clean_extracted_data(extracted_data)

all_docs = cleaned_data + json_docs

print(f"✅ Combined {len(all_docs)} documents")

text_chunks= text_split(all_docs)

text_chunks= text_split(all_docs)
print("length of my chunks:",len(text_chunks))

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

embeddings= download_hugging_face_embeddings()

# Create the vectorstore
docsearch = FAISS.from_documents(text_chunks, embeddings)

# Save locally (optional)
docsearch.save_local("faiss_index")

docsearch = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
