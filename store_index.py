from src.helper import load_csv, load_pdf, load_json, text_split, download_hugging_face_embeddings, clean_extracted_data
from langchain_community.vectorstores import FAISS

# 1. Load and clean data
extracted_data = load_pdf("data/")
json_docs = load_json("data/medquad_data.json")
csv_docs = load_csv("data\Final_Augmented_dataset_Diseases_and_Symptoms.csv")

print(f"✅ Loaded {len(json_docs)} documents from JSON")
print(json_docs[0].page_content)

cleaned_data = clean_extracted_data(extracted_data)
all_docs = cleaned_data + json_docs + csv_docs
print(f"✅ Combined {len(all_docs)} documents")

# 2. Split into text chunks
text_chunks = text_split(all_docs)
print("length of my chunks:", len(text_chunks))

# # 3. Load embeddings (HuggingFace)
embeddings = download_hugging_face_embeddings()
print("✅ Embeddings model loaded")

# 4. Create and save FAISS index
docsearch = FAISS.from_documents(text_chunks, embeddings)
# Save FAISS index
docsearch.save_local("faiss_index_2")
print("✅ FAISS index saved.")

# 5. Load FAISS index
docsearch = FAISS.load_local("faiss_index_2", embeddings, allow_dangerous_deserialization=True)
print("✅ FAISS index loaded from disk.")

# Check size
print(f"✅ Number of documents in FAISS index: {len(docsearch.index_to_docstore_id)}")

# Do a test query
query = "What is acute lymphoblastic leukemia?"
results = docsearch.similarity_search(query, k=3)

print(f"🔍 Search test: Found {len(results)} result(s) for query.")
print("🧠 Top result preview:")
print(results[0].page_content[:500])
