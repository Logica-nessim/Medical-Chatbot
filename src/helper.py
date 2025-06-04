from langchain.document_loaders import PyPDFLoader, DirectoryLoader
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from langchain.document_loaders import PyPDFLoader
from hashlib import md5
from langchain.embeddings import HuggingFaceBgeEmbeddings

#extract data from PDF
def load_pdf(data):
    loader =  DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents=loader.load()
    
    return documents

#extract data from json

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        # Combine Q&A into one document
        if question and answer:
            content = f"Question: {question}\nAnswer: {answer}"
            documents.append(Document(page_content=content))
    
    return documents

# Step 2: Initialize the splitter
def text_split(all_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(all_docs)
    
    return text_chunks



def clean_pdf_text(text):
    # Remove page numbers (e.g., "Page 1", "1 of 300")
    text = re.sub(r"Page\s*\d+|\d+\s*of\s*\d+", "", text)

    # Remove chapter headings
    text = re.sub(r"CHAPTER\s+\w+|Chapter\s+\w+", "", text, flags=re.IGNORECASE)

    # Remove repeated headers/footers
    text = re.sub(r"\n?(CURRENT Medical Diagnosis and Treatment|The GALE ENCYCLOPEDIA of MEDICINE).*?\n", "\n", text, flags=re.IGNORECASE)

    # Remove extra whitespace
    text = text.strip()

    # Skip pages with unwanted keywords
    skip_keywords = [
        "table of contents", "index", "references", "bibliography", 
        "authors", "preface", "contributors", "editorial board","advisory board","contents"
    ]
    if any(keyword in text.lower() for keyword in skip_keywords):
        return ""

    # Skip if the content starts with a keyword (title pages etc.)
    start_skip_phrases = [
        "authors", "preface", "table of contents", "about the author"
    ]
    for phrase in start_skip_phrases:
        if text.lower().startswith(phrase):
            return ""

    # Skip short pages (less than 20 words)
    if len(text.split()) < 20:
        return ""

    return text

def clean_extracted_data(docs):
    cleaned_docs = []
    seen_hashes = set()

    for doc in docs:
        cleaned_text = clean_pdf_text(doc.page_content)
        if cleaned_text:
            # Remove duplicates
            content_hash = md5(cleaned_text.encode("utf-8")).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                doc.page_content = cleaned_text
                cleaned_docs.append(doc)

    return cleaned_docs


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings