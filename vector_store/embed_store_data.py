from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import time
import os
from pathlib import Path
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datasets import load_dataset


data_path = "D:\StudyUET\HOCTAP3_1\KEPKO_KDN\KDN_LangChain_ChatBot\data\data.csv"
# Load the dataset
ds = load_dataset("tuananh18/Eval-RAG-Vietnamese", "legal-data")

def get_documents_from_csv(file_path: str) -> List[Document]:
    loader = CSVLoader(file_path=file_path, encoding='utf-8')
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    data = text_splitter.split_documents(data)
    return data

def load_embed_model(model_name: str = "BAAI/bge-small-en") -> HuggingFaceBgeEmbeddings:
    print(f"Loading embed model: {model_name}")
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings


def vector_store(embeddings: HuggingFaceBgeEmbeddings, documents: Document, save_path: str) -> FAISS:
    start_time = time.time()
    if not os.path.exists(save_path):
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(save_path)
    else:
        db = FAISS.load_local(save_path, embeddings,
                              allow_dangerous_deserialization=True)
    end_time = time.time()
    execution_time = end_time - start_time
    print(
        f"Time taken for make vector embeddings: {execution_time:.2f} seconds")
    return db

data = get_documents_from_csv(data_path)

print(data[0])