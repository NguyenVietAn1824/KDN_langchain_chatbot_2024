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
from langchain_experimental.text_splitter import SemanticChunker


# def get_documents_from_csv(file_path: str) -> List[Document]:
#     loader = CSVLoader(file_path=file_path, encoding='utf-8')
#     data = loader.load()
#     data_list = [doc.page_content for doc in data]  
#     model_kwargs = {"device": "cpu"}
#     encode_kwargs = {"normalize_embeddings": True}
#     embedding_model = HuggingFaceBgeEmbeddings(
#         model_name= "BAAI/bge-small-en", model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
#     )
#     text_splitter = SemanticChunker(
#     embedding_model, breakpoint_threshold_type="percentile")
#     data = text_splitter.create_documents(data_list)
#     return data

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
        print(f"Vector store created and saved at {save_path}")
    else:
        db = FAISS.load_local(save_path, embeddings,
                              allow_dangerous_deserialization=True)
        print(f"Vector store loaded from {save_path}")
    return db

