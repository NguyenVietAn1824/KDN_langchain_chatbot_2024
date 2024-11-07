import streamlit as st
import time
import os
from langchain_core.messages import HumanMessage, AIMessage
import numpy as np
from src.utils.create_chain import chain_query, get_llm, get_history_aware_chain
from src.utils.embed_store_data import load_embed_model, vector_store

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_community.embeddings import HuggingFaceEmbeddings

from src.semantic_router import SemanticRouter, Route
from src.semantic_router.sample import productsSample,chitchatSample
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
import openai
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
load_dotenv()

data_path = Path(__file__).parents[0] / Path("data/data_short.csv")
SAMPLE_PATH = Path(__file__).parents[0] / Path("vector_store/semantic_route_db/sample.json")
DEFAULT_VECTOR_DB_PATH = "vector_store/faiss_db"
llm = "gemma2:2b"

embed_model = "BAAI/bge-small-en"
PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'


# Load LLM, vector store, and retriever once
if "llm" not in st.session_state:
    print("Loading data")
    # data = get_documents_from_csv(data_path)
    embeddings = load_embed_model(model_name=embed_model)
    print("Loading vector store")
    
    db = FAISS.load_local(DEFAULT_VECTOR_DB_PATH, embeddings,
                              allow_dangerous_deserialization=True)
    print(db.index.ntotal)
    query_vector = embeddings.embed_query("Dự án thuộc ngành nghề ưu đãi đầu tư tại địa bàn khó khăn được hưởng ưu đãi như thế nào?")
    results = db.similarity_search(
        "Dự án thuộc ngành nghề ưu đãi đầu tư tại địa bàn khó khăn được hưởng ưu đãi như thế nào?",
        k=2,
        filter={"source": "tweet"},
    )   
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
    st.session_state.db = db
    st.session_state.is_loading = True
    print("Loading LLM")
    st.session_state.llm, response = get_llm(model_name= llm)
    if response:
        st.session_state.is_loading = False
        print("Init response", response)
    # Load retrievers
    
    st.session_state.retriever = st.session_state.db.as_retriever(
        search_type="similarity", search_kwargs={"k": 6})

    st.session_state.session_id = 0
    
    st.session_state.history_aware_chain = get_history_aware_chain(
        st.session_state.llm, st.session_state.retriever)
    
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(model, file_path=SAMPLE_PATH ,routes=[productRoute,chitchatRoute])

def main():
    if "history" not in st.session_state:
        st.session_state.history = []

    st.set_page_config(page_title="ChatBot", page_icon="🧊", layout="wide")
    st.title("ChatBot")

    for message in st.session_state.history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)
    user_query = st.chat_input(
        "Your question", key="user_query", disabled=st.session_state.is_loading)
    
    history = []
    if user_query is not None and user_query != "":
        with st.chat_message("Human"):
            st.markdown(user_query)
        with st.chat_message("AI"):
            response_placeholder = st.empty()
        query = str(user_query)
        query = query.lower()
        response = []
        context = None
        st.session_state.history.append(HumanMessage(user_query))
        guidedRoute = semanticRouter.guide(query)[0]

        if guidedRoute >= 0.12:
            
            history.append("user:" + str(user_query))
            for segment in chain_query(st.session_state.llm, st.session_state.history_aware_chain, user_query, str(st.session_state.session_id)):
                if "answer" in segment:
                    answer_text = segment["answer"]
                    response.append(answer_text)
                    response_placeholder.write("".join(response))
                if context is None and "context" in segment:
                    context = segment["context"]
        else:
            with st.chat_message("AI"):
                answer_text = "Xin lỗi, tôi không có chức năng để phục vụ câu hỏi này"
                response.append(answer_text)
                response_placeholder.write("".join(response))
                # Assuming AIMessage is defined somewhere in your code
                # from your_module import AIMessage
                
                response = []
                
                with st.spinner("Generating response..."):
                    answer_text = "Xin lỗi, tôi không có chức năng để phục vụ câu hỏi này"
                    response.append(answer_text)
                    response_placeholder.write("".join(response))
                    
        st.session_state.history.append(AIMessage("".join(response)))
                    

if __name__ == "__main__":
    main()
