from typing import List
from langchain.schema import Document
import time
import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from src.prompt import (CHATBOT_SYSTEM_PROMPT)
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

load_dotenv()

STORE_MSG_HISTORY = {}


def get_llm(model_name: str = "gemma2:2b", use_api: bool = False):

    llm = ChatOllama(model=model_name, temperature=0.1)
    messages = [
        ("system", "You are a helpful AI assistant."),
        ("human", "hi"),
    ]
    response_init = llm.invoke(messages)
    return llm, response_init


def format_docs(docs: List[Document]) -> str:
    """
    Concatenates the content of all retrieved documents into a single string.

    Args:
        docs (List[Document]): A list of Document objects, where each Document contains page content that needs to be concatenated.

    Returns:
        str: A single string that contains the content of all documents
    """
    return "\n\n".join(doc.page_content for doc in docs)


def chain_query(llm, history_retriever, user_question: str, session_id: str = "abc123"):

    system_prompt = (CHATBOT_SYSTEM_PROMPT)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(
        history_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    answer = conversational_rag_chain.stream({"input": user_question}, config={
        "configurable": {"session_id": session_id}
    })
    return answer


def get_history_aware_chain(llm, retriever):
    contextualize_q_system_prompt = """
    Dựa vào lịch sử trò chuyện và câu hỏi gần nhất của người dùng, có thể tham khảo đến ngữ cảnh trong cuộc trò chuyện trước đó, hãy chuyển đổi câu hỏi thành một câu hỏi độc lập, dễ hiểu mà không cần tham chiếu đến lịch sử.

    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in STORE_MSG_HISTORY:
        STORE_MSG_HISTORY[session_id] = ChatMessageHistory()
    return STORE_MSG_HISTORY[session_id]
