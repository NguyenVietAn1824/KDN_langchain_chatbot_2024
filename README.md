## Build a RAG application using Langchain with advanced skills

## Introduction

This project implements the Retrieval-Augmented Generation (RAG) model using Langchain. The goal of this application is to provide an application that improves the retrieval performance of a dataset of law questions, contributing to providing answers quickly and efficiently. In this project, I have used some advanced techniques such as semantic router to analyze the semantics of questions, adding historical reflection to improve the flow of user conversations, and combining retrieval techniques to improve retrieval efficiency. The LLM model used in this project is google-gemma2b to perform answer generation.

## Features
- **Langchain**: A framework that simplifies working with language models and integrates multiple components like retrieval and generation.
- **Retrieval-Augmented Generation (RAG)**: A machine learning technique that enhances text generation by retrieving relevant information from an external knowledge base.
- **Semantic router** : With Semantic Router, I use a set of available samples, representing the topic of the rule, from which I calculate the similarity of the question with this data set, and from there make a decision about the question type, reduce the need to include distracting questions in the LLM.
- **History reflection** : Improving query answering.
- **Streamlit framework** : A Framework for building interfaces.
- **Google-gemma2b** : Model to perform answer generation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project-name.git

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
6. Run Project
   ```bash
   streamlit run app.py
