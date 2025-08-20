# NullClass Smart Assistant 🤖

This project is a **chatbot developed during my internship at NullClass EdTech**. The chatbot uses **LangChain** and **Google Gemini AI** to provide context-aware answers from FAQs and internal documentation. A **Streamlit frontend** was built to create a working prototype, which was presented to senior engineers for review.

## Features
- Context-aware Q&A using a **Retrieval-Augmented Generation (RAG) pipeline**.
- Handles greetings and politely responds to questions outside the dataset scope.
- Automated email alerts for unresolved queries.
- Interactive prototype demonstrating the chatbot capabilities for senior engineer evaluation.

## Tech Stack
- Python  
- Streamlit  
- LangChain (RAG pipeline, Runnable components)  
- Google Gemini AI (`gemini-1.5-flash`)  
- FAISS vector store  
- HuggingFace Instructor embeddings  
- Gmail API for automated notifications  

## Project Structure
- app.py # Streamlit frontend and chat handling
- langchain_helper.py # RAG chain and email sending functions
- dataset.csv # FAQ dataset
- faiss_index/ # FAISS vector store
- credentials.json # Gmail API credentials (excluded)
- token.json # Gmail API token (excluded)
-.env # Environment variables

  ## Notes
- Implemented as a **prototype for senior engineer review**.
- Demonstrates **RAG pipeline, vector database retrieval, and Streamlit deployment**.
- All sensitive credentials and tokens are excluded from the repository.
- Shows real-world application of AI/ML for customer support automation.
