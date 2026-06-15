# Redi Restaurant Menu Assistant

A restaurant menu chatbot built with **Retrieval-Augmented Generation (RAG)**.

The system helps customers ask natural language questions about menu items, prices, allergens, vegetarian options, vegan options, drinks, lunch menu items, and food recommendations.

This project uses structured CSV menu data, PDF menu information, ChromaDB retrieval, Hugging Face embeddings, Groq/Llama, and Streamlit.

## Live Demo

[Open the Streamlit App](https://restaurant-ragmain-lmule9hdhgfqdvgcrv2rzh.streamlit.app/)

## Project Objective

The objective of this project is to build an AI-powered restaurant assistant that can:

- Answer menu-related questions in natural language
- Recommend dishes based on customer preferences
- Support budget-aware menu search
- Explain allergen information when available
- Support vegetarian and vegan preferences
- Handle food categories such as chicken, fish, lamb, salad, drinks, beer, rice, soup, and bread
- Generate menu-grounded answers using RAG

## What Is RAG?

RAG means **Retrieval-Augmented Generation**.

Instead of allowing the language model to answer only from memory, the system first retrieves relevant information from the restaurant menu data. Then it sends that retrieved context to the language model to generate the final answer.

```text
User Question
      ↓
Retriever
      ↓
Relevant Menu Context
      ↓
Structured Filters
      ↓
Groq / Llama
      ↓
Final Customer Answer

This reduces hallucination and keeps answers grounded in the restaurant menu.

Project Architecture
CSV Menu + PDF Menu
        ↓
Data Cleaning
        ↓
LangChain Documents
        ↓
Chunking
        ↓
Hugging Face Embeddings
        ↓
ChromaDB Vector Store
        ↓
Retriever + Structured Filters
        ↓
RAG Chain
        ↓
Groq / Llama
        ↓
Streamlit Chatbot

Project Structure
restaurant-rag/
│
├── data/
│   ├── swagat_menu_raw.csv
│   ├── swagat_menu.csv
│   └── swagat_menu.pdf
│
├── src/
│   ├── clean_menu_csv.py
│   ├── data_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── streamlit_app.py
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md

Main Components

1. Data Cleaning
File:
src/clean_menu_csv.py
This file cleans the raw restaurant menu CSV and creates the final cleaned CSV used by the RAG system.
Main tasks:
Normalize boolean values
Clean missing values
Normalize allergen codes
Convert allergen codes into readable allergen names
Detect drinks
Fix special cases such as Kingfisher beer and Lassi
Create a search_text field for better retrieval

2. Data Loading
File:
src/data_loader.py
This file loads CSV and PDF menu files and converts them into LangChain Document objects.
Loaders used:
CSVLoader
PyPDFLoader
Each document contains:
page_content
metadata
Metadata helps identify whether the document came from the CSV menu or PDF menu.

3. Chunking
File:
src/chunking.py
This file prepares documents for vector search.
Chunking strategy:
CSV rows are kept complete because each row represents one menu item
PDF documents are split into smaller chunks because PDF text can be long and unstructured
Text splitter used:
RecursiveCharacterTextSplitter

4. Embeddings
File:
src/embeddings.py
Embedding model used:
sentence-transformers/all-MiniLM-L6-v2
Embeddings convert text into numerical vectors. This allows the system to search by meaning instead of only exact keywords.
Example:
"chicken curry"
"Butter Chicken"
"Chicken Tikka Masala"
These texts have similar meanings, so their embeddings will be close in the vector space.

5. Vector Store
File:
src/vector_store.py
Vector database used:
ChromaDB
The vector store saves menu chunks and their embeddings. During retrieval, ChromaDB finds the menu chunks most relevant to the customer question.

6. Retriever
File:
src/retriever.py
The retriever searches ChromaDB and applies structured filters.
It supports:
Query expansion
Price filtering
Vegetarian filtering
Vegan filtering
Allergy filtering
Meal type filtering
Drink and beer detection
Chicken, fish, lamb, duck, shrimp, paneer, salad, soup, bread, rice, lassi, and dessert detection
Spice safety notes
PDF fallback for non-strict menu questions
The retriever is important because it prevents the language model from guessing menu items.

7. RAG Chain
File:
src/rag_chain.py
LLM provider:
Groq
Model used:
llama-3.1-8b-instant
The RAG chain:
Receives the customer question
Calls the retriever
Gets filtered menu items
Builds a safe prompt
Sends menu context to Groq/Llama
Returns a customer-friendly answer

8. Streamlit App
File:
src/streamlit_app.py
The Streamlit app provides the web interface.

Features:
Chat interface
Chat history
Example questions
Sidebar
Clear chat button
Friendly AI waiter responses
Automatic vector store creation when needed

Example Questions
I want mild chicken under 12 euros and no nuts.
Show me vegetarian lunch menu items.
Do you have any fish dishes?
Do you have any beer?
Do you have any salad?
I am allergic to milk. What can I eat?

How To Run Locally
1. Clone the repository
git clone <your-repository-url>
cd restaurant-rag
2. Install dependencies with uv
uv sync
3. Add Groq API key
Create a .env file in the project root:
GROQ_API_KEY=your_api_key_here
4. Create the vector store
uv run python src/vector_store.py
5. Test the RAG chain
uv run python src/rag_chain.py
6. Run the Streamlit app
uv run python -m streamlit run src/streamlit_app.py

Deployment Notes
For Streamlit Community Cloud, add the Groq API key in app secrets using TOML format:
GROQ_API_KEY = "your_api_key_here"
Do not upload your .env file to GitHub.
Security
The API key should stay private.

Make sure .gitignore includes:
.env
.venv/
chroma_db/
__pycache__/
*.pyc
.DS_Store
Tech Stack
Python
LangChain
ChromaDB
Hugging Face Embeddings
sentence-transformers/all-MiniLM-L6-v2
Groq
Llama
Streamlit
pandas
uv

Key Features
Menu-grounded RAG chatbot
CSV and PDF menu support
Structured filtering
Allergy-aware responses
Budget-aware recommendations
Vegetarian and vegan support
Drink and beer detection
PDF fallback for missing menu information
Streamlit web interface

Summary
This project demonstrates how Retrieval-Augmented Generation can be used to build a practical restaurant AI assistant.
By combining cleaned menu data, embeddings, ChromaDB retrieval, structured filtering, Groq/Llama, and Streamlit, the chatbot can answer customer questions in a reliable and menu-grounded way.

