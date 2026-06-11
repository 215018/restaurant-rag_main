# Swagat Restaurant RAG Chatbot

## Project Goal

This project is a Retrieval-Augmented Generation (RAG) chatbot for Swagat Indian Restaurant.

The chatbot helps customers ask questions about the restaurant menu. Customers can ask about dish prices, ingredients, allergens, vegetarian options, vegan options, lunch menu items, drinks, and food preferences.

The main goal is to give accurate menu-based answers instead of guessing from general AI knowledge.

## Dataset

This project uses two menu data sources:

1. `swagat_menu.csv`

This is the structured menu dataset. It contains menu item names, prices, categories, meal types, vegetarian/vegan information, allergen codes, allergen names, and searchable menu text.

2. `swagat_menu.pdf`

This is the original restaurant menu PDF. It gives extra menu context and supports the knowledge base.

## How The System Works

```text
User question
   ↓
Intent extraction
   ↓
Retriever searches ChromaDB
   ↓
Structured filters check price, allergens, food type, lunch menu, and drinks
   ↓
Groq + Llama generate a customer-friendly English answer

Main Features
Answers customer questions using restaurant menu data
Retrieves relevant menu items using semantic search
Filters dishes by price
Handles allergy-related questions
Supports vegetarian and vegan preferences
Supports lunch menu filtering
Supports drink and beer questions
Generates clear English answers using an LLM
Provides allergy safety warnings when information is missing

Technologies Used
Python
LangChain
HuggingFace Embeddings
ChromaDB
Groq
Llama
Streamlit
uv

Project Structure
restaurant-rag/
│
├── data/
│   ├── swagat_menu.csv
│   ├── swagat_menu_raw.csv
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
├── chroma_db/
├── .env
├── pyproject.toml
├── uv.lock
└── README.md

File Explanation
clean_menu_csv.py
This file cleans the raw restaurant menu CSV. It normalizes missing values, fixes drink rows, converts allergen codes into readable allergen names, and creates the final swagat_menu.csv.

data_loader.py
This file loads the cleaned CSV and PDF menu into LangChain Document format. These documents become the knowledge base for the RAG system.

chunking.py
This file prepares documents for retrieval. CSV rows are kept complete so important fields like price and allergens stay together. PDF pages are split into smaller chunks.

embeddings.py
This file converts menu text into embeddings. Embeddings are number representations of text that help the system understand meaning.

vector_store.py
This file stores the embeddings in ChromaDB. ChromaDB is the vector database used to search for relevant menu items.

retriever.py
This file retrieves relevant menu chunks and applies structured filters. It extracts customer intent such as budget, allergy, vegetarian preference, lunch menu request, or beer request.

rag_chain.py
This file connects the retriever with the Groq Llama language model. It creates the final customer-friendly answer.

streamlit_app.py
This file creates the web interface where customers can chat with the AI waiter.

How To Run The Project
1. Install dependencies
uv sync
2. Create the vector database
uv run python src/vector_store.py
3. Run the Streamlit app
uv run streamlit run src/streamlit_app.py
Environment Variables
Create a .env file in the main project folder.

GROQ_API_KEY=your_groq_api_key_here
Do not share this key publicly.

Example Questions
I want mild chicken under 12 euros and no nuts
Show me vegetarian lunch menu items
I am allergic to milk. What can I eat?
Do you have Indian beer?
What vegan dishes do you have?
Which dishes are under 10 euros?

Safety Note
This chatbot helps customers understand the restaurant menu, but allergy-sensitive answers should always be confirmed directly with the restaurant before ordering.

Conclusion
This project demonstrates an advanced restaurant RAG system. It combines structured menu data, PDF context, semantic retrieval, rule-based filtering, and LLM answer generation to help customers choose suitable menu items.

