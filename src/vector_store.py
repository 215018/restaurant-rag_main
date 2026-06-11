# This module creates a vector store using FAISS to store embeddings of document chunks for efficient similarity search.
# Import necessary libraries for vector storage and embedding creation.
from pathlib import Path
from langchain_chroma import Chroma
from chunking import split_documents
from data_loader import load_all_documents
from embeddings import get_embedding_model

# Go up from src/vector_store.py to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Folder where Chroma will save the vector database.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
# This function creates a Chroma vector store from document chunks using the Hugging Face embedding model.
def create_vector_store(chunks):
    # Get the Hugging Face embedding model.
    embedding_model = get_embedding_model()
    # Create a Chroma vector database from documents.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store
# Main function to test the vector store creation process and print some information about the created vector store.
def main():
    # Load CSV and PDF documents.
    documents = load_all_documents()
    # Split documents into chunks.
    chunks = split_documents(documents)
    # Store chunks in Chroma.
    vector_store = create_vector_store(chunks)
    print("Chroma vector store created successfully")
    print(f"Total chunks stored: {len(chunks)}")
    print(f"Saved at: {CHROMA_DIR}")
# Run the main function when this script is executed directly.
if __name__ == "__main__":
    main()