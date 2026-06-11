# This script creates vector embeddings for document chunks using a pre-trained model from the sentence-transformers library.
from langchain_huggingface import HuggingFaceEmbeddings
from chunking import split_documents
from data_loader import load_all_documents
# This module is responsible for creating vector embeddings for document chunks using a pre-trained model from the sentence-transformers library.
def get_embedding_model():
    # Create a Hugging Face embedding model through LangChain.
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
# This function takes a list of document chunks and creates vector embeddings for each chunk using the embedding model.
def create_embeddings(chunks):
    # Get the embedding model.
    embedding_model = get_embedding_model()
    # Get only the text content from each chunk.
    texts = [chunk.page_content for chunk in chunks]
    # Convert chunk texts into numerical vectors.
    embeddings = embedding_model.embed_documents(texts)
    return embeddings
# Main function to test the embedding creation process and print some information about the created embeddings.
def main():
    # Load CSV and PDF documents.
    documents = load_all_documents()
    # Split documents into chunks.
    chunks = split_documents(documents)
    # Create embeddings for the chunks.
    embeddings = create_embeddings(chunks)
    print("Total chunks:", len(chunks))
    print("Total embeddings:", len(embeddings))
    print("Embedding size:", len(embeddings[0]))
# Run the main function when this script is executed directly.
if __name__ == "__main__":
    main()