# This module is responsible for splitting loaded documents into smaller chunks using LangChain's RecursiveCharacterTextSplitter.
# Import necessary libraries for text splitting and document loading.
from langchain_text_splitters import RecursiveCharacterTextSplitter
from data_loader import load_all_documents
# Create a text splitter and split documents into smaller chunks.
def split_documents(documents):
    # Keep CSV rows as complete documents.
    csv_documents = []

    # Split only PDF documents.
    pdf_documents = []

    for document in documents:
        if document.metadata.get("source_type") == "csv_menu":
            csv_documents.append(document)
        else:
            pdf_documents.append(document)
    # Create a text splitter for long PDF documents only.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    # Split PDF documents into chunks.
    pdf_chunks = splitter.split_documents(pdf_documents)
    # Combine complete CSV rows with PDF chunks.
    chunks = csv_documents + pdf_chunks
    return chunks
# Main function to test the document splitting process and print some information about the created chunks.
def main():
    # Load CSV and PDF documents.
    documents = load_all_documents()
    print("Original documents:", len(documents))
    # Split documents.
    chunks = split_documents(documents)
    print("Total chunks:", len(chunks))
    csv_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("source_type") == "csv_menu"
    ]
    pdf_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("source_type") == "pdf_menu"
    ]
    print("CSV chunks:", len(csv_chunks))
    print("PDF chunks:", len(pdf_chunks))
    if chunks:
        print("\nFirst chunk content:")
        print(chunks[0].page_content[:1200])

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)
# Run the main function when this script is executed directly.
if __name__ == "__main__":
    main()
