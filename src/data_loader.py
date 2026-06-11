# This module is responsible for loading CSV and PDF files from the data folder and converting them into LangChain Documents.
# Import necessary libraries for file handling and document loading.
from pathlib import Path
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
# Go up from src/data_loader.py to the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder where CSV and PDF files are stored.
DATA_DIR = PROJECT_ROOT / "data"

# Find all CSV and PDF files in the data folder and return their paths.
def find_data_files():
    # Use only the cleaned CSV file for RAG.
    csv_files = [DATA_DIR / "swagat_menu.csv"]
    # Find all PDF files in the data folder.
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    return csv_files, pdf_files

# Load a single CSV file and convert it into a list of LangChain Documents.
def load_one_csv(csv_path):
    # CSVLoader converts each CSV row into one LangChain Document.
    loader = CSVLoader(
        file_path=str(csv_path),
        encoding="utf-8",
    )
    documents = loader.load()
    # Add extra metadata so we know this document came from CSV.
    for doc in documents:
        doc.metadata["source_type"] = "csv_menu"
        doc.metadata["file_name"] = csv_path.name
    return documents

# Load a single PDF file and convert it into a list of LangChain Documents.
def load_one_pdf(pdf_path):
    # PyPDFLoader converts each PDF page into one LangChain Document.
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    # Add extra metadata so we know this document came from PDF.
    for doc in documents:
        doc.metadata["source_type"] = "pdf_menu"
        doc.metadata["file_name"] = pdf_path.name
    return documents

# Load all CSV and PDF files from the data folder and convert them into a single list of LangChain Documents.
def load_all_documents():
    # Find available CSV and PDF files.
    csv_files, pdf_files = find_data_files()
    # Store all loaded documents here.
    all_documents = []
    # Load every CSV file.
    for csv_file in csv_files:
        csv_documents = load_one_csv(csv_file)
        all_documents.extend(csv_documents)
    # Load every PDF file.
    for pdf_file in pdf_files:
        pdf_documents = load_one_pdf(pdf_file)
        all_documents.extend(pdf_documents)
    return all_documents

# Main function to test the data loading process and print some information about the loaded documents.
def main():
    documents = load_all_documents()
    print("Data loading completed")
    print(f"Total documents loaded: {len(documents)}")
    if documents:
        print("\nFirst document content:")
        print(documents[0].page_content[:1000])
        print("\nFirst document metadata:")
        print(documents[0].metadata)

if __name__ == "__main__":
    main()