import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def ingest_data():
    # 1. Load the document
    file_path = "/home/labuser/assessment/product_details.txt"
    loader = TextLoader(file_path)
    documents = loader.load()

    # 2. Split into chunks
    # Since product_details.txt is structured with products, let's split by double newline first
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 4. Store in ChromaDB
    persist_directory = "/home/labuser/assessment/chroma_db"
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    # Chroma in recent versions automatically persists, but we can call it if needed depending on version
    # vector_db.persist() # Not needed for langchain-community >= 0.0.1
    
    print(f"Stored {len(chunks)} chunks in ChromaDB at {persist_directory}")

if __name__ == "__main__":
    ingest_data()
