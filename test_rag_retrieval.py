import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

def test_retrieval():
    # 1. Initialize Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Load the vector database
    persist_directory = "/home/labuser/assessment/chroma_db"
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # 3. Perform a similarity search
    query = "What is the price of the Wireless Earbuds Elite?"
    results = vector_db.similarity_search(query, k=1)

    print(f"Query: {query}")
    for i, res in enumerate(results):
        print(f"Result {i+1}:")
        print(res.page_content)
        print("-" * 20)

if __name__ == "__main__":
    test_retrieval()
