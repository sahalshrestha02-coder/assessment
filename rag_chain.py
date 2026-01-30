import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # 1. Initialize Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = "/home/labuser/assessment/chroma_db"
    
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # 2. Setup Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Setup LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 4. Define Prompt Template
    template = """Answer the question based only on the following context. 
If the answer is not in the context, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Build LCEL Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    chain = get_rag_chain()
    
    # Test Question
    question = "What are the features of the Power Bank Ultra?"
    print(f"Testing Query: {question}")
    response = chain.invoke(question)
    print("\nResponse:")
    print(response)
    
    print("-" * 30)
    
    # Out of context test
    question2 = "What is the capital of France?"
    print(f"Testing Query: {question2}")
    response2 = chain.invoke(question2)
    print("\nResponse:")
    print(response2)
