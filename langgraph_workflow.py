import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# Import the RAG chain logic from our previous script
from rag_chain import get_rag_chain

# Load environment variables
load_dotenv()

# Define the state of the graph
class GraphState(TypedDict):
    question: str
    category: str
    answer: str

# --- Nodes ---

def classifier(state: GraphState):
    """Categorizes the query into products, returns, or general."""
    print("--- NODE: Classifier ---")
    question = state["question"]
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Classify the following user query into exactly one of these categories: 'products', 'returns', or 'general'.\n"
        "Query: {question}\n"
        "Category:"
    )
    
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"question": question}).strip().lower()
    
    # Handle potential variation in output
    if "products" in category:
        category = "products"
    elif "returns" in category:
        category = "returns"
    else:
        category = "general"
        
    print(f"Query classified as: {category}")
    return {"category": category}

def rag_responder(state: GraphState):
    """Uses RAG to answer product-related queries."""
    print("--- NODE: RAG Responder ---")
    question = state["question"]
    
    rag_chain = get_rag_chain()
    answer = rag_chain.invoke(question)
    
    return {"answer": answer}

def escalation(state: GraphState):
    """Returns an escalation message for unhandled queries."""
    print("--- NODE: Escalation ---")
    category = state["category"]
    
    if category == "returns":
        answer = "For return-related queries, please contact our support team at support@techgear.com or call 1-800-TECH-GEAR. Our returns process typically takes 5-7 business days."
    else:
        answer = "I'm sorry, I cannot handle this general query. Let me escalate this to a human agent who can assist you further."
        
    return {"answer": answer}

# --- Conditional Routing Logic ---

def route_query(state: GraphState) -> Literal["rag_responder", "escalation"]:
    """Routes to the appropriate node based on the category."""
    if state["category"] == "products":
        return "rag_responder"
    else:
        return "escalation"

# --- Build the Graph ---

def create_workflow():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("classifier", classifier)
    workflow.add_node("rag_responder", rag_responder)
    workflow.add_node("escalation", escalation)

    # Set Entry Point
    workflow.set_entry_point("classifier")

    # Add Conditional Edges
    workflow.add_conditional_edges(
        "classifier",
        route_query,
        {
            "rag_responder": "rag_responder",
            "escalation": "escalation"
        }
    )

    # Add Normal Edges
    workflow.add_edge("rag_responder", END)
    workflow.add_edge("escalation", END)

    # Compile the graph
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = create_workflow()
    
    test_queries = [
        "What is the price of the Wireless Earbuds Elite?",
        "I want to return my SmartWatch Pro X, how do I do that?",
        "Who is the CEO of Google?"
    ]
    
    for query in test_queries:
        print(f"\nUser Query: {query}")
        result = app.invoke({"question": query})
        print(f"Final Answer: {result['answer']}")
        print("="*50)
