from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langgraph_workflow import create_workflow
import os

app = FastAPI(title="Product Knowledge Base Chatbot")

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the LangGraph workflow
workflow_app = create_workflow()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    category: str
    answer: str

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    try:
        # Invoke the LangGraph workflow
        result = workflow_app.invoke({"question": request.question})
        
        return QueryResponse(
            question=request.question,
            category=result["category"],
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
