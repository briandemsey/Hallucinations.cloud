# main.py - Minimal FastAPI backend for testing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hallucinations.cloud",
        "https://www.hallucinations.cloud",
        "*"  # Allow all during testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class ModelResponse(BaseModel):
    model: str
    text: str

@app.get("/")
async def root():
    return {"message": "H-LLM API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models_available": 1}

@app.post("/compare")
async def compare(request: QueryRequest) -> List[ModelResponse]:
    # Temporary test response
    return [
        ModelResponse(model="Test Model", text=f"This is a test response for: {request.query}")
    ]
