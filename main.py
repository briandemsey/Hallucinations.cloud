# main.py - H-LLM API for hallucinations.cloud
"""
FastAPI-only implementation for H-LLM Multi-Model Comparison
Optimized for deployment on Render
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai
import cohere

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="H-LLM Multi-Model API",
    description="Compare responses across multiple LLMs",
    version="1.0.0"
)

# Configure CORS - CRITICAL for GoDaddy integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hallucinations.cloud",
        "https://www.hallucinations.cloud",
        "http://hallucinations.cloud",  # if needed
        "http://localhost:*",  # for local testing
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Get API keys
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
grok_key = os.getenv("GROK_API_KEY")
perplexity_key = os.getenv("PERPLEXITY_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# Setup clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None
anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
cohere_client = cohere.Client(cohere_key) if cohere_key else None

if google_key:
    genai.configure(api_key=google_key)

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class ModelResponse(BaseModel):
    model: str
    text: str

# Model calling functions
def call_openai(prompt: str) -> tuple:
    if not openai_client:
        return ("OpenAI", "[OpenAI unavailable: missing API key]")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return ("OpenAI", response.choices[0].message.content.strip())
    except Exception as e:
        return ("OpenAI", f"[OpenAI error: {str(e)}]")

def call_claude(prompt: str) -> tuple:
    if not anthropic_client:
        return ("Claude", "[Claude unavailable: missing API key]")
    try:
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return ("Claude", message.content[0].text.strip())
    except Exception as e:
        return ("Claude", f"[Claude error: {str(e)}]")

def call_gemini(prompt: str) -> tuple:
    if not google_key:
        return ("Gemini", "[Gemini unavailable: missing API key]")
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return ("Gemini", response.text.strip())
    except Exception as e:
        return ("Gemini", f"[Gemini error: {str(e)}]")

def call_cohere(prompt: str) -> tuple:
    if not cohere_client:
        return ("Cohere", "[Cohere unavailable: missing API key]")
    try:
        response = cohere_client.chat(
            message=prompt,
            model='command-r',
            max_tokens=500,
            temperature=0.5
        )
        return ("Cohere", response.text.strip())
    except Exception as e:
        return ("Cohere", f"[Cohere error: {str(e)}]")

def call_deepseek(prompt: str) -> tuple:
    if not deepseek_key:
        return ("Deepseek", "[Deepseek unavailable: missing API key]")
    try:
        deepseek_client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com"
        )
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return ("Deepseek", response.choices[0].message.content.strip())
    except Exception as e:
        return ("Deepseek", f"[Deepseek error: {str(e)}]")

def call_openrouter(prompt: str) -> tuple:
    if not openrouter_key:
        return ("OpenRouter", "[OpenRouter unavailable: missing API key]")
    try:
        openrouter_client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1"
        )
        response = openrouter_client.chat.completions.create(
            model="microsoft/wizardlm-2-8x22b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return ("OpenRouter", response.choices[0].message.content.strip())
    except Exception as e:
        return ("OpenRouter", f"[OpenRouter error: {str(e)}]")

def call_perplexity(prompt: str) -> tuple:
    return ("Perplexity", "[Perplexity temporarily disabled - API configuration needs verification]")

def call_grok(prompt: str) -> tuple:
    if not grok_key:
        return ("Grok", "[Grok unavailable: missing API key]")
    try:
        grok_client = OpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )
        response = grok_client.chat.completions.create(
            model="grok-2-1212",
            messages=[
                {"role": "system", "content": "You are Grok: witty, direct, and accurate with a touch of humor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return ("Grok", response.choices[0].message.content.strip())
    except Exception as e:
        return ("Grok", f"[Grok error: {str(e)}]")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "H-LLM Multi-Model API is running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "compare": "/compare (POST)",
            "models": "/models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": sum([
            1 if openai_key else 0,
            1 if anthropic_key else 0,
            1 if google_key else 0,
            1 if cohere_key else 0,
            1 if deepseek_key else 0,
            1 if openrouter_key else 0,
            1 if grok_key else 0,
        ])
    }

@app.get("/models")
async def get_models():
    """Get list of available models"""
    return {
        "models": {
            "openai": bool(openai_key),
            "claude": bool(anthropic_key),
            "gemini": bool(google_key),
            "cohere": bool(cohere_key),
            "deepseek": bool(deepseek_key),
            "openrouter": bool(openrouter_key),
            "grok": bool(grok_key),
            "perplexity": False  # Temporarily disabled
        }
    }

@app.post("/compare")
async def compare_llms(request: QueryRequest) -> List[ModelResponse]:
    """Compare responses from multiple LLMs"""
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = []
    
    # Map of available models
    model_functions = [
        (openai_key, call_openai),
        (anthropic_key, call_claude),
        (google_key, call_gemini),
        (cohere_key, call_cohere),
        (deepseek_key, call_deepseek),
        (openrouter_key, call_openrouter),
        (grok_key, call_grok),
        # (perplexity_key, call_perplexity),  # Disabled
    ]
    
    # Call each available model
    for api_key, model_func in model_functions:
        if api_key:
            model_name, response_text = model_func(request.query)
            results.append(ModelResponse(model=model_name, text=response_text))
    
    if not results:
        raise HTTPException(
            status_code=503,
            detail="No models available. Please configure at least one API key."
        )
    
    return results

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)