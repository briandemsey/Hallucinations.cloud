"""
H-LLM Multi-Model REST API
FastAPI backend for iOS app
Endpoints: /api/auth/send-otp, /api/auth/verify-otp, /api/query
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="H-LLM Multi-Model API",
    description="Multi-model AI comparison platform with hallucination detection",
    version="2.0.0"
)

# CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import modules (to be created)
from app.auth import send_otp, verify_otp_code
from app.ai_models import query_all_models
from app.analysis import calculate_h_score, run_team_analysis


# ============= REQUEST/RESPONSE MODELS =============

class SendOTPRequest(BaseModel):
    phone_number: str = Field(..., description="Phone number in E.164 format (+1234567890)")

class SendOTPResponse(BaseModel):
    success: bool
    message: str

class VerifyOTPRequest(BaseModel):
    phone_number: str
    code: str = Field(..., min_length=6, max_length=6)

class VerifyOTPResponse(BaseModel):
    success: bool
    message: str
    tokens: Optional[Dict[str, str]] = None  # JWT tokens
    user: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    enable_rag: bool = True
    enable_red_team: bool = True
    enable_blue_team: bool = True
    enable_purple_team: bool = True
    show_metadata: bool = True

class AIResponse(BaseModel):
    model: str
    response: str
    metadata: Optional[Dict[str, Any]] = None

class TeamAnalysis(BaseModel):
    red_team: Optional[str] = None
    blue_team: Optional[str] = None
    purple_team: Optional[str] = None

class HScore(BaseModel):
    final: float
    safety: float
    trust: float
    confidence: float
    quality: float

class QueryResponse(BaseModel):
    success: bool
    responses: List[AIResponse]
    h_score: Optional[HScore] = None
    team_analysis: Optional[TeamAnalysis] = None


# ============= AUTHENTICATION ENDPOINTS =============

@app.post("/api/auth/send-otp", response_model=SendOTPResponse)
async def send_otp_endpoint(request: SendOTPRequest):
    """Send OTP verification code to phone number via Twilio SMS"""
    try:
        success = await send_otp(request.phone_number)

        if success:
            return SendOTPResponse(
                success=True,
                message="Verification code sent successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send verification code")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/verify-otp", response_model=VerifyOTPResponse)
async def verify_otp_endpoint(request: VerifyOTPRequest):
    """Verify OTP code and return JWT tokens"""
    try:
        result = await verify_otp_code(request.phone_number, request.code)

        if result["success"]:
            return VerifyOTPResponse(
                success=True,
                message="Login successful",
                tokens=result["tokens"],
                user=result["user"]
            )
        else:
            raise HTTPException(status_code=401, detail="Invalid verification code")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= QUERY ENDPOINT =============

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Query 8 AI models simultaneously, calculate H-Score, and run Red/Blue/Purple team analysis

    Requires: Authorization header with JWT token
    """
    # TODO: Verify JWT token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - missing token")

    try:
        # Query all 8 AI models in parallel
        model_responses = await query_all_models(
            query=request.query,
            enable_rag=request.enable_rag,
            show_metadata=request.show_metadata
        )

        # Run team analyses if enabled
        team_analysis = None
        if request.enable_red_team or request.enable_blue_team or request.enable_purple_team:
            team_analysis = await run_team_analysis(
                query=request.query,
                responses=model_responses,
                enable_red=request.enable_red_team,
                enable_blue=request.enable_blue_team,
                enable_purple=request.enable_purple_team
            )

        # Calculate H-Score
        h_score = calculate_h_score(
            responses=model_responses,
            red_analysis=team_analysis.get("red_team") if team_analysis else "",
            blue_analysis=team_analysis.get("blue_team") if team_analysis else "",
            purple_analysis=team_analysis.get("purple_team") if team_analysis else ""
        )

        return QueryResponse(
            success=True,
            responses=[
                AIResponse(
                    model=resp["model"],
                    response=resp["response"],
                    metadata=resp.get("metadata")
                )
                for resp in model_responses
            ],
            h_score=HScore(**h_score),
            team_analysis=TeamAnalysis(**team_analysis) if team_analysis else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= HEALTH CHECK =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "H-LLM Multi-Model API",
        "version": "2.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
