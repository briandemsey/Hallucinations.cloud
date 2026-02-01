"""
H-LLM Multi-Model REST API
FastAPI backend for iOS app
Endpoints: /api/auth/send-otp, /api/auth/verify-otp, /api/query, /api/upload, /api/conversation/*
"""
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="H-LLM Multi-Model API",
    description="Multi-model AI comparison platform with hallucination detection",
    version="2.1.0"
)

# CORS middleware for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import modules
from app.auth import send_otp, verify_otp_code
from app.ai_models import query_all_models
from app.analysis import calculate_h_score, run_team_analysis, perform_contradiction_analysis
from app.web_search import get_web_search_context, is_web_search_available
from app.truth_verification import verify_responses
from app.file_processor import extract_text_from_file, validate_file, get_supported_extensions
from app.conversation import (
    create_conversation, add_message, get_history, get_conversation,
    format_context_for_query, export_conversation, conversation_exists
)


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
    enable_contradiction: bool = True
    show_metadata: bool = True
    # New parameters
    enable_web_search: bool = True
    enable_truth_verification: bool = True
    conversation_id: Optional[str] = None
    analysis_depth: Literal["quick", "standard", "comprehensive"] = "standard"
    file_context: Optional[str] = None  # Extracted text from uploaded file

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

class TruthScore(BaseModel):
    overall_score: float
    cross_reference_score: float
    temporal_accuracy: float
    confidence_level: str
    summary: str

class QueryResponse(BaseModel):
    success: bool
    responses: List[AIResponse]
    h_score: Optional[HScore] = None
    team_analysis: Optional[TeamAnalysis] = None
    contradiction_analysis: Optional[str] = None
    # New fields
    truth_score: Optional[TruthScore] = None
    conversation_id: Optional[str] = None
    web_search_used: bool = False

class FileUploadResponse(BaseModel):
    success: bool
    filename: str
    extracted_text: Optional[str] = None
    error: Optional[str] = None
    char_count: int = 0

class ConversationResponse(BaseModel):
    success: bool
    conversation_id: str

class ConversationHistoryResponse(BaseModel):
    success: bool
    conversation_id: str
    messages: List[Dict[str, Any]]

class ConversationExportResponse(BaseModel):
    success: bool
    conversation_id: str
    transcript: str


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


# ============= FILE UPLOAD ENDPOINT =============

@app.post("/api/upload", response_model=FileUploadResponse)
async def upload_file_endpoint(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Upload a file and extract text content.
    Supported formats: PDF, TXT, CSV, DOCX
    Max size: 5MB
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - missing token")

    try:
        # Read file content
        content = await file.read()

        # Validate file
        is_valid, error_msg = validate_file(file.filename, len(content))
        if not is_valid:
            return FileUploadResponse(
                success=False,
                filename=file.filename,
                error=error_msg
            )

        # Extract text
        success, result = extract_text_from_file(content, file.filename)

        if success:
            return FileUploadResponse(
                success=True,
                filename=file.filename,
                extracted_text=result,
                char_count=len(result)
            )
        else:
            return FileUploadResponse(
                success=False,
                filename=file.filename,
                error=result
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= CONVERSATION ENDPOINTS =============

@app.post("/api/conversation/new", response_model=ConversationResponse)
async def create_conversation_endpoint(
    authorization: Optional[str] = Header(None)
):
    """Create a new conversation and return its ID"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - missing token")

    try:
        conversation_id = create_conversation()
        return ConversationResponse(
            success=True,
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/{conversation_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history_endpoint(
    conversation_id: str,
    authorization: Optional[str] = Header(None)
):
    """Get the message history for a conversation"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - missing token")

    try:
        if not conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")

        history = get_history(conversation_id)
        return ConversationHistoryResponse(
            success=True,
            conversation_id=conversation_id,
            messages=history or []
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversation/{conversation_id}/export", response_model=ConversationExportResponse)
async def export_conversation_endpoint(
    conversation_id: str,
    authorization: Optional[str] = Header(None)
):
    """Export conversation as a formatted text transcript"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - missing token")

    try:
        if not conversation_exists(conversation_id):
            raise HTTPException(status_code=404, detail="Conversation not found")

        transcript = export_conversation(conversation_id)
        return ConversationExportResponse(
            success=True,
            conversation_id=conversation_id,
            transcript=transcript or ""
        )
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
    Query 8 AI models simultaneously, calculate H-Score, and run Red/Blue/Purple team analysis.

    Analysis depth options:
    - quick: No team analysis, faster response
    - standard: Current default behavior
    - comprehensive: All features + truth verification + web search

    Requires: Authorization header with JWT token
    """
    # TODO: Verify JWT token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized - missing token")

    try:
        # Build the full query with context
        full_query = request.query

        # Add file context if provided
        if request.file_context:
            full_query = f"DOCUMENT CONTENT:\n{request.file_context}\n\nUSER QUESTION: {request.query}"

        # Add conversation context if provided
        if request.conversation_id and conversation_exists(request.conversation_id):
            conv_context = format_context_for_query(request.conversation_id)
            if conv_context:
                full_query = f"{conv_context}\n{full_query}"
            # Save user message to conversation
            add_message(request.conversation_id, "user", request.query)

        # Get web search context if enabled
        web_context = None
        web_search_used = False
        if request.enable_web_search and (request.analysis_depth in ["standard", "comprehensive"]):
            if is_web_search_available():
                web_context = get_web_search_context(request.query)
                web_search_used = web_context is not None

        # Determine what to run based on analysis depth
        run_team_analyses = request.analysis_depth != "quick"
        run_contradiction = request.enable_contradiction and request.analysis_depth != "quick"
        run_truth = request.enable_truth_verification and request.analysis_depth == "comprehensive"

        # Query all 8 AI models in parallel
        model_responses = await query_all_models(
            query=full_query,
            enable_rag=request.enable_rag,
            show_metadata=request.show_metadata,
            web_context=web_context
        )

        # Run team analyses if enabled
        team_analysis = None
        if run_team_analyses and (request.enable_red_team or request.enable_blue_team or request.enable_purple_team):
            team_analysis = await run_team_analysis(
                query=request.query,
                responses=model_responses,
                enable_red=request.enable_red_team,
                enable_blue=request.enable_blue_team,
                enable_purple=request.enable_purple_team
            )

        # Run contradiction analysis if enabled
        contradiction_analysis = None
        if run_contradiction:
            # Run in thread pool to avoid blocking
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                contradiction_analysis = await loop.run_in_executor(
                    executor,
                    perform_contradiction_analysis,
                    request.query,
                    model_responses
                )

        # Run truth verification if enabled
        truth_score = None
        if run_truth:
            verification = verify_responses(request.query, model_responses)
            truth_score = TruthScore(
                overall_score=verification.get('overall_truth_score', 0.0),
                cross_reference_score=verification.get('cross_reference_score', 0.0),
                temporal_accuracy=verification.get('temporal_accuracy', 0.0),
                confidence_level=verification.get('confidence_level', 'medium'),
                summary=verification.get('verification_summary', '')
            )

        # Calculate H-Score
        h_score = calculate_h_score(
            responses=model_responses,
            red_analysis=team_analysis.get("red_team") if team_analysis else "",
            blue_analysis=team_analysis.get("blue_team") if team_analysis else "",
            purple_analysis=team_analysis.get("purple_team") if team_analysis else ""
        )

        # Create or use conversation ID
        response_conversation_id = request.conversation_id
        if not response_conversation_id:
            response_conversation_id = create_conversation()

        # Save assistant response to conversation (summary of all models)
        if conversation_exists(response_conversation_id):
            # Create a summary response from the first successful model
            for resp in model_responses:
                if not resp["response"].startswith("["):
                    add_message(
                        response_conversation_id,
                        "assistant",
                        resp["response"],
                        {"model": resp["model"], "h_score": h_score.get("final")}
                    )
                    break

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
            team_analysis=TeamAnalysis(**team_analysis) if team_analysis else None,
            contradiction_analysis=contradiction_analysis,
            truth_score=truth_score,
            conversation_id=response_conversation_id,
            web_search_used=web_search_used
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============= HEALTH CHECK =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "features": {
            "web_search": is_web_search_available(),
            "supported_files": get_supported_extensions()
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "H-LLM Multi-Model API",
        "version": "2.1.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
