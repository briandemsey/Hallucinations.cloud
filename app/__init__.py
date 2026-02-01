"""H-LLM Multi-Model API - Application Package"""

# Import all modules for easier access
from app.ai_models import query_all_models
from app.auth import send_otp, verify_otp_code
from app.analysis import calculate_h_score, run_team_analysis, perform_contradiction_analysis
from app.web_search import get_web_search_context, is_web_search_available
from app.truth_verification import TruthVerificationEngine, verify_responses
from app.file_processor import extract_text_from_file, validate_file, get_supported_extensions
from app.conversation import (
    create_conversation, add_message, get_history, get_conversation,
    format_context_for_query, export_conversation, conversation_exists
)
