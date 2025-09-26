# utils/performance.py
"""
Performance utilities for Hallucinations.cloud
Session management, caching, and performance monitoring
"""

import streamlit as st
import time
from typing import Any, Dict

def initialize_session_state():
    """Initialize all session state variables"""

    # Authentication state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # User data
    if "user_data" not in st.session_state:
        st.session_state.user_data = None

    # UI preferences
    if "show_errors" not in st.session_state:
        st.session_state.show_errors = True

    if "show_tokens" not in st.session_state:
        st.session_state.show_tokens = False

    if "auto_hscore" not in st.session_state:
        st.session_state.auto_hscore = True

    if "show_contradictions" not in st.session_state:
        st.session_state.show_contradictions = True

    # Performance tracking
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    # Results cache
    if "results_cache" not in st.session_state:
        st.session_state.results_cache = {}

def log_performance(operation: str, start_time: float, **kwargs):
    """Log performance metrics"""

    duration = time.time() - start_time

    if "performance_log" not in st.session_state:
        st.session_state.performance_log = []

    st.session_state.performance_log.append({
        "operation": operation,
        "duration": duration,
        "timestamp": time.time(),
        **kwargs
    })

    # Keep only last 100 entries
    if len(st.session_state.performance_log) > 100:
        st.session_state.performance_log = st.session_state.performance_log[-100:]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_api_call(model_name: str, prompt: str, api_key_hash: str):
    """Cache API calls to reduce redundant requests"""
    # This function signature allows Streamlit to cache based on inputs
    # The actual API call should be implemented here
    return f"Cached response for {model_name}: {prompt[:50]}..."

def get_performance_stats() -> Dict[str, Any]:
    """Get current performance statistics"""

    if "performance_log" not in st.session_state:
        return {}

    logs = st.session_state.performance_log

    if not logs:
        return {}

    total_operations = len(logs)
    avg_duration = sum(log["duration"] for log in logs) / total_operations
    max_duration = max(log["duration"] for log in logs)

    return {
        "total_operations": total_operations,
        "average_duration": round(avg_duration, 3),
        "max_duration": round(max_duration, 3),
        "total_queries": st.session_state.get("query_count", 0),
        "total_tokens": st.session_state.get("total_tokens", 0)
    }

def clear_cache():
    """Clear all cached data"""

    # Clear Streamlit cache
    st.cache_data.clear()

    # Clear session state cache
    if "results_cache" in st.session_state:
        st.session_state.results_cache = {}

    # Clear performance logs
    if "performance_log" in st.session_state:
        st.session_state.performance_log = []

def display_performance_metrics():
    """Display performance metrics in sidebar"""

    stats = get_performance_stats()

    if not stats:
        return

    st.subheader("📈 Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Queries", stats.get("total_queries", 0))
        st.metric("Avg Response Time", f"{stats.get('average_duration', 0)}s")

    with col2:
        st.metric("Total Tokens", stats.get("total_tokens", 0))
        st.metric("Max Response Time", f"{stats.get('max_duration', 0)}s")

def optimize_session_state():
    """Clean up and optimize session state"""

    # Remove old cached results (keep only last 10)
    if "results_cache" in st.session_state:
        cache = st.session_state.results_cache
        if len(cache) > 10:
            # Keep only the 10 most recent
            sorted_items = sorted(cache.items(),
                                key=lambda x: x[1].get("timestamp", 0),
                                reverse=True)
            st.session_state.results_cache = dict(sorted_items[:10])

    # Clean up old performance logs
    if "performance_log" in st.session_state:
        logs = st.session_state.performance_log
        if len(logs) > 50:
            st.session_state.performance_log = logs[-50:]