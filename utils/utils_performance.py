# utils/performance.py
"""
Performance optimization and session management utilities
Handles caching, session cleanup, and performance monitoring
"""

import streamlit as st
import time
from functools import lru_cache
from typing import Dict, Any

def initialize_session_state():
    """Initialize session state variables with optimization"""
    
    # Query tracking
    if "latest_query_results" not in st.session_state:
        st.session_state.latest_query_results = []
    if "latest_query_text" not in st.session_state:
        st.session_state.latest_query_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # Performance monitoring
    if "app_start_time" not in st.session_state:
        st.session_state.app_start_time = time.time()
    
    # UI state
    if "show_advanced_analysis" not in st.session_state:
        st.session_state.show_advanced_analysis = False
    if "show_performance" not in st.session_state:
        st.session_state.show_performance = False
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

def cleanup_session_state():
    """Clean up temporary session variables to prevent memory bloat"""
    temp_vars = [
        'temp_query_results',
        'temp_analysis_data', 
        'form_submission_temp',
        'validation_errors_temp',
        'email_code_sent',
        'sms_code_sent',
        'auto_query',  # One-time use variable
        'execute_query'  # One-time use variable
    ]
    
    for var in temp_vars:
        if var in st.session_state:
            del st.session_state[var]

@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_user_stats(user_id: str) -> Dict[str, Any]:
    """Cache user statistics to avoid repeated database calls"""
    try:
        # Import here to avoid circular imports
        from database import get_database
        db = get_database()
        return db.get_user_stats(user_id)
    except Exception as e:
        return {
            "total_queries": 0,
            "average_hscore": 0.0,
            "last_query_date": None,
            "error": str(e)
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def cached_subscription_limits(tier: str) -> Dict[str, Any]:
    """Cache subscription limits to avoid repeated lookups"""
    try:
        # Import here to avoid circular imports
        from database import get_database
        db = get_database()
        return db.get_subscription_limits(tier)
    except Exception as e:
        # Fallback limits
        limits = {
            "demo": {"daily_limit": 3, "features": ["basic"]},
            "free": {"daily_limit": 10, "features": ["basic", "history"]},
            "premium": {"daily_limit": 100, "features": ["basic", "history", "advanced"]},
            "pro": {"daily_limit": 1000, "features": ["basic", "history", "advanced", "api"]}
        }
        return limits.get(tier, limits["free"])

@st.cache_resource
def get_cached_database():
    """Cache database instance to avoid reconnections"""
    try:
        from database import get_database
        return get_database()
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

def performance_timer(operation_name: str):
    """Decorator for monitoring operation performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log slow operations (>1 second) if performance monitoring is enabled
            if (end_time - start_time > 1.0 and 
                st.session_state.get("show_performance", False)):
                st.sidebar.caption(f"⚠️ {operation_name}: {end_time - start_time:.1f}s")
            
            return result
        return wrapper
    return decorator

class PerformanceMonitor:
    """Performance monitoring utility class"""
    
    def __init__(self):
        self.timers = {}
        self.enabled = False
    
    def enable(self):
        """Enable performance monitoring"""
        self.enabled = True
        st.session_state.show_performance = True
    
    def disable(self):
        """Disable performance monitoring"""
        self.enabled = False
        st.session_state.show_performance = False
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        if self.enabled:
            self.timers[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if not self.enabled or operation not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[operation]
        del self.timers[operation]
        
        # Display warning for slow operations
        if duration > 1.0:
            st.sidebar.caption(f"⚠️ {operation}: {duration:.1f}s")
        
        return duration
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        stats = {}
        
        # App uptime
        if "app_start_time" in st.session_state:
            stats["uptime"] = time.time() - st.session_state.app_start_time
        
        # Session state size (rough estimate)
        stats["session_size"] = len(str(st.session_state))
        
        # Active timers
        stats["active_timers"] = len(self.timers)
        
        # Memory usage indicators
        stats["cached_items"] = len(st.cache_data.get_stats() if hasattr(st.cache_data, 'get_stats') else {})
        
        return stats

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def optimize_session_state():
    """Optimize session state by removing old/large items"""
    
    # Limit chat history to last 20 messages
    if "chat_history" in st.session_state and len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
    
    # Limit query history to last 50 queries
    if "query_history" in st.session_state and len(st.session_state.query_history) > 50:
        st.session_state.query_history = st.session_state.query_history[-50:]
    
    # Clean up old model suggestions
    if "model_suggestions" in st.session_state and len(st.session_state.model_suggestions) > 10:
        st.session_state.model_suggestions = st.session_state.model_suggestions[-10:]

def clear_caches():
    """Clear all Streamlit caches"""
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Caches cleared successfully")
    except Exception as e:
        st.error(f"❌ Failed to clear caches: {str(e)}")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring"""
    stats = {
        "cache_data_hits": 0,
        "cache_data_misses": 0,
        "cache_resource_items": 0