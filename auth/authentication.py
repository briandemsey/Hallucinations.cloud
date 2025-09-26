# auth/authentication.py
"""
Authentication system for Hallucinations.cloud
Handles user authentication, session management, and subscription checks
"""

import streamlit as st
import os
from typing import Optional, Dict, Any

class StreamlitHLLMAuth:
    """Simplified authentication system for modular architecture"""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize authentication-related session state"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "user_data" not in st.session_state:
            st.session_state.user_data = None
        if "subscription_tier" not in st.session_state:
            st.session_state.subscription_tier = "free"

    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get("authenticated", False)

    def show_authentication_ui(self):
        """Display authentication interface"""
        st.title("🧠 Hallucinations.cloud")
        st.markdown("### Welcome to Multi-Model AI Comparison Platform")

        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("#### Login to Continue")

                # For modular version, provide a demo login
                if st.button("🚀 Enter Demo Mode", use_container_width=True):
                    self._set_demo_session()
                    st.rerun()

                st.markdown("---")
                st.caption("Demo mode provides 5 queries for testing")
                st.caption("Production authentication is available in Hallucinations_9_23.py")

    def _set_demo_session(self):
        """Set up demo session for testing"""
        st.session_state.authenticated = True
        st.session_state.user_data = {
            "email": "demo@hallucinations.cloud",
            "user_id": "demo_user",
            "queries_remaining": 5
        }
        st.session_state.subscription_tier = "demo"

    def logout(self):
        """Logout user and clear session"""
        for key in ["authenticated", "user_data", "subscription_tier"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        return st.session_state.get("user_data")

    def check_query_limit(self) -> bool:
        """Check if user can make another query"""
        if not self.is_authenticated():
            return False

        user_data = self.get_user_info()
        if not user_data:
            return False

        queries_remaining = user_data.get("queries_remaining", 0)
        return queries_remaining > 0

    def decrement_query_count(self):
        """Decrease query count after successful query"""
        if self.is_authenticated() and "user_data" in st.session_state:
            current_count = st.session_state.user_data.get("queries_remaining", 0)
            if current_count > 0:
                st.session_state.user_data["queries_remaining"] = current_count - 1