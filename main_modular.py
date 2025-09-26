#!/usr/bin/env python3
"""
Hallucinations.cloud - Modular Architecture Version
Multi-Model AI Comparison Platform with H-Score Analysis

This is the clean, modular version of the application.
For production, use Hallucinations_9_23.py until migration is complete.
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modular components
from config.settings import initialize_configuration, get_api_clients
from auth.authentication import StreamlitHLLMAuth
from models.ai_models import AIModelsManager
from analysis.hscore import calculate_h_score
from ui.sidebar import render_sidebar
from ui.main_interface import render_main_interface
from ui.results_display import render_results
from utils.performance import initialize_session_state

def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title="Hallucinations.cloud",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize configuration
    initialize_configuration()

    # Initialize session state
    initialize_session_state()

    # Authentication check
    auth_system = StreamlitHLLMAuth()
    if not auth_system.is_authenticated():
        auth_system.show_authentication_ui()
        return

    # Main application interface
    st.title("🧠 Hallucinations.cloud")
    st.markdown("**Multi-Model AI Comparison Platform**")

    # Sidebar
    with st.sidebar:
        render_sidebar()

    # Main interface
    with st.container():
        query_results = render_main_interface()

        if query_results:
            # Display results
            render_results(query_results)

if __name__ == "__main__":
    main()