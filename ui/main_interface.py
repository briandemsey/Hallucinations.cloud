# ui/main_interface.py
"""
Main interface for Hallucinations.cloud
Query input and model selection interface
"""

import streamlit as st
from typing import Dict, Any, Optional
from models.ai_models import AIModelsManager
from auth.authentication import StreamlitHLLMAuth

def render_main_interface() -> Optional[Dict[str, Dict[str, Any]]]:
    """Render the main query interface"""

    st.header("🤖 Multi-Model AI Query")

    # Initialize managers
    ai_manager = AIModelsManager()
    auth_system = StreamlitHLLMAuth()

    # Check authentication and query limits
    if not auth_system.check_query_limit():
        user_info = auth_system.get_user_info()
        if user_info:
            st.error(f"Query limit reached! You have {user_info.get('queries_remaining', 0)} queries remaining.")
        return None

    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask a question that will be sent to all available AI models..."
    )

    # Model selection
    available_models = ai_manager.get_available_models()
    if not available_models:
        st.error("⚠️ No AI models available. Please check your API keys in the .env file.")
        return None

    st.write(f"**Available Models:** {', '.join(available_models)}")

    # Query execution
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        query_all = st.button("🚀 Query All Models", use_container_width=True, type="primary")

    with col2:
        selected_model = st.selectbox("Or query single model:", available_models)
        query_single = st.button("Query Single Model", use_container_width=True)

    with col3:
        user_info = auth_system.get_user_info()
        if user_info:
            st.metric("Queries Remaining", user_info.get("queries_remaining", 0))

    # Process queries
    if query and query_all:
        with st.spinner("Querying all models..."):
            results = ai_manager.query_all_models(query)
            auth_system.decrement_query_count()

            # Store results in session state
            st.session_state.last_results = results
            return results

    elif query and query_single and selected_model:
        with st.spinner(f"Querying {selected_model}..."):
            single_result = ai_manager.query_model(selected_model, query)
            auth_system.decrement_query_count()

            results = {selected_model: single_result}
            st.session_state.last_results = results
            return results

    # Show previous results if available
    if "last_results" in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Previous Results")
        return st.session_state.last_results

    return None