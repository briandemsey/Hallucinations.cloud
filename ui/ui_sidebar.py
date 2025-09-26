# ui/sidebar.py
"""
Sidebar UI Component for Hallucinations.cloud
Handles user dashboard, model suggestions, and relevant queries
"""

import streamlit as st
from config.settings import get_model_list, get_relevant_queries
from auth.authentication import auth_system

def render_sidebar():
    """Render the complete sidebar with all components"""
    
    # User Dashboard Section
    render_user_dashboard()
    
    st.sidebar.divider()
    
    # Model Suggestion Section
    render_model_suggestion()
    
    st.sidebar.divider()
    
    # Available Models Section
    render_available_models()
    
    st.sidebar.divider()
    
    # Relevant Queries Section
    render_relevant_queries()
    
    st.sidebar.divider()
    
    # Advanced Analysis Controls
    render_advanced_controls()

def render_user_dashboard():
    """Render user authentication and dashboard section"""
    auth_system.show_user_sidebar()

def render_model_suggestion():
    """Render model suggestion form"""
    st.sidebar.markdown("### 💡 Suggest an Additional LLM")
    
    suggested_model = st.sidebar.text_input(
        "Suggest a Model", 
        placeholder="Model name...", 
        key="suggest_model"
    )
    
    user_name = st.session_state.get('username', '')
    user_name_input = st.sidebar.text_input(
        "Your Name", 
        key="user_name", 
        value=user_name
    )
    
    if st.sidebar.button("Send Suggestion", use_container_width=True):
        if suggested_model and user_name_input:
            # Store suggestion in session state for potential backend processing
            if "model_suggestions" not in st.session_state:
                st.session_state.model_suggestions = []
            
            suggestion = {
                "model": suggested_model,
                "user": user_name_input,
                "timestamp": st.session_state.get("app_start_time", 0)
            }
            st.session_state.model_suggestions.append(suggestion)
            
            st.sidebar.success("✅ Suggestion submitted!")
            
            # Could integrate with backend here:
            # submit_model_suggestion(suggested_model, user_name_input)
        else:
            st.sidebar.warning("Please fill in both fields")

def render_available_models():
    """Render list of available models"""
    st.sidebar.markdown("### 🧠 Models in Use")
    
    models_list = get_model_list()
    for model in models_list:
        st.sidebar.markdown(f"- {model}")
    
    # Show model count
    st.sidebar.caption(f"Total: {len(models_list)} AI models")

def render_relevant_queries():
    """Render relevant queries section with selection"""
    st.sidebar.markdown("### 🎯 Relevant Queries")
    
    relevant_queries = get_relevant_queries()
    
    selected_relevant_query = st.sidebar.selectbox(
        "Choose a relevant query:",
        ["Select a query..."] + relevant_queries,
        key="relevant_query_selector"
    )
    
    if st.sidebar.button("Use This Query", key="use_relevant_query", use_container_width=True):
        if selected_relevant_query != "Select a query...":
            # Set the query to be used in the main interface
            st.session_state.auto_query = selected_relevant_query
            st.session_state.execute_query = True
            st.rerun()
        else:
            st.sidebar.warning("Please select a query first")

def render_advanced_controls():
    """Render advanced analysis controls"""
    st.sidebar.markdown("### 🛡️ Advanced Analysis")
    
    # Advanced analysis toggle
    if st.sidebar.button(
        "🎯 Run Red/Blue/Purple Team Analysis", 
        key="adv_analysis", 
        help="Advanced security team analysis",
        use_container_width=True
    ):
        st.session_state.show_advanced_analysis = True
    
    # Close advanced analysis if open
    if st.session_state.get("show_advanced_analysis", False):
        if st.sidebar.button(
            "✖️ Close Advanced Analysis", 
            key="close_adv_analysis",
            use_container_width=True
        ):
            st.session_state.show_advanced_analysis = False
    
    # Performance monitoring toggle
    show_performance = st.sidebar.checkbox(
        "📊 Show Performance Metrics",
        value=st.session_state.get("show_performance", False),
        help="Display API response times and performance warnings"
    )
    st.session_state.show_performance = show_performance
    
    # Debug mode toggle (for development)
    if st.session_state.get('username') == 'admin' or st.session_state.get('is_demo', False):
        debug_mode = st.sidebar.checkbox(
            "🔧 Debug Mode",
            value=st.session_state.get("debug_mode", False),
            help="Show additional debugging information"
        )
        st.session_state.debug_mode = debug_mode

def render_performance_metrics():
    """Render performance metrics if enabled"""
    if st.session_state.get("show_performance", False):
        st.sidebar.markdown("### ⚡ Performance")
        
        # App startup time
        if "app_start_time" in st.session_state:
            import time
            uptime = time.time() - st.session_state.app_start_time
            st.sidebar.caption(f"Uptime: {uptime:.1f}s")
        
        # Session state size (for debugging)
        if st.session_state.get("debug_mode", False):
            state_size = len(str(st.session_state))
            st.sidebar.caption(f"Session: {state_size} chars")
        
        # Model availability
        from config.settings import get_api_key_status
        api_status = get_api_key_status()
        available_count = sum(1 for status in api_status.values() if "✅" in status)
        total_count = len(api_status)
        st.sidebar.caption(f"APIs: {available_count}/{total_count} available")

def render_session_info():
    """Render session information for debugging"""
    if st.session_state.get("debug_mode", False):
        st.sidebar.markdown("### 🔧 Debug Info")
        
        # Authentication status
        auth_status = "✅ Authenticated" if st.session_state.get('authenticated') else "❌ Not authenticated"
        st.sidebar.caption(f"Auth: {auth_status}")
        
        # User type
        if st.session_state.get('is_demo'):
            st.sidebar.caption("Type: Demo User")
        elif st.session_state.get('username'):
            st.sidebar.caption(f"Type: {st.session_state.get('subscription_tier', 'unknown')}")
        
        # Recent queries count
        if "latest_query_results" in st.session_state:
            query_count = len(st.session_state.latest_query_results)
            st.sidebar.caption(f"Last query: {query_count} models")

# Call performance metrics at the end of sidebar rendering
def finalize_sidebar():
    """Finalize sidebar with performance metrics and debug info"""
    render_performance_metrics()
    render_session_info()

# Auto-call finalization when this module is imported in main render
def render_complete_sidebar():
    """Complete sidebar rendering including all sections"""
    render_sidebar()
    finalize_sidebar()
