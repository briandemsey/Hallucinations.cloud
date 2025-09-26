# ui/main_interface.py
"""
Main Interface UI Component for Hallucinations.cloud
Handles query input, file uploads, model responses, and H-Score display
"""

import streamlit as st
import time
from typing import Dict, Any, List, Tuple

from config.settings import get_api_key_status
from auth.authentication import check_query_limits
from models.ai_models import AIModelsManager
from analysis.hscore_engine import hscore_engine
from utils.file_processor import FileProcessor
from ui.results_display import ResultsDisplay

def render_main_interface(api_clients: Dict[str, Any]):
    """Render the main application interface"""
    
    # Header and welcome message
    render_header()
    
    # API key status
    render_api_status()
    
    # File attachment section
    file_content, file_info = render_file_upload()
    
    # Main query interface
    render_query_interface(api_clients, file_content, file_info)
    
    # Follow-up conversation
    render_follow_up_conversation(api_clients)

def render_header():
    """Render application header and welcome message"""
    st.title("🧠 Hallucinations.cloud Multi-Model with H-Score & Database")
    st.info("This application is a beta prototype under active development. For suggestions or bug reports, contact support@hallucinations.cloud")
    
    # Welcome message for logged-in user
    if st.session_state.get('username'):
        user_type = "Demo User" if st.session_state.get('is_demo', False) else st.session_state.get('username')
        st.success(f"Welcome back, {user_type}! 👋")

def render_api_status():
    """Render API key status checker"""
    st.subheader("🔐 Environment Key Status Checker")
    
    api_status = get_api_key_status()
    
    # Split into two columns for better layout
    col1, col2 = st.columns(2)
    
    status_items = list(api_status.items())
    mid_point = len(status_items) // 2
    
    with col1:
        for key, status in status_items[:mid_point]:
            st.markdown(f"{key}: {status}")
    
    with col2:
        for key, status in status_items[mid_point:]:
            st.markdown(f"{key}: {status}")

def render_file_upload():
    """Render file attachment section and return file content"""
    st.subheader("📎 File Attachment (Optional)")
    
    # File upload component
    uploaded_file = st.file_uploader(
        "Attach a file to enhance your query with context",
        type=['txt', 'csv', 'xlsx', 'xls', 'json', 'md'],
        help="Upload documents or data files to include in your analysis"
    )
    
    file_content = ""
    file_info = {}
    processing_mode = "analyze"
    
    # Process uploaded file
    if uploaded_file is not None:
        file_processor = FileProcessor()
        file_content, file_info = file_processor.process_uploaded_file(uploaded_file)
        
        # Display file information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ File uploaded: {file_info['name']}")
            st.write(f"**Size:** {file_info['size'] / 1024:.1f} KB")
            st.write(f"**Type:** {file_info['type']}")
        
        with col2:
            processing_mode = st.selectbox(
                "Processing mode:",
                ["analyze", "summarize", "extract", "question"],
                format_func=lambda x: {
                    "analyze": "🔍 Full Analysis",
                    "summarize": "📄 Summary",
                    "extract": "💡 Key Points",
                    "question": "❓ Answer Question"
                }[x]
            )
        
        # Store processing mode for query enhancement
        file_info['processing_mode'] = processing_mode
    
    return file_content, file_info

def render_query_interface(api_clients: Dict[str, Any], file_content: str, file_info: Dict):
    """Render main query interface with model responses and H-Score analysis"""
    st.subheader("🔍 Compare LLMs with H-Score Analysis")
    
    # Check query limits before allowing input
    can_query = check_query_limits()
    
    if not can_query:
        return
    
    # Enhanced query input
    initial_query = ""
    if "auto_query" in st.session_state:
        initial_query = st.session_state.auto_query
        del st.session_state.auto_query
    
    # Show file context indicator
    if file_content and file_info:
        st.info(f"📎 File attached: {file_info['name']} - Your query will include this file's context")
    
    user_query = st.text_input(
        "Enter your question:", 
        value=initial_query, 
        placeholder="Ask something to compare across models..." + (" (file context will be included)" if file_content else "")
    )
    
    # Query execution
    auto_execute = st.session_state.get("execute_query", False)
    if auto_execute:
        st.session_state.execute_query = False
    
    if (st.button("Submit") and user_query) or (auto_execute and user_query):
        execute_query(user_query, api_clients, file_content, file_info)

def execute_query(user_query: str, api_clients: Dict[str, Any], file_content: str, file_info: Dict):
    """Execute query across all models and display results"""
    st.subheader("📊 Model Responses")
    
    # Prepare the query (with file context if available)
    final_query = user_query
    if file_content and file_info and not file_content.startswith("[Error"):
        file_processor = FileProcessor()
        processing_mode = file_info.get('processing_mode', 'analyze')
        final_query = file_processor.create_file_enhanced_prompt(
            user_query, file_content, file_info['name'], processing_mode
        )
        st.info(f"🔄 Query enhanced with {processing_mode} of {file_info['name']}")
    
    # Initialize models manager
    models_manager = AIModelsManager(api_clients)
    available_models = models_manager.get_available_models()
    
    if not available_models:
        st.error("No API keys available! Please set up at least one API key.")
        return
    
    # Execute queries with progress tracking
    results = execute_models_with_progress(models_manager, final_query, available_models)
    
    if not results:
        st.error("No valid responses received from models")
        return
    
    # Store results for advanced analysis
    store_query_results(user_query, final_query, file_info, results)
    
    # Display model responses
    display_model_responses(results, file_info)
    
    # H-Score analysis
    perform_hscore_analysis(user_query, results)
    
    # Traditional contradiction analysis
    perform_contradiction_analysis(user_query, results, file_info, api_clients)

def execute_models_with_progress(models_manager: AIModelsManager, query: str, available_models: List[str]) -> List[Tuple[str, str]]:
    """Execute models with progress bar and performance monitoring"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_models = len(available_models)
    
    for i, model_name in enumerate(available_models):
        display_name = models_manager._get_display_name(model_name)
        status_text.text(f"Querying {display_name}... ({i+1}/{total_models})")
        progress_bar.progress((i + 1) / total_models)
        
        try:
            start_time = time.time()
            result = models_manager._call_single_model(model_name, query)
            end_time = time.time()
            
            # Track slow models for performance monitoring
            if end_time - start_time > 3.0 and st.session_state.get("show_performance", False):
                st.sidebar.caption(f"🐌 {display_name}: {end_time - start_time:.1f}s")
            
            results.append(result)
            
        except Exception as e:
            display_name = models_manager._get_display_name(model_name)
            results.append((display_name, f"[Error: {str(e)}]"))
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

def store_query_results(user_query: str, final_query: str, file_info: Dict, results: List[Tuple[str, str]]):
    """Store query results in session state for advanced analysis"""
    st.session_state.latest_query_results = results
    st.session_state.latest_query_text = user_query
    st.session_state.latest_final_query = final_query
    st.session_state.latest_file_info = file_info if file_info else None

def display_model_responses(results: List[Tuple[str, str]], file_info: Dict):
    """Display model responses with file attachment indicators"""
    for model_name, response in results:
        st.markdown(f"**{model_name}** {'📎' if file_info else ''}")
        
        # Color code responses based on error status
        if response.startswith('[') and 'error' in response.lower():
            st.error(response)
        else:
            st.text_area(
                f"{model_name} response:", 
                value=response, 
                height=150, 
                key=f"response_{model_name}_{hash(response[:50])}"  # Unique key
            )

def perform_hscore_analysis(user_query: str, results: List[Tuple[str, str]]):
    """Perform and display H-Score analysis"""
    st.markdown("---")
    st.markdown("## 🎯 H-Score Reliability Analysis")
    st.caption("Advanced AI hallucination detection and reliability scoring")
    
    with st.spinner("🧠 Calculating H-Score..."):
        try:
            # Calculate H-Score with timing
            start_time = time.time()
            hscore_result = hscore_engine.calculate_h_score(user_query, results)
            end_time = time.time()
            
            # Show timing if slow and performance monitoring is enabled
            calc_time = end_time - start_time
            if calc_time > 1.0 and st.session_state.get("show_performance", False):
                st.sidebar.caption(f"🧠 H-Score: {calc_time:.1f}s")
            
            # Display results
            results_display = ResultsDisplay()
            results_display.display_hscore_results(hscore_result)
            
            # Save to database
            save_hscore_to_database(user_query, results, hscore_result)
            
            # Store H-Score result for advanced analysis
            st.session_state.latest_hscore_result = hscore_result
            
            return hscore_result
            
        except Exception as e:
            st.error(f"H-Score calculation failed: {str(e)}")
            st.info("H-Score analysis requires at least 2 valid model responses")
            return None

def save_hscore_to_database(user_query: str, results: List[Tuple[str, str]], hscore_result):
    """Save H-Score results to database with timing"""
    combined_response = "\n\n".join([f"{name}: {resp}" for name, resp in results])
    
    start_time = time.time()
    save_success = hscore_engine.save_hscore_query(user_query, combined_response, hscore_result)
    end_time = time.time()
    
    # Show database timing if slow and performance monitoring is enabled
    db_time = end_time - start_time
    if db_time > 1.0 and st.session_state.get("show_performance", False):
        st.sidebar.caption(f"💾 Database: {db_time:.1f}s")
    
    if save_success:
        st.success("✅ Query saved to your history!")
    else:
        st.warning("⚠️ Could not save query to database")

def perform_contradiction_analysis(user_query: str, results: List[Tuple[str, str]], file_info: Dict, api_clients: Dict[str, Any]):
    """Perform traditional contradiction analysis"""
    if "openai" not in api_clients:
        return
    
    st.subheader("⚖️ Traditional Contradiction Analysis")
    with st.spinner("Analyzing for contradictions..."):
        model_responses = "\n\n".join([f"{name}: {resp}" for name, resp in results])
        
        contradiction_prompt = f"""
        Analyze these AI model responses for contradictions or significant disagreements:
        
        Original Query: {user_query}
        {'File Context: Analysis of ' + file_info['name'] if file_info else 'No file attached'}
        
        {model_responses}
        
        Provide a brief analysis of any contradictions found, or confirm if responses are generally consistent.
        {'Pay attention to how models interpreted the file data.' if file_info else ''}
        """
        
        try:
            models_manager = AIModelsManager(api_clients)
            contradiction_analysis = models_manager.call_model_for_analysis(contradiction_prompt, "openai")
            
            if not contradiction_analysis[1].startswith('['):
                st.success(contradiction_analysis[1])
                # Store contradiction analysis for advanced features
                st.session_state.latest_contradiction_analysis = contradiction_analysis[1]
            else:
                st.error(f"Contradiction analysis failed: {contradiction_analysis[1]}")
                st.session_state.latest_contradiction_analysis = f"Error: {contradiction_analysis[1]}"
                
        except Exception as e:
            st.error(f"Contradiction analysis failed: {str(e)}")
            st.session_state.latest_contradiction_analysis = f"Error: {str(e)}"

def render_follow_up_conversation(api_clients: Dict[str, Any]):
    """Render follow-up conversation interface"""
    st.subheader("💬 Follow-Up Conversation")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    follow_up_input = st.text_input("Ask a question or follow-up:", key="followup_question")
    
    if st.button("Send Follow-Up") and follow_up_input:
        if "openai" in api_clients:
            handle_follow_up_conversation(follow_up_input, api_clients)
        else:
            st.error("Follow-up conversation requires OpenAI API key.")
    
    # Display conversation history
    display_conversation_history()
    
    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

def handle_follow_up_conversation(follow_up_input: str, api_clients: Dict[str, Any]):
    """Handle follow-up conversation logic"""
    st.session_state.chat_history.append({"role": "user", "content": follow_up_input})
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(st.session_state.chat_history)
    
    try:
        openai_client = api_clients["openai"]
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            max_tokens=500
        )
        assistant_reply = response.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        error_msg = f"[Error: {str(e)}]"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

def display_conversation_history():
    """Display conversation history"""
    if st.session_state.chat_history:
        st.markdown("**Conversation History:**")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
