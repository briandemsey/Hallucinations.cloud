# ui/advanced_analysis.py
"""
Advanced Analysis Component for Hallucinations.cloud
Handles Red/Blue/Purple team analysis and security assessment
"""

import streamlit as st
from typing import Dict, Any, List, Tuple
from models.ai_models import AIModelsManager

def render_advanced_analysis(api_clients: Dict[str, Any]):
    """Render advanced analysis section if enabled"""
    
    if not st.session_state.get("show_advanced_analysis", False):
        return
    
    st.markdown("---")
    st.markdown("## 🛡️ Red/Blue/Purple Team Analysis")
    st.info("Advanced security analysis activated from sidebar controls")
    
    # Check if we have recent query data
    if not _has_query_data():
        _display_no_data_message()
        return
    
    # Create tabs for different team analyses
    tab1, tab2, tab3 = st.tabs(["🔴 Red Team", "🔵 Blue Team", "🟣 Purple Team"])
    
    with tab1:
        render_red_team_analysis(api_clients)
    
    with tab2:
        render_blue_team_analysis(api_clients)
    
    with tab3:
        render_purple_team_analysis(api_clients)

def _has_query_data() -> bool:
    """Check if we have recent query data for analysis"""
    return (hasattr(st.session_state, 'latest_query_results') and 
            st.session_state.latest_query_results)

def _display_no_data_message():
    """Display message when no query data is available"""
    st.info("🎯 Run a query first to enable Red/Blue/Purple team analysis")
    
    if st.button("🔄 Return to Query Interface"):
        st.session_state.show_advanced_analysis = False
        st.rerun()

def render_red_team_analysis(api_clients: Dict[str, Any]):
    """Render Red Team (Adversarial) Analysis"""
    
    st.markdown("### 🔴 Red Team Analysis (Adversarial Testing)")
    st.caption("Identify vulnerabilities and potential attack vectors in AI responses")
    
    if not _can_perform_analysis(api_clients):
        _display_api_requirement_message()
        return
    
    latest_results = st.session_state.latest_query_results
    latest_query = st.session_state.latest_query_text
    
    # Display query info
    st.info(f"Analyzing query: **{latest_query[:100]}{'...' if len(latest_query) > 100 else ''}**")
    
    # Red team analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_depth = st.selectbox(
            "Analysis Depth:",
            ["Basic", "Detailed", "Comprehensive"],
            help="Choose the depth of red team analysis"
        )
    
    with col2:
        focus_areas = st.multiselect(
            "Focus Areas:",
            ["Factual Accuracy", "Bias Detection", "Prompt Injection", "Information Gaps"],
            default=["Factual Accuracy", "Bias Detection"],
            help="Select specific areas to focus the red team analysis"
        )
    
    if st.button("🔴 Run Red Team Analysis", use_container_width=True):
        _execute_red_team_analysis(latest_query, latest_results, analysis_depth, focus_areas, api_clients)

def render_blue_team_analysis(api_clients: Dict[str, Any]):
    """Render Blue Team (Defensive) Analysis"""
    
    st.markdown("### 🔵 Blue Team Analysis (Defensive Assessment)")
    st.caption("Assess reliability and trustworthiness of AI responses")
    
    if not _can_perform_analysis(api_clients):
        _display_api_requirement_message()
        return
    
    latest_results = st.session_state.latest_query_results
    latest_query = st.session_state.latest_query_text
    
    # Display query info
    st.info(f"Analyzing query: **{latest_query[:100]}{'...' if len(latest_query) > 100 else ''}**")
    
    # Blue team analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        verification_level = st.selectbox(
            "Verification Level:",
            ["Standard", "Rigorous", "Forensic"],
            help="Choose the level of verification analysis"
        )
    
    with col2:
        trust_metrics = st.multiselect(
            "Trust Metrics:",
            ["Source Credibility", "Consistency Check", "Uncertainty Analysis", "Evidence Quality"],
            default=["Source Credibility", "Consistency Check"],
            help="Select trust metrics to evaluate"
        )
    
    if st.button("🔵 Run Blue Team Analysis", use_container_width=True):
        _execute_blue_team_analysis(latest_query, latest_results, verification_level, trust_metrics, api_clients)

def render_purple_team_analysis(api_clients: Dict[str, Any]):
    """Render Purple Team (Integrated) Analysis"""
    
    st.markdown("### 🟣 Purple Team Analysis (Integrated Intelligence)")
    st.caption("Combine offensive and defensive perspectives for comprehensive assessment")
    
    if not _can_perform_analysis(api_clients):
        _display_api_requirement_message()
        return
    
    latest_results = st.session_state.latest_query_results
    latest_query = st.session_state.latest_query_text
    latest_hscore = st.session_state.get('latest_hscore_result')
    
    # Display query info
    st.info(f"Analyzing query: **{latest_query[:100]}{'...' if len(latest_query) > 100 else ''}**")
    
    # Purple team analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        integration_mode = st.selectbox(
            "Integration Mode:",
            ["Balanced", "Security-Focused", "Reliability-Focused"],
            help="Choose the focus of the integrated analysis"
        )
    
    with col2:
        include_hscore = st.checkbox(
            "Include H-Score Data",
            value=latest_hscore is not None,
            disabled=latest_hscore is None,
            help="Include H-Score results in the analysis"
        )
    
    if st.button("🟣 Run Purple Team Analysis", use_container_width=True):
        _execute_purple_team_analysis(latest_query, latest_results, latest_hscore if include_hscore else None, 
                                     integration_mode, api_clients)

def _can_perform_analysis(api_clients: Dict[str, Any]) -> bool:
    """Check if we can perform analysis (need at least one AI model)"""
    return len(api_clients) > 0 and any(key in api_clients for key in ["openai", "anthropic", "gemini"])

def _display_api_requirement_message():
    """Display message about API requirements"""
    st.warning("⚠️ Advanced analysis requires at least one AI model API key (OpenAI, Anthropic, or Google)")

def _execute_red_team_analysis(query: str, results: List[Tuple[str, str]], depth: str, focus_areas: List[str], api_clients: Dict[str, Any]):
    """Execute red team analysis"""
    
    with st.spinner("🔴 Running Red Team analysis..."):
        try:
            # Create red team prompt based on options
            red_team_prompt = _create_red_team_prompt(query, results, depth, focus_areas)
            
            # Execute analysis
            models_manager = AIModelsManager(api_clients)
            red_analysis = models_manager.call_model_for_analysis(red_team_prompt, "openai")
            
            if not red_analysis[1].startswith('['):
                # Display results with appropriate formatting
                st.markdown("#### 🔴 Red Team Findings:")
                _display_analysis_results(red_analysis[1], "error")
                
                # Store results for potential export
                st.session_state.latest_red_team_analysis = red_analysis[1]
                
                # Offer additional actions
                _display_red_team_actions(results)
                
            else:
                st.error(f"Red Team analysis failed: {red_analysis[1]}")
                
        except Exception as e:
            st.error(f"Red team analysis failed: {str(e)}")

def _execute_blue_team_analysis(query: str, results: List[Tuple[str, str]], verification_level: str, trust_metrics: List[str], api_clients: Dict[str, Any]):
    """Execute blue team analysis"""
    
    with st.spinner("🔵 Running Blue Team analysis..."):
        try:
            # Create blue team prompt based on options
            blue_team_prompt = _create_blue_team_prompt(query, results, verification_level, trust_metrics)
            
            # Execute analysis
            models_manager = AIModelsManager(api_clients)
            blue_analysis = models_manager.call_model_for_analysis(blue_team_prompt, "openai")
            
            if not blue_analysis[1].startswith('['):
                # Display results with appropriate formatting
                st.markdown("#### 🔵 Blue Team Assessment:")
                _display_analysis_results(blue_analysis[1], "info")
                
                # Store results for potential export
                st.session_state.latest_blue_team_analysis = blue_analysis[1]
                
                # Offer additional actions
                _display_blue_team_actions(results)
                
            else:
                st.error(f"Blue Team analysis failed: {blue_analysis[1]}")
                
        except Exception as e:
            st.error(f"Blue team analysis failed: {str(e)}")

def _execute_purple_team_analysis(query: str, results: List[Tuple[str, str]], hscore_result, integration_mode: str, api_clients: Dict[str, Any]):
    """Execute purple team analysis"""
    
    with st.spinner("🟣 Running Purple Team analysis..."):
        try:
            # Create purple team prompt based on options
            purple_team_prompt = _create_purple_team_prompt(query, results, hscore_result, integration_mode)
            
            # Execute analysis
            models_manager = AIModelsManager(api_clients)
            purple_analysis = models_manager.call_model_for_analysis(purple_team_prompt, "openai")
            
            if not purple_analysis[1].startswith('['):
                # Display results with appropriate formatting
                st.markdown("#### 🟣 Purple Team Intelligence:")
                _display_analysis_results(purple_analysis[1], "success")
                
                # Store results for potential export
                st.session_state.latest_purple_team_analysis = purple_analysis[1]
                
                # Offer additional actions
                _display_purple_team_actions(results, hscore_result)
                
            else:
                st.error(f"Purple Team analysis failed: {purple_analysis[1]}")
                
        except Exception as e:
            st.error(f"Purple team analysis failed: {str(e)}")

def _create_red_team_prompt(query: str, results: List[Tuple[str, str]], depth: str, focus_areas: List[str]) -> str:
    """Create red team analysis prompt"""
    
    depth_instructions = {
        "Basic": "Provide a concise analysis focusing on the most critical vulnerabilities.",
        "Detailed": "Provide a thorough analysis with specific examples and attack scenarios.",
        "Comprehensive": "Provide an exhaustive analysis covering all potential vulnerabilities and attack vectors."
    }
    
    focus_instruction = f"Focus particularly on: {', '.join(focus_areas)}." if focus_areas else ""
    
    prompt = f"""
As a Red Team cybersecurity analyst, analyze these AI responses for potential hallucinations, vulnerabilities, and attack vectors:

Original Query: {query}

AI Model Responses:
{chr(10).join([f"{name}: {resp}" for name, resp in results if not resp.startswith('[')])}

{depth_instructions.get(depth, depth_instructions["Basic"])}
{focus_instruction}

Provide:
1. Hallucination Risk Score (1-10) for each response
2. Specific vulnerability flags detected
3. Potential attack vectors that could exploit these weaknesses
4. Suggested adversarial follow-up questions to test these models further
5. Security recommendations for users

Format your response with clear sections and actionable insights.
"""
    
    return prompt

def _create_blue_team_prompt(query: str, results: List[Tuple[str, str]], verification_level: str, trust_metrics: List[str]) -> str:
    """Create blue team analysis prompt"""
    
    verification_instructions = {
        "Standard": "Apply standard verification practices and trust assessment.",
        "Rigorous": "Apply strict verification standards with detailed fact-checking.",
        "Forensic": "Apply forensic-level analysis with comprehensive evidence evaluation."
    }
    
    metrics_instruction = f"Pay special attention to: {', '.join(trust_metrics)}." if trust_metrics else ""
    
    prompt = f"""
As a Blue Team cybersecurity analyst, assess the reliability and trustworthiness of these AI responses:

Original Query: {query}

AI Model Responses:
{chr(10).join([f"{name}: {resp}" for name, resp in results if not resp.startswith('[')])}

{verification_instructions.get(verification_level, verification_instructions["Standard"])}
{metrics_instruction}

Provide:
1. Reliability score for each response (1-10)
2. Trust indicators found (verifiable claims, appropriate uncertainty language, sources)
3. Evidence quality assessment
4. Verification recommendations for users
5. Defense strategies against identified hallucinations
6. Recommended additional verification steps

Format your response with clear defensive recommendations.
"""
    
    return prompt

def _create_purple_team_prompt(query: str, results: List[Tuple[str, str]], hscore_result, integration_mode: str) -> str:
    """Create purple team analysis prompt"""
    
    hscore_context = ""
    if hscore_result:
        hscore_context = f"""
H-Score Analysis Results:
- Overall Score: {hscore_result.overall_score:.3f}
- Risk Level: {hscore_result.risk_level}
- Component Scores: {hscore_result.component_scores}
- Key Findings: {hscore_result.explanation}
"""
    
    mode_instructions = {
        "Balanced": "Provide a balanced perspective combining both offensive and defensive insights.",
        "Security-Focused": "Emphasize security vulnerabilities while incorporating defensive measures.",
        "Reliability-Focused": "Focus on information reliability while noting potential exploitation risks."
    }
    
    prompt = f"""
As a Purple Team analyst, provide integrated offensive and defensive insights combining both Red Team and Blue Team perspectives:

Original Query: {query}

AI Model Responses:
{chr(10).join([f"{name}: {resp}" for name, resp in results if not resp.startswith('[')])}

{hscore_context}

{mode_instructions.get(integration_mode, mode_instructions["Balanced"])}

Provide:
1. Overall risk assessment combining attack and defense perspectives
2. Integrated threat model for these AI responses
3. Methodology recommendations for users
4. Key learnings about AI model limitations revealed by this query
5. Actionable intelligence for improving AI reliability assessment
6. Strategic recommendations for both offensive testing and defensive measures

Format your response as an executive intelligence briefing.
"""
    
    return prompt

def _display_analysis_results(analysis_text: str, message_type: str):
    """Display analysis results with appropriate formatting"""
    
    if message_type == "error":
        st.error(analysis_text)
    elif message_type == "info":
        st.info(analysis_text)
    elif message_type == "success":
        st.success(analysis_text)
    else:
        st.write(analysis_text)

def _display_red_team_actions(results: List[Tuple[str, str]]):
    """Display additional actions for red team analysis"""
    
    st.markdown("#### 🎯 Red Team Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Generate Attack Scenarios", help="Create specific attack scenarios"):
            st.info("Attack scenario generation would be implemented here")
    
    with col2:
        if st.button("📝 Export Red Team Report", help="Export analysis as a report"):
            _export_team_analysis("red")
    
    with col3:
        if st.button("🔄 Re-analyze with Different Focus", help="Run analysis with different parameters"):
            st.info("Re-analysis with different parameters would be implemented here")

def _display_blue_team_actions(results: List[Tuple[str, str]]):
    """Display additional actions for blue team analysis"""
    
    st.markdown("#### 🛡️ Blue Team Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Generate Verification Checklist", help="Create verification checklist"):
            st.info("Verification checklist generation would be implemented here")
    
    with col2:
        if st.button("📝 Export Blue Team Report", help="Export analysis as a report"):
            _export_team_analysis("blue")
    
    with col3:
        if st.button("🔍 Deep Fact Check", help="Perform additional fact checking"):
            st.info("Deep fact checking would be implemented here")

def _display_purple_team_actions(results: List[Tuple[str, str]], hscore_result):
    """Display additional actions for purple team analysis"""
    
    st.markdown("#### 🎭 Purple Team Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Intelligence Summary", help="Create executive summary"):
            st.info("Intelligence summary generation would be implemented here")
    
    with col2:
        if st.button("📝 Export Purple Team Report", help="Export analysis as a report"):
            _export_team_analysis("purple")
    
    with col3:
        if st.button("🔄 Continuous Monitoring Setup", help="Set up ongoing monitoring"):
            st.info("Continuous monitoring setup would be implemented here")

def _export_team_analysis(team_type: str):
    """Export team analysis results"""
    
    analysis_key = f"latest_{team_type}_team_analysis"
    
    if analysis_key in st.session_state:
        analysis_text = st.session_state[analysis_key]
        
        # Create downloadable content
        timestamp = st.session_state.get('latest_query_text', 'Unknown')[:50]
        filename = f"{team_type}_team_analysis_{timestamp}.txt"
        
        st.download_button(
            label=f"📥 Download {team_type.title()} Team Report",
            data=analysis_text,
            file_name=filename,
            mime="text/plain"
        )
    else:
        st.warning(f"No {team_type} team analysis available to export")
