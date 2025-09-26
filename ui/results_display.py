# ui/results_display.py
"""
Results display component for Hallucinations.cloud
Shows AI responses, H-Score, and analysis
"""

import streamlit as st
from typing import Dict, Any
from analysis.hscore import calculate_h_score, analyze_contradictions

def render_results(results: Dict[str, Dict[str, Any]]):
    """Render AI model results and analysis"""

    if not results:
        return

    st.header("📊 Results Analysis")

    # Calculate H-Score if enabled
    if st.session_state.get('auto_hscore', True):
        h_score = calculate_h_score(results)

        # H-Score display with color coding
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            score_color = "green" if h_score >= 7 else "orange" if h_score >= 4 else "red"
            st.markdown(f"### H-Score: <span style='color:{score_color}'>{h_score}/10</span>", unsafe_allow_html=True)

        with col2:
            if h_score >= 7:
                reliability = "High ✅"
            elif h_score >= 4:
                reliability = "Moderate ⚠️"
            else:
                reliability = "Low ❌"
            st.markdown(f"**Reliability:** {reliability}")

        with col3:
            if h_score >= 8:
                st.success("Excellent consensus across models!")
            elif h_score >= 6:
                st.info("Good agreement between models")
            elif h_score >= 4:
                st.warning("Some disagreement detected")
            else:
                st.error("Significant inconsistencies found")

    st.markdown("---")

    # Individual model responses
    st.header("🤖 Model Responses")

    successful_responses = 0
    total_models = len(results)

    for model_name, result in results.items():
        with st.expander(f"📱 {model_name}", expanded=True):

            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
                if st.session_state.get('show_errors', True):
                    st.code(str(result), language="json")
            else:
                successful_responses += 1

                # Response text
                response_text = result.get("response", "No response")
                st.markdown(f"**Response:**\n\n{response_text}")

                # Token usage (if enabled and available)
                if st.session_state.get('show_tokens', False):
                    tokens = result.get("tokens", 0)
                    if tokens > 0:
                        st.caption(f"Tokens used: {tokens}")

                # Response statistics
                word_count = len(response_text.split())
                char_count = len(response_text)
                st.caption(f"Length: {word_count} words, {char_count} characters")

    # Response summary
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Successful Responses", f"{successful_responses}/{total_models}")

    with col2:
        success_rate = (successful_responses / total_models) * 100 if total_models > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

    # Contradiction analysis
    if st.session_state.get('show_contradictions', True) and successful_responses >= 2:
        contradictions = analyze_contradictions(results)

        st.markdown("---")
        st.header("⚖️ Contradiction Analysis")

        if contradictions:
            st.warning(f"Found {len(contradictions)} potential contradictions:")
            for i, contradiction in enumerate(contradictions, 1):
                st.write(f"{i}. {contradiction}")
        else:
            st.success("✅ No major contradictions detected between models")

    # Advanced analysis section
    with st.expander("🔍 Advanced Analysis"):
        if successful_responses >= 2:

            # Response similarity analysis
            st.subheader("Response Patterns")

            response_lengths = []
            for model_name, result in results.items():
                if "response" in result and not result.get("error"):
                    response_lengths.append(len(result["response"].split()))

            if response_lengths:
                avg_length = sum(response_lengths) / len(response_lengths)
                min_length = min(response_lengths)
                max_length = max(response_lengths)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Length", f"{avg_length:.0f} words")
                with col2:
                    st.metric("Shortest", f"{min_length} words")
                with col3:
                    st.metric("Longest", f"{max_length} words")

            # Export options
            st.subheader("Export Results")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📋 Copy to Clipboard"):
                    # Create formatted text for copying
                    export_text = _format_results_for_export(results, h_score if 'h_score' in locals() else 0)
                    st.code(export_text, language="text")

            with col2:
                if st.button("💾 Save as JSON"):
                    st.download_button(
                        label="Download JSON",
                        data=str(results),
                        file_name="hallucinations_results.json",
                        mime="application/json"
                    )
        else:
            st.info("Advanced analysis requires at least 2 successful responses")

def _format_results_for_export(results: Dict[str, Any], h_score: float) -> str:
    """Format results for text export"""

    export_lines = [
        "=== HALLUCINATIONS.CLOUD RESULTS ===",
        f"H-Score: {h_score}/10",
        f"Timestamp: {st.session_state.get('timestamp', 'Unknown')}",
        "",
        "MODEL RESPONSES:"
    ]

    for model_name, result in results.items():
        export_lines.append(f"\n--- {model_name} ---")
        if "error" in result:
            export_lines.append(f"ERROR: {result['error']}")
        else:
            export_lines.append(result.get("response", "No response"))

    return "\n".join(export_lines)