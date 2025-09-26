# ui/results_display.py
"""
Results Display Component for Hallucinations.cloud
Handles H-Score visualization and results presentation
"""

import streamlit as st
from analysis.hscore_engine import HScoreResult

class ResultsDisplay:
    """Handles display of H-Score results and analysis"""
    
    def __init__(self):
        self.risk_colors = {
            "LOW": "#28a745",      # Green
            "MEDIUM": "#ffc107",   # Yellow
            "HIGH": "#fd7e14",     # Orange
            "CRITICAL": "#dc3545", # Red
            "UNKNOWN": "#6c757d"   # Gray
        }
        
        self.risk_emojis = {
            "LOW": "🟢",
            "MEDIUM": "🟡", 
            "HIGH": "🟠",
            "CRITICAL": "🔴",
            "UNKNOWN": "⚪"
        }
    
    def get_score_color(self, score: float) -> str:
        """Get color for score visualization"""
        if score >= 0.8:
            return self.risk_colors["LOW"]
        elif score >= 0.6:
            return self.risk_colors["MEDIUM"]
        elif score >= 0.4:
            return self.risk_colors["HIGH"]
        else:
            return self.risk_colors["CRITICAL"]
    
    def display_hscore_results(self, hscore_result: HScoreResult):
        """Display comprehensive H-Score results"""
        
        # Main H-Score display
        self._display_main_score(hscore_result)
        
        # Component breakdown
        self._display_component_breakdown(hscore_result)
        
        # Analysis and explanation
        self._display_analysis_explanation(hscore_result)
        
        # Risk assessment
        self._display_risk_assessment(hscore_result)
        
        # Recommendations
        self._display_recommendations(hscore_result)
    
    def _display_main_score(self, hscore_result: HScoreResult):
        """Display the main H-Score with metrics"""
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            score_color = self.get_score_color(hscore_result.overall_score)
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; 
                        background: linear-gradient(45deg, {score_color}20, {score_color}10); 
                        border-radius: 10px; border: 2px solid {score_color};">
                <h1 style="color: {score_color}; margin: 0; font-size: 3em;">
                    {hscore_result.overall_score:.3f}
                </h1>
                <p style="margin: 5px 0; color: #666; font-size: 1.2em;">
                    H-Score (Reliability)
                </p>
                <p style="margin: 0; color: #888; font-size: 0.9em;">
                    Scale: 0.000 (Critical) → 1.000 (Perfect)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            risk_emoji = self.risk_emojis.get(hscore_result.risk_level, "⚪")
            st.metric(
                "Risk Level", 
                f"{risk_emoji} {hscore_result.risk_level}",
                help="Overall reliability assessment"
            )
        
        with col3:
            st.metric(
                "Analysis Time", 
                hscore_result.timestamp.strftime("%H:%M:%S"),
                help="When this analysis was performed"
            )
        
        with col4:
            valid_components = sum(1 for score in hscore_result.component_scores.values() if score > 0)
            st.metric(
                "Components", 
                f"{valid_components}/4",
                help="Agreement, Verification, Certainty, Quality"
            )
    
    def _display_component_breakdown(self, hscore_result: HScoreResult):
        """Display component scores with progress bars"""
        
        st.markdown("### 📊 Component Analysis")
        comp_cols = st.columns(4)
        
        components = [
            ("🤝 Agreement", hscore_result.component_scores['agreement'], 
             "How much AI models agree with each other"),
            ("✅ Verification", hscore_result.component_scores['verification'], 
             "Presence of verifiable facts and sources"),
            ("🎯 Certainty", hscore_result.component_scores['uncertainty'], 
             "Confidence indicators vs uncertainty language"),
            ("⭐ Quality", hscore_result.component_scores['quality'], 
             "Response coherence, depth, and specificity")
        ]
        
        for i, (name, score, help_text) in enumerate(components):
            with comp_cols[i]:
                # Display metric
                st.metric(
                    name, 
                    f"{score:.3f}",
                    delta=self._get_score_delta_text(score),
                    help=help_text
                )
                
                # Progress bar with color coding
                progress_color = self.get_score_color(score)
                st.markdown(f"""
                <div style="background-color: {progress_color}20; height: 12px; 
                           border-radius: 6px; overflow: hidden; margin-top: 5px;">
                    <div style="background-color: {progress_color}; height: 100%; 
                               width: {score*100}%; transition: width 0.3s ease;"></div>
                </div>
                """, unsafe_allow_html=True)
    
    def _get_score_delta_text(self, score: float) -> str:
        """Get delta text for score interpretation"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _display_analysis_explanation(self, hscore_result: HScoreResult):
        """Display analysis explanation"""
        
        st.markdown("### 💡 Analysis Explanation")
        
        # Create explanation with enhanced formatting
        explanation_parts = hscore_result.explanation.split(';')
        
        if len(explanation_parts) > 1:
            # Display main score interpretation
            main_part = explanation_parts[0].strip()
            st.info(f"**Overall Assessment:** {main_part}")
            
            # Display detailed findings
            if len(explanation_parts) > 1:
                st.markdown("**Detailed Findings:**")
                for part in explanation_parts[1:]:
                    part = part.strip()
                    if part:
                        # Add appropriate emoji based on content
                        if "disagreement" in part.lower():
                            st.markdown(f"⚠️ {part.capitalize()}")
                        elif "consensus" in part.lower() or "agreement" in part.lower():
                            st.markdown(f"✅ {part.capitalize()}")
                        elif "uncertainty" in part.lower():
                            st.markdown(f"🤔 {part.capitalize()}")
                        elif "quality" in part.lower():
                            st.markdown(f"📝 {part.capitalize()}")
                        else:
                            st.markdown(f"• {part.capitalize()}")
        else:
            st.info(hscore_result.explanation)
    
    def _display_risk_assessment(self, hscore_result: HScoreResult):
        """Display risk assessment with appropriate styling"""
        
        st.markdown("### ⚡ Risk Assessment")
        
        risk_messages = {
            "CRITICAL": {
                "message": "🚨 **CRITICAL RISK DETECTED** - Do not use this information for important decisions without independent verification",
                "type": "error",
                "details": "The AI responses show significant reliability issues. Multiple verification sources are strongly recommended."
            },
            "HIGH": {
                "message": "⚠️ **HIGH RISK** - Verify information independently before making decisions",
                "type": "warning", 
                "details": "The AI responses have notable reliability concerns. Cross-reference with trusted sources before use."
            },
            "MEDIUM": {
                "message": "🔍 **MEDIUM RISK** - Generally reliable but additional verification recommended for critical use",
                "type": "info",
                "details": "The AI responses appear reasonably reliable but may benefit from verification for important decisions."
            },
            "LOW": {
                "message": "✅ **LOW RISK** - High confidence - information appears reliable and well-supported",
                "type": "success",
                "details": "The AI responses show strong reliability indicators across multiple measures."
            },
            "UNKNOWN": {
                "message": "❓ **UNKNOWN RISK** - Insufficient data for reliable assessment",
                "type": "warning",
                "details": "Unable to determine reliability due to insufficient or invalid model responses."
            }
        }
        
        risk_info = risk_messages.get(hscore_result.risk_level, risk_messages["UNKNOWN"])
        
        # Display main risk message
        if risk_info["type"] == "error":
            st.error(risk_info["message"])
        elif risk_info["type"] == "warning":
            st.warning(risk_info["message"])
        elif risk_info["type"] == "success":
            st.success(risk_info["message"])
        else:
            st.info(risk_info["message"])
        
        # Display additional details
        with st.expander("📋 Risk Assessment Details"):
            st.write(risk_info["details"])
            
            # Score interpretation guide
            st.markdown("**Score Interpretation Guide:**")
            st.markdown("- 0.800 - 1.000: Low Risk (Highly Reliable)")
            st.markdown("- 0.600 - 0.799: Medium Risk (Generally Reliable)")
            st.markdown("- 0.400 - 0.599: High Risk (Verification Needed)")
            st.markdown("- 0.000 - 0.399: Critical Risk (Do Not Trust)")
    
    def _display_recommendations(self, hscore_result: HScoreResult):
        """Display actionable recommendations"""
        
        if not hscore_result.recommendations:
            return
        
        st.markdown("### 🎯 Recommendations")
        
        # Categorize recommendations
        critical_recs = [r for r in hscore_result.recommendations if "⚠️" in r or "Verify" in r]
        improvement_recs = [r for r in hscore_result.recommendations if "Request" in r or "Consider" in r]
        positive_recs = [r for r in hscore_result.recommendations if "✅" in r]
        other_recs = [r for r in hscore_result.recommendations if r not in critical_recs + improvement_recs + positive_recs]
        
        # Display critical recommendations first
        if critical_recs:
            st.markdown("**🚨 Critical Actions:**")
            for rec in critical_recs:
                st.markdown(f"- {rec}")
        
        # Display improvement recommendations
        if improvement_recs:
            st.markdown("**🔧 Improvement Suggestions:**")
            for rec in improvement_recs:
                st.markdown(f"- {rec}")
        
        # Display positive reinforcement
        if positive_recs:
            st.markdown("**✅ Positive Indicators:**")
            for rec in positive_recs:
                st.markdown(f"- {rec}")
        
        # Display other recommendations
        if other_recs:
            st.markdown("**💡 Additional Recommendations:**")
            for rec in other_recs:
                st.markdown(f"- {rec}")
    
    def display_comparison_chart(self, hscore_result: HScoreResult):
        """Display component scores as a radar/bar chart"""
        
        try:
            import plotly.graph_objects as go
            
            # Create bar chart for component scores
            components = list(hscore_result.component_scores.keys())
            scores = list(hscore_result.component_scores.values())
            colors = [self.get_score_color(score) for score in scores]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=components,
                    y=scores,
                    marker_color=colors,
                    text=[f"{score:.3f}" for score in scores],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="H-Score Component Breakdown",
                xaxis_title="Components",
                yaxis_title="Score (0-1)",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            # Fallback if plotly not available
            st.markdown("📊 **Component Scores:**")
            for component, score in hscore_result.component_scores.items():
                st.markdown(f"- {component.title()}: {score:.3f}")
    
    def export_results(self, hscore_result: HScoreResult) -> str:
        """Export H-Score results as formatted text"""
        
        export_text = f"""
H-SCORE ANALYSIS REPORT
Generated: {hscore_result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

OVERALL SCORE: {hscore_result.overall_score:.3f}/1.0
RISK LEVEL: {hscore_result.risk_level}

COMPONENT SCORES:
- Agreement: {hscore_result.component_scores['agreement']:.3f}
- Verification: {hscore_result.component_scores['verification']:.3f}
- Certainty: {hscore_result.component_scores['uncertainty']:.3f}
- Quality: {hscore_result.component_scores['quality']:.3f}

EXPLANATION:
{hscore_result.explanation}

RECOMMENDATIONS:
"""
        for i, rec in enumerate(hscore_result.recommendations, 1):
            export_text += f"{i}. {rec}\n"
        
        return export_text
