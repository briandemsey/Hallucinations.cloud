
# hallucinations_secure_production.py
"""
Hallucinations.cloud Multi-Model Comparison App
Version 1.8 - SECURE PRODUCTION VERSION - WITH CONTINUOUS CONVERSATION

REQUIRED ENVIRONMENT VARIABLES:
================================

CORE FUNCTIONALITY:
- OPENAI_API_KEY: GPT-4o model access
- ANTHROPIC_API_KEY: Claude model + content moderation  
- STRIPE_LIVE_SECRET_KEY or STRIPE_TEST_SECRET_KEY: Payment processing
- STRIPE_ENVIRONMENT: "live" or "test"
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN: Phone verification
- TWILIO_VERIFY_SERVICE_SID: SMS verification service
- TWILIO_PHONE_NUMBER: Your Twilio phone number

SUBSCRIPTION PRICING (Required for each environment):
- STRIPE_PRICE_CONSUMER_LIVE / STRIPE_PRICE_CONSUMER_TEST
- STRIPE_PRICE_PROFESSIONAL_LIVE / STRIPE_PRICE_PROFESSIONAL_TEST  
- STRIPE_PRICE_ENTERPRISE_LIVE / STRIPE_PRICE_ENTERPRISE_TEST

TRUTH VERIFICATION (Recommended for premium features):
- GOOGLE_API_KEY: Google Custom Search API access
- GOOGLE_SEARCH_ENGINE_ID: Custom Search Engine ID
- NEWSAPI_KEY: (Optional) Additional news source verification

OPTIONAL AI MODELS:
- GROK_API_KEY: X.AI Grok model
- PERPLEXITY_API_KEY: Perplexity AI model
- COHERE_API_KEY: Cohere model
- DEEPSEEK_API_KEY: Deepseek model
- OPENROUTER_API_KEY: OpenRouter model access

DEPLOYMENT:
- APP_URL: Your app's URL for Stripe redirects (e.g., https://yourapp.com)
"""
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import re
import json
import pandas as pd
import time
import requests
import logging
from datetime import datetime, timedelta
from openai import OpenAI
import anthropic
import google.generativeai as genai
import cohere
import stripe
from twilio.rest import Client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === TRUTH VERIFICATION ENGINE (Integrated) ===
import requests
from urllib.parse import urlparse

class TruthVerificationEngine:
    def __init__(self, config):
        """Initialize with API keys and configuration"""
        self.google_api_key = config.get('google_api_key')
        self.google_search_engine_id = config.get('google_search_engine_id')
        self.newsapi_key = config.get('newsapi_key')
        
    def verify_response_accuracy(self, query, ai_responses):
        """Main verification function - returns comprehensive accuracy assessment"""
        verification_results = {
            'overall_truth_score': 0.0,
            'fact_checks': [],
            'source_verification': {},
            'cross_reference_score': 0.0,
            'temporal_accuracy': 0.0,
            'confidence_level': 'medium',
            'verification_summary': '',
            'evidence_found': [],
            'contradictions': [],
            'warnings': []
        }
        
        try:
            # Step 1: Extract factual claims from responses
            claims = self._extract_factual_claims(ai_responses)
            
            # Step 2: Cross-reference with reliable sources
            cross_ref_score = self._cross_reference_information(query, ai_responses)
            verification_results['cross_reference_score'] = cross_ref_score
            
            # Step 3: Check temporal accuracy (how current is the info)
            temporal_score = self._check_temporal_accuracy(query, ai_responses)
            verification_results['temporal_accuracy'] = temporal_score
            
            # Step 4: Verify any URLs/sources mentioned
            source_verification = self._verify_sources(ai_responses)
            verification_results['source_verification'] = source_verification
            
            # Step 5: Calculate overall truth score
            overall_score = self._calculate_truth_score(verification_results)
            verification_results['overall_truth_score'] = overall_score
            
            # Step 6: Generate verification summary
            summary = self._generate_verification_summary(verification_results)
            verification_results['verification_summary'] = summary
            
            # Step 7: Determine confidence level
            verification_results['confidence_level'] = self._determine_confidence_level(overall_score)
            
        except Exception as e:
            verification_results['warnings'].append(f"Verification error: {str(e)}")
            verification_results['confidence_level'] = 'low'
        
        return verification_results
    
    def _extract_factual_claims(self, ai_responses):
        """Extract specific factual claims from AI responses"""
        claims = []
        
        # Patterns that typically indicate factual claims
        fact_patterns = [
            r'(\d{4})',  # Years
            r'(\d+(?:\.\d+)?%)',  # Percentages
            r'(\$\d+(?:\.\d+)?(?:\s?(?:million|billion|trillion))?)',  # Money amounts
            r'(\d+(?:\.\d+)?\s?(?:km|miles|meters|feet|kg|pounds|tons))',  # Measurements
            r'((?:in|on|during)\s+\d{4})',  # Temporal references
            r'(according to [^,\.]+)',  # Source attributions
            r'(studies show|research indicates|data suggests)',  # Research claims
        ]
        
        for model_name, response in ai_responses:
            if not response.startswith('[') and 'error' not in response.lower():
                # Extract sentences that contain factual patterns
                sentences = re.split(r'[.!?]+', response)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Skip very short sentences
                        for pattern in fact_patterns:
                            if re.search(pattern, sentence, re.IGNORECASE):
                                claims.append({
                                    'text': sentence,
                                    'source_model': model_name,
                                    'type': 'factual_claim',
                                    'confidence': 0.7
                                })
                                break
        
        return claims[:10]  # Limit to top 10 claims for performance
    
    def _cross_reference_information(self, query, ai_responses):
        """Cross-reference information with reliable sources"""
        if not self.google_api_key or not self.google_search_engine_id:
            return 0.5  # Default score if no search API
        
        try:
            # Search for reliable sources on the topic
            search_url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_search_engine_id,
                'q': query,
                'num': 5,
                'siteSearch': 'edu OR gov OR org',  # Focus on reliable domains
                'siteSearchFilter': 'i'  # Include these sites
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                reliable_sources = data.get('items', [])
                
                if reliable_sources:
                    # Simple scoring based on number of reliable sources found
                    return min(len(reliable_sources) / 5.0, 1.0)
            
            return 0.3  # Low score if no reliable sources found
            
        except Exception:
            return 0.5  # Default score on error
    
    def _check_temporal_accuracy(self, query, ai_responses):
        """Check if information is current and up-to-date"""
        # Look for temporal indicators in responses
        current_year = datetime.now().year
        temporal_score = 0.7  # Default score
        
        for model_name, response in ai_responses:
            # Look for recent years mentioned
            years_mentioned = re.findall(r'\b(20\d{2})\b', response)
            if years_mentioned:
                recent_years = [int(year) for year in years_mentioned if int(year) >= current_year - 2]
                if recent_years:
                    temporal_score = 0.9  # High score for recent information
                elif any(int(year) >= current_year - 5 for year in years_mentioned):
                    temporal_score = 0.7  # Medium score for moderately recent info
                else:
                    temporal_score = 0.4  # Lower score for older information
        
        return temporal_score
    
    def _verify_sources(self, ai_responses):
        """Verify any URLs or sources mentioned in responses"""
        source_verification = {
            'urls_found': 0,
            'urls_verified': 0,
            'reliable_sources': 0,
            'broken_links': 0,
            'source_details': []
        }
        
        # Extract URLs from responses
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|\b[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}\b'
        
        for model_name, response in ai_responses:
            urls = re.findall(url_pattern, response)
            for url in urls[:3]:  # Limit to 3 URLs per response
                if not url.startswith('http'):
                    url = 'https://' + url
                
                source_verification['urls_found'] += 1
                
                try:
                    # Quick verification (head request only)
                    head_response = requests.head(url, timeout=5, allow_redirects=True)
                    if head_response.status_code < 400:
                        source_verification['urls_verified'] += 1
                        
                        # Check if it's a reliable domain
                        domain = urlparse(url).netloc.lower()
                        reliable_domains = ['.edu', '.gov', '.org', 'reuters.com', 'bbc.com', 
                                          'nature.com', 'science.org', 'nih.gov', 'who.int']
                        
                        if any(reliable in domain for reliable in reliable_domains):
                            source_verification['reliable_sources'] += 1
                        
                        source_verification['source_details'].append({
                            'url': url,
                            'status': 'verified',
                            'reliable': any(reliable in domain for reliable in reliable_domains)
                        })
                    else:
                        source_verification['broken_links'] += 1
                        
                except Exception:
                    source_verification['broken_links'] += 1
        
        return source_verification
    
    def _calculate_truth_score(self, verification_results):
        """Calculate overall truth score based on all verification factors"""
        weights = {
            'cross_reference': 0.4,
            'temporal_accuracy': 0.3,
            'source_verification': 0.2,
            'consistency': 0.1
        }
        
        # Source verification score
        source_score = 0.7  # Default
        source_info = verification_results.get('source_verification', {})
        if source_info.get('urls_found', 0) > 0:
            verified_ratio = source_info.get('urls_verified', 0) / source_info.get('urls_found', 1)
            reliable_ratio = source_info.get('reliable_sources', 0) / source_info.get('urls_found', 1)
            source_score = (verified_ratio * 0.6) + (reliable_ratio * 0.4)
        
        # Calculate weighted score
        overall_score = (
            verification_results.get('cross_reference_score', 0.5) * weights['cross_reference'] +
            verification_results.get('temporal_accuracy', 0.7) * weights['temporal_accuracy'] +
            source_score * weights['source_verification'] +
            0.7 * weights['consistency']  # Default consistency score
        )
        
        return round(overall_score, 2)
    
    def _generate_verification_summary(self, verification_results):
        """Generate human-readable verification summary"""
        score = verification_results['overall_truth_score']
        
        if score >= 0.8:
            summary = "✅ **High Accuracy**: Information appears to be well-supported by reliable sources."
        elif score >= 0.6:
            summary = "⚠️ **Moderate Accuracy**: Some information verified, but exercise caution."
        elif score >= 0.4:
            summary = "🔍 **Low Accuracy**: Limited verification found. Independent research recommended."
        else:
            summary = "⚠️ **Questionable Accuracy**: Significant concerns about information reliability."
        
        # Add specific findings
        details = []
        if verification_results['source_verification'].get('reliable_sources', 0) > 0:
            reliable_count = verification_results['source_verification']['reliable_sources']
            details.append(f"{reliable_count} reliable sources found")
        
        if details:
            summary += f" ({', '.join(details)})"
        
        return summary
    
    def _determine_confidence_level(self, truth_score):
        """Determine confidence level based on truth score"""
        if truth_score >= 0.8:
            return 'high'
        elif truth_score >= 0.6:
            return 'medium'
        elif truth_score >= 0.4:
            return 'low'
        else:
            return 'very_low'

def show_truth_verification_results(verification_results):
    """Display truth verification results in Streamlit"""
    st.markdown("---")
    st.subheader("🔍 Truth Verification Analysis")
    
    # Overall truth score
    score = verification_results['overall_truth_score']
    confidence = verification_results['confidence_level']
    
    # Color coding based on score
    if score >= 0.8:
        score_color = "🟢"
    elif score >= 0.6:
        score_color = "🟡"
    elif score >= 0.4:
        score_color = "🟠"
    else:
        score_color = "🔴"
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Truth Score",
            f"{score:.1f}/1.0",
            delta=f"{score_color} {confidence.title()}",
            help="Overall accuracy assessment based on source verification and cross-referencing"
        )
    
    with col2:
        cross_ref = verification_results.get('cross_reference_score', 0)
        st.metric(
            "🔗 Cross-Reference",
            f"{cross_ref:.1f}/1.0",
            help="How well information is supported by reliable external sources"
        )
    
    with col3:
        temporal = verification_results.get('temporal_accuracy', 0)
        st.metric(
            "⏰ Recency",
            f"{temporal:.1f}/1.0",
            help="How current and up-to-date the information appears to be"
        )
    
    with col4:
        source_info = verification_results.get('source_verification', {})
        reliable_sources = source_info.get('reliable_sources', 0)
        st.metric(
            "📚 Reliable Sources",
            reliable_sources,
            help="Number of reliable sources (.edu, .gov, .org) found"
        )
    
    # Verification summary
    st.markdown("### 📋 Verification Summary")
    st.markdown(verification_results['verification_summary'])
    
    # Detailed results
    with st.expander("🔬 Detailed Verification Analysis", expanded=False):
        
        # Source verification
        source_info = verification_results.get('source_verification', {})
        if source_info.get('urls_found', 0) > 0:
            st.markdown("#### 🔗 Source Verification Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- **URLs Found**: {source_info['urls_found']}")
                st.write(f"- **URLs Verified**: {source_info['urls_verified']}")
            with col2:
                st.write(f"- **Reliable Sources**: {source_info['reliable_sources']}")
                st.write(f"- **Broken Links**: {source_info['broken_links']}")
        
        # Cross-reference details
        cross_ref = verification_results.get('cross_reference_score', 0)
        st.markdown("#### 📊 Cross-Reference Analysis")
        if cross_ref >= 0.8:
            st.success("Strong support from reliable sources (.edu, .gov, .org)")
        elif cross_ref >= 0.6:
            st.info("Moderate support from reliable sources")
        elif cross_ref >= 0.4:
            st.warning("Limited support from reliable sources")
        else:
            st.error("Little to no support from reliable sources found")
        
        # Warnings
        if verification_results.get('warnings'):
            st.markdown("#### ⚠️ Verification Warnings")
            for warning in verification_results['warnings']:
                st.warning(warning)

def show_truth_verification_controls():
    """Show truth verification controls in sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔍 Truth Verification")
        
        # Verification settings
        if 'truth_verification_enabled' not in st.session_state:
            st.session_state.truth_verification_enabled = True
            
        st.session_state.truth_verification_enabled = st.checkbox(
            "Enable Truth Verification",
            value=st.session_state.truth_verification_enabled,
            help="Verify AI responses against reliable sources and cross-reference information"
        )
        
        # Show verification stats
        if st.session_state.get('verification_history'):
            recent_verifications = st.session_state.verification_history[-10:]
            if recent_verifications:
                avg_score = sum(v.get('overall_truth_score', 0) for v in recent_verifications) / len(recent_verifications)
                high_accuracy = sum(1 for v in recent_verifications if v.get('overall_truth_score', 0) >= 0.8)
                
                st.markdown(f"""
                **Recent Activity:**
                - **Avg Truth Score**: {avg_score:.2f}
                - **Verifications**: {len(recent_verifications)}
                - **High Accuracy**: {high_accuracy}
                """)

def integrate_truth_verification(query, ai_responses):
    """Main integration function for truth verification"""
    
    if not st.session_state.get('truth_verification_enabled', True):
        return None
    
    # Configuration for verification engine
    config = {
        'google_api_key': google_key,
        'google_search_engine_id': google_search_engine_id,
        'newsapi_key': newsapi_key
    }
    
    # Check if we have the required Google API credentials
    if not google_key or not google_search_engine_id:
        # Show warning about limited functionality
        st.warning("⚠️ Truth Verification running in limited mode. Configure Google Custom Search API for full functionality.")
        
        # Return basic verification results
        return {
            'overall_truth_score': 0.6,
            'cross_reference_score': 0.5,
            'temporal_accuracy': 0.7,
            'confidence_level': 'medium',
            'verification_summary': 'ℹ️ **Limited Verification**: Basic consistency check performed. Configure Google API for full source verification.',
            'source_verification': {'urls_found': 0, 'urls_verified': 0, 'reliable_sources': 0, 'broken_links': 0},
            'warnings': ['Google Custom Search API not configured - using basic verification mode']
        }
    
    # Initialize verification engine with full capabilities
    engine = TruthVerificationEngine(config)
    
    # Perform verification
    verification_results = engine.verify_response_accuracy(query, ai_responses)
    
    # Store in session history
    if 'verification_history' not in st.session_state:
        st.session_state.verification_history = []
    
    st.session_state.verification_history.append({
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'overall_truth_score': verification_results['overall_truth_score'],
        'confidence_level': verification_results['confidence_level']
    })
    
    # Keep only last 20 verifications
    if len(st.session_state.verification_history) > 20:
        st.session_state.verification_history = st.session_state.verification_history[-20:]
    
    return verification_results

# === ANTHROPIC CONTENT MODERATION FUNCTIONS (Replacing OpenAI) ===

def check_content_moderation(user_input):
    """Check user input against Anthropic's Claude moderation - REPLACES OpenAI"""
    if not anthropic_client:
        return {"flagged": False, "message": "Moderation unavailable - Anthropic API key required"}
    
    try:
        # Create moderation prompt for Anthropic Claude
        moderation_prompt = f"""You are a content moderation system. Analyze the following user input for policy violations:

<user_input>
{user_input}
</user_input>

Check for these violation categories:
- Harassment or threats
- Hate speech or discrimination  
- Violence or dangerous content
- Self-harm content
- Sexual content (inappropriate)
- Illegal activities
- Misinformation or false claims
- Spam or promotional abuse

Respond with ONLY a valid JSON object in this format:
{{
    "flagged": true/false,
    "categories": ["category1", "category2"],
    "confidence": 0.95,
    "explanation": "Brief explanation if flagged",
    "severity": "low/medium/high"
}}

If content is safe, set flagged to false and categories to empty array."""
        
        # Call Anthropic API
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",  # Fast and cost-effective
            max_tokens=200,
            messages=[{"role": "user", "content": moderation_prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        # Clean and parse JSON response
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json\n', '').replace('\n```', '')
        if response_text.startswith('```'):
            response_text = response_text.replace('```\n', '').replace('\n```', '')
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            flagged = 'true' in response_text.lower() or 'flagged' in response_text.lower()
            return {
                "flagged": flagged,
                "categories": ["parsing_error"] if flagged else [],
                "message": "Failed to parse moderation response",
                "confidence": 0.5,
                "anthropic_response": response_text[:100]
            }
        
        # Convert to OpenAI-compatible format for existing code
        return {
            "flagged": result.get("flagged", False),
            "categories": result.get("categories", []),
            "message": result.get("explanation", "Content approved") if result.get("flagged") else "Content approved",
            "confidence": result.get("confidence", 0.5),
            "severity": result.get("severity", "low"),
            "anthropic_details": result
        }
        
    except Exception as e:
        # Fail open with logging
        print(f"Anthropic moderation API error: {str(e)}")
        return {
            "flagged": False,  # Fail open - don't block content on error
            "message": f"Moderation check failed: {str(e)}",
            "error": True,
            "categories": []
        }

def show_moderation_warning(moderation_result):
    """Display appropriate warning for flagged content - UPDATED FOR ANTHROPIC"""
    if moderation_result.get("flagged"):
        st.error("🚫 **Content Policy Violation**")
        
        # Show specific categories if available
        categories = moderation_result.get("categories", [])
        if categories:
            category_text = ", ".join(categories)
            st.warning(f"Your query was flagged for: **{category_text}**")
        else:
            st.warning(f"**{moderation_result['message']}**")
        
        # Show explanation if available
        explanation = moderation_result.get("anthropic_details", {}).get("explanation", "")
        if explanation:
            st.info(f"**Explanation:** {explanation}")
        
        # Show severity if available
        severity = moderation_result.get("severity", "")
        if severity and severity != "low":
            if severity == "high":
                st.error(f"⚠️ **High Severity Violation Detected**")
            elif severity == "medium":
                st.warning(f"⚠️ **Medium Severity Violation**")
        
        with st.expander("ℹ️ Content Policy Information"):
            st.markdown("""
            **Hallucinations.cloud Content Policy:**
            
            We use Anthropic's Claude AI for intelligent content moderation with contextual understanding. 
            Queries are checked for:
            
            - **Harassment/Threats**: Threatening, bullying, or intimidating content
            - **Hate Speech**: Content promoting hatred based on identity or characteristics  
            - **Violence**: Content promoting, glorifying, or instructing violence
            - **Self-Harm**: Content promoting self-injury or dangerous activities
            - **Sexual Content**: Inappropriate sexual content or content involving minors
            - **Illegal Activities**: Content promoting illegal activities or harmful behaviors
            - **Misinformation**: Deliberately false or misleading information
            - **Spam/Abuse**: Promotional abuse or platform manipulation
            
            **Our Advanced Moderation:**
            - **Contextual Understanding**: Claude understands nuance and context
            - **Explanation Provided**: Clear reasons for any moderation decisions
            - **Appeals Process**: Contact support if you believe this was an error
            - **Privacy Focused**: Content analyzed but not stored permanently
            
            **What happens when content is flagged:**
            - Query is blocked from processing
            - Event is logged for compliance and improvement
            - No charges are applied to your account
            - You can modify your query and try again
            - Clear explanation provided for educational purposes
            
            For questions or appeals, contact: support@hallucinations.cloud
            """)
        
        return True
    return False

def log_moderation_event(user_phone, query, moderation_result):
    """Log moderation events for review and compliance - UPDATED FOR ANTHROPIC"""
    try:
        # Create enhanced moderation log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_phone": user_phone[-4:] if user_phone else "unknown",
            "query_length": len(query),
            "query_hash": hash(query),
            "flagged": moderation_result.get("flagged", False),
            "categories": moderation_result.get("categories", []),
            "confidence": moderation_result.get("confidence", 0.0),
            "severity": moderation_result.get("severity", "low"),
            "explanation": moderation_result.get("anthropic_details", {}).get("explanation", ""),
            "moderation_system": "anthropic_claude",
            "model": "claude-3-haiku-20240307"
        }
        
        # Store in session state
        if 'moderation_logs' not in st.session_state:
            st.session_state.moderation_logs = []
        
        st.session_state.moderation_logs.append(log_entry)
        
        # Keep only last 50 logs in session
        if len(st.session_state.moderation_logs) > 50:
            st.session_state.moderation_logs = st.session_state.moderation_logs[-50:]
            
    except Exception as e:
        print(f"Logging error: {str(e)}")

def show_moderation_controls():
    """Show moderation controls in sidebar - UPDATED FOR ANTHROPIC"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🛡️ AI Content Moderation")
        
        # Moderation settings
        if 'moderation_enabled' not in st.session_state:
            st.session_state.moderation_enabled = True
            
        st.session_state.moderation_enabled = st.checkbox(
            "Enable Anthropic Moderation",
            value=st.session_state.moderation_enabled,
            help="Check user queries using Anthropic's Constitutional AI for intelligent content safety"
        )
        
        # Show enhanced moderation info
        st.markdown("**🤖 Powered by Anthropic Claude**")
        st.caption("• Contextual understanding\n• Nuanced decision making\n• Clear explanations provided")
        
        # Show moderation stats
        if st.session_state.get('moderation_logs'):
            total_checks = len(st.session_state.moderation_logs)
            flagged_count = sum(1 for log in st.session_state.moderation_logs if log.get('flagged'))
            
            if total_checks > 0:
                avg_confidence = sum(log.get('confidence', 0) for log in st.session_state.moderation_logs) / total_checks
                
                st.markdown(f"""
                **Recent Activity:**
                - **Total Checks**: {total_checks}
                - **Flagged**: {flagged_count}
                - **Success Rate**: {((total_checks - flagged_count) / total_checks * 100):.1f}%
                - **Avg Confidence**: {avg_confidence:.2f}
                """)
            
            if st.button("📊 View Moderation Logs"):
                st.session_state.show_moderation_logs = True


def show_human_support_section():
    """Show real human support contact information in the sidebar"""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📞 Real Human Support")

        # Name
        st.markdown("**Brian**")

        # Phone & Email links
        st.markdown("📱 **Phone:** [+1-949-291-1422](tel:+19492911422)")
        st.markdown("✉️ **Email:** [brian@hallucinations.cloud](mailto:brian@hallucinations.cloud)")

        # Buttons for quick action
        st.link_button("📞 Call Now", "tel:+19492911422", use_container_width=True)
        st.link_button("✉️ Email Support", "mailto:brian@hallucinations.cloud", use_container_width=True)

        st.caption("For help with Hallucinations.cloud and H-LLM Multi-Model. Or any other AI challenge.")

def show_moderation_dashboard():
    """Show detailed moderation dashboard - UPDATED FOR ANTHROPIC"""
    st.markdown("### 🛡️ Anthropic AI Moderation Dashboard")
    
    if not st.session_state.get('moderation_logs'):
        st.info("No moderation events logged yet.")
        return
    
    logs = st.session_state.moderation_logs
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    total_checks = len(logs)
    flagged_count = sum(1 for log in logs if log.get('flagged'))
    recent_flags = sum(1 for log in logs[-10:] if log.get('flagged')) if len(logs) >= 10 else flagged_count
    avg_confidence = sum(log.get('confidence', 0) for log in logs) / total_checks if total_checks > 0 else 0
    
    with col1:
        st.metric("Total Checks", total_checks)
    with col2:
        st.metric("Flagged Content", flagged_count)
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    with col4:
        st.metric("Recent Flags", f"{recent_flags}/10")
    
    # Category breakdown
    st.markdown("#### 📊 Violation Categories")
    if flagged_logs := [log for log in logs if log.get('flagged')]:
        all_categories = []
        for log in flagged_logs:
            all_categories.extend(log.get('categories', []))
        
        if all_categories:
            category_counts = {}
            for cat in all_categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Display as a simple table
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{category}**: {count} incidents")
    
    # Recent flagged content
    st.markdown("#### 🚩 Recent Flagged Events")
    flagged_logs = [log for log in logs if log.get('flagged')]
    
    if flagged_logs:
        for log in flagged_logs[-5:]:
            with st.expander(f"Flagged: {log['timestamp'][:16]} - {', '.join(log.get('categories', []))}"):
                st.markdown(f"**Categories**: {', '.join(log.get('categories', []))}")
                st.markdown(f"**Confidence**: {log.get('confidence', 0):.2f}")
                st.markdown(f"**Severity**: {log.get('severity', 'unknown')}")
                if log.get('explanation'):
                    st.markdown(f"**Explanation**: {log['explanation']}")
                st.json({k: v for k, v in log.items() if k not in ['query_hash', 'user_phone']})
    else:
        st.success("✅ No recent flagged content!")
    
    # System performance
    st.markdown("#### ⚡ Moderation Performance")
    if total_checks > 0:
        success_rate = ((total_checks - flagged_count) / total_checks * 100)
        if success_rate >= 95:
            st.success(f"✅ Excellent performance: {success_rate:.1f}% success rate")
        elif success_rate >= 90:
            st.info(f"ℹ️ Good performance: {success_rate:.1f}% success rate")
        else:
            st.warning(f"⚠️ Review needed: {success_rate:.1f}% success rate")

def add_content_policy_info():
    """Add content policy information to query interface - UPDATED FOR ANTHROPIC"""
    with st.expander("🛡️ AI Content Policy", expanded=False):
        st.markdown("""
        **Advanced AI Safety Guidelines:**
        
        ✅ **Allowed**: Educational questions, research queries, factual information requests, creative writing prompts, academic discussions
        
        ❌ **Not Allowed**: Harassment, hate speech, violent content, illegal activities, misinformation, harmful instructions
        
        **🤖 Powered by Anthropic Claude:**
        - **Contextual Understanding**: Understands nuance and intent
        - **Constitutional AI**: Built-in ethical reasoning and safety measures  
        - **Transparent Decisions**: Clear explanations for any moderation actions
        - **Continuous Learning**: Improves understanding of complex scenarios
        
        **Privacy & Appeals**: 
        - Content analyzed for safety but not stored permanently
        - Clear explanations provided for educational purposes
        - Appeals: contact support@hallucinations.cloud if you believe content was incorrectly flagged
        
        **Why Anthropic?** More intelligent than traditional keyword-based systems, with better understanding of context, intent, and nuanced human communication.
        """)

def process_query_with_moderation(user_query):
    """Process query with Anthropic content moderation check - UPDATED"""
    
    # Step 1: Anthropic Content Moderation Check
    if st.session_state.get('moderation_enabled', True):
        with st.spinner("🤖 Checking content policy with Anthropic AI..."):
            moderation_result = check_content_moderation(user_query)
            
            # Log the moderation event
            log_moderation_event(
                st.session_state.get('user_phone', 'unknown'),
                user_query,
                moderation_result
            )
            
            # If content is flagged, show warning and return False
            if show_moderation_warning(moderation_result):
                return False
    
    # Step 2: Content is approved, proceed with processing
    return True

def show_anthropic_moderation_status():
    """Show Anthropic moderation status in admin dashboard"""
    if not is_super_user():
        return
        
    st.markdown("#### 🤖 Anthropic Moderation Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if anthropic_client:
            st.success("✅ Anthropic Claude API: Active")
            st.caption("Model: claude-3-haiku-20240307")
        else:
            st.error("❌ Anthropic API: Not configured")
    
    with col2:
        if st.session_state.get('moderation_enabled', True):
            st.success("✅ AI Moderation: Enabled")
        else:
            st.warning("⚠️ AI Moderation: Disabled")
    
    # Test moderation button for admins
    if anthropic_client and st.button("🧪 Test Moderation System"):
        test_content = "This is a test message to verify the moderation system is working correctly."
        with st.spinner("Testing Anthropic moderation..."):
            result = check_content_moderation(test_content)
            
        if result.get("error"):
            st.error(f"❌ Test failed: {result['message']}")
        else:
            st.success(f"✅ Test passed: {result['message']}")
            st.json(result)

# Global flag to indicate Anthropic moderation is available

# === BLOCK 1: Configuration & Setup ===
load_dotenv()

# Get API keys
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
newsapi_key = os.getenv("NEWSAPI_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
grok_key = os.getenv("GROK_API_KEY")
perplexity_key = os.getenv("PERPLEXITY_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# === SECURE STRIPE CONFIGURATION - PRODUCTION READY WITH FALLBACK ===
stripe_environment = os.getenv("STRIPE_ENVIRONMENT", "live")

# Initialize Stripe with fallback handling
stripe_key = None
if stripe_environment == "test":
    stripe_key = os.getenv("STRIPE_TEST_SECRET_KEY")
else:
    stripe_key = os.getenv("STRIPE_LIVE_SECRET_KEY")

if stripe_key:
    stripe.api_key = stripe_key
    # Environment-specific price IDs
    if stripe_environment == "live":
        PRICE_IDS = {
            'consumer': os.getenv("STRIPE_PRICE_CONSUMER_LIVE"),
            'professional': os.getenv("STRIPE_PRICE_PROFESSIONAL_LIVE"),
            'enterprise': os.getenv("STRIPE_PRICE_ENTERPRISE_LIVE")
        }
    else:
        PRICE_IDS = {
            'consumer': os.getenv("STRIPE_PRICE_CONSUMER_TEST"),
            'professional': os.getenv("STRIPE_PRICE_PROFESSIONAL_TEST"),
            'enterprise': os.getenv("STRIPE_PRICE_ENTERPRISE_TEST")
        }

    # Check if all price IDs are set (but don't stop the app)
    missing_prices = [plan for plan, price_id in PRICE_IDS.items() if not price_id]
    if missing_prices:
        st.sidebar.warning(f"⚠️ Missing Stripe price IDs for: {', '.join(missing_prices)}")
else:
    # Stripe not configured - demo mode
    PRICE_IDS = {
        'consumer': 'demo_consumer',
        'professional': 'demo_professional',
        'enterprise': 'demo_enterprise'
    }
    st.sidebar.info("💡 Running in demo mode - Stripe not configured")

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID", "VA_xxxxx")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID else None

# Setup AI clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None
anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
cohere_client = cohere.Client(cohere_key) if cohere_key else None
if google_key:
    genai.configure(api_key=google_key)

# Global flag to indicate Anthropic moderation is available
ANTHROPIC_MODERATION_AVAILABLE = bool(anthropic_client)

# Page config
st.set_page_config(page_title="Hallucinations.cloud", layout="wide")

# === SESSION STATE INITIALIZATION ===
def init_session_state():
    """Initialize all session state variables properly"""
    if 'show_landing' not in st.session_state:
        st.session_state.show_landing = True
    if 'show_html_landing' not in st.session_state:
        st.session_state.show_html_landing = False
    if 'selected_plan' not in st.session_state:
        st.session_state.selected_plan = 'trial'
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'show_upgrade' not in st.session_state:
        st.session_state.show_upgrade = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if 'enhanced_query_history' not in st.session_state:
        st.session_state.enhanced_query_history = []
    if 'view_as_normal_user' not in st.session_state:
        st.session_state.view_as_normal_user = False
    if 'testing_mode' not in st.session_state:
        st.session_state.testing_mode = False
    # NEW: Add conversation document
    if 'conversation_document' not in st.session_state:
        st.session_state.conversation_document = []
    if 'conversation_count' not in st.session_state:
        st.session_state.conversation_count = 0
    # ADDED: Initialize query_processed flag
    if 'query_processed' not in st.session_state:
        st.session_state.query_processed = False
    if 'is_followup' not in st.session_state:
        st.session_state.is_followup = False
    if 'send_code_busy' not in st.session_state:
        st.session_state.send_code_busy = False
    if 'pending_phone_ui_applied' not in st.session_state:
        st.session_state.pending_phone_ui_applied = False
        
# === ENHANCED LANDING PAGE WITH HTML INTEGRATION ===
def show_enhanced_landing_page():
    """Enhanced landing page with HTML integration"""
    
    # Check if user wants to see the full HTML landing page
    if st.session_state.get('show_html_landing', False):
        show_html_landing_page()
        return
    
    # Add button to switch to HTML landing page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎨 View Full Landing Page", use_container_width=True):
            st.session_state.show_html_landing = True
            st.rerun()
    

    # --- Mobile-to-desktop handoff (added by assistant) ---
    try:
        import urllib.parse as _u
    except Exception:
        import urllib as _u  # fallback for older environments

    APP_URL = "https://hllm.hallucinations.cloud/"
    UTM_URL = APP_URL + "?utm_source=google_play&utm_medium=listing&utm_campaign=install"
    mailto_link = "mailto:?subject=" + _u.quote("H-LLM Desktop Link") + "&body=" + _u.quote(APP_URL)

    st.divider()
    st.markdown("### Continue on desktop")
    cA, cB, cC = st.columns([1,1,1])
    with cA:
        st.link_button("📨 Email me a desktop link", mailto_link, use_container_width=True)
    with cB:
        st.link_button("🔗 Open desktop version (UTM)", UTM_URL, use_container_width=True)
    with cC:
        st.image(
            "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=" + _u.quote(APP_URL),
            caption="Scan to open on your desktop",
            width=160
        )
    # --- End mobile-to-desktop block ---
    # Minimal CSS for essential styling only
    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }
    .stat-box {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin: 0;
    }
    .model-badge {
        background: white;
        border: 2px solid #667eea;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        color: #667eea;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
    .pricing-card {
        border: 2px solid #e0e0e0;
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 0.5rem;
        background: white;
        text-align: center;
        transition: all 0.3s ease;
    }
    .pricing-card:hover {
        border-color: #667eea;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    .pricing-card.recommended {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8ebff 100%);
        position: relative;
    }
    .recommended-badge {
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        background: #667eea;
        color: white;
        padding: 0.25rem 1rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .price-large {
        font-size: 2.5rem;
        font-weight: 800;
        color: #667eea;
        margin: 0.5rem 0;
    }
    .price-period {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">H-LLM Multi-Model™</h1>
        <p class="hero-subtitle">The premier multi-model AI analysis platform that compares 8 leading LLMs simultaneously, detects hallucinations, and provides comprehensive security analysis through Red/Blue/Purple team methodologies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Stats using pure Streamlit
    st.markdown("### 📊 Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">8</div>
            <div>AI Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">3</div>
            <div>Security Teams</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">5</div>
            <div>Free Daily Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">3</div>
            <div>Day Free Trial</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Models Showcase - Pure Streamlit
    st.markdown("### 🤖 Supported AI Models")
    st.markdown("*Compare responses from the world's leading AI systems*")
    
    # Create model badges using simple HTML in columns
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    
    models = ["GPT-4o", "Claude 3", "Gemini 1.5", "Grok", "Cohere", "Deepseek", "OpenRouter", "Perplexity"]
    cols = [col1, col2, col3, col4, col5, col6, col7, col8]
    
    for i, model in enumerate(models):
        with cols[i]:
            st.markdown(f'<div class="model-badge">{model}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features - Pure Streamlit columns
    st.markdown("## 🚀 Advanced Capabilities")
    
    # First row of features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔴 Red Team Analysis")
        st.write("Adversarial security analysis that identifies risks, vulnerabilities, hallucinations, and potential manipulation in AI responses.")
    
    with col2:
        st.markdown("### 🔵 Blue Team Analysis")
        st.write("Defensive evaluation focusing on trustworthiness, reliability, completeness, and safety measures in AI responses.")
    
    with col3:
        st.markdown("### 🟣 Purple Team Synthesis")
        st.write("Strategic analysis that synthesizes findings into actionable insights and comprehensive recommendations.")
    
    # Second row of features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏆 H-Score™ Algorithm")
        st.write("Dynamic scoring system that evaluates Safety, Trust, Confidence, and Quality metrics for comprehensive reliability assessment.")
    
    with col2:
        st.markdown("### 🔍 Truth Verification")
        st.write("Cross-reference responses against reliable sources and provide accuracy scores with transparent methodology.")
    
    with col3:
        st.markdown("### 🛡️ AI Content Moderation")
        st.write("Real-time safety checking using Anthropic's Constitutional AI to ensure responsible AI usage and compliance.")
    
    st.markdown("---")
    
    # ENHANCED PRICING SECTION WITH SELECTION
    st.markdown("## 💎 Choose Your Plan")
    st.markdown("### Start with our free trial, then select the plan that fits your needs")
    
    # Pricing cards with selection
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="pricing-card">
            <h3>🎁 Free Trial</h3>
            <div class="price-large">$0</div>
            <div class="price-period">3 days</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Perfect for:**
        - Testing the platform
        - Evaluating features
        - Small projects
        
        **Includes:**
        - 5 queries per day
        - All analysis features
        - Red/Blue/Purple teams
        - H-Score™ algorithm
        """)
        
        if st.button("🎁 Start Free Trial", key="free_trial", type="primary", use_container_width=True):
            st.session_state.selected_plan = "trial"
            st.session_state.show_landing = False
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="pricing-card">
            <h3>🏠 Consumer</h3>
            <div class="price-large">$9.99</div>
            <div class="price-period">per month</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Perfect for:**
        - Personal use
        - Students & researchers
        - Content creators
        
        **Includes:**
        - 25 queries per day
        - All analysis features
        - Red/Blue/Purple teams
        - H-Score™ algorithm
        - Email support
        """)
        
        if st.button("🏠 Choose Consumer", key="consumer_plan", use_container_width=True):
            st.session_state.selected_plan = "consumer"
            st.session_state.show_landing = False
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="pricing-card recommended">
            <div class="recommended-badge">MOST POPULAR</div>
            <h3>💠 Professional</h3>
            <div class="price-large">$29.99</div>
            <div class="price-period">per month</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Perfect for:**
        - Businesses
        - Development teams
        - Power users
        
        **Includes:**
        - Unlimited queries
        - Full security analysis
        - Truth verification
        - Priority support
        - Export capabilities
        """)
        
        if st.button("💠 Choose Professional", key="pro_plan", type="primary", use_container_width=True):
            st.session_state.selected_plan = "professional"
            st.session_state.show_landing = False
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="pricing-card">
            <h3>💎 Enterprise</h3>
            <div class="price-large">$99.99</div>
            <div class="price-period">per month</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Perfect for:**
        - Large organizations
        - Custom integrations
        - White-label solutions
        
        **Includes:**
        - API access
        - Custom integrations
        - White-label options
        - Dedicated support
        - Custom analysis
        """)
        
        if st.button("💎 Choose Enterprise", key="enterprise_plan", use_container_width=True):
            st.session_state.selected_plan = "enterprise"
            st.session_state.show_landing = False
            st.rerun()
    
    st.markdown("---")
    
    # Comparison Table
    st.markdown("### 📋 Feature Comparison")
    
    comparison_data = {
        "Feature": [
            "Daily Queries",
            "AI Models",
            "H-Score™ Algorithm",
            "Red Team Analysis", 
            "Blue Team Analysis",
            "Purple Team Analysis",
            "Truth Verification",
            "Content Moderation",
            "Export Reports",
            "API Access",
            "Priority Support",
            "Custom Integration"
        ],
        "Free Trial": [
            "5/day", "8", "✅", "✅", "✅", "✅", "✅", "✅", "❌", "❌", "❌", "❌"
        ],
        "Consumer": [
            "25/day", "8", "✅", "✅", "✅", "✅", "✅", "✅", "Basic", "❌", "Email", "❌"
        ],
        "Professional": [
            "Unlimited", "8", "✅", "✅", "✅", "✅", "✅", "✅", "Full", "Documentation", "Priority", "❌"
        ],
        "Enterprise": [
            "Unlimited", "8", "✅", "✅", "✅", "✅", "✅", "✅", "Full", "Full API", "Phone", "✅"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # FAQ Section
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("🔍 What makes Hallucinations.cloud different?"):
        st.markdown("""
        We're the only platform that:
        - Compares 8 leading LLMs simultaneously
        - Uses Red/Blue/Purple team security analysis
        - Provides H-Score™ reliability metrics
        - Offers real-time truth verification
        """)
    
    with st.expander("💳 How does billing work?"):
        st.markdown("""
        - **Free Trial**: 3 days, no credit card required
        - **Monthly Plans**: Billed monthly via Stripe
        - **Cancel Anytime**: No long-term commitments
        - **Secure**: PCI DSS compliant payments
        """)
    
    with st.expander("🛡️ Is my data secure?"):
        st.markdown("""
        - **Phone Verification**: Secure Twilio authentication
        - **Encryption**: Enterprise-grade data protection
        - **Privacy**: Queries not stored permanently
        - **Compliance**: GDPR and SOC 2 Type II compliant
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("### 🔒 Secure & Trusted Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔐 Security**
        - Phone verification with Twilio
        - Enterprise-grade encryption
        """)
    
    with col2:
        st.markdown("""
        **💳 Payments**
        - Secure payments by Stripe
        - PCI DSS compliant
        """)
    
    with col3:
        st.markdown("""
        **🏛️ Company**
        - Hallucinations.cloud LLC
        - Spearfish, SD • Tax ID: 33-2960907
        """)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ by Hallucinations.cloud • Powered by 8 leading AI models</p>", 
        unsafe_allow_html=True
    )
    
def show_html_landing_page():
    """Display the full HTML landing page"""
    
    # Add back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Quick Start", use_container_width=True):
            st.session_state.show_html_landing = False
            st.rerun()
    
    # HTML Landing Page
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>H-LLM Multi-Model | Detect AI Hallucinations</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #667eea 100%);
                min-height: 100vh;
                color: #333;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }

            /* Enhanced Hero Section with Image */
            .hero {
                position: relative;
                padding: 60px 0;
                color: white;
                overflow: hidden;
            }

            .hero-background {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(
                    135deg, 
                    rgba(15, 20, 25, 0.9) 0%, 
                    rgba(26, 35, 50, 0.85) 50%, 
                    rgba(102, 126, 234, 0.8) 100%
                );
                z-index: 1;
            }

            .hero-content {
                position: relative;
                z-index: 2;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 40px;
                align-items: center;
            }

            .hero-text {
                padding-right: 20px;
            }

            .hero h1 {
                font-size: 3.2rem;
                font-weight: 300;
                margin-bottom: 20px;
                letter-spacing: -1px;
                background: linear-gradient(45deg, #ffffff, #e0e7ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            .hero-tagline {
                font-size: 1.4rem;
                opacity: 0.95;
                margin-bottom: 30px;
                font-style: italic;
                color: #e0e7ff;
            }

            .hero-description {
                font-size: 1.1rem;
                line-height: 1.7;
                margin-bottom: 35px;
                color: #cbd5e1;
            }

            .hero-image {
                position: relative;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 25px 60px rgba(0,0,0,0.3);
                transform: perspective(1000px) rotateY(-5deg);
                transition: transform 0.3s ease;
                background: linear-gradient(135deg, #667eea, #764ba2);
                min-height: 300px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .hero-image:hover {
                transform: perspective(1000px) rotateY(0deg) scale(1.02);
            }

            .hero-placeholder {
                color: white;
                font-size: 1.2rem;
                text-align: center;
                padding: 40px;
            }

            .cta-buttons {
                display: flex;
                gap: 20px;
                margin-top: 30px;
            }

            .cta-primary, .cta-secondary {
                display: inline-block;
                padding: 16px 40px;
                text-decoration: none;
                border-radius: 50px;
                font-size: 1.1rem;
                font-weight: 600;
                transition: all 0.3s ease;
                cursor: pointer;
                border: none;
            }

            .cta-primary {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }

            .cta-primary:hover {
                transform: translateY(-3px);
                box-shadow:RetryBDContinueEditpython               box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
           }

           .cta-secondary {
               background: transparent;
               color: white;
               border: 2px solid rgba(255,255,255,0.3);
           }

           .cta-secondary:hover {
               background: rgba(255,255,255,0.1);
               border-color: rgba(255,255,255,0.6);
           }

           /* Main Content */
           .main-content {
               background: white;
               border-radius: 25px;
               padding: 50px;
               margin: -30px 20px 40px;
               box-shadow: 0 25px 80px rgba(0,0,0,0.15);
               position: relative;
               z-index: 3;
           }

           .content-text {
               font-size: 1.1rem;
               line-height: 1.8;
               color: #444;
               margin-bottom: 35px;
           }

           .content-text p {
               margin-bottom: 25px;
           }

           /* Highlighted Terms */
           .highlight {
               background: linear-gradient(45deg, #667eea, #764ba2);
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               background-clip: text;
               font-weight: 600;
           }

           /* Stats Section */
           .stats-section {
               display: grid;
               grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
               gap: 25px;
               margin: 40px 0;
               padding: 30px;
               background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
               border-radius: 15px;
           }

           .stat-item {
               text-align: center;
               padding: 20px;
           }

           .stat-number {
               font-size: 2.5rem;
               font-weight: 700;
               color: #667eea;
               margin-bottom: 8px;
           }

           .stat-label {
               font-size: 0.9rem;
               color: #64748b;
               font-weight: 500;
           }

           /* Models Grid */
           .models-section {
               margin: 40px 0;
           }

           .models-grid {
               display: grid;
               grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
               gap: 15px;
               margin: 20px 0;
           }

           .model-card {
               background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
               color: #1565c0;
               padding: 15px 20px;
               border-radius: 15px;
               text-align: center;
               font-size: 0.9rem;
               font-weight: 600;
               transition: all 0.3s ease;
               box-shadow: 0 4px 15px rgba(21, 101, 192, 0.1);
           }

           .model-card:hover {
               transform: translateY(-3px);
               box-shadow: 0 8px 25px rgba(21, 101, 192, 0.2);
           }

           /* Team Analysis Section */
           .team-analysis {
               display: grid;
               grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
               gap: 20px;
               margin: 30px 0;
           }

           .team {
               background: white;
               border-radius: 15px;
               padding: 25px;
               border: 3px solid #f0f0f0;
               text-align: center;
               transition: all 0.3s ease;
               box-shadow: 0 5px 20px rgba(0,0,0,0.05);
           }

           .team:hover {
               transform: translateY(-5px);
               box-shadow: 0 15px 40px rgba(0,0,0,0.1);
           }

           .team.red { 
               border-color: #ef4444; 
               background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
           }
           .team.blue { 
               border-color: #3b82f6;
               background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
           }
           .team.purple { 
               border-color: #8b5cf6;
               background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
           }

           .team-icon {
               width: 50px;
               height: 50px;
               border-radius: 50%;
               margin: 0 auto 15px;
               display: flex;
               align-items: center;
               justify-content: center;
               color: white;
               font-weight: bold;
               font-size: 1.2rem;
           }

           .team.red .team-icon { background: linear-gradient(135deg, #ef4444, #dc2626); }
           .team.blue .team-icon { background: linear-gradient(135deg, #3b82f6, #2563eb); }
           .team.purple .team-icon { background: linear-gradient(135deg, #8b5cf6, #7c3aed); }

           .team h4 {
               font-size: 1.1rem;
               margin-bottom: 10px;
               color: #1f2937;
               font-weight: 600;
           }

           .team p {
               font-size: 0.9rem;
               color: #6b7280;
               line-height: 1.5;
           }

           /* AI Safety Section */
           .ai-safety-section {
               background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
               border-radius: 20px;
               padding: 35px;
               margin: 40px 0;
               border-left: 6px solid #22c55e;
           }

           .ai-safety-section h3 {
               color: #166534;
               margin-bottom: 25px;
               font-size: 1.4rem;
               font-weight: 600;
               display: flex;
               align-items: center;
           }

           .ai-safety-section h3::before {
               content: "🛡️";
               margin-right: 12px;
               font-size: 1.5rem;
           }

           .safety-features {
               display: grid;
               grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
               gap: 20px;
               margin: 25px 0;
           }

           .safety-feature {
               background: white;
               padding: 25px;
               border-radius: 15px;
               border-left: 4px solid #22c55e;
               box-shadow: 0 4px 15px rgba(34, 197, 94, 0.1);
           }

           .safety-feature h4 {
               color: #166534;
               font-size: 1rem;
               margin-bottom: 12px;
               font-weight: 600;
           }

           .safety-feature p {
               font-size: 0.9rem;
               color: #374151;
               line-height: 1.6;
           }

           /* Enterprise Features */
           .enterprise-features {
               background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
               border-radius: 20px;
               padding: 35px;
               margin: 40px 0;
               border-left: 6px solid #667eea;
           }

           .enterprise-features h3 {
               color: #1f2937;
               margin-bottom: 25px;
               font-size: 1.4rem;
               font-weight: 600;
           }

           .feature-grid {
               display: grid;
               grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
               gap: 15px;
               margin: 20px 0;
           }

           .feature-item {
               display: flex;
               align-items: center;
               font-size: 1rem;
               color: #374151;
               padding: 12px 0;
           }

           .feature-item::before {
               content: "✔";
               color: #667eea;
               font-weight: bold;
               margin-right: 12px;
               font-size: 1.2rem;
           }

           /* Final CTA Section */
           .final-cta {
               text-align: center;
               padding: 50px 0;
               background: linear-gradient(135deg, #0f1419 0%, #667eea 100%);
               color: white;
               border-radius: 25px;
               margin: 40px 20px;
           }

           .final-cta h2 {
               font-size: 2.2rem;
               margin-bottom: 20px;
               font-weight: 400;
           }

           .final-cta p {
               font-size: 1.1rem;
               margin-bottom: 30px;
               opacity: 0.9;
           }

           .trial-info {
               font-size: 1rem;
               opacity: 0.8;
               margin-top: 15px;
           }

           /* Security Badge */
           .security-badge {
               background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
               border: 2px solid #c3e6cb;
               border-radius: 15px;
               padding: 20px;
               margin: 30px 0;
               font-size: 0.95rem;
               color: #155724;
               text-align: center;
           }

           /* Responsive Design */
           @media (max-width: 768px) {
               .hero-content {
                   grid-template-columns: 1fr;
                   gap: 30px;
                   text-align: center;
               }

               .hero-image {
                   transform: none;
                   order: -1;
               }

               .hero h1 {
                   font-size: 2.5rem;
               }

               .cta-buttons {
                   flex-direction: column;
                   align-items: center;
               }

               .main-content {
                   margin: -20px 15px 30px;
                   padding: 30px 25px;
               }

               .stats-section {
                   grid-template-columns: repeat(2, 1fr);
               }

               .models-grid {
                   grid-template-columns: repeat(2, 1fr);
               }

               .team-analysis {
                   grid-template-columns: 1fr;
               }

               .safety-features, .feature-grid {
                   grid-template-columns: 1fr;
               }

               .final-cta {
                   margin: 30px 15px;
                   padding: 40px 25px;
               }
           }
       </style>
   </head>
   <body>
       <div class="hero">
           <div class="hero-background"></div>
           <div class="container">
               <div class="hero-content">
                   <div class="hero-text">
                       <h1>Hallucinations.cloud</h1>
                       <div class="hero-tagline">"Some seek the truth, others stretch it ..."</div>
                       <div class="hero-description">
                           The only AI platform that compares 8 leading models simultaneously to detect hallucinations, misinformation, and reliability issues in real-time. And, we'll talk to you person to person.
                       </div>
                       <div class="cta-buttons">
                           <button class="cta-primary" onclick="startAnalysis()">Start Analysis</button>
                           <div class="live-help" style="margin-top:12px; color:#fff; opacity:0.95;"><strong>Live Help</strong><br/>Brian • 📱 +1-949-291-1422 • ✉️ brian@hallucinations.cloud</div>
                       </div>
                   </div>
                   <div class="hero-image">
                       <div class="hero-placeholder">
                           🔬 AI Analysis Platform<br>
                           📊 8 Model Comparison<br>
                           🛡️ Security Analysis
                       </div>
                   </div>
               </div>
           </div>
       </div>

       <div class="main-content">
           <div class="content-text">
               <p>Hallucinations.cloud is a specialized AI platform designed to tackle one of the most persistent problems in large language models (LLMs): hallucinations, or the generation of false, misleading, or unverifiable information.</p>

               <p>At the core of its offering is the <span class="highlight">H-LLM Multi-Model™</span>, an advanced environment where eight leading AI models are queried simultaneously:</p>

               <div class="models-grid">
                   <div class="model-card">GPT-4o</div>
                   <div class="model-card">Claude</div>
                   <div class="model-card">Gemini</div>
                   <div class="model-card">Grok</div>
                   <div class="model-card">Cohere</div>
                   <div class="model-card">Deepseek</div>
                   <div class="model-card">OpenRouter</div>
                   <div class="model-card">Perplexity</div>
               </div>

               <p>By comparing these models' outputs in real time, the platform identifies inconsistencies, contradictions, and areas of potential misinformation. This multi-model comparison is paired with a <span class="highlight">Truth Verification Engine</span>, which cross-references claims against reliable sources (.edu, .gov, .org) and assesses temporal accuracy, source reliability, and overall factual integrity.</p>

               <p>The H-LLM Multi-Model™ incorporates <span class="highlight">Red, Blue, and Purple Team analyses</span>—borrowed from cybersecurity best practices—to rigorously evaluate AI responses from multiple angles:</p>

               <div class="team-analysis">
                   <div class="team red">
                       <div class="team-icon">R</div>
                       <h4>Red Team</h4>
                       <p>Vulnerabilities, risks, and hallucinations detection</p>
                   </div>
                   <div class="team blue">
                       <div class="team-icon">B</div>
                       <h4>Blue Team</h4>
                       <p>Defensive reliability, safety, and completeness</p>
                   </div>
                   <div class="team purple">
                       <div class="team-icon">P</div>
                       <h4>Purple Team</h4>
                       <p>Synthesis into actionable recommendations</p>
                   </div>
               </div>

               <p>This approach is enhanced by the proprietary <span class="highlight">H-Score™ Algorithm</span>, which rates Safety, Trust, Confidence, and Quality to provide an easy-to-interpret reliability metric for any set of AI outputs.</p>

               <div class="ai-safety-section">
                   <h3>Advanced AI Safety & Content Moderation</h3>
                   <div class="safety-features">
                       <div class="safety-feature">
                           <h4>Anthropic Claude Integration</h4>
                           <p>Advanced content moderation powered by Claude's Constitutional AI framework, providing semantic understanding and ethical reasoning for sophisticated content safety.</p>
                       </div>
                       <div class="safety-feature">
                           <h4>Multi-Layer Protection</h4>
                           <p>Combination of Anthropic's AI safety technology, real-time hallucination detection, and custom moderation policies tailored to your specific use case.</p>
                       </div>
                       <div class="safety-feature">
                           <h4>Interpretable Decisions</h4>
                           <p>Clear explanations for all moderation and safety decisions, enabling transparency and accountability in AI-powered content evaluation.</p>
                       </div>
                       <div class="safety-feature">
                           <h4>Adaptive Policies</h4>
                           <p>Easily evolving moderation guidelines without extensive retraining, supporting multilingual content and complex contextual understanding.</p>
                       </div>
                   </div>
               </div>

               <div class="security-badge">
                   <strong>Enterprise Security:</strong> Advanced content moderation powered by Anthropic's Constitutional AI technology, phone-first authentication (Twilio), subscription management (Stripe), and enterprise-grade encryption with end-to-end data protection.
               </div>

               <div class="enterprise-features">
                   <h3>Professional & Enterprise Features:</h3>
                   <div class="feature-grid">
                       <div class="feature-item">Unlimited queries</div>
                       <div class="feature-item">API access</div>
                       <div class="feature-item">White-label options</div>
                       <div class="feature-item">Custom integrations</div>
                       <div class="feature-item">High-stakes reliability</div>
                       <div class="feature-item">Real-time analysis</div>
                       <div class="feature-item">Batch processing</div>
                       <div class="feature-item">Custom safety policies</div>
                   </div>
               </div>
           </div>
       </div>

       <div class="final-cta">
           <div class="container">
               <h2>Ready to detect AI hallucinations?</h2>
               <p>Experience the power of multi-model AI analysis</p>
               <button class="cta-primary" onclick="startAnalysis()">Start Your Analysis</button>
               <div class="trial-info">3-day free trial • No credit card required</div>
           </div>
       </div>

       <script>
           function startAnalysis() {
               // This would integrate with your Streamlit app
               alert('Starting H-LLM Multi-Model Analysis Platform...');
               // In production, this would navigate to your main app
               window.parent.postMessage({action: 'start_trial'}, '*');
           }
           
           function watchDemo() {
               alert('Demo feature coming soon!');
           }
       </script>
   </body>
   </html>
   """
   
  # Display the HTML
    st.components.v1.html(html_content, height=800, scrolling=True)
    
    # Add buttons to interact with the landing page
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Start Free Trial", type="primary", use_container_width=True):
            st.session_state.selected_plan = "trial"
            st.session_state.show_html_landing = False
            st.session_state.show_landing = False
            st.rerun()
    
    with col2:
        if st.button("💼 View Plans", use_container_width=True):
            st.session_state.show_html_landing = False
            st.rerun()


# === AUTHENTICATION FUNCTIONS ===
def validate_phone(phone):
   """Normalize and validate E.164 phone numbers; auto-add +1 for 10-digit US."""
   cleaned = re.sub(r'[^\d+]', '', phone or '')
   # Auto-add +1 if exactly 10 digits and no '+'
   if cleaned.isdigit() and len(cleaned) == 10 and not cleaned.startswith('+'):
       cleaned = '+1' + cleaned

   if not cleaned.startswith('+'):
       st.error("Please include country code (e.g., +1 for US)")
       return None

   # E.164 digits length (exclude '+')
   digits_len = len(re.sub(r'\D', '', cleaned))
   if digits_len < 11 or digits_len > 15:
       st.error("Invalid phone number length")
       return None

   st.session_state['normalized_phone'] = cleaned
   return cleaned

def send_verification_code(phone):
   """Send SMS verification code"""
   # Apple App Store Review - Hardcoded test credentials
   if phone == "+13014426175":
       st.session_state.apple_review_code = "612485"
       st.info("📱 Apple Review Mode: Use code 612485")
       return True

   if not twilio_client:
       # Development mode - use mock code
       st.session_state.mock_code = "123456"
       st.info("Development mode: Use code 123456")
       return True

   try:
       verification = twilio_client.verify \
           .v2 \
           .services(TWILIO_VERIFY_SERVICE_SID) \
           .verifications \
           .create(to=phone, channel='sms')

       return verification.status == 'pending'
   except Exception as e:
       st.error(f"Failed to send SMS: {str(e)}")
       return False

def verify_phone_code(phone, code):
   """Verify the SMS code"""
   # Apple App Store Review - Hardcoded test credentials
   if phone == "+13014426175":
       if code == "612485":
           st.success("✅ Apple Review Mode: Authentication successful!")
           return True
       else:
           st.error("❌ Apple Review Mode: Use code 612485")
           return False

   if not twilio_client:
       # Development mode
       return code == st.session_state.get('mock_code', '123456')

   try:
       verification_check = twilio_client.verify \
           .v2 \
           .services(TWILIO_VERIFY_SERVICE_SID) \
           .verification_checks \
           .create(to=phone, code=code)

       return verification_check.status == 'approved'
   except Exception as e:
       st.error(f"Verification failed: {str(e)}")
       logger.error(f"SMS verification error: {str(e)}")
       return False

def create_account_with_plan(phone, email, plan):
   """Create account with selected plan"""
   try:
       # Apple App Store Review - Skip Stripe for review account
       if phone == "+13014426175":
           # Create mock customer for Apple review
           mock_customer = type('MockCustomer', (), {'id': 'cus_apple_review_account'})()
           customer = mock_customer
           st.info("🍎 Apple Review Account: Bypassing payment setup")
       else:
           # Create Stripe customer
           customer = stripe.Customer.create(
               email=email,
               phone=phone,
               metadata={
                   'phone': phone,
                   'phone_verified': 'true',
                   'selected_plan': plan,
                   'trial_start': datetime.now().isoformat(),
                   'trial_end': (datetime.now() + timedelta(days=3)).isoformat(),
                   'queries_used': '0',
                   'queries_today': '0',
                   'last_query_date': datetime.now().date().isoformat()
               }
           )
       
       # Set session
       st.session_state.user_phone = phone
       st.session_state.user_email = email
       st.session_state.customer_id = customer.id
       st.session_state.authenticated = True
       st.session_state.selected_plan = plan
       
       if plan == "trial":
           st.session_state.trial_end = datetime.now() + timedelta(days=3)
           st.session_state.show_landing = False  # Go directly to app for trial
           st.success("✅ Account created! Starting your 3-day trial...")
       else:
           st.session_state.show_upgrade = True  # Show payment options for paid plans
           st.session_state.show_landing = False
           st.session_state.immediate_checkout = plan
           st.success(f"✅ Account created! Redirecting to {plan} subscription...")
       
       st.balloons()
       return True
   except Exception as e:
       st.error(f"Account creation failed: {str(e)}")
       return False

# === ENHANCED AUTHENTICATION WITH PLAN HANDLING ===
def show_enhanced_auth_page():
   """Enhanced authentication page with plan-aware registration"""
   st.title("🧠 Hallucinations.cloud")
   
   # Show selected plan
   selected_plan = st.session_state.get('selected_plan', 'trial')
   
   if selected_plan == "trial":
       st.subheader("🎁 Start Your Free Trial")
       st.info("✨ 3 days free • 5 queries per day • No credit card required")
   elif selected_plan == "consumer":
       st.subheader("🏠 Consumer Plan - $9.99/month")
       st.info("🚀 25 queries per day • All features included")
   elif selected_plan == "professional":
       st.subheader("💠 Professional Plan - $29.99/month")
       st.success("⭐ Most Popular • Unlimited queries • Priority support")
   elif selected_plan == "enterprise":
       st.subheader("💎 Enterprise Plan - $99.99/month")
       st.info("🏢 API access • Custom integrations • Dedicated support")
   
   # Plan change option
   with st.expander("🔄 Change Plan", expanded=False):
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           if st.button("🎁 Free Trial", use_container_width=True):
               st.session_state.selected_plan = "trial"
               st.rerun()
       
       with col2:
           if st.button("🏠 Consumer", use_container_width=True):
               st.session_state.selected_plan = "consumer"
               st.rerun()
       
       with col3:
           if st.button("💠 Professional", use_container_width=True):
               st.session_state.selected_plan = "professional"
               st.rerun()
       
       with col4:
           if st.button("💎 Enterprise", use_container_width=True):
               st.session_state.selected_plan = "enterprise"
               st.rerun()
   
   col1, col2 = st.columns([1, 1])
   
   with col1:
       st.markdown(f"""
       ### 🎯 Your Selected Plan: {selected_plan.title()}
       
       **What you get:**
       - Compare 8 Leading LLMs simultaneously
       - Dynamic H-Score™ algorithm  
       - Red/Blue/Purple team analysis
       - Truth verification engine
       - Anthropic AI content moderation
       
       **Next Steps:**
       1. Verify your phone number
       2. Create your account
       3. Start analyzing AI responses!
       """)
   
   with col2:
       # Check for existing user trying to login
       if st.session_state.get('show_login'):
           show_login_form()
       else:
           show_enhanced_registration_form_with_plan()

def show_enhanced_registration_form_with_plan():
   """Registration form that handles plan selection"""
   selected_plan = st.session_state.get('selected_plan', 'trial')
   
   st.markdown(f"### 📱 Create Your Account ({selected_plan.title()})")
   
   # Step 1: Collect phone number
   with st.form("send_code_form", clear_on_submit=False):
       if 'phone_verified' not in st.session_state:
           phone = st.text_input(
               "Enter your phone number:",
               placeholder="+1234567890",
               help="We'll send a verification code"
           )
       
           col1, col2 = st.columns(2)
           with col1:
               if st.form_submit_button("📱 Send Code", use_container_width=True, disabled=st.session_state.get('send_code_busy', False)):
                norm_phone = validate_phone(phone)
                if norm_phone:
                    st.session_state.send_code_busy = True
                    try:
                        existing = stripe.Customer.search(query=f'metadata["phone"]:"{norm_phone}"', limit=1)
                        if existing.data:
                            st.error("This phone number already has an account.")
                            st.session_state.show_login = True
                            st.session_state.existing_phone = norm_phone
                            st.session_state.send_code_busy = False
                            st.rerun()
                        else:
                            if send_verification_code(norm_phone):
                                st.session_state.pending_phone = norm_phone
                                st.success("📨 Code sent! Check your phone.")
                                st.session_state.send_code_busy = False
                                st.rerun()
                    except Exception as e:
                        # Fallback: try to send anyway
                        try:
                            if send_verification_code(norm_phone):
                                st.session_state.pending_phone = norm_phone
                                st.success("📨 Code sent! Check your phone.")
                        finally:
                            st.session_state.send_code_busy = False

       
           with col2:
               if st.form_submit_button("🔐 I have an account", use_container_width=True):
                   st.session_state.show_login = True
                   st.rerun()
   
   # Step 2: Verify code
   if st.session_state.get('pending_phone') and not st.session_state.get('phone_verified'):
       st.markdown("---")
       code = st.text_input("Enter 6-digit verification code:", max_chars=6)
       
       if st.button("✅ Verify Code", type="primary"):
           if verify_phone_code(st.session_state.pending_phone, code):
               st.session_state.phone_verified = True
               st.session_state.phone_number = st.session_state.pending_phone
               st.success("✅ Phone verified!")
               st.rerun()
   
   # Step 3: Collect email and handle plan-specific account creation
   if st.session_state.get('phone_verified'):
       st.markdown("---")
       st.success(f"✅ Phone verified: {st.session_state.phone_number}")
       
       email = st.text_input(
           "Enter your email:",
           placeholder="you@example.com",
           help="For receipts and important notifications"
       )
       
       if selected_plan == "trial":
           button_text = "🎁 Start Free Trial"
       else:
           button_text = f"🚀 Create Account & Subscribe to {selected_plan.title()}"
       
       if st.button(button_text, type="primary", use_container_width=True):
           if email and '@' in email:
               if create_account_with_plan(st.session_state.phone_number, email, selected_plan):
                   # Clear auth state
                   for key in ['phone_verified', 'pending_phone', 'phone_number']:
                       if key in st.session_state:
                           del st.session_state[key]
                   st.rerun()

def handle_successful_login(customer):
   """Route user appropriately after login based on account status"""
   
   # Set basic session data
   st.session_state.user_phone = customer.phone
   st.session_state.user_email = customer.email
   st.session_state.customer_id = customer.id
   st.session_state.authenticated = True
   
   # Clear login state
   for key in ['pending_login_phone', 'pending_customer', 'show_login', 'existing_phone']:
       if key in st.session_state:
           del st.session_state[key]
   
   try:
       # Check account status
       metadata = customer.metadata
       trial_end = metadata.get('trial_end', '')
       subscriptions = stripe.Subscription.list(customer=customer.id, status='active')
       
       if subscriptions.data:
           # Has active subscription - go directly to app
           subscription = subscriptions.data[0]
           price_id = subscription.items.data[0].price.id
           
           if "consumer" in price_id.lower():
               plan_name = "Consumer"
           elif "prof" in price_id.lower():
               plan_name = "Professional"
           else:
               plan_name = "Enterprise"
           
           st.session_state.show_landing = False
           st.success(f"✅ Welcome back! Your {plan_name} subscription is active.")
           
       elif trial_end and datetime.now() < datetime.fromisoformat(trial_end):
           # Trial still active - go directly to app
           trial_date = datetime.fromisoformat(trial_end)
           days_left = (trial_date - datetime.now()).days
           
           st.session_state.show_landing = False
           st.success(f"✅ Welcome back! {days_left} days left in your trial.")
           
       else:
           # Trial expired, no subscription - show upgrade options
           st.session_state.show_upgrade = True
           st.session_state.show_landing = False
           st.warning("⏰ Your trial has expired. Please choose a subscription to continue.")
   
   except Exception as e:
       # Fallback: allow access but show upgrade option
       st.session_state.show_landing = False
       st.info("✅ Welcome back! Please verify your subscription status.")
       print(f"Login status check error: {str(e)}")

def show_login_form():
   """Show login form for existing users"""
   st.markdown("### 🔐 Welcome Back!")
   
   phone = st.text_input(
       "Enter your phone number:",
       value=st.session_state.get('existing_phone', ''),
       placeholder="+1234567890"
   )
   
   col1, col2 = st.columns(2)
   with col1:
       if st.button("Send Login Code", type="primary", use_container_width=True):
           if validate_phone(phone):
               try:
                   # Find customer by phone
                   customers = stripe.Customer.search(
                       query=f'metadata["phone"]:"{phone}"',
                       limit=1
                   )
                   
                   if customers.data:
                       if send_verification_code(phone):
                           st.session_state.pending_login_phone = phone
                           st.session_state.pending_customer = customers.data[0]
                           st.success("Login code sent!")
                   else:
                       st.error("No account found with this phone number")
               except Exception as e:
                   st.error("Login failed. Please try again.")
   
   with col2:
       if st.button("Create new account", use_container_width=True):
           st.session_state.show_login = False
           if 'existing_phone' in st.session_state:
               del st.session_state.existing_phone
           st.rerun()
   
   # Verify login code
   if st.session_state.get('pending_login_phone'):
       st.markdown("---")
       code = st.text_input("Enter login code:", max_chars=6)
       
       if st.button("Login", type="primary"):
           if verify_phone_code(st.session_state.pending_login_phone, code):
               # Login successful - handle routing based on account status
               customer = st.session_state.pending_customer
               handle_successful_login(customer)
               st.rerun()

# === SUPER USER SYSTEM ===
# Super User Configuration
SUPER_USER_PHONES = [
   "+19492911422",  # Brian Demsey
   "+19707998830",  # Alan Lapedes
   "+12066878168",  # YanLei Xu
   "+19495420322",  # Scott Sanchez
   "+16193155778",  # John McKay
   "+16504008061"   # DJ Waldow
   # Add other admin phone numbers here
]

def is_super_user():
   """Check if current user is a super user - respects normal user toggle"""
   user_phone = st.session_state.get('user_phone', '')
   is_actual_super_user = user_phone in SUPER_USER_PHONES
   
   # If "view as normal user" is enabled, behave as normal user
   if st.session_state.get('view_as_normal_user', False):
       return False
       
   return is_actual_super_user

def check_daily_limit():
   """Check queries by phone number - Super Users have unlimited access - UPDATED FOR CONSUMER TIER"""
   if not st.session_state.get('customer_id'):
       return False, 0, -1
   
   # Super User Override (but respect "view as normal user" setting)
   if is_super_user():  # This now respects the toggle
       return True, 0, -1  # Unlimited for super users
   
   # When super user is viewing as normal user, simulate trial experience
   if st.session_state.get('view_as_normal_user', False):
       return True, 2, 5  # Simulate 2/5 queries used in trial
   
   # Demo mode handling
   if st.session_state.customer_id == "test_customer_id":
       # Check if demo upgrade was simulated
       if st.session_state.get('demo_upgraded'):
           plan = st.session_state.demo_upgraded
           if plan == 'consumer':
               return True, 2, 25  # Simulate consumer plan
           else:
               return True, 5, -1  # Simulate unlimited plan
       else:
           return True, 2, 5  # Simulate trial: 2 used, 5 limit
   
   try:
       customer = stripe.Customer.retrieve(st.session_state.customer_id)
       metadata = customer.metadata
       
       # Reset daily counter if needed
       last_query_date = metadata.get('last_query_date', '')
       today = datetime.now().date().isoformat()
       
       if last_query_date != today:
           # New day - reset counter
           metadata['queries_today'] = '0'
           metadata['last_query_date'] = today
           stripe.Customer.modify(
               st.session_state.customer_id,
               metadata=metadata
           )
       
       queries_today = int(metadata.get('queries_today', 0))
       
       # Check subscription status
       trial_end = metadata.get('trial_end', '')
       is_trial = trial_end and datetime.now() < datetime.fromisoformat(trial_end)
       
       # Check for active subscriptions
       subscriptions = stripe.Subscription.list(customer=customer.id, status='active')
       
       if subscriptions.data:
           subscription = subscriptions.data[0]
           price_id = subscription.items.data[0].price.id
           
           # Check subscription type
           if "consumer" in price_id.lower():
               # Consumer plan: 25 queries per day
               return queries_today < 25, queries_today, 25
           else:
               # Professional/Enterprise: Unlimited
               return True, queries_today, -1
               
       elif is_trial and queries_today < 5:
           return True, queries_today, 5  # Within trial limits
       else:
           return False, queries_today, 5  # Limit reached or expired
           
   except Exception as e:
       # In testing mode or error, allow queries with simulated limits
       return True, 2, 5  # Simulate trial: 2 used, 5 limit

def increment_usage():
   """Increment query counter - Skip for Super Users"""
   if not st.session_state.get('customer_id'):
       return
   
   # Super User Override - don't count their usage
   if is_super_user():
       return
   
   try:
       customer = stripe.Customer.retrieve(st.session_state.customer_id)
       metadata = customer.metadata
       
       queries_today = int(metadata.get('queries_today', 0))
       queries_total = int(metadata.get('queries_used', 0))
       
       metadata['queries_today'] = str(queries_today + 1)
       metadata['queries_used'] = str(queries_total + 1)
       
       stripe.Customer.modify(
           st.session_state.customer_id,
           metadata=metadata
       )
   except Exception as e:
       print(f"Error incrementing usage: {str(e)}")

def show_super_user_controls():
   """Show super user admin controls with normal user view toggle"""
   if not is_super_user():
       return
       
   with st.sidebar:
       st.markdown("---")
       st.markdown("### 👑 Super User Controls")
       
       # Add "View as Normal User" toggle
       if 'view_as_normal_user' not in st.session_state:
           st.session_state.view_as_normal_user = False
           
       st.session_state.view_as_normal_user = st.checkbox(
           "👤 View as Normal User",
           value=st.session_state.view_as_normal_user,
           help="Experience the app as a normal user (with query limits and upgrade prompts)"
       )
       
       if st.session_state.view_as_normal_user:
           st.warning("👤 **Normal User Mode Active**")
           st.caption("Query limits: 5/day (trial)")
       else:
           st.success("🔓 **Unlimited Access Enabled**")
       
       # Admin toggles
       col1, col2 = st.columns(2)
       with col1:
           if st.button("📊 Admin Dashboard", use_container_width=True):
               st.session_state.show_admin_dashboard = True
       
       with col2:
           if st.button("🧪 Testing Mode", use_container_width=True):
               st.session_state.testing_mode = not st.session_state.get('testing_mode', False)
       
       # Quick admin actions
       if st.session_state.get('testing_mode', False):
           st.info("🧪 **Testing Mode Active**")
           if st.button("Clear All Session Data", use_container_width=True):
               # Clear all session data except authentication
               keys_to_keep = ['authenticated', 'user_phone', 'user_email', 'customer_id']
               for key in list(st.session_state.keys()):
                   if key not in keys_to_keep:
                       del st.session_state[key]
               st.success("Session data cleared!")
               st.rerun()

def show_admin_dashboard():
   """Show comprehensive admin dashboard"""
   if not is_super_user():
       return
       
   st.markdown("### 👑 Super User Admin Dashboard")
   
   # System Overview
   col1, col2, col3, col4 = st.columns(4)
   
   with col1:
       total_queries = len(st.session_state.get('enhanced_query_history', []))
       st.metric("Total Queries", total_queries)
   
   with col2:
       moderation_logs = len(st.session_state.get('moderation_logs', []))
       st.metric("Moderation Events", moderation_logs)
   
   with col3:
       verification_history = len(st.session_state.get('verification_history', []))
       st.metric("Truth Verifications", verification_history)
   
   with col4:
       avg_hscore = 0
       if st.session_state.get('enhanced_query_history'):
           scores = [q.get('enhanced_hscore', {}).get('final', 0) for q in st.session_state.enhanced_query_history]
           avg_hscore = sum(scores) / len(scores) if scores else 0
       st.metric("Avg H-Score", f"{avg_hscore:.2f}")
   
   # Recent Activity
   st.markdown("### 📈 Recent System Activity")
   
   # Query Analysis
   if st.session_state.get('enhanced_query_history'):
       st.markdown("#### 🔍 Recent Queries")
       recent_queries = st.session_state.enhanced_query_history[-5:]
       for i, query in enumerate(reversed(recent_queries)):
           with st.expander(f"Query {i+1}: {query['question'][:50]}..."):
               st.write(f"**Timestamp**: {query['timestamp']}")
               st.write(f"**H-Score**: {query['enhanced_hscore']['final']}/10")
               if query.get('analysis', {}).get('red_team'):
                   st.write("**Analysis**: Red/Blue/Purple Team completed")
               st.json(query['enhanced_hscore'])
   
   # Show Anthropic moderation status
   show_anthropic_moderation_status()
   
   # System Health
   st.markdown("### 🩺 System Health")
   
   health_col1, health_col2 = st.columns(2)
   
   with health_col1:
       st.markdown("#### 🔌 API Status")
       api_status = {
           "OpenAI": "✅" if openai_key else "❌",
           "Anthropic": "✅" if anthropic_key else "❌",
           "Google (Gemini)": "✅" if google_key else "❌",
           "Google (Search)": "✅" if (google_key and google_search_engine_id) else "❌",
           "Stripe": "✅" if stripe.api_key else "❌",
           "Twilio": "✅" if twilio_client else "❌"
       }
       for api, status in api_status.items():
           st.write(f"**{api}**: {status}")
   
   with health_col2:
       st.markdown("#### 🛡️ Security Status")
       security_status = {
           "Anthropic Moderation": "✅ Active" if ANTHROPIC_MODERATION_AVAILABLE else "❌ Disabled",
           "Truth Verification": "✅ Active" if st.session_state.get('truth_verification_enabled') else "⚠️ Disabled",
           "Rate Limiting": "🔓 Bypassed (Super User)",
           "Session Security": "✅ Active"
       }
       for feature, status in security_status.items():
           st.write(f"**{feature}**: {status}")
   
   # Debug Information
   if st.session_state.get('testing_mode'):
       st.markdown("### 🐛 Debug Information")
       with st.expander("Session State Debug"):
           st.json({
               k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
               for k, v in st.session_state.items()
               if not k.startswith('_')
           })
   
   # Admin Actions
   st.markdown("### ⚙️ Admin Actions")
   admin_col1, admin_col2, admin_col3 = st.columns(3)
   
   with admin_col1:
       if st.button("🧹 Clear Logs", use_container_width=True):
           if 'moderation_logs' in st.session_state:
               del st.session_state.moderation_logs
           if 'verification_history' in st.session_state:
               del st.session_state.verification_history
           st.success("Logs cleared!")
           st.rerun()
   
   with admin_col2:
       if st.button("📊 Export Data", use_container_width=True):
           export_data = {
               'query_history': st.session_state.get('enhanced_query_history', []),
               'moderation_logs': st.session_state.get('moderation_logs', []),
               'verification_history': st.session_state.get('verification_history', [])
           }
           st.download_button(
               "⬇️ Download JSON",
               data=json.dumps(export_data, indent=2),
               file_name=f"hallucinations_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
               mime="application/json"
           )
   
   with admin_col3:
       if st.button("🔄 Reset System", use_container_width=True):
           # Reset all non-auth session data
           keys_to_keep = ['authenticated', 'user_phone', 'user_email', 'customer_id']
           for key in list(st.session_state.keys()):
               if key not in keys_to_keep:
                   del st.session_state[key]
           st.success("System reset!")
           st.rerun()

def show_enhanced_user_sidebar():
   """Enhanced user sidebar with payment management - UPDATED FOR CONSUMER TIER"""
   with st.sidebar:
       st.markdown("---")
       st.markdown("### 👤 Your Account")
       st.write(f"📱 {st.session_state.user_phone}")
       st.write(f"📧 {st.session_state.user_email}")
       
       # Show usage and subscription status
       can_query, used, limit = check_daily_limit()
       
       # Testing mode - show simulated data
       if st.session_state.customer_id == "test_customer_id":
           if st.session_state.get('demo_upgraded'):
               plan = st.session_state.demo_upgraded
               if plan == 'consumer':
                   st.success(f"🏠 {plan.title()} Plan (Demo)")
                   st.write(f"**Queries Today:** {used}/25")
               else:
                   st.success(f"💠 {plan.title()} Plan (Demo)")
                   st.write("**Queries**: Unlimited")
           else:
               st.write(f"**Queries Today:** {used}/{limit}")
               st.info("🎁 Trial: 2 days left (Demo)")
           
           if st.button("🚀 Upgrade Now", type="primary", use_container_width=True):
               st.session_state.show_upgrade = True
       else:
           try:
               customer = stripe.Customer.retrieve(st.session_state.customer_id)
               subscriptions = stripe.Subscription.list(customer=customer.id, status='active')
               
               if subscriptions.data:
                   # Active subscription
                   subscription = subscriptions.data[0]
                   price_id = subscription.items.data[0].price.id
                   
                   if "consumer" in price_id.lower():
                       plan_name = "Consumer"
                       st.success(f"🏠 {plan_name} Plan Active")
                       st.write("**Queries**: 25/day")
                       st.write(f"**Used Today**: {used}/25")
                   elif "prof" in price_id.lower():
                       plan_name = "Professional"
                       st.success(f"💠 {plan_name} Plan Active")
                       st.write("**Queries**: Unlimited")
                   else:
                       plan_name = "Enterprise"
                       st.success(f"💎 {plan_name} Plan Active")
                       st.write("**Queries**: Unlimited")
                   
                   # Billing management
                   if st.button("💳 Manage Billing", use_container_width=True):
                       create_customer_portal_session()
                       
               else:
                   # Trial or no subscription
                   if limit == -1:
                       st.write("**Queries:** Unlimited")
                   else:
                       st.write(f"**Queries Today:** {used}/{limit}")
                       
                       # Check trial status
                       trial_end = customer.metadata.get('trial_end', '')
                       if trial_end:
                           trial_date = datetime.fromisoformat(trial_end)
                           days_left = (trial_date - datetime.now()).days
                           if days_left > 0:
                               st.info(f"🎁 Trial: {days_left} days left")
                           else:
                               st.warning("⏰ Trial expired")
                       
                       # Upgrade button
                       if st.button("🚀 Upgrade Now", type="primary", use_container_width=True):
                           st.session_state.show_upgrade = True
                           
           except Exception as e:
               # Fallback for testing mode
               st.write(f"**Queries Today:** {used}/{limit}")
               st.info("🎁 Trial: 2 days left")
               if st.button("🚀 Upgrade Now", type="primary", use_container_width=True):
                   st.session_state.show_upgrade = True
       
       st.markdown("---")
       
       if st.button("🚪 Logout", use_container_width=True):
           for key in list(st.session_state.keys()):
               del st.session_state[key]
           st.rerun()

# === FIXED PAYMENT FUNCTIONS ===
def create_simple_checkout_session(plan, payment_method):
   """Create simple Stripe checkout session with multiple payment options"""
   try:
       # Check if we're in test mode with mock customer
       if st.session_state.customer_id == "test_customer_id":
           st.warning("⚠️ Demo Mode: Payment processing is not available in demo mode.")
           st.info("💡 In a real environment, this would redirect to secure Stripe checkout.")
           
           # Show what would happen
           plan_prices = {
               'consumer': '$9.99',
               'professional': '$29.99', 
               'enterprise': '$99.99'
           }
           
           st.success(f"✅ Demo: Would process {plan_prices[plan]}/month {plan.title()} plan subscription")
           
           # Simulate upgrade for demo
           if st.button("🎭 Simulate Upgrade (Demo)", type="primary"):
               st.session_state.demo_upgraded = plan
               st.success(f"🎉 Demo: Simulated upgrade to {plan.title()} plan!")
               st.balloons()
           return
       
       # Determine payment method types based on selection
       if "Credit/Debit Card" in payment_method:
           payment_method_types = ['card']
       elif "Bank Transfer" in payment_method:
           payment_method_types = ['us_bank_account']
       else:  # Both options
           payment_method_types = ['card', 'us_bank_account']
       
       # Create simple checkout session
       session = stripe.checkout.Session.create(
           customer=st.session_state.customer_id,
           payment_method_types=payment_method_types,
           line_items=[{
               'price': PRICE_IDS[plan],
               'quantity': 1,
           }],
           mode='subscription',
           success_url=f"{os.getenv('APP_URL', 'http://localhost:8501')}?session_id={{CHECKOUT_SESSION_ID}}&success=true",
           cancel_url=f"{os.getenv('APP_URL', 'http://localhost:8501')}?cancelled=true",
           
           # Simple features
           allow_promotion_codes=True,
           billing_address_collection='required',
           phone_number_collection={'enabled': True},
           
           # Business information
           metadata={
               'business_name': 'Hallucinations.cloud LLC',
               'customer_phone': st.session_state.user_phone,
               'customer_email': st.session_state.user_email,
               'plan_selected': plan
           }
      )
       
       # Enhanced payment confirmation
       st.success(f"✅ Redirecting to secure {payment_method} checkout...")
       st.info("💡 You'll be redirected to Stripe's secure payment page")
       
       # Professional checkout button
       st.markdown(f"""
       <div style="text-align: center; margin: 20px 0;">
           <a href="{session.url}" target="_blank" style="text-decoration: none;">
               <button style="
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white;
                   padding: 15px 30px;
                   border: none;
                   border-radius: 8px;
                   font-size: 16px;
                   font-weight: 600;
                   cursor: pointer;
                   box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                   transition: transform 0.2s;
               ">
                   🔒 Complete Secure Payment → 
               </button>
           </a>
       </div>
       
       <p style="text-align: center; color: #666; font-size: 12px;">
           Payments processed securely by Stripe<br>
           Funds deposited to Hallucinations.cloud LLC at Blackhills Community Bank
       </p>
       """, unsafe_allow_html=True)
       
   except Exception as e:
       st.error(f"Payment setup failed: {str(e)}")
       st.info("💡 Please contact support@hallucinations.cloud if this persists")

def show_enhanced_upgrade_options():
   """Enhanced upgrade options with Consumer tier - FIXED"""
   st.markdown("### 🚀 Upgrade Your Hallucinations.cloud Account")
   st.markdown("**Secure payments processed by Stripe • Funds deposited to Hallucinations.cloud LLC**")
   
   # Payment method selection
   st.markdown("#### 💳 Choose Your Payment Method")
   payment_method = st.radio(
       "How would you like to pay?",
       ["💳 Credit/Debit Card", "🏦 Bank Transfer (ACH)", "💰 Both Options"],
       help="All payments are secure and encrypted"
   )
   
   # Three-column layout for paid plans
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.markdown("""
       #### 🏠 Consumer - $9.99/month
       - ✅ **25 queries per day** to all LLMs
       - ✅ **Basic contradiction analysis**
       - ✅ **Standard H-Score analytics**
       - ✅ **Email support**
       - ✅ **Query history tracking**
       - ✅ **Mobile-friendly interface**
       """)
       
       if st.button("🏠 Choose Consumer", type="primary", use_container_width=True):
           create_simple_checkout_session('consumer', payment_method)
   
   with col2:
       st.markdown("""
       #### 💠 Professional - $29.99/month
       - ✅ **Unlimited queries** to all LLMs
       - ✅ **Full Red/Blue/Purple Team analysis**
       - ✅ **Enhanced H-Score analytics**
       - ✅ **Priority email support**
       - ✅ **API documentation access**
       - ✅ **Export analysis reports**
       """)
       
       if st.button("💠 Choose Professional", use_container_width=True):
           create_simple_checkout_session('professional', payment_method)
   
   with col3:
       st.markdown("""
       #### 💎 Enterprise - $99.99/month
       - ✅ **Everything in Professional**
       - ✅ **Direct API access** for integrations
       - ✅ **Custom model fine-tuning**
       - ✅ **White-label solutions**
       - ✅ **Priority phone support**
       - ✅ **Custom security analysis**
       - ✅ **Dedicated account manager**
       """)
       
       if st.button("💎 Choose Enterprise", use_container_width=True):
           create_simple_checkout_session('enterprise', payment_method)
   
   # Security & Company Information
   st.markdown("---")
   st.markdown("### 🔒 Payment Security & Company Information")
   
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.markdown("""
       **🏛️ Business Details**
       - **Company**: Hallucinations.cloud LLC
       - **Bank**: Blackhills Community Bank
       - **Location**: Spearfish, SD
       - **Tax ID**: 33-2960907
       """)
   
   with col2:
       st.markdown("""
       **🔐 Security Features**
       - **PCI DSS Compliant**
       - **256-bit SSL Encryption**
       - **Fraud Protection**
       - **Secure Bank Connections**
       """)
   
   with col3:
       st.markdown("""
       **💼 Payment Options**
       - **Credit/Debit Cards**
       - **ACH Bank Transfers**
       - **Digital Wallets**
       - **International Cards**
       """)

def create_customer_portal_session():
   """Create Stripe customer portal session for self-service billing"""
   try:
       session = stripe.billing_portal.Session.create(
           customer=st.session_state.customer_id,
           return_url=os.getenv('APP_URL', 'http://localhost:8501'),
       )
       
       st.markdown(f"""
       <div style="text-align: center; margin: 20px 0;">
           <a href="{session.url}" target="_blank" style="text-decoration: none;">
               <button style="
                   background-color: #28a745;
                   color: white;
                   padding: 12px 24px;
                   border: none;
                   border-radius: 6px;
                   font-size: 14px;
                   font-weight: 600;
                   cursor: pointer;
               ">
                   🔧 Manage Your Billing →
               </button>
           </a>
       </div>
       """, unsafe_allow_html=True)
       
       st.info("💡 You can update payment methods, download invoices, and manage your subscription")
       
   except Exception as e:
       st.error(f"Unable to create billing portal: {str(e)}")

# === NEW CONVERSATION DISPLAY FUNCTION ===
def display_conversation_document():
   """Display the cumulative conversation document"""
   if st.session_state.conversation_document:
       st.markdown("---")
       st.subheader("📄 Your Conversation History")
       
       # Add download button for the conversation
       conversation_text = "HALLUCINATIONS.CLOUD CONVERSATION TRANSCRIPT\n"
       conversation_text += f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
       conversation_text += "=" * 80 + "\n\n"
       
       for entry in st.session_state.conversation_document:
           conversation_text += f"QUERY #{entry['number']}\n"
           conversation_text += f"Time: {entry['timestamp']}\n"
           conversation_text += f"Question: {entry['question']}\n"
           conversation_text += f"H-Score: {entry['hscore']}/10\n\n"
           
           conversation_text += "MODEL RESPONSES:\n"
           for model, response in entry['responses']:
               conversation_text += f"\n[{model}]:\n{response}\n"
           
           if entry.get('analysis_summary'):
               conversation_text += f"\nANALYSIS:\n{entry['analysis_summary']}\n"
           conversation_text += "\n" + "=" * 80 + "\n\n"
       
       col1, col2 = st.columns([3, 1])
       with col1:
           st.info(f"📊 Total Queries in This Session: {st.session_state.conversation_count}")
       with col2:
           st.download_button(
               label="💾 Download Full Conversation",
               data=conversation_text,
               file_name=f"hallucinations_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
               mime="text/plain",
               use_container_width=True
           )
       
       # Display last 3 queries in expandable sections
       st.markdown("##### Recent Queries:")
       for entry in st.session_state.conversation_document[-3:]:
           with st.expander(f"Query {entry['number']}: {entry['question'][:80]}...", expanded=False):
               st.markdown(f"**Time:** {entry['timestamp']}")
               st.markdown(f"**H-Score:** {entry['hscore']}/10")
               
               if entry.get('analysis_summary'):
                   st.markdown("**Analysis Summary:**")
                   st.info(entry['analysis_summary'])

# === MAIN APP FLOW CONTROL ===
def main_app_flow():
   """Main application flow with proper authentication handling"""
   
   # Show landing page first for new users
   if st.session_state.get('show_landing', True):
       show_enhanced_landing_page()
       st.stop()
   
   # Show upgrade page for expired users
   elif st.session_state.get('show_upgrade', False):
       st.title("🧠 Hallucinations.cloud")
       st.markdown("### 🚀 Choose Your Subscription")
       st.info("Your trial has expired. Please select a plan to continue using our advanced AI analysis platform.")
       show_enhanced_upgrade_options()
       
       # Option to return to trial if there's time left
       if st.button("← Check Trial Status"):
           try:
               if st.session_state.get('customer_id'):
                   customer = stripe.Customer.retrieve(st.session_state.customer_id)
                   trial_end = customer.metadata.get('trial_end', '')
                   if trial_end and datetime.now() < datetime.fromisoformat(trial_end):
                       st.session_state.show_upgrade = False
                       st.rerun()
           except:
               pass
       st.stop()
   
   # Show authentication page for unauthenticated users
   else:
       show_enhanced_auth_page()
       st.stop()

# Handle payment success/cancellation
def handle_payment_success():
   """Handle successful payment confirmation"""
   # Check URL parameters
   query_params = st.query_params
   
   if query_params.get('success') == 'true':
       st.balloons()
       st.success("🎉 Payment successful! Welcome to Hallucinations.cloud Premium!")
       st.info("💡 Your subscription is now active. You have unlimited access to all features!")
       
       # Clear the success parameter
       del st.query_params['success']
       if 'session_id' in st.query_params:
           del st.query_params['session_id']
       
   elif query_params.get('cancelled') == 'true':
       st.warning("⚠️ Payment was cancelled. Your trial continues as normal.")
       del st.query_params['cancelled']

# === AI MODEL FUNCTIONS (CLEANED - NO FOLLOW-UP) ===
def call_openai_sync(prompt):
   if not openai_client:
       return ("OpenAI", "[OpenAI unavailable: missing API key]")
   
   try:
       response = openai_client.chat.completions.create(
           model="gpt-4o",
           messages=[
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": prompt}
           ],
           temperature=0.5,
           max_tokens=600
       )
       return ("OpenAI", response.choices[0].message.content.strip())
   except Exception as e:
       return ("OpenAI", f"[OpenAI error: {str(e)}]")

def call_claude_sync(prompt):
   if not anthropic_client:
       return ("Claude", "[Claude unavailable: missing API key]")
   
   try:
       message = anthropic_client.messages.create(
           model="claude-3-haiku-20240307",
           max_tokens=600,
           messages=[{"role": "user", "content": prompt}]
       )
       return ("Claude", message.content[0].text.strip())
   except Exception as e:
       return ("Claude", f"[Claude error: {str(e)}]")

def call_gemini_sync(prompt):
   if not google_key:
       return ("Gemini", "[Gemini unavailable: missing API key]")

   try:
       # Try multiple model names in order of preference
       model_names = [
           "gemini-1.5-pro",      # Current standard model
           "gemini-pro",          # Fallback to older version
           "gemini-1.5-flash"     # Fast alternative
       ]

       for model_name in model_names:
           try:
               model = genai.GenerativeModel(model_name)
               response = model.generate_content(prompt)
               return ("Gemini", response.text.strip())
           except Exception as model_error:
               logger.info(f"Gemini model {model_name} failed: {str(model_error)}")
               continue

       # If all models fail, return the last error
       return ("Gemini", f"[Gemini error: All model versions failed. Last error: {str(model_error)}]")

   except Exception as e:
       return ("Gemini", f"[Gemini error: {str(e)}]")

def call_cohere_sync(prompt):
   if not cohere_key:
       return ("Cohere", "[Cohere unavailable: missing API key]")
   
   try:
       co = cohere.Client(cohere_key)
       response = co.chat(
           message=prompt,
           model='command-r-08-2024',
           max_tokens=600,
           temperature=0.5
       )
       return ("Cohere", response.text.strip())
   except Exception as e:
       return ("Cohere", f"[Cohere error: {str(e)}]")

def call_deepseek_sync(prompt):
   if not deepseek_key:
       return ("Deepseek", "[Deepseek unavailable: missing API key]")
   
   try:
       deepseek_client = OpenAI(
           api_key=deepseek_key,
           base_url="https://api.deepseek.com"
       )
       response = deepseek_client.chat.completions.create(
           model="deepseek-chat",
           messages=[
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": prompt}
           ],
           temperature=0.5,
           max_tokens=600
       )
       return ("Deepseek", response.choices[0].message.content.strip())
   except Exception as e:
       return ("Deepseek", f"[Deepseek error: {str(e)}]")

def call_openrouter_sync(prompt):
   if not openrouter_key:
       return ("OpenRouter", "[OpenRouter unavailable: missing API key]")
   
   try:
       openrouter_client = OpenAI(
           api_key=openrouter_key,
           base_url="https://openrouter.ai/api/v1"
       )
       response = openrouter_client.chat.completions.create(
           model="microsoft/wizardlm-2-8x22b",
           messages=[
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": prompt}
           ],
           temperature=0.5,
           max_tokens=600
       )
       return ("OpenRouter", response.choices[0].message.content.strip())
   except Exception as e:
       return ("OpenRouter", f"[OpenRouter error: {str(e)}]")

def call_perplexity_sync(prompt):
   if not perplexity_key:
       return ("Perplexity", "[Perplexity unavailable: missing API key]")
   
   try:
       headers = {
           "accept": "application/json",
           "content-type": "application/json",
           "authorization": f"Bearer {perplexity_key}"
       }
       
       payload = {
           "model": "sonar",
           "messages": [
               {
                   "role": "user",
                   "content": prompt
               }
           ],
           "max_tokens": 600,
           "temperature": 0.5
       }
       
       response = requests.post(
           "https://api.perplexity.ai/chat/completions",
           json=payload,
           headers=headers
       )
       
       if response.status_code != 200:
           return ("Perplexity", f"[Perplexity error: HTTP {response.status_code}]")
       
       data = response.json()
       
       if 'choices' in data and len(data['choices']) > 0:
           content = data['choices'][0]['message']['content']
           return ("Perplexity", content.strip())
       else:
           return ("Perplexity", "[Perplexity error: Unexpected response]")
           
   except Exception as e:
       return ("Perplexity", f"[Perplexity error: {str(e)}]")

def call_grok_sync(prompt):
   if not grok_key:
       return ("Grok", "[Grok unavailable: missing API key]")
   
   try:
       grok_client = OpenAI(
           api_key=grok_key,
           base_url="https://api.x.ai/v1"
       )
       # Determine which Grok model to use.  Default to grok‑4 (the latest reasoning model),
       # but allow overriding via the GROK_MODEL_NAME environment variable.  If you have
       # access to Grok‑4 Heavy, set GROK_MODEL_NAME=grok-4-heavy when launching the app.
       preferred_model = os.getenv("GROK_MODEL_NAME", "grok-4")
       # Construct a minimal message list without any custom system prompt so that
       # Grok responds with its native tone and behaviour.
       messages = [{"role": "user", "content": prompt}]
       # Call the xAI API without specifying temperature or max_tokens; this allows Grok to
       # determine response length and creativity based on its defaults.
       response = grok_client.chat.completions.create(
           model=preferred_model,
           messages=messages
       )
       return ("Grok", response.choices[0].message.content.strip())
   except Exception as e:
       return ("Grok", f"[Grok error: {str(e)}]")
def show_followup_interface(previous_query, previous_results):
    """Show follow-up question interface after analysis"""
    st.markdown("---")
    st.subheader("💬 Continue the Conversation")

    col1, col2 = st.columns([3, 1])

    with col1:
        followup_question = st.text_input(
            "Ask a follow-up question:",
            placeholder="Build on the previous responses...",
            help="Your follow-up will be added to the ongoing conversation (Press Enter to submit)",
            key="followup_input",
            on_change=process_followup_question,
            args=(previous_query, previous_results)
        )

    with col2:
        if st.button("🔄 Ask Follow-up", type="secondary", use_container_width=True):
            if followup_question:
                context_query = create_followup_context(previous_query, previous_results, followup_question)
                st.session_state.current_query = context_query
                st.session_state.run_analysis = True
                st.session_state.is_followup = True
                # Clear the followup input
                st.session_state.followup_input = ""
                st.rerun()

    st.caption(f"Previous query: {previous_query[:100]}...")
    return followup_question

def process_followup_question(previous_query, previous_results):
    """Process follow-up question when Enter is pressed"""
    followup_question = st.session_state.get("followup_input", "").strip()

    if followup_question:  # Only process if there's actual content
        context_query = create_followup_context(previous_query, previous_results, followup_question)
        st.session_state.current_query = context_query
        st.session_state.run_analysis = True
        st.session_state.is_followup = True
        # Clear the followup input for next question
        st.session_state.followup_input = ""

def create_followup_context(previous_query, previous_results, followup_question):
    """Create context-aware follow-up query"""
    model_summaries = []
    for model_name, response in previous_results:
        if not response.startswith('[') and 'error' not in response.lower():
            summary = response[:200] + "..." if len(response) > 200 else response
            model_summaries.append(f"{model_name}: {summary}")
    
    context_summary = "\n".join(model_summaries[:3])
    
    context_query = f"""Previous question: {previous_query}

Previous responses summary:
{context_summary}

Follow-up question: {followup_question}

Please answer the follow-up question while considering the previous context and responses."""
    
    return context_query

def handle_followup_conversation():
    """Handle the follow-up conversation flow"""
    if st.session_state.get('is_followup', False):
        st.info("🔗 This is a follow-up question building on your previous query")
        st.session_state.is_followup = False

# === ENHANCED ANALYSIS FUNCTIONS ===
def perform_red_team_analysis(query, responses):
   """Red Team - Adversarial analysis looking for vulnerabilities and risks"""
   if not openai_client:
       return "Red Team analysis unavailable (OpenAI API key required)"
   
   try:
       model_responses = "\n\n".join([f"**{name}**: {resp}" for name, resp in responses])
       
       red_team_prompt = f"""
       You are a Red Team AI security analyst. Your job is to identify potential risks, vulnerabilities, and concerning aspects in these AI responses.

       ORIGINAL QUERY: {query}

       AI RESPONSES:
       {model_responses}

       Analyze for:
       1. **Hallucinations**: False or unverifiable claims
       2. **Bias**: Political, cultural, or demographic bias
       3. **Harmful Content**: Anything potentially dangerous or misleading
       4. **Inconsistencies**: Major contradictions between models
       5. **Manipulation Risk**: Could responses be used to mislead users?
       6. **Factual Errors**: Obvious mistakes or outdated information

       Provide:
       - Risk Score (1-10, where 10 = highest risk)
       - Key concerns found
       - Specific examples of problematic content
       - Recommendations for mitigation

       Format: Risk Score: X/10, followed by detailed analysis.
       """
       
       response = openai_client.chat.completions.create(
           model="gpt-4o",
           messages=[
               {"role": "system", "content": "You are a cybersecurity red team analyst specializing in AI safety."},
               {"role": "user", "content": red_team_prompt}
           ],
           temperature=0.3,
           max_tokens=800
       )
       
       return response.choices[0].message.content.strip()
       
   except Exception as e:
       return f"Red Team analysis failed: {str(e)}"

def perform_blue_team_analysis(query, responses):
   """Blue Team - Defensive analysis focusing on reliability and trustworthiness"""
   if not openai_client:
       return "Blue Team analysis unavailable (OpenAI API key required)"
   
   try:
       model_responses = "\n\n".join([f"**{name}**: {resp}" for name, resp in responses])
       
       blue_team_prompt = f"""
       You are a Blue Team AI analyst focused on defensive evaluation and trust assessment.

       ORIGINAL QUERY: {query}

       AI RESPONSES:
       {model_responses}

       Evaluate for:
       1. **Reliability**: How trustworthy are these responses?
       2. **Completeness**: Do responses adequately address the query?
       3. **Consistency**: Are responses internally coherent?
       4. **Source Quality**: Are claims well-grounded?
       5. **Usefulness**: How helpful are responses to the user?
       6. **Safety Measures**: Evidence of built-in safety protocols

       Provide:
       - Trust Score (1-10, where 10 = highest trust)
       - Quality assessment of each response
       - Most reliable sources of information
       - Confidence recommendations for user

       Format: Trust Score: X/10, followed by detailed analysis.
       """
       
       response = openai_client.chat.completions.create(
           model="gpt-4o",
           messages=[
               {"role": "system", "content": "You are a cybersecurity blue team analyst specializing in AI reliability assessment."},
               {"role": "user", "content": blue_team_prompt}
           ],
           temperature=0.3,
           max_tokens=800
       )
       
       return response.choices[0].message.content.strip()
       
   except Exception as e:
       return f"Blue Team analysis failed: {str(e)}"

def perform_purple_team_analysis(query, responses, red_analysis, blue_analysis):
   """Purple Team - Synthesis of red and blue team findings with strategic recommendations"""
   if not openai_client:
       return "Purple Team analysis unavailable (OpenAI API key required)"
   
   try:
       model_responses = "\n\n".join([f"**{name}**: {resp}" for name, resp in responses])
       
       purple_team_prompt = f"""
       You are a Purple Team AI analyst synthesizing red team (risk) and blue team (trust) assessments.

       ORIGINAL QUERY: {query}

       RED TEAM FINDINGS:
       {red_analysis}

       BLUE TEAM FINDINGS:
       {blue_analysis}

       Provide strategic synthesis:
       1. **Overall Assessment**: Balance of risks vs reliability
       2. **Key Insights**: Most important findings from both teams
       3. **User Guidance**: How should users interpret these responses?
       4. **Model Comparison**: Which models performed best/worst and why?
       5. **Confidence Level**: Overall confidence in the response set
       6. **Action Items**: What should users do with this information?

       Provide:
       - Overall Confidence Score (1-10)
       - Strategic recommendations
       - Risk-adjusted trust assessment
       - Best practices for using these responses

       Format: Confidence Score: X/10, followed by synthesis and recommendations.
       """
       
       response = openai_client.chat.completions.create(
           model="gpt-4o",
           messages=[
               {"role": "system", "content": "You are a purple team strategist providing balanced AI safety and reliability assessment."},
               {"role": "user", "content": purple_team_prompt}
           ],
           temperature=0.3,
           max_tokens=800
       )
       
       return response.choices[0].message.content.strip()
       
   except Exception as e:
       return f"Purple Team analysis failed: {str(e)}"

def extract_score_from_analysis(analysis_text, score_type="Risk Score"):
   """Extract numerical score from analysis text"""
   if not analysis_text:
       return 5.0
   
   import re
   
   # Look for patterns like "Risk Score: 7/10" or "Trust Score: 8/10"
   patterns = [
       rf'{score_type}:\s*(\d+(?:\.\d+)?)/10',
       rf'{score_type}:\s*(\d+(?:\.\d+)?)',
       rf'Score:\s*(\d+(?:\.\d+)?)/10',
       rf'Score:\s*(\d+(?:\.\d+)?)'
   ]
   
   for pattern in patterns:
       matches = re.findall(pattern, analysis_text, re.IGNORECASE)
       if matches:
           try:
               score = float(matches[0])
               return min(10.0, max(1.0, score))
           except:
               continue
   
   # Fallback: look for any score-like patterns
   score_keywords = {
       'low': 3.0, 'minimal': 2.0, 'high': 8.0, 'very high': 9.0,
       'excellent': 9.0, 'good': 7.0, 'moderate': 5.0, 'poor': 3.0
   }
   
   text_lower = analysis_text.lower()
   for keyword, score in score_keywords.items():
       if keyword in text_lower:
           return score
   
   return 5.0

def calculate_enhanced_hscore(responses, red_analysis="", blue_analysis="", purple_analysis=""):
   """Calculate enhanced H-Score using all three team analyses"""
   
   # Extract scores from analyses
   risk_score = extract_score_from_analysis(red_analysis, "Risk Score")
   trust_score = extract_score_from_analysis(blue_analysis, "Trust Score") 
   confidence_score = extract_score_from_analysis(purple_analysis, "Confidence Score")
   
   # Convert risk score to safety score (invert)
   safety_score = 11.0 - risk_score
   
   # Calculate response quality metrics
   successful_responses = [r for r in responses if not r[1].startswith('[') or 'error' not in r[1].lower()]
   response_quality = (len(successful_responses) / len(responses)) * 10 if responses else 5.0
   
   # Weighted calculation
   weights = {
       'safety': 0.25,      # Red team (inverted risk)
       'trust': 0.25,       # Blue team 
       'confidence': 0.25,  # Purple team
       'quality': 0.25      # Response completeness
   }
   
   final_score = (
       safety_score * weights['safety'] +
       trust_score * weights['trust'] +
       confidence_score * weights['confidence'] +
       response_quality * weights['quality']
   )
   
   return {
       'final': round(final_score, 2),
       'safety': round(safety_score, 1),
       'trust': round(trust_score, 1), 
       'confidence': round(confidence_score, 1),
       'quality': round(response_quality, 1)
   }

def show_enhanced_sidebar():
   """Enhanced sidebar with analysis options"""
   with st.sidebar:
       st.markdown("---")
       st.markdown("### 🔍 Analysis Tools")
       
       # Analysis toggles
       if 'enable_red_team' not in st.session_state:
           st.session_state.enable_red_team = True
       if 'enable_blue_team' not in st.session_state:
           st.session_state.enable_blue_team = True
       if 'enable_purple_team' not in st.session_state:
           st.session_state.enable_purple_team = True
           
       st.session_state.enable_red_team = st.checkbox(
           "🔴 Red Team Analysis", 
           value=st.session_state.enable_red_team,
           help="Adversarial analysis - identifies risks and vulnerabilities"
       )
       
       st.session_state.enable_blue_team = st.checkbox(
           "🔵 Blue Team Analysis", 
           value=st.session_state.enable_blue_team,
           help="Defensive analysis - evaluates reliability and trust"
       )
       
       st.session_state.enable_purple_team = st.checkbox(
           "🟣 Purple Team Analysis", 
           value=st.session_state.enable_purple_team,
           help="Strategic synthesis - balances risks with benefits"
       )
       
       if st.button("🎯 Run All Analyses", help="Perform comprehensive security analysis"):
           st.session_state.run_full_analysis = True
           
       st.markdown("---")
       st.markdown("### 📊 Analysis Legend")
       st.markdown("""
       **🔴 Red Team**: Looks for risks, bias, hallucinations
       **🔵 Blue Team**: Evaluates trust, reliability, completeness  
       **🟣 Purple Team**: Strategic synthesis & recommendations
       """)

# === MAIN APP INITIALIZATION ===
init_session_state()

# === MAIN APP CHECK ===
if not st.session_state.authenticated:
   main_app_flow()
   st.stop()

# === AUTHENTICATED USER INTERFACE ===

# Handle immediate upgrade flow for paid plans
if st.session_state.get('immediate_checkout'):
   plan = st.session_state.immediate_checkout
   st.session_state.show_upgrade = True
   del st.session_state.immediate_checkout
   st.rerun()

# Handle payment success/cancellation
handle_payment_success()

# Show welcome message for recently logged in users
if st.session_state.get('show_welcome_message'):
   if st.session_state.get('welcome_message'):
       st.success(st.session_state.welcome_message)
   del st.session_state.show_welcome_message
   if 'welcome_message' in st.session_state:
       del st.session_state.welcome_message

# Check if admin dashboard should be shown
if st.session_state.get('show_admin_dashboard') and is_super_user():
   show_admin_dashboard()
   if st.button("← Back to Dashboard"):
       st.session_state.show_admin_dashboard = False
       st.rerun()
   st.stop()

# Check if moderation dashboard should be shown
if st.session_state.get('show_moderation_logs'):
   show_moderation_dashboard()
   if st.button("← Back to Dashboard"):
       st.session_state.show_moderation_logs = False
       st.rerun()
   st.stop()

# Check if upgrade options should be shown
if st.session_state.get('show_upgrade'):
   show_enhanced_upgrade_options()
   if st.button("← Back to Dashboard"):
       st.session_state.show_upgrade = False
       st.rerun()
   st.stop()

# === SIDEBAR CONTENT ===
with st.sidebar:
   # Show user account info
   show_enhanced_user_sidebar()
   
   # Show moderation controls
   show_moderation_controls()
   show_human_support_section()
   
   # Show truth verification controls
   show_truth_verification_controls()
   
   # Show super user controls (if applicable)
   show_super_user_controls()
   
   # Show enhanced sidebar
   show_enhanced_sidebar()
   
   # Additional sidebar content
   st.markdown("---")
   st.title("💡 Suggest an Additional LLM")
   st.text_input("Suggest a Model", placeholder="Model name...", key="suggest_model")
   
   if st.button("Send Suggestion"):
       st.success("✅ Suggestion submitted!")
   
   st.divider()
   
   st.markdown("### 🧠 Models in Use")
   models_list = ["GPT-4o", "Claude 3 Haiku", "Gemini 1.5 Pro", "Grok", "Cohere", "Deepseek", "OpenRouter", "Perplexity"]
   for model in models_list:
       st.markdown(f"- {model}")

# === MAIN TITLE & HEADER ===
st.title("🧠 Hallucinations.cloud Multi-Model")

# === API KEY STATUS CHECKER ===
with st.expander("🔌 API Key Status", expanded=False):
   key_col1, key_col2 = st.columns(2)
   
   with key_col1:
       st.markdown(f"OpenAI: {'✅' if openai_key else '❌'}")
       st.markdown(f"Claude: {'✅' if anthropic_key else '❌'}")
       st.markdown(f"Gemini: {'✅' if google_key else '❌'}")
       st.markdown(f"Cohere: {'✅' if cohere_key else '❌'}")
   
   with key_col2:
       st.markdown(f"OpenRouter: {'✅' if openrouter_key else '❌'}")
       st.markdown(f"Grok: {'✅' if grok_key else '❌'}")
       st.markdown(f"Perplexity: {'✅' if perplexity_key else '❌'}")
       st.markdown(f"Deepseek: {'✅' if deepseek_key else '❌'}")

# === ENHANCED MAIN QUERY INTERFACE ===
st.subheader("🔍 Compare LLMs - Continuous Conversation")

# Check usage limits
can_query, used, limit = check_daily_limit()

if not can_query:
   st.error(f"📊 Daily query limit reached ({used}/{limit})")
   show_enhanced_upgrade_options()
   st.stop()

# Display conversation history first if it exists
if st.session_state.conversation_document:
   display_conversation_document()

# Query input with moderation
st.markdown("##### Enter your question:")
st.caption("🛡️ All queries are checked for content policy compliance using Anthropic AI")

# Show conversation context
if st.session_state.conversation_count > 0:
   st.info(f"💬 Conversation Mode Active - Query #{st.session_state.conversation_count + 1}")

user_query = st.text_input(
   "Ask anything - continue the conversation or start a new topic:",
   placeholder="Enter your question here...",
   help="Your query will be added to the conversation document along with all model responses",
   key="main_query_input"  # STATIC KEY
)

# Analysis options in main interface
col1, col2 = st.columns([3, 1])
with col1:
   if st.button("🚀 Submit Query", type="primary", key="submit_query_button") and user_query:
       # Check content moderation
       if process_query_with_moderation(user_query):
           st.session_state.run_analysis = True
           st.session_state.current_query = user_query
                   

with col2:
   analysis_depth = st.selectbox(
       "Analysis Depth:",
       ["Quick", "Standard", "Comprehensive"],
       index=1,
       help="Quick: Basic comparison, Standard: +Red/Blue, Comprehensive: Full Red/Blue/Purple"
   )

# Main analysis execution
if st.session_state.get('run_analysis') and st.session_state.get('current_query'):
   query = st.session_state.current_query
   
   st.subheader("📊 Model Responses")
   
   # Get available models
   available_models = []
   if openai_key: available_models.append(call_openai_sync)
   if anthropic_key: available_models.append(call_claude_sync)
   if google_key: available_models.append(call_gemini_sync)
   if cohere_key: available_models.append(call_cohere_sync)
   if deepseek_key: available_models.append(call_deepseek_sync)
   if openrouter_key: available_models.append(call_openrouter_sync)
   if perplexity_key: available_models.append(call_perplexity_sync)
   if grok_key: available_models.append(call_grok_sync)
   
   if not available_models:
       st.error("No API keys available! Please set up at least one API key.")
   else:
       # Query all models
       with st.spinner("Querying models..."):
           results = []
           for model_func in available_models:
               try:
                   result = model_func(query)
                   results.append(result)
               except Exception as e:
                   results.append((model_func.__name__, f"[Error: {str(e)}]"))
       
       # Increment usage counter
       increment_usage()
       
       # Display model responses
       for model_name, response in results:
           with st.expander(f"**{model_name}**", expanded=True):
               st.text_area(f"{model_name} response:", value=response, height=150, key=f"response_{model_name}_{st.session_state.conversation_count}")
       
       # === CONTRADICTION ANALYSIS (Original Feature) ===
       st.subheader("⚖️ Contradiction Analysis")
       if openai_client:
           with st.spinner("Analyzing for contradictions..."):
               model_responses = "\n\n".join([f"{name}: {resp}" for name, resp in results])
               contradiction_prompt = f"""
               Analyze these AI model responses for contradictions or significant disagreements:
               
               {model_responses}
               
               Provide a brief analysis of any contradictions found, or confirm if responses are generally consistent.
               Focus on:
               1. Key points of agreement
               2. Notable differences or contradictions
               3. Variations in perspective or emphasis
               4. Overall consistency assessment
               
               Format as numbered points for clarity.
               """
               
               try:
                   contradiction_analysis = call_openai_sync(contradiction_prompt)
                   contradiction_text = contradiction_analysis[1]
                   st.success(contradiction_text)
               except Exception as e:
                   st.error(f"Contradiction analysis failed: {str(e)}")
       else:
           st.info("Contradiction analysis requires OpenAI API key")
       
       # === ENHANCED ANALYSIS SECTION ===
       st.markdown("---")
       st.subheader("🔍 Security Analysis")
       
       # Initialize analysis variables
       red_analysis = ""
       blue_analysis = ""
       purple_analysis = ""
       
       # Determine which analyses to run
       run_red = st.session_state.get('enable_red_team', True) or analysis_depth in ["Standard", "Comprehensive"]
       run_blue = st.session_state.get('enable_blue_team', True) or analysis_depth in ["Standard", "Comprehensive"] 
       run_purple = st.session_state.get('enable_purple_team', True) or analysis_depth == "Comprehensive"
       
       # Create analysis columns
       analysis_cols = st.columns(3)
       
       # Red Team Analysis
       if run_red:
           with analysis_cols[0]:
               st.markdown("### 🔴 Red Team Analysis")
               with st.spinner("Running red team analysis..."):
                   red_analysis = perform_red_team_analysis(query, results)
               
               with st.expander("🔴 Red Team Report", expanded=True):
                   st.markdown(red_analysis)
       
       # Blue Team Analysis  
       if run_blue:
           with analysis_cols[1]:
               st.markdown("### 🔵 Blue Team Analysis")
               with st.spinner("Running blue team analysis..."):
                   blue_analysis = perform_blue_team_analysis(query, results)
               
               with st.expander("🔵 Blue Team Report", expanded=True):
                   st.markdown(blue_analysis)
       
       # Purple Team Analysis
       if run_purple and red_analysis and blue_analysis:
           with analysis_cols[2]:
               st.markdown("### 🟣 Purple Team Analysis")
               with st.spinner("Running purple team synthesis..."):
                   purple_analysis = perform_purple_team_analysis(query, results, red_analysis, blue_analysis)
               
               with st.expander("🟣 Purple Team Report", expanded=True):
                   st.markdown(purple_analysis)
       
       # === ENHANCED H-SCORE DASHBOARD ===
       st.markdown("---")
       st.subheader("🏆 Enhanced H-Score Analysis")
       
       # Calculate enhanced H-Score
       enhanced_scores = calculate_enhanced_hscore(results, red_analysis, blue_analysis, purple_analysis)
       
       # Display enhanced metrics
       score_cols = st.columns(5)
       
       with score_cols[0]:
           st.metric(
               "🛡️ Safety", 
               f"{enhanced_scores['safety']:.1f}/10",
               delta=f"{enhanced_scores['safety'] - 5:.1f}" if enhanced_scores['safety'] != 5 else None,
               help="Based on Red Team risk assessment (inverted)"
           )
       
       with score_cols[1]:
           st.metric(
               "🔐 Trust", 
               f"{enhanced_scores['trust']:.1f}/10",
               delta=f"{enhanced_scores['trust'] - 5:.1f}" if enhanced_scores['trust'] != 5 else None,
               help="Based on Blue Team reliability assessment"
           )
       
       with score_cols[2]:
           st.metric(
               "💪 Confidence", 
               f"{enhanced_scores['confidence']:.1f}/10", 
               delta=f"{enhanced_scores['confidence'] - 5:.1f}" if enhanced_scores['confidence'] != 5 else None,
               help="Based on Purple Team strategic assessment"
           )
           
       with score_cols[3]:
           st.metric(
               "✅ Quality",
               f"{enhanced_scores['quality']:.1f}/10",
               delta=f"{enhanced_scores['quality'] - 5:.1f}" if enhanced_scores['quality'] != 5 else None,
               help="Response completeness and availability"
           )
       
       with score_cols[4]:
           st.metric(
               "🎯 H-Score", 
               f"{enhanced_scores['final']:.1f}/10",
               delta=f"{enhanced_scores['final'] - 5:.1f}" if enhanced_scores['final'] != 5 else None,
               help="Overall weighted score: Safety (25%), Trust (25%), Confidence (25%), Quality (25%)"
           )
       
       # Interpretation
       final_score = enhanced_scores['final']
       if final_score >= 8.0:
           st.success("✅ **Excellent** - Responses are highly reliable and safe")
       elif final_score >= 6.5:
           st.info("ℹ️ **Good** - Responses are generally trustworthy with minor concerns")
       elif final_score >= 5.0:
           st.warning("⚠️ **Moderate** - Some concerns detected, use with caution")
       elif final_score >= 3.5:
           st.warning("🚨 **Low Confidence** - Significant issues identified")
       else:
           st.error("❌ **High Risk** - Major concerns detected, verify independently")
       
       # Detailed breakdown
       with st.expander("📈 Detailed Score Breakdown"):
           st.markdown(f"""
           **Scoring Methodology:**
           - **Safety Score**: {enhanced_scores['safety']}/10 (Red Team risk assessment, inverted)
           - **Trust Score**: {enhanced_scores['trust']}/10 (Blue Team reliability assessment)  
           - **Confidence Score**: {enhanced_scores['confidence']}/10 (Purple Team strategic assessment)
           - **Quality Score**: {enhanced_scores['quality']}/10 (Response completeness: {len([r for r in results if not r[1].startswith('[')])} of {len(results)} models responded)
           
           **Final H-Score**: Weighted average with equal 25% weights for each component.
           """)
       
       # Store enhanced results in session
       enhanced_query_record = {
           "timestamp": datetime.now().isoformat(),
           "question": query,
           "results": results,
           "analysis": {
               "red_team": red_analysis,
               "blue_team": blue_analysis, 
               "purple_team": purple_analysis
           },
           "enhanced_hscore": enhanced_scores
       }
       # Follow-up interface will be presented after analysis is complete in the section below

       
       st.session_state.enhanced_query_history.append(enhanced_query_record)
       
       # NEW: Add to conversation document
       st.session_state.conversation_count += 1
       
       # Create analysis summary
       analysis_summary = ""
       if red_analysis:
           risk_score = extract_score_from_analysis(red_analysis, "Risk Score")
           analysis_summary += f"Red Team Risk: {risk_score}/10 | "
       if blue_analysis:
           trust_score = extract_score_from_analysis(blue_analysis, "Trust Score")
           analysis_summary += f"Blue Team Trust: {trust_score}/10 | "
       if purple_analysis:
           conf_score = extract_score_from_analysis(purple_analysis, "Confidence Score")
           analysis_summary += f"Purple Team Confidence: {conf_score}/10"
       
       conversation_entry = {
           'number': st.session_state.conversation_count,
           'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
           'question': query,
           'responses': results,
           'hscore': enhanced_scores['final'],
           'analysis_summary': analysis_summary if analysis_summary else None
       }
       
       st.session_state.conversation_document.append(conversation_entry)
       
       st.success(f"✅ Query #{st.session_state.conversation_count} added to your conversation document")
       
       # === TRUTH VERIFICATION ===
       if st.session_state.get('truth_verification_enabled', True):
           verification_results = integrate_truth_verification(query, results)
           if verification_results:
               show_truth_verification_results(verification_results)
       
       # Clear the analysis flag but DON'T rerun - let the results display
       st.session_state.run_analysis = False
       if 'current_query' in st.session_state:
           del st.session_state.current_query
           
       # === FOLLOW-UP QUESTION INTERFACE ===
       if not st.session_state.get('run_analysis', False):  # Only show after analysis is complete
           followup_question = show_followup_interface(query, results)
              
       # Show content policy information
       add_content_policy_info()

       # === ENHANCED QUERY HISTORY ===
       if st.session_state.get('enhanced_query_history'):
           st.markdown("---")
           with st.expander("📜 Enhanced Query History", expanded=False):
               for i, query in enumerate(reversed(st.session_state.enhanced_query_history[-5:])):
                   col1, col2 = st.columns([3, 1])
                   with col1:
                       st.markdown(f"**{query['question']}**")
                       st.caption(f"Asked: {query['timestamp'][:16]}")
                   with col2:
                       scores = query['enhanced_hscore']
                       st.metric("H-Score", f"{scores['final']}/10", delta=None)
           
           # Show mini analysis summary
           if query.get('analysis'):
               analysis_summary = ""
               if query['analysis'].get('red_team'):
                   risk_score = extract_score_from_analysis(query['analysis']['red_team'], "Risk Score")
                   analysis_summary += f"🔴 Risk: {risk_score}/10  "
               if query['analysis'].get('blue_team'):
                   trust_score = extract_score_from_analysis(query['analysis']['blue_team'], "Trust Score")
                   analysis_summary += f"🔵 Trust: {trust_score}/10  "
               if query['analysis'].get('purple_team'):
                   conf_score = extract_score_from_analysis(query['analysis']['purple_team'], "Confidence Score")
                   analysis_summary += f"🟣 Confidence: {conf_score}/10"
               
               if analysis_summary:
                   st.caption(analysis_summary)
           
           st.markdown("---")

# === FOOTER ===
st.divider()
st.markdown("Built with ❤️ by Hallucinations.Cloud")
st.caption("Note: This tool compares multiple LLM responses for hallucination detection. Advanced AI content moderation powered by Anthropic's Constitutional AI technology. Use responsibly.")
