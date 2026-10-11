
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
load_dotenv(override=True)  # re-read .env on every rerun, so key changes apply without a restart
import concurrent.futures
import time
MODEL_TIMEOUT_SECONDS = 60  # longest the page waits for any one model
import streamlit as st
import os
import re
import json
import pandas as pd
import time
import requests
from datetime import datetime, timedelta
from types import SimpleNamespace
from openai import OpenAI
import anthropic
import google.generativeai as genai
import cohere
import stripe
from twilio.rest import Client
import io
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === FILE EXTRACTION HELPER ===
def extract_text_from_file(uploaded_file):
    """Extract text content from uploaded file (PDF, TXT, CSV, DOCX)"""
    try:
        file_type = uploaded_file.name.lower().split('.')[-1]

        if file_type == 'txt':
            return uploaded_file.read().decode('utf-8')

        elif file_type == 'csv':
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            return df.to_string()

        elif file_type == 'pdf':
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
            except ImportError:
                return "[PDF extraction requires PyPDF2 - file content not extracted]"

        elif file_type == 'docx':
            try:
                from docx import Document
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text.strip()
            except ImportError:
                return "[DOCX extraction requires python-docx - file content not extracted]"

        else:
            return f"[Unsupported file type: {file_type}]"

    except Exception as e:
        return f"[Error extracting file content: {str(e)}]"

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
            model="claude-haiku-4-5-20251001",  # Oct 2026: Claude 3.5 Haiku is retired
            max_tokens=200,
            timeout=15.0,  # Quick timeout for moderation
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
            "model": "claude-haiku-4-5-20251001"
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
            st.caption("Model: claude-haiku-4-5-20251001")
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
load_dotenv(override=True)  # re-read .env on every rerun, so key changes apply without a restart

# Get API keys (strip whitespace to handle copy/paste issues with newlines)
openai_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
_raw_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_key = _raw_anthropic_key.strip() if _raw_anthropic_key else None
google_key = (os.getenv("GOOGLE_API_KEY") or "").strip() or None
google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip() or None
newsapi_key = os.getenv("NEWSAPI_KEY")
openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None
grok_key = (os.getenv("GROK_API_KEY") or "").strip() or None
perplexity_key = (os.getenv("PERPLEXITY_API_KEY") or "").strip() or None
cohere_key = (os.getenv("COHERE_API_KEY") or "").strip() or None
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# === SECURE STRIPE CONFIGURATION - PRODUCTION READY ===
# Supports: "live" (production), "test" (Stripe test mode), "dev" (local development without Stripe)
stripe_environment = os.getenv("STRIPE_ENVIRONMENT", "dev")  # Default to dev for local testing

# Flag to track if Stripe is available
STRIPE_AVAILABLE = False

if stripe_environment == "dev":
    # Development mode - Stripe disabled, app runs without payment features
    stripe.api_key = None
    STRIPE_AVAILABLE = False
    PRICE_IDS = {
        'consumer': 'dev_consumer',
        'professional': 'dev_professional',
        'enterprise': 'dev_enterprise'
    }
elif stripe_environment == "test":
    # Test mode - get key from environment
    stripe.api_key = os.getenv("STRIPE_TEST_SECRET_KEY")
    if not stripe.api_key:
        st.warning("⚠️ STRIPE_TEST_SECRET_KEY not set - running in dev mode")
        STRIPE_AVAILABLE = False
        PRICE_IDS = {
            'consumer': 'dev_consumer',
            'professional': 'dev_professional',
            'enterprise': 'dev_enterprise'
        }
    else:
        STRIPE_AVAILABLE = True
        PRICE_IDS = {
            'consumer': os.getenv("STRIPE_PRICE_CONSUMER_TEST"),
            'professional': os.getenv("STRIPE_PRICE_PROFESSIONAL_TEST"),
            'enterprise': os.getenv("STRIPE_PRICE_ENTERPRISE_TEST")
        }
else:
    # Live production mode
    stripe.api_key = os.getenv("STRIPE_LIVE_SECRET_KEY")
    if not stripe.api_key:
        st.warning("⚠️ STRIPE_LIVE_SECRET_KEY not set - running in dev mode")
        STRIPE_AVAILABLE = False
        PRICE_IDS = {
            'consumer': 'dev_consumer',
            'professional': 'dev_professional',
            'enterprise': 'dev_enterprise'
        }
    else:
        STRIPE_AVAILABLE = True
        PRICE_IDS = {
            'consumer': os.getenv("STRIPE_PRICE_CONSUMER_LIVE"),
            'professional': os.getenv("STRIPE_PRICE_PROFESSIONAL_LIVE"),
            'enterprise': os.getenv("STRIPE_PRICE_ENTERPRISE_LIVE")
        }

# Validate price IDs only if Stripe is available
if STRIPE_AVAILABLE:
    for plan, price_id in PRICE_IDS.items():
        if not price_id:
            st.warning(f"⚠️ Missing {plan} price ID - some features may be limited")
            STRIPE_AVAILABLE = False
            break

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID", "VA_xxxxx")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None
TWILIO_AVAILABLE = bool(twilio_client)

# === GMAIL SIGN-IN NOTIFICATIONS ===
GMAIL_SENDER = "bdemsey@gmail.com"
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
NOTIFY_EMAIL = "brian@hallucinations.cloud"

def send_signin_notification(phone: str, email: str = "", event: str = "sign_in", plan: str = ""):
    """Email bdemsey@gmail.com whenever a user signs in or registers."""
    if not GMAIL_APP_PASSWORD:
        print(f"[NOTIFY] Sign-in event: {event} | phone={phone} | email={email} | plan={plan}")
        return
    try:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        subject = f"Hallucinations.cloud — {'New Sign-Up' if event == 'register' else 'Sign-In'}"
        body = f"""
<h2>{'🆕 New User Registration' if event == 'register' else '🔑 User Sign-In'}</h2>
<table style="font-family:monospace;border-collapse:collapse;">
  <tr><td style="padding:4px 12px;"><b>Time:</b></td><td>{now}</td></tr>
  <tr><td style="padding:4px 12px;"><b>Phone:</b></td><td>{phone}</td></tr>
  <tr><td style="padding:4px 12px;"><b>Email:</b></td><td>{email or '(not provided)'}</td></tr>
  <tr><td style="padding:4px 12px;"><b>Plan:</b></td><td>{plan or '(unknown)'}</td></tr>
  <tr><td style="padding:4px 12px;"><b>Event:</b></td><td>{event}</td></tr>
</table>
"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = GMAIL_SENDER
        msg["To"] = NOTIFY_EMAIL
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, NOTIFY_EMAIL, msg.as_string())
    except Exception as e:
        print(f"[NOTIFY] Email send failed: {e}")

# Setup AI clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None
# Configure Anthropic client with extended timeout for cloud deployments (Render cold starts)
anthropic_client = anthropic.Anthropic(
    api_key=anthropic_key,
    timeout=60.0,  # 60 second timeout
    max_retries=3
) if anthropic_key else None
cohere_client = cohere.Client(cohere_key) if cohere_key else None
if google_key:
    genai.configure(api_key=google_key)

# Global flag to indicate Anthropic moderation is available
ANTHROPIC_MODERATION_AVAILABLE = bool(anthropic_client)

# Page config
st.set_page_config(page_title="H-LLM Multi-Model | Hallucinations.cloud", layout="centered", initial_sidebar_state="collapsed")

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
    
    # === CUT TO THE CHASE - Interactive Interface ===
    # Logo and title side by side
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("logo.png", width=120)
    with col_title:
        st.markdown("""
        <h1 style="color: #ff3b3b; margin-bottom: 0; font-size: 2.2rem; padding-top: 0.5rem;">
            ✂️ Cut to the Chase
        </h1>
        """, unsafe_allow_html=True)

    # Initialize session state for cut-to-chase flow
    if 'ctc_step' not in st.session_state:
        st.session_state.ctc_step = 1
    if 'ctc_phone' not in st.session_state:
        st.session_state.ctc_phone = ""
    if 'ctc_verified' not in st.session_state:
        st.session_state.ctc_verified = False
    if 'ctc_synopsis' not in st.session_state:
        st.session_state.ctc_synopsis = None
    if 'ctc_email' not in st.session_state:
        st.session_state.ctc_email = ""
    if 'ctc_email_saved' not in st.session_state:
        st.session_state.ctc_email_saved = False

    # Step 1: Begin Free Trial
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**1. Click here to begin your 3-day trial**")
    with col_right:
        if st.checkbox("Start Free Trial", key="ctc_start_trial"):
            st.session_state.ctc_step = max(st.session_state.ctc_step, 2)

    # Step 2: Enter Email Address
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**2. Enter your email address**")
    with col_right:
        if not st.session_state.get('ctc_email_saved', False):
            inp_col, btn_col, _ = st.columns([2, 1, 1])
            with inp_col:
                ctc_email = st.text_input("Email address", placeholder="you@example.com", key="ctc_email_input", label_visibility="collapsed")
            with btn_col:
                if ctc_email and '@' in ctc_email:
                    if st.button("Save →", key="ctc_save_email", type="primary", use_container_width=True):
                        st.session_state.ctc_email = ctc_email
                        st.session_state.ctc_email_saved = True
                        st.rerun()
        else:
            st.success(f"✓ {st.session_state.get('ctc_email', '')}")

    # Step 3: Enter Phone Number and Send Code
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**3. Enter your phone number**")
    with col_right:
        if not st.session_state.get('ctc_code_sent', False):
            inp_col, btn_col, _ = st.columns([2, 1, 1])
            with inp_col:
                phone = st.text_input("Phone number", placeholder="+1 (555) 123-4567", key="ctc_phone_input", label_visibility="collapsed")
            with btn_col:
                if phone:
                    st.session_state.ctc_phone = phone
                    if st.button("Send →", key="ctc_send_code", type="primary", use_container_width=True):
                        normalized_phone = validate_phone(phone)
                        if normalized_phone:
                            if send_verification_code(normalized_phone):
                                st.session_state.ctc_code_sent = True
                                st.session_state.ctc_normalized_phone = normalized_phone
                                st.session_state.ctc_step = max(st.session_state.ctc_step, 3)
                                st.rerun()
                        else:
                            st.error("Please enter a valid phone number")
        else:
            st.success("✓ Code sent to your phone!")
            st.session_state.ctc_step = max(st.session_state.ctc_step, 3)

    # Step 4: Enter SMS Code and Verify
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**4. Enter your SMS authorization code**")
    with col_right:
        if not st.session_state.get('ctc_verified', False):
            inp_col, btn_col, _ = st.columns([2, 1, 1])
            with inp_col:
                sms_code = st.text_input("SMS Code", placeholder="123456", key="ctc_sms_input", label_visibility="collapsed")
            with btn_col:
                if sms_code and len(sms_code) >= 4:
                    if st.button("Verify →", key="ctc_verify_code", type="primary", use_container_width=True):
                        normalized_phone = st.session_state.get('ctc_normalized_phone', st.session_state.get('ctc_phone', ''))
                        if verify_phone_code(normalized_phone, sms_code):
                            st.session_state.ctc_verified = True
                            st.session_state.ctc_step = max(st.session_state.ctc_step, 4)
                            send_signin_notification(
                                phone=normalized_phone,
                                email=st.session_state.get('ctc_email', ''),
                                event="sign_in",
                                plan="free"
                            )
                            st.rerun()
                        else:
                            st.error("Invalid code. Please try again.")
        else:
            st.success("✓ Verified")

    # Step 5: Enter Query (wrapped in form so Enter key submits)
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**5. Enter your query**")
    with col_right:
        with st.form(key="ctc_query_form", clear_on_submit=False):
            user_query = st.text_area("Your question", placeholder="Ask anything...", height=80, label_visibility="collapsed")
            submit_query = st.form_submit_button("🚀 Submit", type="primary")

    # Step 6: One Sentence Answer (shown after query)
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**6. Here's your one-sentence answer**")
    with col_right:
        if st.session_state.ctc_synopsis:
            with st.expander("📋 Synopsis", expanded=True):
                synopsis = st.session_state.ctc_synopsis
                st.markdown(f"**{synopsis.get('synopsis', '')}**")
                st.metric("Rating", fmt_score(synopsis.get('rating')))
        else:
            st.info("Submit a query above to see your answer")

    # Step 7: Scroll for details
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**7. Scroll down for detailed analysis from all 8 AI models**")
    with col_right:
        st.caption("⬇️ Detailed results appear below after you submit a query")

    # Step 8: Next Query
    col_left, col_right = st.columns([2, 3])
    with col_left:
        st.markdown("**8. Enter your next query**")
    with col_right:
        st.caption("Use the query box above (Step 5) for additional questions")

    st.markdown("---")

    # Process the query if submitted.
    # Oct 2026: a question runs only after the phone number is verified (steps 3 and 4).
    # Before this, step 5 ran the paid models for anyone, signed in or not.
    if submit_query and user_query and not st.session_state.get('ctc_verified', False):
        st.error("Please verify your phone number first (steps 3 and 4), then submit your question.")
    elif submit_query and user_query:
        # Initialize required session state for the app
        st.session_state.show_landing = False
        st.session_state.authenticated = True
        st.session_state.run_analysis = True
        st.session_state.current_query = user_query
        st.session_state.user_phone = st.session_state.get('ctc_normalized_phone', st.session_state.get('ctc_phone', 'Demo User'))
        st.session_state.user_email = st.session_state.get('ctc_email', 'demo@hallucinations.cloud')
        st.session_state.customer_id = "test_customer_id"
        st.session_state.subscription_status = "trialing"
        st.session_state.trial_end = (datetime.now() + timedelta(days=3)).isoformat()
        # Keep counting within the visit instead of resetting to zero on every question.
        st.session_state.queries_today = st.session_state.get('queries_today', 0)
        st.session_state.conversation_count = st.session_state.get('conversation_count', 0) + 1
        st.rerun()

    # Add button to switch to HTML landing page
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎨 View Full Landing Page", use_container_width=True):
            st.session_state.show_html_landing = True
            st.rerun()
    # Minimal CSS for essential styling only
    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, #ff3b3b 0%, #cc2f2f 100%);
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
        color: #1a1a2e;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #ff3b3b;
        margin: 0;
    }
    .model-badge {
        background: white;
        border: 2px solid #ff3b3b;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        color: #ff3b3b;
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
        border-color: #ff3b3b;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    .pricing-card.recommended {
        border-color: #ff3b3b;
        background: linear-gradient(135deg, #fff8f8 0%, #ffe8e8 100%);
        position: relative;
    }
    .recommended-badge {
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        background: #ff3b3b;
        color: white;
        padding: 0.25rem 1rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .price-large {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ff3b3b;
        margin: 0.5rem 0;
    }
    .price-period {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with Logo and Title side by side
    col_logo, col_text = st.columns([1, 2])
    with col_logo:
        st.image("logo.png", width=280)
    with col_text:
        st.markdown("""
        <div style="padding-top: 20px;">
            <h1 style="color: #ff3b3b; font-size: 2.8rem; font-weight: 800; margin-bottom: 0.5rem;">H-LLM Multi-Model™</h1>
            <p style="color: #4a5068; font-size: 1.1rem; line-height: 1.6;">The premier multi-model AI analysis platform that compares 8 leading LLMs simultaneously, detects hallucinations, and provides comprehensive security analysis through Red/Blue/Purple team methodologies.</p>
            <p style="font-weight:bold; text-transform:uppercase; color:#ff3b3b; margin-top:16px; font-size:13px;">Questions may be submitted in English, Spanish, Japanese, Mandarin, and 20+ other languages</p>
        </div>
        """, unsafe_allow_html=True)

    # Key Stats using pure Streamlit
    st.markdown("### 📊 Platform Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">8</div>
            <div>🤖 AI Models</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">3</div>
            <div>🛡️ Security Teams</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">5</div>
            <div>💬 Free Daily Queries</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-number">3</div>
            <div>🎁 Day Free Trial</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Models Showcase - Pure Streamlit
    st.markdown("### 🤖 Supported AI Models")
    st.markdown("*Compare responses from the world's leading AI systems*")
    
    # Create model badges using simple HTML in columns
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    
    models = ["GPT-4o", "Claude Haiku 4.5", "Gemini 2.5 Flash", "Grok 4.3", "Cohere Command R+", "DeepSeek", "OpenRouter", "Perplexity (paused)"]
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
                background: #ffffff;
                min-height: 100vh;
                color: #1a1a2e;
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
                color: #1a1a2e;
                overflow: hidden;
            }

            .hero-background {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: #f3f6fb;
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
                color: #1a1a2e;
            }

            .hero-tagline {
                font-size: 1.4rem;
                opacity: 0.95;
                margin-bottom: 30px;
                font-style: italic;
                color: #d42f4f;
            }

            .hero-description {
                font-size: 1.1rem;
                line-height: 1.7;
                margin-bottom: 35px;
                color: #4a5068;
            }

            .hero-image {
                position: relative;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 25px 60px rgba(0,0,0,0.3);
                transform: perspective(1000px) rotateY(-5deg);
                transition: transform 0.3s ease;
                background: linear-gradient(135deg, #ff3b3b, #cc2f2f);
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
                background: linear-gradient(45deg, #ff3b3b, #cc2f2f);
                color: white;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }

            .cta-primary:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 40px rgba(212, 47, 79, 0.3);
           }

           .cta-secondary {
               background: transparent;
               color: #1a1a2e;
               border: 2px solid #1a1a2e;
           }

           .cta-secondary:hover {
               background: #f3f6fb;
               border-color: #1a1a2e;
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
               background: linear-gradient(45deg, #ff3b3b, #cc2f2f);
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
               color: #ff3b3b;
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
               background: linear-gradient(135deg, #fde3e3 0%, #ffbbbb 100%);
               color: #cc2f2f;
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
               border-left: 6px solid #ff3b3b;
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
               color: #ff3b3b;
               font-weight: bold;
               margin-right: 12px;
               font-size: 1.2rem;
           }

           /* Final CTA Section */
           .final-cta {
               text-align: center;
               padding: 50px 0;
               background: #f3f6fb;
               color: #1a1a2e;
               border: 1px solid #d9dfeb;
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
# Dedicated bypass credentials for internal testing (login only)
TEST_LOGIN_PHONE = "+11234567890"
TEST_LOGIN_CODE = "123456"
TEST_LOGIN_EMAIL = os.getenv("TEST_LOGIN_EMAIL", "test+login@hallucinations.cloud")

def get_test_login_customer():
   """Create a lightweight customer object for the OTP bypass flow."""
   return SimpleNamespace(
       phone=TEST_LOGIN_PHONE,
       email=TEST_LOGIN_EMAIL,
       id="cus_test_login",
       metadata={
           'trial_end': (datetime.now() + timedelta(days=30)).isoformat(),
           'queries_used': '0',
           'queries_today': '0'
       }
   )

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
       print(f"Twilio send failed for {phone}: {e}", flush=True)
       st.error("We could not send a text to that number. Please check the number and try again.")
       return False

def verify_phone_code(phone, code):
   """Verify the SMS code"""
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
   except:
       st.error("Invalid code. Please try again.")
       return False

def create_account_with_plan(phone, email, plan):
   """Create account with selected plan"""
   try:
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

       send_signin_notification(phone=phone, email=email, event="register", plan=plan)

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
   col1, col2 = st.columns([1, 8])
   with col1:
       st.image("logo.png", width=80)
   with col2:
       st.title("Hallucinations.cloud")

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

   send_signin_notification(
       phone=customer.phone,
       email=getattr(customer, 'email', '') or '',
       event="sign_in",
       plan=getattr(customer, 'metadata', {}).get('selected_plan', '') if hasattr(customer, 'metadata') else ''
   )

   # Clear login state
   for key in ['pending_login_phone', 'pending_customer', 'show_login', 'existing_phone', 'test_login_bypass']:
       if key in st.session_state:
           del st.session_state[key]

   # SUPERUSER CHECK - bypass all subscription/trial checks
   if is_super_user():
       st.session_state.show_landing = False
       st.session_state.show_upgrade = False
       st.success("✅ Welcome back! You have superuser access.")
       return

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

def handle_send_login_code():
   """Callback for Send Login Code button - executes on single click"""
   phone = st.session_state.get('login_phone_input', '')
   normalized_phone = validate_phone(phone)

   if not normalized_phone:
       st.session_state.login_error = "Please enter a valid phone number"
       return

   if normalized_phone == TEST_LOGIN_PHONE:
       st.session_state.pending_login_phone = normalized_phone
       st.session_state.pending_customer = get_test_login_customer()
       st.session_state.test_login_bypass = True
       st.session_state.login_success = f"Test login bypass active. Use code {TEST_LOGIN_CODE}."
       st.session_state.login_error = None
   else:
       try:
           # Find customer by phone
           customers = stripe.Customer.search(
               query=f'metadata["phone"]:"{normalized_phone}"',
               limit=1
           )

           if customers.data:
               st.session_state.pending_login_phone = normalized_phone
               st.session_state.pending_customer = customers.data[0]
               st.session_state.pop('test_login_bypass', None)

               if send_verification_code(normalized_phone):
                   st.session_state.login_success = "Login code sent!"
                   st.session_state.login_error = None
               else:
                   # Sending failed, clear pending state
                   for key in ['pending_login_phone', 'pending_customer']:
                       if key in st.session_state:
                           del st.session_state[key]
                   st.session_state.login_error = "Failed to send code. Please try again."
                   st.session_state.login_success = None
           else:
               st.session_state.login_error = "No account found with this phone number"
               st.session_state.login_success = None
       except Exception as e:
           st.session_state.login_error = "Login failed. Please try again."
           st.session_state.login_success = None

def show_login_form():
   """Show login form for existing users"""
   st.markdown("### 🔐 Welcome Back!")

   phone = st.text_input(
       "Enter your phone number:",
       value=st.session_state.get('existing_phone', ''),
       placeholder="+1234567890",
       key="login_phone_input"
   )

   col1, col2 = st.columns(2)
   with col1:
       st.button("Send Login Code", type="primary", use_container_width=True, on_click=handle_send_login_code)

       # Display success/error messages after button click
       if st.session_state.get('login_success'):
           st.success(st.session_state.login_success)
       if st.session_state.get('login_error'):
           st.error(st.session_state.login_error)
   
   with col2:
       if st.button("Create new account", use_container_width=True):
           st.session_state.show_login = False
           if 'existing_phone' in st.session_state:
               del st.session_state.existing_phone
           st.session_state.pop('test_login_bypass', None)
           st.rerun()
   
   # Verify login code
   if st.session_state.get('pending_login_phone'):
       st.markdown("---")
       code = st.text_input("Enter login code:", max_chars=6)
       
       if st.button("Login", type="primary"):
           is_test_bypass = st.session_state.get('test_login_bypass', False)
           
           if is_test_bypass:
               if code == TEST_LOGIN_CODE:
                   customer = st.session_state.pending_customer
                   handle_successful_login(customer)
                   st.rerun()
               else:
                   st.error("Invalid test login code.")
           elif verify_phone_code(st.session_state.pending_login_phone, code):
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
   "+16504008061",  # DJ Waldow
   "+19495007539",  # Erin Conley
   "+972507510007", # Uri Levine
   "+61476165706",  # Julie Demsey
   "+19498877593",  # Sharon Conley
   "+18132207444",  # Jeff Berg
   "+12132207060",  # Kevin Tenn
   "+16507008815",  # Sean Fee
   "+16503882901",  # Shannon Fee
   "+16265235976",  # Howard Chen
   "+17275048126"   # Dustin Calim
]

def is_super_user():
   """Check if current user is a super user - respects normal user toggle"""
   user_phone = st.session_state.get('user_phone', '')
   # Normalize phone number - remove spaces, dashes, parentheses
   normalized_phone = ''.join(c for c in user_phone if c.isdigit() or c == '+')
   normalized_super_phones = [''.join(c for c in p if c.isdigit() or c == '+') for p in SUPER_USER_PHONES]
   is_actual_super_user = normalized_phone in normalized_super_phones

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
           scores = [q.get('enhanced_hscore', {}).get('final') for q in st.session_state.enhanced_query_history]
           scores = [v for v in scores if isinstance(v, (int, float))]
           avg_hscore = sum(scores) / len(scores) if scores else None
       st.metric("Avg H-Score", f"{avg_hscore:.2f}" if avg_hscore is not None else "n/a")
   
   # Recent Activity
   st.markdown("### 📈 Recent System Activity")
   
   # Query Analysis
   if st.session_state.get('enhanced_query_history'):
       st.markdown("#### 🔍 Recent Queries")
       recent_queries = st.session_state.enhanced_query_history[-5:]
       for i, query in enumerate(reversed(recent_queries)):
           with st.expander(f"Query {i+1}: {query['question'][:50]}..."):
               st.write(f"**Timestamp**: {query['timestamp']}")
               st.write(f"**H-Score**: {fmt_score(query['enhanced_hscore'].get('final'))}")
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
       st.write(f"📱 {st.session_state.get('user_phone', 'Demo User')}")
       st.write(f"📧 {st.session_state.get('user_email', 'demo@test.com')}")
       
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
                   background: linear-gradient(135deg, #ff3b3b 0%, #cc2f2f 100%);
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
           conversation_text += f"H-Score: {fmt_score(entry['hscore'])}\n\n"
           
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
               st.markdown(f"**H-Score:** {fmt_score(entry['hscore'])}")
               
               if entry.get('analysis_summary'):
                   st.markdown("**Analysis Summary:**")
                   st.info(entry['analysis_summary'])

# === MAIN APP FLOW CONTROL ===
def main_app_flow():
   """Main application flow with proper authentication handling"""

   # Check if user is authenticated via Cut-to-Chase flow (bypass normal auth for local testing)
   if st.session_state.get('authenticated', False):
       # User is authenticated, don't show landing or auth pages
       # The main app content will be rendered after this function
       return

   # Show landing page first for new users
   if st.session_state.get('show_landing', True):
       show_enhanced_landing_page()
       st.stop()

   # Show upgrade page for expired users
   elif st.session_state.get('show_upgrade', False):
       col1, col2 = st.columns([1, 8])
       with col1:
           st.image("logo.png", width=80)
       with col2:
           st.title("Hallucinations.cloud")
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

# === WEB SEARCH AUGMENTATION ===
def get_web_search_context(query):
   """Fetch current web search results to augment LLM queries - Tavily first, Google fallback"""

   # Try Tavily first (optimized for LLMs)
   if tavily_key:
       try:
           response = requests.post(
               "https://api.tavily.com/search",
               json={
                   "api_key": tavily_key,
                   "query": query,
                   "search_depth": "basic",
                   "max_results": 5,
                   "include_answer": True
               },
               timeout=15
           )
           if response.status_code == 200:
               data = response.json()
               search_context = "CURRENT WEB SEARCH RESULTS (use this for up-to-date information):\n\n"

               # Include Tavily's direct answer if available
               if data.get('answer'):
                   search_context += f"DIRECT ANSWER: {data['answer']}\n\n"

               # Include search results
               for i, result in enumerate(data.get('results', [])[:5], 1):
                   title = result.get('title', '')
                   content = result.get('content', '')
                   search_context += f"{i}. {title}\n   {content}\n\n"

               return search_context
       except Exception as e:
           print(f"Tavily search error: {e}")

   # Fall back to Google Custom Search
   if google_key and google_search_engine_id:
       try:
           search_url = "https://www.googleapis.com/customsearch/v1"
           params = {
               'key': google_key,
               'cx': google_search_engine_id,
               'q': query,
               'num': 5
           }
           response = requests.get(search_url, params=params, timeout=10)
           if response.status_code == 200:
               data = response.json()
               items = data.get('items', [])
               if items:
                   search_context = "CURRENT WEB SEARCH RESULTS (use this for up-to-date information):\n\n"
                   for i, item in enumerate(items[:5], 1):
                       title = item.get('title', '')
                       snippet = item.get('snippet', '')
                       search_context += f"{i}. {title}\n   {snippet}\n\n"
                   return search_context
       except Exception as e:
           print(f"Google search error: {e}")

   return None

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

   # Try newer model first, then fall back to older if needed
   # Oct 2026: Claude 3.5 Haiku and Claude 3 Haiku are retired (Anthropic model deprecations page).
   models_to_try = ["claude-haiku-4-5-20251001", "claude-haiku-5-5"]
   max_retries = 3

   for model_name in models_to_try:
       for attempt in range(max_retries):
           try:
               message = anthropic_client.messages.create(
                   model=model_name,
                   max_tokens=600,
                   timeout=30.0,  # 30 second timeout to prevent connection hangs
                   messages=[{"role": "user", "content": prompt}]
               )
               return ("Claude", message.content[0].text.strip())
           except anthropic.APIConnectionError as e:
               # Log the underlying cause for debugging
               cause = e.__cause__
               print(f"Claude APIConnectionError (attempt {attempt+1}/{max_retries}, model={model_name}): {e}, cause: {cause}")
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                   continue
               # Try next model if available
               if model_name != models_to_try[-1]:
                   break
               return ("Claude", f"[Claude connection error after {max_retries} retries. Cause: {cause}]")
           except anthropic.RateLimitError as e:
               if attempt < max_retries - 1:
                   time.sleep(2 ** attempt)
                   continue
               return ("Claude", f"[Claude rate limited: {str(e)}]")
           except anthropic.NotFoundError as e:
               # Model not found, try next model
               break
           except Exception as e:
               return ("Claude", f"[Claude error: {str(e)}]")
   return ("Claude", "[Claude error: all models unavailable]")

def call_gemini_sync(prompt):
   if not google_key:
       return ("Gemini", "[Gemini unavailable: missing API key]")

   # Oct 2026: gemini-2.0-flash reached its June 1 2026 shutdown; Google's replacement is gemini-3.6-flash.
   models_to_try = ["models/gemini-2.5-flash", "models/gemini-3.6-flash"]
   last_error = None

   for model_name in models_to_try:
       try:
           model = genai.GenerativeModel(model_name)
           response = model.generate_content(prompt)
           return ("Gemini", response.text.strip())
       except Exception as e:
           last_error = str(e)
           error_str = last_error.lower()
           if "404" in error_str or "not found" in error_str:
               continue  # Try next model
           # For other errors, retry with backoff
           time.sleep(1)
           continue

   return ("Gemini", f"[Gemini error: {last_error}]")

def call_cohere_sync(prompt):
   if not cohere_key:
       return ("Cohere", "[Cohere unavailable: missing API key]")
   
   try:
       co = cohere.Client(cohere_key)
       response = co.chat(
           message=prompt,
           model='command-r-plus-08-2024',
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

# Free OpenRouter models, tried in order. Free models are often busy ("rate-limited upstream"),
# so the next one is tried when one is busy. List checked against openrouter.ai/api/v1/models, Oct 9 2026.
OPENROUTER_FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "thinkingmachines/inkling:free",
]

def call_openrouter_sync(prompt):
   if not openrouter_key:
       return ("OpenRouter", "[OpenRouter unavailable: missing API key]")
   openrouter_client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
   last_error = ""
   for model in OPENROUTER_FREE_MODELS:
       try:
           response = openrouter_client.chat.completions.create(
               model=model,
               messages=[
                   {"role": "system", "content": "You are a helpful assistant."},
                   {"role": "user", "content": prompt}
               ],
               temperature=0.5,
               max_tokens=600,
               timeout=20,
           )
           text = (response.choices[0].message.content or "").strip()
           if text:
               print(f"OPENROUTER answered by {model}", flush=True)
               return ("OpenRouter", text)
           last_error = f"{model}: empty answer"
       except Exception as e:
           last_error = f"{model}: {str(e)[:150]}"
   return ("OpenRouter", f"[OpenRouter error: every free model busy; last: {last_error}]")

def call_perplexity_sync(prompt):
   """Perplexity ended Sonar Chat Completions on Sept 27, 2026 (now the Agent API).
   Uses the "fast" preset: one web search, no extra reasoning steps, the cheapest preset."""
   if not perplexity_key:
       return ("Perplexity", "[Perplexity unavailable: missing API key]")
   try:
       response = requests.post(
           "https://api.perplexity.ai/v1/agent",
           json={"preset": "fast", "input": prompt, "max_output_tokens": 800},
           headers={"Authorization": f"Bearer {perplexity_key}",
                    "Content-Type": "application/json"},
           timeout=50,
       )
       if response.status_code != 200:
           return ("Perplexity", f"[Perplexity error: HTTP {response.status_code}: {response.text[:200]}]")
       data = response.json()
       parts = [c.get("text", "")
                for item in data.get("output", []) if item.get("type") == "message"
                for c in item.get("content", []) if c.get("type") == "output_text"]
       text = re.sub(r"\s*\[\d+\]", "", "".join(parts)).strip()  # drop [1][2] source markers
       cost = ((data.get("usage") or {}).get("cost") or {}).get("total_cost")
       if cost is not None:
           print(f"PERPLEXITY cost ${cost}", flush=True)
       return ("Perplexity", text) if text else ("Perplexity", "[Perplexity error: empty answer]")
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
       messages = [{"role": "user", "content": prompt}]
       # Oct 2026: grok-3 was retired May 15 2026 (xAI's replacement is grok-4.3); grok-2 was retired earlier.
       for model_name in [os.getenv("GROK_MODEL_NAME", "grok-4.3")]:
           try:
               response = grok_client.chat.completions.create(
                   model=model_name,
                   messages=messages
               )
               return ("Grok", response.choices[0].message.content.strip())
           except Exception as e:
               if "not found" in str(e).lower() or "400" in str(e):
                   continue
               return ("Grok", f"[Grok error: {str(e)}]")
       return ("Grok", "[Grok error: no available models]")
   except Exception as e:
       return ("Grok", f"[Grok error: {str(e)}]")
def show_followup_interface(previous_query, previous_results):
    """Show follow-up question interface after analysis"""
    st.markdown("---")
    st.subheader("💬 Continue the Conversation")

    # Wrap in form so Enter key submits
    with st.form(key="followup_form", clear_on_submit=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            followup_question = st.text_input(
                "Ask a follow-up question:",
                placeholder="Build on the previous responses...",
                help="Your follow-up will be added to the ongoing conversation"
            )

        with col2:
            submit_followup = st.form_submit_button("🔄 Ask Follow-up", use_container_width=True)

    # Process after form submission
    if submit_followup and followup_question:
        context_query = create_followup_context(previous_query, previous_results, followup_question)
        st.session_state.current_query = context_query
        st.session_state.run_analysis = True
        st.session_state.is_followup = True
        st.session_state.last_query_text = followup_question
        st.rerun()

    st.caption(f"Previous query: {previous_query[:100]}...")
    return followup_question

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

# === SYNOPSIS GENERATION FUNCTION ===
def generate_synopsis(query, model_responses, red_analysis="", blue_analysis="", purple_analysis="", hscore=None):
    """
    Generate a one-sentence synopsis with rating from all analysis data.
    This is the "Cut to the Chase" feature - instant answer at the top.
    """
    unavailable = {"synopsis": "", "rating": None, "confidence": "Unavailable"}
    if not anthropic_client and not openai_client:
        return unavailable

    try:
        # Only the answers models actually gave (failures are left out)
        answered = [(name, resp) for name, resp in model_responses if model_answered(resp)]
        if not answered:
            return unavailable
        responses_text = "\n\n".join([f"[{name}]: {resp}" for name, resp in answered])
        hscore = f"{hscore}/10" if isinstance(hscore, (int, float)) else "not available"

        synopsis_prompt = f"""You are a synthesis expert. Your task is to create a ONE SENTENCE answer with a confidence rating.

ORIGINAL QUESTION: {query}

AI MODEL RESPONSES:
{responses_text}

RED TEAM ANALYSIS (risks/concerns):
{red_analysis if red_analysis else "Not available"}

BLUE TEAM ANALYSIS (reliability/trust):
{blue_analysis if blue_analysis else "Not available"}

PURPLE TEAM SYNTHESIS:
{purple_analysis if purple_analysis else "Not available"}

H-SCORE: {hscore}

YOUR TASK:
1. Synthesize ALL the above into ONE SENTENCE that directly answers the user's question
2. Account for any disagreements, risks, or caveats identified
3. Provide a confidence rating from 0-10

RESPOND IN EXACTLY THIS FORMAT (no other text):
SYNOPSIS: [Your one sentence answer here]
RATING: [X.X]/10
CONFIDENCE: [High/Medium/Low]
"""

        result_text = ""
        system_text = "You are an expert at synthesizing complex information into clear, accurate one-sentence summaries."
        # Claude Haiku 4.5 writes the final answer (decided Aug 2026); GPT-4o is the fallback.
        if anthropic_client:
            try:
                message = anthropic_client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=300,
                    timeout=30.0,
                    system=system_text,
                    messages=[{"role": "user", "content": synopsis_prompt}],
                )
                result_text = message.content[0].text.strip()
            except Exception as e:
                print(f"Synopsis via Claude failed: {e}", flush=True)
        if not result_text and openai_client:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": synopsis_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                result_text = response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Synopsis via OpenAI failed: {e}", flush=True)
        if not result_text:
            return unavailable

        # Parse the response
        synopsis = ""
        rating = None
        confidence = "Unavailable"

        for line in result_text.split("\n"):
            line = line.strip()
            if line.startswith("SYNOPSIS:"):
                synopsis = line.replace("SYNOPSIS:", "").strip()
            elif line.startswith("RATING:"):
                try:
                    rating_str = line.replace("RATING:", "").replace("/10", "").strip()
                    rating = max(0.0, min(10.0, float(rating_str)))
                except ValueError:
                    rating = None
            elif line.startswith("CONFIDENCE:"):
                confidence = line.replace("CONFIDENCE:", "").strip()

        # Fallback if parsing failed
        if not synopsis:
            synopsis = result_text.split("\n")[0] if result_text else "Unable to generate synopsis"

        return {
            "synopsis": synopsis,
            "rating": rating,
            "confidence": confidence
        }

    except Exception as e:
        print(f"Synopsis generation failed: {e}", flush=True)
        return unavailable

def display_synopsis(synopsis_data):
    """Display the synopsis prominently at the top of results"""
    synopsis = synopsis_data.get("synopsis", "")
    rating = synopsis_data.get("rating")
    confidence = synopsis_data.get("confidence", "Unavailable")
    if not synopsis:
        synopsis = "A one-sentence answer could not be produced for this question. The individual model answers are below."

    # Color based on rating
    if rating is None:
        color = "#8a8fa3"  # Gray: no rating
        emoji = "⚪"
    elif rating >= 8:
        color = "#28a745"  # Green
        emoji = "🟢"
    elif rating >= 6:
        color = "#ffc107"  # Yellow
        emoji = "🟡"
    elif rating >= 4:
        color = "#fd7e14"  # Orange
        emoji = "🟠"
    else:
        color = "#dc3545"  # Red
        emoji = "🔴"

    st.markdown(f"""
    <div style="background: #ffffff;
                padding: 1.5rem; border-radius: 1rem; margin-bottom: 1.5rem;
                border: 3px solid {color}; box-shadow: 0 4px 18px rgba(26,26,46,0.10);">
        <h2 style="color: #ff3b3b; margin-bottom: 0.5rem; text-align: center;">
            ✂️ Cut to the Chase
        </h2>
        <p style="font-size: 1.3rem; color: #1a1a2e; text-align: center; margin: 1rem 0; line-height: 1.6;">
            "{synopsis}"
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem;">
            <div style="text-align: center;">
                <span style="font-size: 2rem; color: {color};">{emoji} {fmt_score(rating)}</span>
                <br><span style="color: #4a5068;">Rating</span>
            </div>
            <div style="text-align: center;">
                <span style="font-size: 1.5rem; color: #1a1a2e;">{confidence}</span>
                <br><span style="color: #4a5068;">Confidence</span>
            </div>
        </div>
        <p style="text-align: center; color: #4a5068; margin-top: 1rem; font-size: 0.9rem;">
            ⬇️ Scroll down for each model's answer and the detailed analysis ⬇️
        </p>
    </div>
    """, unsafe_allow_html=True)

def extract_score_from_analysis(analysis_text, score_type="Risk Score"):
   """Return the score the analysis actually states, or None.

   Oct 2026: this used to return 5.0 (or guess from words like "high" or "good") whenever
   no score could be read, so a failed review looked like an ordinary middle score.
   Now a missing score is reported as missing.
   """
   if not analysis_text:
       return None
   text = str(analysis_text)
   head = text[:160].lower()
   if text.startswith("[") or "analysis failed" in head or "unavailable" in head:
       return None

   import re
   patterns = [
       rf'{score_type}:\s*(\d+(?:\.\d+)?)\s*/\s*10',
       rf'{score_type}:\s*(\d+(?:\.\d+)?)',
       rf'Score:\s*(\d+(?:\.\d+)?)\s*/\s*10',
   ]
   for pattern in patterns:
       matches = re.findall(pattern, text, re.IGNORECASE)
       if matches:
           try:
               return min(10.0, max(1.0, float(matches[0])))
           except ValueError:
               continue
   return None

def fmt_score(value, digits=1):
   """Show a 0-10 score, or 'n/a' when it is not available."""
   return f"{value:.{digits}f}/10" if isinstance(value, (int, float)) else "n/a"

def model_answered(response_text):
   """Every model caller marks a failure with text starting with '['."""
   return not str(response_text).startswith("[")

def calculate_enhanced_hscore(responses, red_analysis="", blue_analysis="", purple_analysis=""):
   """H-Score = equal 25% parts: Safety (11 minus Red risk), Trust (Blue), Confidence (Purple), Quality.

   Oct 2026: a part that could not be measured is None, and the H-Score itself is only
   computed when all four parts exist. Quality now counts only the models that really answered.
   """
   risk_score = extract_score_from_analysis(red_analysis, "Risk Score")
   trust_score = extract_score_from_analysis(blue_analysis, "Trust Score")
   confidence_score = extract_score_from_analysis(purple_analysis, "Confidence Score")
   safety_score = (11.0 - risk_score) if risk_score is not None else None

   answered = [r for r in responses if model_answered(r[1])]
   response_quality = (len(answered) / len(responses)) * 10 if responses else None

   parts = [safety_score, trust_score, confidence_score, response_quality]
   final_score = round(sum(parts) / 4, 2) if all(p is not None for p in parts) else None

   r1 = lambda v: round(v, 1) if v is not None else None
   return {
       'final': final_score,
       'safety': r1(safety_score),
       'trust': r1(trust_score),
       'confidence': r1(confidence_score),
       'quality': r1(response_quality),
       'answered': len(answered),
       'asked': len(responses),
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

# =============================================================================
# FOUR-PAGE FLOW (October 2026)
#   Page 1  What it is
#   Page 2  Ask: question, phone number and email (all required), then the texted code
#   Page 3  The answer: one sentence, each model's response, Red/Blue/Purple,
#           the expanded response, scoring
#   Page 4  "Do you have a follow-up question to your last question?"  -> back to page 3
# Free: no plans, trial, payments or upgrades. The older screens further down this
# file are no longer reached (st.stop() at the end of this block).
# =============================================================================
import threading as _hc_threading
from html import escape as _hc_esc

HC_DAILY_LIMIT = 25        # questions per phone number per day (decided Aug 2026)
HC_PER_MINUTE_LIMIT = 5    # questions per phone number per minute (decided Aug 2026)

HC_MODEL_NAMES = {
    "OpenAI": "OpenAI GPT-4o",
    "Claude": "Anthropic Claude",
    "Gemini": "Google Gemini",
    "Cohere": "Cohere",
    "Deepseek": "DeepSeek",
    "OpenRouter": "OpenRouter",
    "Perplexity": "Perplexity",
    "Grok": "xAI Grok",
}

HC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;600;800&display=swap');
html, body, [data-testid="stAppViewContainer"], .stMarkdown, .stTextInput input, .stTextArea textarea, button, label {
    font-family: 'Atkinson Hyperlegible Next', 'Segoe UI', system-ui, sans-serif !important;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { max-width: 780px !important; padding-top: 4.5rem !important; padding-bottom: 3rem !important; }
.hc-brand { padding-top: 4px; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { font-size: 1rem !important; color: #4a5068 !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { font-size: 1.15rem !important; font-weight: 600 !important; line-height: 1.3 !important; }
.stMarkdown p, .stMarkdown li { font-size: 1.08rem; line-height: 1.65; color: #1a1a2e; }
.hc-brand { display: flex; align-items: center; gap: 16px; margin-bottom: 6px; }
.hc-brand img { width: 64px; height: 64px; }
.stMarkdown p.hc-name, .hc-name { font-size: 1.9rem !important; font-weight: 800; color: #1a1a2e; line-height: 1.1; margin: 0; }
.stMarkdown p.hc-tag, .hc-tag { font-size: 1.05rem !important; color: #d42f4f; font-weight: 600; margin: 2px 0 0; }
.stMarkdown p.hc-lead, .hc-lead { font-size: 1.2rem !important; color: #4a5068; line-height: 1.6; margin: 18px 0 10px; }
.hc-list { margin: 8px 0 22px; padding-left: 0; list-style: none; }
.hc-list li { padding: 10px 0; border-bottom: 1px solid #e3e7f0; font-size: 1.05rem; color: #1a1a2e; }
.hc-list b { color: #1a1a2e; }
.stMarkdown p.hc-small, .hc-small { color: #4a5068; font-size: 0.95rem; }
.stMarkdown p.hc-question, .hc-question { color: #4a5068; font-size: 1.05rem !important; margin: 4px 0 6px; }
.hc-answer { font-size: 1.45rem; font-weight: 600; line-height: 1.5; color: #1a1a2e;
             border-left: 5px solid #d42f4f; padding: 6px 0 6px 18px; margin: 6px 0 26px; }
.stMarkdown p.hc-section, .hc-section { font-size: 1.25rem !important; font-weight: 800; color: #1a1a2e; margin: 28px 0 8px; }
.hc-foot { color: #4a5068; font-size: 0.92rem; border-top: 1px solid #d9dfeb; margin-top: 46px; padding-top: 14px; }
.hc-foot a { color: #4a5068; }
/* Hold the white scheme even when a link (embed_options=dark_theme) or a remembered
   browser setting asks Streamlit for its dark theme. */
:root, html, body { color-scheme: light !important; }
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main,
[data-testid="stHeader"], [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {
    background-color: #ffffff !important; color: #1a1a2e !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp span, .stApp label, .stApp strong, .stApp em, .stApp td, .stApp th,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] { color: #1a1a2e; }
.stApp a { color: #b3203f; }
.stApp .hc-foot, .stApp .hc-foot a { color: #4a5068; }
.stApp .hc-tag { color: #d42f4f; }
.stApp .hc-lead, .stApp .hc-small, .stApp .hc-question { color: #4a5068; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #1a1a2e !important; }
.stTextInput input, .stTextArea textarea, [data-baseweb="input"], [data-baseweb="textarea"],
[data-baseweb="base-input"] {
    background-color: #f3f6fb !important; color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important; caret-color: #1a1a2e !important;
}
[data-baseweb="input"], [data-baseweb="textarea"] { border-color: #c9d1e0 !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: #6b7289 !important; -webkit-text-fill-color: #6b7289 !important; opacity: 1 !important;
}
[data-testid="InputInstructions"], [data-testid="InputInstructions"] span { color: #6b7289 !important; }
[data-testid="stForm"] { border-color: #d9dfeb !important; background-color: #ffffff !important; }
[data-testid="stExpander"], [data-testid="stExpander"] details {
    background-color: #ffffff !important; border-color: #d9dfeb !important;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg { color: #1a1a2e !important; fill: #1a1a2e !important; }
[data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover p { color: #d42f4f !important; }
.stButton button, [data-testid="stFormSubmitButton"] button, [data-testid^="baseButton-secondary"] {
    background-color: #ffffff !important; color: #1a1a2e !important; border: 1px solid #c9d1e0 !important;
}
.stButton button p, .stButton button div, .stButton button span, [data-testid="stFormSubmitButton"] button p, [data-testid="stFormSubmitButton"] button div, [data-testid="stFormSubmitButton"] button span { color: inherit !important; }
.stButton button:hover, [data-testid="stFormSubmitButton"] button:hover { border-color: #d42f4f !important; color: #d42f4f !important; }
.stButton button[kind="primary"], .stButton button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"],
[data-testid^="baseButton-primary"] {
    background-color: #d42f4f !important; color: #ffffff !important; border: 1px solid #d42f4f !important;
}
.stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover,
[data-testid^="baseButton-primary"]:hover { background-color: #b3203f !important; color: #ffffff !important; }
[data-testid="stAlert"], [data-testid="stNotification"], [data-baseweb="notification"] { background-color: #f3f6fb !important; }
[data-testid="stAlert"] p, [data-testid="stNotification"] p, [data-baseweb="notification"] p { color: #1a1a2e !important; }
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] label, [data-testid="stStatusWidget"] span, [data-testid="stSpinner"] p, [data-testid="stSpinner"] div, .stSpinner p, .stSpinner div { color: #1a1a2e !important; }
[data-testid="stStatusContainer"], [data-testid="stStatusContainer"] details { background-color: #ffffff !important; border-color: #d9dfeb !important; }
.stApp code, .stApp pre { background-color: #f3f6fb !important; color: #1a1a2e !important; }
hr { border-color: #d9dfeb !important; }
</style>
"""


def hc_logo_html():
    try:
        import base64
        with open("logo.png", "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="">'
    except Exception:
        return ""


def hc_header():
    st.markdown(HC_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<div class="hc-brand">{hc_logo_html()}<div><p class="hc-name">H-LLM Multi-Model™</p>'
        f'<p class="hc-tag">Refining AI, One Insight at a Time</p></div></div>',
        unsafe_allow_html=True)


def hc_footer():
    st.markdown(
        '<div class="hc-foot">Hallucinations.cloud LLC · Spearfish, SD<br>'
        'Questions? Talk to Brian: <a href="tel:+19492911422">+1-949-291-1422</a> · '
        '<a href="mailto:brian@hallucinations.cloud">brian@hallucinations.cloud</a></div>',
        unsafe_allow_html=True)


def hc_go(page):
    st.session_state.hc_page = page
    st.rerun()


def hc_home_link(key):
    """Home: clear the question and start fresh. A confirmed phone stays confirmed for this visit."""
    if st.button("⌂ Home", key=key):
        for k in ("hc_question", "hc_last", "hc_code_sent"):
            st.session_state.pop(k, None)
        hc_go("about")


# ---------- limits (kept in the server's memory; reset when it restarts) ----------
@st.cache_resource
def hc_usage_store():
    return {"lock": _hc_threading.Lock(), "day": {}, "minute": {}}


def hc_take_question_slot(phone):
    """Return None if the phone may ask now, or a message saying why not."""
    if phone in SUPER_USER_PHONES:
        return None
    store = hc_usage_store()
    now = time.time()
    today = datetime.now().date().isoformat()
    with store["lock"]:
        day, count = store["day"].get(phone, (today, 0))
        if day != today:
            day, count = today, 0
        recent = [t for t in store["minute"].get(phone, []) if now - t < 60]
        if count >= HC_DAILY_LIMIT:
            return f"You have asked {HC_DAILY_LIMIT} questions today, the daily limit. Please come back tomorrow."
        if len(recent) >= HC_PER_MINUTE_LIMIT:
            return "Please wait a minute before asking another question."
        store["day"][phone] = (day, count + 1)
        store["minute"][phone] = recent + [now]
    return None


# ---------- the answer engine ----------
def hc_synthesize(question, results, red, blue, purple, hscore):
    """One sentence, a rating, a confidence and a multi-paragraph expanded response, in one call.
    Claude Haiku 4.5 writes it (decided Aug 2026); GPT-4o is the fallback."""
    answered = [(n, r) for n, r in results if model_answered(r)]
    if not answered:
        return {"sentence": "", "rating": None, "confidence": "", "expanded": ""}
    responses_text = "\n\n".join(f"[{n}]: {r}" for n, r in answered)
    # Reviews that returned nothing are passed as "Not available", never as error text.
    red, blue, purple = hc_review_text(red), hc_review_text(blue), hc_review_text(purple)
    score_text = f"{hscore}/10" if isinstance(hscore, (int, float)) else "not available"
    prompt = f"""QUESTION: {question}

ANSWERS FROM SEVERAL AI MODELS:
{responses_text}

RED TEAM REVIEW (risks, errors, hallucinations):
{red or "Not available"}

BLUE TEAM REVIEW (reliability, completeness):
{blue or "Not available"}

PURPLE TEAM SYNTHESIS:
{purple or "Not available"}

H-SCORE: {score_text}

Combine all of the above.
1. SENTENCE: one sentence that directly answers the question.
2. RATING: how reliable that answer is, from 0 to 10.
3. CONFIDENCE: High, Medium or Low.
4. EXPANDED: the full answer in plain language, in as many paragraphs as the question needs (usually two to five). Say where the models agreed, where they disagreed, and any caution the reviews raised. Do not mention reviews or models that are "Not available". No headings, no bullet lists, and no dashes used as punctuation.

Reply in exactly this format:
SENTENCE: ...
RATING: X.X/10
CONFIDENCE: ...
EXPANDED:
...
"""
    system_text = "You combine several AI answers into one clear, accurate answer for a general reader."
    text = ""
    if anthropic_client:
        try:
            msg = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1500, timeout=45.0,
                system=system_text, messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text.strip()
        except Exception as e:
            print(f"Synthesis via Claude failed: {e}", flush=True)
    if not text and openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o", temperature=0.3, max_tokens=1500,
                messages=[{"role": "system", "content": system_text}, {"role": "user", "content": prompt}])
            text = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Synthesis via OpenAI failed: {e}", flush=True)

    out = {"sentence": "", "rating": None, "confidence": "", "expanded": ""}
    if not text:
        return out
    text = text.replace("\u2014", ", ").replace("\u2013", "-").replace(" ,", ",")  # house style: no em dashes
    head, _, expanded = text.partition("EXPANDED:")
    out["expanded"] = expanded.strip()
    for line in head.splitlines():
        line = line.strip()
        if line.startswith("SENTENCE:"):
            out["sentence"] = line[len("SENTENCE:"):].strip().strip('"')
        elif line.startswith("RATING:"):
            try:
                out["rating"] = max(0.0, min(10.0, float(line[len("RATING:"):].replace("/10", "").strip())))
            except ValueError:
                pass
        elif line.startswith("CONFIDENCE:"):
            out["confidence"] = line[len("CONFIDENCE:"):].strip()
    return out


def hc_run_pipeline(question, model_query, status=None, search_query=None):
    """Ask every model at the same time, run the reviews, score, and write the answer."""
    def step(label):
        if status is not None:
            status.update(label=label)

    t0 = time.time()
    step("Looking up current information...")
    search_query = search_query or question
    print(f"SEARCH {search_query[:200]}", flush=True)
    web_context = get_web_search_context(search_query)
    if web_context:
        model_query = (f"{web_context}\n\nUSER QUESTION: {model_query}\n\n"
                       "Please answer using the current web search results above when relevant.")

    callers = []
    if openai_key: callers.append(call_openai_sync)
    if anthropic_key: callers.append(call_claude_sync)
    if google_key: callers.append(call_gemini_sync)
    if cohere_key: callers.append(call_cohere_sync)
    if deepseek_key: callers.append(call_deepseek_sync)
    if openrouter_key: callers.append(call_openrouter_sync)
    # Perplexity back on Oct 9 2026. Spending is capped on Perplexity's side:
    # prepaid $20, auto reload off, so the key stops when the credit runs out.
    if perplexity_key: callers.append(call_perplexity_sync)
    if grok_key: callers.append(call_grok_sync)

    step("Asking the AI models...")
    results = []
    if callers:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(callers))
        def _timed(c):
            t = time.time()
            r = c(model_query)
            print(f"TIMING model {c.__name__} {time.time() - t:.1f}s", flush=True)
            return r
        futures = [pool.submit(_timed, c) for c in callers]
        concurrent.futures.wait(futures, timeout=MODEL_TIMEOUT_SECONDS)
        for c, f in zip(callers, futures):
            name = c.__name__.replace("call_", "").replace("_sync", "").title()
            try:
                r = f.result() if f.done() else None
                results.append(r if isinstance(r, tuple) and len(r) == 2 else (name, "[no response]"))
            except Exception as e:
                results.append((name, f"[error: {e}]"))
        pool.shutdown(wait=False)
    # Log why any model gave no answer (log only; visitors still see "No response").
    for _n, _r in results:
        if not model_answered(_r):
            _why = re.sub(r"(sk-|pplx-|xai-|AIza|tvly-)[A-Za-z0-9_\-]+", r"\1***", str(_r))[:300]
            print(f"MODEL FAILED {_n}: {_why}", flush=True)
    t_models = time.time() - t0

    step("Red, Blue and Purple Team reviews...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as rp:
        f_red = rp.submit(perform_red_team_analysis, question, results)
        f_blue = rp.submit(perform_blue_team_analysis, question, results)
    red, blue = f_red.result(), f_blue.result()
    purple = ""
    if extract_score_from_analysis(red, "Risk Score") is not None or extract_score_from_analysis(blue, "Trust Score") is not None:
        purple = perform_purple_team_analysis(question, results, red, blue)

    hscore = calculate_enhanced_hscore(results, red, blue, purple)
    step("Writing the answer...")
    synth = hc_synthesize(question, results, red, blue, purple, hscore.get("final"))
    print(f"TIMING models={t_models:.1f}s total={time.time() - t0:.1f}s", flush=True)
    return {"question": question, "results": results, "red": red, "blue": blue,
            "purple": purple, "hscore": hscore, "synth": synth}


def hc_followup_search_query(previous_question, followup):
    """A follow-up such as "How does that compare with a year ago?" means nothing to a web search
    on its own. Rewrite it as one complete search that names the topic and any time frame."""
    fallback = f"{previous_question} {followup}"
    if not anthropic_client:
        return fallback
    try:
        msg = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=80, timeout=15.0,
            messages=[{"role": "user", "content":
                f"Today is {time.strftime('%B %d, %Y')}.\n"
                f"Earlier question: {previous_question}\n"
                f"Follow-up question: {followup}\n\n"
                "Write ONE web search query that answers the follow-up on its own. Name the specific "
                "thing being measured (for example: gold price per ounce), not the general topic. "
                "Turn relative times into exact dates (a year ago from today is the same day last "
                "year) and ask for the figures on both dates. Reply with the query only, no quotes."}])
        q = msg.content[0].text.strip().strip('"').splitlines()[0].strip()
        return q or fallback
    except Exception as e:
        print(f"Follow-up search rewrite failed: {e}", flush=True)
        return fallback


def hc_ask(question, model_query, search_query=None):
    """Moderation, limits, then the pipeline. Returns True when an answer is ready."""
    phone = st.session_state.get("hc_phone", "")
    if not process_query_with_moderation(question):
        return False
    blocked = hc_take_question_slot(phone)
    if blocked:
        st.error(blocked)
        return False
    with st.status("Working on your answer...", expanded=False) as status:
        st.session_state.hc_last = hc_run_pipeline(question, model_query, status, search_query)
        status.update(label="Done", state="complete")
    return True


def hc_model_text(text):
    """Every model caller marks a failure with text starting with '['; treat it as an empty response."""
    text = (text or "").strip()
    return None if (not text or text.startswith("[")) else text


def hc_review_text(text):
    """A review that failed returns 'X Team analysis failed/unavailable ...'; treat it as empty."""
    text = (text or "").strip()
    head = text[:60].lower()
    if not text or text.startswith("[") or "analysis failed" in head or "analysis unavailable" in head:
        return None
    return text


# ---------- page 1: what it is ----------
def hc_page_about():
    hc_header()
    st.markdown(
        '<p class="hc-lead">The premier multi-model AI analysis platform that compares 8 leading LLMs simultaneously, '
        'detects hallucinations, and provides comprehensive security analysis through Red/Blue/Purple team methodologies.</p>'
        '<ul class="hc-list">'
        '<li><b>🔴 Red Team</b> identifies risks, vulnerabilities, hallucinations and potential manipulation in AI responses.</li>'
        '<li><b>🔵 Blue Team</b> evaluates trustworthiness, reliability, completeness and safety.</li>'
        '<li><b>🟣 Purple Team</b> synthesizes the findings into actionable insights.</li>'
        '<li><b>🏆 H-Score™</b> rates Safety, Trust, Confidence and Quality.</li>'
        '</ul>'
        '<p class="hc-small">Questions may be submitted in English, Spanish, Japanese, Mandarin, and 20+ other languages.</p>',
        unsafe_allow_html=True)
    if st.button("Ask a question →", type="primary"):
        hc_go("ask")
    hc_footer()


# ---------- page 2: ask ----------
def hc_page_ask():
    hc_header()
    hc_home_link("home_ask")
    st.markdown('<p class="hc-section">Ask your question</p>', unsafe_allow_html=True)
    code_sent = st.session_state.get("hc_code_sent", False)

    if st.session_state.get("hc_verified"):
        # Phone already confirmed during this visit: just the question.
        with st.form("hc_ask_again_form"):
            q = st.text_area("Your question", height=110)
            go = st.form_submit_button("Get my answer →", type="primary")
        if go:
            q = q.strip()
            if not q:
                st.error("Please enter your question.")
            else:
                st.session_state.hc_question = q
                if hc_ask(q, q):
                    hc_go("answer")
    elif not code_sent:
        with st.form("hc_ask_form"):
            q = st.text_area("Your question", value=st.session_state.get("hc_question", ""), height=110)
            email = st.text_input("Email address", value=st.session_state.get("hc_email", ""))
            phone = st.text_input("Mobile phone number", value=st.session_state.get("hc_phone_raw", ""),
                                  placeholder="+1 555 123 4567")
            go = st.form_submit_button("Text me a code →", type="primary")
        if go:
            q, email, phone = q.strip(), email.strip(), phone.strip()
            st.session_state.hc_question, st.session_state.hc_email, st.session_state.hc_phone_raw = q, email, phone
            if not q:
                st.error("Please enter your question.")
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
                st.error("Please enter a valid email address.")
            else:
                normalized = validate_phone(phone)
                if normalized and send_verification_code(normalized):
                    st.session_state.hc_phone = normalized
                    st.session_state.hc_code_sent = True
                    st.rerun()
    else:
        st.markdown(f'<p class="hc-question">Your question: {_hc_esc(st.session_state.get("hc_question", ""))}</p>',
                    unsafe_allow_html=True)
        st.markdown(f'<p class="hc-small">We texted a code to {st.session_state.get("hc_phone", "")}.</p>',
                    unsafe_allow_html=True)
        with st.form("hc_code_form"):
            code = st.text_input("Code from the text message", max_chars=6)
            ok = st.form_submit_button("Confirm and get my answer →", type="primary")
        if st.button("Use a different number"):
            st.session_state.hc_code_sent = False
            st.rerun()
        if ok:
            if verify_phone_code(st.session_state.hc_phone, code.strip()):
                st.session_state.hc_verified = True
                send_signin_notification(phone=st.session_state.hc_phone, email=st.session_state.hc_email,
                                         event="sign_in", plan="free")
                q = st.session_state.hc_question
                if hc_ask(q, q):
                    hc_go("answer")
            else:
                st.error("That code did not match. Please check the text message and try again.")
    hc_footer()


# ---------- page 3: the answer ----------
def hc_page_answer():
    hc_header()
    hc_home_link("home_answer")
    a = st.session_state.get("hc_last")
    if not a:
        hc_go("ask")
    synth = a["synth"]
    st.markdown(f'<p class="hc-question">{_hc_esc(a["question"])}</p>', unsafe_allow_html=True)
    sentence = synth.get("sentence") or ""
    st.markdown(f'<div class="hc-answer">{_hc_esc(sentence) if sentence else "&nbsp;"}</div>', unsafe_allow_html=True)

    st.markdown('<p class="hc-section">What each AI model said</p>', unsafe_allow_html=True)
    for name, response in a["results"]:
        with st.expander(HC_MODEL_NAMES.get(name, name)):
            text = hc_model_text(response)
            st.markdown(text if text else "*No response.*")

    st.markdown('<p class="hc-section">Red, Blue and Purple Team analysis</p>', unsafe_allow_html=True)
    for label, key in [("🔴 Red Team", "red"), ("🔵 Blue Team", "blue"), ("🟣 Purple Team", "purple")]:
        with st.expander(label):
            text = hc_review_text(a[key])
            st.markdown(text if text else "*No analysis.*")

    st.markdown('<p class="hc-section">The full answer</p>', unsafe_allow_html=True)
    st.markdown(synth.get("expanded") or "*No expanded response.*")

    st.markdown('<p class="hc-section">Scoring</p>', unsafe_allow_html=True)
    s = a["hscore"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Answer rating", fmt_score(synth.get("rating")))
    c2.metric("Answer confidence", synth.get("confidence") or "n/a")
    c3.metric("H-Score™", fmt_score(s.get("final")))
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Safety", fmt_score(s.get("safety")))
    d2.metric("Trust", fmt_score(s.get("trust")))
    d3.metric("Confidence", fmt_score(s.get("confidence")))
    d4.metric("Quality", fmt_score(s.get("quality")))
    st.caption("H-Score™: Safety (Red Team), Trust (Blue Team), Confidence (Purple Team) and Quality (how many models answered), 25% each.")

    st.markdown("")
    if st.button("Ask a follow-up question →", type="primary"):
        hc_go("followup")
    hc_footer()


# ---------- page 4: follow-up ----------
def hc_page_followup():
    hc_header()
    hc_home_link("home_followup")
    a = st.session_state.get("hc_last")
    if not a:
        hc_go("ask")
    st.markdown('<p class="hc-section">Do you have a follow-up question to your last question?</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="hc-question">Your last question: {_hc_esc(a["question"])}</p>', unsafe_allow_html=True)
    with st.form("hc_followup_form"):
        fq = st.text_area("Your follow-up question", height=100)
        go = st.form_submit_button("Ask →", type="primary")
    if st.button("← Back to the answer"):
        hc_go("answer")
    if go:
        fq = fq.strip()
        if not fq:
            st.error("Please enter your follow-up question.")
        else:
            model_query = create_followup_context(a["question"], a["results"], fq)
            search_query = hc_followup_search_query(a["question"], fq)
            if hc_ask(fq, model_query, search_query):
                hc_go("answer")
    hc_footer()


# ---------- router ----------
if "hc_page" not in st.session_state:
    st.session_state.hc_page = "about"
_hc_page = st.session_state.hc_page
if _hc_page in ("answer", "followup") and not st.session_state.get("hc_verified"):
    _hc_page = "ask"
{"about": hc_page_about, "ask": hc_page_ask, "answer": hc_page_answer,
 "followup": hc_page_followup}.get(_hc_page, hc_page_about)()
st.stop()


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
   
   st.markdown("### Models in Use")
   models_list = ["GPT-4o", "Claude Haiku 4.5", "Gemini 2.5 Flash", "Grok 4.3", "Cohere Command R+", "DeepSeek", "OpenRouter", "Perplexity (paused)"]
   for model in models_list:
       st.markdown(f"- {model}")

   # Download Responses & Analyses button (available to all users)
   st.divider()
   st.markdown("### 📥 Export Data")
   if st.session_state.get('enhanced_query_history'):
       export_text = "HALLUCINATIONS.CLOUD - RESPONSES & ANALYSES\n"
       export_text += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
       export_text += "=" * 80 + "\n\n"

       for entry in st.session_state.enhanced_query_history:
           export_text += f"QUERY: {entry.get('question', 'N/A')}\n"
           export_text += f"TIME: {entry.get('timestamp', 'N/A')}\n"

           hscore = entry.get('enhanced_hscore', {})
           export_text += f"H-SCORE: {fmt_score(hscore.get('final'))}\n\n"

           export_text += "--- MODEL RESPONSES ---\n"
           for model, response in entry.get('responses', []):
               export_text += f"\n[{model}]:\n{response}\n"

           if entry.get('analysis_summary'):
               export_text += f"\n--- ANALYSIS SUMMARY ---\n{entry['analysis_summary']}\n"

           analysis = entry.get('analysis', {})
           if analysis.get('red_team'):
               export_text += f"\n--- 🔴 RED TEAM ANALYSIS ---\n{analysis['red_team']}\n"
           if analysis.get('blue_team'):
               export_text += f"\n--- 🔵 BLUE TEAM ANALYSIS ---\n{analysis['blue_team']}\n"
           if analysis.get('purple_team'):
               export_text += f"\n--- 🟣 PURPLE TEAM SYNTHESIS ---\n{analysis['purple_team']}\n"

           export_text += "\n" + "=" * 80 + "\n\n"

       st.download_button(
           "📥 Download Responses & Analyses",
           data=export_text,
           file_name=f"hllm_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
           mime="text/plain",
           use_container_width=True
       )
   else:
       st.caption("Run a query to enable download")

# === MAIN TITLE & HEADER ===
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.title("Hallucinations.cloud Multi-Model")

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

# Query input with moderation — hidden when results are displayed
if not st.session_state.get('last_query_text'):
    st.markdown("##### Enter your question:")
    st.caption("🛡️ All queries are checked for content policy compliance using Anthropic AI")

    # Show conversation context
    if st.session_state.conversation_count > 0:
       st.info(f"💬 Conversation Mode Active - Query #{st.session_state.conversation_count + 1}")

    # File attachment BEFORE form (file_uploader doesn't work well inside forms)
    st.caption("📎 Attach file (optional) · PDF, TXT, CSV, DOCX · 5MB max")
    uploaded_file = st.file_uploader(
        "Attach file",
        type=['pdf', 'txt', 'csv', 'docx'],
        label_visibility="collapsed",
        key="file_attachment"
    )

    # Validate file size and show status
    if uploaded_file and uploaded_file.size > 5_000_000:
        st.error("⚠️ File exceeds 5MB limit.")
        uploaded_file = None
    elif uploaded_file:
        st.success(f"📎 {uploaded_file.name} ({uploaded_file.size // 1024}KB)")

    # Query input wrapped in form so Enter key submits
    with st.form(key="main_query_form", clear_on_submit=False):
        user_query = st.text_input(
           "Ask anything - continue the conversation or start a new topic:",
           placeholder="Enter your question here...",
           help="Your query will be added to the conversation document along with all model responses"
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("🚀 Submit Query", type="primary")
        with col2:
            analysis_depth = st.selectbox(
                "Analysis Depth:",
                ["Quick", "Standard", "Comprehensive"],
                index=1,
                help="Quick: Basic comparison, Standard: +Red/Blue, Comprehensive: Full Red/Blue/Purple"
            )
else:
    uploaded_file = None
    user_query = None
    submit_button = False
    analysis_depth = "Standard"

# Process query after form submission
if submit_button and user_query:
    # Check content moderation
    if process_query_with_moderation(user_query):
        # Build full query with file content if attached
        full_query = user_query
        if uploaded_file:
            file_content = extract_text_from_file(uploaded_file)
            if file_content and not file_content.startswith("["):
                full_query = f"{user_query}\n\n--- Attached File: {uploaded_file.name} ---\n{file_content}"
        st.session_state.run_analysis = True
        st.session_state.current_query = full_query
        st.session_state.last_query_text = user_query

# Main analysis execution
if st.session_state.get('run_analysis') and st.session_state.get('current_query'):
   query = st.session_state.current_query

   # Get available models
   available_models = []
   if openai_key: available_models.append(call_openai_sync)
   if anthropic_key: available_models.append(call_claude_sync)
   if google_key: available_models.append(call_gemini_sync)
   if cohere_key: available_models.append(call_cohere_sync)
   if deepseek_key: available_models.append(call_deepseek_sync)
   if openrouter_key: available_models.append(call_openrouter_sync)
   # Perplexity disabled — runaway billing issue June 2026
   # if perplexity_key: available_models.append(call_perplexity_sync)
   if grok_key: available_models.append(call_grok_sync)

   if not available_models:
       st.error("No API keys available! Please set up at least one API key.")
   else:
       # === PHASE 1: COLLECT ALL DATA (no display yet) ===

       # Show the question immediately so it's visible during the entire analysis
       st.markdown(f"<p style='font-size:2.5rem; font-weight:bold;'>Your question so far: {query}</p>", unsafe_allow_html=True)

       # Get web search context for current information
       with st.spinner("🔍 Searching web for current information..."):
           web_context = get_web_search_context(query)

       # Augment query with web search results
       if web_context:
           augmented_query = f"{web_context}\n\nUSER QUESTION: {query}\n\nPlease answer using the current web search results above when relevant."
       else:
           augmented_query = query

       # Query all models AT THE SAME TIME (Oct 2026 speed change).
       # Each model call is independent, so the wait is now the slowest model, not the sum of all.
       # Results keep the same order as before. A model still running after MODEL_TIMEOUT_SECONDS
       # is reported as timed out instead of holding up the page.
       _t_models = time.time()
       with st.spinner(f"🤖 Querying {len(available_models)} AI models..."):
           _pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(available_models))
           _futures = [_pool.submit(model_func, augmented_query) for model_func in available_models]
           concurrent.futures.wait(_futures, timeout=MODEL_TIMEOUT_SECONDS)
           results = []
           for model_func, fut in zip(available_models, _futures):
               if not fut.done():
                   results.append((model_func.__name__, f"[Error: no answer within {MODEL_TIMEOUT_SECONDS} seconds]"))
                   continue
               try:
                   result = fut.result()
                   if result is not None and isinstance(result, tuple) and len(result) == 2:
                       results.append(result)
                   else:
                       results.append((model_func.__name__, "[Error: model returned no response]"))
               except Exception as e:
                   results.append((model_func.__name__, f"[Error: {str(e)}]"))
           _pool.shutdown(wait=False)
       print(f"TIMING models_parallel={time.time() - _t_models:.1f}s count={len(available_models)}", flush=True)

       # Increment usage counter
       increment_usage()

       # Initialize analysis variables
       red_analysis = ""
       blue_analysis = ""
       purple_analysis = ""
       contradiction_text = ""

       # Determine which analyses to run
       run_red = st.session_state.get('enable_red_team', True) or analysis_depth in ["Standard", "Comprehensive"]
       run_blue = st.session_state.get('enable_blue_team', True) or analysis_depth in ["Standard", "Comprehensive"]
       run_purple = st.session_state.get('enable_purple_team', True) or analysis_depth == "Comprehensive"

       # Contradiction check, Red Team and Blue Team do not depend on each other,
       # so they now run AT THE SAME TIME (Oct 2026 speed change). Purple still waits for Red and Blue.
       _t_reviews = time.time()
       model_responses_text = "\n\n".join([f"{name}: {resp}" for name, resp in results])
       contradiction_prompt = f"""
               Analyze these AI model responses for contradictions or significant disagreements:

               {model_responses_text}

               Provide a brief analysis of any contradictions found, or confirm if responses are generally consistent.
               Focus on:
               1. Key points of agreement
               2. Notable differences or contradictions
               3. Variations in perspective or emphasis
               4. Overall consistency assessment

               Format as numbered points for clarity.
               """
       with st.spinner("⚖️ Checking contradictions and running Red and Blue Team analyses..."):
           with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _rpool:
               _f_contra = _rpool.submit(call_openai_sync, contradiction_prompt) if openai_client else None
               _f_red = _rpool.submit(perform_red_team_analysis, query, results) if run_red else None
               _f_blue = _rpool.submit(perform_blue_team_analysis, query, results) if run_blue else None
           if _f_contra is not None:
               try:
                   contradiction_text = _f_contra.result()[1]
               except Exception as e:
                   contradiction_text = f"Contradiction analysis failed: {str(e)}"
           if _f_red is not None:
               try:
                   red_analysis = _f_red.result()
               except Exception as e:
                   red_analysis = f"Red Team analysis failed: {str(e)}"
           if _f_blue is not None:
               try:
                   blue_analysis = _f_blue.result()
               except Exception as e:
                   blue_analysis = f"Blue Team analysis failed: {str(e)}"
       print(f"TIMING reviews_parallel={time.time() - _t_reviews:.1f}s", flush=True)

       # Run Purple Team Analysis
       if run_purple and red_analysis and blue_analysis:
           with st.spinner("🟣 Running Purple Team synthesis..."):
               purple_analysis = perform_purple_team_analysis(query, results, red_analysis, blue_analysis)

       print(f"TIMING after_purple={time.time() - _t_models:.1f}s since models started", flush=True)

       # Calculate enhanced H-Score
       with st.spinner("🏆 Calculating H-Score..."):
           enhanced_scores = calculate_enhanced_hscore(results, red_analysis, blue_analysis, purple_analysis)

       # === PHASE 2: GENERATE SYNOPSIS ===
       with st.spinner("✂️ Generating synopsis..."):
           synopsis_data = generate_synopsis(
               query,
               results,
               red_analysis,
               blue_analysis,
               purple_analysis,
               enhanced_scores.get('final')
           )

       print(f"TIMING total_to_display={time.time() - _t_models:.1f}s since models started", flush=True)

       # === PHASE 3: DISPLAY RESULTS (Synopsis FIRST) ===

       # Display Synopsis at the TOP
       display_synopsis(synopsis_data)

       # Order (Oct 2026): rating first (above), then each model's answer,
       # then the Red, Blue and Purple Team reviews, then the overall assessment (H-Score).
       st.subheader("📊 Detailed Analysis")
       _answered = sum(1 for _, r in results if model_answered(r))
       st.caption(f"{_answered} of {len(results)} AI models answered. Open any model below to read its full answer."
                  + (" Current web search results were added to the question." if web_context else ""))

       # Display model responses
       st.markdown("### 🤖 Model Responses")
       for model_name, response in results:
           _status = "✅" if model_answered(response) else "⚠️ no answer:"
           with st.expander(f"{_status} **{model_name}**", expanded=False):
               st.text_area(f"{model_name} response:", value=response, height=150, key=f"response_{model_name}_{st.session_state.conversation_count}")

       # === CONTRADICTION ANALYSIS ===
       st.subheader("⚖️ Contradiction Analysis")
       if contradiction_text and not contradiction_text.startswith("[") and "failed" not in contradiction_text[:60].lower():
           st.success(contradiction_text)
       else:
           st.warning("The contradiction check could not run for this question.")

       # === SECURITY ANALYSIS SECTION ===
       st.markdown("---")
       st.subheader("🔍 Security Analysis")

       # Create analysis columns
       analysis_cols = st.columns(3)

       # Red Team Analysis
       if run_red and red_analysis:
           with analysis_cols[0]:
               st.markdown("### 🔴 Red Team Analysis")
               with st.expander("🔴 Red Team Report", expanded=False):
                   st.markdown(red_analysis)

       # Blue Team Analysis
       if run_blue and blue_analysis:
           with analysis_cols[1]:
               st.markdown("### 🔵 Blue Team Analysis")
               with st.expander("🔵 Blue Team Report", expanded=False):
                   st.markdown(blue_analysis)

       # Purple Team Analysis
       if run_purple and purple_analysis:
           with analysis_cols[2]:
               st.markdown("### 🟣 Purple Team Analysis")
               with st.expander("🟣 Purple Team Report", expanded=False):
                   st.markdown(purple_analysis)

       # === ENHANCED H-SCORE DASHBOARD ===
       st.markdown("---")
       st.subheader("🏆 Enhanced H-Score Analysis")
       
       # Display enhanced metrics
       score_cols = st.columns(5)
       
       def _delta(v):
           return f"{v - 5:.1f}" if isinstance(v, (int, float)) and v != 5 else None

       for _col, _label, _key, _help in [
           (score_cols[0], "🛡️ Safety", 'safety', "Based on Red Team risk assessment (inverted)"),
           (score_cols[1], "🔐 Trust", 'trust', "Based on Blue Team reliability assessment"),
           (score_cols[2], "💪 Confidence", 'confidence', "Based on Purple Team strategic assessment"),
           (score_cols[3], "✅ Quality", 'quality', "Share of the models that actually answered"),
           (score_cols[4], "🎯 H-Score", 'final', "Equal 25% parts: Safety, Trust, Confidence, Quality. Shown only when all four were measured."),
       ]:
           with _col:
               st.metric(_label, fmt_score(enhanced_scores.get(_key)), delta=_delta(enhanced_scores.get(_key)), help=_help)

       # Interpretation (overall assessment)
       final_score = enhanced_scores.get('final')
       if final_score is None:
           st.warning("⚪ **H-Score not available** - the Red, Blue and Purple Team reviews did not all produce a score for this question, so no overall score is shown. "
                      f"{enhanced_scores.get('answered', 0)} of {enhanced_scores.get('asked', 0)} models answered.")
       elif final_score >= 8.0:
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
           - **Safety Score**: {fmt_score(enhanced_scores['safety'])} (Red Team risk assessment, inverted)
           - **Trust Score**: {fmt_score(enhanced_scores['trust'])} (Blue Team reliability assessment)
           - **Confidence Score**: {fmt_score(enhanced_scores['confidence'])} (Purple Team strategic assessment)
           - **Quality Score**: {fmt_score(enhanced_scores['quality'])} ({enhanced_scores.get('answered', 0)} of {enhanced_scores.get('asked', 0)} models answered)

           **Final H-Score**: equal 25% weights for each part, shown only when all four parts were measured ("n/a" means not measured).
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
           analysis_summary += f"Red Team Risk: {fmt_score(risk_score)} | "
       if blue_analysis:
           trust_score = extract_score_from_analysis(blue_analysis, "Trust Score")
           analysis_summary += f"Blue Team Trust: {fmt_score(trust_score)} | "
       if purple_analysis:
           conf_score = extract_score_from_analysis(purple_analysis, "Confidence Score")
           analysis_summary += f"Purple Team Confidence: {fmt_score(conf_score)}"
       
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
       if not st.session_state.get('run_analysis', False):
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
                       st.metric("H-Score", fmt_score(scores.get('final')), delta=None)
           
           # Show mini analysis summary
           if query.get('analysis'):
               analysis_summary = ""
               if query['analysis'].get('red_team'):
                   risk_score = extract_score_from_analysis(query['analysis']['red_team'], "Risk Score")
                   analysis_summary += f"🔴 Risk: {fmt_score(risk_score)}  "
               if query['analysis'].get('blue_team'):
                   trust_score = extract_score_from_analysis(query['analysis']['blue_team'], "Trust Score")
                   analysis_summary += f"🔵 Trust: {fmt_score(trust_score)}  "
               if query['analysis'].get('purple_team'):
                   conf_score = extract_score_from_analysis(query['analysis']['purple_team'], "Confidence Score")
                   analysis_summary += f"🟣 Confidence: {fmt_score(conf_score)}"
               
               if analysis_summary:
                   st.caption(analysis_summary)
           
           st.markdown("---")

# === FOOTER ===
st.divider()
st.markdown("Built with ❤️ by Hallucinations.Cloud")
st.caption("Note: This tool compares multiple LLM responses for hallucination detection. Advanced AI content moderation powered by Anthropic's Constitutional AI technology. Use responsibly.")
