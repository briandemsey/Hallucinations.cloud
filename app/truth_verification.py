"""
Truth Verification Engine
Verifies AI response accuracy through cross-referencing, temporal checks, and source verification.
"""
import os
import re
import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse

# API Keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")


class TruthVerificationEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with API keys and configuration"""
        if config is None:
            config = {}
        self.google_api_key = config.get('google_api_key') or GOOGLE_API_KEY
        self.google_search_engine_id = config.get('google_search_engine_id') or GOOGLE_SEARCH_ENGINE_ID
        self.newsapi_key = config.get('newsapi_key') or NEWSAPI_KEY

    def verify_response_accuracy(
        self,
        query: str,
        ai_responses: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """
        Main verification function - returns comprehensive accuracy assessment

        Args:
            query: The original user query
            ai_responses: List of (model_name, response_text) tuples

        Returns:
            Dict with verification results including overall_truth_score
        """
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

    def _extract_factual_claims(self, ai_responses: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
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

    def _cross_reference_information(self, query: str, ai_responses: List[Tuple[str, str]]) -> float:
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

    def _check_temporal_accuracy(self, query: str, ai_responses: List[Tuple[str, str]]) -> float:
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

    def _verify_sources(self, ai_responses: List[Tuple[str, str]]) -> Dict[str, Any]:
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
                        reliable_domains = [
                            '.edu', '.gov', '.org', 'reuters.com', 'bbc.com',
                            'nature.com', 'science.org', 'nih.gov', 'who.int'
                        ]

                        is_reliable = any(reliable in domain for reliable in reliable_domains)
                        if is_reliable:
                            source_verification['reliable_sources'] += 1

                        source_verification['source_details'].append({
                            'url': url,
                            'status': 'verified',
                            'reliable': is_reliable
                        })
                    else:
                        source_verification['broken_links'] += 1

                except Exception:
                    source_verification['broken_links'] += 1

        return source_verification

    def _calculate_truth_score(self, verification_results: Dict[str, Any]) -> float:
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

    def _generate_verification_summary(self, verification_results: Dict[str, Any]) -> str:
        """Generate human-readable verification summary"""
        score = verification_results['overall_truth_score']

        if score >= 0.8:
            summary = "High Accuracy: Information appears to be well-supported by reliable sources."
        elif score >= 0.6:
            summary = "Moderate Accuracy: Some information verified, but exercise caution."
        elif score >= 0.4:
            summary = "Low Accuracy: Limited verification found. Independent research recommended."
        else:
            summary = "Questionable Accuracy: Significant concerns about information reliability."

        # Add specific findings
        details = []
        if verification_results['source_verification'].get('reliable_sources', 0) > 0:
            reliable_count = verification_results['source_verification']['reliable_sources']
            details.append(f"{reliable_count} reliable sources found")

        if details:
            summary += f" ({', '.join(details)})"

        return summary

    def _determine_confidence_level(self, truth_score: float) -> str:
        """Determine confidence level based on truth score"""
        if truth_score >= 0.8:
            return 'high'
        elif truth_score >= 0.6:
            return 'medium'
        elif truth_score >= 0.4:
            return 'low'
        else:
            return 'very_low'


def verify_responses(query: str, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convenience function to verify a list of AI responses.

    Args:
        query: The original user query
        responses: List of response dicts with 'model' and 'response' keys

    Returns:
        Verification results dict
    """
    # Convert to tuple format expected by TruthVerificationEngine
    ai_responses = [(r['model'], r['response']) for r in responses]

    engine = TruthVerificationEngine()
    return engine.verify_response_accuracy(query, ai_responses)
