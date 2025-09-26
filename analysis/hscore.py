# analysis/hscore.py
"""
H-Score Analysis Engine for Hallucinations.cloud
Proprietary scoring algorithm for AI response reliability
"""

import streamlit as st
from typing import Dict, List, Any, Tuple
import re
import json
from collections import Counter

class HScoreEngine:
    """Proprietary H-Score calculation engine"""

    def __init__(self):
        self.weight_consistency = 0.3
        self.weight_accuracy = 0.3
        self.weight_completeness = 0.2
        self.weight_confidence = 0.2

    def calculate_h_score(self, responses: Dict[str, Dict[str, Any]]) -> float:
        """Calculate H-Score from multiple AI responses"""
        if not responses or len(responses) < 2:
            return 0.0

        # Extract response texts
        response_texts = {}
        for model, data in responses.items():
            if "response" in data and not data.get("error"):
                response_texts[model] = data["response"]

        if len(response_texts) < 2:
            return 0.0

        # Calculate individual components
        consistency_score = self._calculate_consistency(response_texts)
        accuracy_score = self._calculate_accuracy(response_texts)
        completeness_score = self._calculate_completeness(response_texts)
        confidence_score = self._calculate_confidence(response_texts)

        # Weighted H-Score calculation
        h_score = (
            consistency_score * self.weight_consistency +
            accuracy_score * self.weight_accuracy +
            completeness_score * self.weight_completeness +
            confidence_score * self.weight_confidence
        ) * 10  # Scale to 0-10

        return round(h_score, 1)

    def _calculate_consistency(self, responses: Dict[str, str]) -> float:
        """Calculate consistency between responses"""
        if len(responses) < 2:
            return 0.0

        # Extract key facts and claims from each response
        all_facts = []
        for model, response in responses.items():
            facts = self._extract_facts(response)
            all_facts.extend(facts)

        # Count common facts across responses
        fact_counts = Counter(all_facts)
        common_facts = sum(1 for count in fact_counts.values() if count > 1)
        total_unique_facts = len(fact_counts)

        if total_unique_facts == 0:
            return 0.0

        consistency = common_facts / total_unique_facts
        return min(consistency, 1.0)

    def _calculate_accuracy(self, responses: Dict[str, str]) -> float:
        """Calculate accuracy indicators"""
        # This is a simplified accuracy calculation
        # In production, this would integrate with fact-checking APIs

        accuracy_indicators = 0
        total_responses = len(responses)

        for response in responses.values():
            # Check for uncertainty markers (good for accuracy)
            uncertainty_patterns = [
                r"might be", r"could be", r"possibly", r"perhaps",
                r"it seems", r"appears to be", r"likely", r"probably"
            ]

            # Check for absolute claims (potential accuracy risk)
            absolute_patterns = [
                r"definitely", r"absolutely", r"certainly", r"always",
                r"never", r"impossible", r"guaranteed"
            ]

            uncertainty_count = sum(len(re.findall(pattern, response, re.IGNORECASE))
                                  for pattern in uncertainty_patterns)
            absolute_count = sum(len(re.findall(pattern, response, re.IGNORECASE))
                               for pattern in absolute_patterns)

            # Balanced responses with some uncertainty are more accurate
            if uncertainty_count > 0 and absolute_count <= uncertainty_count:
                accuracy_indicators += 1

        return accuracy_indicators / total_responses if total_responses > 0 else 0.0

    def _calculate_completeness(self, responses: Dict[str, str]) -> float:
        """Calculate response completeness"""
        if not responses:
            return 0.0

        response_lengths = [len(response.split()) for response in responses.values()]
        avg_length = sum(response_lengths) / len(response_lengths)

        # Responses between 50-300 words are considered complete
        if 50 <= avg_length <= 300:
            return 1.0
        elif avg_length < 50:
            return avg_length / 50
        else:
            # Penalize overly long responses
            return max(0.5, 1.0 - ((avg_length - 300) / 300))

    def _calculate_confidence(self, responses: Dict[str, str]) -> float:
        """Calculate confidence indicators"""
        confidence_score = 0
        total_responses = len(responses)

        for response in responses.values():
            # Look for confidence indicators
            high_confidence_patterns = [
                r"according to", r"research shows", r"studies indicate",
                r"data suggests", r"evidence shows", r"experts agree"
            ]

            low_confidence_patterns = [
                r"I think", r"in my opinion", r"I believe", r"I'm not sure",
                r"it's unclear", r"hard to say"
            ]

            high_conf_count = sum(len(re.findall(pattern, response, re.IGNORECASE))
                                for pattern in high_confidence_patterns)
            low_conf_count = sum(len(re.findall(pattern, response, re.IGNORECASE))
                               for pattern in low_confidence_patterns)

            # Balance between confidence and humility
            if high_conf_count > low_conf_count:
                confidence_score += 0.8
            elif low_conf_count > 0:
                confidence_score += 0.6
            else:
                confidence_score += 0.4

        return confidence_score / total_responses if total_responses > 0 else 0.0

    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts and claims from text"""
        # Simple fact extraction - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', text)
        facts = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                # Extract numbers and proper nouns as facts
                numbers = re.findall(r'\d+(?:\.\d+)?', sentence)
                proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', sentence)

                for num in numbers:
                    facts.append(f"number_{num}")
                for noun in proper_nouns:
                    facts.append(f"entity_{noun.lower()}")

                # Extract key phrases
                if any(word in sentence.lower() for word in ['is', 'was', 'are', 'were']):
                    facts.append(sentence[:50])  # First 50 chars as fact

        return facts

def calculate_h_score(responses: Dict[str, Dict[str, Any]]) -> float:
    """Convenience function to calculate H-Score"""
    engine = HScoreEngine()
    return engine.calculate_h_score(responses)

def analyze_contradictions(responses: Dict[str, Dict[str, Any]]) -> List[str]:
    """Analyze contradictions between responses"""
    contradictions = []

    response_texts = []
    model_names = []

    for model, data in responses.items():
        if "response" in data and not data.get("error"):
            response_texts.append(data["response"])
            model_names.append(model)

    if len(response_texts) < 2:
        return contradictions

    # Simple contradiction detection
    for i in range(len(response_texts)):
        for j in range(i + 1, len(response_texts)):
            text1 = response_texts[i].lower()
            text2 = response_texts[j].lower()

            # Look for contradictory patterns
            if ("yes" in text1 and "no" in text2) or ("no" in text1 and "yes" in text2):
                contradictions.append(f"{model_names[i]} and {model_names[j]} give contradictory yes/no answers")

            # Check for contradictory numbers
            nums1 = re.findall(r'\d+(?:\.\d+)?', text1)
            nums2 = re.findall(r'\d+(?:\.\d+)?', text2)

            if nums1 and nums2:
                for n1 in nums1:
                    for n2 in nums2:
                        if abs(float(n1) - float(n2)) / max(float(n1), float(n2)) > 0.5:
                            contradictions.append(f"{model_names[i]} and {model_names[j]} provide significantly different numbers: {n1} vs {n2}")

    return contradictions