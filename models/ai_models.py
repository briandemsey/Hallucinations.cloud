# models/ai_models.py
"""
AI Models Manager for Hallucinations.cloud
Handles all AI model integrations and API calls
"""

import streamlit as st
import os
import requests
from typing import Dict, List, Any, Optional
from config.settings import get_api_clients, get_api_keys

class AIModelsManager:
    """Manages all AI model integrations"""

    def __init__(self):
        self.api_keys = get_api_keys()
        self.clients = get_api_clients()
        self.models = {
            "GPT-4o": "gpt-4o",
            "Claude 3 Haiku": "claude-3-haiku-20240307",
            "Gemini Pro": "gemini-1.5-pro-latest",
            "Cohere Command-R": "command-r",
            "Deepseek Chat": "deepseek-chat",
            "OpenRouter WizardLM": "microsoft/wizardlm-2-8x22b",
            "Grok": "grok-2-1212",
            "Perplexity Sonar": "sonar-medium-online"
        }

    def get_available_models(self) -> List[str]:
        """Return list of available models based on API keys"""
        available = []
        if self.api_keys.get("openai"): available.append("GPT-4o")
        if self.api_keys.get("anthropic"): available.append("Claude 3 Haiku")
        if self.api_keys.get("google"): available.append("Gemini Pro")
        if self.api_keys.get("cohere"): available.append("Cohere Command-R")
        if self.api_keys.get("deepseek"): available.append("Deepseek Chat")
        if self.api_keys.get("openrouter"): available.append("OpenRouter WizardLM")
        if self.api_keys.get("grok"): available.append("Grok")
        if self.api_keys.get("perplexity"): available.append("Perplexity Sonar")
        return available

    def query_model(self, model_name: str, prompt: str) -> Dict[str, Any]:
        """Query a specific AI model"""
        try:
            if model_name == "GPT-4o":
                return self._call_openai(prompt)
            elif model_name == "Claude 3 Haiku":
                return self._call_anthropic(prompt)
            elif model_name == "Gemini Pro":
                return self._call_gemini(prompt)
            elif model_name == "Cohere Command-R":
                return self._call_cohere(prompt)
            elif model_name == "Deepseek Chat":
                return self._call_deepseek(prompt)
            elif model_name == "OpenRouter WizardLM":
                return self._call_openrouter(prompt)
            elif model_name == "Grok":
                return self._call_grok(prompt)
            elif model_name == "Perplexity Sonar":
                return self._call_perplexity(prompt)
            else:
                return {"error": f"Unknown model: {model_name}"}
        except Exception as e:
            return {"error": f"Error calling {model_name}: {str(e)}"}

    def query_all_models(self, prompt: str) -> Dict[str, Dict[str, Any]]:
        """Query all available models with the same prompt"""
        results = {}
        available_models = self.get_available_models()

        progress_bar = st.progress(0)
        total_models = len(available_models)

        for i, model_name in enumerate(available_models):
            st.write(f"Querying {model_name}...")
            results[model_name] = self.query_model(model_name, prompt)
            progress_bar.progress((i + 1) / total_models)

        progress_bar.empty()
        return results

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI GPT-4o"""
        if not self.clients.get("openai"):
            return {"error": "OpenAI client not initialized"}

        response = self.clients["openai"].chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return {
            "response": response.choices[0].message.content,
            "model": "GPT-4o",
            "tokens": response.usage.total_tokens if response.usage else 0
        }

    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude"""
        if not self.clients.get("anthropic"):
            return {"error": "Anthropic client not initialized"}

        response = self.clients["anthropic"].messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "response": response.content[0].text,
            "model": "Claude 3 Haiku",
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }

    def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        """Call Google Gemini"""
        if not self.clients.get("google"):
            return {"error": "Google client not initialized"}

        model = self.clients["google"].GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return {
            "response": response.text,
            "model": "Gemini Pro",
            "tokens": 0  # Gemini doesn't provide token count easily
        }

    def _call_cohere(self, prompt: str) -> Dict[str, Any]:
        """Call Cohere Command-R"""
        if not self.clients.get("cohere"):
            return {"error": "Cohere client not initialized"}

        response = self.clients["cohere"].chat(
            model="command-r",
            message=prompt,
            max_tokens=1000
        )
        return {
            "response": response.text,
            "model": "Cohere Command-R",
            "tokens": 0  # Cohere token counting varies
        }

    def _call_deepseek(self, prompt: str) -> Dict[str, Any]:
        """Call Deepseek via API"""
        if not self.api_keys.get("deepseek"):
            return {"error": "Deepseek API key not found"}

        headers = {
            "Authorization": f"Bearer {self.api_keys['deepseek']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }

        response = requests.post("https://api.deepseek.com/v1/chat/completions",
                               headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result["choices"][0]["message"]["content"],
                "model": "Deepseek Chat",
                "tokens": result.get("usage", {}).get("total_tokens", 0)
            }
        else:
            return {"error": f"Deepseek API error: {response.status_code}"}

    def _call_openrouter(self, prompt: str) -> Dict[str, Any]:
        """Call OpenRouter"""
        if not self.api_keys.get("openrouter"):
            return {"error": "OpenRouter API key not found"}

        headers = {
            "Authorization": f"Bearer {self.api_keys['openrouter']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "microsoft/wizardlm-2-8x22b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                               headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result["choices"][0]["message"]["content"],
                "model": "OpenRouter WizardLM",
                "tokens": result.get("usage", {}).get("total_tokens", 0)
            }
        else:
            return {"error": f"OpenRouter API error: {response.status_code}"}

    def _call_grok(self, prompt: str) -> Dict[str, Any]:
        """Call Grok via API"""
        if not self.api_keys.get("grok"):
            return {"error": "Grok API key not found"}

        # Placeholder for Grok API implementation
        return {"error": "Grok API integration pending"}

    def _call_perplexity(self, prompt: str) -> Dict[str, Any]:
        """Call Perplexity Sonar"""
        if not self.api_keys.get("perplexity"):
            return {"error": "Perplexity API key not found"}

        headers = {
            "Authorization": f"Bearer {self.api_keys['perplexity']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "sonar-medium-online",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }

        response = requests.post("https://api.perplexity.ai/chat/completions",
                               headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return {
                "response": result["choices"][0]["message"]["content"],
                "model": "Perplexity Sonar",
                "tokens": result.get("usage", {}).get("total_tokens", 0)
            }
        else:
            return {"error": f"Perplexity API error: {response.status_code}"}