# config/settings.py
"""
Configuration management for Hallucinations.cloud
Handles API keys, client initialization, and app settings
"""

import os
import streamlit as st
from openai import OpenAI
import anthropic
import google.generativeai as genai
import cohere

# Global configuration dictionary
CONFIG = {
    "app_name": "Hallucinations.cloud",
    "version": "1.0.0",
    "debug": False,
    "cache_ttl": 300,  # 5 minutes
    "max_file_size": 10 * 1024 * 1024,  # 10MB
}

def initialize_configuration():
    """Initialize application configuration"""
    # Store config in session state for access across modules
    if "app_config" not in st.session_state:
        st.session_state.app_config = CONFIG

def get_api_keys():
    """Get all API keys from environment variables"""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "google": os.getenv("GOOGLE_API_KEY"),
        "cohere": os.getenv("COHERE_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "grok": os.getenv("GROK_API_KEY"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
        "stripe": os.getenv("STRIPE_LIVE_SECRET_KEY") or os.getenv("STRIPE_TEST_SECRET_KEY"),
        "twilio_sid": os.getenv("TWILIO_ACCOUNT_SID"),
        "twilio_token": os.getenv("TWILIO_AUTH_TOKEN"),
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_key": os.getenv("SUPABASE_KEY")
    }

def get_api_clients():
    """Initialize and return all API clients"""
    keys = get_api_keys()
    clients = {}

    # OpenAI client
    if keys["openai"]:
        clients["openai"] = OpenAI(api_key=keys["openai"])

    # Anthropic client
    if keys["anthropic"]:
        clients["anthropic"] = anthropic.Anthropic(api_key=keys["anthropic"])

    # Google Gemini client
    if keys["google"]:
        genai.configure(api_key=keys["google"])
        clients["google"] = genai

    # Cohere client
    if keys["cohere"]:
        clients["cohere"] = cohere.Client(keys["cohere"])

    return clients

def get_model_list():
    """Return list of available AI models"""
    return [
        "GPT-4o",
        "Claude 3 Haiku",
        "Gemini Pro",
        "Cohere Command-R",
        "Deepseek Chat",
        "OpenRouter WizardLM",
        "Grok",
        "Perplexity Sonar"
    ]

def get_subscription_tiers():
    """Return subscription pricing tiers"""
    return {
        "free": {"queries_per_day": 5, "price": 0, "duration_days": 3},
        "consumer": {"queries_per_day": 25, "price": 9.99, "duration_days": 30},
        "professional": {"queries_per_day": -1, "price": 29.99, "duration_days": 30},  # -1 = unlimited
        "enterprise": {"queries_per_day": -1, "price": 99.99, "duration_days": 30, "api_access": True}
    }