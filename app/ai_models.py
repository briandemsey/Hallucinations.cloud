"""
AI Models module - 8 AI model integration
OpenAI GPT-4o, Claude Sonnet, Gemini, Cohere, DeepSeek, OpenRouter, Perplexity, Grok
"""
import os
from openai import OpenAI
import anthropic
import google.generativeai as genai
import cohere
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# API Keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# Initialize AI clients
openai_client = None
anthropic_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

if ANTHROPIC_API_KEY:
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)


# ============= AI MODEL FUNCTIONS =============

def call_openai(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """OpenAI GPT-4o"""
    if not openai_client:
        return {"model": "OpenAI", "response": "[OpenAI unavailable: missing API key]"}

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to current information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=600
        )

        answer = response.choices[0].message.content.strip()

        return {"model": "OpenAI", "response": answer}
    except Exception as e:
        return {"model": "OpenAI", "response": f"[OpenAI error: {str(e)}]"}


def call_claude(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """Claude Sonnet 4.5"""
    if not anthropic_client:
        return {"model": "Claude", "response": "[Claude unavailable: missing API key]"}

    try:
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )

        answer = message.content[0].text.strip()

        return {"model": "Claude", "response": answer}
    except Exception as e:
        return {"model": "Claude", "response": f"[Claude error: {str(e)}]"}


def call_gemini(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """Google Gemini 2.5 Flash"""
    if not GOOGLE_API_KEY:
        return {"model": "Gemini", "response": "[Gemini unavailable: missing API key]"}

    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        answer = response.text.strip()

        return {"model": "Gemini", "response": answer}
    except Exception as e:
        return {"model": "Gemini", "response": f"[Gemini error: {str(e)}]"}


def call_cohere(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """Cohere Command R"""
    if not COHERE_API_KEY:
        return {"model": "Cohere", "response": "[Cohere unavailable: missing API key]"}

    try:
        co = cohere.Client(COHERE_API_KEY)
        response = co.chat(
            message=prompt,
            model='command-r-08-2024',
            max_tokens=600,
            temperature=0.5
        )

        answer = response.text.strip()

        return {"model": "Cohere", "response": answer}
    except Exception as e:
        return {"model": "Cohere", "response": f"[Cohere error: {str(e)}]"}


def call_deepseek(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """DeepSeek"""
    if not DEEPSEEK_API_KEY:
        return {"model": "DeepSeek", "response": "[DeepSeek unavailable: missing API key]"}

    try:
        deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
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

        answer = response.choices[0].message.content.strip()

        return {"model": "DeepSeek", "response": answer}
    except Exception as e:
        return {"model": "DeepSeek", "response": f"[DeepSeek error: {str(e)}]"}


def call_openrouter(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """OpenRouter"""
    if not OPENROUTER_API_KEY:
        return {"model": "OpenRouter", "response": "[OpenRouter unavailable: missing API key]"}

    try:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
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

        answer = response.choices[0].message.content.strip()

        return {"model": "OpenRouter", "response": answer}
    except Exception as e:
        return {"model": "OpenRouter", "response": f"[OpenRouter error: {str(e)}]"}


def call_perplexity(prompt: str) -> Dict[str, Any]:
    """Perplexity"""
    if not PERPLEXITY_API_KEY:
        return {"model": "Perplexity", "response": "[Perplexity unavailable: missing API key]"}

    try:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {PERPLEXITY_API_KEY}"
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
            return {"model": "Perplexity", "response": f"[Perplexity error: HTTP {response.status_code}]"}

        data = response.json()

        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            return {"model": "Perplexity", "response": content.strip()}
        else:
            return {"model": "Perplexity", "response": "[Perplexity error: Unexpected response]"}

    except Exception as e:
        return {"model": "Perplexity", "response": f"[Perplexity error: {str(e)}]"}


def call_grok(prompt: str, enable_rag: bool = True, show_metadata: bool = False) -> Dict[str, Any]:
    """xAI Grok"""
    if not GROK_API_KEY:
        return {"model": "Grok", "response": "[Grok unavailable: missing API key]"}

    try:
        grok_client = OpenAI(
            api_key=GROK_API_KEY,
            base_url="https://api.x.ai/v1"
        )
        preferred_model = os.getenv("GROK_MODEL_NAME", "grok-beta")
        messages = [{"role": "user", "content": prompt}]

        response = grok_client.chat.completions.create(
            model=preferred_model,
            messages=messages
        )

        answer = response.choices[0].message.content.strip()

        return {"model": "Grok", "response": answer}
    except Exception as e:
        return {"model": "Grok", "response": f"[Grok error: {str(e)}]"}


# ============= PARALLEL QUERY EXECUTION =============

async def query_all_models(
    query: str,
    enable_rag: bool = True,
    show_metadata: bool = False
) -> List[Dict[str, Any]]:
    """
    Query all 8 AI models in parallel using ThreadPoolExecutor

    Args:
        query: User's query string
        enable_rag: Enable RAG (currently not implemented)
        show_metadata: Show model metadata

    Returns:
        List of dicts: [{"model": str, "response": str}, ...]
    """
    # All model functions
    model_functions = [
        call_openai,
        call_claude,
        call_gemini,
        call_cohere,
        call_deepseek,
        call_openrouter,
        call_perplexity,
        call_grok
    ]

    results = []

    # Execute all models in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_model = {
            executor.submit(func, query, enable_rag, show_metadata): func
            for func in model_functions[:-1]  # All except Perplexity
        }
        # Perplexity has different signature
        future_to_model[executor.submit(call_perplexity, query)] = call_perplexity

        for future in as_completed(future_to_model):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Model execution error: {str(e)}")

    return results
