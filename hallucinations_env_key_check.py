import streamlit as st
import os

st.title("🔐 Environment Key Status Checker")

required_keys = {
    "OPENAI_API_KEY": "OpenAI",
    "OPENROUTER_API_KEY": "OpenRouter",
    "ANTHROPIC_API_KEY": "Claude",
    "GOOGLE_API_KEY": "Gemini",
    "GROK_API_KEY": "Grok",
    "PERPLEXITY_API_KEY": "Perplexity"
}

for env_var, model_name in required_keys.items():
    key_present = os.getenv(env_var) is not None
    status_emoji = "✅" if key_present else "❌"
    st.markdown(
        f"<div style='color: {'green' if key_present else 'red'}; "
        f"font-size: 14px;'>🔐 Render sees `{env_var}` ({model_name}): <strong>{key_present}</strong> {status_emoji}</div>",
        unsafe_allow_html=True
    )
