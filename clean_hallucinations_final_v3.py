# hallucinations_app_final_clean.py
"""
Hallucinations.cloud Multi-Model Comparison App
FINAL CLEAN VERSION - No duplicates
"""

import streamlit as st
import os
import pandas as pd
import requests
from datetime import datetime
from openai import OpenAI
import anthropic
import google.generativeai as genai
import cohere
from dotenv import load_dotenv

# === BLOCK 1: Configuration & Setup ===
load_dotenv()

# Get API keys
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
grok_key = os.getenv("GROK_API_KEY")
perplexity_key = os.getenv("PERPLEXITY_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# Setup clients
openai_client = OpenAI(api_key=openai_key) if openai_key else None
anthropic_client = anthropic.Anthropic(api_key=anthropic_key) if anthropic_key else None
cohere_client = cohere.Client(cohere_key) if cohere_key else None
if google_key:
    genai.configure(api_key=google_key)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Page config
st.set_page_config(page_title="Hallucinations.cloud", layout="wide")

# === BLOCK 2: Sidebar ===
with st.sidebar:
    st.title("💡 Suggest an Additional LLM")
    st.text_input("Suggest a Model", placeholder="Model name...", key="suggest_model")
    st.text_input("Your Name", key="user_name")
    st.text_input("Your Email", key="user_email")
    
    if st.button("Send Suggestion"):
        st.success("✅ Suggestion submitted!")
    
    st.divider()
    
    st.markdown("### 🧠 Models in Use")
    models_list = ["GPT-4o", "Claude 3 Haiku", "Gemini 1.5 Pro", "Grok", "Cohere", "Deepseek", "OpenRouter", "Perplexity"]
    for model in models_list:
        st.markdown(f"- {model}")

# === RELEVANT QUERIES SECTION ===
with st.sidebar:
    st.markdown("### 🎯 Relevant Queries")
    
    relevant_queries = [
        "How much damage do AI hallucinations really cause?",
        "Give some examples of recent AI hallucinations in the news",
        "What are the most serious consequences of AI hallucinations in healthcare?",
        "How can users identify when an AI is hallucinating?",
        "What are the most famous ChatGPT hallucination incidents?",
        "Why do large language models hallucinate?",
        "How do AI hallucinations affect trust in AI systems?",
        "Show me examples of AI making up fake legal cases",
        "What techniques do companies use to reduce hallucinations?",
        "How dangerous are hallucinations in autonomous vehicles?",
        "What financial losses have companies suffered from AI hallucinations?",
        "How have AI hallucinations affected journalism?"
    ]
    
    selected_relevant_query = st.selectbox(
        "Choose a relevant query:",
        ["Select a query..."] + relevant_queries,
        key="relevant_query_selector"
    )
    
    if st.button("Use This Query", key="use_relevant_query") and selected_relevant_query != "Select a query...":
        st.session_state.auto_query = selected_relevant_query
        st.session_state.execute_query = True
        st.rerun()

# === ADVANCED ANALYSIS SECTION ===
with st.sidebar:
    st.markdown("### 🛡️ Advanced Analysis")
    
    # Always show the button for now - we can see query results exist
    if st.button("🎯 Run Red/Blue/Purple Team Analysis", key="adv_analysis", help="Advanced security team analysis"):
        st.session_state.show_advanced_analysis = True
    
    if st.session_state.get("show_advanced_analysis", False):
        if st.button("✖️ Close Advanced Analysis", key="close_adv_analysis"):
            st.session_state.show_advanced_analysis = False

# === BLOCK 3: Main Title & Header ===
st.title("🧠 Hallucinations.cloud Multi-Model")
st.info("This application is a beta prototype under active development. For suggestions or bug reports, contact support@hallucinations.cloud")

# === BLOCK 4: API Key Status Checker ===
st.subheader("🔐 Environment Key Status Checker")

key_col1, key_col2 = st.columns(2)

with key_col1:
    st.markdown(f"OpenAI (OPENAI_API_KEY): {'✅ True' if openai_key else '❌ False'}")
    st.markdown(f"Claude (ANTHROPIC_API_KEY): {'✅ True' if anthropic_key else '❌ False'}")
    st.markdown(f"Gemini (GOOGLE_API_KEY): {'✅ True' if google_key else '❌ False'}")
    st.markdown(f"Cohere (COHERE_API_KEY): {'✅ True' if cohere_key else '❌ False'}")

with key_col2:
    st.markdown(f"OpenRouter (OPENROUTER_API_KEY): {'✅ True' if openrouter_key else '❌ False'}")
    st.markdown(f"Grok (GROK_API_KEY): {'✅ True' if grok_key else '❌ False'}")
    st.markdown(f"Perplexity (PERPLEXITY_API_KEY): {'✅ True' if perplexity_key else '❌ False'}")
    st.markdown(f"Deepseek (DEEPSEEK_API_KEY): {'✅ True' if deepseek_key else '❌ False'}")

# === BLOCK 5: Model Caller Functions ===
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
            max_tokens=500
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
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return ("Claude", message.content[0].text.strip())
    except Exception as e:
        return ("Claude", f"[Claude error: {str(e)}]")

def call_gemini_sync(prompt):
    if not google_key:
        return ("Gemini", "[Gemini unavailable: missing API key]")
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return ("Gemini", response.text.strip())
    except Exception as e:
        return ("Gemini", f"[Gemini error: {str(e)}]")

def call_cohere_sync(prompt):
    if not cohere_key:
        return ("Cohere", "[Cohere unavailable: missing API key]")
    try:
        co = cohere.Client(cohere_key)
        response = co.chat(
            message=prompt,
            model='command-r',
            max_tokens=500,
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
            max_tokens=500
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
            max_tokens=500
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
        
        # Using the verified working model
        payload = {
            "model": "sonar",  # This model is confirmed to work!
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.5
        }
        
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 401:
            return ("Perplexity", "[Perplexity error: Invalid API key. Please check your PERPLEXITY_API_KEY]")
        elif response.status_code == 429:
            return ("Perplexity", "[Perplexity error: Rate limit exceeded]")
        elif response.status_code == 400:
            try:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', str(error_data))
                return ("Perplexity", f"[Perplexity error: {error_msg}]")
            except:
                return ("Perplexity", f"[Perplexity error: HTTP 400 - {response.text[:300]}]")
        elif response.status_code != 200:
            return ("Perplexity", f"[Perplexity error: HTTP {response.status_code} - {response.text[:200]}]")
        
        data = response.json()
        
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            return ("Perplexity", content.strip())
        else:
            return ("Perplexity", f"[Perplexity error: Unexpected response - {str(data)[:200]}]")
            
    except requests.exceptions.RequestException as e:
        return ("Perplexity", f"[Perplexity network error: {str(e)}]")
    except Exception as e:
        return ("Perplexity", f"[Perplexity error: {type(e).__name__} - {str(e)}]")

def call_grok_sync(prompt):
    if not grok_key:
        return ("Grok", "[Grok unavailable: missing API key]")
    try:
        grok_client = OpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )
        response = grok_client.chat.completions.create(
            model="grok-2-1212",
            messages=[
                {"role": "system", "content": "You are Grok: witty, direct, and accurate with a touch of humor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return ("Grok", response.choices[0].message.content.strip())
    except Exception as e:
        return ("Grok", f"[Grok error: {str(e)}]")

# === BLOCK 6: Main Query Interface ===
st.subheader("🔍 Compare LLMs")

initial_query = ""
if "auto_query" in st.session_state:
    initial_query = st.session_state.auto_query
    del st.session_state.auto_query

user_query = st.text_input("Enter your factual question:", value=initial_query, placeholder="Ask something to compare across models...")

auto_execute = st.session_state.get("execute_query", False)
if auto_execute:
    st.session_state.execute_query = False

if (st.button("Submit") and user_query) or (auto_execute and user_query):
    st.subheader("📊 Model Responses")
    
    available_models = []
    if openai_key:
        available_models.append(call_openai_sync)
    if anthropic_key:
        available_models.append(call_claude_sync)
    if google_key:
        available_models.append(call_gemini_sync)
    if cohere_key:
        available_models.append(call_cohere_sync)
    if deepseek_key:
        available_models.append(call_deepseek_sync)
    if openrouter_key:
        available_models.append(call_openrouter_sync)
    if perplexity_key:
        available_models.append(call_perplexity_sync)
    if grok_key:
        available_models.append(call_grok_sync)
    
    if not available_models:
        st.error("No API keys available! Please set up at least one API key.")
    else:
        with st.spinner("Querying models..."):
            results = []
            for model_func in available_models:
                try:
                    result = model_func(user_query)
                    results.append(result)
                except Exception as e:
                    results.append((model_func.__name__, f"[Error: {str(e)}]"))
        
        for model_name, response in results:
            st.markdown(f"**{model_name}**")
            st.text_area(f"{model_name} response:", value=response, height=150, key=f"response_{model_name}")
        
        query_record = {
            "timestamp": datetime.now().isoformat(),
            "question": user_query,
            "results": results
        }
        st.session_state.query_history.append(query_record)
        
        if openai_client:
            st.subheader("⚖️ Contradiction Analysis")
            with st.spinner("Analyzing for contradictions..."):
                model_responses = "\n\n".join([f"{name}: {resp}" for name, resp in results])
                contradiction_prompt = f"""
                Analyze these AI model responses for contradictions or significant disagreements:
                
                {model_responses}
                
                Provide a brief analysis of any contradictions found, or confirm if responses are generally consistent.
                """
                
                try:
                    contradiction_analysis = call_openai_sync(contradiction_prompt)
                    st.success(contradiction_analysis[1])
                except Exception as e:
                    st.error(f"Contradiction analysis failed: {str(e)}")

# === ADVANCED ANALYSIS DISPLAY ===
if st.session_state.get("show_advanced_analysis", False):
    st.markdown("---")
    st.markdown("## 🛡️ Red/Blue/Purple Team Analysis")
    st.info("Advanced security analysis activated from sidebar controls")
    
    tab1, tab2, tab3 = st.tabs(["🔴 Red Team", "🔵 Blue Team", "🟣 Purple Team"])
    
    with tab1:
        st.markdown("### 🔴 Red Team Analysis (Adversarial Testing)")
        if openai_client and st.session_state.get("query_history"):
            latest_query = st.session_state.query_history[-1]
            latest_results = latest_query.get("results", [])
            
            with st.spinner("Running Red Team analysis..."):
                red_team_prompt = f"""
                As a Red Team analyst, analyze these AI responses for potential hallucinations and vulnerabilities:
                
                Original Query: {latest_query.get('question', 'N/A')}
                
                Responses: {chr(10).join([f"{name}: {resp}" for name, resp in latest_results])}
                
                Provide:
                1. Hallucination Risk Score (1-10)
                2. Specific vulnerability flags detected
                3. Suggested adversarial follow-up questions to test these models further
                4. Attack vectors that could exploit these weaknesses
                """
                
                try:
                    red_analysis = call_openai_sync(red_team_prompt)
                    st.error(red_analysis[1])
                except Exception as e:
                    st.error(f"Red team analysis failed: {str(e)}")
        else:
            st.info("Run a query first to enable Red Team analysis")
    
    with tab2:
        st.markdown("### 🔵 Blue Team Analysis (Defensive Assessment)")
        if openai_client and st.session_state.get("query_history"):
            latest_query = st.session_state.query_history[-1]
            latest_results = latest_query.get("results", [])
            
            blue_team_prompt = f"""
            As a Blue Team analyst, assess the reliability and trustworthiness of these AI responses:
            
            Original Query: {latest_query.get('question', 'N/A')}
            
            Responses: {chr(10).join([f"{name}: {resp}" for name, resp in latest_results])}
            
            Provide:
            1. Reliability score for each response (1-10)
            2. Trust indicators found (verifiable claims, appropriate uncertainty language)
            3. Verification recommendations for users
            4. Defense strategies against identified hallucinations
            """
            
            try:
                with st.spinner("Running Blue Team analysis..."):
                    blue_analysis = call_openai_sync(blue_team_prompt)
                    st.info(blue_analysis[1])
            except Exception as e:
                st.error(f"Blue team analysis failed: {str(e)}")
        else:
            st.info("Run a query first to enable Blue Team analysis")
    
    with tab3:
        st.markdown("### 🟣 Purple Team Analysis (Integrated Intelligence)")
        if openai_client and st.session_state.get("query_history"):
            latest_query = st.session_state.query_history[-1]
            latest_results = latest_query.get("results", [])
            
            purple_team_prompt = f"""
            As a Purple Team analyst, provide integrated offensive and defensive insights:
            
            Original Query: {latest_query.get('question', 'N/A')}
            
            Responses: {chr(10).join([f"{name}: {resp}" for name, resp in latest_results])}
            
            Provide:
            1. Overall risk assessment combining attack and defense perspectives
            2. Methodology recommendations for users
            3. Key learnings about AI model limitations revealed by this query
            4. Actionable intelligence for improving AI reliability assessment
            """
            
            try:
                with st.spinner("Running Purple Team analysis..."):
                    purple_analysis = call_openai_sync(purple_team_prompt)
                    st.success(purple_analysis[1])
            except Exception as e:
                st.error(f"Purple team analysis failed: {str(e)}")
        else:
            st.info("Run a query first to enable Purple Team analysis")

# === FOLLOW-UP CONVERSATION ===
st.subheader("💬 Follow-Up Conversation")

follow_up_input = st.text_input("Ask a question or follow-up:", key="followup_question")

if st.button("Send Follow-Up") and follow_up_input:
    if openai_client:
        st.session_state.chat_history.append({"role": "user", "content": follow_up_input})
        
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.extend(st.session_state.chat_history)
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
                max_tokens=500
            )
            assistant_reply = response.choices[0].message.content.strip()
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        except Exception as e:
            error_msg = f"[Error: {str(e)}]"
            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    else:
        st.error("Follow-up conversation requires OpenAI API key.")

if st.session_state.chat_history:
    st.markdown("**Conversation History:**")
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")

if st.button("Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

# === FOOTER ===
st.divider()
st.markdown("Built with ❤️ by Hallucinations.Cloud")
st.caption("Note: This tool compares multiple LLM responses for hallucination detection. Use responsibly.")