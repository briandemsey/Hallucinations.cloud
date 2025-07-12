import pandas as pd
import os

def append_history_to_csv(row, filename="query_history_log.csv"):
    df = pd.DataFrame([row])
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False)
    else:
        df.to_csv(filename, index=False)


import streamlit as st
import os
import asyncio
import aiohttp
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import anthropic
import google.generativeai as genai
from datetime import datetime

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

openai_client = OpenAI(api_key=openai_key)
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
genai.configure(api_key=google_key)

if "history" not in st.session_state:
    st.session_state["history"] = []

async def call_openai(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful, accurate assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.5
        )
        return ("GPT-4o", response.choices[0].message.content.strip())
    except Exception as e:
        return ("GPT-4o", f"[OpenAI error: {str(e)}]")

async def call_claude(prompt):
    try:
        message = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return ("Claude", message.content[0].text.strip())
    except Exception as e:
        return ("Claude", f"[Claude error: {str(e)}]")

async def call_gemini(prompt):
    model_id = "models/gemini-1.5-pro"
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return ("Gemini", response.text.strip())
    except Exception as e:
        return ("Gemini", f"[Gemini error: {str(e)}]")

async def call_grok_simulated(prompt):
    try:
        grok_prompt = (
            "You are Grok (simulated): witty, direct, and accurate. "
            "Answer with clarity but a bit of irreverent flair.\n\n" + prompt
        )
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are Grok (simulated), a sharp-tongued but truthful assistant."},
                {"role": "user", "content": grok_prompt}
            ],
            temperature=0.5
        )
        return ("Grok (simulated)", response.choices[0].message.content.strip())
    except Exception as e:
        return ("Grok (simulated)", f"[Grok error: {str(e)}]")


async def call_openrouter(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openrouter/mistralai/mixtral-8x7b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers) as response:
                result = await response.json()
                return ("OpenRouter", result["choices"][0]["message"]["content"].strip())
    except Exception as e:
        return ("OpenRouter", f"[OpenRouter error: {str(e)}]")


async def gather_answers(prompt):
    return await asyncio.gather(
        call_openai(prompt),
        call_openrouter(prompt),
        call_claude(prompt),
        call_gemini(prompt),
        call_grok_simulated(prompt)
    )

# Removed incomplete function definition
    if st.session_state["history"]:
        query_options = [
            f'{i+1}: {row["Timestamp"][:19]} - {row["Question"][:40]}'
            for i, row in enumerate(st.session_state["history"])
        ]
        selected_index = st.sidebar.selectbox("Select a past query to rerun:", [""] + query_options)
        if selected_index and selected_index != "":
            selected_idx = int(selected_index.split(":")[0]) - 1
            user_query = st.session_state["history"][selected_idx]["Question"]


def run_app():
    past_question = None  # Ensure safe default
    st.title("Hallucinations.cloud Multi-Model")
    st.write("Compare model responses and provide your feedback.")

    user_query = st.text_input("Enter your factual question:")
    if past_question and not user_query.strip():
        user_query = past_question

    # Save to local CSV
    import csv
    from datetime import datetime
    history_file = "query_history.csv"
    def append_to_history(question, response, model_used):
        with open(history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), question, model_used, response])
        

    if st.button("Submit"):
        if not user_query.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Contacting models..."):
            results = asyncio.run(gather_answers(user_query))

        st.subheader("🔍 Model Responses")
        feedback_row = {"Timestamp": datetime.now().isoformat(), "Question": user_query}
        for name, answer in results:
            st.markdown(f"**{name}**")
            st.text_area(f"{name} says:", answer, height=200)
            feedback = st.radio(f"Was {name} helpful?", ["✅ Yes", "❌ No"], key=name)
            feedback_row[f"{name}_Feedback"] = feedback

        joined = "\n".join([f"{n}: {a}" for n, a in results])
        contradiction_prompt = (
            "Review the following model responses to the same question. "
            "Do they contradict each other? Respond YES or NO and explain.\n\n" + joined
        )
        try:
            contradiction_response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a contradiction detector."},
                    {"role": "user", "content": contradiction_prompt}
                ],
                temperature=0.2
            ).choices[0].message.content.strip()
        except Exception as e:
            contradiction_response = f"[Contradiction check error: {str(e)}]"

        st.subheader("⚖️ Contradiction Check")
        st.write(contradiction_response)
        feedback_row["Contradiction_Check"] = contradiction_response
        st.session_state["history"].append(feedback_row)
        append_history_to_csv(feedback_row)

    if st.button("Export Feedback as CSV"):
        df = pd.DataFrame(st.session_state["history"])
        st.download_button("Download CSV", df.to_csv(index=False), file_name="feedback_log.csv", mime="text/csv")
    if st.session_state["history"]:
        query_options = [
            f'{i+1}: {row["Timestamp"][:19]} - {row["Question"][:40]}'
            for i, row in enumerate(st.session_state["history"])
        ]
        selected_index = st.sidebar.selectbox("Select a past query to rerun:", [""] + query_options)
        if selected_index and selected_index != "":
            selected_idx = int(selected_index.split(":")[0]) - 1
            user_query = st.session_state["history"][selected_idx]["Question"]


run_app()



# === Follow-Up Conversation Section ===
st.subheader("💬 Conversation")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Ask a question or follow-up:")

if st.button("Submit Follow-Up") and user_input:
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": "You are a helpful assistant."}] + st.session_state["chat_history"]

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5
        )
        assistant_reply = response.choices[0].message.content.strip()
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        assistant_reply = f"[Error: {str(e)}]"
        st.session_state["chat_history"].append({"role": "assistant", "content": assistant_reply})

st.markdown("---")
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**GPT-4o:** {msg['content']}")

if st.button("Clear Conversation"):
    st.session_state["chat_history"] = []
# === Sidebar User Input Section ===
with st.sidebar:
    st.subheader("💡 Suggest a Model")
    st.write("Should you wish to include another model, please indicate in the following space.")

    suggested_model = st.text_input("Model Suggestion")
    user_name = st.text_input("Your Name")
    user_email = st.text_input("Your Email")

# === Add Suggested Model to List Dynamically ===
model_list = ["GPT-4o", "Claude 3 Haiku", "Gemini 1.5 Pro", "Grok (simulated)", "OpenRouter"]
if suggested_model:
    if suggested_model not in model_list:
        model_list.append(suggested_model)

with st.sidebar:
    st.header("🧠 Models in Use")
    for model in model_list:
        st.markdown(f"- {model}")


