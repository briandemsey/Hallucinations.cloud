import streamlit as st

st.set_page_config(page_title="Hallucinations.cloud", layout="wide")
st.title("🧠 Hallucinations.cloud: Query Checker")

query = st.text_input("Enter a query to evaluate:")

if query:
    with st.spinner("Evaluating hallucination risk..."):
        # Placeholder for evaluation logic
        st.write(f"**Query**: `{query}`")
        st.success("✅ No hallucination detected.")
        st.caption("This is a placeholder result. Future versions will use LLM comparison and scoring.")