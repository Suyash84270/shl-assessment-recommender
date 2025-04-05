import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    path = "shl_catalog_expanded.csv"
    if not os.path.exists(path):
        st.error("CSV file not found.")
        st.stop()
    return pd.read_csv(path)

# Dummy LLM logic
def query_ollama(prompt):
    return """
1. **Cognitive Ability Test** – To assess problem-solving and logical thinking.
2. **Personality Inventory** – To evaluate team fit and behavior.
3. **Situational Judgement Test** – For workplace decision-making skills.
"""

st.set_page_config(page_title="SHL Assessment Recommender")
st.title("🤖 SHL Assessment Recommender (Demo)")

user_input = st.text_area("Paste Job Description or Query:")

if st.button("Recommend"):
    df = load_data()
    if user_input.strip() == "":
        st.warning("Please enter something.")
    else:
        assessment_text = "\n".join([
            f"{row['name']}: {row['description']}" 
            for _, row in df.iterrows()
        ])
        prompt = f"Available assessments:\n{assessment_text}\nUser query: {user_input}"
        response = query_ollama(prompt)
        st.markdown(response)
