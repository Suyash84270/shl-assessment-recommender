import streamlit as st
import pandas as pd
import os
import requests

# Load assessment catalog
@st.cache_data
def load_data():
    path = "shl_catalog_expanded.csv"
    if not os.path.exists(path):
        st.error("CSV file not found.")
        st.stop()
    return pd.read_csv(path)

# Query local TinyLlama model via Ollama
def query_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "tinyllama",  # Change this if you're using a different model name
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"❌ Error communicating with TinyLlama: {e}"

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender")
st.title("🤖 SHL Assessment Recommender (TinyLlama-Powered)")

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
        prompt = f"""
Given the following available assessments:

{assessment_text}

Based on the user query below, recommend the most relevant SHL assessments:

Query: {user_input}
"""
        response = query_ollama(prompt)
        st.markdown(response)
