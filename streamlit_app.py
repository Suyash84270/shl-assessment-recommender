import streamlit as st
import pandas as pd
import os
import requests

# Your Hugging Face token (used directly — insecure for public sharing)
HF_API_TOKEN = "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"

# Load assessment catalog
@st.cache_data
def load_data():
    path = "shl_catalog_expanded.csv"
    if not os.path.exists(path):
        st.error("CSV file not found.")
        st.stop()
    return pd.read_csv(path)

# Query Hugging Face Inference API directly using token
def query_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list):
            return result[0]["generated_text"]
        else:
            return result.get("generated_text", "No response generated.")
    except Exception as e:
        return f"❌ Error communicating with Hugging Face API: {e}"

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender")
st.title("🤖 SHL Assessment Recommender (HF LLM Powered)")

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
Given the following SHL assessments:

{assessment_text}

Suggest the most relevant assessments based on the job description:

{user_input}
"""
        with st.spinner("Generating recommendations..."):
            response = query_huggingface(prompt)
        st.markdown(response)
