import streamlit as st
import pandas as pd
import os
import requests

# Direct token (for demo/testing — NOT for public sharing)
HF_API_TOKEN = "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"

# Load assessment catalog
@st.cache_data
def load_data():
    path = "shl_catalog_expanded.csv"
    if not os.path.exists(path):
        st.error("CSV file not found.")
        st.stop()
    return pd.read_csv(path)

# Query Hugging Face API with Falcon model
def query_huggingface(prompt):
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
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
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return result.get("generated_text", "⚠️ No meaningful response received.")
    except Exception as e:
        return f"❌ Error communicating with Hugging Face API: {e}"

# Streamlit UI
st.set_page_config(page_title="SHL Assessment Recommender")
st.title("🤖 SHL Assessment Recommender (Falcon-7B Powered)")

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
You are an intelligent assistant. Based on the following SHL assessments:

{assessment_text}

Please suggest the most relevant SHL assessments for the following job description:

{user_input}
"""
        with st.spinner("Thinking..."):
            response = query_huggingface(prompt)
        st.markdown(response)
