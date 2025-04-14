import os
import streamlit as st
import pandas as pd
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize model globally
model = None

# Configure Gemini and set model
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Gemini setup error: {e}")
else:
    st.error("🚫 Gemini API key not found in .env file.")

# ----------------------
# PDF & CSV Helpers
# ----------------------
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return "".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return ""

def load_csv_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"CSV Error: {e}")
        return None

# ----------------------
# MAIN FUNCTION
# ----------------------
def llm_multimodal_section():
    global model  # ✅ This ensures you're using the model defined above

    st.title(" Gemini Q&A on Documents & Data")
    st.write("Ask questions about datasets and documents using Gemini AI.")

    if model is None:
        st.warning("Gemini model is not initialized. Please check your API key.")
        return

    datasets = {
        " ACity Handbook (PDF)": "datasets/ACity_handbook.pdf"
    }

    dataset_name = st.selectbox("Choose a dataset to analyze:", list(datasets.keys()))
    selected_path = datasets[dataset_name]
    content = ""

    if selected_path.endswith(".csv"):
        df = load_csv_data(selected_path)
        if df is not None:
            st.subheader(" Dataset Preview")
            st.dataframe(df.head())
            content = df.to_string(index=False)

    elif selected_path.endswith(".pdf"):
        content = extract_text_from_pdf(selected_path)
        st.subheader(" PDF Preview")
        st.text(content[:1000] + "..." if len(content) > 1000 else content)

    st.subheader(" Ask a Question About the Selected Dataset")
    question = st.text_input("Enter your question below:")
    if st.button("Submit Question"):
        if not question or not content:
            st.warning("Make sure both a dataset and a question are provided.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content([content, question])
                    st.success("Gemini says:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"LLM Error: {e}")
