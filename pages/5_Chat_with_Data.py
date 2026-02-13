#pages/5_Chat_with_Data.py
import streamlit as st
import pandas as pd
import ollama
from ai_local import MODEL_NAME

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b3d 50%, #1a1a2e 100%);
    }
    .page-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .page-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #b8b8d1;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    h3 {
        color: white;
    }
    .suggestions {
        color: #b8b8d1;
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">💬 Chat with Your Data</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Ask questions in plain English</div>', unsafe_allow_html=True)

try:
    ollama.list()
except:
    st.error("❌ Ollama not running. Please start Ollama.")
    st.stop()

if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("⚠️ No dataset. Please upload first.")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

df = st.session_state["df"]

schema = "\n".join([f"- {c} ({df[c].dtype}, unique: {df[c].nunique()})" for c in df.columns])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about your data...")


if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    system_prompt = f"""You are a data analyst.
Answer ONLY using this dataset schema:

{schema}

Provide clear answers. If relevant, give Python code using DataFrame named 'df'.
Do not invent columns."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ollama.chat(model=MODEL_NAME, messages=messages)
                answer = response["message"]["content"]
                
                st.markdown(answer)
                
                st.session_state["messages"].append({"role": "assistant", "content": answer})
            
            except Exception as e:
                error_msg = f"⚠️ Error: {e}"
                st.error(error_msg)
                st.session_state["messages"].append({"role": "assistant", "content": error_msg})

st.markdown("")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.switch_page("pages/1_Upload.py")
