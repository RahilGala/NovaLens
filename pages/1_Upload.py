# pages/1_Upload.py
import streamlit as st
import pandas as pd
import json
from ai_local import ask_ai

st.set_page_config(page_title="Upload Data", page_icon="📂", layout="wide", initial_sidebar_state="collapsed")

# Dark mode CSS
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
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 50px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.6) !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: white;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #b8b8d1;
    }
    
    .stDataFrame {
        border-radius: 8px;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📂 Upload Your Data</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Upload your dataset and let AI understand it</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a file (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    try:
        with st.spinner("🔄 Loading your data..."):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state['df'] = df
            st.session_state['filename'] = uploaded_file.name
        
        st.success(f"✅ Loaded: **{uploaded_file.name}**")
        
        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            numeric_cols = df.select_dtypes(include=['number']).shape[1]
            st.metric("Numeric Columns", numeric_cols)
        with col4:
            text_cols = df.select_dtypes(include=['object']).shape[1]
            st.metric("Text Columns", text_cols)
        
        st.markdown("")
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("")
        st.markdown("### 🧠 AI Dataset Analysis")
        
        if st.button("🤖 Let AI Analyze This Dataset", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is analyzing your dataset..."):
                schema_info = []
                for col in df.columns:
                    col_info = {
                        "name": col,
                        "type": str(df[col].dtype),
                        "missing": int(df[col].isnull().sum()),
                        "unique": int(df[col].nunique()),
                        "sample_values": df[col].dropna().head(3).tolist()
                    }
                    schema_info.append(col_info)
                
                ai_prompt = f"""You are a professional data scientist. Analyze this dataset and provide insights.

Dataset: {uploaded_file.name}
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Details:
{json.dumps(schema_info, indent=2, default=str)}

Provide analysis in this EXACT JSON format:
{{
    "dataset_summary": "Brief 2-3 sentence summary",
    "data_quality": {{
        "overall_score": "Good/Fair/Poor",
        "issues": ["list of issues"]
    }},
    "key_columns": ["important columns"],
    "possible_analyses": [
        {{"type": "Analysis type", "description": "What insights", "business_value": "Why it matters"}}
    ],
    "recommended_next_steps": ["next steps"]
}}

Respond with ONLY valid JSON, no extra text."""
                
                try:
                    system_msg = {"role": "system", "content": "You are a data science expert. Return only valid JSON."}
                    user_msg = {"role": "user", "content": ai_prompt}
                    
                    ai_response = ask_ai([system_msg, user_msg])
                    
                    cleaned_response = ai_response.strip()
                    if cleaned_response.startswith("```"):
                        lines = cleaned_response.split("\n")
                        cleaned_response = "\n".join([l for l in lines if not l.strip().startswith("```")])
                    
                    insights = json.loads(cleaned_response)
                    st.session_state['ai_insights'] = insights
                    
                    st.markdown("## 🎯 AI Analysis Results")
                    
                    st.markdown("### 📋 What This Dataset Is About")
                    st.info(insights.get('dataset_summary', 'Analysis unavailable'))
                    
                    st.markdown("### 🔍 Data Quality Assessment")
                    quality = insights.get('data_quality', {})
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        score = quality.get('overall_score', 'Unknown')
                        if score == "Good":
                            st.success(f"**Quality: {score}**")
                        elif score == "Fair":
                            st.warning(f"**Quality: {score}**")
                        else:
                            st.error(f"**Quality: {score}**")
                    
                    with col2:
                        issues = quality.get('issues', [])
                        if issues:
                            st.markdown("**Issues Found:**")
                            for issue in issues:
                                st.markdown(f"- {issue}")
                        else:
                            st.markdown("No major issues detected ✅")
                    
                    st.markdown("### 🔑 Most Important Columns")
                    key_cols = insights.get('key_columns', [])
                    if key_cols:
                        cols_display = st.columns(min(len(key_cols), 4))
                        for i, col in enumerate(key_cols[:4]):
                            with cols_display[i % 4]:
                                st.info(f"**{col}**")
                    
                    st.markdown("### 💡 What You Can Do With This Data")
                    analyses = insights.get('possible_analyses', [])
                    if analyses:
                        for i, analysis in enumerate(analyses, 1):
                            with st.expander(f"{i}. {analysis.get('type', 'Analysis')}"):
                                st.markdown(f"**What:** {analysis.get('description', '')}")
                                st.markdown(f"**Why It Matters:** {analysis.get('business_value', '')}")
                    
                    st.markdown("### 🚀 Recommended Next Steps")
                    next_steps = insights.get('recommended_next_steps', [])
                    if next_steps:
                        for i, step in enumerate(next_steps, 1):
                            st.markdown(f"{i}. {step}")
                    
                    st.success("✅ AI analysis complete!")
                
                except json.JSONDecodeError:
                    st.error("AI response could not be parsed. Showing raw response:")
                    st.code(ai_response)
                except Exception as e:
                    st.error(f"AI analysis error: {e}")
        
        elif 'ai_insights' in st.session_state and st.session_state['ai_insights']:
            st.info("💡 Previous AI analysis available. Click the button above to refresh.")
        
        st.markdown("")
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("➡️ Continue to Data Cleaning", use_container_width=True):
                st.switch_page("pages/2_Cleaning.py")
    
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

else:
    st.info("""
    ### 📤 Upload Instructions
    
    1. Click the upload button above
    2. Select your CSV, Excel, or JSON file
    3. AI will automatically analyze it
    4. Get instant insights and recommendations
    
    **Supported Formats:** CSV, XLSX, JSON
    """)
