# pages/2_Cleaning.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from ai_local import ask_ai

st.set_page_config(page_title="Data Cleaning", page_icon="🧹", layout="wide", initial_sidebar_state="collapsed")

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

st.markdown('<div class="page-title">🧹 AI-Powered Data Cleaning</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Let AI automatically clean your data</div>', unsafe_allow_html=True)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ No data found! Please upload a file first.")
    if st.button("← Go to Upload Page"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

df = st.session_state['df']

st.markdown("### 📊 Current Dataset Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{df.shape[0]:,}")
with col2:
    st.metric("Columns", df.shape[1])
with col3:
    total_missing = df.isnull().sum().sum()
    st.metric("Missing Values", f"{total_missing:,}")
with col4:
    duplicates = df.duplicated().sum()
    st.metric("Duplicate Rows", duplicates)

st.divider()

st.markdown("### 🤖 AI Automatic Cleaning")

if st.button("🧠 Let AI Clean This Dataset", type="primary", use_container_width=True):
    with st.spinner("🧠 AI is creating a cleaning plan..."):
        quality_report = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "duplicates": int(df.duplicated().sum()),
            "columns": []
        }
        
        for col in df.columns:
            col_report = {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing_count": int(df[col].isnull().sum()),
                "missing_percent": round(float(df[col].isnull().sum() / len(df) * 100), 2),
                "unique_values": int(df[col].nunique())
            }
            quality_report["columns"].append(col_report)
        
        ai_prompt = f"""You are a data cleaning expert. Create a cleaning plan.

Dataset Quality:
{json.dumps(quality_report, indent=2, default=str)}

Return EXACT JSON format:
{{
    "plan_summary": "Brief cleaning strategy",
    "steps": [
        {{
            "action": "remove_duplicates|fill_missing|drop_column|standardize_nulls",
            "target": "column_name or list",
            "method": "mean|median|mode|drop",
            "reason": "Why needed"
        }}
    ],
    "expected_impact": "What improves"
}}

Rules: Use mean/median for numeric, mode for categorical. Only drop if >90% missing. Be conservative.

Respond ONLY with valid JSON."""
        
        try:
            system_msg = {"role": "system", "content": "Return only valid JSON."}
            user_msg = {"role": "user", "content": ai_prompt}
            
            ai_response = ask_ai([system_msg, user_msg])
            
            cleaned_response = ai_response.strip()
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                cleaned_response = "\n".join([l for l in lines if not l.strip().startswith("```")])
            
            cleaning_plan = json.loads(cleaned_response)
            
            st.markdown("## 📋 AI Cleaning Plan")
            st.info(cleaning_plan.get('plan_summary', 'Cleaning plan generated'))
            
            st.markdown("### 🔧 Cleaning Steps")
            steps = cleaning_plan.get('steps', [])
            
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}: {step.get('action', '').replace('_', ' ').title()}"):
                    st.markdown(f"**Target:** {step.get('target')}")
                    st.markdown(f"**Method:** {step.get('method')}")
                    st.markdown(f"**Reason:** {step.get('reason')}")
            
            st.success(cleaning_plan.get('expected_impact', 'Data quality will improve'))
            
            if st.button("✅ Apply Cleaning Plan", type="primary"):
                with st.spinner("🔄 Applying cleaning..."):
                    original_shape = df.shape
                    cleaning_log = []
                    
                    for step in steps:
                        action = step.get('action')
                        target = step.get('target')
                        method = step.get('method')
                        
                        try:
                            if action == 'remove_duplicates':
                                before = len(df)
                                df = df.drop_duplicates()
                                removed = before - len(df)
                                cleaning_log.append(f"✅ Removed {removed} duplicates")
                            
                            elif action == 'standardize_nulls':
                                df.replace(["?", "na", "NA", "null", "-", " ", ""], np.nan, inplace=True)
                                cleaning_log.append("✅ Standardized nulls")
                            
                            elif action == 'fill_missing':
                                cols = [target] if isinstance(target, str) and target != 'all' else (target if isinstance(target, list) else df.columns.tolist())
                                
                                for col in cols:
                                    if col not in df.columns:
                                        continue
                                    
                                    missing_before = df[col].isnull().sum()
                                    
                                    if df[col].dtype in ['float64', 'int64']:
                                        if method == 'mean':
                                            df[col].fillna(df[col].mean(), inplace=True)
                                        elif method == 'median':
                                            df[col].fillna(df[col].median(), inplace=True)
                                    else:
                                        if method == 'mode' and not df[col].mode().empty:
                                            df[col].fillna(df[col].mode()[0], inplace=True)
                                    
                                    filled = missing_before - df[col].isnull().sum()
                                    if filled > 0:
                                        cleaning_log.append(f"✅ Filled {filled} in '{col}' ({method})")
                            
                            elif action == 'drop_column':
                                cols = [target] if isinstance(target, str) else target
                                cols = [c for c in cols if c in df.columns]
                                if cols:
                                    df.drop(columns=cols, inplace=True)
                                    cleaning_log.append(f"✅ Dropped: {', '.join(cols)}")
                        
                        except Exception as e:
                            cleaning_log.append(f"⚠️ {action}: {str(e)}")
                    
                    st.session_state['df'] = df
                    st.session_state['cleaning_history'] = cleaning_log
                    
                    st.success("✅ Cleaning complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows Before", f"{original_shape[0]:,}")
                        st.metric("Rows After", f"{df.shape[0]:,}", delta=df.shape[0] - original_shape[0])
                    with col2:
                        st.metric("Columns Before", original_shape[1])
                        st.metric("Columns After", df.shape[1], delta=df.shape[1] - original_shape[1])
                    
                    st.markdown("### 📝 Cleaning Log")
                    for log in cleaning_log:
                        st.markdown(f"- {log}")
                    
                    st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()

if 'cleaning_history' in st.session_state and st.session_state['cleaning_history']:
    st.markdown("### 📜 Cleaning History")
    for log in st.session_state['cleaning_history']:
        st.markdown(f"- {log}")

st.markdown("### 🔍 Data Preview")
st.dataframe(df.head(20), use_container_width=True)

st.markdown("### 💾 Download")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Cleaned CSV", data=csv, file_name="cleaned_data.csv", mime="text/csv", use_container_width=True)

st.markdown("")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("➡️ Continue to Visualization", use_container_width=True):
        st.switch_page("pages/3_Visualization.py")
