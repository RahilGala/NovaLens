# pages/3_Visualization.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from ai_local import ask_ai

st.set_page_config(page_title="Visualization", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

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
    h3 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">📊 AI-Generated Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">AI creates professional dashboards automatically</div>', unsafe_allow_html=True)

if "df" not in st.session_state or st.session_state["df"] is None:
    st.warning("No dataset found. Please upload first.")
    if st.button("← Go to Upload"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

df = st.session_state["df"]

schema = [{"name": c, "dtype": str(df[c].dtype), "unique": int(df[c].nunique())} for c in df.columns]

st.markdown("### 🤖 Generate Dashboard")

if st.button("🧠 Let AI Create Dashboard", type="primary", use_container_width=True):
    with st.spinner("🧠 AI is creating dashboard..."):
        
        ai_prompt = f"""Create a dashboard plan in EXACT JSON:
{{
    "executive_summary": {{
        "overview": "2-3 sentence summary",
        "key_findings": ["finding1", "finding2", "finding3"],
        "recommendations": ["rec1", "rec2"]
    }},
    "kpis": [
        {{"name": "KPI name", "value_column": "column", "aggregation": "sum|mean|count", "insight": "what it means"}}
    ],
    "charts": [
        {{"title": "Chart title", "chart_type": "bar|line|scatter|histogram|pie", "x": "column", "y": "column", "aggregation": "sum|mean|count|null", "business_insight": "insight"}}
    ]
}}

Dataset: {len(df)} rows, {len(df.columns)} columns
Columns: {json.dumps(schema, default=str)}

Rules: 3-5 KPIs, 5-7 charts. Match types to data. ONLY valid JSON."""
        
        try:
            system_msg = {"role": "system", "content": "Return ONLY valid JSON."}
            user_msg = {"role": "user", "content": ai_prompt}
            
            ai_response = ask_ai([system_msg, user_msg])
            
            cleaned_response = ai_response.strip()
            if cleaned_response.startswith("```"):
                lines = cleaned_response.split("\n")
                cleaned_response = "\n".join([l for l in lines if not l.strip().startswith("```")])
            
            dashboard_plan = json.loads(cleaned_response)
            st.session_state['dashboard_plan'] = dashboard_plan
            st.session_state['analysis_complete'] = True
            
            st.success("✅ Dashboard created!")
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

if st.session_state.get('analysis_complete') and 'dashboard_plan' in st.session_state:
    dashboard = st.session_state['dashboard_plan']
    
    st.markdown("## 📋 Executive Summary")
    summary = dashboard.get('executive_summary', {})
    st.info(summary.get('overview', ''))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Key Findings")
        for finding in summary.get('key_findings', []):
            st.markdown(f"- {finding}")
    
    with col2:
        st.markdown("### 💡 Recommendations")
        for rec in summary.get('recommendations', []):
            st.markdown(f"- {rec}")
    
    st.divider()
    
    st.markdown("## 📈 Key Metrics")
    kpis = dashboard.get('kpis', [])
    if kpis:
        kpi_cols = st.columns(len(kpis))
        for i, kpi in enumerate(kpis):
            with kpi_cols[i]:
                col_name = kpi.get('value_column')
                agg = kpi.get('aggregation', 'sum')
                
                if col_name and col_name in df.columns:
                    try:
                        if agg == 'sum':
                            value = df[col_name].sum()
                        elif agg == 'mean':
                            value = df[col_name].mean()
                        elif agg == 'count':
                            value = df[col_name].count()
                        else:
                            value = df[col_name].sum()
                        
                        display = f"{value/1000000:.2f}M" if value > 1000000 else f"{value/1000:.2f}K" if value > 1000 else f"{value:.2f}"
                        
                        st.metric(kpi.get('name', col_name), display)
                        st.caption(kpi.get('insight', ''))
                    except:
                        pass
    
    st.divider()
    
    st.markdown("## 📊 Visualizations")
    
    charts = dashboard.get('charts', [])
    for i, chart in enumerate(charts, 1):
        st.markdown(f"### {i}. {chart.get('title', 'Chart')}")
        
        chart_type = chart.get('chart_type')
        x_col = chart.get('x')
        y_col = chart.get('y')
        
        try:
            plot_df = df.copy()
            
            if chart_type == 'bar' and x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.bar(plot_df, x=x_col, y=y_col, title=chart.get('title'))
            elif chart_type == 'line' and x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.line(plot_df, x=x_col, y=y_col, title=chart.get('title'))
            elif chart_type == 'scatter' and x_col and y_col and x_col in df.columns and y_col in df.columns:
                fig = px.scatter(plot_df, x=x_col, y=y_col, title=chart.get('title'))
            elif chart_type == 'histogram' and x_col and x_col in df.columns:
                fig = px.histogram(plot_df, x=x_col, title=chart.get('title'))
            elif chart_type == 'pie' and x_col and x_col in df.columns:
                counts = df[x_col].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title=chart.get('title'))
            else:
                continue
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            insight = chart.get('business_insight', '')
            if insight:
                st.info(f"💡 {insight}")
        
        except Exception as e:
            st.warning(f"Could not create chart: {e}")
        
        st.divider()

st.markdown("")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("➡️ Continue to Machine Learning", use_container_width=True):
        st.switch_page("pages/4_Machine_Learning.py")