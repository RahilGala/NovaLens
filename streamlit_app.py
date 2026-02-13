# Home.py
import streamlit as st

st.set_page_config(
    page_title="NovaLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dark mode CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d1b3d 50%, #1a1a2e 100%);
    }
    
    .hero-title {
        text-align: center;
        font-size: 5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 300;
        color: #b8b8d1;
        margin-bottom: 3rem;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.3);
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #b8b8d1;
        line-height: 1.4;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 3rem !important;
        border-radius: 50px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.6) !important;
    }
    
    .status-indicator {
        position: fixed;
        bottom: 1.5rem;
        right: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown('<div class="hero-title">NovaLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Your Personal AI Data Scientist</div>', unsafe_allow_html=True)

# Feature Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📂</span>
        <div class="feature-title">Upload</div>
        <div class="feature-desc">Instant AI analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🧹</span>
        <div class="feature-title">Clean</div>
        <div class="feature-desc">Auto data fixes</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">Visualize</div>
        <div class="feature-desc">Smart dashboards</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <div class="feature-title">Predict</div>
        <div class="feature-desc">ML models</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">💬</span>
        <div class="feature-title">Chat</div>
        <div class="feature-desc">Ask anything</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# CTA Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🚀 Start Analysis", use_container_width=True):
        st.switch_page("pages/1_Upload.py")

# System Status
try:
    import ollama
    ollama.list()
    status_html = '<div class="status-indicator">✅ AI Online</div>'
except:
    status_html = '<div class="status-indicator">⚠️ Start Ollama</div>'

st.markdown(status_html, unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'ai_insights' not in st.session_state:
    st.session_state['ai_insights'] = {}
if 'cleaning_history' not in st.session_state:
    st.session_state['cleaning_history'] = []
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
