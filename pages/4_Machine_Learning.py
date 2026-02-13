# pages/4_Machine_Learning.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from ai_local import ask_ai

st.set_page_config(page_title="ML Predictions", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")

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
    
    h3, h2, h4 {
        color: white;
    }
    
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">🎯 AI-Powered Predictions</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">AI builds and trains machine learning models automatically</div>', unsafe_allow_html=True)

if 'df' not in st.session_state or st.session_state['df'] is None:
    st.warning("⚠️ No data found! Please upload a file first.")
    if st.button("← Go to Upload Page"):
        st.switch_page("pages/1_Upload.py")
    st.stop()

df = st.session_state['df']

st.divider()

st.markdown("### 🤖 Automatic ML Model Builder")
st.markdown("#### Step 1: What do you want to predict?")

target_col = st.selectbox(
    "Select the target variable (what you want to predict):",
    df.columns,
    help="AI will automatically determine the best model type"
)

if target_col:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique Values", df[target_col].nunique())
    with col2:
        st.metric("Missing Values", df[target_col].isnull().sum())
    with col3:
        st.metric("Data Type", str(df[target_col].dtype))
    
    st.divider()
    
    st.markdown("#### Step 2: Let AI Analyze and Build Model")
    
    if st.button("🧠 Let AI Build Prediction Model", type="primary", use_container_width=True):
        with st.spinner("🧠 AI is analyzing your data and building a prediction model..."):
            
            schema = []
            for col in df.columns:
                if col != target_col:
                    schema.append({
                        "name": col,
                        "dtype": str(df[col].dtype),
                        "unique": int(df[col].nunique()),
                        "missing": int(df[col].isnull().sum())
                    })
            
            target_info = {
                "name": target_col,
                "dtype": str(df[target_col].dtype),
                "unique": int(df[target_col].nunique())
            }
            
            ai_prompt = f"""You are a machine learning expert. Analyze this prediction task.

Target Variable:
{json.dumps(target_info, indent=2, default=str)}

Available Features:
{json.dumps(schema, indent=2, default=str)}

Dataset size: {len(df)} rows

Create ML strategy in EXACT JSON format:
{{
    "problem_type": "classification|regression",
    "reasoning": "Why you chose this",
    "recommended_model": "random_forest|logistic_regression|linear_regression|gradient_boosting|decision_tree",
    "model_reasoning": "Why this model",
    "feature_selection": ["column names to use"],
    "feature_reasoning": "Why these features",
    "expected_accuracy": "Performance range",
    "business_application": "How to use predictions"
}}

Rules:
- Classification if target <20 unique values
- Regression if continuous numeric
- Random Forest is best for both
- Exclude high missing (>50%) columns

Respond ONLY with valid JSON, no extra text."""
            
            try:
                system_msg = {"role": "system", "content": "You are ML expert. Return only valid JSON."}
                user_msg = {"role": "user", "content": ai_prompt}
                
                ai_response = ask_ai([system_msg, user_msg])
                
                cleaned_response = ai_response.strip()
                if cleaned_response.startswith("```"):
                    lines = cleaned_response.split("\n")
                    cleaned_response = "\n".join([l for l in lines if not l.strip().startswith("```")])
                
                ml_strategy = json.loads(cleaned_response)
                st.session_state['ml_strategy'] = ml_strategy
                
                st.rerun()
            
            except json.JSONDecodeError:
                st.error("❌ AI response could not be parsed.")
                st.code(ai_response)
            except Exception as e:
                st.error(f"❌ Error: {e}")

if 'ml_strategy' in st.session_state:
    
    st.markdown("## 📋 AI ML Strategy")
    
    ml_strategy = st.session_state['ml_strategy']
    
    col1, col2 = st.columns(2)
    
    with col1:
        problem_type = ml_strategy.get('problem_type', 'unknown')
        if problem_type == 'classification':
            st.success(f"**Problem Type:** Classification")
        else:
            st.info(f"**Problem Type:** Regression")
        st.markdown(f"*{ml_strategy.get('reasoning', '')}*")
    
    with col2:
        model_name = ml_strategy.get('recommended_model', 'random_forest')
        st.success(f"**Selected Model:** {model_name.replace('_', ' ').title()}")
        st.markdown(f"*{ml_strategy.get('model_reasoning', '')}*")
    
    st.markdown("### 🎯 Selected Features")
    features = ml_strategy.get('feature_selection', [])
    st.info(f"**Features:** {', '.join(features)}")
    st.caption(ml_strategy.get('feature_reasoning', ''))
    
    st.markdown("### 💡 Business Application")
    st.info(ml_strategy.get('business_application', ''))
    
    st.markdown("### 📊 Expected Performance")
    st.success(ml_strategy.get('expected_accuracy', ''))
    
    st.divider()
    
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("🔄 Training model..."):
            
            try:
                feature_cols = [f for f in features if f in df.columns]
                
                if not feature_cols:
                    st.error("No valid features found!")
                    st.stop()
                
                X = df[feature_cols].copy()
                y = df[target_col].copy()
                
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                
                for col in X.columns:
                    if X[col].dtype == 'object':
                        if X[col].isnull().any():
                            X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown', inplace=True)
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                
                X.fillna(X.mean(), inplace=True)
                
                label_encoder = None
                if problem_type == 'classification' and y.dtype == 'object':
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if problem_type == 'classification':
                    if model_name == 'random_forest':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif model_name == 'logistic_regression':
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingClassifier(random_state=42)
                    else:
                        model = DecisionTreeClassifier(random_state=42)
                else:
                    if model_name == 'random_forest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_name == 'linear_regression':
                        model = LinearRegression()
                    elif model_name == 'gradient_boosting':
                        model = GradientBoostingRegressor(random_state=42)
                    else:
                        model = DecisionTreeRegressor(random_state=42)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.session_state['trained_model'] = {
                    'model': model,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'features': feature_cols,
                    'problem_type': problem_type,
                    'label_encoder': label_encoder
                }
                
                st.success("✅ Model trained successfully!")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Training error: {e}")

if 'trained_model' in st.session_state:
    
    st.divider()
    st.markdown("## 📈 Model Performance Results")
    
    model_data = st.session_state['trained_model']
    problem_type = model_data['problem_type']
    y_test = model_data['y_test']
    y_pred = model_data['y_pred']
    model = model_data['model']
    features = model_data['features']
    
    if problem_type == 'classification':
        
        accuracy = accuracy_score(y_test, y_pred)
        
        n_classes = len(np.unique(y_test))
        avg_method = 'binary' if n_classes == 2 else 'weighted'
        
        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
        
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Precision", f"{precision:.2%}")
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        with col4:
            st.metric("F1-Score", f"{f1:.2%}")
        
        st.divider()
        
        st.markdown("### 📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues")
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            st.markdown("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True, height=400)
        
        st.divider()
        
        st.markdown("### 🧠 AI Performance Interpretation")
        
        if accuracy >= 0.9:
            performance_level = "Excellent"
            st.success(f"🎉 **{performance_level}** - Accuracy: {accuracy:.1%}\n\nThe model correctly predicts {accuracy:.1%} of cases. This model can be confidently used for predictions.")
        elif accuracy >= 0.75:
            performance_level = "Good"
            st.info(f"✅ **{performance_level}** - Accuracy: {accuracy:.1%}\n\nThe model is suitable for most use cases. Monitor performance on real data.")
        elif accuracy >= 0.6:
            performance_level = "Fair"
            st.warning(f"⚠️ **{performance_level}** - Accuracy: {accuracy:.1%}\n\nConsider gathering more data or engineering better features.")
        else:
            performance_level = "Poor"
            st.error(f"❌ **{performance_level}** - Accuracy: {accuracy:.1%}\n\nThe model needs significant improvement.")
    
    else:
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAE", f"{mae:.2f}")
        with col4:
            st.metric("MSE", f"{mse:.2f}")
        
        st.divider()
        
        st.markdown("### 📊 Prediction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Actual vs Predicted")
            fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect', line=dict(color='red', dash='dash', width=2)))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Residuals")
            residuals = y_test - y_pred
            
            fig_res = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Error'})
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(height=400)
            st.plotly_chart(fig_res, use_container_width=True)
        
        st.divider()
        
        st.markdown("### 🧠 AI Performance Interpretation")
        
        if r2 >= 0.8:
            performance_level = "Excellent"
            st.success(f"🎉 **{performance_level}** - R² Score: {r2:.4f}\n\nThe model explains {r2*100:.1f}% of variance. Predictions are highly reliable.")
        elif r2 >= 0.6:
            performance_level = "Good"
            st.info(f"✅ **{performance_level}** - R² Score: {r2:.4f}\n\nThis model performs well for most predictions.")
        elif r2 >= 0.4:
            performance_level = "Fair"
            st.warning(f"⚠️ **{performance_level}** - R² Score: {r2:.4f}\n\nConsider more data or features.")
        else:
            performance_level = "Poor"
            st.error(f"❌ **{performance_level}** - R² Score: {r2:.4f}\n\nThe model struggles with predictions.")
    
    if hasattr(model, 'feature_importances_'):
        st.divider()
        st.markdown("### 🌟 Feature Importance")
        
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig_imp, use_container_width=True)
        
        top_features = feature_imp_df.head(3)['Feature'].tolist()
        st.info(f"💡 **Key Drivers:** {', '.join(top_features)}")
    
    st.divider()
    st.markdown("### 📄 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_text = f"""ML Model Performance Report
Target: {target_col}
Problem: {problem_type}
Model: {ml_strategy.get('recommended_model', 'Unknown')}

Performance:
"""
        if problem_type == 'classification':
            report_text += f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}"
        else:
            report_text += f"R² Score: {r2:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}"
        
        st.download_button("📥 Download Report", data=report_text, file_name="model_report.txt", mime="text/plain", use_container_width=True)
    
    with col2:
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        if problem_type == 'regression':
            results_df['Error'] = y_test - y_pred
        
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv_data, file_name="predictions.csv", mime="text/csv", use_container_width=True)

st.markdown("")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("➡️ Continue to Chat", use_container_width=True):
        st.switch_page("pages/5_Chat_with_Data.py")