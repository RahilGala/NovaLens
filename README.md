# 🧠 NovaLens AI - Complete Setup & User Guide

> **Your Personal AI Data Scientist**  
> Upload data. Get insights. No coding required.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Quick Start](#quick-start)
5. [How to Use NovaLens](#how-to-use-novalens)
6. [Page-by-Page Guide](#page-by-page-guide)
7. [Troubleshooting](#troubleshooting)
8. [FAQs](#faqs)

---

## 🎯 Overview

**NovaLens AI** is an intelligent data science platform that acts as your personal AI data scientist. It automatically:
- Analyzes and understands your datasets
- Cleans data quality issues
- Creates professional dashboards
- Builds machine learning models
- Answers questions about your data

**No coding required. No data science knowledge needed.**

---

## 💻 System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **RAM:** 8GB (12GB recommended)
- **Disk Space:** 10GB free
- **CPU:** 4+ cores recommended
- **Python:** 3.8 or higher

### Recommended System
- **RAM:** 16GB
- **CPU:** 6+ cores
- **GPU:** Not required

---

## 🚀 Installation Guide

Follow these steps to set up NovaLens AI.

### Step 1: Install Ollama

#### **macOS:**
```bash
brew install ollama
```

#### **Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### **Windows:**
Download from: https://ollama.ai/download

**Verify:**
```bash
ollama --version
```

---

### Step 2: Start Ollama Service

```bash
ollama serve
```

**Keep this terminal open.** Ollama must stay running.

---

### Step 3: Download AI Model

```bash
ollama pull qwen2.5:7b
```

This downloads ~4.7GB. Takes 5-10 minutes.

**Verify:**
```bash
ollama list
```

---

### Step 4: Install Python

Check if installed:
```bash
python3 --version
```

**If not installed:**
- **macOS:** `brew install python@3.11`
- **Linux:** `sudo apt install python3.11 python3-pip`
- **Windows:** Download from python.org

---

### Step 5: Get NovaLens Project

```bash
cd /path/to/novalens-ai
```

---

### Step 6: Create Virtual Environment

```bash
# Create environment
python3 -m venv venv

# Activate it
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

---

### Step 7: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 8: Verify Installation

```bash
python -c "import ollama; print('✅ Ollama ready!')"
streamlit --version
```

---

## ⚡ Quick Start

```bash
# Make sure:
# 1. Ollama is running (ollama serve)
# 2. Virtual environment is activated (venv)
# 3. You're in the project folder

streamlit run Home.py
```

**Open browser:** http://localhost:8501

---

## 📖 How to Use NovaLens

### The 5-Step Workflow

```
Home → Upload → Clean → Visualize → Predict → Chat
```

Each page has AI-powered buttons that do the work automatically.

---

## 📄 Page-by-Page Guide

### 🏠 Home Page

**Purpose:** Introduction and entry point

**Actions:**
- Click **"🚀 Start Analyzing Your Data"**

---

### 📂 Page 1: Upload Data

**Purpose:** Upload and understand your dataset

**Steps:**
1. Upload CSV, Excel, or JSON file
2. View dataset overview (rows, columns, types)
3. Click **"🤖 Let AI Analyze This Dataset"**
4. Review AI insights:
   - Dataset summary
   - Data quality score
   - Key columns
   - Possible analyses
   - Next steps
5. Click **"➡️ Continue to Data Cleaning"**

**What AI tells you:**
- What your data is about
- Data quality (Good/Fair/Poor)
- Issues found
- What analysis you can do

---

### 🧹 Page 2: Data Cleaning

**Purpose:** Automatically fix data quality issues

**Steps:**
1. View current status (missing values, duplicates)
2. Click **"🧠 Let AI Clean This Dataset"**
3. Review AI's cleaning plan:
   - What problems exist
   - How AI will fix each
   - Expected impact
4. Click **"✅ Apply This Cleaning Plan"**
5. Review cleaning log
6. Download cleaned data (optional)
7. Click **"➡️ Continue to Visualization"**

**What AI fixes:**
- Missing values (fills intelligently)
- Duplicate rows (removes)
- Wrong data types (converts)
- Useless columns (drops if 90%+ empty)

---

### 📊 Page 3: Charts & Analysis

**Purpose:** Generate professional dashboard with insights

**Steps:**
1. Click **"🧠 Let AI Create Dashboard & Insights"**
2. Review executive summary
3. View KPIs (3-5 key metrics)
4. Explore visualizations (5-7 charts)
5. Read business insights for each chart
6. Download reports (optional)
7. Click **"➡️ Continue to Machine Learning"**

**What you get:**
- Executive summary
- Key performance indicators
- Multiple chart types (bar, line, scatter, pie, histogram)
- Business insights
- Downloadable reports

---

### 🎯 Page 4: Machine Learning

**Purpose:** Build prediction models automatically

**Steps:**
1. Select what to predict (target variable)
2. View target statistics
3. Click **"🧠 Let AI Build Prediction Model"**
4. Review AI's strategy:
   - Problem type (Classification/Regression)
   - Model selected
   - Features chosen
   - Reasoning
   - Expected accuracy
5. Click **"🚀 Train Model Now"**
6. View comprehensive results:
   
   **Classification:**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - Error distribution
   - Performance interpretation
   
   **Regression:**
   - R², RMSE, MAE, MAPE
   - Actual vs Predicted plot
   - Residual plot
   - Error distribution
   - Quality breakdown

7. View feature importance
8. Download performance report & predictions
9. Click **"➡️ Continue to Chat"**

**What AI does:**
- Detects problem type automatically
- Selects best algorithm
- Chooses predictive features
- Trains and evaluates model
- Explains results in plain English

---

### 💬 Page 5: Chat with Data

**Purpose:** Ask questions about your data

**Steps:**
1. View suggested questions OR type your own
2. Ask questions like:
   - "What are the key insights?"
   - "Show me average sales by region"
   - "Are there any correlations?"
   - "What should I focus on?"
3. Get answers with:
   - Business explanations
   - Python code (if relevant)
   - Visualizations
4. Click **"▶️ Run This Code"** to execute
5. Continue conversation
6. Click **"🔄 Start New Analysis"** for new dataset

**Examples:**
- "Summarize this dataset"
- "What trends do you see?"
- "Which columns are most important?"
- "Create a chart showing X vs Y"

---

## 🔧 Troubleshooting

### "Ollama is not running"
```bash
ollama serve
```

### "Model not found"
```bash
ollama pull qwen2.5:7b
ollama list
```

### "Module not found"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Port 8501 in use"
```bash
streamlit run Home.py --server.port 8502
```

### "AI is slow"
- Close other applications
- First response is always slower
- Consider smaller model for 8GB RAM systems

### "ML page stuck at Step 1"
Make sure you have the updated version with `st.rerun()` fix.

### "Charts not showing"
- Refresh browser
- Clear cache
- Try Chrome

---

## ❓ FAQs

**Q: Do I need internet?**  
A: Only for initial setup. After that, 100% offline.

**Q: Is my data safe?**  
A: Yes! Everything runs locally. No data leaves your computer.

**Q: What file formats?**  
A: CSV, Excel (.xlsx), JSON

**Q: How large can datasets be?**  
A: Recommended: <100K rows, <100 columns, <50MB

**Q: How accurate are predictions?**  
A: 60-95% depending on data quality. AI shows you the accuracy.

**Q: Can I export results?**  
A: Yes! Downloads available for cleaned data, reports, predictions.

**Q: Can I change AI model?**  
A: Yes! Edit `ai_local.py` and change `MODEL_NAME`.

**Q: Works on M1/M2 Mac?**  
A: Yes! Optimized for Apple Silicon.

**Q: Commercial use?**  
A: Yes! Free to use for business.

---

## 📞 Support

- **Documentation:** See `/mnt/user-data/outputs/` folder
- **Ollama Docs:** https://ollama.ai/docs
- **Issues:** GitHub Issues page

---

## ⭐ Quick Reference

```bash
# 1. Start Ollama
ollama serve

# 2. Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Start NovaLens
streamlit run Home.py

# 4. Open browser
http://localhost:8501
```

---

## 🎯 Workflow Summary

1. **Upload** → AI analyzes dataset
2. **Clean** → AI fixes data issues
3. **Visualize** → AI creates dashboard
4. **Predict** → AI builds ML model
5. **Chat** → Ask anything about your data

**Total time:** 5-10 minutes from upload to predictions!

---

## 🔑 Key Features

✅ **100% Local** - No internet needed  
✅ **100% Private** - Data never leaves your computer  
✅ **100% Automatic** - AI does everything  
✅ **0% Coding** - Just click buttons  
✅ **Professional Output** - Dashboard & reports  

---

**🎉 Ready to analyze data like a pro! 🎉**

*NovaLens AI v2.0 - AI-Powered Edition*
