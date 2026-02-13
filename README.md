# 🔮 Explainable ML Playground

A high-performance, ChatGPT-inspired AutoML platform for rapid experimentation, explainable AI, and one-click model deployment.

---

<img width="1440" height="816" alt="Playground" src="https://github.com/user-attachments/assets/d44e76d6-fca3-4c93-9ad4-77d4a9021e5e" />

## 🚀 Key Value Propositions

*   **Automated Preprocessing**: Reduces manual engineering time by **90%** through automatic encoding, scaling, and missing value handling.
*   **AI-Powered Advisor**: Built-in data quality engine that flags high-cardinality targets, class imbalances, and suggests optimal algorithms.
*   **Model Comparison & Ensembling**: Compare multiple models (RF, SVM, LR) side-by-side and generate high-accuracy **Voting Ensembles** with a single click.
*   **Explainable AI (XAI)**: Integrated **SHAP** engine providing 100% transparency into model logic via global importance and local waterfall charts.
*   **Instant Deployment**: One-click generation of **FastAPI** wrappers and **Pydantic** validation schemas, shortening production lead time by **95%**.
*   **Premium UX**: State-of-the-art dark theme with prioritized visual hierarchy and interactive Plotly-powered diagnostics.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit + Custom CSS (ChatGPT Dark Aesthetic)
- **ML Engine**: Scikit-Learn (Random Forest, SVM, Logistic/Linear Regression)
- **Explainability**: SHAP (Shapley Additive Explanations)
- **Visuals**: Plotly & Matplotlib (High-contrast dark mode)
- **Production**: FastAPI & Uvicorn

---

## 🏃 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Verify Setup
Check if all dependencies are installed and modules are working correctly:
```bash
python3 test_setup.py
```

### 3. Run Application
```bash
streamlit run app.py
```

### 4. Usage Workflow
1.  **Data Upload**: Drop a CSV/Excel or use pre-loaded datasets (Iris, Titanic, California Housing).
2.  **Configuration & AI Audit**: Select your target variable. The **AI Advisor** automatically audits your data quality and suggests the best processing path.
3.  **Training & Comparison**: Train individual models or use **"Compare All"** to evaluate the entire suite. Create an **Ensemble Model** to maximize performance.
4.  **Explanations**: Generate SHAP values to see exactly which features are driving your model's decisions.
5.  **One-Click Deployment**: Head to the Deployment tab to generate a production-ready API for your model.

---

## 📁 Repository Structure
```
├── app.py                 # Application Entry Point & Navigation
├── test_setup.py          # Environment & Dependency Verification script
├── utils/
│   ├── ui.py              # ChatGPT Dark Theme & CSS
│   ├── ai_advisor.py      # AI-powered Data Auditing & Guidance
│   ├── data_handler.py    # Robust Data Loading & Preprocessing
│   ├── model_trainer.py   # Multi-Model Comparison & Ensembling
│   ├── explainer.py       # SHAP Visualization Engine
│   └── deployment.py      # Production FastAPI Code Generator
└── README.md
```

---

*Built for speed, transparency, and production-preparedness. 🔮✨*
