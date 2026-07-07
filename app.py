
"""
app.py
------
Streamlit web application for the Credit Card Fraud Detection project.

Provides a clean, modern, multi-page interface covering:
    Home | Project Overview | Dataset Information | Model Selection |
    Upload & Predict | Performance Dashboard | Charts | About Project

Run with:
    streamlit run app.py

Author: Anand D
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from predict import load_trained_models, predict_fraud, summarize_predictions
from utils import (
    RANDOM_STATE,
    TARGET_COLUMN,
    all_models_exist,
    dataset_exists,
    get_dataset_path,
)

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #264653;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F7F9FB;
        border: 1px solid #E3E8EC;
        border-radius: 10px;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        color: #264653;
    }
    .fraud-badge {
        background-color: #E63946;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .legit-badge {
        background-color: #2A9D8F;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.markdown("## 💳 Fraud Detection")
st.sidebar.caption("ML-powered transaction analysis")

PAGES = [
    "🏠 Home",
    "📋 Project Overview",
    "🗂️ Dataset Information",
    "🤖 Model Selection",
    "📤 Upload & Predict",
    "📊 Performance Dashboard",
    "📈 Charts",
    "ℹ️ About Project",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
if all_models_exist():
    st.sidebar.success("Models loaded and ready.")
else:
    st.sidebar.warning("Models not trained yet. Run `python train.py` first.")

if dataset_exists():
    st.sidebar.success("Dataset detected.")
else:
    st.sidebar.info("Dataset not found (optional for prediction-only use).")


# --------------------------------------------------------------------------
# Helper: cached model loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    """Load and cache the trained models so they aren't reloaded on every
    Streamlit interaction (which would be slow)."""
    return load_trained_models()


# --------------------------------------------------------------------------
# Page: Home
# --------------------------------------------------------------------------

if page == "🏠 Home":
    st.markdown('<div class="main-header">💳 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">A machine learning system that detects fraudulent '
        'transactions using supervised and unsupervised learning.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Purpose")
        st.write(
            "Identify fraudulent credit card transactions in highly "
            "imbalanced data, where fraud represents a tiny fraction of "
            "all transactions."
        )
    with col2:
        st.markdown("### 🧠 Models")
        st.write(
            "**XGBoost Classifier** (supervised) and **Isolation Forest** "
            "(unsupervised anomaly detection) are trained and compared "
            "side by side."
        )
    with col3:
        st.markdown("### ⚖️ Imbalance Handling")
        st.write(
            "**SMOTE** oversampling is applied to the training data so "
            "the model learns meaningful fraud patterns instead of "
            "ignoring the minority class."
        )

    st.markdown("---")
    st.markdown("### Get Started")
    st.write(
        "Use the sidebar to explore the dataset, review model performance, "
        "or upload your own transaction data on the **Upload & Predict** page."
    )

# --------------------------------------------------------------------------
# Page: Project Overview
# --------------------------------------------------------------------------

elif page == "📋 Project Overview":
    st.markdown('<div class="main-header">📋 Project Overview</div>', unsafe_allow_html=True)

    st.markdown("""
This project builds an end-to-end machine learning pipeline to detect
fraudulent credit card transactions.

**Pipeline stages:**
1. **Data Loading & Cleaning** — missing value checks, duplicate removal.
2. **Feature Scaling** — standardizing the `Time` and `Amount` columns.
3. **Class Imbalance Handling** — SMOTE oversampling on the training split.
4. **Model Training** — an XGBoost Classifier (supervised) and an
   Isolation Forest (unsupervised anomaly detector).
5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion
   matrices, and full classification reports.
6. **Visualization** — 8 professional charts covering distribution,
   correlation, model performance, and feature importance.
7. **Deployment** — this Streamlit application for interactive predictions.
    """)

    st.markdown("### Tech Stack")
    tech_cols = st.columns(4)
    techs = ["Python 3.11+", "scikit-learn", "XGBoost", "imbalanced-learn",
             "pandas / numpy", "matplotlib", "joblib", "Streamlit"]
    for i, tech in enumerate(techs):
        tech_cols[i % 4].info(tech)

# --------------------------------------------------------------------------
# Page: Dataset Information
# --------------------------------------------------------------------------

elif page == "🗂️ Dataset Information":
    st.markdown('<div class="main-header">🗂️ Dataset Information</div>', unsafe_allow_html=True)

    if not dataset_exists():
        st.error(
            f"Dataset file `{get_dataset_path()}` was not found in the project "
            "root.\n\nDownload the **Credit Card Fraud Detection** dataset from "
            "Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), "
            "rename it to `dataset.csv`, and place it in this project's root "
            "folder."
        )
    else:
        df = pd.read_csv(get_dataset_path())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Features", df.shape[1] - 1)
        fraud_count = int((df[TARGET_COLUMN] == 1).sum()) if TARGET_COLUMN in df.columns else 0
        col3.metric("Fraudulent", f"{fraud_count:,}")
        fraud_pct = (fraud_count / len(df) * 100) if len(df) > 0 else 0
        col4.metric("Fraud Rate", f"{fraud_pct:.4f}%")

        st.markdown("### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Column Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success("No missing values found in the dataset.")
        else:
            st.dataframe(missing[missing > 0], use_container_width=True)

# --------------------------------------------------------------------------
# Page: Model Selection
# --------------------------------------------------------------------------

elif page == "🤖 Model Selection":
    st.markdown('<div class="main-header">🤖 Model Selection</div>', unsafe_allow_html=True)
    st.write("Choose which model to use for predictions on this page. Your "
             "choice will carry over to the **Upload & Predict** page.")

    model_choice = st.radio(
        "Select a model:",
        ["XGBoost", "Isolation Forest"],
        index=0 if st.session_state.get("model_choice", "XGBoost") == "XGBoost" else 1,
        horizontal=True,
    )
    st.session_state["model_choice"] = model_choice

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌲 XGBoost Classifier")
        st.write(
            "- **Type:** Supervised (learns from labeled fraud examples)\n"
            "- **Strength:** Very high precision/recall when labeled data "
            "is available\n"
            "- **Trained on:** SMOTE-balanced data\n"
        )
    with col2:
        st.markdown("#### 🌳 Isolation Forest")
        st.write(
            "- **Type:** Unsupervised (detects anomalies without labels)\n"
            "- **Strength:** Can catch novel fraud patterns not seen "
            "during training\n"
            "- **Trained on:** Original (unbalanced) data distribution\n"
        )

    st.success(f"Currently selected model: **{model_choice}**")

# --------------------------------------------------------------------------
# Page: Upload & Predict
# --------------------------------------------------------------------------

elif page == "📤 Upload & Predict":
    st.markdown('<div class="main-header">📤 Upload CSV & Predict Fraud</div>', unsafe_allow_html=True)

    if not all_models_exist():
        st.error(
            "No trained models found. Please run `python train.py` first "
            "to train and save the models."
        )
    else:
        model_choice = st.session_state.get("model_choice", "XGBoost")
        st.info(f"Using model: **{model_choice}** (change this on the Model Selection page)")

        uploaded_file = st.file_uploader(
            "Upload a CSV file of transactions to check for fraud", type=["csv"]
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.markdown("#### Preview of Uploaded Data")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("🔍 Predict Fraud", type="primary"):
                with st.spinner("Running predictions..."):
                    xgb_model, iso_model, scaler = get_models()
                    result = predict_fraud(
                        df, xgb_model, iso_model, scaler, model_choice=model_choice
                    )
                    summary = summarize_predictions(result)

                st.markdown("### Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Transactions", f"{summary['total']:,}")
                col2.metric("Fraudulent Detected", f"{summary['fraud_count']:,}")
                col3.metric("Fraud Percentage", f"{summary['fraud_percentage']}%")

                st.markdown("#### Prediction Details")
                st.dataframe(result, use_container_width=True)

                csv_bytes = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Prediction Results as CSV",
                    data=csv_bytes,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )
        else:
            st.caption(
                "Tip: your CSV should contain the same feature columns the "
                "model was trained on (Time, V1-V28, Amount). The target "
                "column, if present, will be ignored automatically."
            )

# --------------------------------------------------------------------------
# Page: Performance Dashboard
# --------------------------------------------------------------------------

elif page == "📊 Performance Dashboard":
    st.markdown('<div class="main-header">📊 Performance Dashboard</div>', unsafe_allow_html=True)

    comparison_path = "model_comparison.png"
    if os.path.isfile(comparison_path):
        st.image(comparison_path, caption="Model Comparison Across Metrics", use_container_width=True)
    else:
        st.warning(
            "Performance charts not found. Run `python train.py` to generate "
            "evaluation charts."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### XGBoost")
        for fname, caption in [
            ("confusion_matrix_xgboost.png", "Confusion Matrix"),
            ("roc_curve_xgboost.png", "ROC Curve"),
            ("precision_recall_curve_xgboost.png", "Precision-Recall Curve"),
        ]:
            if os.path.isfile(fname):
                st.image(fname, caption=caption, use_container_width=True)

    with col2:
        st.markdown("#### Isolation Forest")
        for fname, caption in [
            ("confusion_matrix_isolation_forest.png", "Confusion Matrix"),
            ("roc_curve_isolation_forest.png", "ROC Curve"),
            ("precision_recall_curve_isolation_forest.png", "Precision-Recall Curve"),
        ]:
            if os.path.isfile(fname):
                st.image(fname, caption=caption, use_container_width=True)

# --------------------------------------------------------------------------
# Page: Charts
# --------------------------------------------------------------------------

elif page == "📈 Charts":
    st.markdown('<div class="main-header">📈 Data Visualizations</div>', unsafe_allow_html=True)

    charts = [
        ("class_distribution.png", "Class Distribution"),
        ("correlation_heatmap.png", "Correlation Heatmap"),
        ("fraud_vs_normal_distribution.png", "Fraud vs Normal Distribution"),
        ("feature_importance_xgboost.png", "Feature Importance (XGBoost)"),
    ]

    any_found = False
    for fname, caption in charts:
        if os.path.isfile(fname):
            any_found = True
            st.image(fname, caption=caption, use_container_width=True)
            st.markdown("---")

    if not any_found:
        st.warning("No charts found yet. Run `python train.py` to generate them.")

# --------------------------------------------------------------------------
# Page: About Project
# --------------------------------------------------------------------------

elif page == "ℹ️ About Project":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
**Project:** Credit Card Fraud Detection using Machine Learning

**Author:** Anand D

**Purpose:** Academic / portfolio project demonstrating an end-to-end
supervised + unsupervised machine learning pipeline for financial fraud
detection, including imbalanced-data handling, model evaluation, and
deployment via Streamlit.

**Random State:** `{RANDOM_STATE}` (used throughout for reproducibility)

**Dataset:** [Credit Card Fraud Detection — Kaggle]
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Disclaimer:** This project is for educational purposes. It should not
be used as-is in a production financial system without further testing,
monitoring, and regulatory review.
    """)
