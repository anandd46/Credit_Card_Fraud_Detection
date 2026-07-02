# 💳 Credit Card Fraud Detection using Machine Learning

A production-quality, end-to-end machine learning project that detects fraudulent credit card transactions in highly imbalanced data — combining a **supervised XGBoost classifier** with an **unsupervised Isolation Forest** anomaly detector, deployed through an interactive **Streamlit** web application.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📖 Project Description

Fraudulent transactions make up a tiny fraction of all credit card activity, which makes fraud detection a textbook example of a **severe class imbalance problem**. A naive model can score 99%+ accuracy by simply predicting "legitimate" every time — while catching zero fraud.

This project builds a complete pipeline that:

- Cleans and prepares real-world imbalanced transaction data
- Balances the training data using **SMOTE**
- Trains and compares **two fundamentally different modeling approaches**: a supervised gradient-boosted classifier (XGBoost) and an unsupervised anomaly detector (Isolation Forest)
- Evaluates both models with metrics that actually matter for fraud detection (Precision, Recall, F1, ROC-AUC — not just Accuracy)
- Ships a polished Streamlit interface for uploading new transactions and getting instant fraud predictions

---

## ✨ Features

- 🔍 Automatic dataset detection with friendly setup instructions if missing
- 🧹 Full preprocessing pipeline: missing values, duplicates, scaling, stratified splitting
- ⚖️ SMOTE-based class imbalance handling
- 🤖 Two trained models: **XGBoost Classifier** and **Isolation Forest**
- 📊 8 auto-generated, publication-quality visualizations
- 🌐 Multi-page Streamlit web app with CSV upload & downloadable predictions
- 💾 Model persistence via Joblib — train once, reuse instantly
- 🧾 Fully typed, documented, PEP8-compliant Python code

---

## 🛠️ Technology Stack

| Category | Tools |
|---|---|
| Language | Python 3.11+ |
| Data Handling | pandas, numpy |
| Machine Learning | scikit-learn, xgboost |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Visualization | matplotlib, seaborn |
| Model Persistence | joblib |
| Web App | Streamlit |

---

## 🧠 Machine Learning Algorithms

### 1. XGBoost Classifier (Supervised)
A gradient-boosted decision tree ensemble trained on SMOTE-balanced data. Chosen for its strong performance on tabular data, built-in feature importance, and speed.

### 2. Isolation Forest (Unsupervised)
An anomaly-detection model trained on the *original, unbalanced* distribution. It isolates anomalies (fraud) by exploiting the fact that they require fewer random partitions to separate from the bulk of "normal" data — no labels required.

---

## 📂 Dataset

This project uses the **[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** from Kaggle — 284,807 anonymized European card transactions (PCA-transformed features `V1`–`V28`, plus `Time`, `Amount`, and `Class`).

> ⚠️ The dataset is **not included** in this repository due to size and licensing. See [GUIDE.md](GUIDE.md) for exact download and setup instructions. The application automatically detects whether `dataset.csv` exists and will tell you exactly what to do if it's missing.

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
# Download from Kaggle, rename to dataset.csv, place in the project root
```

Full beginner-friendly walkthrough: see **[GUIDE.md](GUIDE.md)**.

---

## ▶️ Usage

**Train both models:**
```bash
python train.py
```

**Run predictions from the command line:**
```bash
python predict.py path/to/transactions.csv
```

**Launch the web app:**
```bash
streamlit run app.py
```

---

## 🖼️ Screenshots

> _Add screenshots after running the app locally — recommended pages: Home, Upload & Predict, Performance Dashboard._

| Home | Upload & Predict | Performance Dashboard |
|---|---|---|
| `screenshots/home.png` | `screenshots/predict.png` | `screenshots/dashboard.png` |

---

## 📏 Evaluation Metrics

Both models are evaluated on an identical held-out test set using:

- **Accuracy** — overall correctness (least informative for imbalanced data)
- **Precision** — of all flagged transactions, how many were truly fraud
- **Recall** — of all actual fraud, how many were caught
- **F1 Score** — harmonic mean of precision and recall
- **ROC-AUC** — overall separability between classes
- **Confusion Matrix** — full breakdown of TP / TN / FP / FN
- **Classification Report** — per-class precision/recall/F1

Results and charts are generated automatically in the project root after running `train.py`.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                  # Streamlit web application
├── train.py                # Model training pipeline
├── predict.py               # Prediction logic (CLI + app)
├── preprocess.py            # Data loading & preprocessing
├── utils.py                 # Shared constants & helpers
├── visualize.py              # Chart generation
├── requirements.txt
├── README.md
├── GUIDE.md
├── LICENSE
├── .gitignore
├── dataset.csv              # (not included — see GUIDE.md)
├── xgb_model.pkl             # generated after training
├── isolation_model.pkl       # generated after training
├── scaler.pkl                # generated after training
└── *.png                     # generated charts after training
```

No subfolders — everything lives in the project root by design.

---

## 🚀 Future Improvements

- Add deep learning baselines (autoencoders) for anomaly detection
- Hyperparameter tuning via Optuna/GridSearchCV
- Real-time streaming prediction (Kafka integration)
- Model explainability with SHAP values
- Dockerize the application for consistent deployment
- Add automated CI/CD testing pipeline

---

## 👤 Author

**Anand D**
MCA (AI & Data Science), Amrita Vishwa Vidyapeetham

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Kaggle — Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Machine Learning Group, ULB)
- [scikit-learn](https://scikit-learn.org/) documentation
- [XGBoost](https://xgboost.readthedocs.io/) documentation
- [imbalanced-learn](https://imbalanced-learn.org/) documentation
- [Streamlit](https://streamlit.io/) documentation
