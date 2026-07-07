

"""
predict.py
----------
Loads the saved, trained models (XGBoost and Isolation Forest) and the
fitted scaler, and runs fraud predictions on new transaction data.

Used both as a standalone script (predict on a CSV given via command
line) and as a module imported by app.py (the Streamlit interface).

Usage (standalone):
    python predict.py path/to/new_transactions.csv

Author: Anand D
"""

from __future__ import annotations

import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import (
    ISOLATION_MODEL_FILENAME,
    SCALER_FILENAME,
    TARGET_COLUMN,
    XGB_MODEL_FILENAME,
    all_models_exist,
    load_model,
    log,
)


def load_trained_models() -> Tuple[XGBClassifier, IsolationForest, StandardScaler]:
    """
    Load the previously trained XGBoost model, Isolation Forest model,
    and fitted StandardScaler from disk.

    Returns:
        A tuple (xgb_model, isolation_model, scaler).

    Raises:
        FileNotFoundError: If any of the required model files are
            missing, with guidance to run train.py first.
    """
    if not all_models_exist():
        raise FileNotFoundError(
            "One or more trained model files are missing. Please run "
            "'python train.py' first to train and save the models before "
            "predicting."
        )
    xgb_model = load_model(XGB_MODEL_FILENAME)
    iso_model = load_model(ISOLATION_MODEL_FILENAME)
    scaler = load_model(SCALER_FILENAME)
    return xgb_model, iso_model, scaler


def _prepare_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Prepare a raw uploaded transaction DataFrame for prediction: drop the
    target column if present, and scale 'Time'/'Amount' using the
    ALREADY-FITTED scaler (never re-fit on new data, which would leak
    information and produce inconsistent scaling versus training).

    Args:
        df: Raw input DataFrame (may or may not include the target column).
        scaler: The scaler fitted during training.

    Returns:
        A DataFrame of features ready to feed into the models.
    """
    features = df.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    columns_to_scale = [col for col in ["Time", "Amount"] if col in features.columns]
    if columns_to_scale:
        features[columns_to_scale] = scaler.transform(features[columns_to_scale])

    return features


def predict_fraud(
    df: pd.DataFrame,
    xgb_model: XGBClassifier,
    iso_model: IsolationForest,
    scaler: StandardScaler,
    model_choice: str = "XGBoost",
) -> pd.DataFrame:
    """
    Predict which transactions in the given DataFrame are fraudulent.

    Args:
        df: Raw uploaded transaction data.
        xgb_model: The trained XGBoost classifier.
        iso_model: The trained Isolation Forest model.
        scaler: The fitted StandardScaler from training.
        model_choice: Either "XGBoost" or "Isolation Forest".

    Returns:
        The original DataFrame with two extra columns appended:
            - 'Prediction': 0 (legitimate) or 1 (fraud)
            - 'Fraud_Probability': model confidence score (0-1) that the
              transaction is fraudulent.
    """
    features = _prepare_features(df, scaler)
    result = df.copy()

    if model_choice == "XGBoost":
        preds = xgb_model.predict(features)
        scores = xgb_model.predict_proba(features)[:, 1]
    elif model_choice == "Isolation Forest":
        raw_preds = iso_model.predict(features)  # 1 = normal, -1 = anomaly
        preds = np.where(raw_preds == -1, 1, 0)
        raw_scores = -iso_model.decision_function(features)
        # Normalize anomaly scores to a 0-1 range for a consistent,
        # human-readable "probability-like" display in the UI.
        min_s, max_s = raw_scores.min(), raw_scores.max()
        scores = (raw_scores - min_s) / (max_s - min_s + 1e-9)
    else:
        raise ValueError(
            f"Unknown model_choice '{model_choice}'. "
            "Expected 'XGBoost' or 'Isolation Forest'."
        )

    result["Prediction"] = preds
    result["Fraud_Probability"] = np.round(scores, 4)
    return result


def summarize_predictions(result_df: pd.DataFrame) -> dict:
    """
    Summarize prediction results: counts and percentage of fraud vs
    legitimate transactions.

    Args:
        result_df: The DataFrame returned by predict_fraud().

    Returns:
        A dictionary with 'total', 'fraud_count', 'legit_count', and
        'fraud_percentage'.
    """
    total = len(result_df)
    fraud_count = int((result_df["Prediction"] == 1).sum())
    legit_count = total - fraud_count
    fraud_percentage = (fraud_count / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_percentage": round(fraud_percentage, 4),
    }


def main() -> None:
    """Command-line entry point: predict fraud on a CSV file path argument."""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    log(f"Loading transactions from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    xgb_model, iso_model, scaler = load_trained_models()

    result = predict_fraud(df, xgb_model, iso_model, scaler, model_choice="XGBoost")
    summary = summarize_predictions(result)

    log(f"Total transactions : {summary['total']}")
    log(f"Fraudulent         : {summary['fraud_count']}")
    log(f"Legitimate         : {summary['legit_count']}")
    log(f"Fraud percentage   : {summary['fraud_percentage']}%")

    output_path = "prediction_results.csv"
    result.to_csv(output_path, index=False)
    log(f"Predictions saved to '{output_path}'.")


if __name__ == "__main__":
    main()
