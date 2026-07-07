

"""
train.py
--------
Main training script for the Credit Card Fraud Detection project.

Runs the full pipeline:
    1. Load & preprocess the dataset
    2. Balance the training data with SMOTE
    3. Train an XGBoost Classifier (supervised)
    4. Train an Isolation Forest (unsupervised anomaly detection)
    5. Evaluate both models on the same held-out test set
    6. Generate all visualizations
    7. Save both trained models (+ the fitted scaler) to disk

Usage:
    python train.py

Author: Anand D
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from preprocess import run_full_preprocessing
from utils import (
    ISOLATION_MODEL_FILENAME,
    RANDOM_STATE,
    SCALER_FILENAME,
    XGB_MODEL_FILENAME,
    log,
    save_model,
    timeit,
)
from visualize import generate_all_visualizations


# --------------------------------------------------------------------------
# Model training functions
# --------------------------------------------------------------------------

@timeit
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Train an XGBoost Classifier on the SMOTE-balanced training data.

    XGBoost is a gradient-boosted tree ensemble that builds decision
    trees sequentially, with each new tree correcting the errors of the
    previous ones. It is chosen here because it:
        * Handles complex non-linear feature interactions well.
        * Is fast to train even on large, high-dimensional data.
        * Provides built-in feature importance scores.
        * Consistently performs strongly on tabular fraud datasets in
          both industry and Kaggle competitions.

    Args:
        X_train: Balanced training features.
        y_train: Balanced training labels.

    Returns:
        The fitted XGBClassifier.
    """
    log("Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


@timeit
def train_isolation_forest(X_train: pd.DataFrame, contamination: float) -> IsolationForest:
    """
    Train an Isolation Forest for unsupervised anomaly detection.

    Isolation Forest works by randomly partitioning the feature space
    with decision trees; anomalies (fraud) are, on average, isolated
    into their own leaf node in far fewer splits than normal points,
    because they are "few and different". This makes it well suited to
    fraud detection scenarios where labels may be scarce or unreliable,
    since it does not require labeled fraud examples to train.

    Note: Isolation Forest is trained WITHOUT SMOTE and without labels
    (unsupervised), using only the original (unbalanced) training
    features, which is the correct and realistic way to use this model.

    Args:
        X_train: Training features (unbalanced, original distribution).
        contamination: Expected proportion of anomalies (fraud) in the
            data, used to set the decision threshold.

    Returns:
        The fitted IsolationForest.
    """
    log("Training Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate_model(
    y_true: pd.Series, y_pred: np.ndarray, y_scores: np.ndarray, model_name: str
) -> Dict[str, object]:
    """
    Compute and display the full evaluation suite for a model's
    predictions: accuracy, precision, recall, F1, ROC-AUC, confusion
    matrix, and the full classification report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels (0/1).
        y_scores: Predicted probabilities/scores for the positive class,
            used for ROC-AUC.
        model_name: Human-readable model name for logging.

    Returns:
        A dictionary containing all computed values, ready to be passed
        into the visualization module.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"])

    log(f"\n===== {model_name} Evaluation =====")
    log(f"Accuracy : {accuracy:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall   : {recall:.4f}")
    log(f"F1 Score : {f1:.4f}")
    log(f"ROC AUC  : {roc_auc:.4f}")
    log(f"Confusion Matrix:\n{cm}")
    log(f"Classification Report:\n{report}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def main() -> None:
    """Run the full training pipeline end to end."""
    log("=" * 70)
    log("CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE")
    log("=" * 70)

    # 1. Preprocessing (load, clean, scale, split, SMOTE)
    (
        X_train_res,
        X_test,
        y_train_res,
        y_test,
        X_train_original,
        y_train_original,
        df_scaled,
        scaler,
    ) = run_full_preprocessing()

    # 2. Train XGBoost on the SMOTE-balanced data
    xgb_model, xgb_train_time = train_xgboost(X_train_res, y_train_res)

    # 3. Train Isolation Forest on the ORIGINAL (unbalanced) training data.
    #    Isolation Forest is unsupervised, so it should learn the true
    #    anomaly rate of real transactions rather than the artificially
    #    balanced 50/50 distribution produced by SMOTE.
    true_fraud_rate = y_train_original.mean()
    contamination = float(np.clip(true_fraud_rate, 0.001, 0.5))
    iso_model, iso_train_time = train_isolation_forest(
        X_train_original, contamination=contamination
    )

    # 4. Predict + time predictions on the test set
    import time as _time

    start = _time.perf_counter()
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_time = _time.perf_counter() - start
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    log(f"XGBoost prediction time: {xgb_pred_time:.4f} seconds")

    start = _time.perf_counter()
    iso_raw_pred = iso_model.predict(X_test)  # returns 1 (normal) / -1 (anomaly)
    iso_pred_time = _time.perf_counter() - start
    # Convert Isolation Forest output to match our 0=legit, 1=fraud convention.
    iso_pred = np.where(iso_raw_pred == -1, 1, 0)
    # decision_function: lower (more negative) score = more anomalous.
    # We invert it so higher score = more likely fraud, matching XGBoost's
    # score convention for ROC/PR curve plotting.
    iso_scores = -iso_model.decision_function(X_test)
    log(f"Isolation Forest prediction time: {iso_pred_time:.4f} seconds")

    # 5. Evaluate both models
    xgb_results = evaluate_model(y_test, xgb_pred, xgb_scores, "XGBoost")
    iso_results = evaluate_model(y_test, iso_pred, iso_scores, "Isolation Forest")

    xgb_results["feature_importances"] = xgb_model.feature_importances_
    xgb_results["feature_names"] = list(X_train_res.columns)

    # 6. Generate all visualizations
    generate_all_visualizations(
        df_scaled,
        {"XGBoost": xgb_results, "Isolation Forest": iso_results},
    )

    # 7. Save models + scaler
    log("Saving Models...")
    save_model(xgb_model, XGB_MODEL_FILENAME)
    save_model(iso_model, ISOLATION_MODEL_FILENAME)
    save_model(scaler, SCALER_FILENAME)

    log("=" * 70)
    log("Done.")
    log("=" * 70)
    log(f"XGBoost training time : {xgb_train_time:.4f}s | prediction time: {xgb_pred_time:.4f}s")
    log(f"Isolation Forest training time: {iso_train_time:.4f}s | prediction time: {iso_pred_time:.4f}s")


if __name__ == "__main__":
    main()
