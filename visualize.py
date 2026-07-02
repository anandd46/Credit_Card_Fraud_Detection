"""
visualize.py
------------
Generates all professional matplotlib visualizations required for the
Credit Card Fraud Detection project and saves them automatically to the
project root as PNG files.

Author: Anand D
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so this works headless too
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from utils import IMAGE_DIR, TARGET_COLUMN, log

# Consistent, professional color palette used across every chart.
PALETTE = {
    "legit": "#2E86AB",
    "fraud": "#E63946",
    "primary": "#264653",
    "secondary": "#2A9D8F",
    "accent": "#E9C46A",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    }
)


def _save(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to disk and close it to free memory."""
    path = f"{IMAGE_DIR}/{filename}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved chart: {path}")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """
    Bar chart showing the raw count of legitimate vs fraudulent
    transactions, illustrating the severe class imbalance.
    """
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraudulent (1)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=[PALETTE["legit"], PALETTE["fraud"]])
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Class Distribution: Legitimate vs Fraudulent Transactions",
                 fontweight="bold")
    ax.set_ylabel("Number of Transactions")
    ax.set_yscale("log")
    _save(fig, "class_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of feature correlations across the dataset."""
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    _save(fig, "correlation_heatmap.png")


def plot_confusion_matrix_chart(
    y_true: Sequence[int], y_pred: Sequence[int], model_name: str
) -> None:
    """Confusion matrix heatmap for a given model's predictions."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Legitimate", "Fraud"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {model_name}", fontweight="bold")
    _save(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_roc_curve_chart(
    y_true: Sequence[int], y_scores: Sequence[float], model_name: str
) -> float:
    """
    Plot the ROC curve for a model and return the AUC score.

    Returns:
        The area under the ROC curve.
    """
    from sklearn.metrics import auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["secondary"], lw=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}", fontweight="bold")
    ax.legend(loc="lower right")
    _save(fig, f"roc_curve_{model_name.lower().replace(' ', '_')}.png")
    return roc_auc


def plot_precision_recall_curve_chart(
    y_true: Sequence[int], y_scores: Sequence[float], model_name: str
) -> None:
    """Plot the Precision-Recall curve, which is especially informative
    for highly imbalanced fraud datasets (more so than ROC alone)."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE["accent"], lw=2, label=model_name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontweight="bold")
    ax.legend(loc="lower left")
    _save(fig, f"precision_recall_curve_{model_name.lower().replace(' ', '_')}.png")


def plot_feature_importance(
    feature_names: Sequence[str], importances: Sequence[float], top_n: int = 15
) -> None:
    """Horizontal bar chart of the top-N most important features from XGBoost."""
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order][::-1]
    top_importances = [importances[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top_features, top_importances, color=PALETTE["primary"])
    ax.set_title(f"Top {top_n} Feature Importances - XGBoost", fontweight="bold")
    ax.set_xlabel("Importance Score")
    _save(fig, "feature_importance_xgboost.png")


def plot_fraud_vs_normal_distribution(df: pd.DataFrame) -> None:
    """
    Overlaid histogram comparing the transaction 'Amount' distribution
    for fraudulent vs legitimate transactions.
    """
    if "Amount" not in df.columns:
        log("Column 'Amount' not present; skipping fraud-vs-normal chart.")
        return

    fraud = df[df[TARGET_COLUMN] == 1]["Amount"]
    legit = df[df[TARGET_COLUMN] == 0]["Amount"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(legit, bins=50, color=PALETTE["legit"], alpha=0.8)
    axes[0].set_title("Legitimate Transaction Amounts", fontweight="bold")
    axes[0].set_xlabel("Amount (scaled)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(fraud, bins=50, color=PALETTE["fraud"], alpha=0.8)
    axes[1].set_title("Fraudulent Transaction Amounts", fontweight="bold")
    axes[1].set_xlabel("Amount (scaled)")
    axes[1].set_ylabel("Frequency")

    fig.suptitle("Fraud vs Normal Transaction Amount Distribution", fontweight="bold")
    _save(fig, "fraud_vs_normal_distribution.png")


def plot_model_comparison(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1, and
    ROC-AUC across all trained models.

    Args:
        metrics: Dict mapping model_name -> dict of metric_name -> value.
            Example:
                {
                    "XGBoost": {"Accuracy": 0.999, "Precision": 0.95, ...},
                    "Isolation Forest": {...},
                }
    """
    model_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())

    x = np.arange(len(metric_names))
    width = 0.8 / len(model_names)
    colors = [PALETTE["primary"], PALETTE["fraud"], PALETTE["secondary"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        values = [metrics[model_name][m] for m in metric_names]
        ax.bar(
            x + i * width,
            values,
            width,
            label=model_name,
            color=colors[i % len(colors)],
        )

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Across Evaluation Metrics", fontweight="bold")
    ax.legend()
    _save(fig, "model_comparison.png")


def generate_all_visualizations(
    df: pd.DataFrame,
    results: Dict[str, Dict[str, object]],
) -> None:
    """
    Convenience function that generates every required visualization in
    one call. Intended to be used from train.py.

    Args:
        df: The full (scaled) dataset including the target column.
        results: Dict keyed by model name, each containing at minimum
            'y_true', 'y_pred', 'y_scores', and optionally
            'feature_importances' and 'feature_names' (for XGBoost).
    """
    log("Generating Graphs...")

    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_fraud_vs_normal_distribution(df)

    comparison_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, data in results.items():
        plot_confusion_matrix_chart(data["y_true"], data["y_pred"], model_name)
        roc_auc = plot_roc_curve_chart(data["y_true"], data["y_scores"], model_name)
        plot_precision_recall_curve_chart(data["y_true"], data["y_scores"], model_name)

        comparison_metrics[model_name] = {
            "Accuracy": data["accuracy"],
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1 Score": data["f1"],
            "ROC AUC": roc_auc,
        }

        if "feature_importances" in data and "feature_names" in data:
            plot_feature_importance(data["feature_names"], data["feature_importances"])

    plot_model_comparison(comparison_metrics)
    log("All graphs generated and saved successfully.")
