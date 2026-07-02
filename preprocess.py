"""
preprocess.py
-------------
Handles everything related to loading and preparing the Credit Card Fraud
Detection dataset:

    * Dataset auto-detection (with a friendly message if missing)
    * Missing value checks
    * Duplicate removal
    * Data information / statistics summaries
    * Class distribution reporting
    * Feature scaling
    * Train/test splitting
    * SMOTE oversampling for the imbalanced classes

Author: Anand D
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import RANDOM_STATE, TARGET_COLUMN, dataset_exists, get_dataset_path, log


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection dataset from the project root.

    The Kaggle dataset ("creditcard.csv") is too large (and licensed) to
    ship inside a GitHub repository, so this function checks for its
    presence at runtime instead of assuming it exists.

    Returns:
        The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file cannot be found, with a
            friendly, actionable message telling the user exactly where
            to place it.
    """
    if not dataset_exists():
        raise FileNotFoundError(
            "\n"
            "==================================================================\n"
            " Dataset not found!\n"
            "==================================================================\n"
            " This project expects a file named 'dataset.csv' in the project\n"
            " root folder (the same folder as train.py).\n"
            "\n"
            " Steps to fix this:\n"
            "   1. Download the 'Credit Card Fraud Detection' dataset from\n"
            "      Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "   2. Rename the downloaded file to 'dataset.csv'\n"
            "   3. Place it in this project's root folder\n"
            "   4. Re-run this script\n"
            "==================================================================\n"
        )

    log("Loading Dataset...")
    df = pd.read_csv(get_dataset_path())
    log(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# Missing values & duplicates
# --------------------------------------------------------------------------

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check each column for missing (NaN) values.

    Fraud datasets sourced from real payment processors are usually clean,
    but this check is essential to catch any corrupted rows before
    training, since ML models cannot handle NaNs directly.

    Args:
        df: The input DataFrame.

    Returns:
        A Series indexed by column name with the count of missing values
        in each column.
    """
    log("Checking for missing values...")
    missing = df.isnull().sum()
    total_missing = int(missing.sum())
    if total_missing == 0:
        log("No missing values found.")
    else:
        log(f"Found {total_missing} missing values across the dataset.")
    return missing


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate transaction rows from the dataset.

    Duplicate rows can bias the model by over-representing certain
    transactions, artificially inflating both training and test accuracy.

    Args:
        df: The input DataFrame.

    Returns:
        A new DataFrame with duplicate rows removed.
    """
    log("Removing duplicate rows...")
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df_clean)
    log(f"Removed {removed} duplicate rows. Remaining rows: {len(df_clean)}")
    return df_clean


# --------------------------------------------------------------------------
# Data summaries
# --------------------------------------------------------------------------

def show_data_info(df: pd.DataFrame) -> None:
    """
    Print structural information about the dataset (dtypes, non-null
    counts, memory usage) -- equivalent to pandas' df.info().

    Args:
        df: The input DataFrame.
    """
    log("Dataset Info:")
    df.info()


def show_data_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, std, min, max, quartiles) for
    every numeric column in the dataset.

    Args:
        df: The input DataFrame.

    Returns:
        A DataFrame containing the summary statistics.
    """
    log("Computing dataset statistics...")
    stats = df.describe()
    return stats


def show_class_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Report how many transactions are legitimate (Class = 0) vs fraudulent
    (Class = 1), and the percentage each class represents.

    This is the single most important diagnostic for a fraud detection
    project: it demonstrates *why* the dataset is imbalanced and *why*
    techniques like SMOTE are required at all.

    Args:
        df: The input DataFrame. Must contain the target column.

    Returns:
        A Series with the raw counts per class.
    """
    counts = df[TARGET_COLUMN].value_counts()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    log("Class Distribution:")
    for cls in counts.index:
        label = "Fraudulent" if cls == 1 else "Legitimate"
        log(f"  Class {cls} ({label}): {counts[cls]} transactions "
            f"({percentages[cls]:.4f}%)")
    return counts


# --------------------------------------------------------------------------
# Feature scaling
# --------------------------------------------------------------------------

def scale_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler (zero mean, unit variance).

    The Kaggle dataset's 'Time' and 'Amount' columns are on a very
    different scale than the PCA-transformed V1-V28 features, which can
    destabilize distance-based and gradient-based algorithms. Scaling
    puts every feature on comparable footing.

    Args:
        df: The input DataFrame (features + target column).

    Returns:
        A tuple of (scaled_dataframe, fitted_scaler). The fitted scaler
        is returned so it can be reused later on new/unseen data during
        prediction (it must NOT be refit on new data).
    """
    log("Scaling features (Time and Amount)...")
    df_scaled = df.copy()
    scaler = StandardScaler()

    # Only 'Time' and 'Amount' need scaling -- the V1-V28 columns are
    # already PCA components and are approximately standardized already.
    columns_to_scale = [col for col in ["Time", "Amount"] if col in df_scaled.columns]

    if columns_to_scale:
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
        log(f"Scaled columns: {columns_to_scale}")
    else:
        log("No 'Time'/'Amount' columns found to scale; skipping.")

    return df_scaled, scaler


# --------------------------------------------------------------------------
# Train/test split
# --------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets (80/20), using
    stratified sampling so the fraud/legitimate ratio is preserved in
    both splits.

    Args:
        df: The scaled DataFrame including the target column.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    log("Splitting data into train and test sets (80/20, stratified)...")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,  # reproducibility
        stratify=y,                  # preserve class ratio in both splits
    )
    log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# SMOTE (Synthetic Minority Over-sampling Technique)
# --------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training data only (never to the test set, which
    must reflect the real-world imbalanced distribution).

    Why fraud datasets are imbalanced:
        In real-world payment systems, fraudulent transactions typically
        make up well under 1% of all transactions. Fraudsters are rare
        by nature and card issuers actively block obvious fraud before
        it is even recorded, so the datasets we train on are inherently
        skewed toward legitimate transactions.

    Why SMOTE is needed:
        A classifier trained on raw imbalanced data can achieve >99%
        accuracy simply by always predicting "legitimate" -- which is
        useless for catching fraud. SMOTE generates synthetic minority
        class (fraud) samples by interpolating between existing fraud
        examples and their nearest neighbours, giving the model enough
        balanced signal to actually learn fraud patterns instead of
        ignoring the minority class.

    Advantages:
        * Reduces bias toward the majority class without simply
          duplicating existing minority rows (which would cause
          overfitting to identical points).
        * Improves recall on the minority (fraud) class, which is the
          metric that matters most in fraud detection.
        * Works directly on the feature space, so it integrates cleanly
          with any scikit-learn-compatible model.

    Disadvantages:
        * Synthetic samples can introduce noise if minority class
          examples are not well separated from the majority class,
          potentially blurring the decision boundary.
        * Increases training set size and therefore training time.
        * Does not consider the majority class distribution when
          generating synthetic points, which can occasionally produce
          unrealistic feature combinations.

    Args:
        X_train: Training features (imbalanced).
        y_train: Training labels (imbalanced).

    Returns:
        A tuple (X_resampled, y_resampled) with a balanced 1:1 class
        ratio.
    """
    log("Applying SMOTE to balance the training data...")
    log(f"Before SMOTE -> {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    log(f"After SMOTE  -> {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled


# --------------------------------------------------------------------------
# Full preprocessing pipeline (convenience wrapper)
# --------------------------------------------------------------------------

def run_full_preprocessing() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
    pd.DataFrame, pd.Series, pd.DataFrame, StandardScaler,
]:
    """
    Run the complete preprocessing pipeline end to end: load, clean,
    inspect, scale, split, and balance the dataset.

    Returns:
        A tuple:
            (X_train_resampled, X_test, y_train_resampled, y_test,
             X_train_original, y_train_original, full_scaled_df, fitted_scaler)

        X_train_original/y_train_original (the pre-SMOTE training split)
        are returned separately because Isolation Forest is unsupervised
        and should be trained on the true, unbalanced data distribution
        rather than the SMOTE-balanced one.
    """
    df = load_dataset()
    check_missing_values(df)
    df = remove_duplicates(df)
    show_data_info(df)
    show_data_statistics(df)
    show_class_distribution(df)

    df_scaled, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test, X_train, y_train, df_scaled, scaler
