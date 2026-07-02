"""
utils.py
--------
Shared utility functions and constants used across the Credit Card Fraud
Detection project. Centralizing these here avoids code duplication between
train.py, predict.py, preprocess.py, and app.py.

Author: Anand D
"""

from __future__ import annotations

import functools
import os
import time
from datetime import datetime
from typing import Any, Callable

import joblib

# --------------------------------------------------------------------------
# Global constants
# --------------------------------------------------------------------------

# Random seed used everywhere (train/test split, SMOTE, model init) so that
# results are 100% reproducible between runs.
RANDOM_STATE: int = 42

# Name of the dataset file the program expects to find in the project root.
# We deliberately use a relative path (never an absolute path) so the
# project works on any machine/OS without modification.
DATASET_FILENAME: str = "dataset.csv"

# Filenames for the two trained models, saved with joblib in the project
# root (no subfolders, as required).
XGB_MODEL_FILENAME: str = "xgb_model.pkl"
ISOLATION_MODEL_FILENAME: str = "isolation_model.pkl"
SCALER_FILENAME: str = "scaler.pkl"

# Directory (project root) images are saved into. Kept as "." so no new
# folders are created.
IMAGE_DIR: str = "."

# Target column name in the Kaggle Credit Card Fraud Detection dataset.
TARGET_COLUMN: str = "Class"


# --------------------------------------------------------------------------
# Logging helper
# --------------------------------------------------------------------------

def log(message: str) -> None:
    """
    Print a timestamped progress message to the console.

    This gives the user clear, professional feedback about what the
    pipeline is doing at every stage (e.g. "Loading Dataset...",
    "Applying SMOTE...", "Training XGBoost...").

    Args:
        message: The message to display.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


# --------------------------------------------------------------------------
# Timing decorator
# --------------------------------------------------------------------------

def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that measures and logs how long a function takes to run.

    Used to report training time and prediction time for each model,
    which is a required metric in the evaluation report.

    Args:
        func: The function being timed.

    Returns:
        A wrapped function that logs its own execution time and also
        returns that elapsed time (in seconds) alongside the original
        return value as a tuple: (result, elapsed_seconds).
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        log(f"'{func.__name__}' completed in {elapsed:.4f} seconds")
        return result, elapsed

    return wrapper


# --------------------------------------------------------------------------
# File path helpers
# --------------------------------------------------------------------------

def get_dataset_path() -> str:
    """
    Return the relative path to the dataset file.

    Returns:
        The relative path string (never absolute), e.g. "dataset.csv".
    """
    return DATASET_FILENAME


def dataset_exists() -> bool:
    """
    Check whether the dataset file exists in the project root.

    Returns:
        True if the dataset file is present, False otherwise.
    """
    return os.path.isfile(get_dataset_path())


# --------------------------------------------------------------------------
# Model persistence helpers
# --------------------------------------------------------------------------

def save_model(model: Any, filename: str) -> None:
    """
    Save a trained model (or any Python object) to disk using joblib.

    Args:
        model: The fitted model/object to persist.
        filename: Destination filename (relative path, project root).
    """
    joblib.dump(model, filename)
    log(f"Saved '{filename}' to disk.")


def load_model(filename: str) -> Any:
    """
    Load a previously saved model from disk using joblib.

    Args:
        filename: Filename of the saved model (relative path).

    Returns:
        The deserialized model/object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"Model file '{filename}' was not found. Please run "
            f"'python train.py' first to train and save the models."
        )
    return joblib.load(filename)


def model_exists(filename: str) -> bool:
    """
    Check whether a saved model file already exists.

    Used to decide whether to retrain or simply load the cached model,
    avoiding unnecessary retraining every time the app starts.

    Args:
        filename: Filename of the model to check.

    Returns:
        True if the file exists, False otherwise.
    """
    return os.path.isfile(filename)


def all_models_exist() -> bool:
    """
    Check whether both trained models and the scaler already exist on disk.

    Returns:
        True if xgb_model.pkl, isolation_model.pkl, and scaler.pkl are all
        present, False otherwise.
    """
    return (
        model_exists(XGB_MODEL_FILENAME)
        and model_exists(ISOLATION_MODEL_FILENAME)
        and model_exists(SCALER_FILENAME)
    )
