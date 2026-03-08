"""
prepare.py — Fixed constants, data prep, and evaluation for autoresearch-tabular.

DO NOT MODIFY. This file is the ground truth: dataset loading, preprocessing,
train/val/test splits, and the evaluate() function. The agent only edits train.py.

Usage: uv run prepare.py  (run once to download and cache the dataset)
"""

import os
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# Constants (do not modify)
# ---------------------------------------------------------------------------

TASK = "classification"          # "classification" or "regression"
TIME_BUDGET = 120                # seconds — each experiment has 2 minutes of training time
RANDOM_STATE = 42
CACHE_DIR = Path.home() / ".cache" / "autoresearch-tabular"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATASET_FILE = CACHE_DIR / "adult.data"

# Column names for Adult Income dataset
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

NUMERIC_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

TARGET_COL = "income"

# ---------------------------------------------------------------------------
# Data download + cache
# ---------------------------------------------------------------------------

def download_dataset():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_FILE.exists():
        print(f"Dataset already cached at {DATASET_FILE}")
        return
    print(f"Downloading Adult Income dataset...")
    r = requests.get(DATASET_URL, timeout=30)
    r.raise_for_status()
    DATASET_FILE.write_bytes(r.content)
    print(f"Saved to {DATASET_FILE}")


def load_and_preprocess():
    download_dataset()

    df = pd.read_csv(
        DATASET_FILE,
        names=COLUMNS,
        sep=", ",
        engine="python",
        na_values="?",
    )
    df = df.dropna()

    # Encode target
    df[TARGET_COL] = (df[TARGET_COL].str.strip() == ">50K").astype(int)

    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[CATEGORICAL_COLS + NUMERIC_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    # Scale numerics (columns at end of feature matrix)
    n_cat = len(CATEGORICAL_COLS)
    scaler = StandardScaler()
    X[:, n_cat:] = scaler.fit_transform(X[:, n_cat:])

    # Split: 60% train, 20% val, 20% test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
    )  # 0.25 of 0.8 = 0.2 overall

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Load data at import time (cached in module scope)
# ---------------------------------------------------------------------------

X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

N_FEATURES = X_train.shape[1]
N_CLASSES = int(np.max(y_train)) + 1

# ---------------------------------------------------------------------------
# Evaluation (ground truth — do not modify)
# ---------------------------------------------------------------------------

def evaluate(model, training_seconds: float) -> dict:
    """
    Evaluate model on validation set.
    model must implement predict() and predict_proba() (sklearn API).
    Returns a dict with val_accuracy, val_auc, training_seconds, total_seconds.
    """
    t0 = time.time()

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    val_accuracy = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_prob)

    total_seconds = training_seconds + (time.time() - t0)

    results = {
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "n_features": N_FEATURES,
        "n_train_samples": len(X_train),
    }

    print("---")
    print(f"val_accuracy:     {val_accuracy:.6f}")
    print(f"val_auc:          {val_auc:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"n_features:       {N_FEATURES}")
    print(f"n_train_samples:  {len(X_train)}")

    return results


# ---------------------------------------------------------------------------
# Run standalone to verify setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Task:            {TASK}")
    print(f"Time budget:     {TIME_BUDGET}s")
    print(f"Train samples:   {len(X_train)}")
    print(f"Val samples:     {len(X_val)}")
    print(f"Test samples:    {len(X_test)}")
    print(f"Features:        {N_FEATURES}")
    print(f"Classes:         {N_CLASSES}")
    print()
    print("Setup OK. Run 'uv run train.py' to verify the baseline.")
