"""
prepare.py — Fixed constants, data prep, and evaluation for autoresearch-tabular.

READ-ONLY during experiment runs. The agent only edits train.py.

To use a different dataset, create a new prepare.py following the same contract:
- Export: X_train, X_val, X_test, y_train, y_val, y_test, N_FEATURES, N_CLASSES
- Export: TASK, TIME_BUDGET, RANDOM_STATE
- Export: evaluate(model, training_seconds) -> dict
- evaluate() must print "val_auc:" as a grep-able line

Usage: uv run prepare.py  (run once to download and cache the dataset)
"""

import os
import sys
import time
import warnings
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

    # Split FIRST, before any encoding/scaling (60/20/20)
    X_df = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL].values.astype(np.int32)

    X_tv, X_test_df, y_tv, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv
    )  # 0.25 of 0.8 = 0.2 overall
    X_train_df = X_train_df.copy()
    X_val_df = X_val_df.copy()
    X_test_df = X_test_df.copy()

    # Fit LabelEncoders on TRAIN only, transform all splits
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(X_train_df[col].astype(str))
        known = set(le.classes_)
        for split_name, split_df in [("train", X_train_df), ("val", X_val_df), ("test", X_test_df)]:
            col_vals = split_df[col].astype(str)
            unseen = set(col_vals.unique()) - known
            if unseen:
                warnings.warn(
                    f"Column '{col}' in {split_name} has unseen categories {unseen}, mapping to -1"
                )
                split_df.loc[:, col] = col_vals.map(
                    lambda x, _le=le, _known=known: _le.transform([x])[0] if x in _known else -1
                )
            else:
                split_df.loc[:, col] = le.transform(col_vals)

    X_train = X_train_df.values.astype(np.float32)
    X_val = X_val_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    # Fit StandardScaler on TRAIN only, transform all splits
    n_cat = len(CATEGORICAL_COLS)
    scaler = StandardScaler()
    X_train[:, n_cat:] = scaler.fit_transform(X_train[:, n_cat:])
    X_val[:, n_cat:] = scaler.transform(X_val[:, n_cat:])
    X_test[:, n_cat:] = scaler.transform(X_test[:, n_cat:])

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
    Evaluate model on validation and test sets.
    model must implement predict() and predict_proba() (sklearn API).
    Returns a dict with val/test accuracy, val/test auc, timing info.
    val_auc is the optimization target. test metrics are for final reporting only.
    """
    t0 = time.time()

    # Validation metrics (optimization target)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    val_accuracy = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_prob)

    # Test metrics (final generalization check — not for optimization)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob)

    total_seconds = training_seconds + (time.time() - t0)

    results = {
        "val_accuracy": val_accuracy,
        "val_auc": val_auc,
        "test_accuracy": test_accuracy,
        "test_auc": test_auc,
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
    print(f"test_accuracy:    {test_accuracy:.6f}")
    print(f"test_auc:         {test_auc:.6f}")

    # Enforce time budget
    if training_seconds > TIME_BUDGET:
        print(f"\nOVER BUDGET: training took {training_seconds:.1f}s (budget: {TIME_BUDGET}s)")
        sys.exit(1)

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
