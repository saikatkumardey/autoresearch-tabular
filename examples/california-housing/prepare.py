"""
prepare.py — California Housing regression.

READ-ONLY during experiment runs. The agent only edits train.py.

To use: copy this file to the repo root, replacing the default prepare.py.
  cp examples/california-housing/prepare.py . && uv run prepare.py && uv run train.py

Contract:
- Export: X_train, X_val, X_test, y_train, y_val, y_test, N_FEATURES, N_CLASSES
- Export: TASK, TIME_BUDGET, RANDOM_STATE
- Export: evaluate(model, training_seconds) -> dict
- evaluate() prints "val_rmse:" as a grep-able line (lower is better)

NOTE: For regression, update program.md to grep "^val_rmse:" and note lower is better.
"""

import sys
import time
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK = "regression"
TIME_BUDGET = 120
RANDOM_STATE = 42
N_CLASSES = 0  # regression — no classes

# ---------------------------------------------------------------------------
# Data loading + preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess():
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    # Split FIRST (60/20/20)
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE
    )

    # Fit StandardScaler on TRAIN only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Keep float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Load data at import time
# ---------------------------------------------------------------------------

X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

N_FEATURES = X_train.shape[1]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, training_seconds: float) -> dict:
    """
    Evaluate model on validation and test sets.
    model must implement predict() (sklearn API).
    val_rmse is the optimization target (lower is better).
    test metrics are for final reporting only.
    """
    t0 = time.time()

    y_pred = model.predict(X_val)
    val_rmse = root_mean_squared_error(y_val, y_pred)
    val_r2 = r2_score(y_val, y_pred)

    y_test_pred = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    total_seconds = training_seconds + (time.time() - t0)

    results = {
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "n_features": N_FEATURES,
        "n_train_samples": len(X_train),
    }

    print("---")
    print(f"val_rmse:         {val_rmse:.6f}")
    print(f"val_r2:           {val_r2:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"n_features:       {N_FEATURES}")
    print(f"n_train_samples:  {len(X_train)}")
    print(f"test_rmse:        {test_rmse:.6f}")
    print(f"test_r2:          {test_r2:.6f}")

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
    print(f"Classes:         {N_CLASSES} (regression)")
    print()
    print("Setup OK. Run 'uv run train.py' to verify the baseline.")
    print("NOTE: Update program.md to grep '^val_rmse:' (lower is better).")
