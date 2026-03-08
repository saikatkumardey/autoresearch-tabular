"""
train.py — The only file the agent modifies.

Baseline: LogisticRegression. The agent replaces/improves this:
try XGBoost, LightGBM, MLP, ensembles, feature engineering, stacking, etc.

Usage: uv run train.py
"""

import time
from sklearn.linear_model import LogisticRegression

from prepare import (
    X_train, X_val, y_train, y_val,
    TASK, TIME_BUDGET, N_FEATURES, N_CLASSES,
    evaluate,
)

# ---------------------------------------------------------------------------
# Model — change everything below this line
# ---------------------------------------------------------------------------

model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")

# ---------------------------------------------------------------------------
# Training (measure wall-clock training time)
# ---------------------------------------------------------------------------

t0 = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t0

# ---------------------------------------------------------------------------
# Evaluation (do not modify)
# ---------------------------------------------------------------------------

results = evaluate(model, training_seconds)
