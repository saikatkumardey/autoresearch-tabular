"""
train.py — Experiment 4: LightGBM default params
Hypothesis: LightGBM is often faster and matches XGBoost on tabular data
"""

import time
from lightgbm import LGBMClassifier

from prepare import (
    X_train, X_val, y_train, y_val,
    TASK, TIME_BUDGET, N_FEATURES, N_CLASSES,
    evaluate,
)

model = LGBMClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

t0 = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t0

results = evaluate(model, training_seconds)
