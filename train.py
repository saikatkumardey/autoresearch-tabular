"""
train.py — Experiment 3: XGBoost default params
Hypothesis: gradient boosting should beat random forest on this dataset
"""

import time
from xgboost import XGBClassifier

from prepare import (
    X_train, X_val, y_train, y_val,
    TASK, TIME_BUDGET, N_FEATURES, N_CLASSES,
    evaluate,
)

model = XGBClassifier(
    n_estimators=300,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t0

results = evaluate(model, training_seconds)
