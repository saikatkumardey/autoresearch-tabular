"""
train.py — Experiment 2: RandomForest n_estimators=200
Hypothesis: ensemble of decision trees should outperform LR on this tabular data
"""

import time
from sklearn.ensemble import RandomForestClassifier

from prepare import (
    X_train, X_val, y_train, y_val,
    TASK, TIME_BUDGET, N_FEATURES, N_CLASSES,
    evaluate,
)

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

t0 = time.time()
model.fit(X_train, y_train)
training_seconds = time.time() - t0

results = evaluate(model, training_seconds)
