"""
train.py — Experiment 6: LightGBM with early stopping on val set
Hypothesis: let val_auc guide the number of trees instead of fixed n_estimators
"""

import time
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from prepare import (
    X_train, X_val, y_train, y_val,
    TASK, TIME_BUDGET, N_FEATURES, N_CLASSES,
    evaluate,
)

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
)
training_seconds = time.time() - t0

results = evaluate(model, training_seconds)
