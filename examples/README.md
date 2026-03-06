# Examples

Reference `prepare.py` implementations for different datasets. Each shows how to adapt the autoresearch-tabular contract to a new problem.

## How to use

1. Copy the example's `prepare.py` to the repo root:
   ```bash
   cp examples/<dataset>/prepare.py .
   ```
2. Verify setup:
   ```bash
   uv run prepare.py
   ```
3. Run a baseline experiment:
   ```bash
   uv run train.py
   ```
4. For regression datasets, update `program.md` to grep `^val_rmse:` instead of `^val_auc:` and note that **lower is better**.

## Available datasets

| Dataset | Task | Metric | Features | Samples |
|---------|------|--------|----------|---------|
| [Adult Income](../prepare.py) (default) | classification | val_auc | 14 (8 cat, 6 num) | ~30k |
| [Titanic](titanic/) | classification | val_auc | 7 (3 cat, 4 num) | ~891 |
| [Diabetes](diabetes/) | regression | val_rmse | 10 (all num) | 442 |
| [California Housing](california-housing/) | regression | val_rmse | 8 (all num) | ~20,640 |

## Contract

Every `prepare.py` must export:

- `X_train, X_val, X_test, y_train, y_val, y_test` — data splits
- `N_FEATURES, N_CLASSES` — dataset metadata (N_CLASSES=0 for regression)
- `TASK, TIME_BUDGET, RANDOM_STATE` — constants
- `evaluate(model, training_seconds) -> dict` — prints grep-able metric line

See the [root prepare.py](../prepare.py) docstring for the full contract.
