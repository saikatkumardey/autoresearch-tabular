# Diabetes Regression

Regression: predict disease progression one year after baseline.

- **Source:** `sklearn.datasets.load_diabetes` (built-in, no download needed)
- **Task:** regression
- **Metric:** val_rmse (lower is better), also reports val_r2
- **Features:** 10 numeric (age, sex, bmi, blood pressure, 6 serum measurements)
- **Samples:** 442 (very small)
- **Missing data:** none

## Usage

```bash
cp examples/diabetes/prepare.py .
uv run prepare.py    # verify setup
uv run train.py      # run baseline
```

**Important:** Update `program.md` to grep `^val_rmse:` instead of `^val_auc:`, and note that **lower is better** for RMSE.
