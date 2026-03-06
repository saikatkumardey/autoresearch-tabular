# California Housing Regression

Regression: predict median house value (in $100k units).

- **Source:** `sklearn.datasets.fetch_california_housing` (downloaded via sklearn on first call)
- **Task:** regression
- **Metric:** val_rmse (lower is better), also reports val_r2
- **Features:** 8 numeric (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **Samples:** ~20,640
- **Missing data:** none

## Usage

```bash
cp examples/california-housing/prepare.py .
uv run prepare.py    # download and verify
uv run train.py      # run baseline
```

**Important:** Update `program.md` to grep `^val_rmse:` instead of `^val_auc:`, and note that **lower is better** for RMSE.
