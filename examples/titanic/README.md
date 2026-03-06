# Titanic Survival Classification

Binary classification: predict passenger survival (0/1).

- **Source:** [Titanic dataset](https://github.com/datasciencedojo/datasets) (downloaded and cached)
- **Task:** classification
- **Metric:** val_auc (higher is better)
- **Features:** 7 (3 categorical: Sex, Embarked, Pclass; 4 numeric: Age, Fare, SibSp, Parch)
- **Samples:** ~891
- **Missing data:** Age (~20% missing, median-imputed from train set)

## Usage

```bash
cp examples/titanic/prepare.py .
uv run prepare.py    # download and verify
uv run train.py      # run baseline
```

The agent greps `^val_auc:` — same as the default Adult Income dataset. No changes to `program.md` needed.
