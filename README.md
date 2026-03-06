# autoresearch-tabular

Autonomous ML research for tabular datasets. Give an AI agent a training setup and let it experiment overnight.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for traditional tabular ML.

## How it works

The agent modifies `train.py`, trains a model, checks if the metric improved, keeps or discards the commit, and repeats.

Three files:

- **prepare.py** — read-only. Data loading, preprocessing, splits, and `evaluate()`. Swap this file to change datasets.
- **train.py** — the only file the agent edits. Model, hyperparameters, feature engineering.
- **program.md** — agent instructions. Point your agent here and let it go.

Default dataset: [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult) (binary classification). Default metric: **val_auc**.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync                                             # install deps
uv run prepare.py                                   # download dataset
uv run train.py                                     # run baseline
```

## Autonomous mode

```
Have a look at program.md and let's kick off a new experiment!
```

The agent creates a branch, runs experiments in a loop, and logs results to `results.tsv`. Each experiment has a **2-minute time budget**.

## Examples

See [`examples/`](examples/) for ready-made `prepare.py` configs:

| Dataset | Task | Metric | Samples |
|---------|------|--------|---------|
| Adult Income (default) | classification | val_auc | ~30k |
| [Titanic](examples/titanic/) | classification | val_auc | ~891 |
| [Diabetes](examples/diabetes/) | regression | val_rmse | 442 |
| [California Housing](examples/california-housing/) | regression | val_rmse | ~20k |

To use an example: `cp examples/<dataset>/prepare.py . && uv run prepare.py`

## Extending

Create a new `prepare.py` following the contract:

- **Exports:** `X_train, X_val, X_test, y_train, y_val, y_test, N_FEATURES, N_CLASSES`
- **Constants:** `TASK, TIME_BUDGET, RANDOM_STATE`
- **`evaluate(model, training_seconds)`** — prints a grep-able metric line (e.g. `val_auc:` or `val_rmse:`)
- **Splits:** fit encoders/scalers on train only, transform all splits

For regression, update `program.md` to grep the new metric and note whether higher or lower is better.

## Design

- **Single file to modify.** Diffs stay reviewable.
- **Fixed time budget.** Experiments are comparable.
- **Self-contained.** No cloud APIs. One machine, one metric.
- **Loop forever.** The agent never stops unless you interrupt it.
