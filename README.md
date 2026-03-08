# autoresearch-tabular

Autonomous ML research for tabular datasets — classification and regression.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch), which does the same thing for LLM pretraining. Same concept, different domain: give an AI agent a real ML training setup and let it experiment autonomously overnight.

The agent modifies `train.py`, trains a model, checks if val_auc improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better model.

## How it works

Three files that matter:

- **prepare.py** — fixed. Data download, preprocessing, train/val/test splits, and the `evaluate()` function. Do not modify.
- **train.py** — the only file the agent edits. Model definition and training loop. Everything is fair game: model choice, hyperparameters, feature engineering, ensembles.
- **program.md** — instructions for the agent. Point your agent here and let it go.

Default dataset: [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult) (binary classification: predict income >50K). Default metric: **val_auc** (higher is better).

Time budget per experiment: **2 minutes**. Enough for XGBoost and LightGBM runs; fast enough to get ~30 experiments/hour.

## Setup

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download dataset and verify setup (one-time, ~10 seconds)
uv run prepare.py

# 4. Run a single baseline experiment
uv run train.py
```

## Autonomous mode

Spin up Claude, Codex, or any agent in this repo:

```
Have a look at program.md and let's kick off a new experiment!
```

The agent will create a branch, run experiments in a loop, and log results to `results.tsv`.

## Design principles (inherited from the original)

- **Single file to modify.** Agent only touches `train.py`. Diffs stay reviewable.
- **Fixed time budget.** Experiments are comparable regardless of what the agent tries.
- **Self-contained.** No cloud APIs, no distributed training. One machine, one dataset, one metric.
- **Loop forever.** The agent never stops unless you interrupt it.

## Extending

To switch to regression, replace `prepare.py` with a regression dataset (e.g. California Housing from sklearn) and change `evaluate()` to return val_rmse and val_r2. Update `program.md` to reflect the new metric.

---

Credit: concept and structure from [@karpathy](https://github.com/karpathy/autoresearch). This fork adapts it for traditional tabular ML.
