# autoresearch-tabular

Autonomous ML research for tabular datasets. The agent experiments with models, hyperparameters,
and feature engineering — iterating until it finds the best val_auc on the Adult Income dataset.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar8`). The branch
   `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, preprocessing, and `evaluate()`. Do not modify.
   - `train.py` — the file you modify. Model definition and training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch-tabular/adult.data` exists.
   If not, run `uv run prepare.py` to download it.
5. **Initialize results.tsv**: Create with header row. Baseline val_auc comes from the first run.
6. **Confirm and go**: Confirm setup looks good, then start the experiment loop.

## Experimentation

Each experiment runs for a **fixed time budget of 2 minutes** (wall-clock training time).
Launch as: `uv run train.py > run.log 2>&1`

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Model choice: LogisticRegression, RandomForest, XGBoost, LightGBM, MLP, ensembles, stacking
  - Hyperparameters: learning rate, depth, regularization, n_estimators, etc.
  - Feature engineering: polynomial features, interactions, binning, log transforms
  - Ensembles: VotingClassifier, StackingClassifier, blending
  - Any sklearn-compatible model API

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading,
  preprocessing, and the `evaluate()` function.
- Install new packages. You can only use what's in `pyproject.toml`:
  scikit-learn, xgboost, lightgbm, pandas, numpy, requests, matplotlib.
- Modify the `evaluate()` function. It is the ground truth metric.

**The goal: get the highest val_auc.** Higher is better.

Since the time budget is fixed at 2 minutes, models that train in 10 seconds and models that
train in 90 seconds are both fine — what matters is the val_auc at the end. The only constraint
is the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 val_auc improvement
that adds 50 lines of brittle code is probably not worth it. A simplification that matches
performance is always a win.

## Output format

The script prints a summary like:

```
---
val_accuracy:     0.856700
val_auc:          0.923400
training_seconds: 45.3
total_seconds:    46.1
n_features:       14
n_train_samples:  23478
```

Extract key metric: `grep "^val_auc:" run.log`

## Logging results

Log every experiment to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_auc	status	description
```

Columns:
1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.923400) — use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short description of what this experiment tried

Example:
```
commit	val_auc	status	description
a1b2c3d	0.856700	keep	baseline LogisticRegression C=1.0
b2c3d4e	0.901200	keep	RandomForest n_estimators=200
c3d4e5f	0.912800	keep	XGBoost default params
d4e5f6g	0.914100	keep	XGBoost lr=0.05 depth=6
e5f6g7h	0.000000	0.0	crash	StackingClassifier OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar8`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Form a hypothesis — what might improve val_auc? Write it down in a comment in train.py.
3. Modify `train.py` with the experimental idea.
4. git commit
5. Run the experiment: `uv run train.py > run.log 2>&1`
6. Read results: `grep "^val_auc:" run.log`
7. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the error.
   - If it is a simple bug (typo, import error), fix and re-run.
   - If the idea is fundamentally broken, log as crash and move on.
8. Record in results.tsv.
9. If val_auc improved (higher), **keep** the commit — advance the branch.
10. If val_auc is equal or worse, `git reset --hard HEAD~1` — discard.

**Ideas to try (in rough order of expected payoff):**
- XGBoost with default params (usually beats LR on tabular)
- LightGBM (often faster than XGBoost, comparable accuracy)
- Tune XGBoost/LightGBM hyperparameters: n_estimators, max_depth, learning_rate, subsample
- RandomForest with n_estimators=500+
- Feature engineering: polynomial features, log(1+x) on skewed numerics, interaction terms
- Ensembles: VotingClassifier combining best models
- Stacking: use LR/Ridge as meta-learner on top of base models
- Calibration: CalibratedClassifierCV to improve probability estimates (helps AUC)
- One-hot encoding vs label encoding for categoricals

**NEVER STOP**: Once the experiment loop begins, do NOT pause to ask the human if you should
continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human
might be asleep. You are autonomous. If you run out of ideas, think harder — try combining
previous near-misses, try more radical feature engineering, try meta-learning. The loop runs
until the human interrupts you, period.

Each experiment takes ~2 minutes. In 8 hours you can run ~240 experiments. Make them count.
