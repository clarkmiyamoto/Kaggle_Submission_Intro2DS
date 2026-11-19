# CLAUDE.md

Kaggle [Tabular Playground Series, Oct 2022](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022)
(Rocket League). Each row is one frame of a game; predict the probability that team A and
team B score within the next 10 seconds. Metric: **mean columnwise log loss**.

## This repo: the final submission

This repo holds our final Kaggle submission (public 0.18296, private 0.17610) and what is
needed to rebuild it. `README.md`, "Final submission", lists its files;
`docs/experiments/final_submission.md` describes the model.

- **To reproduce the final model, follow `REPRODUCE.md`.** It works with Modal
  (`bash scripts/reproduce_final.sh modal`) or without it (`bash scripts/reproduce_final.sh
  local`, on CUDA, Apple MPS or CPU) and ends with `src.verify_submission` comparing the new
  CSV with the submitted one. Run `bash scripts/reproduce_final.sh smoke` first.
- Don't edit `configs/final/`, `configs/host/`,
  `submissions/final_full24_wide2_drop0.05_trval_3seeds.csv` or the committed `results/`
  files: they record what was submitted and how it was validated.

## Data

`data/` is gitignored and not in the repo; `src/data.py` looks for it next to itself. Get it
with `README.md`, "Setup" (`scripts/download_data.sh`, then `uv run python -m src.make_splits`),
or link an existing copy (`ln -s /path/to/data data`). `git status --short` must not list
`data`. The root `.gitignore` ignores `/data`, which covers a link: keep that rule without a
trailing slash, since `data/` matches directories only.

## Environment

- uv only: `uv sync`, `uv run python -m src.<module>` from the repo root, `uv add <pkg>`
  (commit `uv.lock` in the same commit).
- `host_net` takes `device: auto` (cuda, then mps, then cpu) and `compile: reduce-overhead`
  (CUDA graphs, CUDA only). On an M4 Max, MPS is 5-10x faster than CPU.
- Modal is optional (`REPRODUCE.md`, "Path A"). The container runs the committed HEAD:
  commit first, then `uv run modal run --detach scripts/modal_train.py::main --gpu A10G
  --module src.retrain --configs <a.yaml,b.yaml>`, and fetch with
  `uv run modal run scripts/modal_train.py::fetch --name <run name>`. Runs on one GPU type are
  close to reproducible; A10 and A10G differ by about seed noise (`fit_info.device_name`).
- ~19M train + val rows. Use polars lazily (`pl.scan_parquet`, only the columns you need).

## Data and splits

```python
from src.data import load_split, load_kaggle_test, FEATURES, TARGETS, META
```

| Split | Games | Use |
| --- | --- | --- |
| `train` | 5,893 | Fit models, features, preprocessing |
| `val` | 736 | Validation; the final submission also trains on it (`src.retrain`) |
| `test` | 736 | Labeled holdout: never trained on; score only when asked |
| `data/test.csv` | Kaggle's, unlabeled | Submissions only |

Splits are by game and frozen in `splits/game_splits.csv`.

## Hard rules

- Model inputs are `FEATURES` plus registered feature sets. Never a `META` column:
  `event_time` + `team_scoring_next` reproduce the targets, and no `META` column exists in
  Kaggle's test set. `META` columns may be training labels only, through a model's
  `TRAIN_LABELS` (`run.py` passes them to `fit` as `labels_tr`).
- Features must be computable from a single row. Kaggle's test set has only `id` +
  `FEATURES`, so no lags, rolling windows or per-game aggregates.
- Fit scalers, imputers and encoders on `train` only. (`src.retrain`, for the final
  submission only, fits on `train` + `val` by design.)
- Never edit `splits/game_splits.csv` or the split logic in `src/make_splits.py`.
- Score with `src.metrics.score`: two binary log losses averaged, on probabilities, never
  logits or margins.
- `accuracy`, `auc` and `ap` in result files are diagnostics only: accuracy is about 0.94 for
  every model (always predicting "no goal" scores 0.9424), and AUC and AP ignore calibration.
- Claim an improvement on `val` only with the paired bootstrap of `src.compare`, on 3 seeds.

## Commands

| Task | Command |
| --- | --- |
| Train on `train`, score on `val` | `uv run python -m src.run <config>.yaml` |
| Train on `train` + `val` for a submission | `uv run python -m src.retrain <config>.yaml [--device ...] [--max-hours none]` |
| List scored runs | `uv run python -m src.leaderboard` |
| Compare / ensemble runs on `val` | `uv run python -m src.compare BASE NEW`, `uv run python -m src.blend <prefix>` |
| Build a submission CSV | `uv run python -m src.submit --name <name> <run_id> ...` |
| Check a CSV against the submitted one | `uv run python -m src.verify_submission submissions/<name>.csv` |
| Check HostNet's invariances | `uv run python -m scripts.check_host_net` |

## Git

- Commit code, configs, `results/*.json`, `splits/*.csv`, `uv.lock`. Never `data/`,
  `runs/`, model files or `submissions/`, except the submitted CSV, which is committed as
  the reference for `src.verify_submission`.
- Ask before editing core: `src/data.py`, `src/make_splits.py`, `src/metrics.py`,
  `src/run.py`, `src/retrain.py`, `splits/`.
