# Kaggle Tabular Playground Series, Oct 2022: final submission

Kaggle's [Tabular Playground Series, Oct 2022](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022)
(Rocket League). Each row is one frame of a game; the task is the probability that team A and
team B score within the next 10 seconds, scored by mean columnwise log loss.

This repo holds our final submission and everything needed to rebuild it.

## Final submission

**Kaggle: public 0.18296, private 0.17610** (submitted Nov 13, 2026).

A 3-seed ensemble of HostNet, our port of the competition host's net, 2x wide with dropout
0.05 and 1..10 s auxiliary heads, trained for a fixed 24 passes on every game of our `train`
and `val` splits. Recipe name `host_full24_wide2_drop0.05`.

| | Files |
| --- | --- |
| Submitted CSV | `submissions/final_full24_wide2_drop0.05_trval_3seeds.csv` |
| Configs of the 3 submitted runs | `configs/final/final_full24_wide2_drop0.05_trval_s{0,1,2}.yaml` |
| Results of the 3 submitted runs | `results/2026-11-13_final_full24_wide2_drop0.05_trval_s{0,1,2}_36d09c1.json` |
| Validation: the same recipe trained on `train` only, scored on `val` | `configs/host/host_full24_wide2_drop0.05_f1.0_s{0,1,2}.yaml`, `results/2026-10-30_host_full24_wide2_drop0.05_f1.0_s*.json`: 0.17949, 0.17950, 0.17955; 3-seed ensemble 0.17919 (`uv run python -m src.leaderboard`) |
| Write-up | `docs/experiments/final_submission.md` |
| Code | `src/models/host_net.py`, `src/features/host.py`, `src/retrain.py`, `src/submit.py`, `src/verify_submission.py` |

**To reproduce it, follow [`REPRODUCE.md`](REPRODUCE.md)**: setup, then one command with Modal
(`bash scripts/reproduce_final.sh modal`) or without it (`bash scripts/reproduce_final.sh local`),
ending in a check against the submitted CSV.

Why this recipe: it was the best single recipe on `val` among the full-data runs we tried, and
its last checkpoint is within about 0.0001 of its best, which matters because a model trained
on `val` has nothing left to pick a checkpoint with. Details in
`docs/experiments/final_submission.md`.

`src/models/` and `src/features/` still contain options that the final recipe does not use (an
MLP, other feature sets, extra HostNet switches). The configs above set everything the final
model uses.

## Setup

Packages are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).
`pyproject.toml` lists the dependencies, `uv.lock` pins exact versions, and
`.python-version` pins Python 3.13 (uv downloads it if needed).

1. Accept the competition rules at https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/rules, or downloads will fail.
2. Install dependencies into `.venv/` and log in to Kaggle:
   ```bash
   uv sync
   uv run kaggle auth login
   ```
3. Download the data (~3.5 GB zip, ~9.8 GB unzipped) into `data/`:
   ```bash
   uv run scripts/download_data.sh
   ```
4. Build the train / val / test splits (from the repo root):
   ```bash
   uv run python -m src.make_splits
   ```
   This writes `data/processed/{train,val,test}.parquet`. It should print
   `Recomputed split matches .../splits/game_splits.csv` and this table:

   | split | games | rows       | A_rate | B_rate | no_goal_rows |
   |-------|-------|------------|--------|--------|--------------|
   | train | 5893  | 16,992,436 | 0.0570 | 0.0552 | 0.2278       |
   | val   | 736   | 2,102,175  | 0.0595 | 0.0558 | 0.2180       |
   | test  | 736   | 2,103,425  | 0.0558 | 0.0559 | 0.2203       |

## Splits

- 80 / 10 / 10 by **game**: every frame of a game lands in the same split, since frames
  ~0.1 s apart are nearly identical and would otherwise leak between splits.
- The game → split mapping is committed in `splits/game_splits.csv` and is the source of
  truth. `make_splits` recomputes it (SHA-256 of each `game_num`, no RNG) and stops if
  it disagrees with the committed file.
- Our `test` split is a **labeled holdout from Kaggle's training data**. It is not
  `data/test.csv`, which is Kaggle's unlabeled test set, used only for submissions.

## Loading data

Run code with `uv run python ...` (or activate `.venv`); add packages with `uv add <pkg>`.

```python
from src.data import load_split, FEATURES, TARGETS

train = load_split("train")          # polars DataFrame (.to_pandas() needs pandas + pyarrow)
X, y = train[FEATURES], train[TARGETS]
groups = train["game_num"]           # use for grouped CV
```

Only use `FEATURES` as model inputs. The other columns (`game_num`, `event_id`,
`event_time`, `player_scoring_next`, `team_scoring_next`) are not in Kaggle's test set,
and `event_time` + `team_scoring_next` reproduce the targets exactly.
