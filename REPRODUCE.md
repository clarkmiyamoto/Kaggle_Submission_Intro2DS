# Reproducing the final submission

This page rebuilds `submissions/final_full24_wide2_drop0.05_trval_3seeds.csv`, the CSV
submitted to Kaggle (**public 0.18296, private 0.17610**), from the raw competition data. It
covers two paths: on Modal GPUs, and on your own machine without Modal. Every step is a
command; the last one says whether the result matches.

## The final method

| | |
| --- | --- |
| Model | HostNet (`src/models/host_net.py`, `type: host_net`), a port of the competition host's net: a ball MLP, per-player and per-pair 1x1 convs, max-pooling per team, a dense head. Opponent pairing fixed (`opponent_pairs: fixed`), every layer 2x wide (`width: 2.0`), dropout 0.05 |
| Inputs | `features: [host]` (`src/features/host.py`): raw ball and player positions, velocities, boost, a demolished flag per player. No scaling. Every input comes from a single row |
| Objective | A 3-class softmax (A scores / nobody / B scores) at horizons 1..10 s; the 10 s head is the prediction, the 1..9 s heads are auxiliary (`aux_weight: 1.0`) |
| Training | Adam (eps 1e-7), batch 4096, bf16 autocast, LR 0.01 stepped down to 0.00025 over 24 passes, mirror augmentation. Last checkpoint, no early stopping |
| Data | Every game of our `train` and `val` splits (6,629 games, 19,094,611 rows). Never our `test` split |
| Inference | Mean over the 4 mirror combinations (`tta: flips`) |
| Ensemble | Seeds 0, 1, 2, probabilities averaged with equal weights |
| Configs | `configs/final/final_full24_wide2_drop0.05_trval_s{0,1,2}.yaml` |
| Code | `src/retrain.py` (train on train + val, predict Kaggle's test), `src/submit.py` (average into a CSV), `src/verify_submission.py` (compare with the submitted CSV) |

Evidence that the recipe works, from the same recipe trained on `train` only and scored on
`val` (`configs/host/host_full24_wide2_drop0.05_f1.0_s{0,1,2}.yaml`, see
`uv run python -m src.leaderboard`): 0.17949, 0.17950, 0.17955 per seed, 0.17919 for
the 3-seed ensemble. Details: `docs/experiments/final_submission.md`.

## 0. Setup (both paths)

You need git, [uv](https://docs.astral.sh/uv/getting-started/installation/), a Kaggle account
that has accepted the [competition rules](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/rules),
and about 15 GB of free disk.

```bash
git clone <this repo> && cd <this repo>      # a git checkout: run ids carry the commit SHA
uv sync                                      # Python 3.13 and pinned packages (uv.lock)
uv run kaggle auth login                     # or put a token in ~/.kaggle/kaggle.json
bash scripts/download_data.sh                # ~3.5 GB zip -> data/ (~9.8 GB of CSVs)
uv run python -m src.make_splits             # data/processed/{train,val,test}.parquet
```

`make_splits` must print `Recomputed split matches .../splits/game_splits.csv` and the split
table in `README.md` (train 5,893 games, val 736, test 736). If it stops with a mismatch,
the data differs from what the submission used: do not continue.

If the data is already on disk, link it instead of downloading: `ln -s /path/to/data data`
(the folder holding `train_0.csv` ... `test.csv`, `sample_submission.csv`, `*_dtypes.csv`).

**Smoke test first** (about 10 min on an M4 Max): trains each seed on 20 games per split,
then runs the averaging and comparison steps.

```bash
bash scripts/reproduce_final.sh smoke
```

It should end with `FAIL` (20-game models are far from the real ones) followed by
`smoke: FAIL is expected from 20-game models; the pipeline ran end to end`. Any error
before that line is a setup problem.

## Path A: with Modal

Closest to the original: the submitted runs trained on Modal `A10G` GPUs with this code.

```bash
uv run modal setup                           # once: log in
bash scripts/modal_upload_data.sh            # once: data/ -> the kaggle-rl-data volume (~3.4 GB)
git status                                   # must be clean: Modal runs the committed HEAD
bash scripts/reproduce_final.sh modal        # 3 seeds in parallel on A10G, then fetch, CSV, check
```

- **Time and cost:** about 20 min per seed, in parallel; about $2.3 in all (Nov 2026 prices).
- **GPU type matters.** Keep the default `A10G`. On one GPU type a compiled run is close to
  reproducible; across types (even A10 vs A10G) runs differ by about seed noise.
  `reproduce_final.sh modal <GPU>` picks another.
- `reproduce_final.sh modal` keeps the terminal attached until the runs finish; closing it
  stops them. To survive a closed terminal, run the steps by hand:

```bash
uv run modal run --detach scripts/modal_train.py::main --gpu A10G --module src.retrain \
    --configs configs/final/final_full24_wide2_drop0.05_trval_s0.yaml,configs/final/final_full24_wide2_drop0.05_trval_s1.yaml,configs/final/final_full24_wide2_drop0.05_trval_s2.yaml
# when the 3 runs are done (logs: uv run modal volume get kaggle-rl-outputs logs/<config name>.log -):
uv run modal run scripts/modal_train.py::fetch --name final_full24_wide2_drop0.05_trval
```

then continue with "Build and check the CSV" below.

## Path B: without Modal

The same script trains the 3 seeds one after another on this machine. `device: auto` picks
CUDA, then Apple MPS, then CPU. `compile: reduce-overhead` applies on CUDA only.

| Machine | Command | Time per seed |
| --- | --- | --- |
| NVIDIA GPU (Linux) | `bash scripts/reproduce_final.sh local` | ~20 min on an A10G; scales with the GPU |
| Apple silicon (MPS) | `bash scripts/reproduce_final.sh local --max-hours none` | ~2 h on an M4 Max (measured on 400 games and scaled to 19.1M rows); ~6 h for 3 seeds |
| CPU only | `bash scripts/reproduce_final.sh local --device cpu --max-hours none` | not measured; MPS is 5-10x faster than CPU on an M4 Max, so expect 10-20 h |

- **`--max-hours none` is needed off CUDA.** The configs cap a run at 3 hours (sized for a
  GPU), and `src/retrain.py` refuses a run that stopped before its 24 passes, so a slow
  device would fail after 3 hours. `--device` and `--max-hours` change only where and how
  long it may run, not the recipe, and are recorded under `overrides` in the result JSON.
- **Memory:** the full train + val matrix is 19.1M rows x 54 float32 inputs (4.1 GB), plus a
  copy on the device. The Modal containers had 24 GB of RAM; plan for at least that.
- One seed by hand: `uv run python -m src.retrain configs/final/final_full24_wide2_drop0.05_trval_s0.yaml [--device ...] [--max-hours none]`.
  It writes `runs/<date>_final_full24_wide2_drop0.05_trval_s0_<sha>/preds_kaggle.parquet` and
  `results/<same run id>.json`.

## Build and check the CSV

`reproduce_final.sh` does this itself and writes `submissions/repro_<mode>_<sha>.csv`. By
hand, after the 3 seeds have finished:

```bash
sha=$(git rev-parse --short HEAD)
RUNS=$(for s in 0 1 2; do ls runs | grep -E "_final_full24_wide2_drop0.05_trval_s${s}_${sha}$" | sort | tail -1; done)
uv run python -m src.submit --name repro_$sha $RUNS          # -> submissions/repro_<sha>.csv
uv run python -m src.verify_submission submissions/repro_$sha.csv
```

`src.verify_submission` compares the new CSV with the submitted one on all 701,143 rows of
Kaggle's test set and prints one verdict:

| Verdict | Meaning | Expected from |
| --- | --- | --- |
| `IDENTICAL` | every probability within 1e-6 | possible only on the same GPU type and software (Modal A10G) |
| `MATCH` | as close as the train-only 3-seed ensemble that shares the seeds (logit corr >= 0.993) | a rerun whose runs stayed close to the originals |
| `PASS` | within the gap between two seeds of the recipe (logit corr >= 0.985) | a rerun on other hardware |
| `FAIL` | further apart than two seeds | a different setup: data, features, schedule, or a run that did not finish |

`IDENTICAL`, `MATCH` and `PASS` all mean the method is reproduced; they exit 0. The
thresholds and the measurements behind them are in the docstring of
`src/verify_submission.py`.

The definitive check is Kaggle's scorer (late submissions are still scored):

```bash
uv run kaggle competitions submit -c tabular-playground-series-oct-2022 \
    -f submissions/<the reproduced CSV> -m "reproduction of final_full24_wide2_drop0.05_trval_3seeds"
```

A rerun that passes should score close to public 0.18296 / private 0.17610: single seeds of
the recipe differ by under 1e-4 on `val`, and the public leaderboard, a smaller sample, is
noisier.

## Optional: re-derive the val evidence

The val numbers above come from the train-only runs. To rerun one (about 22 min on an A10;
on Modal, `scripts/modal_train.py::main` without `--module` runs `src.run`):

```bash
uv run python -m src.run configs/host/host_full24_wide2_drop0.05_f1.0_s0.yaml
uv run python -m src.leaderboard             # the rerun next to the committed 0.17949
```

`src.blend` needs one result per seed: to score a rerun 3-seed ensemble
(`uv run python -m src.blend host_full24_wide2_drop0.05_f1.0`), move the committed
`results/2026-10-30_host_full24_wide2_drop0.05_f1.0_s*.json` out of `results/` first.
