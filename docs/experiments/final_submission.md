# Final submission: HostNet retrained on train + val (Nov 2026)

**Kaggle: public 0.18296, private 0.17610** (submission 58319096,
`final_full24_wide2_drop0.05_trval_3seeds.csv`).

## What was submitted

- **Recipe:** `host_full24_wide2_drop0.05`, the best full-data single recipe on val among the
  ones we tried: trained on `train` only, it scored 0.17949, 0.17950 and 0.17955 on `val`
  (seeds 0-2, mean 0.17951), and 0.17919 as a 3-seed ensemble
  (`configs/host/host_full24_wide2_drop0.05_f1.0_s*.yaml`, `results/2026-10-30_*`).
  - The host net with the opponent pairing fixed and the 1..10 s heads at weight 1.
  - Every layer 2x wide, dropout 0.05, bf16, batch 4096, LR 0.01 stepped down over 24 passes.
  - The 4-mirror TTA.
- **Training data:** every game of `train` and `val` (6,629 games, 19.1M rows). Our labeled
  `test` split was not used for training and was not scored.
- **Members:** 3 seeds (0, 1, 2), probabilities averaged with equal weights.
- **Why this recipe:**
  - With val in the training set, nothing can choose a checkpoint, so each run keeps its last
    one after the full 24 passes.
  - This recipe's last checkpoint is within +0.00002 to +0.00011 of its best on full data, so
    keeping the last one costs almost nothing.
  - Recipes that need val to pick checkpoints (snapshot ensembles) were left out.
  - It is not the lowest val number we measured: a 7-run ensemble of wide nets (this recipe plus
    two related variants) scored 0.17905, but it was not retrained on train + val or
    submitted.

## How

To reproduce it from scratch, with or without Modal, follow `REPRODUCE.md` at the repo root.

```bash
uv run modal run --detach scripts/modal_train.py::main --gpu A10 --module src.retrain \
    --configs configs/final/final_full24_wide2_drop0.05_trval_s0.yaml,...s1.yaml,...s2.yaml
uv run python -m src.submit --name final_full24_wide2_drop0.05_trval_3seeds <the 3 run ids>
uv run kaggle competitions submit -c tabular-playground-series-oct-2022 -f submissions/<name>.csv -m "..."
```

- **`src/retrain.py`** trains on train + val for the fixed schedule. It evaluates once, at the
  end (`eval_every_epochs` = `max_epochs`), so the kept checkpoint is the last one, and it
  asserts the run trained all 24 passes. It writes Kaggle predictions and a result JSON with
  `trained_on: [train, val]` and no val score; `src.leaderboard` skips such files.
- **`src/submit.py`** averages the runs' Kaggle predictions after joining on `id`. It checks
  the result against `sample_submission.csv`: same ids in the same order, no nulls,
  probabilities strictly inside (0, 1).
- **Cost:** about 20 min of training per run on A10 / A10G, about $2.3 in all.

## Checks before submitting

There is no val score for a model trained on val, so each new run was compared with the 3
train-only runs of the same recipe on Kaggle's test rows.

| | mean p (A, B) | logit correlation | KL |
| --- | --- | --- | --- |
| train + val runs, seeds 0-2 | 0.0559-0.0560, 0.0557-0.0558 | 0.985-0.987 among themselves | 0.00095-0.00106 |
| train-only runs, seeds 0-2 | 0.0562-0.0565, 0.0560-0.0563 | 0.984-0.987 among themselves | 0.00093-0.00100 |
| train + val vs train-only | | 0.983-0.987 | 0.00094-0.00113 |

Each retrained run differs from the train-only runs about as much as two seeds differ from
each other, so the retrain behaves like more seeds of the same model.

## Not done

- **Scoring `test`:** the retrained model was not scored on our labeled `test` split, which is
  kept for when it is asked for.
- **A comparison submission** of the existing train-only ensemble, which would measure on the
  leaderboard what retraining on val is worth (estimated at about 0.0001).
