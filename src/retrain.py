"""Retrain a finished recipe on train + val for a fixed schedule, for a Kaggle submission.

    uv run python -m src.retrain configs/final/<name>.yaml
    uv run python -m src.retrain configs/final/<name>.yaml --limit-games 20   # local smoke test
    uv run python -m src.retrain configs/final/<name>.yaml --device cpu --max-hours none

`--device` and `--max-hours` override the config's `device` and `max_hours` for the hardware
at hand; the recipe (data, schedule, model) is unchanged and the overrides are recorded in the
result JSON. The final configs set `max_hours: 3.0`, sized for a CUDA GPU (~20 min per run on an
A10G). A slower device would hit that cap before the 24 passes and fail the full-schedule
assertion, so pass `--max-hours none` on CPU or on a slow Mac.

What differs from src.run:
- Trains on every game of `train` and `val` (6,629). Never `test`, which stays the labeled
  holdout.
- Val is in the training set, so it can't choose a checkpoint or report a score. The config
  must evaluate only once, at the end (`eval_every_epochs` == `max_epochs`), so the kept
  checkpoint is the last one. The script asserts the run trained the full schedule. Use only
  recipes whose last checkpoint is measured to be close to their best
  (`host_full24_wide2_drop0.05`: +0.00002 to +0.00011 on full data).
- Writes only runs/<run_id>/preds_kaggle.parquet, plus results/<run_id>.json with
  `trained_on: [train, val]` and no `score`. src.leaderboard skips such files. No val
  predictions are written, since they would be in-sample.
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from src.data import SPLITS_CSV, TARGETS
from src.models import MODELS
from src.preprocess import Preprocessor
from src.run import RESULTS_DIR, RUNS_DIR, build_inputs, git_state, scan, seed_everything

TRAIN_ON = ["train", "val"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--limit-games", type=int, help="smoke test: only this many games per split")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], help="override model.device")
    ap.add_argument("--max-hours", help="override model.max_hours: hours, or `none` for no time cap")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    model_cfg = dict(cfg["model"])
    overrides = {}
    if a.device:
        overrides["device"] = a.device
    if a.max_hours:
        overrides["max_hours"] = None if a.max_hours.lower() == "none" else float(a.max_hours)
    model_cfg.update(overrides)
    assert float(model_cfg["eval_every_epochs"]) == float(model_cfg["max_epochs"]), \
        "retrain keeps the last checkpoint: set eval_every_epochs == max_epochs"
    assert cfg.get("predict") == ["kaggle"], "retrain predicts Kaggle's test set only"

    sha, dirty = git_state()
    tag = "_smoke" if a.limit_games else ""
    run_id = f"{dt.date.today().isoformat()}_{cfg['name']}{tag}_{sha}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"retrain {run_id} on {TRAIN_ON}{'  (DIRTY TREE)' if dirty else ''}"
          f"{f'  overrides {overrides}' if overrides else ''}", flush=True)
    seed_everything(int(cfg.get("seed", 0)))

    model_cls = MODELS[model_cfg.pop("type")]
    labels = list(getattr(model_cls, "TRAIN_LABELS", []))
    splits = pl.read_csv(SPLITS_CSV)
    frames, n_games = [], 0
    for split in TRAIN_ON:
        games = sorted(splits.filter(pl.col("split") == split)["game_num"].to_list())
        if a.limit_games:
            games = games[:a.limit_games]
        n_games += len(games)
        lf, inputs = build_inputs(scan(split).filter(pl.col("game_num").is_in(games)), cfg["features"])
        frames.append(lf.select(inputs + TARGETS + labels).collect())
    train = pl.concat(frames)
    print(f"train + val: {n_games} games, {len(train):,} rows, {len(inputs)} inputs", flush=True)

    pre = Preprocessor(**cfg.get("preprocess", {})).fit(train.lazy(), inputs)
    X = pre.transform(train.lazy())
    y = train.select(TARGETS).to_numpy().astype(np.float32)
    extra = {"labels_tr": train.select(labels)} if labels else {}
    del train
    # fit() needs a val set for its (single, final) eval; give it a slice of its own training rows,
    # labelled as what it is: the number it prints is in-sample and is not reported
    n_eval = min(len(X), 200_000)
    model = model_cls(**model_cfg)
    t0 = time.time()
    model.fit(X, y, X[:n_eval], y[:n_eval], **extra)
    train_time = time.time() - t0
    fi = model.fit_info
    assert fi["stop_reason"] == "max_epochs" and fi["stopped_step"] == fi["best_step"], \
        f"did not train the full schedule or kept a non-final checkpoint: {fi['stop_reason']}, " \
        f"best {fi['best_step']} vs stopped {fi['stopped_step']}"

    frame = scan("kaggle").collect()
    p = model.predict_proba(pre.transform(build_inputs(frame.lazy(), cfg["features"])[0]))
    assert p.shape == (len(frame), 2) and np.isfinite(p).all()
    frame.select("id").hstack(pl.DataFrame({t: p[:, i] for i, t in enumerate(TARGETS)})).write_parquet(
        run_dir / "preds_kaggle.parquet")

    fi = {k: v for k, v in fi.items() if k != "history"}
    fi["in_sample_final_eval"] = fi.pop("best_val_loss_no_tta")  # on training rows: not a score
    result = {
        "run_id": run_id,
        "config_path": str(a.config),
        "config": cfg,
        "git_sha": sha,
        "dirty": dirty,
        "trained_on": TRAIN_ON,
        "limit_games": a.limit_games,
        "overrides": overrides,
        "seed": int(cfg.get("seed", 0)),
        "n_train_games": n_games,
        "n_train_rows": len(X),
        "inputs": len(inputs),
        "train_time_s": round(train_time, 1),
        "kaggle_mean_p": [float(p[:, 0].mean()), float(p[:, 1].mean())],
        "fit_info": fi,
    }
    if not a.limit_games:
        (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps(result, indent=1))
    else:
        (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(f"wrote {run_dir / 'preds_kaggle.parquet'}; Kaggle mean p {result['kaggle_mean_p']}", flush=True)


if __name__ == "__main__":
    main()
