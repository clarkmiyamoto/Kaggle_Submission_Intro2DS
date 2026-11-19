"""Run one experiment: config -> features -> preprocess (fit on train) -> model -> preds -> score.

    uv run python -m src.run configs/<name>.yaml

Writes runs/<run_id>/preds_<set>.parquet and results/<run_id>.json, where
run_id = <date>_<config name>_<short sha>.
"""
import argparse
import datetime as dt
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from src.data import (META, PROCESSED_DIR, ROOT, SPLITS, SPLITS_CSV, TARGETS,
                      load_kaggle_test, load_split)
from src.features import FEATURE_SETS
from src.metrics import diagnostics, score
from src.models import MODELS
from src.preprocess import Preprocessor

RESULTS_DIR = ROOT / "results"
RUNS_DIR = ROOT / "runs"
N_VAL_ROWS = 2_102_175


def subsample_games(games: list[int], frac: float) -> list[int]:
    """Same games for everyone, and a 10% sample sits inside the 20% one."""
    order = sorted(games, key=lambda g: hashlib.sha256(f"train-frac:{g}".encode()).hexdigest())
    return order[: round(len(order) * frac)]


def git_state() -> tuple[str, bool]:
    """Short SHA, and whether tracked code differs from it (results/ and runs/ don't count)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                    check=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD"), bool(
        git("status", "--porcelain", "--", ".", ":!results", ":!runs"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_inputs(lf: pl.LazyFrame, feature_sets: list[str]) -> tuple[pl.LazyFrame, list[str]]:
    cols = []
    for name in feature_sets:
        add, provides = FEATURE_SETS[name]
        lf = add(lf)
        cols += [c for c in provides if c not in cols]
    bad = set(cols) & set(META + TARGETS)
    assert not bad, f"META / TARGETS columns used as inputs: {sorted(bad)}"
    return lf, cols


def scan(split: str) -> pl.LazyFrame:
    if split == "kaggle":
        return load_kaggle_test().lazy()  # a CSV: parsed in full on each call
    assert split in SPLITS, split
    return pl.scan_parquet(PROCESSED_DIR / f"{split}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    cfg_path = ap.parse_args().config
    cfg = yaml.safe_load(cfg_path.read_text())

    sha, dirty = git_state()
    run_id = f"{dt.date.today().isoformat()}_{cfg['name']}_{sha}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}{'  (DIRTY TREE)' if dirty else ''}", flush=True)
    seed_everything(int(cfg.get("seed", 0)))

    # 1. train cut to train_frac of its games (by hash), full val
    frac = float(cfg.get("train_frac", 1.0))
    splits = pl.read_csv(SPLITS_CSV)
    games = subsample_games(splits.filter(pl.col("split") == "train")["game_num"].to_list(), frac)

    # 2. feature sets; no META column is ever an input. A model may declare TRAIN_LABELS:
    # META columns it gets for train rows as extra training labels (e.g. host_net's 1-9 s
    # horizons), passed to fit as `labels_tr`, never as inputs
    model_cfg = dict(cfg["model"])
    model_cls = MODELS[model_cfg.pop("type")]
    labels = list(getattr(model_cls, "TRAIN_LABELS", []))
    assert set(labels) <= set(META), f"TRAIN_LABELS must be META columns: {labels}"
    lf_tr, inputs = build_inputs(scan("train").filter(pl.col("game_num").is_in(games)), cfg["features"])
    train = lf_tr.select(inputs + TARGETS + labels).collect()
    print(f"train: {len(games)} games, {len(train):,} rows, {len(inputs)} inputs", flush=True)

    # 3. preprocessing fit on train only
    pre = Preprocessor(**cfg.get("preprocess", {})).fit(train.lazy(), inputs)
    X_tr = pre.transform(train.lazy())
    y_tr = train.select(TARGETS).to_numpy().astype(np.float32)
    extra = {"labels_tr": train.select(labels)} if labels else {}
    del train
    lf_val, _ = build_inputs(scan("val"), cfg["features"])
    X_val = pre.transform(lf_val)
    y_val = load_split("val", columns=TARGETS).to_numpy().astype(np.float32)
    assert len(X_val) == len(y_val) == N_VAL_ROWS

    # 4. fit; val is only for early stopping
    model = model_cls(**model_cfg)
    t0 = time.time()
    model.fit(X_tr, y_tr, X_val, y_val, **extra)
    train_time = time.time() - t0

    # 5. predictions, in load_split / Kaggle row order
    preds = {}
    for split in dict.fromkeys(["val"] + list(cfg.get("predict", ["val"]))):
        if split == "val":
            X = X_val
        else:
            frame = scan(split).collect()
            X = pre.transform(build_inputs(frame.lazy(), cfg["features"])[0])
        p = model.predict_proba(X)
        assert p.shape == (len(X), 2)
        df = pl.DataFrame({t: p[:, i] for i, t in enumerate(TARGETS)})
        if split == "kaggle":
            df = frame.select("id").hstack(df)
        df.write_parquet(run_dir / f"preds_{split}.parquet")
        preds[split] = p

    # 6. score the full val with the one scorer, log the run; diagnostics are secondary
    s = score(y_val, preds["val"])
    diag = diagnostics(y_val, preds["val"])
    result = {
        "run_id": run_id,
        "config_path": str(cfg_path),
        "config": cfg,
        "git_sha": sha,
        "dirty": dirty,
        "train_frac": frac,
        "seed": int(cfg.get("seed", 0)),
        "n_train_games": len(games),
        "n_train_rows": len(X_tr),
        "inputs": len(inputs),
        **s,
        **diag,
        "train_time_s": round(train_time, 1),
        "fit_info": getattr(model, "fit_info", {}),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{run_id}.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"val score {s['score']:.5f}  (A {s['logloss_A']:.5f}, B {s['logloss_B']:.5f})"
          f"  train {train_time:.0f}s  -> results/{run_id}.json", flush=True)
    print(f"  accuracy {diag['accuracy']:.4f} (all 'no goal' {diag['accuracy_majority']:.4f})"
          f"  auc {diag['auc']:.4f}  ap {diag['ap']:.4f}", flush=True)


if __name__ == "__main__":
    main()
