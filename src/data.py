"""Paths, column groups and loaders shared across the project."""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data"
PROCESSED_DIR = RAW_DIR / "processed"
SPLITS_CSV = ROOT / "splits" / "game_splits.csv"

SPLITS = ("train", "val", "test")

# Model inputs: exactly the columns of Kaggle's test.csv (minus `id`).
# make_splits.py asserts this matches test_dtypes.csv.
FEATURES = (
    [f"ball_{k}_{a}" for k in ("pos", "vel") for a in "xyz"]
    + [f"p{p}_{c}" for p in range(6) for c in
       [f"{k}_{a}" for k in ("pos", "vel") for a in "xyz"] + ["boost"]]
    + [f"boost{i}_timer" for i in range(6)]
)

TARGETS = ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]

# Train-only columns. Never use as model inputs: event_time + team_scoring_next
# reproduce the targets exactly, and none of these exist in Kaggle's test.csv.
# game_num is kept for grouped cross-validation.
META = ["game_num", "event_id", "event_time", "player_scoring_next", "team_scoring_next"]


def raw_schema(name: str) -> dict:
    """Polars schema from Kaggle's {name}_dtypes.csv (float16 widened to float32)."""
    d = pl.read_csv(RAW_DIR / f"{name}_dtypes.csv")
    m = {"int8": pl.Int8, "int32": pl.Int32, "int64": pl.Int64,
         "float16": pl.Float32, "float32": pl.Float32, "object": pl.String}
    return {c: m[t] for c, t in zip(d["column"], d["dtype"])}


def load_split(name: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Load one of our splits ('train', 'val', 'test'). Use .to_pandas() if needed."""
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {name!r}")
    return pl.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns)


def load_kaggle_test() -> pl.DataFrame:
    """Kaggle's unlabeled test.csv, used only for making submissions."""
    return pl.read_csv(RAW_DIR / "test.csv", schema_overrides=raw_schema("test"))
