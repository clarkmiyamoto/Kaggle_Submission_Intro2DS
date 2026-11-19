"""Split Kaggle's training games into our train / val / test sets.

Our `test` split is a *labeled* holdout carved out of Kaggle's train_*.csv.
It is NOT Kaggle's data/test.csv, which is unlabeled and only for submissions.

Whole games go to a single split, so near-identical frames from one game never
end up on both sides. Games are ordered by SHA-256 of f"{SALT}:{game_num}" and
cut at exact counts, which gives the same answer on every machine and library
version. The mapping is committed as splits/game_splits.csv; if that file
exists, this script recomputes the split and refuses to continue on mismatch.

Usage (from the repo root):  uv run python -m src.make_splits
"""
import hashlib

import polars as pl

from src.data import (FEATURES, META, PROCESSED_DIR, RAW_DIR, SPLITS,
                      SPLITS_CSV, TARGETS, raw_schema)

SALT = "tps-oct-2022"
VAL_FRAC = 0.10
TEST_FRAC = 0.10

SPLITS_SCHEMA = {"game_num": pl.Int32, "split": pl.String}


def assign_games(games: list[int]) -> pl.DataFrame:
    order = sorted(games, key=lambda g: hashlib.sha256(f"{SALT}:{g}".encode()).hexdigest())
    n_val = int(len(order) * VAL_FRAC)
    n_test = int(len(order) * TEST_FRAC)
    n_train = len(order) - n_val - n_test
    split = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return pl.DataFrame({"game_num": order, "split": split}, schema=SPLITS_SCHEMA).sort("game_num")


def main():
    schema = raw_schema("train")
    assert FEATURES == [c for c in raw_schema("test") if c != "id"], "FEATURES out of sync with test_dtypes.csv"
    assert set(schema) == set(FEATURES + META + TARGETS), "unexpected columns in train_dtypes.csv"

    files = sorted(RAW_DIR.glob("train_[0-9].csv"))
    assert len(files) == 10, f"expected train_0..9.csv in {RAW_DIR}, found {len(files)}; run `uv run scripts/download_data.sh`"
    raw = pl.scan_csv(files, schema_overrides=schema)

    games = raw.select(pl.col("game_num").unique()).collect()["game_num"].to_list()
    mapping = assign_games(games)
    if SPLITS_CSV.exists():
        committed = pl.read_csv(SPLITS_CSV, schema=SPLITS_SCHEMA)
        if not committed.equals(mapping):
            raise SystemExit(f"Recomputed split differs from {SPLITS_CSV}. "
                             "If you changed the split on purpose, delete that file and rerun.")
        print(f"Recomputed split matches {SPLITS_CSV}")
    else:
        SPLITS_CSV.parent.mkdir(exist_ok=True)
        mapping.write_csv(SPLITS_CSV)
        print(f"Wrote {SPLITS_CSV}")

    PROCESSED_DIR.mkdir(exist_ok=True)
    for name in SPLITS:
        split_games = mapping.filter(pl.col("split") == name)["game_num"].to_list()
        out = PROCESSED_DIR / f"{name}.parquet"
        (raw.filter(pl.col("game_num").is_in(split_games))
            .sort(["game_num", "event_id", "event_time"])
            .sink_parquet(out))
        print(f"Wrote {out}")

    # Sanity checks: compare these numbers with the table in README.md.
    game_sets = {n: set(pl.read_parquet(PROCESSED_DIR / f"{n}.parquet", columns=["game_num"])["game_num"].unique())
                 for n in SPLITS}
    assert not (game_sets["train"] & game_sets["val"] or game_sets["train"] & game_sets["test"]
                or game_sets["val"] & game_sets["test"]), "a game appears in more than one split"
    assert set().union(*game_sets.values()) == set(games), "some games are missing from the splits"

    summary = pl.concat([
        pl.scan_parquet(PROCESSED_DIR / f"{n}.parquet").select(
            pl.lit(n).alias("split"),
            pl.col("game_num").n_unique().alias("games"),
            pl.len().alias("rows"),
            pl.col(TARGETS[0]).mean().round(4).alias("A_rate"),
            pl.col(TARGETS[1]).mean().round(4).alias("B_rate"),
            pl.col("team_scoring_next").is_null().mean().round(4).alias("no_goal_rows"),
        ).collect()
        for n in SPLITS
    ])
    assert summary["rows"].sum() == raw.select(pl.len()).collect().item(), "row count mismatch with raw CSVs"
    with pl.Config(tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(summary)


if __name__ == "__main__":
    main()
