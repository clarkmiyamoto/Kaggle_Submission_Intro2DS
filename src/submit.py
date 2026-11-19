"""A Kaggle submission CSV from saved predictions: one run, or the equal-weight mean of several.

    uv run python -m src.submit --name final_trval_3seeds RUN_ID [RUN_ID ...]
    uv run kaggle competitions submit -c tabular-playground-series-oct-2022 \\
        -f submissions/final_trval_3seeds.csv -m "final_trval_3seeds"

Each RUN_ID is a directory under runs/ holding preds_kaggle.parquet (`id` + both targets).
Several runs are averaged in probability space after joining on `id`. The output follows
data/sample_submission.csv: the same ids in the same order and the same column names. It is
checked to have no missing ids, no nulls and probabilities strictly inside (0, 1).
Writes submissions/<name>.csv (gitignored).
"""
import argparse

import numpy as np
import polars as pl

from src.data import RAW_DIR, ROOT, TARGETS
from src.run import RUNS_DIR

OUT_DIR = ROOT / "submissions"
EPS = 1e-7  # the scorer's clip; keep submitted probabilities inside it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run ids (directories under runs/) with preds_kaggle.parquet")
    ap.add_argument("--name", required=True, help="output file stem")
    a = ap.parse_args()

    sample = pl.read_csv(RAW_DIR / "sample_submission.csv")
    assert sample.columns == ["id"] + TARGETS, sample.columns
    ps = []
    for run in a.runs:
        df = pl.read_parquet(RUNS_DIR / run / "preds_kaggle.parquet")
        assert df.columns == ["id"] + TARGETS and df["id"].n_unique() == len(df) == len(sample), (run, df.shape)
        joined = sample.select("id").join(df, on="id", how="left")
        assert joined.null_count().sum_horizontal().item() == 0, f"{run}: ids missing from the predictions"
        ps.append(joined.select(TARGETS).to_numpy().astype(np.float64))
        print(f"  {run}: mean p A {ps[-1][:, 0].mean():.4f}, B {ps[-1][:, 1].mean():.4f}")
    p = np.clip(np.mean(ps, axis=0), EPS, 1 - EPS)
    out = sample.select("id").hstack(pl.DataFrame({t: p[:, i] for i, t in enumerate(TARGETS)}))
    assert out.height == sample.height and out["id"].equals(sample["id"])
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{a.name}.csv"
    out.write_csv(path)
    print(f"wrote {path}: {out.height:,} rows, mean of {len(ps)} run(s); mean p A {p[:, 0].mean():.4f}, "
          f"B {p[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
