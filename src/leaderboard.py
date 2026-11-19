"""Print every run in results/*.json as one table, best val score first.

    uv run python -m src.leaderboard                       # all runs
    uv run python -m src.leaderboard --train-frac 0.1      # only comparable runs
    uv run python -m src.leaderboard --name mlp256         # config name contains
"""
import argparse
import json

import polars as pl

from src.run import RESULTS_DIR


def load_results() -> pl.DataFrame:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        if "score" not in r:  # src.retrain runs (trained on train + val): no val score to rank
            continue
        m, info = r["config"]["model"], r.get("fit_info", {})
        rows.append({
            "run_id": r["run_id"],
            "name": r["config"]["name"],
            "train_frac": r["train_frac"],
            "games": r["n_train_games"],
            "model": m["type"] + (str(m["hidden"]) if "hidden" in m else ""),
            "features": "+".join(r["config"]["features"]),
            "seed": r["seed"],
            "score": r["score"],
            "logloss_A": r["logloss_A"],
            "logloss_B": r["logloss_B"],
            # diagnostics (src.metrics.diagnostics); missing in results written before them
            "acc": r.get("accuracy"),
            "auc": r.get("auc"),
            "ap": r.get("ap"),
            "best_step": info.get("best_step"),
            "stop": info.get("stop_reason"),
            "train_s": r["train_time_s"],
            "dirty": r["dirty"],
        })
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float)
    ap.add_argument("--name", help="substring of the config name")
    args = ap.parse_args()
    df = load_results()
    if args.train_frac is not None:
        df = df.filter(pl.col("train_frac") == args.train_frac)
    if args.name:
        df = df.filter(pl.col("name").str.contains(args.name, literal=True))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=60,
                   tbl_hide_dataframe_shape=True, float_precision=5):
        print(df.sort("score").drop("run_id"))


if __name__ == "__main__":
    main()
