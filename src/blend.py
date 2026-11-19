"""Equal-weight ensembles of saved predictions, scored on the full val.

    uv run python -m src.blend host_fast_auxw1_f1.0
    uv run python -m src.blend host_fast_auxw1_f1.0 host_fast_auxw1_wide2_f1.0 --seeds 0,1,2

Each argument is a group, as in src.compare: every result named `<prefix>_s<seed>`. The
ensemble averages the members' probabilities (`--mean prob`, the default) or log-odds
(`--mean logit`) with equal weights, so nothing is fitted on val. It reports the ensemble's
val score with src.metrics.score, its paired-bootstrap difference (over val games) from the
best member, and its difference from the members' mean score. `--out NAME` also writes
runs/NAME/preds_val.parquet and, when every member has one, runs/NAME/preds_kaggle.parquet,
in the members' row order, for a submission.

An ensemble of k seeds costs k runs: compare it with the single runs as a recipe, not as
one more seed.
"""
import argparse

import numpy as np
import polars as pl

from src.compare import load_group, preds, preds_path
from src.data import TARGETS, load_split
from src.metrics import paired_bootstrap, score
from src.run import RUNS_DIR


def combine(ps: list[np.ndarray], mean: str) -> np.ndarray:
    if mean == "prob":
        return np.mean(ps, axis=0)
    z = np.mean([np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)) for p in ps], axis=0)
    return 1 / (1 + np.exp(-z))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+", help="config-name prefixes; every seed of each is a member")
    ap.add_argument("--seeds", help="comma-separated seeds to keep")
    ap.add_argument("--mean", choices=["prob", "logit"], default="prob")
    ap.add_argument("--out", help="write the ensemble's predictions to runs/OUT/")
    args = ap.parse_args()
    keep = None if args.seeds is None else {int(s) for s in args.seeds.split(",")}
    members = [r for g in args.groups for s, r in sorted(load_group(g).items()) if keep is None or s in keep]
    fracs = {r["train_frac"] for r in members}
    assert len(fracs) == 1, f"train_frac differs between members: {fracs}"
    if any(r["dirty"] for r in members):
        print("WARNING: some members come from a dirty tree")

    y = load_split("val", columns=TARGETS).to_numpy().astype(np.float64)
    games = load_split("val", columns=["game_num"])["game_num"].to_numpy()
    ps = [preds(r).astype(np.float64) for r in members]
    ens = combine(ps, args.mean)
    scores = [score(y, p)["score"] for p in ps]
    print(f"train_frac {fracs.pop()}, {len(members)} members, {args.mean} mean")
    for r, s in zip(members, scores):
        print(f"  {r['config']['name']:>40} {s:.5f}  ({r['run_id']})")
    s = score(y, ens)
    best = int(np.argmin(scores))
    d, lo, hi = paired_bootstrap(y, ens, ps[best], games)
    print(f"ensemble {s['score']:.5f}  (A {s['logloss_A']:.5f}, B {s['logloss_B']:.5f})")
    print(f"  vs best member  ({scores[best]:.5f}): {d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    print(f"  vs member mean  ({np.mean(scores):.5f}): {s['score'] - np.mean(scores):+.5f}")

    if args.out:
        out = RUNS_DIR / args.out
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({t: ens[:, i].astype(np.float32) for i, t in enumerate(TARGETS)}).write_parquet(out / "preds_val.parquet")
        kaggle = [preds_path(r).with_name("preds_kaggle.parquet") for r in members]
        if all(k.exists() for k in kaggle):
            frames = [pl.read_parquet(k) for k in kaggle]
            assert all(f["id"].equals(frames[0]["id"]) for f in frames), "Kaggle row order differs"
            pk = combine([f.select(TARGETS).to_numpy().astype(np.float64) for f in frames], args.mean)
            frames[0].select("id").hstack(pl.DataFrame({t: pk[:, i].astype(np.float32)
                                                        for i, t in enumerate(TARGETS)})).write_parquet(out / "preds_kaggle.parquet")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
