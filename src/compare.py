"""Compare groups of runs against a base group, paired by seed, with the paired bootstrap
over val games.

    uv run python -m src.compare BASE NEW [NEW ...]
    uv run python -m src.compare goalabl_mlp64_raw goalabl_mlp64_goal

A group is every result whose config name is `<prefix>_s<seed>`. All groups must have the
same train_frac and the same seeds; run sK of each NEW is paired with run sK of BASE. With
several NEW groups, a summary table follows the per-group blocks.

`--seeds 0` keeps only those seeds, for single-seed explorations. With one seed the verdict
covers val-game sampling only, not seed noise: treat it as a screen, not a claim.

Verdict (fixed rule):
  better  every seed pair's 95% interval is below 0 AND NEW's worst seed beats BASE's best
  worse   the mirror image
  no detectable difference otherwise
The bootstrap resamples val games only; the seed ranges cover initialisation noise.
"""
import argparse
import json
import re

import numpy as np
import polars as pl

from src.data import TARGETS, load_split
from src.metrics import paired_bootstrap
from src.run import RESULTS_DIR, RUNS_DIR


def load_group(prefix: str) -> dict[int, dict]:
    """seed -> result dict for every result named `<prefix>_s<seed>`."""
    pattern = re.compile(rf"{re.escape(prefix)}_s(\d+)")
    group: dict[int, dict] = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        m = pattern.fullmatch(r["config"]["name"])
        if not m:
            continue
        seed = int(m[1])
        if seed in group:
            raise SystemExit(f"two results for {r['config']['name']}: {group[seed]['run_id']} and "
                             f"{r['run_id']}; move one out of results/ first")
        group[seed] = r
    if not group:
        raise SystemExit(f"no results named {prefix}_s<seed> in {RESULTS_DIR}")
    return group


def preds_path(r: dict):
    path = RUNS_DIR / r["run_id"] / "preds_val.parquet"
    if not path.exists():
        raise SystemExit(f"preds not found: {path}. The run was made in another checkout; "
                         "copy its runs/<run_id>/ here or rerun its config.")
    return path


def preds(r: dict) -> np.ndarray:
    return pl.read_parquet(preds_path(r)).select(TARGETS).to_numpy()


def compare(base_name, base, new_name, new, y, games) -> str:
    """Print one paired comparison block and return the verdict."""
    print(f"\n{'seed':>4} {base_name:>24} {new_name:>24}   diff (new - base) [95% CI]")
    cis = []
    for seed in sorted(base):
        d, lo, hi = paired_bootstrap(y, preds(new[seed]), preds(base[seed]), games)
        cis.append((lo, hi))
        print(f"{seed:>4} {base[seed]['score']:>24.5f} {new[seed]['score']:>24.5f}   "
              f"{d:+.5f} [{lo:+.5f}, {hi:+.5f}]")
    sb = np.array([r["score"] for r in base.values()])
    sn = np.array([r["score"] for r in new.values()])
    print(f"mean {sb.mean():>24.5f} {sn.mean():>24.5f}   {sn.mean() - sb.mean():+.5f}")
    print(f"range {f'[{sb.min():.5f}, {sb.max():.5f}]':>23} {f'[{sn.min():.5f}, {sn.max():.5f}]':>24}")
    if all(hi < 0 for _, hi in cis) and sn.max() < sb.min():
        verdict = "better"
    elif all(lo > 0 for lo, _ in cis) and sn.min() > sb.max():
        verdict = "worse"
    else:
        verdict = "no detectable difference"
    print(f"verdict: {new_name} {verdict}" if verdict != "no detectable difference" else f"verdict: {verdict}")
    # diagnostics: seed means, for context only (the verdict is on log loss)
    rows = [(label, [[r.get(key) for r in g.values()] for g in (base, new)])
            for label, key in (("acc", "accuracy"), ("auc", "auc"), ("ap", "ap"))]
    rows = [(label, vb, vn) for label, (vb, vn) in rows if None not in vb + vn]
    if rows:
        print("diagnostics, seed means (context only; the verdict is on log loss):")
    for label, vb, vn in rows:
        print(f"{label:>4} {np.mean(vb):>24.4f} {np.mean(vn):>24.4f}   {np.mean(vn) - np.mean(vb):+.4f}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="config-name prefix of the reference group")
    ap.add_argument("new", nargs="+", help="config-name prefix(es) of the groups being tested")
    ap.add_argument("--seeds", help="comma-separated seeds to keep, e.g. 0 for single-seed runs")
    args = ap.parse_args()
    keep = None if args.seeds is None else {int(s) for s in args.seeds.split(",")}
    pick = lambda g: g if keep is None else {s: r for s, r in g.items() if s in keep}
    base = pick(load_group(args.base))
    news = {name: pick(load_group(name)) for name in args.new}
    for name, g in news.items():
        assert sorted(g) == sorted(base), f"seeds differ: {args.base} {sorted(base)} vs {name} {sorted(g)}"
    every = [*base.values(), *(r for g in news.values() for r in g.values())]
    fracs = {r["train_frac"] for r in every}
    assert len(fracs) == 1, f"train_frac differs between runs: {fracs}"
    for r in every:
        preds_path(r)
    if any(r["dirty"] for r in every):
        print("WARNING: some of these results come from a dirty tree")

    y = load_split("val", columns=TARGETS).to_numpy().astype(np.float32)
    games = load_split("val", columns=["game_num"])["game_num"].to_numpy()
    print(f"train_frac {fracs.pop()}, seeds {sorted(base)}")
    verdicts = {name: compare(args.base, base, name, g, y, games) for name, g in news.items()}
    if len(news) > 1:
        print(f"\nsummary vs {args.base} (seed mean, [range]):")
        for name, g in [(args.base, base), *news.items()]:
            s = np.array([r["score"] for r in g.values()])
            print(f"  {name:>28} {s.mean():.5f} [{s.min():.5f}, {s.max():.5f}]  {verdicts.get(name, 'base')}")


if __name__ == "__main__":
    main()
