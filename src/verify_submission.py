"""Check a reproduced Kaggle submission against the committed final submission.

    uv run python -m src.verify_submission submissions/<reproduced>.csv
    uv run python -m src.verify_submission submissions/<a>.csv --reference submissions/<b>.csv

A rerun of the final recipe is bit-identical only on the same software and GPU type (the
submitted runs trained on a Modal A10G). Anywhere else, run-to-run float differences grow
over 24 passes, and the result differs from the reference the way another seed would. So
the check is statistical. Per target, it compares the two CSVs on Kaggle's 701,143 test rows:

- `corr`: Pearson correlation of the logits
- `kl`: mean Bernoulli KL(reference || reproduction)
- `dmean`: difference in mean probability

The thresholds come from the saved Kaggle predictions of the final recipe (Nov 2026):

| pair | corr | kl | dmean |
| --- | --- | --- | --- |
| two train + val runs, seeds 0 vs 1 | 0.9851 | 0.00106 | +0.00005 |
| train-only 3-seed ensemble vs the submitted train + val ensemble | 0.9947 | 0.00037 | +0.00035 |
| train-only single seeds vs the submitted ensemble | 0.988-0.991 | 0.00067-0.00071 | +0.0002-0.0006 |

Verdicts:
- IDENTICAL: every probability within 1e-6 of the reference.
- MATCH: as close as the train-only 3-seed ensemble that shares the seeds (corr >= 0.993,
  kl <= 0.0005, |dmean| <= 0.0005).
- PASS: within the gap between two seeds of the recipe (corr >= 0.985, kl <= 0.0011,
  |dmean| <= 0.001). The same method; a 3-seed rerun whose runs diverged from the originals
  (other hardware) should land here or in MATCH.
- FAIL: further apart than two seeds of the recipe. Something differs: data, features,
  preprocessing, schedule, or the run did not finish.

Exit code 0 for IDENTICAL, MATCH or PASS, 1 for FAIL.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.data import ROOT, TARGETS

REFERENCE = ROOT / "submissions" / "final_full24_wide2_drop0.05_trval_3seeds.csv"
EPS = 1e-7
MATCH = {"corr": 0.993, "kl": 0.0005, "dmean": 0.0005}
PASS = {"corr": 0.985, "kl": 0.0011, "dmean": 0.001}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.clip(p, EPS, 1 - EPS), np.clip(q, EPS, 1 - EPS)
    return float((p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))).mean())


def _within(stats: list[dict], limits: dict) -> bool:
    return all(s["corr"] >= limits["corr"] and s["kl"] <= limits["kl"] and abs(s["dmean"]) <= limits["dmean"]
               for s in stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", type=Path, help="submission CSV to check")
    ap.add_argument("--reference", type=Path, default=REFERENCE, help=f"default: {REFERENCE.relative_to(ROOT)}")
    a = ap.parse_args()

    ref = pl.read_csv(a.reference)
    cand = pl.read_csv(a.candidate)
    for name, df in (("reference", ref), ("candidate", cand)):
        assert df.columns == ["id"] + TARGETS, f"{name} columns: {df.columns}"
    assert cand.height == ref.height and cand["id"].equals(ref["id"]), \
        "candidate must list the reference's ids in the same order (src.submit writes them that way)"
    R = ref.select(TARGETS).to_numpy().astype(np.float64)
    C = cand.select(TARGETS).to_numpy().astype(np.float64)
    assert np.isfinite(C).all() and ((C > 0) & (C < 1)).all(), "candidate probabilities must be inside (0, 1)"

    stats = []
    print(f"candidate {a.candidate}\nreference {a.reference}  ({ref.height:,} rows)")
    for i, t in enumerate(TARGETS):
        s = {"corr": float(np.corrcoef(_logit(C[:, i]), _logit(R[:, i]))[0, 1]),
             "kl": _kl(R[:, i], C[:, i]),
             "dmean": float(C[:, i].mean() - R[:, i].mean()),
             "max_abs": float(np.abs(C[:, i] - R[:, i]).max())}
        stats.append(s)
        print(f"  {t}: mean p {C[:, i].mean():.5f} vs {R[:, i].mean():.5f} (dmean {s['dmean']:+.5f}), "
              f"logit corr {s['corr']:.5f}, kl {s['kl']:.6f}, max |dp| {s['max_abs']:.2e}")

    if all(s["max_abs"] <= 1e-6 for s in stats):
        verdict = "IDENTICAL"
    elif _within(stats, MATCH):
        verdict = "MATCH"
    elif _within(stats, PASS):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    print(f"{verdict}: " + {
        "IDENTICAL": "the same predictions as the reference",
        "MATCH": "as close as the train-only 3-seed ensemble that shares the seeds",
        "PASS": "within the gap between two seeds of the recipe",
        "FAIL": "further from the reference than two seeds of the recipe are from each other",
    }[verdict])
    sys.exit(1 if verdict == "FAIL" else 0)


if __name__ == "__main__":
    main()
