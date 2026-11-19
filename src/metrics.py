"""The one scorer: mean columnwise log loss, and a paired bootstrap over val games."""
import numpy as np

EPS = 1e-7  # one clip value for everyone


def _binary_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def score(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Mean columnwise log loss. Both arrays (n, 2), columns in TARGETS order."""
    assert y_true.shape == p.shape and p.ndim == 2 and p.shape[1] == 2, "expected (n, 2) in TARGETS order"
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "pass probabilities, not logits"
    a = _binary_logloss(y_true[:, 0], p[:, 0])
    b = _binary_logloss(y_true[:, 1], p[:, 1])
    return {"logloss_A": a, "logloss_B": b, "score": (a + b) / 2}


def _row_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row log loss averaged over the two targets; its mean equals score()['score']."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)


def paired_bootstrap(y, p_new, p_base, games, n_boot=1000, seed=0) -> tuple[float, float, float]:
    """score(p_new) - score(p_base) and its 95% interval, resampling val games with replacement.
    Negative = p_new is better."""
    d = _row_loss(y, p_new) - _row_loss(y, p_base)
    _, g = np.unique(games, return_inverse=True)
    sums, counts = np.bincount(g, weights=d), np.bincount(g)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC by the rank-sum formula; tied scores share their average rank."""
    order = np.argsort(p, kind="mergesort")
    _, first, counts = np.unique(p[order], return_index=True, return_counts=True)
    ranks = np.empty(len(p))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    pos = y == 1
    n_pos = int(pos.sum())  # integer counts: float32 labels would round n_pos * (n_pos + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos)))


def _average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under precision-recall: sum of (recall step) * precision over distinct thresholds."""
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order] == 1, p[order]
    last = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]  # last row of each tied score
    tp = np.cumsum(ys)[last]  # integer counts
    precision = tp / (last + 1)
    recall_step = np.diff(np.r_[0, tp]) / ys.sum()
    return float((recall_step * precision).sum())


def diagnostics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Secondary metrics, per target and averaged. Not for model selection: claims, comparisons
    and early stopping use score(). Accuracy thresholds at p > 0.5; with ~5.7% positives,
    always predicting "no goal" already scores `accuracy_majority` (~0.94)."""
    assert y_true.shape == p.shape and p.ndim == 2 and p.shape[1] == 2, "expected (n, 2) in TARGETS order"
    out = {}
    for name, fn in [("accuracy", lambda y, q: float(((q > 0.5) == y).mean())),
                     ("accuracy_majority", lambda y, q: float(max(y.mean(), 1 - y.mean()))),
                     ("auc", _auc), ("ap", _average_precision)]:
        a, b = fn(y_true[:, 0], p[:, 0]), fn(y_true[:, 1], p[:, 1])
        out |= {f"{name}_A": a, f"{name}_B": b, name: (a + b) / 2}
    return out
