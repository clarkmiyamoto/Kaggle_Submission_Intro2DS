"""Preprocessing fit on train only, applied identically to every other set.

Config block (all keys optional):
    preprocess:
      scaler: standard   # standard | none. Mean / std per column from train, nulls ignored
      fill_null: 0       # applied AFTER scaling, so 0 = the train mean of that column
                         # (demolished players give nulls)

Order: scale with train statistics, then fill nulls. Output is Float32.
"""
import numpy as np
import polars as pl


class Preprocessor:
    def __init__(self, scaler: str = "standard", fill_null: float = 0.0):
        assert scaler in ("standard", "none"), f"unknown scaler {scaler!r}"
        self.scaler = scaler
        self.fill_null = float(fill_null)
        self.stats: dict[str, tuple[float, float]] = {}

    def fit(self, lf_train: pl.LazyFrame, columns: list[str]) -> "Preprocessor":
        self.columns = list(columns)
        if self.scaler == "standard":
            row = lf_train.select(
                [pl.col(c).mean().alias(f"{c}__mean") for c in columns]
                + [pl.col(c).std().alias(f"{c}__std") for c in columns]
            ).collect().row(0, named=True)
            for c in columns:
                std = row[f"{c}__std"]
                self.stats[c] = (row[f"{c}__mean"], std if std and std > 0 else 1.0)
        return self

    def exprs(self) -> list[pl.Expr]:
        out = []
        for c in self.columns:
            e = pl.col(c).cast(pl.Float32)
            if self.scaler == "standard":
                mean, std = self.stats[c]
                e = (e - mean) / std
            out.append(e.fill_null(self.fill_null).fill_nan(self.fill_null).cast(pl.Float32).alias(c))
        return out

    def transform(self, lf: pl.LazyFrame) -> np.ndarray:
        X = lf.select(self.exprs()).collect().to_numpy().astype(np.float32, copy=False)
        assert np.isfinite(X).all(), "non-finite values after preprocessing"
        return X
