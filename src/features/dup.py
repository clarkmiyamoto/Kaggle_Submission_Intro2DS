"""A CONTROL, not a feature: exact copies of the raw positions and velocities.

Adding copies of existing inputs gives the model no new information, but it changes how an
MLP optimises them; for mlp64 that alone was worth about -0.001 on val
(an earlier experiment, not in this repo). A feature set's gain counts only if it also beats
a control of the same width, and this is the 42-column control for `goal`. Never use it in
a model meant to be good.

Columns: `<c>_dup` for each of the 42 position / velocity columns of the ball and p0..p5.
Nulls (demolished players) wherever the source column is null. Float32.
Works on DataFrames and LazyFrames, including Kaggle's test set.
"""
import polars as pl

from src.features.goal import ENTITIES

KINEMATIC = [f"{e}_{k}_{a}" for e in ENTITIES for k in ("pos", "vel") for a in "xyz"]
DUP_FEATURES = [f"{c}_dup" for c in KINEMATIC]


def add_dup_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Return df with the DUP_FEATURES columns appended."""
    return df.with_columns([pl.col(c).cast(pl.Float32).alias(f"{c}_dup") for c in KINEMATIC])
