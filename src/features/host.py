"""The input layout of the competition host's solution (dster, "Rocket League TPS Host
solution" + its "preprocessing" notebook on Kaggle): raw positions, velocities and boost, a
demolished flag per player, and constant stand-ins for demolished players' nulls.

Columns, in this order (the host model in src/models/host_net.py reads them by position):
  host_ball_{pos,vel}_{x,y,z}                                     6
  host_p{i}_{pos,vel}_{x,y,z}, host_p{i}_boost, host_p{i}_demoed   8 per player, p0..p5
Boost-pad timers are dropped, as in the host's preprocessing.

A demolished player (all of its pos / vel / boost null; they are never partly null) gets
demoed = 1 and the host's dummy state "hovering above the stadium, flying up, no boost":
  pos_x = vel_x = -20 + 20 * (i % 3),  pos_y = vel_y = -20 + 40 * (i // 3),
  pos_z = 100, vel_z = 10, boost = 0.
No nulls remain. Values are not scaled: the host feeds raw values, and the model's flips
negate coordinates, which is exact only on unscaled inputs (use `preprocess: {scaler: none}`).
Output is Float32. Works on DataFrames and LazyFrames, including Kaggle's test set.
"""
import polars as pl

KINEMATIC = [f"{k}_{a}" for k in ("pos", "vel") for a in "xyz"]
PLAYER = KINEMATIC + ["boost", "demoed"]
HOST_FEATURES = ([f"host_ball_{c}" for c in KINEMATIC]
                 + [f"host_p{i}_{c}" for i in range(6) for c in PLAYER])


def demo_fill(i: int) -> dict[str, float]:
    """The host's stand-in values for player i while demolished."""
    x, y = -20.0 + 20 * (i % 3), -20.0 + 40 * (i // 3)
    return {"pos_x": x, "vel_x": x, "pos_y": y, "vel_y": y, "pos_z": 100.0, "vel_z": 10.0,
            "boost": 0.0}


def host_feature_exprs() -> list[pl.Expr]:
    exprs = [pl.col(f"ball_{c}").cast(pl.Float32).alias(f"host_ball_{c}") for c in KINEMATIC]
    for i in range(6):
        fill = demo_fill(i)
        exprs += [pl.col(f"p{i}_{c}").fill_null(fill[c]).cast(pl.Float32).alias(f"host_p{i}_{c}")
                  for c in KINEMATIC + ["boost"]]
        exprs.append(pl.col(f"p{i}_boost").is_null().cast(pl.Float32).alias(f"host_p{i}_demoed"))
    return exprs


def add_host_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Return df with the HOST_FEATURES columns appended."""
    return df.with_columns(host_feature_exprs())
