"""Goal-centred warped coordinates: near a goal, small moves span more of the feature range;
far from it, distances are compressed logarithmically.

For each goal centre g (from goal.py) and spread scale s (field units; the field is about
160 x 210 x 40, the goal mouth about 36 wide), every entity's position x and velocity v are
mapped by
  r = |x - g|,  rho(r) = s * ln(1 + r / s)    (rho ~ r for r << s, ~ s ln(r/s) for r >> s)
  position  x' = (x - g) * rho(r) / r
  velocity  v' = J v,  J = rho'(r) r^ r^T + (rho(r) / r) (I - r^ r^T),  rho'(r) = s / (s + r)
v' is the velocity of the warped point. Its radial part is approach speed * s / (s + r),
roughly s / (time to reach the goal), so motion toward a nearby goal counts for more.
Smaller s spreads the region near the goal more; s = inf is no warp (x' = x - g, v' = v),
the control for "more columns" without the nonlinearity.

Columns, for e in {ball, p0..p5}, goal in {Ascore, Bscore}, k in pos_x..vel_z:
  {e}_w{s}_{goal}_{k}       e.g. ball_w10_Ascore_pos_y, p3_winf_Bscore_vel_x
Configs name the set `warp_s<s>`, e.g. `warp_s10`, `warp_s2.5`, `warp_sinf`
(resolved in src/features/__init__.py). Demolished players give nulls. Output is Float32.
Works on DataFrames and LazyFrames, including Kaggle's test set.
"""
import math

import polars as pl

from src.features.goal import ENTITIES, GOALS

AXES = "xyz"


def scale_tag(s: float) -> str:
    """Canonical text for s in names: 10 and 10.0 -> '10', inf -> 'inf'."""
    return "inf" if math.isinf(s) else f"{s:g}"


def warp_features(s: float) -> list[str]:
    t = scale_tag(s)
    return [f"{e}_w{t}_{g}_{k}_{a}" for e in ENTITIES for g in GOALS for k in ("pos", "vel") for a in AXES]


def warp_feature_exprs(s: float) -> list[pl.Expr]:
    if not s > 0:
        raise ValueError(f"warp scale must be > 0 (or inf), got {s}")
    t = scale_tag(s)
    exprs = []
    for e in ENTITIES:
        pos = [pl.col(f"{e}_pos_{a}") for a in AXES]
        vel = [pl.col(f"{e}_vel_{a}") for a in AXES]
        for g, centre in GOALS.items():
            d = [p - c for p, c in zip(pos, centre)]
            if math.isinf(s):
                new_pos, new_vel = d, vel
            else:
                r = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2).sqrt()
                # rho(r) / r and rho'(r); both -> 1 as r -> 0
                ratio = pl.when(r > 0).then(s * (r / s).log1p() / r).otherwise(1.0)
                radial = s / (s + r)
                v_r = pl.when(r > 0).then((vel[0] * d[0] + vel[1] * d[1] + vel[2] * d[2]) / r).otherwise(0.0)
                unit = [pl.when(r > 0).then(di / r).otherwise(0.0) for di in d]
                new_pos = [di * ratio for di in d]
                # J v = ratio * v + (rho' - ratio) * (v . r^) r^
                new_vel = [ratio * vi + (radial - ratio) * v_r * ui for vi, ui in zip(vel, unit)]
            exprs += [x.cast(pl.Float32).alias(f"{e}_w{t}_{g}_pos_{a}") for x, a in zip(new_pos, AXES)]
            exprs += [x.cast(pl.Float32).alias(f"{e}_w{t}_{g}_vel_{a}") for x, a in zip(new_vel, AXES)]
    return exprs


def add_warp_features(df: pl.DataFrame | pl.LazyFrame, s: float) -> pl.DataFrame | pl.LazyFrame:
    """Return df with the warp_features(s) columns appended."""
    return df.with_columns(warp_feature_exprs(s))
