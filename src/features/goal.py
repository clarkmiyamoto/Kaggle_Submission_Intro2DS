"""Goal-relative features: how far the ball and each player are from each goal,
and how fast / how directly they are moving toward it.

Field geometry (data units are Rocket League units / 50):
  y is the goal axis. Team A (p0-p2) scores into the goal at y = +102.4,
  Team B (p3-p5) scores into the goal at y = -102.4. x is side to side, z is height.

Goals are named by the team that scores into them, to line up with TARGETS:
  "Ascore" = goal at y = +102.4 (Team B defends it)
  "Bscore" = goal at y = -102.4 (Team A defends it)

For each entity e in {ball, p0..p5} and goal g in {Ascore, Bscore}:
  {e}_dist_{g}      distance from e to the centre of the goal mouth
  {e}_approach_{g}  velocity . unit(goal - pos): speed toward the goal, in velocity
                    units; positive = getting closer
  {e}_cos_{g}       approach / speed: cosine between velocity and the direction to
                    the goal (1 = straight at it, -1 = straight away); null when speed is 0

Inputs with nulls (demolished players) give null features.

Usage:
    from src.data import load_split, FEATURES
    from src.features.goal import add_goal_features, GOAL_FEATURES
    train = add_goal_features(load_split("train"))
    X = train[FEATURES + GOAL_FEATURES]
Works on DataFrames and LazyFrames, including Kaggle's test set.
"""
import polars as pl

GOAL_Y = 5120 / 50                 # goal line
GOAL_Z = 642.775 / 2 / 50          # half the goal height: aim at the middle of the mouth
GOALS = {"Ascore": (0.0, GOAL_Y, GOAL_Z), "Bscore": (0.0, -GOAL_Y, GOAL_Z)}

ENTITIES = ["ball"] + [f"p{i}" for i in range(6)]
GOAL_FEATURES = [f"{e}_{stat}_{g}" for e in ENTITIES for g in GOALS
                 for stat in ("dist", "approach", "cos")]


def goal_feature_exprs() -> list[pl.Expr]:
    exprs = []
    for e in ENTITIES:
        px, py, pz = (pl.col(f"{e}_pos_{a}") for a in "xyz")
        vx, vy, vz = (pl.col(f"{e}_vel_{a}") for a in "xyz")
        speed = (vx**2 + vy**2 + vz**2).sqrt()
        for g, (gx, gy, gz) in GOALS.items():
            dx, dy, dz = gx - px, gy - py, gz - pz
            dist = (dx**2 + dy**2 + dz**2).sqrt()
            approach = (vx * dx + vy * dy + vz * dz) / dist
            exprs += [
                dist.alias(f"{e}_dist_{g}"),
                approach.alias(f"{e}_approach_{g}"),
                pl.when(speed > 0).then(approach / speed).alias(f"{e}_cos_{g}"),
            ]
    return exprs


def add_goal_features(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Return df with the GOAL_FEATURES columns appended."""
    return df.with_columns(goal_feature_exprs())
