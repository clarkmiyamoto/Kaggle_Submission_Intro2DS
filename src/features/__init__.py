"""Registry of feature sets that configs can name under `features:`."""
import re

from src.data import FEATURES
from src.features.dup import DUP_FEATURES, add_dup_features
from src.features.goal import GOAL_FEATURES, add_goal_features
from src.features.host import HOST_FEATURES, add_host_features
from src.features.warp import add_warp_features, warp_features

WARP_NAME = re.compile(r"warp_s(inf|\d+(?:\.\d+)?)")


class _FeatureSets(dict):
    """The fixed sets below, plus parametrised names resolved on lookup (`FEATURE_SETS[name]`):
    `warp_s<s>` -> goal-centred warp with spread scale s > 0, or `warp_sinf` for no warp."""

    def __missing__(self, name: str):
        m = WARP_NAME.fullmatch(name)
        if not m:
            raise KeyError(f"unknown feature set {name!r}")
        s = float(m[1])
        if not s > 0:
            raise ValueError(f"{name}: warp scale must be > 0 (or inf)")
        return (lambda df: add_warp_features(df, s), warp_features(s))


# name -> (function that appends the columns, the columns it provides)
FEATURE_SETS = _FeatureSets({
    "raw": (lambda df: df, FEATURES),
    "goal": (add_goal_features, GOAL_FEATURES),
    "dup": (add_dup_features, DUP_FEATURES),  # control only: copies of raw positions / velocities
    "host": (add_host_features, HOST_FEATURES),  # the host solution's 54-column layout; use alone
})
