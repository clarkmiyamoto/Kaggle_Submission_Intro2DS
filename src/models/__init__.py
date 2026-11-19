"""The Model interface and the registry that configs name under `model.type`."""
from typing import Protocol

import numpy as np

from src.models.host_net import HostNet
from src.models.nn import MLP


class Model(Protocol):
    def fit(self, X_tr, y_tr, X_val, y_val) -> None: ...  # X_val / y_val only for early stopping
    def predict_proba(self, X) -> np.ndarray: ...         # (n, 2) probabilities, TARGETS order

    # Optional: a dict of training facts (best step, stop reason, ...) that run.py copies
    # into results/<run_id>.json as `fit_info`.


MODELS = {  # name -> class, constructed with the config's model params
    "mlp": MLP,
    "host_net": HostNet,  # the competition host's permutation-invariant net
}
