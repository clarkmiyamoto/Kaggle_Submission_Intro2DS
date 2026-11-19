"""MLP for both targets: shared hidden layers, two logits, sigmoid at predict time.

`hidden: []` is logistic regression (one linear layer per target), trained with the same
loop, so the LR baseline and the MLPs differ only in capacity.

Training: AdamW (or `optimizer.name`) on random-permutation minibatches, with the whole
train matrix held on `device`. The output bias starts at the train log-odds of each target
(`init_output_bias: prior`), so no steps are spent learning the base rate.

Early stopping: val loss is computed densely at first (the gap between evals grows by
`eval_growth` x step) and then every `eval_every` steps, capped at one epoch, because wide
MLPs on few games overfit within tens of steps. Training stops once
max(`patience_epochs`, `min_patience_steps`) of training has passed without a val
improvement larger than `min_delta`, or at `max_steps`, and the best weights are restored.
`lr_schedule: {name: plateau, ...}` multiplies the LR by `factor` (down to `min_lr`) once
max(`patience_epochs`, `min_patience_steps`) has passed since the last improvement or cut.
Both windows are in steps, so the eval schedule only sets their resolution.
Inputs must already be scaled and null-free (see src/preprocess.py).
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
OPTIMIZERS = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}


def focal_loss(logits: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


class MLP:
    def __init__(self, hidden=(256, 128), dropout=0.0, activation="relu", loss="bce",
                 focal_gamma=2.0, init_output_bias="prior", optimizer=None, lr_schedule=None,
                 batch_size=4096, max_steps=100_000, eval_every=1000, eval_growth=0.25,
                 patience_epochs=2.0, min_patience_steps=2000, min_delta=0.0, device="mps",
                 eval_batch_size=262_144):
        self.hidden = list(hidden)
        self.dropout = float(dropout)
        self.activation = activation
        self.loss = loss
        self.focal_gamma = float(focal_gamma)
        assert init_output_bias in ("prior", "zero"), f"unknown init_output_bias {init_output_bias!r}"
        self.init_output_bias = init_output_bias
        self.optimizer = {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4, **(optimizer or {})}
        self.lr_schedule = lr_schedule or {"name": "none"}
        assert self.lr_schedule["name"] in ("none", "plateau"), f"unknown lr_schedule {self.lr_schedule}"
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.eval_every = int(eval_every)
        self.eval_growth = float(eval_growth)
        self.patience_epochs = float(patience_epochs)
        self.min_patience_steps = int(min_patience_steps)
        self.min_delta = float(min_delta)
        self.device = torch.device(device)
        self.eval_batch_size = int(eval_batch_size)
        assert loss in ("bce", "focal"), f"unknown loss {loss!r}"
        self.net = None
        self.fit_info = {}

    def _build(self, n_in: int) -> nn.Module:
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), ACTIVATIONS[self.activation]()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        return nn.Sequential(*layers).to(self.device)

    def _train_loss(self, logits, y):
        if self.loss == "focal":
            return focal_loss(logits, y, self.focal_gamma)
        return F.binary_cross_entropy_with_logits(logits, y)

    @torch.no_grad()
    def _logits(self, X: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        return torch.cat([self.net(X[i:i + self.eval_batch_size])
                          for i in range(0, len(X), self.eval_batch_size)])

    def fit(self, X_tr, y_tr, X_val, y_val) -> None:
        Xt = torch.from_numpy(X_tr).to(self.device)
        yt = torch.from_numpy(y_tr.astype(np.float32)).to(self.device)
        Xv = torch.from_numpy(X_val).to(self.device)
        yv = torch.from_numpy(y_val.astype(np.float32)).to(self.device)

        self.net = self._build(Xt.shape[1])
        if self.init_output_bias == "prior":
            rate = yt.mean(0).clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                self.net[-1].bias.copy_(torch.log(rate / (1 - rate)))
        opt_cfg = dict(self.optimizer)
        opt_cls = OPTIMIZERS[opt_cfg.pop("name")]
        # YAML reads "1e-3" as a string, so cast every numeric optimizer knob
        opt = opt_cls(self.net.parameters(), **{k: float(v) for k, v in opt_cfg.items()})

        n = len(Xt)
        steps_per_epoch = math.ceil(n / self.batch_size)
        eval_every = min(self.eval_every, steps_per_epoch)
        patience_steps = max(self.patience_epochs * steps_per_epoch, self.min_patience_steps)
        sched = self.lr_schedule
        plateau = sched["name"] == "plateau"
        if plateau:
            plateau_steps = max(float(sched.get("patience_epochs", 0.5)) * steps_per_epoch,
                                int(sched.get("min_patience_steps", 500)))
            factor, min_lr = float(sched.get("factor", 0.3)), float(sched.get("min_lr", 1e-6))

        best, best_step, best_state = math.inf, 0, None
        history, lr_cuts, step, next_eval, stop_reason = [], [], 0, 1, "max_steps"
        run_loss, run_n = torch.zeros((), device=self.device), 0
        t0 = time.time()
        while stop_reason == "max_steps" and step < self.max_steps:
            perm = torch.randperm(n, device=self.device)
            for i in range(steps_per_epoch):
                self.net.train()
                idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                loss = self._train_loss(self.net(Xt[idx]), yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                run_loss += loss.detach()
                run_n += 1
                step += 1
                if step < next_eval and step < self.max_steps:
                    continue
                next_eval = step + min(eval_every, max(1, math.ceil(step * self.eval_growth)))
                # val loss = mean of the two binary log losses (the metric, unclipped)
                val = F.binary_cross_entropy_with_logits(self._logits(Xv), yv).item()
                lr = opt.param_groups[0]["lr"]
                history.append({"step": step, "train_loss": (run_loss / run_n).item(),
                                "val_loss": val, "lr": lr})
                run_loss.zero_(); run_n = 0
                if val < best - self.min_delta:
                    best, best_step = val, step
                    best_state = copy.deepcopy(self.net.state_dict())
                elif plateau and step - max([best_step, *lr_cuts]) >= plateau_steps and lr > min_lr:
                    for g in opt.param_groups:
                        g["lr"] = max(lr * factor, min_lr)
                    lr_cuts.append(step)
                if len(history) % 10 == 0 or step - best_step >= patience_steps:
                    print(f"  step {step:>7} ep {step / steps_per_epoch:6.2f}  train {history[-1]['train_loss']:.5f}"
                          f"  val {val:.5f}  best {best:.5f}@{best_step}  lr {lr:.1e}"
                          f"  {time.time() - t0:5.0f}s", flush=True)
                if step - best_step >= patience_steps:
                    stop_reason = "patience"
                    break
                if step >= self.max_steps:
                    break

        self.net.load_state_dict(best_state)
        self.fit_info = {
            "best_step": best_step,
            "stopped_step": step,
            "max_steps": self.max_steps,
            "stop_reason": stop_reason,
            "best_is_last_eval": best_step == history[-1]["step"],
            "best_is_first_eval": best_step == history[0]["step"],
            "steps_per_epoch": steps_per_epoch,
            "eval_every": eval_every,
            "patience_steps": patience_steps,
            "plateau_steps": plateau_steps if plateau else None,
            "lr_cuts": lr_cuts,
            "epochs": step / steps_per_epoch,
            "best_val_loss": best,
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "device": str(self.device),
            "history": history,
        }

    def predict_proba(self, X) -> np.ndarray:
        logits = self._logits(torch.from_numpy(X).to(self.device))
        return torch.sigmoid(logits).cpu().numpy()
