"""The competition host's network (dster, "Rocket League TPS Host solution" on Kaggle),
ported from Keras to PyTorch. Inputs are the 54 `host` columns (src/features/host.py),
unscaled.

Architecture, as in the host's notebook (every Dense / 1x1 conv uses Mish; a nested width
list [a, b] is two layers followed by one BatchNorm, a bare width one layer then BatchNorm):
  ball (6)  -> dense [[48, 36], [36, 32]]                                    -> b (32)
  player i  -> [own 8 features, team flag, ball - own pos/vel (6), b]  (47)
            -> per-player 1x1 conv [[81, 64], [64, 48], 36]                  -> p_i (36)
  pairs     -> [p_j, p_i] for each teammate j != i, and for each j of the
               "opponent" set; per-pair 1x1 conv [[81, 64], [64, 48], 36],
               max-pooled over j (one conv stack for each of the two sets)   -> t_i, o_i
  per team  -> [p_i, t_i, o_i] -> 1x1 conv [[81, 64], [64, 48], 36], max over the team's 3
  head      -> [b, team A, team B] (104) -> dense [[81, 64], [64, 48], 42]
            -> softmax over 3 classes: A scores within 10 s / nobody does / B does
P(A) and P(B) are classes 0 and 2. The two targets never co-occur, so the class comes from
TARGETS alone.

Options beyond the host's notebook (defaults reproduce it, except that the host also has
the 1..9 s heads):
  horizons      [10] or e.g. [1, ..., 10]: one 3-class head per horizon, "scores within h s",
                weighted 1 for 10 s and `aux_weight` (host: 0.3) for the rest. The labels come
                from the META columns in TRAIN_LABELS, which run.py passes to fit() as
                `labels_tr` (never as inputs). Predictions use the 10 s head.
  goal_features the goal features of src/features/goal.py (distance, approach speed, cosine
                to each goal) for the ball and each player, computed inside the net after
                augmentation, so a y mirror also swaps the goals.
  input_scale   divide positions by the field's half-size, velocities by a car's top speed
                and boost by 100 (the host feeds raw values).
  warp_scale    s > 0: the goal-centred warp of src/features/warp.py (positions and
                velocities relative to each goal, log-compressed beyond ~s) for the ball and
                each player, computed inside the net after augmentation like goal_features.
  width         multiply every layer width by this (1 = the host's net).
  dropout       p > 0: dropout after every BatchNorm group, in every stack.
  head_dropout  p > 0: dropout on the head's output only, just before the final linear layer.
  backbone      `host` (this net) or `mlp`: a plain MLP on the 54 inputs, `mlp_hidden` in the
                same group notation, with no built-in symmetry at all.
  frame         frame averaging (Puny et al., arXiv:2110.03336) with the whole group as the
                frame: the net's 3-class logits are averaged over every element of `flips`
                (the 4 mirrors), `perms` (the 36 within-team reorderings) or `all` (144), with
                A / B swapped back after a y mirror. The result is exactly invariant (for a y
                mirror, equivariant) to that group in training as in inference. The groups are
                small enough to average over in full, so no input-dependent frame is needed.
                The host's net is already invariant to `perms`; for it only `flips` changes
                anything. TTA is skipped when the frame contains the mirrors.
  ema_epochs    keep an exponential moving average of the weights and of the BatchNorm
                statistics, decay 1 - 1 / (ema_epochs x steps per pass), warmed up as
                min(decay, (1 + t) / (10 + t)). Val is scored and the best checkpoint kept on
                the averaged weights; the live weights' val score is logged as `val_loss_raw`.
  optimizer     `name: adam` (the host's, default) or `adamw`, which needs an explicit
                `weight_decay`, applied (decoupled) to weight matrices only, not to biases or
                BatchNorm parameters.
  pair_geometry the mates / opponents pair stacks also get the raw relative geometry of each
                pair (pos_j - pos_i, vel_j - vel_i, distance), computed after augmentation.
  pool          `max` (the host's) or `maxmean`: every pooling (over pair partners, over a
                team's players) concatenates the max and the mean, so the net can count
                (e.g. how many defenders are goal-side), which a max alone cannot.
  input_noise   sigma > 0: in training, add Gaussian noise to every input except `demoed`, with
                std sigma x (half-field size for positions, a car's top speed for velocities,
                100 for boost), after augmentation.
  sam_rho       rho > 0: sharpness-aware minimization (Foret et al. 2021). Each step takes the
                gradient at w + rho * g / |g| (g the gradient at w, the same batch), and
                the optimizer applies it at w. Two forward / backward passes per step.
  distill       {alpha, folds, teacher_epochs}: self-distillation with cross-fitted teachers.
                The training games are split into `folds` (by game); for each fold the same
                recipe is trained on the other folds for `teacher_epochs` passes, and its TTA +
                snapshot probabilities for every horizon on this fold's rows become soft targets.
                The final net trains on alpha x one-hot + (1 - alpha) x teacher, the teacher's
                class vector reversed with the labels under a y mirror.
  snapshot_dump path: after training, save the val probabilities (TTA, float32) of the
                `snapshot_dump_n` best evals there (npz with probs, epochs, val_no_tta), for
                offline snapshot studies.
  snapshots     k > 1: keep the k best evals (val without TTA), each at least
                `snapshot_gap_epochs` passes from the others, and predict the mean of their
                (TTA) probabilities: a snapshot ensemble from one run.
  rounds        2: a second per-player stack after the first pooling. Each player's embedding,
                with [its team's, the other team's] pooled summaries, goes through another stack
                like `full` and is pooled again (one more round of message passing).
  ball_path     the ball stack also gets, per goal line, the time for the ball to reach it and
                where it would cross: x folded off the side walls, z under gravity clamped to
                floor and ceiling (HostNetModule._path), computed after augmentation.
  scorer_weight w > 0: an extra head, one logit per player from its final embedding before
                team pooling, trained with cross-entropy over the 6 players to name who scores,
                on rows with a goal within 10 s (META player_scoring_next as a training label),
                weight w. The label follows the player through mirrors and reorderings.
  backbone attn a transformer over the ball and the 6 players (AttnModule); `attn: {dim, layers,
                heads}`, `dropout` inside it.
  precision     `fp32` (default), `tf32` (matmuls in TF32) or `bf16` (autocast) for the
                training steps and the in-training val evals. The final predictions are always
                fp32.

`opponent_pairs`: the host builds the "opponent" set with tf.roll on the player axis, not
the team axis, so each player is paired with the 3 members of its own team (itself
included). Teams only meet in the head. `host` reproduces this; `fixed` rolls on the team
axis, pairing each player with the 3 opponents. Either way the net is exactly invariant to
reordering players within a team (scripts/check_host_net.py).

Training follows the host:
  - Adam (eps 1e-7, Keras's default), batch 168, cross-entropy on the 3 classes.
  - Keras-style init (Glorot-uniform weights, zero biases); BatchNorm momentum 0.01 and
    eps 1e-3 in torch terms (Keras 0.99 / 1e-3).
  - Augmentation, drawn per row and per step: mirror x (50%); mirror y (50%), which also
    swaps the teams and the A / B labels; shuffle players within each team. The shuffle
    cannot change the output of this net, so it is kept only to match the host.
  - Step LR schedule in passes over train: `lr_schedule.boundaries` / `values`. The host's
    is 2e-3, then 1.5e-3 from pass 1, 1e-3 from 3, 5e-4 from 5, 2e-4 from 7, 1e-4 from 10,
    5e-5 from 13.
  - Test-time augmentation (`tta: flips`): average the probabilities over the 4 mirror
    combinations, swapping A / B back after a y mirror. This is the expectation of the host's
    16 random draws, since player shuffles have no effect.
The host trains for a fixed 11.25 hours with no validation. Here val is scored with the
competition metric every `eval_every_epochs` passes (without TTA), training stops after
`max_epochs` passes, `max_hours`, or `patience_epochs` passes without a gain, and the best
weights are restored.

Speed: at batch 168 a step is a few hundred tiny kernels, so it is bound by launch overhead,
not FLOPs. `device: auto` picks cuda, then mps, then cpu. On cuda, `compile: <mode>`
torch.compiles the training forward + loss (`reduce-overhead` also replays it as a CUDA
graph); it is ignored on other devices. Evaluation always runs the uncompiled net.
"""
import contextlib
import copy
import itertools
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from src.features.goal import GOALS

N_IN = 54  # len(src.features.host.HOST_FEATURES)
BALL, PLAYER = 6, 8  # ball: pos xyz, vel xyz; player: pos xyz, vel xyz, boost, demoed
GOAL = 6  # goal_features: distance, approach speed, cosine, to each of the 2 goals
WARP = 12  # warp_scale: warped position and velocity (3 + 3) relative to each of the 2 goals
# input_scale (data units are Rocket League units / 50): positions over the field's half-size,
# velocities over a car's top speed, boost over its maximum. No shift, so mirrors stay exact.
POS_SCALE = [4096 / 50, 5120 / 50, 2044 / 50]
VEL_SCALE = 2300 / 50


class ChannelBN(nn.BatchNorm1d):
    """BatchNorm over the last axis with statistics pooled over every other axis (Keras's
    channels-last BatchNormalization)."""

    def forward(self, x):
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)


def stack(d_in: int, widths, dropout: float = 0.0) -> tuple[nn.Sequential, int]:
    """The host's batched_denses / batched_convNds: Dense+Mish layers, BatchNorm per group
    (then dropout, if p > 0)."""
    layers = []
    for group in widths:
        for w in group if isinstance(group, (list, tuple)) else [group]:
            layers += [nn.Linear(d_in, w), nn.Mish()]
            d_in = w
        layers.append(ChannelBN(d_in, eps=1e-3, momentum=0.01))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers), d_in


def keras_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# ball_path: ball-centre bounds in data units (uu / 50): side walls, floor, ceiling; gravity 650 uu/s^2
BALL_R = 92.75 / 50
WALL_X, FLOOR_Z, CEIL_Z = 4096 / 50 - BALL_R, BALL_R, 2044 / 50 - BALL_R
GRAVITY = 650 / 50
GOAL_HALF_W, CROSSBAR = 892.755 / 50, 642.775 / 50
PATH_T_MAX = 10.0
PATH = 6  # per goal: time to the goal line, x and z there (walls folded, gravity), scaled


def fold(u: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Reflect u into [lo, hi] as a ball bouncing elastically between two walls."""
    w = hi - lo
    m = torch.remainder(u - lo, 2 * w)
    return lo + torch.where(m < w, m, 2 * w - m)


PAIR_GEO = 7  # pair_geometry: pos_j - pos_i (3), vel_j - vel_i (3), |pos_j - pos_i|


def _pair_geo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(..., 8) raw player rows a (partner j) and b (self i) -> (..., 7) relative geometry."""
    d = a[..., :6] - b[..., :6]
    return torch.cat([d, d[..., :3].norm(dim=-1, keepdim=True)], dim=-1)


class HostNetModule(nn.Module):
    def __init__(self, opponent_pairs: str = "host", n_heads: int = 1, goal_features: bool = False,
                 input_scale: bool = False, warp_scale: float | None = None, width: float = 1.0,
                 dropout: float = 0.0, head_dropout: float = 0.0, pair_geometry: bool = False,
                 pool: str = "max", scorer_head: bool = False, ball_path: bool = False, rounds: int = 1):
        super().__init__()
        assert rounds in (1, 2), f"rounds must be 1 or 2, got {rounds}"
        self.rounds = rounds
        self.scorer_head = scorer_head
        self.ball_path = ball_path
        assert pool in ("max", "maxmean"), f"unknown pool {pool!r}"
        self.pool = pool
        k = 2 if pool == "maxmean" else 1  # pooled width multiplier
        assert opponent_pairs in ("host", "fixed"), f"unknown opponent_pairs {opponent_pairs!r}"
        self.opponent_dim = 2 if opponent_pairs == "host" else 1  # roll axis of (B, team, player, ...)
        self.goal_features, self.input_scale, self.warp_scale = goal_features, input_scale, warp_scale
        self.head_dropout = head_dropout
        self.pair_geometry = pair_geometry
        extra = (GOAL if goal_features else 0) + (WARP if warp_scale else 0)
        geo = PAIR_GEO if pair_geometry else 0
        w = lambda ws: [[round(width * x) for x in g] if isinstance(g, list) else round(width * g) for g in ws]
        widths = w([[81, 64], [64, 48], 36])
        self.ball, d_ball = stack(BALL + extra + (PATH if ball_path else 0), w([[48, 36], [36, 32]]), dropout)
        self.player, d_p = stack(PLAYER + extra + 1 + BALL + d_ball, widths, dropout)
        self.mates, d_t = stack(2 * d_p + geo, widths, dropout)
        self.opps, d_o = stack(2 * d_p + geo, widths, dropout)
        self.full, d_f = stack(d_p + k * (d_t + d_o), widths, dropout)
        if rounds == 2:  # each player again, with both teams' pooled summaries
            self.full2, d_f2 = stack(d_f + 2 * k * d_f, widths, dropout)
            assert d_f2 == d_f
        self.head, d_h = stack(d_ball + 2 * k * d_f, w([[81, 64], [64, 48], 42]), dropout)
        self.out = nn.Linear(d_h, 3 * n_heads)  # one 3-class softmax per horizon
        if scorer_head:  # one logit per player from its final (pre-pooling) embedding
            self.scorer = nn.Linear(d_f, 1)
        self.register_buffer("team", torch.tensor([0.0, 1.0]).view(1, 2, 1, 1).repeat(1, 1, 3, 1))
        self.register_buffer("goals", torch.tensor(list(GOALS.values())))  # (2, 3)
        self.goal_y = abs(float(GOALS["Ascore"][1]))
        self.register_buffer("ball_scale", torch.tensor(POS_SCALE + [VEL_SCALE] * 3))
        self.register_buffer("player_scale", torch.tensor(POS_SCALE + [VEL_SCALE] * 3 + [100.0, 1.0]))
        keras_init(self)

    def _goal(self, pos, vel):
        """(..., 3) position and velocity -> (..., 6): per goal, distance, approach speed and
        cosine as in src/features/goal.py (cosine 0 when not moving)."""
        d = self.goals - pos.unsqueeze(-2)  # (..., 2, 3)
        dist = d.norm(dim=-1)
        approach = (vel.unsqueeze(-2) * d).sum(-1) / dist
        speed = vel.norm(dim=-1, keepdim=True)
        cos = torch.where(speed > 0, approach / speed.clamp_min(1e-12), torch.zeros_like(approach))
        if self.input_scale:
            dist, approach = dist / 100, approach / VEL_SCALE
        return torch.stack([dist, approach, cos], dim=-1).flatten(-2)  # goal-major, like goal.py

    def _warp(self, pos, vel):
        """(..., 3) position and velocity -> (..., 12): per goal, the warped position and
        velocity of src/features/warp.py at spread scale s = warp_scale."""
        s = self.warp_scale
        d = pos.unsqueeze(-2) - self.goals  # (..., 2, 3)
        r = d.norm(dim=-1, keepdim=True)
        safe = r.clamp_min(1e-12)
        ratio = torch.where(r > 0, s * torch.log1p(r / s) / safe, torch.ones_like(r))  # rho(r) / r
        radial = s / (s + r)  # rho'(r)
        unit = torch.where(r > 0, d / safe, torch.zeros_like(d))
        v = vel.unsqueeze(-2)
        v_r = (v * unit).sum(-1, keepdim=True)
        new_vel = ratio * v + (radial - ratio) * v_r * unit  # J v
        return torch.cat([d * ratio, new_vel], dim=-1).flatten(-2)  # goal-major, like warp.py

    def _path(self, ball):
        """(B, 6) raw ball -> (B, 6): for each goal line (y = -102.4, then +102.4), the time for
        the ball to reach it at its current y velocity (capped at 10 s, also when moving away),
        and where it would be then: x reflected off the side walls (elastic, no corners), z
        under gravity, clamped to the floor and ceiling (no bounce). Scaled by 10 s,
        the goal's half-width and the crossbar height."""
        x, y, z, vx, vy, vz = ball.unbind(-1)
        out = []
        for gy in (-self.goal_y, self.goal_y):
            t = (gy - y) / torch.where(vy.abs() > 1e-3, vy, torch.full_like(vy, 1e-3))
            t = torch.where(t > 0, t, torch.full_like(t, PATH_T_MAX)).clamp(max=PATH_T_MAX)
            xh = fold(x + vx * t, -WALL_X, WALL_X)
            zh = (z + vz * t - 0.5 * GRAVITY * t * t).clamp(FLOOR_Z, CEIL_Z)  # lands and stays down
            out += [t / PATH_T_MAX, xh / GOAL_HALF_W, zh / CROSSBAR]
        return torch.stack(out, dim=-1)

    def _pool(self, v, dim):
        if self.pool == "max":
            return v.amax(dim=dim)
        return torch.cat([v.amax(dim=dim), v.mean(dim=dim)], dim=-1)

    def forward(self, x):  # (B, 54) -> (B, n_heads, 3) logits
        n = len(x)
        ball, players = x[:, :BALL], x[:, BALL:].reshape(n, 2, 3, PLAYER)
        gb, gp = [], []  # computed after augmentation, so a y mirror also swaps the goals
        if self.goal_features:
            gb.append(self._goal(ball[:, :3], ball[:, 3:]))
            gp.append(self._goal(players[..., :3], players[..., 3:6]))
        if self.warp_scale:
            gb.append(self._warp(ball[:, :3], ball[:, 3:]))
            gp.append(self._warp(players[..., :3], players[..., 3:6]))
        path = [self._path(ball)] if self.ball_path else []  # raw units, before input_scale
        if self.input_scale:
            ball, players = ball / self.ball_scale, players / self.player_scale
        b = self.ball(torch.cat([ball, *gb, *path], dim=-1))
        tile = lambda v: v[:, None, None, :].expand(n, 2, 3, v.shape[-1])
        p = self.player(torch.cat([players, *gp, self.team.expand(n, 2, 3, 1),
                                   tile(ball) - players[..., :BALL], tile(b)], dim=-1))
        # pairs[:, team, i, k] = p[:, team, i - k]: k = 0 is the player itself
        roll_pairs = lambda v: torch.stack([torch.roll(v, shifts=k, dims=2) for k in range(3)], dim=3)
        pairs = roll_pairs(p)
        selves = p.unsqueeze(3).expand_as(pairs)
        opp = torch.roll(pairs, shifts=1, dims=self.opponent_dim)
        mate_in = [pairs[:, :, :, 1:], selves[:, :, :, 1:]]
        opp_in = [opp, selves]
        if self.pair_geometry:  # the same pairing on the raw rows (after augmentation)
            raw = roll_pairs(players)
            raw_self = players.unsqueeze(3).expand_as(raw)
            raw_opp = torch.roll(raw, shifts=1, dims=self.opponent_dim)
            mate_in.append(_pair_geo(raw[:, :, :, 1:], raw_self[:, :, :, 1:]))
            opp_in.append(_pair_geo(raw_opp, raw_self))
        t = self._pool(self.mates(torch.cat(mate_in, dim=-1)), 3)
        o = self._pool(self.opps(torch.cat(opp_in, dim=-1)), 3)
        per_player = self.full(torch.cat([p, t, o], dim=-1))  # (B, 2, 3, d_f)
        f = self._pool(per_player, 2)  # (B, 2, 36 or 72): pooled per team
        if self.rounds == 2:  # [own team, other team] summaries, broadcast to each player
            ctx = torch.cat([f, f.flip(1)], dim=-1)[:, :, None, :].expand(n, 2, 3, -1)
            per_player = self.full2(torch.cat([per_player, ctx], dim=-1))
            f = self._pool(per_player, 2)
        h = self.head(torch.cat([b, f.reshape(n, -1)], dim=-1))
        if self.head_dropout > 0:
            h = F.dropout(h, self.head_dropout, self.training)
        out = self.out(h).reshape(n, -1, 3)
        if self.scorer_head:  # packed as 2 extra "heads": team A's 3 players, then team B's
            out = torch.cat([out, self.scorer(per_player).reshape(n, 2, 3)], dim=1)
        return out


class MLPModule(nn.Module):
    """A plain MLP on the 54 inputs (Mish, BatchNorm per group, Keras init, as above). It has
    none of the host net's symmetry: reordering players changes its output."""

    def __init__(self, n_heads: int = 1, hidden=((256, 256), 128), dropout: float = 0.0,
                 head_dropout: float = 0.0):
        super().__init__()
        self.body, d = stack(N_IN, hidden, dropout)
        self.head_dropout = head_dropout
        self.out = nn.Linear(d, 3 * n_heads)
        keras_init(self)

    def forward(self, x):  # (B, 54) -> (B, n_heads, 3) logits
        h = self.body(x)
        if self.head_dropout > 0:
            h = F.dropout(h, self.head_dropout, self.training)
        return self.out(h).reshape(len(x), -1, 3)


class AttnModule(nn.Module):
    """A transformer over 7 tokens: the ball and the 6 players. No positional encoding, so it is
    equivariant to reordering players; a team flag tells the teams apart. Inputs are scaled as
    `input_scale` does (positions / half-field, velocities / top speed, boost / 100). Tokens:
      ball:   [ball (6)]                                     -> Linear -> d
      player: [player (8), team flag, ball - player pos/vel (6)] -> Linear -> d
    `layers` pre-LayerNorm encoder layers (`heads` heads, GELU feed-forward 4d, `dropout`), then
    a final LayerNorm. Read-out: [ball token, max and mean of each team's player tokens] (5d)
    -> MLP [2d, d] (GELU) -> 3 logits per horizon. Exactly invariant to reordering players
    within a team."""

    def __init__(self, n_heads: int = 1, dim: int = 96, layers: int = 4, heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.ball_in = nn.Linear(BALL, dim)
        self.player_in = nn.Linear(PLAYER + 1 + BALL, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(5 * dim, 2 * dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 3 * n_heads))
        self.register_buffer("team", torch.tensor([0.0, 1.0]).view(1, 2, 1, 1).repeat(1, 1, 3, 1))
        self.register_buffer("ball_scale", torch.tensor(POS_SCALE + [VEL_SCALE] * 3))
        self.register_buffer("player_scale", torch.tensor(POS_SCALE + [VEL_SCALE] * 3 + [100.0, 1.0]))

    def forward(self, x):  # (B, 54) -> (B, n_heads, 3) logits
        n = len(x)
        ball = x[:, :BALL] / self.ball_scale
        players = x[:, BALL:].reshape(n, 2, 3, PLAYER) / self.player_scale
        rel = ball[:, None, None, :] - players[..., :BALL]
        tok_p = self.player_in(torch.cat([players, self.team.expand(n, 2, 3, 1), rel], dim=-1))
        tok = torch.cat([self.ball_in(ball)[:, None], tok_p.reshape(n, 6, -1)], dim=1)  # (B, 7, d)
        h = self.encoder(tok)
        pl_ = h[:, 1:].reshape(n, 2, 3, -1)
        z = torch.cat([h[:, 0], pl_.amax(dim=2).flatten(1), pl_.mean(dim=2).flatten(1)], dim=-1)
        return self.head(z).reshape(n, -1, 3)


# Group elements of augmentations() (index = 72 fx + 36 fy + 6 pa + pb) averaged over by `frame`
FRAMES = {"flips": [0, 36, 72, 108], "perms": list(range(36)), "all": list(range(144))}


class FrameAverage(nn.Module):
    """Frame averaging with the whole group G as the frame: mean over g in G of the backbone's
    logits on g x, with A / B swapped back after a y mirror. Exactly invariant to the x mirror
    and player reorderings in G, and equivariant (A <-> B) to the y mirror, in train mode too:
    the transformed copies of a batch are the same set for g x as for x, so BatchNorm sees
    the same statistics."""

    def __init__(self, backbone: nn.Module, frame: str):
        super().__init__()
        self.backbone = backbone
        idx, sign, flip_y = augmentations()
        elements = FRAMES[frame]
        self.register_buffer("idx", idx[elements])
        self.register_buffer("sign", sign[elements])
        self.register_buffer("flip_y", flip_y[elements].view(1, -1, 1, 1))

    def forward(self, x):  # (B, 54) -> (B, n_heads, 3) logits
        n, k = len(x), len(self.idx)
        z = self.backbone((x[:, self.idx] * self.sign).reshape(n * k, -1)).reshape(n, k, -1, 3)
        return torch.where(self.flip_y, z.flip(-1), z).mean(dim=1)


def augmentations() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All 2 x 2 x 6 x 6 augmentations as (gather index (144, 54), sign (144, 54), y-mirror
    flag (144,)): augmented[:, j] = sign[j] * x[:, index[j]]."""
    cols = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
    player_cols = cols + ["boost", "demoed"]
    perms = list(itertools.permutations(range(3)))
    idx, sign, flip_y = [], [], []
    for fx, fy, pa, pb in itertools.product((0, 1), (0, 1), perms, perms):
        def s(col):
            return -1.0 if (fx and col.endswith("_x")) or (fy and col.endswith("_y")) else 1.0
        i, g = list(range(BALL)), [s(c) for c in cols]
        for slot in range(6):  # slot takes the player at its team's permuted position, then
            team, pos = divmod(slot, 3)  # the other team's when mirrored in y
            src = 3 * team + (pa, pb)[team][pos]
            src = (src + 3) % 6 if fy else src
            i += [BALL + PLAYER * src + k for k in range(PLAYER)]
            g += [s(c) for c in player_cols]
        idx.append(i), sign.append(g), flip_y.append(bool(fy))
    return torch.tensor(idx), torch.tensor(sign), torch.tensor(flip_y)


def scorer_label_map(idx: torch.Tensor) -> torch.Tensor:
    """(144, 54) augmentation gather index -> (144, 6): the slot that player p lands in under
    each augmentation (augmented slot s holds source player (idx[a, BALL + 8 s] - BALL) / 8)."""
    src = (idx[:, BALL::PLAYER][:, :6] - BALL) // PLAYER  # (144, 6): source player of each slot
    inv = torch.empty_like(src)
    inv.scatter_(1, src, torch.arange(6, device=idx.device).expand_as(src).contiguous())
    return inv


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)


def target_classes(y: np.ndarray) -> np.ndarray:
    """(n, 2) TARGETS -> class 0 (A scores within 10 s), 1 (nobody), 2 (B)."""
    assert not (y[:, 0] * y[:, 1]).any(), "both teams score within 10 s on some row"
    return (1 - y[:, 0] + y[:, 1]).astype(np.int64)


def horizon_classes(labels: pl.DataFrame, horizons: list[float]) -> np.ndarray:
    """(n, len(horizons)) classes as in target_classes, for "scores within h seconds", from
    the META columns event_time (minus the time to the next goal) and team_scoring_next."""
    t = labels["event_time"].to_numpy()
    a = (labels["team_scoring_next"] == "A").fill_null(False).to_numpy()
    b = (labels["team_scoring_next"] == "B").fill_null(False).to_numpy()
    out = np.ones((len(t), len(horizons)), np.int64)
    for j, h in enumerate(horizons):
        out[a & (t >= -h), j] = 0
        out[b & (t >= -h), j] = 2
    return out


class HostNet:
    # META columns run.py passes to fit() as training labels (never inputs): the 1..9 s heads
    TRAIN_LABELS = ["event_time", "team_scoring_next", "player_scoring_next", "game_num"]

    def __init__(self, opponent_pairs="host", horizons=(10,), aux_weight=0.3, goal_features=False,
                 input_scale=False, warp_scale=None, width=1.0, dropout=0.0, head_dropout=0.0,
                 backbone="host", mlp_hidden=((256, 256), 128), frame="none", ema_epochs=None,
                 pair_geometry=False, pool="max", input_noise=0.0, attn=None, snapshots=1,
                 scorer_weight=0.0, ball_path=False, snapshot_dump=None, snapshot_dump_n=10, rounds=1,
                 snapshot_gap_epochs=1.0, distill=None, sam_rho=0.0,
                 augment=True, tta="flips", batch_size=168, optimizer=None, lr_schedule=None,
                 max_epochs=20.0, max_hours=None, eval_every_epochs=0.25, patience_epochs=None,
                 device="auto", compile=None, precision="fp32", eval_batch_size=32_768):
        self._init_kwargs = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        assert tta in ("flips", "none"), f"unknown tta {tta!r}"
        assert backbone in ("host", "mlp", "attn"), f"unknown backbone {backbone!r}"
        assert frame == "none" or frame in FRAMES, f"unknown frame {frame!r}"
        assert backbone == "host" or not (goal_features or input_scale or warp_scale or width != 1.0
                                          or pair_geometry or pool != "max"), \
            "goal_features / input_scale / warp_scale / width / pair_geometry are options of the host backbone"
        self.pair_geometry = bool(pair_geometry)
        self.pool = pool
        self.scorer_weight = float(scorer_weight)
        self.ball_path = bool(ball_path)
        self.rounds = int(rounds)
        assert self.rounds == 1 or backbone == "host", "rounds is an option of the host backbone"
        assert not self.ball_path or backbone == "host", "ball_path is an option of the host backbone"
        assert self.scorer_weight == 0 or (backbone == "host" and frame == "none"), \
            "scorer_weight needs the host backbone without frame averaging"
        self.distill = None if not distill else {"alpha": 0.5, "folds": 2, "teacher_epochs": 14.0, **distill}
        assert not self.distill or (self.scorer_weight == 0 and backbone == "host" and frame == "none"), \
            "distill: host backbone, no scorer head, no frame averaging"
        self.sam_rho = float(sam_rho)
        self.input_noise = float(input_noise)
        self.attn = {"dim": 96, "layers": 4, "heads": 4, **(attn or {})}
        self.opponent_pairs = opponent_pairs
        self.horizons = [float(h) for h in horizons]
        assert 10.0 in self.horizons, "the 10 s horizon is the target"
        self.i10 = self.horizons.index(10.0)
        self.loss_weights = [1.0 if h == 10 else float(aux_weight) for h in self.horizons]
        self.goal_features = bool(goal_features)
        self.input_scale = bool(input_scale)
        self.warp_scale = None if warp_scale is None else float(warp_scale)
        assert self.warp_scale is None or self.warp_scale > 0, "warp_scale must be > 0"
        self.width = float(width)
        self.dropout, self.head_dropout = float(dropout), float(head_dropout)
        self.backbone, self.mlp_hidden, self.frame = backbone, mlp_hidden, frame
        self.ema_epochs = None if ema_epochs is None else float(ema_epochs)
        self.snapshots, self.snapshot_gap = int(snapshots), float(snapshot_gap_epochs)
        self.snapshot_dump, self.snapshot_dump_n = snapshot_dump, int(snapshot_dump_n)
        self.snapshot_states = []
        self.augment = bool(augment)
        self.tta = tta
        self.batch_size = int(batch_size)
        self.optimizer = {"name": "adam", "lr": 2e-3, "eps": 1e-7, **(optimizer or {})}
        assert self.optimizer["name"] in ("adam", "adamw"), f"unknown optimizer {self.optimizer['name']!r}"
        assert self.optimizer["name"] == "adam" or "weight_decay" in self.optimizer, \
            "adamw needs an explicit weight_decay (torch's default is 0.01)"
        sched = lr_schedule or {"boundaries": [1, 3, 5, 7, 10, 13],
                                "values": [2e-3, 1.5e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]}
        self.boundaries = [float(b) for b in sched["boundaries"]]
        self.values = [float(v) for v in sched["values"]]
        assert len(self.values) == len(self.boundaries) + 1, "need one more value than boundaries"
        self.max_epochs = float(max_epochs)
        self.max_hours = None if max_hours is None else float(max_hours)
        self.eval_every_epochs = float(eval_every_epochs)
        self.patience_epochs = None if patience_epochs is None else float(patience_epochs)
        self.device = resolve_device(device)
        self.compile = compile if self.device.type == "cuda" else None
        assert precision in ("fp32", "tf32", "bf16"), f"unknown precision {precision!r}"
        self.precision = precision
        self.eval_batch_size = int(eval_batch_size)
        self._augs = tuple(a.to(self.device) for a in augmentations())
        self.net = None
        self.fit_info = {}

    def _lr(self, epoch: float) -> float:
        return self.values[sum(epoch >= b for b in self.boundaries)]

    def _build(self) -> nn.Module:
        n_heads = len(self.horizons)
        if self.backbone == "host":
            net = HostNetModule(self.opponent_pairs, n_heads, self.goal_features, self.input_scale,
                                self.warp_scale, self.width, self.dropout, self.head_dropout,
                                self.pair_geometry, self.pool, self.scorer_weight > 0, self.ball_path,
                                self.rounds)
        elif self.backbone == "attn":
            net = AttnModule(n_heads, int(self.attn["dim"]), int(self.attn["layers"]), int(self.attn["heads"]),
                             self.dropout)
        else:
            net = MLPModule(n_heads, self.mlp_hidden, self.dropout, self.head_dropout)
        return net if self.frame == "none" else FrameAverage(net, self.frame)

    def _optimizer(self) -> torch.optim.Optimizer:
        cfg = dict(self.optimizer)
        name = cfg.pop("name")
        cfg = {k: float(v) for k, v in cfg.items()}  # YAML reads "1e-3" as a string
        if name == "adam":
            return torch.optim.Adam(self.net.parameters(), **cfg)
        params = list(self.net.parameters())  # adamw: decay weight matrices only
        groups = [{"params": [p for p in params if p.ndim >= 2]},
                  {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, **cfg)

    @torch.no_grad()
    def _probs(self, X: torch.Tensor, tta: str, net: nn.Module | None = None) -> torch.Tensor:
        """(n, 3) class probabilities; with tta='flips', averaged over the 4 mirrors (unless the
        frame already averages over them)."""
        net = self.net if net is None else net
        net.eval()
        idx, sign, flip_y = self._augs
        if self.frame in ("flips", "all"):
            tta = "none"
        combos = [0] if tta == "none" else [c for c in range(len(idx)) if c % 36 == 0]  # identity perms
        batch = self.eval_batch_size // (len(FRAMES[self.frame]) if self.frame != "none" else 1)
        out = []
        for i in range(0, len(X), batch):
            xb = X[i:i + batch]
            acc = 0
            for c in combos:
                pr = torch.softmax(net(xb[:, idx[c]] * sign[c])[:, self.i10], dim=-1)
                acc = acc + (pr.flip(-1) if flip_y[c] else pr)
            out.append(acc / len(combos))
        return torch.cat(out)

    @torch.no_grad()
    def _probs_all(self, X: torch.Tensor) -> torch.Tensor:
        """(n, n_heads, 3) class probabilities for every horizon, averaged over the 4 mirrors (A / B
        swapped back after a y mirror) and over the snapshots if there are any."""
        idx, sign, flip_y = self._augs
        combos = [c for c in range(len(idx)) if c % 36 == 0] if self.tta == "flips" else [0]
        n_h = len(self.horizons)
        live = copy.deepcopy(self.net.state_dict())
        states = self.snapshot_states or [live]
        acc = torch.zeros(len(X), n_h, 3, device=self.device)
        for st in states:
            self.net.load_state_dict(st)
            self.net.eval()
            for i in range(0, len(X), self.eval_batch_size):
                xb = X[i:i + self.eval_batch_size]
                for c in combos:
                    pr = torch.softmax(self.net(xb[:, idx[c]] * sign[c])[:, :n_h].float(), dim=-1)
                    acc[i:i + len(xb)] += pr.flip(-1) if flip_y[c] else pr
        self.net.load_state_dict(live)
        return acc / (len(states) * len(combos))

    def _soft_targets(self, X_tr, y_tr, X_val, y_val, labels_tr) -> torch.Tensor:
        """Cross-fitted teachers: split the training games into `folds`; for each fold, train
        the same recipe (no distillation, `teacher_epochs` passes over its share) on the other
        folds and predict every horizon, with TTA and snapshots, on this fold's rows."""
        games = labels_tr["game_num"].to_numpy()
        ug = np.unique(games)
        fold_of = dict(zip(ug, np.random.default_rng(0).permutation(len(ug)) % int(self.distill["folds"])))
        fold = np.array([fold_of[g] for g in games])
        soft = torch.zeros(len(X_tr), len(self.horizons), 3, device=self.device)
        info = []
        for f in range(int(self.distill["folds"])):
            tr = fold != f
            kw = {**self._init_kwargs, "distill": None, "snapshot_dump": None,
                  "max_epochs": float(self.distill["teacher_epochs"])}
            teacher = HostNet(**kw)
            print(f"  teacher {f}: {tr.sum():,} rows ({len(np.unique(games[tr]))} games)", flush=True)
            teacher.fit(X_tr[tr], y_tr[tr], X_val, y_val, labels_tr=labels_tr.filter(pl.Series(tr)))
            rows = torch.from_numpy(np.flatnonzero(~tr)).to(self.device)
            soft[rows] = teacher._probs_all(torch.from_numpy(X_tr[~tr]).to(self.device))
            pv = teacher.predict_proba(X_val)
            pv = np.clip(pv, 1e-7, 1 - 1e-7)
            tv = float(-(y_val * np.log(pv) + (1 - y_val) * np.log(1 - pv)).mean())
            info.append({"fold": f, "val_tta": tv, "best_epoch": teacher.fit_info["best_epoch"]})
            print(f"  teacher {f}: val with TTA {tv:.5f}", flush=True)
            del teacher
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self._teacher_info = info
        return soft

    def _autocast(self):
        """bf16 autocast for training and in-training evals when `precision: bf16`."""
        if self.precision == "bf16":
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _val_score(self, Xv, yv, net: nn.Module | None = None) -> float:
        """The competition metric on val (no TTA): mean of the two binary log losses."""
        with self._autocast():
            pr = self._probs(Xv, "none", net)[:, [0, 2]].float().clamp(1e-7, 1 - 1e-7)
        return F.binary_cross_entropy(pr, yv).item()

    def _check_compiled_dropout(self, train_loss, xb, cb) -> None:
        """Dropout inside a CUDA graph must draw a new mask on every replay. Run the compiled
        training loss (forward + backward, as in training) a few times on one batch, on a copy
        of the net so the real one is untouched; the losses must differ after warm-up."""
        live = self.net
        self.net = copy.deepcopy(live)
        losses = []
        for _ in range(5):
            if self.compile == "reduce-overhead":
                torch.compiler.cudagraph_mark_step_begin()
            loss = train_loss(xb, cb)
            self.net.zero_grad(set_to_none=True)  # as in training: never accumulate into graph outputs
            loss.backward()
            losses.append(loss.item())
        self.net = live
        print(f"  compiled dropout check, losses on one batch: {[f'{v:.6f}' for v in losses]}", flush=True)
        assert len(set(losses[2:])) > 1, f"dropout masks repeat under compile={self.compile}: {losses}"

    def fit(self, X_tr, y_tr, X_val, y_val, labels_tr=None) -> None:
        assert X_tr.shape[1] == N_IN, f"expected the 54 `host` features, got {X_tr.shape[1]}"
        classes = target_classes(y_tr)[:, None]
        if len(self.horizons) > 1:
            assert labels_tr is not None, "horizons other than 10 s need run.py's labels_tr"
            classes = horizon_classes(labels_tr, self.horizons)
            assert (classes[:, self.i10] == target_classes(y_tr)).all(), "10 s labels disagree with TARGETS"
        Xt = torch.from_numpy(X_tr).to(self.device)
        ct = torch.from_numpy(classes).to(self.device)  # (n, n_heads)
        n_heads = len(self.horizons)
        if self.scorer_weight > 0:  # scorer slot (0-5) when a goal follows within 10 s, else -1
            assert labels_tr is not None, "scorer_weight needs run.py's labels_tr"
            sc = labels_tr["player_scoring_next"].fill_null(-1).to_numpy().astype(np.int64)
            sc[labels_tr["event_time"].to_numpy() < -10.0] = -1
            assert ((sc >= 0) == (target_classes(y_tr) != 1)).all(), "scorer labels disagree with TARGETS"
            ct = torch.cat([ct, torch.from_numpy(sc).to(self.device)[:, None]], dim=1)  # last column
            scorer_inv = scorer_label_map(self._augs[0]).to(self.device)
        Xv = torch.from_numpy(X_val).to(self.device)
        yv = torch.from_numpy(y_val.astype(np.float32)).to(self.device)
        aug_idx, aug_sign, aug_fy = self._augs
        weights = torch.tensor(self.loss_weights, device=self.device)
        # input_noise: Gaussian noise per input, in units of the input_scale divisors (none on `demoed`)
        noise_scale = torch.tensor(POS_SCALE + [VEL_SCALE] * 3
                                   + (POS_SCALE + [VEL_SCALE] * 3 + [100.0, 0.0]) * 6, device=self.device)

        soft = None
        if self.distill:
            assert labels_tr is not None and "game_num" in labels_tr.columns, "distill needs game_num"
            soft = self._soft_targets(X_tr, y_tr, X_val, y_val, labels_tr)
            alpha = float(self.distill["alpha"])

        self.net = self._build().to(self.device)
        opt = self._optimizer()
        matmul_precision = torch.get_float32_matmul_precision()  # restored after training
        if self.precision == "tf32":
            torch.set_float32_matmul_precision("high")

        def train_loss(xb, cb):  # sum over horizons of weight * mean cross-entropy, as in Keras
            with self._autocast():
                logits = self.net(xb).float()
            hc = cb[:, :n_heads]
            ce = F.cross_entropy(logits[:, :n_heads].flatten(0, 1), hc.flatten(), reduction="none")
            loss = (ce.view(hc.shape).mean(0) * weights).sum()
            if self.scorer_weight > 0:  # mean CE over rows with a goal within 10 s (static shapes)
                sc = cb[:, n_heads]
                valid = (sc >= 0).float()
                sce = F.cross_entropy(logits[:, n_heads:].reshape(len(sc), 6), sc.clamp_min(0), reduction="none")
                loss = loss + self.scorer_weight * (sce * valid).sum() / valid.sum().clamp_min(1.0)
            return loss

        def train_loss_soft(xb, cb, sb):  # distill: target = alpha * one-hot + (1 - alpha) * teacher
            with self._autocast():
                logits = self.net(xb).float()
            target = alpha * F.one_hot(cb, 3).float() + (1 - alpha) * sb
            ce = -(target * F.log_softmax(logits, dim=-1)).sum(-1)  # (B, n_heads)
            return (ce.mean(0) * weights).sum()

        if self.compile:
            train_loss = torch.compile(train_loss, mode=self.compile)
            train_loss_soft = torch.compile(train_loss_soft, mode=self.compile)
            if self.dropout > 0 or self.head_dropout > 0:
                rows = torch.arange(min(len(Xt), self.batch_size), device=self.device)
                if soft is None:
                    self._check_compiled_dropout(train_loss, Xt[rows], ct[rows])
                else:
                    self._check_compiled_dropout(lambda xb, cb: train_loss_soft(xb, cb, soft[rows]),
                                                 Xt[rows], ct[rows])

        n = len(Xt)
        steps_per_epoch = n // self.batch_size  # the host drops the last partial batch
        eval_every = max(1, round(self.eval_every_epochs * steps_per_epoch))
        max_steps = round(self.max_epochs * steps_per_epoch)
        patience = None if self.patience_epochs is None else self.patience_epochs * steps_per_epoch
        ema, ema_decay = None, None
        if self.ema_epochs is not None:  # a shadow copy; training never reads it
            ema = copy.deepcopy(self.net)
            ema_decay = 1 - 1 / (self.ema_epochs * steps_per_epoch)
            ema_t = [v for v in ema.state_dict().values() if v.is_floating_point()]
            live_t = [v for v in self.net.state_dict().values() if v.is_floating_point()]

        best, best_step, best_state = math.inf, 0, None
        pool = []  # snapshots > 1: (val, step, state) of every eval (~2 MB each for the 2x wide net)
        history, step, stop_reason, lr = [], 0, "max_epochs", None
        run_loss, run_n = torch.zeros((), device=self.device), 0
        t0 = time.time()
        while stop_reason == "max_epochs" and step < max_steps:
            perm = torch.randperm(n, device=self.device)
            for i in range(steps_per_epoch):
                new_lr = self._lr(step / steps_per_epoch)
                if new_lr != lr:
                    lr = new_lr
                    for g in opt.param_groups:
                        g["lr"] = lr
                self.net.train()
                rows = perm[i * self.batch_size:(i + 1) * self.batch_size]
                xb, cb = Xt[rows], ct[rows]
                if self.augment:
                    a = torch.randint(len(aug_idx), (len(rows),), device=self.device)
                    xb = xb.gather(1, aug_idx[a]) * aug_sign[a]
                    if self.scorer_weight > 0:
                        hc, sc = cb[:, :-1], cb[:, -1]
                        hc = torch.where(aug_fy[a, None], 2 - hc, hc)
                        sc = torch.where(sc >= 0, scorer_inv[a, sc.clamp_min(0)], sc)
                        cb = torch.cat([hc, sc[:, None]], dim=1)
                    else:
                        cb = torch.where(aug_fy[a, None], 2 - cb, cb)
                    if soft is not None:  # the teacher's class vectors reverse under a y mirror too
                        sb = soft[rows]
                        sb = torch.where(aug_fy[a, None, None], sb.flip(-1), sb)
                elif soft is not None:
                    sb = soft[rows]
                if self.input_noise > 0:
                    xb = xb + self.input_noise * noise_scale * torch.randn_like(xb)
                if self.compile == "reduce-overhead":
                    torch.compiler.cudagraph_mark_step_begin()
                loss = train_loss(xb, cb) if soft is None else train_loss_soft(xb, cb, sb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.sam_rho > 0:  # SAM: the gradient at w + rho * g / |g|, applied at w
                    with torch.no_grad():
                        params = [p for p in self.net.parameters() if p.grad is not None]
                        grads = [p.grad.detach().clone() for p in params]  # copy: graph-owned memory
                        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
                        eps = [g * (self.sam_rho / (norm + 1e-12)) for g in grads]
                        torch._foreach_add_(params, eps)
                    loss = loss.detach().clone()
                    if self.compile == "reduce-overhead":
                        torch.compiler.cudagraph_mark_step_begin()
                    loss2 = train_loss(xb, cb) if soft is None else train_loss_soft(xb, cb, sb)
                    opt.zero_grad(set_to_none=True)
                    loss2.backward()
                    with torch.no_grad():
                        torch._foreach_sub_(params, eps)
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(ema_t, live_t, 1 - min(ema_decay, (1 + step) / (10 + step)))
                run_loss += loss.detach()
                run_n += 1
                step += 1
                if step % eval_every and step < max_steps:
                    continue
                val = self._val_score(Xv, yv, ema)  # the EMA weights when there are any
                hours = (time.time() - t0) / 3600
                history.append({"step": step, "epoch": step / steps_per_epoch,
                                "train_loss": (run_loss / run_n).item(), "val_loss": val,
                                "lr": lr, "hours": hours})
                if ema is not None:
                    history[-1]["val_loss_raw"] = self._val_score(Xv, yv)
                run_loss.zero_(); run_n = 0
                if val < best:
                    best, best_step = val, step
                    best_state = copy.deepcopy((self.net if ema is None else ema).state_dict())
                if self.snapshots > 1 or self.snapshot_dump:
                    pool.append((val, step, best_state if best_step == step else
                                 copy.deepcopy((self.net if ema is None else ema).state_dict())))
                raw = "" if ema is None else f"  raw {history[-1]['val_loss_raw']:.5f}"
                print(f"  step {step:>8} ep {step / steps_per_epoch:6.2f}  train {history[-1]['train_loss']:.5f}"
                      f"  val {val:.5f}{raw}  best {best:.5f}@{best_step}  lr {lr:.1e}"
                      f"  {hours * 60:6.1f} min", flush=True)
                if patience is not None and step - best_step >= patience:
                    stop_reason = "patience"
                elif self.max_hours is not None and hours >= self.max_hours:
                    stop_reason = "max_hours"
                if stop_reason != "max_epochs" or step >= max_steps:
                    break

        torch.set_float32_matmul_precision(matmul_precision)  # final predictions in fp32
        if ema is not None:
            self.net = ema  # predictions use the averaged weights
        if self.snapshot_dump:
            self._dump_snapshots(pool, Xv, steps_per_epoch)
        self.net.load_state_dict(best_state)
        if self.snapshots > 1:
            chosen = self._select_snapshots(pool, steps_per_epoch, keep=self.snapshots)
            self.snapshot_states = [st for _, _, st in chosen]
        self.fit_info = {
            "precision": self.precision,
            "best_step": best_step,
            "best_epoch": best_step / steps_per_epoch,
            "stopped_step": step,
            "epochs": step / steps_per_epoch,
            "stop_reason": stop_reason,
            "best_is_last_eval": best_step == history[-1]["step"],
            "steps_per_epoch": steps_per_epoch,
            "eval_every": eval_every,
            "best_val_loss_no_tta": best,
            **({} if ema is None else {
                "ema_decay": ema_decay,
                "best_val_loss_raw_no_tta": min(h["val_loss_raw"] for h in history)}),
            **({} if self.snapshots <= 1 else {
                "snapshot_epochs": [round(st / steps_per_epoch, 2) for _, st, _ in chosen],
                "snapshot_val_no_tta": [round(v, 6) for v, _, _ in chosen]}),
            **({} if not self.distill else {"teachers": self._teacher_info}),
            "n_params": sum(p.numel() for p in self.net.parameters()),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name() if self.device.type == "cuda" else None,
            "compile": self.compile,
            "train_time_per_epoch_s": round(3600 * history[-1]["hours"] / (step / steps_per_epoch), 1),
            "history": history,
        }

    def _dump_snapshots(self, pool, Xv, steps_per_epoch) -> None:
        """Save the val probabilities (with TTA, float32) of the `snapshot_dump_n` best evals, with
        their passes and val losses without TTA, so snapshot rules can be compared offline."""
        from pathlib import Path
        top = sorted(pool, key=lambda t: t[0])[:self.snapshot_dump_n]
        probs = []
        for _, _, st in top:
            self.net.load_state_dict(st)
            probs.append(self._probs(Xv, self.tta)[:, [0, 2]].cpu().numpy().astype(np.float32))
        path = Path(self.snapshot_dump)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, probs=np.stack(probs), epochs=np.array([st / steps_per_epoch for _, st, _ in top]),
                 val_no_tta=np.array([v for v, _, _ in top]))
        print(f"  dumped {len(top)} snapshots' val probabilities -> {path}", flush=True)

    def _select_snapshots(self, pool, steps_per_epoch, keep):
        """Greedy: the lowest-val evals, each at least snapshot_gap passes from those already taken."""
        chosen = []
        for item in sorted(pool, key=lambda t: t[0]):
            if all(abs(item[1] - c[1]) >= self.snapshot_gap * steps_per_epoch for c in chosen):
                chosen.append(item)
            if len(chosen) == keep:
                break
        return chosen

    def predict_proba(self, X) -> np.ndarray:
        Xt = torch.from_numpy(X).to(self.device)
        if self.snapshots > 1:  # mean of the snapshots' (TTA) probabilities
            live = copy.deepcopy(self.net.state_dict())
            acc = 0
            for st in self.snapshot_states:
                self.net.load_state_dict(st)
                acc = acc + self._probs(Xt, self.tta)
            self.net.load_state_dict(live)
            return (acc / len(self.snapshot_states))[:, [0, 2]].cpu().numpy()
        p = self._probs(Xt, self.tta)
        return p[:, [0, 2]].cpu().numpy()
