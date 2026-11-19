"""Sanity checks for src/models/host_net.py; run before trusting a change to it.

    uv run python -m scripts.check_host_net

1. Both `opponent_pairs` modes, with or without the extra options, are exactly invariant to
   reordering players within a team. The in-net goal features match src/features/goal.py,
   and the 10 s horizon labels from event_time / team_scoring_next match TARGETS.
2. `host` mode reproduces the host's tf.roll: the "opponent" branch of a team-A player does
   not see team B at all; `fixed` mode's does.
3. The augmentation tables: index 0 is the identity, every entry is a signed column
   permutation, a y mirror moves team B into team A's slots with y negated, mirrors undo
   themselves, and an x mirror only negates x columns.
4. TTA over the mirrors returns A / B in the right order.
5. Frame averaging: the MLP backbone is not invariant to anything on its own; with
   `frame: all` it is invariant to all 144 transforms (A / B swapped under a y mirror), in
   train and eval mode; with `frame: perms` it is invariant to reorderings but not to the
   mirrors; the host net with `frame: flips` is mirror-equivariant.
"""
import itertools

import numpy as np
import torch

from src.models.host_net import BALL, PLAYER, AttnModule, FrameAverage, scorer_label_map, HostNet, HostNetModule, MLPModule, augmentations

torch.manual_seed(0)
X = torch.randn(512, 54, dtype=torch.float64) * 30


def permute_within_teams(x, pa, pb):
    out = x.clone()
    for team, perm in ((0, pa), (1, pb)):
        for slot, src in enumerate(perm):
            dst, srcc = BALL + PLAYER * (3 * team + slot), BALL + PLAYER * (3 * team + src)
            out[:, dst:dst + PLAYER] = x[:, srcc:srcc + PLAYER]
    return out


def check_invariance():
    variants = [("host", {}), ("fixed", {}),
                ("fixed", {"n_heads": 10, "goal_features": True, "input_scale": True, "warp_scale": 5.0}),
                ("fixed", {"pair_geometry": True}), ("host", {"pair_geometry": True, "width": 2.0}), ("fixed", {"pool": "maxmean", "pair_geometry": True}), ("fixed", {"ball_path": True, "pool": "maxmean"}), ("fixed", {"rounds": 2, "pool": "maxmean"}), ("host", {"rounds": 2}),
                ("attn", {"n_heads": 10}), ("attn", {"dropout": 0.0, "layers": 2})]
    for mode, opts in variants:
        net = (AttnModule(**opts) if mode == "attn" else HostNetModule(mode, **opts)).double()
        for train in (True, False):
            net.train(train)
            base = net(X)
            worst = max((net(permute_within_teams(X, pa, pb)) - base).abs().max().item()
                        for pa, pb in itertools.product(itertools.permutations(range(3)), repeat=2))
            assert worst < 1e-9, (mode, opts, train, worst)
        print(f"ok  {mode:5s} {opts}: invariant to within-team reordering (max diff {worst:.1e})")


def check_goal_features_and_horizons():
    """The in-net goal features equal src/features/goal.py on real rows, and the 10 s
    horizon labels rebuild TARGETS."""
    import polars as pl

    from src.data import PROCESSED_DIR, TARGETS
    from src.features.goal import GOAL_FEATURES, add_goal_features
    from src.features.host import HOST_FEATURES, add_host_features
    from src.models.host_net import horizon_classes, target_classes

    df = pl.read_parquet(PROCESSED_DIR / "val.parquet").head(50_000)
    x = torch.from_numpy(add_host_features(df).select(HOST_FEATURES).to_numpy().astype(np.float64))
    want = add_goal_features(df).select(GOAL_FEATURES).to_numpy().astype(np.float64).reshape(len(df), 7, 6)
    net = HostNetModule(goal_features=True).double()
    players = x[:, BALL:].reshape(-1, 2, 3, PLAYER)
    got = torch.cat([net._goal(x[:, None, :3], x[:, None, 3:BALL]),
                     net._goal(players[..., :3], players[..., 3:6]).reshape(-1, 6, 6)], dim=1).numpy()
    ok = ~np.isnan(want)  # goal.py gives nulls for demolished players (the net sees stand-ins)
    assert np.allclose(got[ok], want[ok], atol=1e-3), np.abs(got[ok] - want[ok]).max()
    from src.features.warp import add_warp_features, warp_features
    want = add_warp_features(df, 5.0).select(warp_features(5.0)).to_numpy().astype(np.float64).reshape(len(df), 7, 12)
    net = HostNetModule(warp_scale=5.0).double()
    got = torch.cat([net._warp(x[:, None, :3], x[:, None, 3:BALL]),
                     net._warp(players[..., :3], players[..., 3:6]).reshape(-1, 6, 12)], dim=1).numpy()
    ok = ~np.isnan(want)
    assert np.allclose(got[ok], want[ok], atol=1e-3), np.abs(got[ok] - want[ok]).max()
    y = df.select(TARGETS).to_numpy().astype(np.float32)
    h = horizon_classes(df.select(["event_time", "team_scoring_next"]), [1, 5, 10])
    assert (h[:, 2] == target_classes(y)).all() and (h[:, 0] != 1).sum() <= (h[:, 1] != 1).sum()
    print("ok  in-net goal and warp features match src/features/{goal,warp}.py; "
          "10 s horizon labels match TARGETS")


def check_opponent_branch():
    seen = {}
    for mode in ("host", "fixed"):
        net = HostNetModule(mode).double().eval()
        out = []
        net.opps.register_forward_hook(lambda m, i, o: out.append(o.amax(dim=3)))
        x2 = X.clone()
        x2[:, BALL + 3 * PLAYER:] += 5.0  # move every team-B player
        net(X), net(x2)
        seen[mode] = (out[1][:, 0] - out[0][:, 0]).abs().max().item()  # team A's opponent features
    assert seen["host"] == 0 and seen["fixed"] > 1e-3, seen
    print(f"ok  host mode's 'opponent' branch ignores team B; fixed mode's sees it "
          f"(change {seen['fixed']:.2f})")


def check_augmentations():
    idx, sign, flip_y = augmentations()
    assert idx.shape == sign.shape == (144, 54) and flip_y.sum() == 72
    assert (idx[0] == torch.arange(54)).all() and (sign[0] == 1).all() and not flip_y[0]
    for i, s in zip(idx, sign):
        assert sorted(i.tolist()) == list(range(54)) and set(s.abs().tolist()) == {1.0}
    names = ([f"ball_{c}" for c in ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z")]
             + [f"p{p}_{c}" for p in range(6)
                for c in ("pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "boost", "demoed")])
    apply = lambda c, x: x[:, idx[c]] * sign[c]
    fy, fx, both = 36, 72, 108  # identity player orders
    y = apply(fy, X)
    for j, n in enumerate(names):
        src = n if n.startswith("ball") else f"p{(int(n[1]) + 3) % 6}{n[2:]}"
        want = -X[:, names.index(src)] if n.endswith("_y") else X[:, names.index(src)]
        assert torch.equal(y[:, j], want), n
    assert flip_y[fy] and not flip_y[fx] and flip_y[both]
    for c in (fy, fx, both):
        assert torch.equal(apply(c, apply(c, X)), X)
    x = apply(fx, X)
    for j, n in enumerate(names):
        assert torch.equal(x[:, j], -X[:, j] if n.endswith("_x") else X[:, j]), n
    print("ok  augmentation tables: signed permutations; mirrors as specified and self-inverse")


def check_tta():
    model = HostNet(device="cpu", tta="flips")
    model.net = HostNetModule().eval()
    model._augs = augmentations()
    x = X.float()
    # mirror y by hand and swap A / B back: TTA over flips must give the same average
    _, sign, _ = model._augs
    p = model._probs(x, "flips")
    p_mirrored = model._probs(x[:, augmentations()[0][36]] * sign[36], "flips").flip(-1)
    assert torch.allclose(p, p_mirrored, atol=1e-5), (p - p_mirrored).abs().max()
    assert torch.allclose(p.sum(-1), torch.ones(len(p)), atol=1e-5)
    print("ok  TTA over mirrors is itself mirror-equivariant (A <-> B under a y mirror)")


def check_frames():
    idx, sign, flip_y = augmentations()

    def worst(net, elements):
        """Largest change in the logits over the given transforms, A / B swapped back after a y mirror."""
        base = net(X)
        return max((net(X[:, idx[c]] * sign[c]).flip(-1) if flip_y[c] else net(X[:, idx[c]] * sign[c]))
                   .sub(base).abs().max().item() for c in elements)

    mirrors, perms = [36, 72, 108], list(range(1, 36))
    torch.manual_seed(0)
    mlp = MLPModule(n_heads=2).double()
    for train in (True, False):
        mlp.train(train)
        assert worst(mlp, perms) > 1e-3 and worst(mlp, mirrors) > 1e-3  # no symmetry of its own
        fa_all = FrameAverage(mlp, "all").double().train(train)
        assert worst(fa_all, range(1, 144)) < 1e-9, worst(fa_all, range(1, 144))
        fa_perms = FrameAverage(mlp, "perms").double().train(train)
        assert worst(fa_perms, perms) < 1e-9 and worst(fa_perms, mirrors) > 1e-3
        fa_flips = FrameAverage(HostNetModule("fixed", n_heads=2).double(), "flips").train(train)
        assert worst(fa_flips, mirrors) < 1e-9 and worst(fa_flips, perms) < 1e-9
    print("ok  frame averaging: MLP + all invariant to the 144 transforms (train and eval mode); "
          "MLP + perms not mirror-invariant; host net + flips mirror-equivariant")


if __name__ == "__main__":
    check_invariance()
    check_goal_features_and_horizons()
    check_opponent_branch()
    check_augmentations()
    check_tta()
    check_frames()
    print(f"n_params {sum(p.numel() for p in HostNetModule().parameters()):,}")


def check_scorer_head():
    """The scorer labels follow the player through every augmentation, and the per-player
    scorer logits move with the players when they are reordered within a team."""
    idx, sign, _ = augmentations()
    inv = scorer_label_map(idx)
    for a in range(len(idx)):
        aug = X[:, idx[a]] * sign[a]
        for p in range(6):
            slot = inv[a, p].item()
            got = aug[:, BALL + PLAYER * slot:BALL + PLAYER * (slot + 1)].abs()
            want = X[:, BALL + PLAYER * p:BALL + PLAYER * (p + 1)].abs()
            assert torch.equal(got, want), (a, p, slot)
    net = HostNetModule("fixed", n_heads=10, scorer_head=True, pool="maxmean").double().eval()
    out = net(X)
    assert out.shape == (len(X), 12, 3)
    for pa, pb in itertools.product(itertools.permutations(range(3)), repeat=2):
        o2 = net(permute_within_teams(X, pa, pb))
        assert (o2[:, :10] - out[:, :10]).abs().max() < 1e-9
        sc, sc2 = out[:, 10:].reshape(-1, 6), o2[:, 10:].reshape(-1, 6)
        perm = list(pa) + [3 + q for q in pb]  # slot s of the permuted rows holds player perm[s]
        assert (sc2 - sc[:, perm]).abs().max() < 1e-9
    print("ok  scorer head: labels follow players through all 144 augmentations; per-player logits "
          "are equivariant to reordering")


check_scorer_head()


def check_soft_targets_follow_mirrors():
    """Distillation targets: _probs_all is equivariant to the y mirror at every horizon (class
    vectors reversed), so flipping a stored teacher vector under a y mirror, as training does,
    matches what the teacher would have said about the mirrored row."""
    h = HostNet(opponent_pairs="fixed", horizons=[1, 5, 10], device="cpu", pool="maxmean")
    h.net = h._build().double()
    idx, sign, flip_y = h._augs
    fy = 36  # y mirror, identity player orders
    x = X[:64]
    h.net = h.net.float()
    xf = x.float()
    p = h._probs_all(xf)
    pm = h._probs_all(xf[:, idx[fy]] * sign[fy])
    assert p.shape == (64, 3, 3) and torch.allclose(pm, p.flip(-1), atol=1e-5), (pm - p.flip(-1)).abs().max()
    print("ok  distill: teacher probabilities for every horizon reverse under a y mirror (soft targets follow the labels)")


check_soft_targets_follow_mirrors()
