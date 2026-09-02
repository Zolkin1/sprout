# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fit the trackability critic on closed-loop rollouts, and run the three gates.

Reads what ``scripts/rsl_rl/record_rollouts.py`` wrote -- one row per generator re-plan plus
one per episode -- and fits ``V(s) = -P(fail | s)`` as a Bernoulli failure classifier warm
started from the generator's own conditioning encoder.

**The split is by EPISODE and that is not a preference.**  With a terminal-only reward every
window of an episode carries the SAME label, so a window-level split puts near-duplicate
states either side of it and the held-out metrics measure memorisation.  For the same reason
the effective sample size is close to the EPISODE count, not the window count: 25 windows from
one episode are one label observed from 25 vantage points, and this script prints both numbers
so the second is never mistaken for the first.

Two variants are always fitted, and their gap is a measurement -- see
``robot_rl.flow_matching.value``.

Gates, in the order they should be read (``robot_rl.flow_matching.value_gates``):

1. **calibration**, which is testable even where discrimination is impossible;
2. **AUC by PHASE**, never in aggregate -- a critic that has merely learned which tile is hard
   scores a perfect aggregate AUC and a flat, high phase curve;
3. **lead time**, measured against the FALL and not just against the end. If the critic only
   rises once the base is past recovery, the advantage arrives after the chunk that caused the
   failure and nothing is attributable. It also sets ``n`` empirically.

Usage::

    python scripts/flow_matching/fit_value.py <rollout stem> \\
        --fm_checkpoint <fm.pt> --out logs/value/<name>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE.parent / "rsl_rl"),):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import rollout_outcomes as ro  # noqa: E402

from robot_rl.flow_matching import online, transforms, value, value_gates  # noqa: E402

#: Metres per unit on the goal-offset features. DECLARED, not fitted: a critic's notion of
#: "far from the goal" should be a property of the robot, not of whichever rollout happened to
#: be on disk when the normalizer ran. 5 m spans the approach on every family measured.
GOAL_SCALE_M: float = 5.0


def _load(stem: str, max_windows: int) -> tuple[dict, dict, dict]:
    """Read both tables and join them on ``episode_uid``.

    Args:
        stem: The rollout stem.
        max_windows: Cap, so a multi-gigabyte recording can be smoke-tested.

    Returns:
        ``(windows, episodes, per_window_labels)``.

    Raises:
        SystemExit: When the join is not exact. A window whose episode has no outcome row
            cannot be labelled, and silently dropping it would quietly change the training
            distribution.
    """
    # Only what the critic reads. `fm_plan` alone is 11.3 kB/window -- 1.6 GB over a 6k
    # episode rollout -- and the value function is a function of the STATE: the chunk enters
    # the advantage through s_{t+n}, never through V. Loading it would cost a third of the
    # file's RAM to hold an array nothing touches.
    W, wa = ro.read_rollout_windows(stem + ".h5", columns=[
        "episode_uid", "replan_index", "family_code", "history", "history_null", "extero",
        "cmd_future", "goal_downrange_m", "goal_lateral_m", "goal_height_err_m",
        "proprio", "base_twist_yaw", "height_scan",
        # The quality head's target. Requested BY NAME like everything else: this loader
        # takes an explicit column list (it never reads `fm_plan`, 11.3 kB/window), so a
        # column that is not named is simply absent and the head silently trains on nothing.
        # That is exactly what happened -- the target code was right and the fit reported no
        # quality metrics at all, twice, because nobody asked for the column.
        #
        # No `try` around it: `read_rollout_windows` RAISES on an absent column, and it
        # already refuses any file whose schema_version is not 2 -- and `track_err_prev` is
        # part of what schema 2 MEANS. So a v2 file without it is malformed, and failing
        # loudly is right.
        "track_err_prev"])
    E, ea = ro.read_rollout_episodes(stem + ".episodes.h5")
    if max_windows and W["episode_uid"].size > max_windows:
        keep = np.arange(max_windows)
        W = {k: v[keep] for k, v in W.items()}

    live = ~E["censored"].astype(bool)
    reward = ro.terminal_reward(E)
    # `first_bad` is what drop-after-latch below cuts on: the fall crossing or the strict
    # violation, whichever came FIRST. Taken from the same table `terminal_reward` scored, so
    # the label and the cut cannot disagree about when the episode ended.
    first_bad = ro.first_bad_step(E)
    outcome = {int(u): (float(r), bool(l), int(ff), int(fam), int(fb))
               for u, r, l, ff, fam, fb in zip(E["episode_uid"], reward, live,
                                               E["first_fall_step"], E["family_code"],
                                               first_bad)}
    missing = sorted({int(u) for u in W["episode_uid"]} - set(outcome))
    if missing:
        raise SystemExit(
            f"{len(missing)} window episode(s) have no outcome row (e.g. {missing[:5]}). The "
            "two tables must join exactly on episode_uid; dropping the unmatched ones would "
            "change the training distribution without saying so.")
    ep_terrain = {int(u): int(t) for u, t in zip(E["episode_uid"], E["terrain_id"])}
    lab = {
        "failed": np.array([outcome[int(u)][0] == -1.0 for u in W["episode_uid"]]),
        "live": np.array([outcome[int(u)][1] for u in W["episode_uid"]]),
        "first_fall_step": np.array([outcome[int(u)][2] for u in W["episode_uid"]]),
        "family_code": np.array([outcome[int(u)][3] for u in W["episode_uid"]]),
    }
    # DROP EVERY WINDOW AT OR AFTER THE FIRST BAD STEP -- the fall crossing or the strict
    # tracking violation, whichever came first.
    #
    # The recording LATCHES rather than terminating, which is what keeps the threshold an
    # offline knob: lowering a bound just moves the latch earlier and drops more rows, a
    # relabel instead of a re-collection. But every consumer downstream of the recording
    # treats a violation as a TERMINATION, so the training set has to be the one termination
    # would have produced -- windows `0 .. t-1`, labelled by the strict outcome. That is
    # bit-for-bit the same dataset, because under termination the later states never exist.
    #
    # Without this the fit is asked to do two impossible things at once. The post-latch
    # windows LOOK healthy -- the robot recovered, which is why the episode kept going -- yet
    # carry the failure label, so the survival head is fitted on states termination would
    # never have shown it. And the quality target for a PRE-latch window sums chunk rewards
    # from chunks that termination prevents, so `G` is contaminated by post-failure behaviour
    # rather than being absorbing at the failure. Measured on `v4_big`: 33,428 of 446,131
    # windows (7.49%), 20,580 of them after a violation.
    #
    # Placed BEFORE the return computation on purpose: `normalized_returns` recurses backwards
    # over each episode's remaining chunks, so dropping first is what makes the tail absorbing
    # instead of merely zeroed at the end.
    K = int(ea.get("fm_update_every", 12))
    first_bad = np.array([outcome[int(u)][4] for u in W["episode_uid"]])
    step_of = W["replan_index"].astype(np.int64) * K
    alive = ~((first_bad >= 0) & (step_of >= first_bad))
    if not alive.all():
        n_ep_touched = int(np.unique(W["episode_uid"][~alive]).size)
        print(f"[fit] drop-after-latch: {int((~alive).sum())}/{alive.size} windows "
              f"({100 * (~alive).mean():.2f}%) at or after the first bad step, over "
              f"{n_ep_touched} episode(s). These do not exist under termination.")
        W = {k: v[alive] for k, v in W.items()}
        lab = {k: v[alive] for k, v in lab.items()}

    # The QUALITY head's target. Unconditional: the column is part of schema 2 and the
    # reader refuses anything else, so a guard here would be dead code pretending to be a
    # fallback. `--quality_bins 0` is how the survival-only control arm is selected.
    from robot_rl.flow_matching import advantage as adv_mod  # noqa: PLC0415
    spec = _tracking_spec_from(ea)
    r, r_ok = adv_mod.chunk_rewards_from_track_err(
        W["track_err_prev"], W["episode_uid"], W["replan_index"], spec)
    g, g_ok = adv_mod.normalized_returns(
        r, r_ok, W["episode_uid"], W["replan_index"], lab["failed"],
        gamma=float(spec.gamma))
    lab["chunk_reward"], lab["g"], lab["g_valid"] = r, g, g_ok
    print(f"[fit] quality target from track_err_prev: {int(g_ok.sum())}/{g_ok.size} "
          f"windows, median G {np.median(g[g_ok]):.4f}, "
          f"spec {'FITTED' if not spec.is_bootstrap else 'BOOTSTRAP'}")
    return W, {"cols": E, "attrs": ea, "ep_terrain": ep_terrain}, lab


def _tracking_spec_from(episode_attrs: dict):
    """The spec the recording's strict columns were written under.

    Read from the FILE, never re-defaulted: the recorder stamps it precisely so the columns
    are interpretable without knowing which code ran, and silently substituting today's
    default would relabel someone else's artifact.

    Args:
        episode_attrs: Attrs from the episodes table.

    Returns:
        A ``TrackingErrorSpec``.

    Raises:
        SystemExit: If the recording carries no spec. That is a pre-stamp file, and guessing
            one would make its `first_violation_step` mean something it never meant.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import (  # noqa: PLC0415
        TrackingErrorSpec,
    )

    raw = episode_attrs.get("tracking_spec")
    if raw is None:
        raise SystemExit(
            "this rollout has a track_err_prev block but no `tracking_spec` attr, so the "
            "thresholds and kernel widths its columns were written under are unknown. "
            "Re-record, or fit with --quality_bins 0 to train the survival head alone.")
    return TrackingErrorSpec.from_dict(raw)


def _splits(uids: np.ndarray, tiles: np.ndarray, val_frac: float, seed: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Two validation masks, and the DIFFERENCE between them is the measurement.

    * ``val_ep`` -- held-out EPISODES from tiles the critic trained on. Splitting by episode
      and not by window is already necessary: with a terminal-only reward every window of an
      episode carries the same label, so a window split puts near-duplicate states either
      side of it and the held-out score measures memorisation of the episode.
    * ``val_tile`` -- held-out TILES, which the critic has never seen at all.

    The gap between the two answers the question the phase curve CANNOT. A high AUC on the
    first chunk of an episode has two readings -- the critic has memorised which tile is
    hard, or the depth image genuinely shows the terrain ahead and the outcome really is
    partly predictable before anything has happened -- and those call for opposite responses.
    Held-out tiles separate them: under memorisation the tile-held-out score collapses, under
    real perception it holds.

    Args:
        uids: ``[n]`` per-window episode ids.
        tiles: ``[n]`` per-window terrain tile ids.
        val_frac: Fraction held out, of each.
        seed: RNG seed.

    Returns:
        ``(val_episodes, val_tiles)``, each ``[n]`` bool. Disjoint: the episode split is
        drawn only from TRAINING tiles, so the two measure different things rather than
        overlapping.
    """
    rng = np.random.default_rng(seed)
    tl = np.unique(tiles)
    n_t = max(1, int(round(val_frac * tl.size)))
    held_tiles = set(rng.choice(tl, size=min(n_t, tl.size), replace=False).tolist())
    val_tile = np.array([int(t) in held_tiles for t in tiles])

    ep_pool = np.unique(uids[~val_tile])
    n_e = max(1, int(round(val_frac * ep_pool.size)))
    held_ep = set(rng.choice(ep_pool, size=min(n_e, ep_pool.size), replace=False).tolist())
    val_ep = np.array([int(u) in held_ep for u in uids]) & ~val_tile
    return val_ep, val_tile


def _batch(ctx, W: dict, idx: np.ndarray, priv: bool, device,
           scan_hw) -> value.ValueInputs:
    """Build one normalized batch from raw recorded rows.

    Normalized with the GENERATOR's own normalizers, so the critic sees exactly what the
    generator saw. Re-fitting them here would put the two models in different input spaces
    while every shape still agreed.
    """
    to = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)  # noqa: E731
    hist = ctx.state_norm.normalize(to(W["history"][idx]))
    ext = ctx.extero_norm.normalize(to(W["extero"][idx]).reshape(len(idx), -1))
    cmd = None
    if ctx.cmd_dim > 0:
        steps = torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=device)
        cmd = ctx.cmd_norm.normalize(to(W["cmd_future"][idx])[:, steps, :])
    scan = proprio = None
    if priv:
        # The height scan records -inf for a MISSED RAY, and one non-finite value makes the
        # whole batch's loss NaN.
        # `sanitize_extero` is the generator's own guard for exactly this.
        scan = transforms.sanitize_extero(
            to(W["height_scan"][idx]), "height_scan").reshape(len(idx), *scan_hw)
        proprio = to(W["proprio"][idx])
    # Goal offsets on a DECLARED metre scale rather than a fitted statistic, so a critic's
    # notion of "far" is a property of the robot and not of whichever rollout was on disk --
    # the same idiom as the generator's `contract` normalization.
    goal = to(np.stack([W["goal_downrange_m"][idx], W["goal_lateral_m"][idx],
                        W["goal_height_err_m"][idx]], axis=-1)) / GOAL_SCALE_M
    goal = torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0)
    twist = to(W["base_twist_yaw"][idx])
    # ONBOARD keeps only the ANGULAR half: an IMU measures body rates and the yaw frame is
    # reachable from those plus its own roll/pitch, but nothing on the robot measures the
    # linear half without odometry. Slicing it away here is what makes the two variants'
    # gap a statement about information rather than about architecture.
    ctx = torch.cat([goal, twist if priv else twist[:, 3:]], dim=-1)
    return value.ValueInputs(history=hist, extero=ext, cmd_tokens=cmd,
                             history_null=to(W["history_null"][idx], torch.bool),
                             scan=scan, proprio=proprio, context=ctx)


def _predict(model, ctx, W, idx, priv, device, scan_hw, batch):
    """``P(fail)`` over ``idx``, in eval mode and without gradients."""
    was = model.training
    model.eval()
    out = np.zeros(idx.size, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, idx.size, batch):
            j = idx[i:i + batch]
            out[i:i + j.size] = model.p_fail(
                _batch(ctx, W, j, priv, device, scan_hw)).cpu().numpy()
    model.train(was)
    return out


@torch.no_grad()
def _predict_quality(model, ctx, W, idx, priv, device, scan_hw, batch: int) -> np.ndarray:
    """``G_hat`` over ``idx``, batched, in eval mode.

    Args:
        model: The critic.
        ctx: The FM context.
        W: The window table.
        idx: Row indices.
        priv: Privileged variant.
        device: Torch device.
        scan_hw: Height-scan grid.
        batch: Rows per forward pass.

    Returns:
        ``[len(idx)]`` predicted normalized returns.
    """
    was = model.training
    model.eval()
    out = np.empty(idx.size, dtype=np.float64)
    for i in range(0, idx.size, batch):
        j = idx[i:i + batch]
        out[i:i + j.size] = model.quality(
            _batch(ctx, W, j, priv, device, scan_hw)).cpu().numpy()
    model.train(was)
    return out


def _fit(ctx, W, lab, train, val, *, priv: bool, epochs: int, batch: int, lr: float,
         device, label: str, scan_hw, wb=None, val_tile=None,
         quality_bins: int = 0, quality_weight: float = 1.0
         ) -> tuple[value.FlowValueModel, np.ndarray]:
    """Train one variant and return it plus held-out ``P(fail)``.

    Validation runs EVERY epoch, on both scopes. Without it the only visible signal is the
    training loss, which falls whether the model is learning the task or the tiles -- so
    there is no way to see overfitting happen, and no basis for choosing the epoch count
    other than guessing. The per-epoch unseen-tile AUC is the curve that shows it.
    """
    import copy  # noqa: PLC0415

    model = value.FlowValueModel(
        copy.deepcopy(ctx.model),
        scan_hw=(tuple(scan_hw) if priv else None),
        proprio_dim=(int(W["proprio"].shape[-1]) if priv else 0),
        proprio_steps=(int(W["proprio"].shape[1]) if priv else 0),
        context_dim=(3 + (6 if priv else 3)),
        quality_bins=int(quality_bins),
    ).to(device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    y = torch.as_tensor(lab["failed"].astype(np.float32), device=device)
    # The QUALITY target. `g_valid` excludes the last window of every episode (its chunk's
    # consequences were never recorded) and any episode with no usable reward at all; those
    # rows train the survival head only, rather than being handed a zero -- which is the worst
    # possible return and would teach the critic that every episode's last decision was bad.
    use_q = int(quality_bins) > 0 and lab.get("g") is not None
    g_t = (torch.as_tensor(lab["g"].astype(np.float32), device=device) if use_q else None)
    g_ok = (torch.as_tensor(lab["g_valid"].astype(bool), device=device) if use_q else None)
    if use_q:
        print(f"  [{label}] quality head on: {int(lab['g_valid'].sum())}/{lab['g_valid'].size} "
              f"windows carry a return target "
              f"(median G {np.median(lab['g'][lab['g_valid']]):.3f})")
    tr = np.flatnonzero(train)
    rng = np.random.default_rng(0)
    best_state, best_auc, best_epoch, last_auc = None, -1.0, 0, float("nan")
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        tot = 0.0
        for i in range(0, tr.size, batch):
            j = tr[i:i + batch]
            opt.zero_grad(set_to_none=True)
            xb = _batch(ctx, W, j, priv, device, scan_hw)
            fail_logit, q_logits = model.heads(xb)     # ONE encoder pass for both heads
            loss = lossf(fail_logit, y[j])
            if use_q and q_logits is not None:
                jt = torch.as_tensor(j, device=device)
                m = g_ok[jt]
                if bool(m.any()):
                    loss = loss + float(quality_weight) * value.quality_loss(
                        q_logits[m], g_t[jt][m], model.quality_support)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[{label}] the loss went non-finite on epoch {ep + 1}. Before this "
                    "check the run printed 'BCE nan' three times and went on to report an "
                    "aggregate AUC of 0.647 from an untrained head -- a number produced by "
                    "a broken fit is worse than no number. The usual cause is a non-finite "
                    "input: the height scan carries -inf for missed rays.")
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * j.size
        bce = tot / max(tr.size, 1)
        # Named for what it IS. With the quality head on this is BCE + w * cross-entropy over
        # 51 bins, and the second term's floor is the target's own entropy rather than zero --
        # so it reads ~3.4 where the survival-only loss reads ~0.31, and calling it "BCE"
        # invites reading a working two-head fit as a broken one.
        name = "loss (BCE + quality CE)" if use_q else "BCE"
        print(f"  [{label}] epoch {ep + 1}/{epochs}  train {name} {bce:.4f}")
        log = {f"{label}/train_{'loss' if use_q else 'bce'}": bce, f"{label}/epoch": ep + 1}
        y_np = lab["failed"]
        for scope, m in (("val_seen_tiles", val), ("val_UNSEEN_tiles", val_tile)):
            if m is None or not m.any():
                continue
            j = np.flatnonzero(m)
            a = value_gates.auc(_predict(model, ctx, W, j, priv, device, scan_hw, batch),
                                y_np[j])
            log[f"{label}/{scope}/auc_epoch"] = a
            print(f"      {scope} AUC {a:.3f}")
            if use_q:
                # The quality head needs DIFFERENT questions from the classifier, and the
                # first is whether it did anything: r2 is measured against the CONSTANT
                # predictor, so <= 0 means it collapsed to the mean -- a real stationary
                # point here, and one whose calibration curve is a single point on the
                # diagonal (measured ECE 0.00000), so calibration alone would pass it. rho is
                # what the selector needs, since best-of-K takes an argmax and only ordering
                # matters.
                jq = j[lab["g_valid"][j]]
                if jq.size:
                    qg = value_gates.return_gates(
                        _predict_quality(model, ctx, W, jq, priv, device, scan_hw, batch),
                        lab["g"][jq])
                    log.update({f"{label}/{scope}/q_r2": qg["r2"],
                                f"{label}/{scope}/q_rho": qg["rho"],
                                f"{label}/{scope}/q_rmse": qg["rmse"]})
                    print(f"      {scope} quality r2 {qg['r2']:+.3f}  rho {qg['rho']:.3f} "
                          f"(shuffled {qg['rho_shuffled']:+.3f})  rmse {qg['rmse']:.4f}  "
                          f"pred_sd {qg['pred_std']:.3f} vs target_sd {qg['target_std']:.3f}")
            if scope == "val_seen_tiles" and np.isfinite(a):
                last_auc = a
                if a > best_auc:
                    best_auc, best_epoch = a, ep + 1
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
        if wb is not None:
            wb.log(log)
    if best_state is not None:
        # Select on the fold's OWN validation episodes -- seen tiles, unseen episodes. The
        # unseen-TILE score is the generalisation measure and must not also be the selection
        # criterion, or it stops being a holdout. With folds every window gets scored by a
        # model that did not train on it, so this split costs nothing.
        print(f"  [{label}] restoring epoch {best_epoch} (val AUC {best_auc:.3f} against "
              f"{last_auc:.3f} at the end)")
        model.load_state_dict(best_state)
    model.eval()
    out = np.zeros(W["episode_uid"].size, dtype=np.float64)
    with torch.no_grad():
        allidx = np.arange(out.size)
        for i in range(0, out.size, batch):
            j = allidx[i:i + batch]
            out[j] = model.p_fail(_batch(ctx, W, j, priv, device, scan_hw)).cpu().numpy()
    return model, out


def main() -> int:
    """Entry point."""
    # Line-buffer stdout. A fit is a ~1 h job that is normally redirected to a file, where
    # python block-buffers stdout -- so every `[fit]` line sits in a 8 kB buffer while wandb's
    # own output (stderr, unbuffered) lands immediately. The log then shows the wandb banner
    # and nothing else, which reads as "still loading" when the job is several epochs in.
    # That misread has now cost twice, here and in `record_rollouts.py`'s progress print.
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stem", help="rollout stem: <stem>.h5 and <stem>.episodes.h5")
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.25,
                   help="Fraction of TILES held out entirely, and (at --folds 1) of episodes.")
    p.add_argument("--quality_bins", type=int, default=51,
                   help="Bins for the QUALITY head's distribution over the normalized "
                        "tracking return. 0 trains the survival head ALONE, which is "
                        "bit-for-bit the model that shipped and is therefore the control arm. "
                        "Ignored on a rollout with no track_err_prev block.")
    p.add_argument("--quality_weight", type=float, default=1.0,
                   help="Weight on the quality cross-entropy against the survival BCE. Both "
                        "are per-sample means over comparable ranges, so 1.0 is a real "
                        "starting point rather than a placeholder -- but the two heads share "
                        "a frozen trunk and only their own readouts, so they cannot fight for "
                        "capacity the way two heads on one token would.")
    p.add_argument("--folds", type=int, default=1,
                   help="K-fold cross-fit over EPISODES, for labelling. At K>1 every window "
                        "on a training tile gets an OUT-OF-FOLD prediction from a model that "
                        "never saw its episode -- which is what stops the advantage from "
                        "being a lookup of the realized outcome. Costs the same wall-clock "
                        "as K=1 (each fold trains on 1/K less data), at the price of each "
                        "fold model seeing less.")
    p.add_argument("--variants", type=str, default="V_priv,V_onboard",
                   help="Which critics to fit, comma-separated.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Opt-OUT, matching `scripts/flow_matching/train.py` and the RL agent cfgs.
    p.add_argument("--wandb", action="store_true", default=True,
                   help="Log to Weights & Biases (default; use --no_wandb to disable).")
    p.add_argument("--no_wandb", dest="wandb", action="store_false",
                   help="Disable W&B logging.")
    p.add_argument("--wandb_project", type=str, default="robot_rl_value")
    p.add_argument("--run_name", type=str, default="",
                   help="W&B run name; defaults to the output directory's basename.")
    args = p.parse_args()

    stem = args.stem[:-3] if args.stem.endswith(".h5") else args.stem
    stem = stem[:-9] if stem.endswith(".episodes") else stem
    W, E, lab = _load(stem, args.max_windows)
    keep = lab["live"]
    if not keep.all():
        # Censored episodes have NO outcome: they neither fell nor failed to arrive, they were
        # cut when the rollout hit its ceiling. Scoring them would teach the critic that
        # whatever was happening at the ceiling predicts failure, which is a property of the
        # harness.
        print(f"[fit] dropping {int((~keep).sum())} window(s) from censored episodes")
        W = {k: v[keep] for k, v in W.items()}
        lab = {k: v[keep] for k, v in lab.items()}

    # Resolved ONCE and carried, because three places need it -- the checkpoint stamp, the
    # sidecar's gamma, and the printed provenance -- and re-deriving it per fold would let
    # them disagree if the derivation ever grew a default.
    track_spec = _tracking_spec_from(E["attrs"])
    uids = W["episode_uid"]
    n_ep = int(np.unique(uids).size)
    n_fail_ep = int(np.unique(uids[lab["failed"]]).size)
    print(f"[fit] {uids.size} windows over {n_ep} episodes "
          f"({n_fail_ep} failed, {100 * n_fail_ep / max(n_ep, 1):.1f}%)")
    print(f"[fit] EFFECTIVE sample size is the EPISODE count ({n_ep}), not the window count: "
          "every window of an episode carries the same label.")

    tiles = np.array([E["ep_terrain"].get(int(u), -1) for u in uids])
    val_ep, val_tile = _splits(uids, tiles, args.val_frac, args.seed)
    train = ~val_ep & ~val_tile
    print(f"[fit] {int(np.unique(tiles).size)} tiles: "
          f"{int(np.unique(tiles[train]).size)} train / "
          f"{int(np.unique(tiles[val_tile]).size)} HELD OUT")
    print(f"[fit] episodes: {int(np.unique(uids[train]).size)} train / "
          f"{int(np.unique(uids[val_ep]).size)} val (seen tiles) / "
          f"{int(np.unique(uids[val_tile]).size)} val (unseen tiles)")

    if args.folds < 1:
        raise SystemExit(f"--folds must be at least 1, got {args.folds}")
    if args.folds == 1:
        folds = [(train, val_ep)]
    else:
        # Partition the TRAINING-tile episodes into K folds. The tile holdout is excluded
        # from every fold's training set, so it stays a clean generalisation measure.
        pool = np.unique(uids[~val_tile])
        rng_f = np.random.default_rng(args.seed + 1)
        assign = {int(u): int(i % args.folds)
                  for i, u in enumerate(rng_f.permutation(pool))}
        fold_of = np.array([assign.get(int(u), -1) for u in uids])
        folds = []
        for k in range(args.folds):
            va_k = (fold_of == k)
            folds.append(((~val_tile) & ~va_k, va_k))
        sizes = [int(np.unique(uids[v]).size) for _, v in folds]
        print(f"[fit] {args.folds}-fold cross-fit over episodes: fold sizes {sizes}")

    scan_hw = E["attrs"].get("height_scan_hw")
    if not scan_hw:
        raise SystemExit(
            "the recording does not state the height scan's (height, width). 176 rays is "
            "both 11x16 and 16x11, and under a convolution a transposed grid is scrambled "
            "input rather than a reordered vector -- so it is refused rather than guessed. "
            "Re-record; `record_rollouts.py` reads it from the sensor's own contract.")
    # Never let telemetry take down a fit. An uninstalled wandb, a missing login or a
    # network blip would otherwise kill the run before its first epoch; the JSON and the
    # npz are written regardless.
    wb = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415
            wb = wandb.init(project=args.wandb_project,
                            name=args.run_name or os.path.basename(os.path.abspath(args.out)),
                            config={"stem": stem, "fm_checkpoint": args.fm_checkpoint,
                                    "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                                    "val_frac": args.val_frac, "seed": args.seed,
                                    "n_episodes": n_ep, "n_windows": int(uids.size),
                                    "n_failed_episodes": n_fail_ep,
                                    "n_tiles": int(np.unique(tiles).size)})
            # Declare epoch as the x-axis, one per (variant, fold). Without this every
            # series is plotted against wandb's auto-incrementing `_step`, and because the
            # variants and folds train SEQUENTIALLY into one run they land on disjoint parts
            # of that axis -- V_priv on steps 0..N-1 and V_onboard on N..2N-1. Each curve
            # then looks like a stub and overlaying them looks offset, which is a plotting
            # artefact rather than anything about the models.
            for label in ("V_priv", "V_onboard"):
                for k in range(max(int(args.folds), 1)):
                    tag = label if args.folds == 1 else f"{label}/f{k}"
                    wandb.define_metric(f"{tag}/epoch")
                    wandb.define_metric(f"{tag}/*", step_metric=f"{tag}/epoch")
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}\n"
                  f"       The fit continues; gates.json and predictions.npz still land in "
                  f"{args.out}.\n"
                  "       Pass --no_wandb to silence this, or fix the login (wandb login).")

    ctx = online.load_fm_context(args.fm_checkpoint, device=args.device)
    results = {}
    preds = {}
    qpreds = {}
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in wanted if v not in ("V_priv", "V_onboard")]
    if unknown:
        raise SystemExit(f"--variants: unknown {unknown}; choose from V_priv, V_onboard")
    os.makedirs(args.out, exist_ok=True)
    for label, priv in (("V_priv", True), ("V_onboard", False)):
        if label not in wanted:
            continue
        print(f"[fit] {label}")
        p_all = np.zeros(uids.size, dtype=np.float64)
        # The QUALITY head's out-of-fold prediction, assembled exactly like p_fail so the two
        # are on the same footing when `label_advantage` combines them into one V.
        q_all = (np.zeros(uids.size, dtype=np.float64) if int(args.quality_bins) > 0 else None)
        for k, (tr_k, va_k) in enumerate(folds):
            if len(folds) > 1:
                print(f"  [{label}] fold {k + 1}/{len(folds)}  "
                      f"{int(np.unique(uids[tr_k]).size)} train episodes")
            model, p_k = _fit(ctx, W, lab, tr_k, va_k,
                              quality_bins=int(args.quality_bins),
                              quality_weight=float(args.quality_weight),
                              priv=priv, epochs=args.epochs,
                              batch=args.batch, lr=args.lr, device=args.device,
                              label=(label if len(folds) == 1 else f"{label}/f{k}"),
                              scan_hw=scan_hw, wb=wb, val_tile=val_tile)
            # OUT OF FOLD: each window takes the prediction of the model that did not train
            # on its episode. The held-out TILES are out-of-sample for every fold, so they
            # take the mean -- averaging there is a free variance reduction, and doing it on
            # a training fold would silently reintroduce the in-sample reading.
            p_all[va_k] = p_k[va_k]
            p_all[val_tile] += p_k[val_tile] / len(folds)
            if q_all is not None:
                q_k = _predict_quality(model, ctx, W, np.arange(uids.size), priv,
                                       args.device, scan_hw, args.batch)
                q_all[va_k] = q_k[va_k]
                q_all[val_tile] += q_k[val_tile] / len(folds)
            # Checkpoint it. Without this the fit is write-only: the gates land on disk but
            # the thing that produced them does not, so labelling a rollout or scoring
            # candidate chunks would mean re-fitting from scratch.
            name = f"{label}.pt" if len(folds) == 1 else f"{label}.fold{k}.pt"
            # The QUALITY head's shape and the reward it was fitted against travel WITH the
            # checkpoint. `quality_bins` because a consumer has to build the head before it
            # can load into it, and a strict `load_state_dict` on a mis-sized head fails with
            # a shape error rather than a diagnosis. `tracking_spec` because the head predicts
            # a return under one set of sigmas, bounds and gamma: a selector that measures the
            # reward under a DIFFERENT spec and adds it to this head's prediction is summing
            # two different quantities, and both are plausible floats. This is the fourth
            # instance in this repo of "a checkpoint's saved config is the only record of what
            # it was trained against"; see CLAUDE.md on `extero_clip` for the first.
            torch.save({"model": model.state_dict(), "variant": label,
                        "fm_checkpoint": os.path.abspath(args.fm_checkpoint),
                        "scan_hw": list(scan_hw), "proprio_hw": list(W["proprio"].shape[1:]),
                        "context_dim": int(3 + (6 if priv else 3)), "privileged": bool(priv),
                        "epochs": int(args.epochs), "stem": stem,
                        "fold": int(k), "num_folds": int(len(folds)),
                        "quality_bins": int(args.quality_bins),
                        "quality_weight": float(args.quality_weight),
                        "tracking_spec": track_spec.to_dict(),
                        "gamma": float(track_spec.gamma)},
                       os.path.join(args.out, name))
            del model
        preds[label] = p_all
        if q_all is not None:
            qpreds[label] = q_all
        results[label] = {}
        # With folds the seen-tile scope is EVERY training-tile window, each scored out of
        # fold -- a far larger sample than the single held-out episode split it replaces.
        seen_scope = (~val_tile) if len(folds) > 1 else val_ep
        for scope, mask in (("val_seen_tiles", seen_scope), ("val_UNSEEN_tiles", val_tile)):
            print(f"  -- {scope} --")
            results[label][scope] = _gates(p_all, lab, W, mask)
        if wb is not None:
            for scope in ("val_seen_tiles", "val_UNSEEN_tiles"):
                r = results[label][scope]
                lead = r["lead_to_fall"]["lead"]
                good = np.isfinite(lead)
                wb.log({
                    f"{label}/{scope}/auc": r["auc"],
                    f"{label}/{scope}/ece": r["calibration"]["ece"],
                    f"{label}/{scope}/mce": r["calibration"]["mce"],
                    # The phase curve as separate scalars: a flat, high one is the shape that
                    # matters, and it is unreadable folded into a single number.
                    **{f"{label}/{scope}/auc_phase{i}": (None if np.isnan(a) else float(a))
                       for i, a in enumerate(r["auc_by_phase"]["auc"])},
                    f"{label}/{scope}/lead_to_fall_chunks":
                        (float(np.nanmedian(lead[good])) if good.any() else None),
                    f"{label}/{scope}/lead_warned_frac": float(good.mean()) if lead.size else None,
                })
        a_seen = results[label]["val_seen_tiles"]["auc"]
        a_new = results[label]["val_UNSEEN_tiles"]["auc"]
        print(f"  generalisation gap (seen - unseen tiles): {a_seen - a_new:+.3f}"
              + ("   <-- large: the critic is reading the TILE, not the state"
                 if a_seen - a_new > 0.10 else ""))

    gap = float("nan")
    if "V_priv" in preds and "V_onboard" in preds:
        gap = float(np.mean(np.abs(preds["V_priv"] - preds["V_onboard"])))
        print(f"\n[fit] V_priv / V_onboard mean |dP| = {gap:.4f}  "
              "(how much outcome-relevant information the generator cannot see)")
    with open(os.path.join(args.out, "gates.json"), "w") as f:
        json.dump({"gap_mean_abs": gap, "n_episodes": n_ep, "n_windows": int(uids.size),
                   "results": results}, f, indent=2, default=_jsonable)
    np.savez(os.path.join(args.out, "predictions.npz"), episode_uid=uids,
             replan_index=W["replan_index"], val=val_ep, val_tile=val_tile, tile=tiles,
             failed=lab["failed"],
             first_fall_step=lab["first_fall_step"], family_code=lab["family_code"],
             num_folds=int(args.folds),
             # The quality half of the sidecar: the head's out-of-fold G_hat, the MEASURED
             # per-chunk reward, and the realized return with its validity mask. Saved so
             # `label_advantage` can form the combined V without re-reading a 12 GB rollout
             # -- and so the shift out of `track_err_prev` is undone exactly once, here.
             chunk_reward=lab["chunk_reward"], g=lab["g"], g_valid=lab["g_valid"],
             gamma=float(track_spec.gamma),
             **{f"p_{k}": v for k, v in preds.items()},
             **{f"q_{k}": v for k, v in qpreds.items()})
    if wb is not None:
        if np.isfinite(gap):
            wb.log({"gap_mean_abs": gap})
            wb.summary["gap_mean_abs"] = gap
        wb.finish()
    print(f"[fit] wrote {args.out}/gates.json and predictions.npz")
    return 0


def _jsonable(o):
    """JSON encoder for the numpy the gates return."""
    if isinstance(o, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v
                for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        v = o.item()
        return None if (isinstance(v, float) and not np.isfinite(v)) else v
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def _gates(p_all, lab, W, val) -> dict:
    """Run the three gates on the held-out split and print them."""
    uids, ix = W["episode_uid"], W["replan_index"]
    y = lab["failed"]
    m = val
    # Phase: position within the episode, 0 at the first chunk and 1 at the last.
    last = {int(u): int(ix[uids == u].max()) for u in np.unique(uids)}
    phase = np.array([ix[i] / max(last[int(uids[i])], 1) for i in range(uids.size)])

    cal = value_gates.calibration(p_all[m], y[m])
    byph = value_gates.auc_by_phase(p_all[m], y[m], phase[m], n_bins=5)
    # Against the FALL, in CHUNKS: first_fall_step is a control step, and a chunk is K of them.
    k = 12
    ref = np.where(lab["first_fall_step"] >= 0, lab["first_fall_step"] / k, np.nan)
    lead_end = value_gates.lead_time(p_all[m], uids[m], ix[m], y[m])
    lead_fall = value_gates.lead_time(p_all[m], uids[m], ix[m], y[m], reference=ref[m])

    print(f"    aggregate AUC {value_gates.auc(p_all[m], y[m]):.3f}   "
          f"ECE {cal['ece']:.3f}  MCE {cal['mce']:.3f}")
    print("    AUC by phase: " + "  ".join(
        f"{lo:.1f}-{hi:.1f}:{'  nan' if np.isnan(a) else f'{a:5.3f}'}"
        for lo, hi, a in zip(byph["bin_lo"], byph["bin_hi"], byph["auc"])))
    for nm, L in (("to end", lead_end), ("to FALL", lead_fall)):
        good = np.isfinite(L["lead"])
        if good.any():
            print(f"    lead time {nm}: median {np.nanmedian(L['lead'][good]):.1f} chunks "
                  f"({np.nanmedian(L['lead'][good]) * 0.24:.2f} s) over "
                  f"{int(good.sum())}/{L['lead'].size} failures")
        else:
            print(f"    lead time {nm}: NEVER rose on any failure")
    return {"auc": value_gates.auc(p_all[m], y[m]), "calibration": cal,
            "auc_by_phase": byph,
            "lead_to_end": {k2: v for k2, v in lead_end.items()},
            "lead_to_fall": {k2: v for k2, v in lead_fall.items()}}


if __name__ == "__main__":
    raise SystemExit(main())
