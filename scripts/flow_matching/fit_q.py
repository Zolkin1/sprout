# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fit ``Q(s, a)`` over the generator's own chunks, and measure it against ``V(s)``.

``fit_value.py`` fits ``V(s)`` -- a state score, which answers identically for every candidate
drawn at one state and therefore cannot rank them.  Best-of-K needs a ranking.  This fits the
same architecture with the emitted chunk added as a token stream
(:class:`~robot_rl.flow_matching.qvalue.FlowQModel`) and reports the one number that decides
whether the action carries information: **within-state ranking accuracy**.

**No new data was collected for this.**  ``record_rollouts.py`` has stored ``fm_plan`` -- the
emitted reference window, i.e. the ACTION -- all along, so ``logs/rollouts/v2/`` is already an
``(s, a, outcome)`` table: 342,711 windows from the deployed pair with
``fm_zero_noise=False``, i.e. genuinely stochastic draws.

**Why on-policy data suffices**, without the same-state branching the original plan called
for: a Q-head does not need many actions at literally one state, only that the action vary
independently of the state often enough to be identifiable.  Measured on this rollout, 65,076
``(clip, replan)`` groups carry two or more independently drawn chunks and 42,261 carry four
or more -- the branch structure, arrived at by repetition across 128 envs rather than by
simulator restore.  Those are NEAR-replicates: two episodes on one clip have diverged by
re-plan 20 even though they share terrain, command and start, so the early re-plans are the
tight ones and the metric below is reported against re-plan depth.

**The label is SUCCESS**, notwithstanding the column being called ``failed``.
``rollout_outcomes.terminal_reward`` scores an episode ``0.0`` only if it ARRIVED and did not
fall first, ``-1.0`` otherwise -- so the complement is "did not do the task", not "fell over".
That is deliberate: if merely staying upright scored well, standing still would score like
climbing a staircase with none of the risk.

**Two exclusions, for different reasons.**  ``flat`` is dropped because its goal labelling is
not trusted.  Censored episodes are dropped because they have no outcome at all -- scoring one
would teach the head that whatever was happening at the recorder's ceiling predicts failure,
which is a property of the harness.  ``jump_off_front`` is KEPT: it is the forward dismount and
is observable to a forward-facing camera.  The excluded family elsewhere in this repo is
``jump_off``, which jumps BACKWARDS -- do not conflate them.

**The headline metric is rank-based, so calibration cannot confound it.**  Within a
``(clip, replan)`` group holding both a successful and a failed draw, does the model score the
successful one better?  ``V`` can only do this through the small state differences between
those episodes; ``Q`` sees the chunks.  The gap between them is what the action is worth.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rollout_outcomes as ro  # noqa: E402
from robot_rl.flow_matching import online, qvalue, transforms, value, value_gates  # noqa: E402

#: Declared metre scale for the goal offsets, mirroring ``fit_value.GOAL_SCALE_M`` so the two
#: models read the same numbers.
GOAL_SCALE_M = 5.0

#: Family codes dropped from training. ``flat`` only -- see the module docstring for why
#: ``jump_off_front`` is NOT here.
DROP_FAMILIES = (0,)


def _load(stem: str, max_windows: int) -> tuple[dict, dict, dict]:
    """Read the windows and episodes, join on ``episode_uid``, and drop what cannot be used.

    Args:
        stem: Rollout stem, without the ``.h5``.
        max_windows: Cap for smoke tests; 0 for all.

    Returns:
        ``(windows, episode_meta, labels)``.

    Raises:
        SystemExit: If the join is inexact, or if the file's real row count cannot be read --
            the recording pre-allocates and only ``num_windows_written`` rows are real, and
            the rest decode as ``clip 0, replan 0, family flat``.
    """
    cols = ["episode_uid", "replan_index", "clip_id", "family_code", "history", "history_null",
            "extero", "cmd_future", "fm_plan", "goal_downrange_m", "goal_lateral_m",
            "goal_height_err_m", "proprio", "base_twist_yaw", "height_scan"]
    W, wa = ro.read_rollout_windows(stem + ".h5", columns=cols)
    n_real = int(wa.get("num_windows_written", 0))
    if not n_real:
        raise SystemExit(
            f"{stem}.h5 has no num_windows_written attr. The recorder PRE-ALLOCATES "
            "(450,000 rows on the v2 file for 342,711 real ones) and the remainder are zeros "
            "that decode as a valid-looking flat window at clip 0, replan 0. Without that "
            "attribute there is no way to tell them apart, and training on them is training "
            "on 24% zeros.")
    if W["episode_uid"].size > n_real:
        W = {k: v[:n_real] for k, v in W.items()}
        print(f"[fit_q] sliced {W['episode_uid'].size:,} real rows (file allocates more)")
    if max_windows and W["episode_uid"].size > max_windows:
        W = {k: v[:max_windows] for k, v in W.items()}

    E, ea = ro.read_rollout_episodes(stem + ".episodes.h5")
    live = ~E["censored"].astype(bool)
    reward = ro.terminal_reward(E)
    out = {int(u): (float(r), bool(l)) for u, r, l in zip(E["episode_uid"], reward, live)}
    missing = sorted({int(u) for u in W["episode_uid"]} - set(out))
    if missing:
        raise SystemExit(f"{len(missing)} window episode(s) have no outcome row "
                         f"(e.g. {missing[:5]}); the join must be exact.")
    lab = {"failed": np.array([out[int(u)][0] == -1.0 for u in W["episode_uid"]]),
           "live": np.array([out[int(u)][1] for u in W["episode_uid"]])}
    ep_terrain = {int(u): int(t) for u, t in zip(E["episode_uid"], E["terrain_id"])}

    keep = lab["live"] & ~np.isin(W["family_code"], DROP_FAMILIES)
    print(f"[fit_q] keeping {int(keep.sum()):,} of {keep.size:,} windows "
          f"({int((~lab['live']).sum()):,} censored, "
          f"{int(np.isin(W['family_code'], DROP_FAMILIES).sum()):,} dropped families)")
    W = {k: v[keep] for k, v in W.items()}
    lab = {k: v[keep] for k, v in lab.items()}
    return W, {"attrs": ea, "ep_terrain": ep_terrain}, lab


def _batch(ctx, W, idx, device, scan_hw, action_steps):
    """One batch of :class:`~robot_rl.flow_matching.qvalue.QInputs`.

    Normalizer order mirrors ``fit_value._batch`` exactly so ``Q`` and ``V`` read identical
    state.  The ACTION is normalized with ``state_norm`` -- the emitted plan nodes live in the
    same space as the history nodes, which is what that normalizer was fitted on.

    Args:
        ctx: The loaded FM context.
        W: The window columns.
        idx: Row indices.
        device: Torch device.
        scan_hw: ``(H, W)`` height-scan grid.
        action_steps: Node indices to keep from the ``(F, D)`` plan.

    Returns:
        The batch.
    """
    to = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)  # noqa: E731
    hist = ctx.state_norm.normalize(to(W["history"][idx]))
    ext = ctx.extero_norm.normalize(to(W["extero"][idx]).reshape(len(idx), -1))
    cmd = None
    if ctx.cmd_dim > 0:
        steps = torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=device)
        cmd = ctx.cmd_norm.normalize(to(W["cmd_future"][idx])[:, steps, :])
    scan = transforms.sanitize_extero(
        to(W["height_scan"][idx]), "height_scan").reshape(len(idx), *scan_hw)
    goal = to(np.stack([W["goal_downrange_m"][idx], W["goal_lateral_m"][idx],
                        W["goal_height_err_m"][idx]], axis=-1)) / GOAL_SCALE_M
    goal = torch.nan_to_num(goal, nan=0.0, posinf=0.0, neginf=0.0)
    st = value.ValueInputs(
        history=hist, extero=ext, cmd_tokens=cmd,
        history_null=to(W["history_null"][idx], torch.bool),
        scan=scan, proprio=to(W["proprio"][idx]),
        context=torch.cat([goal, to(W["base_twist_yaw"][idx])], dim=-1))
    a = ctx.state_norm.normalize(to(W["fm_plan"][idx])[:, action_steps, :])
    return qvalue.QInputs(state=st, action=a)


def within_state_ranking(score: np.ndarray, failed: np.ndarray, clip: np.ndarray,
                         ridx: np.ndarray) -> dict:
    """Does the model rank the surviving draw above the failing one, at the same state?

    The metric best-of-K actually depends on, and the only one immune to miscalibration: it
    compares scores WITHIN a ``(clip, replan)`` group, so any monotone distortion of the score
    cancels.  Pairs are formed only where a group holds both outcomes -- a group that all
    survived or all failed carries no ranking information and contributes nothing.

    Args:
        score: Predicted ``P(fail)``; lower should mean safer.
        failed: True where the episode did not do the task.
        clip: Per-window clip id.
        ridx: Per-window re-plan index.

    Returns:
        ``{"acc", "n_pairs", "n_groups", "by_depth"}``. ``acc`` 0.5 is chance.
    """
    key = clip.astype(np.int64) * 100000 + ridx
    order = np.argsort(key, kind="mergesort")
    k_s, starts = np.unique(key[order], return_index=True)
    ends = list(starts[1:]) + [len(order)]
    wins = tot = 0
    by_depth: dict[int, list] = {}
    for s, e in zip(starts, ends):
        rows = order[s:e]
        f = failed[rows]
        if f.all() or not f.any():
            continue
        good, bad = score[rows[~f]], score[rows[f]]
        # All good-vs-bad pairs in the group. Ties count as half, the standard AUC convention;
        # scoring them as wins would flatter a model that collapsed to a constant.
        w = (good[:, None] < bad[None, :]).sum() + 0.5 * (good[:, None] == bad[None, :]).sum()
        n = good.size * bad.size
        wins += float(w)
        tot += int(n)
        d = int(ridx[rows[0]])
        by_depth.setdefault(min(d // 5 * 5, 30), [0.0, 0]);  by_depth[min(d // 5 * 5, 30)][0] += float(w)
        by_depth[min(d // 5 * 5, 30)][1] += int(n)
    return {"acc": (wins / tot if tot else float("nan")), "n_pairs": tot,
            "n_groups": len(k_s),
            "by_depth": {k: (v[0] / v[1] if v[1] else float("nan"), v[1])
                         for k, v in sorted(by_depth.items())}}


def _predict(model, ctx, W, idx, device, scan_hw, action_steps, batch) -> np.ndarray:
    """``P(fail | s, a)`` over ``idx``, eval mode, no grad.

    Args:
        model: The Q model.
        ctx: FM context.
        W: Window columns.
        idx: Rows to score.
        device: Torch device.
        scan_hw: Scan grid.
        action_steps: Strided plan node indices.
        batch: Chunk size.

    Returns:
        ``(len(idx),)`` float64.
    """
    was = model.training
    model.eval()
    out = np.zeros(idx.size, dtype=np.float64)
    with torch.no_grad():
        for i in range(0, idx.size, batch):
            j = idx[i:i + batch]
            out[i:i + j.size] = model.p_fail(
                _batch(ctx, W, j, device, scan_hw, action_steps)).cpu().numpy()
    model.train(was)
    return out


def main() -> int:
    """Fit Q and report it against V on the ranking metric."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout", required=True, help="Rollout stem, without .h5")
    p.add_argument("--fm_checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val_frac", type=float, default=0.08)
    p.add_argument("--tile_holdout", type=float, default=0.2)
    p.add_argument("--max_windows", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--wandb_project", default="robot_rl_qvalue")
    p.add_argument("--run_name", default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    W, meta, lab = _load(args.rollout, args.max_windows)
    scan_hw = tuple(meta["attrs"]["height_scan_hw"])
    uids = W["episode_uid"]
    tiles = np.array([meta["ep_terrain"][int(u)] for u in uids])

    # Held-out TILES first, then held-out EPISODES from what remains. Splitting on windows
    # would put two windows of one episode either side of the line, and they share an outcome.
    held_t = set(rng.choice(np.unique(tiles), max(1, int(len(np.unique(tiles))
                                                        * args.tile_holdout)), replace=False).tolist())
    val_tile = np.isin(tiles, list(held_t))
    pool = np.unique(uids[~val_tile])
    held_e = set(rng.choice(pool, max(1, int(pool.size * args.val_frac)), replace=False).tolist())
    val_ep = np.isin(uids, list(held_e)) & ~val_tile
    train = ~val_tile & ~val_ep
    print(f"[fit_q] train {int(train.sum()):,} | val_ep {int(val_ep.sum()):,} | "
          f"val_tile {int(val_tile.sum()):,} over {len(np.unique(tiles))} tiles")
    print(f"[fit_q] label: failed={lab['failed'].mean():.3f} of windows")

    ctx = online.load_fm_context(args.fm_checkpoint, device=args.device)
    # The action is strided by the generator's own cmd_steps -- the offsets its conditioning
    # already treats as informative -- rather than by an independent choice.
    action_steps = torch.as_tensor(ctx.cmd_steps, dtype=torch.long, device=args.device)
    n_a = len(ctx.cmd_steps)
    model = qvalue.FlowQModel(
        copy.deepcopy(ctx.model), action_dim=int(W["fm_plan"].shape[-1]), action_steps=n_a,
        scan_hw=scan_hw, proprio_dim=int(W["proprio"].shape[-1]),
        proprio_steps=int(W["proprio"].shape[1]), context_dim=9).to(args.device)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    y = torch.as_tensor(lab["failed"].astype(np.float32), device=args.device)

    wb = None
    if not args.no_wandb:
        import wandb  # noqa: PLC0415

        wb = wandb.init(project=args.wandb_project,
                        name=args.run_name or os.path.basename(args.out),
                        config={**vars(args), "n_action_tokens": n_a,
                                "n_train": int(train.sum()), "scan_hw": list(scan_hw),
                                "trainable": sum(q.numel() for q in model.trainable_parameters())})
        print(f"[fit_q] wandb: {wb.url}")

    tr = np.flatnonzero(train)
    best = (float("inf"), -1)
    for ep in range(args.epochs):
        model.train()
        rng.shuffle(tr)
        run, nb = 0.0, 0
        for i in range(0, tr.size, args.batch_size):
            j = tr[i:i + args.batch_size]
            loss = lossf(model(_batch(ctx, W, j, args.device, scan_hw, action_steps)), y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            opt.step()
            run += float(loss.item())
            nb += 1
            if wb is not None and nb % 200 == 0:
                wb.log({"train/loss": run / nb, "epoch": ep})

        row = {"epoch": ep + 1, "train_loss": run / max(nb, 1)}
        for scope, m in (("val_ep", val_ep), ("val_tile", val_tile)):
            idx = np.flatnonzero(m)
            if not idx.size:
                continue
            pf = _predict(model, ctx, W, idx, args.device, scan_hw, action_steps, 256)
            f = lab["failed"][idx]
            rk = within_state_ranking(pf, f, W["clip_id"][idx], W["replan_index"][idx])
            row[f"{scope}/auc"] = value_gates.auc(pf, f)
            row[f"{scope}/rank_acc"] = rk["acc"]
            row[f"{scope}/rank_pairs"] = rk["n_pairs"]
            if scope == "val_tile":
                row["val_tile/rank_by_depth"] = rk["by_depth"]
        print(f"[fit_q] ep {row['epoch']:>2} loss {row['train_loss']:.4f}  "
              f"val_ep AUC {row.get('val_ep/auc', float('nan')):.3f} "
              f"rank {row.get('val_ep/rank_acc', float('nan')):.3f}  |  "
              f"val_tile AUC {row.get('val_tile/auc', float('nan')):.3f} "
              f"rank {row.get('val_tile/rank_acc', float('nan')):.3f}", flush=True)
        if wb is not None:
            wb.log({k: v for k, v in row.items() if not isinstance(v, dict)})
        if row["train_loss"] < best[0]:
            best = (row["train_loss"], ep)
            torch.save({"model": model.state_dict(), "fm_checkpoint": args.fm_checkpoint,
                        "scan_hw": list(scan_hw), "proprio_hw": list(W["proprio"].shape[1:]),
                        "context_dim": 9, "action_steps": n_a,
                        "action_dim": int(W["fm_plan"].shape[-1]), "epoch": ep},
                       os.path.join(args.out, "Q.pt"))
        with open(os.path.join(args.out, "history.json"), "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    print(f"\n[fit_q] best train loss {best[0]:.4f} at epoch {best[1] + 1} -> {args.out}/Q.pt")
    print("[fit_q] READ the RANK accuracy, not the AUC. AUC is a state-level number that a "
          "state-only V\n        already scores well on; rank accuracy asks whether the "
          "ACTION was informative, which\n        is the only thing best-of-K depends on. 0.5 "
          "is chance.")
    if wb is not None:
        wb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
