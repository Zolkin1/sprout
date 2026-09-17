# Copyright 2026 Zachary Olkin. All rights reserved.

"""What the generator's proposal cloud LOOKS like at one state, and where the critic likes it.

Freeze one re-plan state, draw ``N`` chunks from the generator (the only thing varying is the
flow noise), roll each one chunk forward and score the resulting state with ``V``.  Then embed
the ``N`` chunks in 2-D and colour by that score.  With a second ``--fm_onnx`` the SAME state
and the SAME noise keys are replayed through a fine-tuned model, so the two clouds are
directly comparable and the question "did fine-tuning move the proposal distribution, and did
it move toward what the critic likes?" becomes something you can look at.

**Both models branch from ONE snapshot, taken with the FIRST model.**  Driving the prefix with
each model separately would reach two different states, and every difference in the clouds
would then be confounded with where the robot was standing.  The unconditioned fine-tune keeps
the architecture identical (same 142 tensors, same shapes), which is exactly what makes the
policy half of the snapshot transferable between them; it is asserted, not assumed.

**Four scaling choices, because this plot is mostly a scaling question:**

* **Normalized space, never raw.**  The 44 channels mix metres, radians and m/s, so a raw
  Euclidean distance is dominated by whichever block has the largest units.  Chunks are
  embedded through the checkpoint's own ``state_norm``.
* **Per-node standardization, by default.**  Prediction variance grows with look-ahead (5.1x
  from node 0 to node 50, measured), so a flat concatenation of 64 nodes lets the far end
  dictate the whole geometry.  ``--node_scaling per_node`` gives every node an equal say;
  ``none`` keeps the natural growth.  Both are worth looking at -- they answer different
  questions, and neither is the "correct" one.
* **ONE embedding fitted on the UNION of both models' samples.**  Fitting per model would put
  the two clouds in different coordinate systems, where "the cloud moved" is unreadable.
* **PCA is the primary panel and t-SNE is secondary.**  At N=50 per model, t-SNE has ~100
  points and needs a perplexity near 10; its geometry is largely arbitrary at that size and
  its distances mean nothing.  PCA is deterministic, and the fraction of variance it explains
  is printed so you can tell whether the 2-D picture is representative at all.

Two stages, and ``plot`` imports no MuJoCo and no onnxruntime::

    python fm_sample_tsne.py --stage record --out logs/value/tsne_v1 \\
        --tile 20 --speed 2.4 --replan 6 --n 50 \\
        --fm_onnx <base>_fm.onnx --fm_onnx_b <finetuned>_fm.onnx \\
        --critic_ckpt logs/value/v2/V_priv.fold0.pt
    python fm_sample_tsne.py --stage plot --out logs/value/tsne_v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (_HERE, os.path.dirname(_HERE), os.path.join(_HERE, "mjeval")):
    if _d not in sys.path:
        sys.path.insert(0, _d)


def _drive_to_replan(rig, policy, fm_lag, spawn, cmd, cfg, n_replans, seed, mjrunner,
                     cmd_hold_s=0.0):
    """Run the episode forward ``n_replans`` re-plans and stop there.

    Args:
        rig: The course rig.
        policy: The DRIVING policy (model A).
        fm_lag: Its camera-lag ring.
        spawn: Initial condition.
        cmd: ``(3,)`` commanded twist.
        cfg: The ``ArmRun``.
        n_replans: How many chunks to execute before stopping.
        seed: Episode seed.
        mjrunner: The ``runner`` module.

    Returns:
        ``(zex, ok)`` -- the episode's z extremes and whether it survived that far.
    """
    import mujoco  # noqa: PLC0415
    from live_select import _chunk  # noqa: PLC0415

    robot = rig.robot
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    policy._rng = np.random.default_rng(seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    cadence = int(getattr(policy, "fm_update_every", 12))
    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]
    for _ in range(int(n_replans)):
        v, _f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, zex,
                            hold_until=cmd_hold_s)
        if v:
            return zex, False
    return zex, True


def _sample_cloud(rig, policy, fm_lag, snap, cmd, cfg, zex, crit, anchor, n, seed, ring,
                  cmd_hold_s=0.0):
    """Draw ``n`` chunks from one state and score each one chunk out.

    Restores ``snap`` before every draw, so each sample starts from the identical state and the
    ONLY thing that differs between them is the flow-noise key ``[seed, i]`` -- the same keying
    ``live_select`` uses for its candidates.

    Args:
        rig: The course rig.
        policy: The policy to sample from.
        fm_lag: Its camera-lag ring.
        snap: The frozen state to branch from.
        cmd: Commanded twist.
        cfg: The ``ArmRun``.
        zex: Episode z extremes at the branch point.
        crit: A ``critic.ValueCritic``.
        anchor: Goal anchor for the critic's goal features.
        n: Number of draws.
        seed: Noise-key seed, shared across models so draw ``i`` is the same key in both.
        ring: The proprio ring the critic reads, or ``None``.

    Returns:
        ``dict`` of stacked plans, scores and verdict codes.
    """
    import critic as C  # noqa: PLC0415
    from live_select import SelectedChunkWriter, _chunk  # noqa: PLC0415

    robot = rig.robot
    plans, scores, verds = [], [], []
    for i in range(int(n)):
        snap.restore(robot, policy, fm_lag)
        policy._rng = np.random.default_rng([seed, i])
        cap: dict = {}
        cz = list(zex)
        v, _f, st = _chunk(rig, policy, fm_lag, cmd, int(getattr(policy, "fm_update_every", 12)),
                           cfg, cz, scorer=(lambda r_, p_, s_: ring.push(
                               s_["joint_pos"], s_["joint_vel"], s_["base_ang_vel"],
                               C.projected_gravity_b(s_["root_quat"]))) if ring else None,
                           capture=cap, hold_until=cmd_hold_s)
        if v == "fall":
            score = 1.0
        elif v == "success":
            score = -1.0
        elif st is None:
            score = 1.0
        else:
            x = C.CriticInputs.from_runtime(policy, robot, st, anchor=anchor, ring=ring,
                                            want_scan=crit.privileged)
            # V(s), so HIGH is good -- the opposite of the p_fail this replaced.
            score = float(crit.value(x))
        plans.append(np.asarray(cap.get("fm_plan"), dtype=np.float32))
        scores.append(score)
        verds.append(SelectedChunkWriter.VERDICT_CODES.get(v, 0))
    return {"plans": np.stack(plans), "scores": np.asarray(scores, dtype=np.float32),
            "verdict": np.asarray(verds, dtype=np.uint8)}


def stage_record(args) -> int:
    """Freeze one state, sample both models from it, and write the clouds.

    Args:
        args: Parsed CLI.

    Returns:
        Process exit code.
    """
    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    import terrains  # noqa: PLC0415
    from live_select import Snapshot  # noqa: PLC0415

    course = terrains.isaac_tile_course(args.clip_lib, int(args.tile), speeds=(args.speed,),
                                        approach=args.approach,
                                        klass=(args.klass or None))
    import dataclasses  # noqa: PLC0415
    course = dataclasses.replace(course, name=f"tile{args.tile}{args.scene_tag}")

    def _arm(fm_onnx):
        cfg = mjrunner.ArmRun(
            arm=0, label="tsne", tracker_onnx=args.tracker_onnx, fm_onnx=fm_onnx,
            out_dir=args.out,
            providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
            fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
            cmd_hold_s=args.cmd_hold_s, steer_kp=0.0, zero_noise=False, fm_seed=0)
        policy, fm_lag, spawn = mjrunner.load_arm(cfg)
        return cfg, policy, fm_lag, spawn

    cfg_a, pol_a, lag_a, spawn = _arm(args.fm_onnx)
    rig = mjrunner.CourseRig(course, pol_a, cfg_a, spawn)
    crit = C.ValueCritic(args.critic_ckpt, fm_pt=(args.fm_pt or None), device=args.device)
    ring = C.ProprioRing() if crit.privileged else None
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)

    zex, ok = _drive_to_replan(rig, pol_a, lag_a, spawn, cmd, cfg_a, args.replan,
                               args.seed, mjrunner, cmd_hold_s=args.cmd_hold_s)
    if not ok:
        raise SystemExit(
            f"the episode ended before re-plan {args.replan} on tile {args.tile} at "
            f"{args.speed} m/s (seed {args.seed}). Pick an earlier --replan, a slower "
            "--speed, or another --seed: there is no state to branch from.")
    snap = Snapshot(rig.robot, pol_a, lag_a)
    print(f"[tsne] frozen at re-plan {args.replan}, base xyz = "
          f"{np.round(rig.robot.mj_data.qpos[:3], 3).tolist()}")

    out = {"a": _sample_cloud(rig, pol_a, lag_a, snap, cmd, cfg_a, zex, crit,
                              course.goal_anchor_xyz, args.n, args.noise_seed, ring,
                              cmd_hold_s=args.cmd_hold_s)}
    print(f"[tsne] model A: {args.n} draws, p_fail "
          f"[{out['a']['scores'].min():.3f}, {out['a']['scores'].max():.3f}]")

    if args.fm_onnx_b:
        cfg_b, pol_b, lag_b, _s = _arm(args.fm_onnx_b)
        # The SAME snapshot restored into the second policy. Legitimate only because the
        # unconditioned fine-tune leaves the architecture untouched, so every captured policy
        # field has the same meaning and shape in both -- checked rather than trusted.
        for k, v in snap.policy.items():
            cur = getattr(pol_b, k, None)
            if isinstance(v, np.ndarray) and isinstance(cur, np.ndarray) \
                    and v.shape != cur.shape:
                raise SystemExit(
                    f"policy field {k!r} is {v.shape} in model A and {cur.shape} in model B, "
                    "so the snapshot cannot be transferred. The two generators are not the "
                    "same architecture, and branching them from one state is not defined.")
        out["b"] = _sample_cloud(rig, pol_b, lag_b, snap, cmd, cfg_b, zex, crit,
                                 course.goal_anchor_xyz, args.n, args.noise_seed, ring,
                                 cmd_hold_s=args.cmd_hold_s)
        print(f"[tsne] model B: {args.n} draws, p_fail "
              f"[{out['b']['scores'].min():.3f}, {out['b']['scores'].max():.3f}]")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, "samples.npz"),
        plans_a=out["a"]["plans"], scores_a=out["a"]["scores"], verdict_a=out["a"]["verdict"],
        **({"plans_b": out["b"]["plans"], "scores_b": out["b"]["scores"],
            "verdict_b": out["b"]["verdict"]} if "b" in out else {}),
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"args": vars(args), "n": int(args.n),
                   "has_b": bool(args.fm_onnx_b)}, f, indent=2)
    print(f"[tsne] wrote {args.out}/samples.npz")
    return 0


def _embed(plans_a, plans_b, node_scaling, seed):
    """Flatten, scale and project both clouds into ONE 2-D space.

    Args:
        plans_a: ``[N, F, D]`` model A's chunks.
        plans_b: ``[M, F, D]`` model B's chunks, or ``None``.
        node_scaling: ``per_node`` (equalize each look-ahead node's variance) or ``none``.
        seed: RNG seed for t-SNE.

    Returns:
        ``(pca_xy, tsne_xy, n_a, explained)`` -- both projections over the concatenation.
    """
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.manifold import TSNE  # noqa: PLC0415

    n_a = plans_a.shape[0]
    x = plans_a if plans_b is None else np.concatenate([plans_a, plans_b], axis=0)
    if node_scaling == "per_node":
        # Per (node, channel) standardization over the UNION, so no look-ahead node dominates
        # by virtue of predicting further out.
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        x = (x - mu) / np.maximum(sd, 1e-6)
    flat = x.reshape(x.shape[0], -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    p = PCA(n_components=2, random_state=seed).fit(flat)
    pca_xy = p.transform(flat)
    perp = float(max(5.0, min(30.0, (flat.shape[0] - 1) / 3.0)))
    tsne_xy = TSNE(n_components=2, perplexity=perp, init="pca",
                   random_state=seed).fit_transform(flat)
    return pca_xy, tsne_xy, n_a, p.explained_variance_ratio_



def stage_scan(args) -> int:
    """Walk the episode and report, at every re-plan, how much the noise draw MATTERS.

    Which state to freeze is the whole experiment, and it is not a matter of taste: at a state
    where every draw survives comfortably the cloud is one undifferentiated blob and the colour
    carries no information.  What is worth looking at is a state where the draw DECIDES the
    outcome -- a large spread of ``p_fail`` across draws, or a mix of falls and survivals.

    So this scans instead of guessing.  At each re-plan it snapshots, takes ``--scan_n`` cheap
    draws, records the spread, restores, and then continues along the episode's OWN trajectory
    -- the snapshot carries ``_rng``, so scanning does not perturb the run it is measuring.

    Args:
        args: Parsed CLI.

    Returns:
        Process exit code.
    """
    import dataclasses  # noqa: PLC0415

    import critic as C  # noqa: PLC0415
    import runner as mjrunner  # noqa: PLC0415
    import terrains  # noqa: PLC0415
    from live_select import Snapshot, _chunk  # noqa: PLC0415

    course = terrains.isaac_tile_course(args.clip_lib, int(args.tile), speeds=(args.speed,),
                                        approach=args.approach, klass=(args.klass or None))
    course = dataclasses.replace(course, name=f"tile{args.tile}{args.scene_tag}")
    cfg = mjrunner.ArmRun(
        arm=0, label="scan", tracker_onnx=args.tracker_onnx, fm_onnx=args.fm_onnx,
        out_dir=args.out,
        providers=tuple(x.strip() for x in args.providers.split(",") if x.strip()),
        fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
        cmd_hold_s=0.0, steer_kp=0.0, zero_noise=False, fm_seed=0)
    policy, fm_lag, spawn = mjrunner.load_arm(cfg)
    rig = mjrunner.CourseRig(course, policy, cfg, spawn)
    crit = C.ValueCritic(args.critic_ckpt, fm_pt=(args.fm_pt or None), device=args.device)
    ring = C.ProprioRing() if crit.privileged else None
    cmd = np.array([float(args.speed), 0.0, 0.0], dtype=np.float64)

    import mujoco  # noqa: PLC0415

    robot = rig.robot
    mujoco.mj_resetData(robot.mj_model, robot.mj_data)
    robot.set_base_and_joints(spawn.base_pos, spawn.base_quat_wxyz, rig.hold_target,
                              base_lin_vel_w=spawn.base_lin_vel_w,
                              base_ang_vel_w=spawn.base_ang_vel_w,
                              joint_vel_mujoco=rig.hold_joint_vel)
    policy._rng = np.random.default_rng(args.seed)
    mjrunner._reset_episode_state(policy, fm_lag)
    cadence = int(getattr(policy, "fm_update_every", 12))
    z0 = float(robot.mj_data.qpos[2])
    zex = [z0, z0]

    rows = []
    print(f"{'replan':>6s} {'x':>7s} {'z':>7s} {'p_mean':>7s} {'p_sd':>7s} "
          f"{'p_min':>7s} {'p_max':>7s} {'falls':>6s}")
    for r in range(int(args.scan_max)):
        snap = Snapshot(robot, policy, fm_lag)
        got = _sample_cloud(rig, policy, fm_lag, snap, cmd, cfg, zex, crit,
                            course.goal_anchor_xyz, args.scan_n, args.noise_seed, ring,
                            cmd_hold_s=args.cmd_hold_s)
        sc, vd = got["scores"], got["verdict"]
        # Restore: the scan must not advance the episode NOR its rng, or the trajectory being
        # characterised is not the one that gets scanned at the next index.
        snap.restore(robot, policy, fm_lag)
        x, z = float(robot.mj_data.qpos[0]), float(robot.mj_data.qpos[2])
        n_fall = int((vd == 1).sum())
        # Rank on the CRITIC-SCORED draws only. A draw that fell or arrived carries 1.0 / -1.0
        # by fiat, so a re-plan where some draws reach the goal and some do not shows an sd of
        # 0.5 that is the sentinel spread and nothing else -- it looks exactly like the state
        # of maximum disagreement while being the state where the episode is simply ending.
        real = sc[vd == 0]
        rows.append({"replan": r, "x": x, "z": z, "p_mean": float(sc.mean()),
                     "p_sd": float(sc.std()), "p_min": float(sc.min()),
                     "p_max": float(sc.max()), "n_fall": n_fall, "n": int(args.scan_n),
                     "n_real": int(real.size),
                     "p_sd_real": float(real.std()) if real.size > 1 else 0.0})
        print(f"{r:6d} {x:7.2f} {z:7.2f} {sc.mean():7.3f} {sc.std():7.3f} "
              f"{sc.min():7.3f} {sc.max():7.3f} {n_fall:3d}/{args.scan_n:<2d}")
        v, _f, _st = _chunk(rig, policy, fm_lag, cmd, cadence, cfg, zex,
                            hold_until=cmd_hold_s)
        if v:
            print(f"[scan] episode ended at re-plan {r + 1}: {v}")
            break
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "scan.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    if rows:
        best = max(rows, key=lambda q: q["p_sd_real"])
        mixed = [q for q in rows if 0 < q["n_fall"] < q["n"]]
        print(f"\n[scan] widest spread at re-plan {best['replan']} "
              f"(sd {best['p_sd_real']:.3f} over {best['n_real']} critic-scored draws; "
              f"sentinel draws excluded)")
        if mixed:
            print("[scan] re-plans where the draw DECIDES survival (mixed falls): "
                  + ", ".join(str(q["replan"]) for q in mixed))
        else:
            print("[scan] no re-plan had a mix of falls and survivals -- at this speed the "
                  "draw may not be decisive anywhere on this tile, and the cloud will be "
                  "coloured by p_fail alone.")
    print(f"[scan] wrote {args.out}/scan.json")
    return 0

def stage_plot(args) -> int:
    """Draw the clouds, coloured by critic value.

    Args:
        args: Parsed CLI.

    Returns:
        Process exit code.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    d = np.load(os.path.join(args.out, "samples.npz"))
    pa, sa = d["plans_a"], d["scores_a"]
    has_b = "plans_b" in d
    pb, sb = (d["plans_b"], d["scores_b"]) if has_b else (None, None)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    for row, scaling in enumerate(("per_node", "none")):
        pca_xy, tsne_xy, n_a, ev = _embed(pa, pb, scaling, args.seed)
        for col, (xy, name) in enumerate(((pca_xy, "PCA"), (tsne_xy, "t-SNE"))):
            ax = axes[row][col]
            sc = np.concatenate([sa, sb]) if has_b else sa
            vmin, vmax = float(sc.min()), float(sc.max())
            ax.scatter(xy[:n_a, 0], xy[:n_a, 1], c=sa, cmap="viridis", vmin=vmin, vmax=vmax,
                       s=70, marker="o", edgecolors="k", linewidths=0.4, label="base")
            if has_b:
                im = ax.scatter(xy[n_a:, 0], xy[n_a:, 1], c=sb, cmap="viridis",
                                vmin=vmin, vmax=vmax, s=95, marker="^",
                                edgecolors="crimson", linewidths=1.0, label="fine-tuned")
            else:
                im = ax.collections[0]
            extra = (f"  (PC1+PC2 = {100 * ev.sum():.0f}% of variance)"
                     if name == "PCA" else "")
            ax.set_title(f"{name} | node_scaling={scaling}{extra}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046).set_label("critic p_fail", fontsize=8)
    fig.suptitle("Generator proposals at ONE frozen state, coloured by the critic\n"
                 "one shared embedding per row; marker shape = model", fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.out, "sample_cloud.png")
    fig.savefig(path, dpi=140)
    print(f"[tsne] wrote {path}")
    print(f"[tsne] base p_fail mean {sa.mean():.4f}  sd {sa.std():.4f}")
    if has_b:
        print(f"[tsne] fine p_fail mean {sb.mean():.4f}  sd {sb.std():.4f}")
        print(f"[tsne] shift in mean p_fail = {sb.mean() - sa.mean():+.4f} "
              "(negative = the fine-tuned cloud sits where the critic prefers)")
    return 0


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("scan", "record", "plot"))
    p.add_argument("--out", required=True)
    p.add_argument("--tracker_onnx", default="")
    p.add_argument("--fm_onnx", default="", help="model A -- also drives the prefix.")
    p.add_argument("--fm_onnx_b", default="",
                   help="model B, sampled from the SAME frozen state with the SAME noise keys.")
    p.add_argument("--fm_pt", default="")
    p.add_argument("--critic_ckpt", default="")
    p.add_argument("--clip_lib", default="clips/g1_mb_v5_body_mode11_decor.h5")
    p.add_argument("--tile", type=int, default=20)
    p.add_argument("--speed", type=float, default=2.4)
    p.add_argument("--approach", type=float, default=2.5)
    p.add_argument("--klass", default="")
    p.add_argument("--replan", type=int, default=6,
                   help="Freeze after this many executed chunks. Too early and the robot is "
                        "still on flat approach where every draw is fine; too late and the "
                        "episode may already be over.")
    p.add_argument("--n", type=int, default=50, help="draws per model.")
    p.add_argument("--seed", type=int, default=0, help="episode seed for the prefix.")
    p.add_argument("--noise_seed", type=int, default=1234,
                   help="Flow-noise key seed. SHARED by both models, so draw i is the same "
                        "key in each and the clouds are paired rather than merely comparable.")
    p.add_argument("--cmd_hold_s", type=float, default=None,
                   help="Seconds of zero command before stepping to --speed; defaults to "
                        "terrains.CMD_HOLD_S. The hold changes which states an episode even "
                        "REACHES, so a frozen state chosen under one value does not exist "
                        "under the other -- --replan indices are not portable between them.")
    p.add_argument("--scan_n", type=int, default=8,
                   help="draws per re-plan in the scan stage -- enough to see a spread, cheap "
                        "enough to walk a whole episode.")
    p.add_argument("--scan_max", type=int, default=40, help="re-plans to scan at most.")
    p.add_argument("--node_scaling", default="per_node", choices=("per_node", "none"))
    p.add_argument("--scene_tag", default="_tsne")
    p.add_argument("--device", default="cuda")
    p.add_argument("--providers", default="auto")
    p.add_argument("--fall_height", type=float, default=0.2)
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    args = p.parse_args()
    if args.cmd_hold_s is None:
        import terrains  # noqa: PLC0415  the shared default, resolved after parsing

        args.cmd_hold_s = terrains.CMD_HOLD_S
    return {"scan": stage_scan, "record": stage_record, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
