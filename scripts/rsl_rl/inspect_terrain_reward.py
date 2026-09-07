# Copyright 2026 Zachary Olkin. All rights reserved.

"""Verify the terrain-consistency reward BEFORE anything is trained on it.

The reward is a function of ``(plan, terrain)`` alone -- no rollout, no simulator -- so it can
be computed on plans that are already on disk.  That is what this script does, and it is
deliberately the gate in front of the critic and the fine-tune: a reward nobody has looked at
is a reward that gets optimised in the wrong direction for a week.

Four stages, sharing ``<out>/terrain_reward.json``::

    --stage frame      is the plan PLACED right? (the one check everything else rests on)
    --stage calibrate  what velocity threshold marks a stance node, and how well?
    --stage grade      the reward over recorded plans: distributions, outcome splits, sweeps
    --stage reward     write the per-window terrain reward as a sidecar for the critic
    --stage report     plots from what grade wrote
    --stage all        frame -> calibrate -> grade -> report

``frame`` comes first for a reason.  Every element of the placement -- the tile corner
recovered from the recorded goal, the anchor, the yaw-frame lift, the FK -- lands in one
number: how far the plan's first-node soles sit off the terrain under them.  If that has a
systematic offset then the reward is grading a displacement, and every number after it is
about the harness rather than about the generator.

``calibrate`` reports **three separate arms**, never pooled, because no single ground truth
for foot contact is both correct on terrain and independent of the SDF:

* **A -- flat clips, independent.**  The clip library's own ``contact`` flag, on clips where
  its flat-ground-at-z=0 assumption is actually true.  No SDF involved.  The trustworthy arm.
* **B -- terrain clips, SDF-recomputed flag.**  The library's flag is *replaced* (not
  compared against) by ``min sole-corner sd < contact_z_thresh`` on the tile's real boxes,
  because on terrain the recorded flag is simply wrong -- see
  ``test_clip_contact_flag_is_flat_ground.py``.  Correct on terrain, but not independent of
  the SDF, and labelled as such.
* **C -- measured contact.**  Isaac's own foot contact sensors, which are declared in the
  scene and read by nothing.  Correct on terrain AND independent, but it has to be recorded
  first, so it reports as unavailable until the recorder logs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from motion_gen import features as MGF                                    # noqa: E402
from motion_gen.clip_output import foot_box_corners                       # noqa: E402
from motion_gen.joint_names import G1_JOINT_NAMES                         # noqa: E402
from motion_gen.math_utils import quat_apply                              # noqa: E402
from motion_gen.plan_geometry import (                                    # noqa: E402
    CONTACT_DILATE_NODES,
    CONTACT_MAX_GAP_NODES,
    DEFAULT_COLLIDER_SPACING,
    CONTACT_SPEED_THRESH,
    FOOT_BODIES,
    PlanColliderModel,
    plan_collider_points,
    plan_contact,
)
from motion_gen.terrain_consistency import (                              # noqa: E402
    TerrainConsistencySpec,
    terrain_consistency,
)
import terrain_reward_io as trio                                          # noqa: E402

#: Velocity thresholds (m/s) the calibration sweeps.
THRESHOLD_SWEEP: tuple[float, ...] = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50)

#: Dilation radii (nodes) the grading sweeps, so the buffer's cost is a number.
DILATE_SWEEP: tuple[int, ...] = (0, 1, 2, 3)

#: The library's own contact test tolerance (m), mirrored from ``ClipOutputBuilder``.
CONTACT_Z_THRESH: float = 0.02

#: Control dt (s).  The plan's nodes are contiguous control steps.
CONTROL_DT: float = 0.02

#: Temperature of the penalty -> reward map (in penalty units).
#:
#: ``normalized_returns`` -- the critic's quality target -- assumes a per-chunk reward bounded
#: in ``(0, 1]``: that is what makes its ``(1 - gamma)`` factor give a declarable ``[0, 1]``
#: support instead of a fitted one.  The terrain terms are unbounded PENALTIES, so they are
#: mapped ``R = exp(-penalty / tau)``: 1 for a geometrically perfect plan, falling toward 0.
#: Same shape as the tracking reward this replaces, which is itself a product of exp kernels
#: bounded in ``(0, 1]`` -- so every downstream assumption (the two-hot bins, the tail
#: convention, the ``[0, 1]`` support) is untouched.
#:
#: 1.0 because the reference weights already put each term near 1.0 at its own p95, so a unit
#: temperature spends the reward's range on exactly the band the terms were scaled to.
REWARD_TAU: float = 1.0

#: Name stamped into the sidecar and checked by ``label_advantage``.
REWARD_SOURCE: str = "terrain"

#: Target SDF working-set size, in point-box cells, used to pick the window chunk.
#:
#: The SDF's cost and memory are ``windows * nodes * points * boxes``. With surface sampling
#: that is ~1900 points instead of 46, so a fixed window chunk that was fine before now asks
#: for tens of GB. The chunk is DERIVED from the plan's actual shape instead.
SDF_TARGET_CELLS: int = 20_000_000


# ---------------------------------------------------------------------------------------
# stage: frame
# ---------------------------------------------------------------------------------------
def _episode_index(episodes_path: str) -> dict[str, np.ndarray]:
    """Per-episode columns this script joins against, keyed by name.

    Args:
        episodes_path: The ``*.episodes.h5`` beside the rollout.

    Returns:
        Mapping of column name -> array, plus ``goal_pos_w`` stacked ``(E, 3)``.
    """
    want = ("episode_uid", "terrain_id", "reached_goal", "first_fall_step", "steps_survived",
            "n_replans", "censored", "max_tilt_rad", "min_base_height_w")
    with h5py.File(episodes_path, "r") as f:
        out = {k: f[k][:] for k in want if k in f}
        out["goal_pos_w"] = np.stack(
            [f["goal_pos_w_x"][:], f["goal_pos_w_y"][:], f["goal_pos_w_z"][:]], axis=-1)
    return out


def _window_rows(rollout: str, n_max: int, seed: int) -> np.ndarray:
    """A reproducible random subset of written window rows.

    Random rather than the first ``n``: rows are written in env-major order per re-plan, so a
    prefix is a handful of episodes' opening plans and nothing else.

    Args:
        rollout: The rollout ``.h5``.
        n_max: Maximum rows to take.
        seed: RNG seed.

    Returns:
        Sorted row indices (sorted so the h5 reads stay sequential).
    """
    n = trio.rollout_meta(rollout)["_num_written"]
    if n_max <= 0 or n_max >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n_max, replace=False))


def _grade_rows(
    rollout: str,
    episodes: dict[str, np.ndarray],
    table: trio.TileTable,
    model: PlanColliderModel,
    rows: np.ndarray,
    spec: TerrainConsistencySpec,
    thresh: float,
    dilate: int,
    max_gap: int = CONTACT_MAX_GAP_NODES,
    chunk: int | None = None,
    keep_nodes: bool = False,
) -> dict[str, np.ndarray]:
    """Grade a set of rollout windows, in chunks.

    Args:
        rollout: The rollout ``.h5``.
        episodes: The per-episode index from :func:`_episode_index`.
        table: The tile table.
        model: The collider model.
        rows: Window row indices to grade.
        spec: The reward constants.
        thresh: Contact speed threshold (m/s).
        dilate: Contact dilation radius (nodes).
        chunk: Rows per read; ``None`` derives it from :data:`SDF_TARGET_CELLS`.
        keep_nodes: Also return the per-node arrays (large; for the viewer / report).

    Returns:
        Mapping of column name -> ``(len(rows), ...)`` array.  Always includes the scalars,
        the joined episode outcome, and the node-0 placement residual.
    """
    meta = trio.rollout_meta(rollout)
    blocks = tuple(meta.get("reference_blocks") or trio.DEFAULT_BLOCKS)
    nj = len(G1_JOINT_NAMES)
    uid_to_ep = {int(u): i for i, u in enumerate(episodes["episode_uid"])}
    corner_ep = trio.corner_from_goal(table, episodes["terrain_id"], episodes["goal_pos_w"])

    scalars = ("contact_penalty", "penetration_sum_m",
               "contact_reward", "penetration_volume_m3", "penetration_sq",
               "penetration_mean", "clearance", "support_fraction", "contact_coverage",
               "exempt_fraction", "max_penetration_m",
               "contact_penalty_served", "penetration_sum_m_served",
               "contact_reward_served", "penetration_volume_m3_served",
               "support_fraction_served", "contact_coverage_served")
    acc: dict[str, list] = {k: [] for k in scalars}
    for k in ("episode_uid", "replan_index", "terrain_id", "reached_goal", "fell",
              "anchor_pitch_rad", "node0_sole_z", "node0_top_z", "row"):
        acc[k] = []
    if keep_nodes:
        for k in ("node_contact_penalty", "node_penetration_m", "node_sd_min",
                  "node_foot_clearance", "node_foot_abs", "foot_speed"):
            acc[k] = []

    if chunk is None:
        f_nodes = trio.rollout_meta(rollout)["_plan_nodes"]
        chunk = max(8, SDF_TARGET_CELLS // max(f_nodes * model.num_points, 1))
    with h5py.File(rollout, "r") as f:
        for lo in range(0, len(rows), chunk):
            idx = rows[lo : lo + chunk]
            plan = torch.as_tensor(f["fm_plan"][idx], dtype=torch.float64)
            a_pos = torch.as_tensor(f["anchor_pos"][idx], dtype=torch.float64)
            a_quat = torch.as_tensor(f["anchor_quat"][idx], dtype=torch.float64)
            uid = f["episode_uid"][idx]
            rep = f["replan_index"][idx]

            ep_rows = np.array([uid_to_ep[int(u)] for u in uid])
            tid = episodes["terrain_id"][ep_rows]
            corner = corner_ep[ep_rows]

            base_pos, base_rot, q, _bq = trio.plan_to_world(plan, a_pos, a_quat, nj, 4, blocks)
            base_pos = base_pos.to(torch.float64)
            base_rot = base_rot.to(torch.float64)
            q = q.to(torch.float64)
            pts = plan_collider_points(base_pos, base_rot, q, model)
            contact = plan_contact(pts, model, dt=CONTROL_DT, thresh=thresh, dilate=dilate,
                                   max_gap=max_gap)
            terrain = trio.terrain_for_rows(table, tid, corner)
            out = terrain_consistency(pts, contact, terrain, model, spec,
                                      chunk_nodes=max(1, 8_000_000 //
                                                      max(len(idx) * model.num_points, 1)))

            for k in scalars:
                acc[k].append(out[k].numpy())
            acc["episode_uid"].append(uid)
            acc["replan_index"].append(rep)
            acc["terrain_id"].append(tid)
            acc["reached_goal"].append(episodes["reached_goal"][ep_rows])
            acc["fell"].append((episodes["first_fall_step"][ep_rows] >= 0).astype(np.uint8))
            acc["row"].append(idx)
            # The anchor's own pitch, to check it is NOT a confound: the window lift is
            # yaw-only, so the anchor's tilt is dropped by construction -- this is the
            # measurement that says so rather than the assumption.
            g = quat_apply(a_quat[:, [3, 0, 1, 2]],
                           torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(
                               len(idx), 3))
            acc["anchor_pitch_rad"].append(
                torch.arccos(g[:, 2].clamp(-1.0, 1.0)).numpy())
            # Placement residual at node 0: lowest sole corner vs the terrain under it.
            sole = pts[:, 0][:, model.sole_index.reshape(-1)]                  # (N, 8, 3)
            low = sole[..., 2].argmin(dim=1)
            p_low = sole[torch.arange(len(idx)), low]
            top = terrain.top_z_at(p_low[:, None, :2])[:, 0]
            acc["node0_sole_z"].append(p_low[:, 2].numpy())
            acc["node0_top_z"].append(top.numpy())
            if keep_nodes:
                acc["node_contact_penalty"].append(
                    out["node_contact_penalty"].numpy().astype(np.float32))
                acc["node_penetration_m"].append(
                    out["node_penetration_m"].numpy().astype(np.float32))
                acc["node_foot_abs"].append(
                    out["node_foot_abs"].numpy().astype(np.float16))
                acc["node_sd_min"].append(out["node_sd_min"].numpy().astype(np.float16))
                acc["node_foot_clearance"].append(
                    out["node_foot_clearance"].numpy().astype(np.float16))
                acc["foot_speed"].append(contact["foot_speed"].numpy().astype(np.float16))

    return {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}


def stage_frame(args, state: dict) -> None:
    """Check the placement, and refuse to proceed if the plan is not on the terrain."""
    rollout = args.rollout + ".h5"
    episodes = _episode_index(args.rollout + ".episodes.h5")
    clip = args.clip or trio.rollout_meta(rollout)["clip_library_path"]
    clip = clip if os.path.isabs(clip) else str(MGF._REPO_ROOT / clip)
    table = trio.load_tile_table(clip)
    print(f"\n=== stage frame: {os.path.basename(args.rollout)} ===")
    model, _names = _plan_model(rollout, args.isaac_joint_names, args.spacing)

    rows = _window_rows(rollout, args.frame_windows, args.seed)
    got = _grade_rows(rollout, episodes, table, model, rows,
                      TerrainConsistencySpec(), CONTACT_SPEED_DEFAULT, 0, args.max_gap)
    res = trio.placement_residual(got["node0_sole_z"], got["node0_top_z"])
    inside = float(np.mean(np.isfinite(got["node0_top_z"])))

    print(f"  windows           {len(rows)}")
    print(f"  clip library      {clip}")
    print(f"  tiles             {list(table.terrain_ids)}")
    print(f"  node-0 sole over its own tile footprint: {inside:.4f}")
    print("  node-0 lowest-sole gap above the terrain surface (m):")
    for k in ("median_m", "mean_m", "p05_m", "p95_m", "frac_within_5cm"):
        print(f"    {k:16s} {res[k]:+.4f}")
    state["frame"] = {"clip": clip, "n_windows": int(len(rows)),
                      "frac_over_footprint": inside, **res}
    if inside < 0.90:
        raise SystemExit(
            f"only {inside:.1%} of node-0 soles land over their own tile footprint. The "
            "placement is being computed in the wrong frame, so no reward number below "
            "would mean anything. (augment_registration.py refuses on the same 90% test.)")
    if abs(res["median_m"]) > 0.10:
        raise SystemExit(
            f"the median sole gap is {res['median_m']:+.3f} m. A plan's opening node is the "
            "robot's own current pose, so this should be near zero; a systematic offset "
            "means the reward is grading a displacement, not a plan.")


#: The threshold, read from the reward's own constant rather than restated -- a second copy
#: here would let the inspector calibrate one number and the reward use another.
CONTACT_SPEED_DEFAULT: float = CONTACT_SPEED_THRESH


def _plan_model(rollout: str, override: str,
                spacing: float | None = None) -> tuple[PlanColliderModel, list[str]]:
    """The collider model built in the joint order the RECORDED PLAN is actually in.

    Separated out and printed because getting it wrong is silent: the clip order and the
    policy order differ on 27 of the G1's 29 joints, and FK'ing a plan under the clip order
    produces a plausible wrong skeleton that passes a placement check. Measured here before
    the fix: it moved the plan's median foot speed from 0.19 to 0.72 m/s and its contact
    coverage from 43% to 3%, i.e. it looked exactly like a real finding about the generator.

    Args:
        rollout: The rollout ``.h5``.
        override: ``--isaac_joint_names``, passed through.
        spacing: Collider surface-sample spacing (m), or ``None`` for extreme points.

    Returns:
        ``(model, joint_names)``.
    """
    names = trio.isaac_joint_names(rollout, override)
    if list(names) == list(G1_JOINT_NAMES):
        raise SystemExit(
            "the resolved plan joint order is identical to motion_gen's CLIP order. On the "
            "G1 these differ on 27 of 29 joints, so this means the artifact was not read "
            "and the order was defaulted -- refusing rather than FK'ing a wrong skeleton.")
    print(f"  plan joint order  {len(names)} names from the policy artifact "
          f"(differs from the clip order on "
          f"{sum(a != b for a, b in zip(names, G1_JOINT_NAMES))} of {len(names)} positions)")
    model = PlanColliderModel(MGF.DEFAULT_MJCF, list(names), dtype=torch.float64,
                              spacing=spacing)
    print(f"  colliders         {model.num_points} points over {len(model.body_order)} "
          f"bodies, {model.total_area_m2:.4f} m^2 sampled "
          f"({'surface samples, r=0' if spacing else 'extreme points, r>0'}), "
          f"{model.sole_index.shape[1]} sole points per foot")
    return model, list(names)


# ---------------------------------------------------------------------------------------
# stage: calibrate
# ---------------------------------------------------------------------------------------
def _clip_sole_corners(lib: h5py.File, frame: str, n_clips: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """World sole-corner positions and the recorded contact flag for one foot.

    Reads the ankle pose straight out of ``y_des`` -- no FK chain and no window lift, so arm
    A really is independent of everything the reward computes.

    Args:
        lib: An open clip library.
        frame: The contact frame name.
        n_clips: Clips to read.

    Returns:
        ``(corners (N, T, 4, 3), valid (N, T) bool)``.
    """
    pn = json.loads(lib.attrs["pos_output_names"])
    pos = lib["y_des"][:n_clips][:, :, [pn.index(f"{frame}:pos_{a}") for a in "xyz"]]
    ori = lib["y_des"][:n_clips][:, :, [pn.index(f"{frame}:ori_{a}") for a in "wxyz"]]
    n, t, _ = pos.shape
    q = torch.as_tensor(ori.reshape(-1, 4), dtype=torch.float64)
    c = torch.as_tensor(np.asarray(foot_box_corners(dtype=torch.float64)))
    off = quat_apply(q[:, None, :].expand(-1, 4, 4).reshape(-1, 4),
                     c[None].expand(q.shape[0], 4, 3).reshape(-1, 3)
                     ).reshape(n, t, 4, 3).numpy()
    corners = pos[:, :, None, :] + off
    lens = lib["clip_len"][:n_clips]
    valid = np.arange(t)[None, :] < lens[:, None]
    return corners, valid


def _speed_from_corners(corners: np.ndarray, dt: float) -> np.ndarray:
    """Min-over-corners speed by central finite difference; ``(N, T)``."""
    v = np.zeros_like(corners)
    v[:, 1:-1] = (corners[:, 2:] - corners[:, :-2]) / (2.0 * dt)
    v[:, 0] = (corners[:, 1] - corners[:, 0]) / dt
    v[:, -1] = (corners[:, -1] - corners[:, -2]) / dt
    return np.linalg.norm(v, axis=-1).min(axis=-1)


def _pr(pred: np.ndarray, truth: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    """Precision / recall / f1 / rates of a boolean detector.

    Args:
        pred: Predicted flag.
        truth: Reference flag.
        keep: Mask of usable samples.

    Returns:
        The summary, with counts so a suspiciously easy precision can be read against the
        base rate it was achieved against.
    """
    p, t = pred[keep], truth[keep]
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "pred_rate": float(p.mean()), "base_rate": float(t.mean()),
            "n": int(keep.sum()), "false_positives": fp}


def _sdf_contact_flag(corners: np.ndarray, table: trio.TileTable,
                      terrain_id: np.ndarray) -> np.ndarray:
    """Arm B's reference flag: the library's geometric test, on the REAL terrain.

    Same rule the library applies (a sole corner within ``CONTACT_Z_THRESH`` of the ground)
    with the flat-ground assumption removed.  Clip world coordinates are already tile-local
    for a tiles-then-clips library, so no corner offset is applied.

    Args:
        corners: ``(N, T, 4, 3)`` world sole corners.
        table: The tile table.
        terrain_id: ``(N,)`` per-clip tile id.

    Returns:
        ``(N, T)`` bool.
    """
    n, t = corners.shape[:2]
    rows = torch.as_tensor([table.row_of(int(x)) for x in terrain_id], dtype=torch.long)
    from motion_gen.terrain.box_sdf import BoxUnionTerrain              # noqa: PLC0415
    terrain = BoxUnionTerrain(table.boxes, table.mask).index(rows)
    pts = torch.as_tensor(corners.reshape(n, t * 4, 3), dtype=table.boxes.dtype)
    sd = terrain.sdf(pts).reshape(n, t, 4).amin(dim=-1).numpy()
    return sd < CONTACT_Z_THRESH


def stage_calibrate(args, state: dict) -> None:
    """Sweep the velocity threshold against each available ground truth."""
    out: dict = {"arms": {}}
    for lib_path in args.calib_lib:
        path = lib_path if os.path.isabs(lib_path) else str(MGF._REPO_ROOT / lib_path)
        with h5py.File(path, "r") as lib:
            if lib["y_des"].shape[0] == 0:
                print(f"  [skip] {os.path.basename(path)} holds no clips (terrain only)")
                continue
            frames = json.loads(lib.attrs["contact_frames"])
            n_clips = min(int(lib["y_des"].shape[0]), args.calib_clips)
            ttc = lib["terrain_type_code"][:n_clips]
            tid = lib["terrain_id"][:n_clips]
            dt = float(lib.attrs["dt"])
            table = trio.load_tile_table(path) if np.any(ttc != 0) else None
            recorded = lib["contact"][:n_clips]
            per_frame = {}
            for j, frame in enumerate(frames):
                corners, valid = _clip_sole_corners(lib, frame, n_clips)
                speed = _speed_from_corners(corners, dt)
                flat = (ttc == 0)[:, None] & valid
                terr = (ttc != 0)[:, None] & valid
                arms = {}
                if flat.any():
                    truth = recorded[:, :, j].astype(bool)
                    arms["A_flat_library_flag"] = {
                        str(th): _pr(speed < th, truth, flat) for th in THRESHOLD_SWEEP}
                if terr.any() and table is not None:
                    truth_b = _sdf_contact_flag(corners, table, tid)
                    arms["B_terrain_sdf_flag"] = {
                        str(th): _pr(speed < th, truth_b, terr) for th in THRESHOLD_SWEEP}
                    # How wrong the recorded flag is on terrain, quantified rather than
                    # asserted -- this is the number that justifies replacing it.
                    rec = recorded[:, :, j].astype(bool)
                    arms["_recorded_vs_sdf_on_terrain"] = _pr(rec, truth_b, terr)
                per_frame[frame] = arms
            out["arms"][os.path.basename(path)] = per_frame
    out["C_measured_contact"] = (
        "unavailable: left_foot_contact / right_foot_contact are declared in the scene cfg "
        "and read by nothing, so no recording carries them yet. Stage 4 logs them.")

    print("\n=== stage calibrate ===")
    for lib_name, per_frame in out["arms"].items():
        print(f"  {lib_name}")
        for frame, arms in per_frame.items():
            for arm, table_ in arms.items():
                if arm.startswith("_"):
                    r = table_
                    print(f"    {frame} :: recorded flag vs SDF flag on TERRAIN -> "
                          f"precision {r['precision']:.3f} recall {r['recall']:.3f} "
                          f"(base {r['base_rate']:.3f}, n={r['n']})")
                    continue
                print(f"    {frame} :: {arm}")
                for th, r in table_.items():
                    print(f"      thr={float(th):.2f} m/s  precision={r['precision']:.3f} "
                          f"recall={r['recall']:.3f} f1={r['f1']:.3f} "
                          f"pred={r['pred_rate']:.3f} base={r['base_rate']:.3f} "
                          f"fp={r['false_positives']}")
    print(f"  arm C: {out['C_measured_contact']}")
    state["calibrate"] = out


# ---------------------------------------------------------------------------------------
# stage: grade
# ---------------------------------------------------------------------------------------
def _summary(x: np.ndarray) -> dict[str, float]:
    """Median / mean / quantile summary of one column."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95)),
            "max": float(x.max())}


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank AUC of ``score`` against a boolean ``label``, by the Mann-Whitney identity.

    Args:
        score: Per-sample score.
        label: Boolean outcome.

    Returns:
        The AUC, or ``nan`` if either class is empty.
    """
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def stage_grade(args, state: dict) -> None:
    """Compute the reward over recorded plans and report what it separates."""
    rollout = args.rollout + ".h5"
    episodes = _episode_index(args.rollout + ".episodes.h5")
    clip = state.get("frame", {}).get("clip") or args.clip \
        or trio.rollout_meta(rollout)["clip_library_path"]
    clip = clip if os.path.isabs(clip) else str(MGF._REPO_ROOT / clip)
    table = trio.load_tile_table(clip)
    spec = TerrainConsistencySpec(served_nodes=args.served_nodes)
    rows = _window_rows(rollout, args.grade_windows, args.seed)
    print(f"\n=== stage grade: {len(rows)} windows, thresh={args.thresh} "
          f"dilate={args.dilate} ===")
    model, _names = _plan_model(rollout, args.isaac_joint_names, args.spacing)
    got = _grade_rows(rollout, episodes, table, model, rows, spec, args.thresh, args.dilate,
                      args.max_gap, keep_nodes=True)

    np.savez_compressed(os.path.join(args.out, "graded.npz"),
                        **{k: v for k, v in got.items()})

    fell = got["fell"].astype(bool)
    goal = got["reached_goal"].astype(bool)
    report: dict = {"n_windows": int(len(rows)), "thresh": args.thresh,
                    "dilate": args.dilate, "served_nodes": args.served_nodes,
                    "overall": {}, "by_outcome": {}, "auc": {}, "by_tile": {},
                    "sweeps": {}}

    cols = ("contact_penalty", "penetration_sum_m", "contact_penalty_served",
            "penetration_sum_m_served", "contact_reward", "penetration_volume_m3",
            "penetration_mean", "max_penetration_m", "support_fraction",
            "contact_coverage", "exempt_fraction")
    for c in cols:
        report["overall"][c] = _summary(got[c])
    for c in ("contact_penalty", "penetration_sum_m", "contact_reward",
              "penetration_volume_m3", "penetration_mean", "max_penetration_m",
              "support_fraction", "contact_coverage"):
        report["by_outcome"][c] = {
            "fell": _summary(got[c][fell]), "survived": _summary(got[c][~fell]),
            "reached_goal": _summary(got[c][goal]), "no_goal": _summary(got[c][~goal])}
        report["auc"][c] = {"vs_fell": _auc(got[c], fell), "vs_goal": _auc(got[c], goal)}
    for tid in np.unique(got["terrain_id"]):
        m = got["terrain_id"] == tid
        report["by_tile"][int(tid)] = {c: _summary(got[c][m]) for c in
                                       ("contact_penalty", "penetration_sum_m",
                                        "contact_reward", "contact_coverage")}
    # Is the anchor's own tilt a confound? The window lift is yaw-only, so it should not be.
    pitch = got["anchor_pitch_rad"]
    ok = np.isfinite(pitch)
    for c in ("contact_penalty", "penetration_sum_m", "contact_reward",
              "penetration_volume_m3", "max_penetration_m"):
        report["auc"][f"{c}_vs_anchor_tilt_corr"] = float(
            np.corrcoef(pitch[ok], got[c][ok])[0, 1]) if ok.sum() > 8 else float("nan")

    # The dilation sweep: what the buffer costs in penetration sensitivity.
    sweep_rows = rows[: min(len(rows), args.sweep_windows)]
    for r in DILATE_SWEEP:
        s = _grade_rows(rollout, episodes, table, model, sweep_rows, spec, args.thresh, r,
                        args.max_gap)
        report["sweeps"][f"dilate_{r}"] = {
            "penetration_sum_m": _summary(s["penetration_sum_m"]),
            "exempt_fraction": _summary(s["exempt_fraction"]),
            "max_penetration_m": _summary(s["max_penetration_m"])}
    for th in (0.03, 0.05, 0.08, 0.10, 0.20):
        s = _grade_rows(rollout, episodes, table, model, sweep_rows, spec, th, args.dilate,
                        args.max_gap)
        report["sweeps"][f"thresh_{th}"] = {
            "contact_penalty": _summary(s["contact_penalty"]),
            "contact_coverage": _summary(s["contact_coverage"]),
            "penetration_sum_m": _summary(s["penetration_sum_m"]),
            "exempt_fraction": _summary(s["exempt_fraction"])}

    _print_grade(report)
    state["grade"] = report


def _print_grade(r: dict) -> None:
    """Print the grade report in the order it should be read."""
    print("  overall")
    for c, s in r["overall"].items():
        if s.get("n"):
            print(f"    {c:26s} median {s['median']:+.5f}  mean {s['mean']:+.5f}  "
                  f"p05 {s['p05']:+.5f}  p95 {s['p95']:+.5f}  max {s['max']:+.5f}")
    print("  by outcome (median)")
    for c, d in r["by_outcome"].items():
        parts = "  ".join(f"{k} {v['median']:+.4f}" for k, v in d.items() if v.get("n"))
        print(f"    {c:26s} {parts}")
    print("  separation (AUC; 0.5 = no information)")
    for c, d in r["auc"].items():
        if isinstance(d, dict):
            print(f"    {c:26s} vs_fell {d['vs_fell']:.4f}  vs_goal {d['vs_goal']:.4f}")
        else:
            print(f"    {c:26s} {d:+.4f}")
    print("  by tile (median contact_penalty / penetration_sum_m / contact_reward / cover)")
    for tid, d in r["by_tile"].items():
        print(f"    tile {tid}  {d['contact_penalty']['median']:+.5f}  "
              f"{d['penetration_sum_m']['median']:+.5f}  "
              f"{d['contact_reward']['median']:+.4f}  "
              f"{d['contact_coverage']['median']:.3f}")
    print("  sweeps")
    for k, d in r["sweeps"].items():
        parts = "  ".join(f"{n}={v['median']:+.6f}" for n, v in d.items() if v.get("n"))
        print(f"    {k:16s} {parts}")


# ---------------------------------------------------------------------------------------
# stage: reward
# ---------------------------------------------------------------------------------------
def stage_reward(args, state: dict) -> None:
    """Write the per-window terrain reward as a sidecar the critic can be fitted on.

    The critic's quality target is built from a per-window reward array.  The terrain reward
    is NOT in the rollout file and cannot be derived from it -- it is a function of the plan
    and the terrain -- so it is computed here and handed over as a sidecar keyed on
    ``(episode_uid, replan_index)``.

    A sidecar rather than a new column in the recording, deliberately: the thresholds, the
    deadband, the per-body weights and the term weights all stay OFFLINE knobs, so
    re-weighting the reward costs a minute here instead of a re-record.  That is the same
    argument ``record_rollouts`` makes for storing the tracking-error VECTOR rather than a
    reward.

    Every constant that shaped the numbers is stamped alongside them, and the file carries a
    content hash, so ``fit_value`` can record WHICH reward a critic was fitted on and
    ``label_advantage`` can refuse to mix two.
    """
    rollout = args.rollout + ".h5"
    episodes = _episode_index(args.rollout + ".episodes.h5")
    clip = state.get("frame", {}).get("clip") or args.clip \
        or trio.rollout_meta(rollout)["clip_library_path"]
    clip = clip if os.path.isabs(clip) else str(MGF._REPO_ROOT / clip)
    table = trio.load_tile_table(clip)
    spec = TerrainConsistencySpec(served_nodes=args.served_nodes)

    rows = _window_rows(rollout, args.reward_windows, args.seed)
    print(f"\n=== stage reward: {len(rows)} windows ===")
    model, names = _plan_model(rollout, args.isaac_joint_names, args.spacing)
    got = _grade_rows(rollout, episodes, table, model, rows, spec, args.thresh, args.dilate,
                      args.max_gap)

    penalty = got["contact_penalty"] * spec.contact_weight \
        + got["penetration_sum_m"] * spec.penetration_weight
    reward = np.exp(-penalty / float(args.tau))

    meta = {
        "reward_source": REWARD_SOURCE,
        "tau": float(args.tau),
        "thresh": float(args.thresh), "dilate": int(args.dilate),
        "max_gap": int(args.max_gap), "spacing": args.spacing,
        "contact_weight": float(spec.contact_weight),
        "penetration_weight": float(spec.penetration_weight),
        "contact_deadband_m": float(spec.contact_deadband_m),
        "penetration_tol_m": float(spec.penetration_tol_m),
        "penetration_default_weight": float(spec.penetration_default_weight),
        "body_penetration_weights": spec.body_penetration_weights,
        "served_nodes": int(spec.served_nodes),
        "rollout": os.path.abspath(args.rollout), "clip": clip,
        "n_windows": int(len(rows)), "joint_names": list(names),
    }
    payload = {
        "episode_uid": got["episode_uid"].astype(np.int64),
        "replan_index": got["replan_index"].astype(np.int32),
        "reward": reward.astype(np.float64),
        "penalty": penalty.astype(np.float64),
        "contact_penalty": got["contact_penalty"].astype(np.float64),
        "penetration_sum_m": got["penetration_sum_m"].astype(np.float64),
        "meta_json": json.dumps(meta, sort_keys=True),
    }
    digest = hashlib.sha256(
        payload["episode_uid"].tobytes() + payload["replan_index"].tobytes()
        + payload["reward"].tobytes() + payload["meta_json"].encode()).hexdigest()[:16]
    payload["reward_hash"] = digest

    path = args.reward_out or os.path.join(args.out, "terrain_reward_sidecar.npz")
    np.savez_compressed(path, **payload)
    q = np.quantile(reward, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  R = exp(-penalty / {args.tau:g}), support (0, 1]")
    print(f"    penalty  median {np.median(penalty):.5f}  p95 {np.quantile(penalty, .95):.5f}"
          f"  max {penalty.max():.5f}")
    print(f"    reward   p05 {q[0]:.4f}  p25 {q[1]:.4f}  median {q[2]:.4f}  p75 {q[3]:.4f}"
          f"  p95 {q[4]:.4f}")
    print(f"    fraction above 0.99: {float((reward > 0.99).mean()):.4f}  "
          f"below 0.5: {float((reward < 0.5).mean()):.4f}")
    print(f"  wrote {path}  (hash {digest})")
    state["reward"] = {"path": path, "hash": digest, **meta,
                       "reward_median": float(np.median(reward))}


# ---------------------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------------------
def stage_report(args, state: dict) -> None:
    """Plot what ``grade`` wrote.  Imports matplotlib only here."""
    import matplotlib                                                   # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                     # noqa: PLC0415
    from plot_style import apply_paper_style                            # noqa: PLC0415

    apply_paper_style()
    got = np.load(os.path.join(args.out, "graded.npz"))
    fell = got["fell"].astype(bool)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    cp = np.log10(np.maximum(got["contact_penalty"], 1e-6))
    ax[0, 0].hist([cp[~fell], cp[fell]], bins=40, label=["survived", "fell"],
                  density=True, histtype="step")
    ax[0, 0].set_xlabel(r"$\log_{10}$ contact penalty (m)"); ax[0, 0].legend()
    pen = np.log10(np.maximum(got["penetration_sum_m"], 1e-12))
    ax[0, 1].hist([pen[~fell], pen[fell]], bins=40, label=["survived", "fell"],
                  density=True, histtype="step")
    ax[0, 1].set_xlabel(r"$\log_{10}$ penetration sum (m)")
    ax[0, 1].legend()
    ax[0, 2].hist(got["contact_coverage"], bins=40)
    ax[0, 2].set_xlabel("contact coverage (fraction of foot-nodes marked)")

    ax[1, 0].plot(got["node_penetration_m"].mean(axis=0), label="mean")
    ax[1, 0].plot(np.quantile(got["node_penetration_m"], 0.95, axis=0), label="p95")
    ax[1, 0].set_xlabel("plan node"); ax[1, 0].set_ylabel("penetration"); ax[1, 0].legend()
    clr = got["node_foot_clearance"].astype(np.float64)
    ax[1, 1].plot(np.median(clr[:, :, 0], axis=0), label="left")
    ax[1, 1].plot(np.median(clr[:, :, 1], axis=0), label="right")
    ax[1, 1].axhline(0.0, color="k", lw=0.8)
    ax[1, 1].set_xlabel("plan node"); ax[1, 1].set_ylabel("foot clearance (m)")
    ax[1, 1].legend()
    ax[1, 2].scatter(got["anchor_pitch_rad"], got["contact_penalty"], s=1, alpha=0.15)
    ax[1, 2].set_xlabel("anchor tilt from upright (rad)")
    ax[1, 2].set_ylabel("contact penalty (m)")
    ax[1, 2].set_yscale("log")

    fig.tight_layout()
    path = os.path.join(args.out, "terrain_reward.png")
    fig.savefig(path, dpi=140)
    print(f"\n=== stage report: wrote {path} ===")
    state.setdefault("report", {})["png"] = path


# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all",
                   choices=("frame", "calibrate", "grade", "reward", "report", "all"))
    p.add_argument("--rollout", required=True,
                   help="rollout stem, WITHOUT .h5 (its .episodes.h5 must sit beside it)")
    p.add_argument("--clip", default="", help="clip library; default: the rollout's own")
    p.add_argument("--calib_lib", action="append", default=[],
                   help="clip library for the contact calibration; repeatable")
    p.add_argument("--calib_clips", type=int, default=200,
                   help="clips per calibration library")
    p.add_argument("--out", default="", help="output dir; default <rollout dir>/terrain_reward")
    p.add_argument("--frame_windows", type=int, default=4096)
    p.add_argument("--grade_windows", type=int, default=20000)
    p.add_argument("--sweep_windows", type=int, default=4096)
    p.add_argument("--reward_windows", type=int, default=0,
                   help="windows to grade for the sidecar; 0 = ALL of them, which is what a "
                        "critic fit wants")
    p.add_argument("--reward_out", default="", help="sidecar path; default <out>/terrain_reward_sidecar.npz")
    p.add_argument("--tau", type=float, default=REWARD_TAU,
                   help="temperature of the penalty -> reward map, R = exp(-penalty/tau)")
    p.add_argument("--thresh", type=float, default=CONTACT_SPEED_DEFAULT)
    p.add_argument("--dilate", type=int, default=CONTACT_DILATE_NODES)
    p.add_argument("--max_gap", type=int, default=CONTACT_MAX_GAP_NODES,
                   help="close runs of non-contact this long between two stances")
    p.add_argument("--served_nodes", type=int, default=12)
    p.add_argument("--spacing", type=float, default=DEFAULT_COLLIDER_SPACING,
                   help="collider surface-sample spacing (m); 0 for extreme points only")
    p.add_argument("--isaac_joint_names", default="",
                   help="policy_parameters.yaml (or a run dir) stating joint_names_isaac; "
                        "default: derived from the rollout's own checkpoint_path")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    """Run the requested stages, sharing one json."""
    args = build_parser().parse_args()
    args.spacing = args.spacing if args.spacing and args.spacing > 0 else None
    if not args.out:
        args.out = os.path.join(os.path.dirname(os.path.abspath(args.rollout)),
                                "terrain_reward")
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    state_path = os.path.join(args.out, "terrain_reward.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}

    stages = ("frame", "calibrate", "grade", "reward", "report") if args.stage == "all" \
        else (args.stage,)
    for name in stages:
        if name == "calibrate" and not args.calib_lib:
            print("\n=== stage calibrate: skipped (no --calib_lib given) ===")
            continue
        {"frame": stage_frame, "calibrate": stage_calibrate, "grade": stage_grade,
         "reward": stage_reward, "report": stage_report}[name](args, state)
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, default=float)
    print(f"\nwrote {state_path}")


if __name__ == "__main__":
    main()
