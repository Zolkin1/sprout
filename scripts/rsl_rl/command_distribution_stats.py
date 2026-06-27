# Copyright 2026 Zachary Olkin. All rights reserved.

"""Command-distribution statistics for motion clips and recorded FM datasets.

This is a training-data triage tool. Given a motion-clip library (``.h5``) or a
recorded ``(obs, action)`` dataset (``.npz``, the kind consumed by the
flow-matching trainer), it reports how the *commanded velocity* is distributed
over time so you can judge whether the data covers the regimes you care about
and rebalance the generation distributions if not.

Every control timestep is placed into exactly one of five mutually-exclusive
command-dynamics buckets (so the percentages sum to 100% of time):

* ``steady_standing``  -- command held constant, in the standing bucket.
* ``steady_moving``    -- command held constant, in a moving skill.
* ``enter_standing``   -- command ramping from a moving skill *into* standing
                          (coming to a stop).
* ``exit_standing``    -- command ramping *out of* standing into a moving skill
                          (starting from a stop).
* ``change_moving``    -- command ramping between two moving skills / speeds
                          (e.g. walk -> run), no standing endpoint involved.

From these the headline numbers fall out:

* ``% commanded steady state``       = steady_standing + steady_moving
* ``% changing velocities``          = enter + exit + change_moving
* ``% entering or exiting standing`` = enter_standing + exit_standing

Separately it reports the fraction of *time* spent in each skill (occupancy),
plus a few distribution diagnostics (commanded vx percentiles, start/stop event
counts). A ramp is attributed to standing-entry/exit by its *endpoints* (the
settled skill before and after the ramp), so the whole stop/start ramp counts as
such -- not just the single step that crosses the standing velocity boundary.

Skill labels:

* Clips carry a per-step ``skill_id`` and a ``skill_list``; these are used
  directly (authoritative).
* Recorded datasets carry no skill labels (the FM model is command-conditioned),
  so the skill is *inferred* per step from ``target_vel`` using velocity buckets
  -- the same box-nearest rule the data generator uses
  (``skill_bucket.bucket_for_velocity``). The default buckets match the G1
  walk/run setup; override with ``--buckets-json`` to match a different task.

The tool is numpy-only (plus ``h5py`` for clips, optional ``matplotlib`` for
``--plot``); it does not import Isaac Lab, so it runs quickly anywhere.

Examples::

    # A single clip library
    python scripts/rsl_rl/command_distribution_stats.py clips/g1_teacher_v1.h5

    # All recorded shards used for an FM run, aggregated, with plots
    python scripts/rsl_rl/command_distribution_stats.py \
        logs/.../recorded_data/ --plot --out-dir /tmp/cmd_stats

    # Recorded data with custom skill buckets + an expected-mix comparison
    python scripts/rsl_rl/command_distribution_stats.py data.npz \
        --buckets-json my_buckets.json --reference-mix clips/skill_mix.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

# ``recorded_data_io`` is a sibling module in this directory (numpy-only, no
# Isaac Lab); importing it works whether run as a script or imported.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recorded_data_io import load_recorded_data  # noqa: E402

# Default velocity buckets for skill inference on recorded data (no stored
# skill labels). Mirrors ``g1_walk_run_env_cfg`` / ``g1_clf_multiskill_base``:
# only lin_vel_x is partitioned; lin_vel_y / ang_vel_z default to (0, 0). Order
# defines the integer skill ids and the lowest-index tie-break. Override with
# ``--buckets-json`` for other tasks (e.g. terrain skills, lateral/yaw ranges).
DEFAULT_BUCKETS: dict[str, dict[str, tuple[float, float]]] = {
    "walk_forward": {"lin_vel_x": (0.1, 1.5)},
    "running": {"lin_vel_x": (1.5, 3.7)},
    "standing": {"lin_vel_x": (0.0, 0.1)},
}

# Command-dynamics categories, in display order. The first five partition time;
# index into this list is the per-step category code.
CATEGORIES = ("steady_standing", "steady_moving", "enter_standing", "exit_standing", "change_moving")
_CAT_IDX = {name: i for i, name in enumerate(CATEGORIES)}

# Name used to identify the standing skill in a skill list (case-insensitive).
STANDING_NAME = "standing"


@dataclass
class MotionData:
    """A normalized per-timestep view of commanded velocity + skill labels.

    Multiple clips/episodes are concatenated along the single time axis; segment
    boundaries are encoded by ``prev_valid`` (False at the first step of each
    segment), so step-to-step deltas never span a boundary.

    Attributes:
        vel: ``[M, 3]`` commanded velocity per step ``(vx, vy, wz)``.
        skill: ``[M]`` skill id per step, indexing ``skill_list``.
        prev_valid: ``[M]`` bool; True where ``vel[m-1]`` is the temporal
            predecessor of step ``m`` (i.e. same segment, contiguous).
        skill_list: Ordered skill names; ``skill`` indexes into this.
        standing_idx: Index of the standing skill in ``skill_list``, or None.
        dt: Control timestep in seconds.
        n_segments: Number of contiguous segments (clips/episodes).
        sources: Human-readable descriptions of the contributing files.
        fmt: ``"clip"``, ``"recorded"``, or ``"mixed"``.
        has_command: Whether a real commanded velocity was available (False if a
            clip lacked ``target_vel``; only skill occupancy is then meaningful).
        perturb_active: ``[M]`` uint8 per-step perturbation flag (``1`` while a
            perturbation window was active), aligned to ``vel``/``skill``; ``None``
            for clips or recordings without it (record_data.py ``--perturb``).
    """

    vel: np.ndarray
    skill: np.ndarray
    prev_valid: np.ndarray
    skill_list: list[str]
    standing_idx: int | None
    dt: float
    n_segments: int
    sources: list[str]
    fmt: str
    has_command: bool
    perturb_active: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Skill inference (numpy port of skill_bucket.bucket_for_velocity)
# ---------------------------------------------------------------------------

def bucket_for_velocity_np(
    vel: np.ndarray, lin_x: np.ndarray, lin_y: np.ndarray, ang_z: np.ndarray
) -> np.ndarray:
    """Per-step skill id by L2 distance from ``vel`` to each bucket's box.

    Numpy port of ``...traj_tracking.skill_bucket.bucket_for_velocity`` (with no
    eligibility mask -- every skill is eligible here). For each skill the edge
    distance is the L2 distance from ``vel`` to the nearest point of its
    axis-aligned ``[lo, hi]`` box (zero inside it); the argmin wins, with ties
    (e.g. a velocity on a shared bucket boundary) broken to the lowest index.

    Args:
        vel: ``[M, 3]`` velocities ``(vx, vy, wz)``.
        lin_x: ``[S, 2]`` ``(min, max)`` lin-x bucket per skill.
        lin_y: ``[S, 2]`` ``(min, max)`` lin-y bucket per skill.
        ang_z: ``[S, 2]`` ``(min, max)`` ang-z bucket per skill.

    Returns:
        ``[M]`` int array of skill indices.
    """
    vx, vy, wz = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]              # [M, 1]
    cx = np.minimum(np.maximum(vx, lin_x[:, 0]), lin_x[:, 1])       # [M, S]
    cy = np.minimum(np.maximum(vy, lin_y[:, 0]), lin_y[:, 1])
    cw = np.minimum(np.maximum(wz, ang_z[:, 0]), ang_z[:, 1])
    dist2 = (vx - cx) ** 2 + (vy - cy) ** 2 + (wz - cw) ** 2        # [M, S]
    return np.argmin(dist2, axis=1)


def _bucket_tables(buckets: dict[str, dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(skill_list, lin_x, lin_y, ang_z)`` arrays from a buckets dict.

    Missing axes default to ``(0.0, 0.0)`` (the ``VelocityBucketCfg`` default),
    so a bucket specifying only ``lin_vel_x`` pins lateral/yaw to zero.
    """
    skill_list = list(buckets.keys())

    def col(attr: str) -> np.ndarray:
        return np.array(
            [tuple(buckets[s].get(attr, (0.0, 0.0))) for s in skill_list], dtype=np.float64
        )

    return skill_list, col("lin_vel_x"), col("lin_vel_y"), col("ang_vel_z")


def _standing_index(skill_list: list[str]) -> int | None:
    """Index of the standing skill in ``skill_list`` (case-insensitive), or None."""
    for i, s in enumerate(skill_list):
        if s.lower() == STANDING_NAME:
            return i
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_clip_file(path: str) -> MotionData:
    """Load a motion-clip library (``.h5``) into a :class:`MotionData`.

    Reads the HDF5 directly (mirroring ``clips.clip_io`` layout) to avoid the
    Isaac Lab package import chain. Uses the stored per-step ``skill_id`` /
    ``skill_list`` for skills and ``target_vel`` for the command; trims to the
    number of clips actually written.

    Args:
        path: Path to a clip-library ``.h5``.

    Returns:
        A :class:`MotionData` (one segment per clip).
    """
    import h5py

    with h5py.File(path, "r") as f:
        def attr(name: str, default=None):
            if name not in f.attrs:
                return default
            v = f.attrs[name]
            if isinstance(v, (bytes, str)):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    return v
            return v.item() if isinstance(v, np.generic) else v

        skill_list = list(attr("skill_list", []) or [])
        dt = float(attr("dt", 0.02))
        if "skill_id" not in f:
            raise ValueError(f"{path}: not a clip library (no 'skill_id' dataset).")
        n_written = int(attr("num_clips_written", attr("num_clips", f["skill_id"].shape[0])))
        skill_id = f["skill_id"][:n_written].astype(np.int64)              # [N, T]
        target_vel = f["target_vel"][:n_written] if "target_vel" in f else None

    n_clips, T = skill_id.shape
    skill = skill_id.reshape(-1)                                            # [N*T]

    # target_vel may be absent (older lib) or an all-zero placeholder (dataset
    # created but never written); treat both as "no command" -> skill-only.
    has_command = target_vel is not None and bool(np.any(target_vel))
    if has_command:
        vel = target_vel.reshape(-1, 3).astype(np.float64)                 # [N*T, 3]
    else:
        vel = np.zeros((skill.shape[0], 3), dtype=np.float64)

    # prev_valid: False at each clip's first step (clip-major, time-minor flatten).
    prev_valid = np.ones(skill.shape[0], dtype=bool)
    prev_valid[::T] = False

    return MotionData(
        vel=vel, skill=skill, prev_valid=prev_valid, skill_list=skill_list,
        standing_idx=_standing_index(skill_list), dt=dt, n_segments=n_clips,
        sources=[f"{os.path.basename(path)} (clip: {n_clips} clips x {T} steps)"],
        fmt="clip", has_command=has_command,
    )


def load_recorded_file(path: str, buckets: dict[str, dict]) -> MotionData:
    """Load a recorded FM dataset (``.npz``) into a :class:`MotionData`.

    The recorded ``target_vel`` is the commanded velocity that drove the
    reference (gathered straight from the clip, so it has the same piecewise
    ramp character). Skills are inferred per step from ``target_vel`` via
    ``buckets``. Segments are reconstructed from ``episode_id`` /
    ``step_in_episode`` so deltas never span an episode or env boundary.

    Args:
        path: Path to a recorded ``.npz`` (see ``recorded_data_io``).
        buckets: Velocity buckets for skill inference.

    Returns:
        A :class:`MotionData` (one segment per recorded episode).
    """
    data, meta = load_recorded_data(path)
    tv = data.get("target_vel")
    if tv is None:
        raise ValueError(
            f"{path}: recorded dataset has no 'target_vel'. Re-record on a clip "
            "task that exposes the command velocity (command_velocity_available)."
        )
    tv = tv.astype(np.float64)
    ep = data["episode_id"]
    st = data["step_in_episode"]

    skill_list, lin_x, lin_y, ang_z = _bucket_tables(buckets)
    skill = bucket_for_velocity_np(tv, lin_x, lin_y, ang_z)

    # Sort into (episode, step) order so each episode's steps are contiguous and
    # ascending, regardless of the env-interleaved row-major storage order.
    order = np.lexsort((st, ep))
    ep_s, st_s = ep[order], st[order]
    tv_s, skill_s = tv[order], skill[order]

    prev_valid = np.zeros(tv_s.shape[0], dtype=bool)
    if tv_s.shape[0] > 1:
        prev_valid[1:] = (ep_s[1:] == ep_s[:-1]) & (st_s[1:] == st_s[:-1] + 1)
    n_segments = int((~prev_valid).sum())  # segment starts == rows w/o a predecessor

    # Per-step perturbation flag (recovery recordings); sort into the same order.
    pa = data.get("perturb_active")
    perturb_active = pa[order].astype(np.uint8) if pa is not None else None

    return MotionData(
        vel=tv_s, skill=skill_s, prev_valid=prev_valid, skill_list=skill_list,
        standing_idx=_standing_index(skill_list), dt=float(meta.dt), n_segments=n_segments,
        sources=[f"{os.path.basename(path)} (recorded: {tv_s.shape[0]} steps, "
                 f"{n_segments} episodes, task={meta.task_name})"],
        fmt="recorded", has_command=True, perturb_active=perturb_active,
    )


def merge_motion_data(items: list[MotionData]) -> MotionData:
    """Concatenate several :class:`MotionData` into one, unifying skill lists.

    Skill ids are remapped onto a union skill list (first-seen order across
    inputs) so clips with different skill orderings -- or a clip mixed with
    recorded data -- aggregate correctly. ``prev_valid`` concatenation is safe
    because each input's first step is already a segment boundary.

    Args:
        items: Non-empty list of loaded :class:`MotionData`.

    Returns:
        A single merged :class:`MotionData`.
    """
    if len(items) == 1:
        return items[0]

    global_list: list[str] = []
    for md in items:
        for s in md.skill_list:
            if s not in global_list:
                global_list.append(s)

    vels, skills, pvs, sources = [], [], [], []
    for md in items:
        remap = np.array([global_list.index(s) for s in md.skill_list], dtype=np.int64)
        skills.append(remap[md.skill] if md.skill_list else md.skill)
        vels.append(md.vel)
        pvs.append(md.prev_valid)
        sources.extend(md.sources)

    dts = {round(md.dt, 6) for md in items}
    if len(dts) > 1:
        print(f"[WARN] inputs have differing dt {sorted(dts)}; using {items[0].dt}.")

    # Perturbation flag only carries through if every input has it (else None).
    if all(md.perturb_active is not None for md in items):
        perturb_active = np.concatenate([md.perturb_active for md in items])
    else:
        perturb_active = None

    fmts = {md.fmt for md in items}
    return MotionData(
        vel=np.concatenate(vels), skill=np.concatenate(skills),
        prev_valid=np.concatenate(pvs), skill_list=global_list,
        standing_idx=_standing_index(global_list), dt=items[0].dt,
        n_segments=sum(md.n_segments for md in items), sources=sources,
        fmt=fmts.pop() if len(fmts) == 1 else "mixed",
        has_command=all(md.has_command for md in items),
        perturb_active=perturb_active,
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(md: MotionData, lin_thresh: float, ang_thresh: float) -> dict:
    """Compute command-dynamics + skill-occupancy statistics.

    Each step is assigned one category in :data:`CATEGORIES`. A step is
    "changing" if any command component moved more than its threshold since the
    previous step; maximal runs of changing steps are then classified by their
    settled endpoints (skill just before vs the skill the ramp lands on) so a
    whole stop/start ramp is attributed to standing entry/exit.

    Args:
        md: The motion data to analyze.
        lin_thresh: Per-step abs-delta threshold (m/s) for vx/vy to count as
            "changing".
        ang_thresh: Per-step abs-delta threshold (rad/s) for wz.

    Returns:
        A dict of arrays/scalars consumed by :func:`format_report` /
        :func:`make_plots`.
    """
    vel, skill, pv = md.vel, md.skill, md.prev_valid
    M = vel.shape[0]
    S = len(md.skill_list)
    std = md.standing_idx

    occupancy = np.bincount(skill, minlength=S).astype(np.int64)  # steps per skill

    # Default every step to steady, classified by its own skill.
    is_standing_step = (skill == std) if std is not None else np.zeros(M, dtype=bool)
    cat = np.where(is_standing_step, _CAT_IDX["steady_standing"], _CAT_IDX["steady_moving"])

    n_enter = n_exit = n_change = 0
    if md.has_command and M > 1:
        # changing[m] (only where m has a predecessor): command moved since m-1.
        dv = np.abs(vel - np.roll(vel, 1, axis=0))
        changing = (dv[:, 0] > lin_thresh) | (dv[:, 1] > lin_thresh) | (dv[:, 2] > ang_thresh)
        changing &= pv  # invalid at segment boundaries

        # Maximal runs of consecutive changing steps (boundaries are False, so
        # runs never span segments).
        starts = np.flatnonzero(changing & ~np.r_[False, changing[:-1]])
        ends = np.flatnonzero(changing & ~np.r_[changing[1:], False])
        for s, e in zip(starts, ends):
            before = skill[s - 1]          # settled skill before the ramp
            after = skill[e]               # skill the ramp lands on
            before_st = std is not None and before == std
            after_st = std is not None and after == std
            if before_st ^ after_st:
                if after_st:
                    cat[s:e + 1] = _CAT_IDX["enter_standing"]
                    n_enter += 1
                else:
                    cat[s:e + 1] = _CAT_IDX["exit_standing"]
                    n_exit += 1
            else:
                cat[s:e + 1] = _CAT_IDX["change_moving"]
                n_change += 1

    cat_counts = np.bincount(cat, minlength=len(CATEGORIES)).astype(np.int64)

    # Commanded forward-speed distribution (diagnostic), split by moving vs all.
    vx = vel[:, 0]
    moving = ~is_standing_step
    pct = [5, 25, 50, 75, 95, 99]
    vx_pctl_all = np.percentile(vx, pct) if M else np.full(len(pct), np.nan)
    vx_pctl_moving = np.percentile(vx[moving], pct) if moving.any() else np.full(len(pct), np.nan)

    # Perturbation coverage (recovery recordings): per-step flag -> total time +
    # window count (contiguous runs of active steps, not spanning segments).
    has_perturb = md.perturb_active is not None
    perturb_steps = perturb_windows = 0
    if has_perturb:
        active_p = md.perturb_active.astype(bool)
        perturb_steps = int(active_p.sum())
        prev_active = np.r_[False, active_p[:-1]]
        # A window starts at an active step that either follows an inactive step or
        # begins a new segment (so windows never merge across a segment boundary).
        starts_p = active_p & (~prev_active | ~pv)
        perturb_windows = int(starts_p.sum())

    return {
        "M": M,
        "dt": md.dt,
        "n_segments": md.n_segments,
        "skill_list": md.skill_list,
        "standing_idx": std,
        "has_command": md.has_command,
        "fmt": md.fmt,
        "sources": md.sources,
        "occupancy": occupancy,
        "cat_counts": cat_counts,
        "n_enter": n_enter,
        "n_exit": n_exit,
        "n_change": n_change,
        "vx_pct": pct,
        "vx_pctl_all": vx_pctl_all,
        "vx_pctl_moving": vx_pctl_moving,
        "vx": vx,  # for plotting
        "has_perturb": has_perturb,
        "perturb_steps": perturb_steps,
        "perturb_windows": perturb_windows,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(n: int, total: int) -> str:
    """Format ``n/total`` as a right-aligned percentage string."""
    return f"{100.0 * n / total:6.2f}%" if total else "   -  "


def format_report(stats: dict, reference_mix: dict | None = None) -> str:
    """Render the human-readable text report.

    Args:
        stats: Output of :func:`analyze`.
        reference_mix: Optional ``{skill: prob}`` expected *sampling* mix (e.g.
            from ``clips/skill_mix.json``) shown alongside the measured *time*
            occupancy. Note these measure different things (stride-sampling
            probability vs. dwell time), so they are not expected to match.

    Returns:
        The report as a single string.
    """
    M = stats["M"]
    dt = stats["dt"]
    cc = stats["cat_counts"]
    skill_list = stats["skill_list"]
    L: list[str] = []

    L.append("=" * 72)
    L.append("COMMAND DISTRIBUTION STATS")
    L.append("=" * 72)
    for s in stats["sources"]:
        L.append(f"  - {s}")
    total_s = M * dt
    L.append(f"format: {stats['fmt']}   steps: {M:,}   segments: {stats['n_segments']:,}   "
             f"dt: {dt:.4f}s   duration: {total_s:,.1f}s ({total_s / 60.0:,.1f} min)")
    if stats["standing_idx"] is None:
        L.append("[WARN] no 'standing' skill present -> standing entry/exit reported as 0.")
    if not stats["has_command"]:
        L.append("[WARN] no commanded velocity in input -> dynamics are all 'steady'; "
                 "only skill occupancy is meaningful.")

    # -- Headline command dynamics --
    steady = int(cc[_CAT_IDX["steady_standing"]] + cc[_CAT_IDX["steady_moving"]])
    enter = int(cc[_CAT_IDX["enter_standing"]])
    exit_ = int(cc[_CAT_IDX["exit_standing"]])
    change_moving = int(cc[_CAT_IDX["change_moving"]])
    changing = enter + exit_ + change_moving
    L.append("")
    L.append("-- Command dynamics (% of time) --")
    L.append(f"  steady state .................... {_pct(steady, M)}   ({steady:,} steps)")
    L.append(f"  changing velocity .............. {_pct(changing, M)}   ({changing:,} steps)")
    L.append(f"  entering/exiting standing ...... {_pct(enter + exit_, M)}   ({enter + exit_:,} steps)")

    # -- Full partition --
    L.append("")
    L.append("-- Full partition (% of time) --")
    labels = {
        "steady_standing": "steady - standing",
        "steady_moving": "steady - moving",
        "exit_standing": "ramp - exiting standing (start)",
        "enter_standing": "ramp - entering standing (stop)",
        "change_moving": "ramp - moving<->moving",
    }
    for name in ("steady_standing", "steady_moving", "exit_standing", "enter_standing", "change_moving"):
        n = int(cc[_CAT_IDX[name]])
        L.append(f"  {labels[name]:<34} {_pct(n, M)}   ({n:,} steps)")

    # -- Start/stop events --
    L.append("")
    L.append("-- Start/stop events --")
    L.append(f"  stops  (-> standing) : {stats['n_enter']:,}")
    L.append(f"  starts (<- standing) : {stats['n_exit']:,}")
    L.append(f"  moving speed changes : {stats['n_change']:,}")

    # -- Perturbation coverage (recovery recordings only) --
    if stats.get("has_perturb"):
        ps, pw = stats["perturb_steps"], stats["perturb_windows"]
        L.append("")
        L.append("-- Perturbation coverage (record_data.py --perturb) --")
        L.append(f"  perturbed time .............. {_pct(ps, M)}   ({ps:,} steps)")
        L.append(f"  clean time (FM prediction) . {_pct(M - ps, M)}   ({M - ps:,} steps)")
        mean_len = (ps / pw * dt) if pw else 0.0
        L.append(f"  perturbation windows ....... {pw:,}   (mean {mean_len:.2f}s each)")

    # -- Skill occupancy --
    occ = stats["occupancy"]
    L.append("")
    L.append("-- Skill occupancy (% of time) --")
    ref = {k.lower(): float(v) for k, v in (reference_mix or {}).items()}
    ref_total = sum(ref.values()) or 1.0
    header = f"  {'skill':<16} {'time%':>8} {'steps':>12}"
    if ref:
        header += f"   {'ref(sample)%':>13}"
    L.append(header)
    for i, name in enumerate(skill_list):
        row = f"  {name:<16} {_pct(int(occ[i]), M):>8} {int(occ[i]):>12,}"
        if ref:
            rv = ref.get(name.lower())
            row += f"   {(100.0 * rv / ref_total):>12.2f}%" if rv is not None else f"   {'-':>13}"
        L.append(row)
    if ref:
        L.append("  (ref% is the configured stride-sampling mix, not a time target -- "
                 "shown for context only.)")

    # -- Commanded vx distribution --
    L.append("")
    L.append("-- Commanded vx distribution (m/s) --")
    pcs = stats["vx_pct"]
    L.append("  pctl:        " + "  ".join(f"p{p}" for p in pcs))
    L.append("  all steps:   " + "  ".join(f"{v:5.2f}" for v in stats["vx_pctl_all"]))
    L.append("  moving only: " + "  ".join(f"{v:5.2f}" for v in stats["vx_pctl_moving"]))

    L.append("=" * 72)
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def make_plots(stats: dict, out_dir: str, buckets: dict | None, reference_mix: dict | None) -> list[str]:
    """Write diagnostic PNGs: dynamics partition, skill occupancy, vx histogram.

    Args:
        stats: Output of :func:`analyze`.
        out_dir: Directory to write PNGs into (created if missing).
        buckets: Velocity buckets (for vx histogram boundary markers), or None.
        reference_mix: Optional expected sampling mix overlaid on occupancy.

    Returns:
        List of written file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    M = stats["M"]
    written: list[str] = []

    # 1. Command-dynamics partition.
    cc = stats["cat_counts"]
    order = ("steady_standing", "steady_moving", "exit_standing", "enter_standing", "change_moving")
    vals = [100.0 * cc[_CAT_IDX[n]] / M for n in order]
    colors = ["#4477aa", "#66ccee", "#228833", "#ccbb44", "#ee6677"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    bars = ax.bar(range(len(order)), vals, color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([n.replace("_", "\n") for n in order])
    ax.set_ylabel("% of time")
    ax.set_title("Command dynamics partition")
    ax.bar_label(bars, fmt="%.1f%%", padding=2)
    ax.grid(True, axis="y", alpha=0.3)
    p = os.path.join(out_dir, "dynamics_partition.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 2. Skill occupancy (vs reference mix if given).
    occ = stats["occupancy"]
    skill_list = stats["skill_list"]
    occ_pct = [100.0 * c / M for c in occ]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = np.arange(len(skill_list))
    if reference_mix:
        ref = {k.lower(): float(v) for k, v in reference_mix.items()}
        rt = sum(ref.values()) or 1.0
        ref_pct = [100.0 * ref.get(s.lower(), np.nan) / rt for s in skill_list]
        ax.bar(x - 0.2, occ_pct, 0.4, label="time occupancy", color="#4477aa")
        ax.bar(x + 0.2, ref_pct, 0.4, label="ref sampling mix", color="#ee6677")
        ax.legend()
    else:
        b = ax.bar(x, occ_pct, color="#4477aa")
        ax.bar_label(b, fmt="%.1f%%", padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(skill_list, rotation=20, ha="right")
    ax.set_ylabel("% of time")
    ax.set_title("Skill occupancy")
    ax.grid(True, axis="y", alpha=0.3)
    p = os.path.join(out_dir, "skill_occupancy.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # 3. Commanded vx histogram with bucket boundaries.
    if stats["has_command"]:
        vx = stats["vx"]
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.hist(vx, bins=80, color="#66ccee", edgecolor="#225577", linewidth=0.3)
        if buckets:
            edges = sorted({e for b in buckets.values() for e in b.get("lin_vel_x", ())})
            for e in edges:
                ax.axvline(e, color="#aa3377", ls="--", lw=1.0, alpha=0.7)
        ax.set_xlabel("commanded vx (m/s)")
        ax.set_ylabel("steps")
        ax.set_title("Commanded forward-velocity distribution")
        ax.grid(True, alpha=0.3)
        p = os.path.join(out_dir, "vx_histogram.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        written.append(p)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _expand_inputs(paths: list[str]) -> list[str]:
    """Expand directories/globs into a flat list of ``.h5`` / ``.npz`` files."""
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.h5")))
            out += sorted(glob.glob(os.path.join(p, "*.npz")))
        elif any(ch in p for ch in "*?[") and not os.path.exists(p):
            out += sorted(glob.glob(p))
        else:
            out.append(p)
    if not out:
        raise SystemExit(f"No .h5/.npz inputs found in: {paths}")
    return out


def _load_buckets(path: str | None) -> dict[str, dict]:
    """Load a buckets JSON (``{skill: {lin_vel_x: [lo, hi], ...}}``) or default."""
    if path is None:
        return DEFAULT_BUCKETS
    with open(path) as f:
        raw = json.load(f)
    # Allow the file to nest under a "velocity_buckets" key.
    return raw.get("velocity_buckets", raw)


def main() -> None:
    """Parse args, load + analyze the inputs, print the report, optionally plot."""
    ap = argparse.ArgumentParser(
        description="Command-distribution statistics for motion clips / recorded FM datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+",
                    help="Clip .h5 / recorded .npz files, directories, or globs (aggregated).")
    ap.add_argument("--buckets-json", default=None,
                    help="Velocity buckets for skill inference on recorded data "
                         "(default: G1 walk/run). Ignored for clips (they carry skill_id).")
    ap.add_argument("--reference-mix", default=None,
                    help="JSON with an expected skill sampling mix (e.g. clips/skill_mix.json) "
                         "to show next to the measured time occupancy.")
    ap.add_argument("--lin-thresh", type=float, default=1e-4,
                    help="Per-step abs-delta (m/s) on vx/vy above which the command is 'changing'.")
    ap.add_argument("--ang-thresh", type=float, default=1e-4,
                    help="Per-step abs-delta (rad/s) on wz above which the command is 'changing'.")
    ap.add_argument("--per-file", action="store_true",
                    help="Also print a separate report for each input file.")
    ap.add_argument("--plot", action="store_true", help="Write diagnostic PNGs.")
    ap.add_argument("--out-dir", default="command_stats_out",
                    help="Directory for --plot PNGs.")
    args = ap.parse_args()

    buckets = _load_buckets(args.buckets_json)
    reference_mix = None
    if args.reference_mix:
        with open(args.reference_mix) as f:
            raw = json.load(f)
        reference_mix = raw.get("skill_probs", raw)

    files = _expand_inputs(args.inputs)
    loaded: list[MotionData] = []
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".h5":
                md = load_clip_file(path)
            elif ext == ".npz":
                md = load_recorded_file(path, buckets)
            else:
                print(f"[SKIP] {path}: unknown extension {ext!r}")
                continue
        except (ValueError, OSError) as exc:
            print(f"[SKIP] {path}: {exc}")
            continue
        loaded.append(md)
        if args.per_file:
            print(format_report(analyze(md, args.lin_thresh, args.ang_thresh), reference_mix))
            print()

    if not loaded:
        raise SystemExit("No inputs loaded successfully.")

    merged = merge_motion_data(loaded)
    stats = analyze(merged, args.lin_thresh, args.ang_thresh)
    if args.per_file and len(loaded) > 1:
        print("### AGGREGATE ###")
    print(format_report(stats, reference_mix))

    if args.plot:
        written = make_plots(stats, args.out_dir, buckets, reference_mix)
        print("\nplots written:")
        for p in written:
            print(f"  {p}")


if __name__ == "__main__":
    main()
