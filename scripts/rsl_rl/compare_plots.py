# Copyright 2026 Zachary Olkin. All rights reserved.

"""Cross-policy comparison plots + stats for the multi-policy eval.

This module is the *aggregation* half of the comparison eval (see
``eval_compare_policies.py``).  It is intentionally pure ``numpy`` +
``matplotlib`` with **no torch / isaaclab imports** so it never touches the
simulator and can be unit-tested without a GPU.

Each worker subprocess rolls out one policy across the requested forward
velocities and writes a pre-aggregated dump (``<name>.npz`` + ``<name>.meta.json``).
This module loads those dumps into :class:`WorkerResult` objects and emits, for
every policy's ``<log_dir>/policy_comparison/`` folder, an identical set of
comparison figures + a ``comparison_stats.csv`` (long format) and a
``comparison_summary.txt`` (human-scannable tables).

Plot conventions mirror ``play_plots.py``: the ``Agg`` backend, ``dpi=300``,
``bbox_inches="tight"``, grouped-bar style from ``plot_moe_weights`` and the
line style from ``plot_moe_weights_vs_velocity`` (default rcParams, NOT
publication style).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    """One policy's pre-aggregated comparison metrics across velocities.

    All numeric arrays are indexed by velocity along axis 0 (length ``V``).
    ``K`` = number of termination terms, ``D`` = number of position outputs,
    ``D2`` = number of velocity outputs, ``F`` = number of tracked frames,
    ``G`` = number of CLF subgroups.
    """

    label: str
    log_dir: str
    velocities: np.ndarray                       # [V]

    clf_v_mean: np.ndarray                        # [V]
    clf_v_std: np.ndarray                         # [V]
    clf_subgroup_mean: np.ndarray                 # [V, G]
    clf_subgroup_std: np.ndarray                  # [V, G]

    vel_track_err_mean: np.ndarray               # [V]
    vel_track_err_std: np.ndarray                 # [V]
    vx_actual_mean: np.ndarray                    # [V]
    vx_actual_std: np.ndarray                     # [V]

    term_frac: np.ndarray                         # [V, K]
    term_counts: np.ndarray                       # [V, K]
    episode_ends: np.ndarray                      # [V]

    pos_err_mse: np.ndarray                       # [V, D]
    pos_err_mse_std: np.ndarray                   # [V, D]
    vel_err_mse: np.ndarray                       # [V, D2]
    vel_err_mse_std: np.ndarray                   # [V, D2]
    frame_track_err_mean: np.ndarray              # [V, F]
    frame_track_err_std: np.ndarray               # [V, F]

    term_names: list[str]
    pos_names: list[str]
    vel_names: list[str]
    clf_subgroup_names: list[str]
    frame_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def had_clf(self) -> bool:
        """Whether the policy's task exposed a CLF Lyapunov value."""
        return bool(self.meta.get("had_clf", self.clf_v_mean.size > 0))

    @property
    def had_subgroups(self) -> bool:
        """Whether CLF subgroup contributions were captured."""
        return bool(self.meta.get("had_subgroups", len(self.clf_subgroup_names) > 0))


def _meta_path_for(npz_path: str) -> str:
    """Return the sidecar ``.meta.json`` path for a dump ``.npz`` path."""
    return os.path.splitext(npz_path)[0] + ".meta.json"


def load_worker_dump(npz_path: str) -> WorkerResult:
    """Load one worker dump (``.npz`` + ``.meta.json``) into a WorkerResult.

    Args:
        npz_path: Path to the worker's ``.npz`` array dump. The metadata sidecar
            is resolved as the same path with a ``.meta.json`` extension.

    Returns:
        The populated :class:`WorkerResult`.
    """
    with open(_meta_path_for(npz_path)) as fh:
        meta = json.load(fh)
    z = np.load(npz_path, allow_pickle=False)

    def _arr(key: str, default_shape: tuple[int, ...]) -> np.ndarray:
        return z[key] if key in z.files else np.empty(default_shape, dtype=np.float64)

    velocities = z["velocities"]
    nv = velocities.shape[0]
    return WorkerResult(
        label=str(meta.get("policy_label", os.path.basename(npz_path))),
        log_dir=str(meta.get("log_dir", "")),
        velocities=velocities,
        clf_v_mean=_arr("clf_v_mean", (nv,)),
        clf_v_std=_arr("clf_v_std", (nv,)),
        clf_subgroup_mean=_arr("clf_subgroup_mean", (nv, 0)),
        clf_subgroup_std=_arr("clf_subgroup_std", (nv, 0)),
        vel_track_err_mean=_arr("vel_track_err_mean", (nv,)),
        vel_track_err_std=_arr("vel_track_err_std", (nv,)),
        vx_actual_mean=_arr("vx_actual_mean", (nv,)),
        vx_actual_std=_arr("vx_actual_std", (nv,)),
        term_frac=_arr("term_frac", (nv, 0)),
        term_counts=_arr("term_counts", (nv, 0)),
        episode_ends=_arr("episode_ends", (nv,)),
        pos_err_mse=_arr("pos_err_mse", (nv, 0)),
        pos_err_mse_std=_arr("pos_err_mse_std", (nv, 0)),
        vel_err_mse=_arr("vel_err_mse", (nv, 0)),
        vel_err_mse_std=_arr("vel_err_mse_std", (nv, 0)),
        frame_track_err_mean=_arr("frame_track_err_mean", (nv, 0)),
        frame_track_err_std=_arr("frame_track_err_std", (nv, 0)),
        term_names=list(meta.get("term_names", [])),
        pos_names=list(meta.get("pos_names", [])),
        vel_names=list(meta.get("vel_names", [])),
        clf_subgroup_names=list(meta.get("clf_subgroup_names", [])),
        frame_names=list(meta.get("frame_names", [])),
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _policy_colors(n: int) -> list:
    """Return ``n`` distinct colors, one per policy (tab10/tab20 cycle)."""
    if n <= 10:
        return list(plt.cm.tab10.colors[:n])
    return [plt.cm.tab20(i % 20) for i in range(n)]


def _vel_label(vx: float) -> str:
    """Compact label for a forward-velocity value, e.g. ``vx=1.50``."""
    return f"vx={float(vx):.2f}"


def _grouped_bar(
    ax,
    categories: list[str],
    series_labels: list[str],
    values: np.ndarray,
    errors: np.ndarray | None,
    colors: list,
    ylabel: str,
    title: str,
) -> None:
    """Draw a grouped bar chart: one cluster per category, one bar per series.

    Args:
        ax: Target matplotlib axis.
        categories: ``C`` cluster labels (x axis).
        series_labels: ``S`` series labels (one bar color each, e.g. policies).
        values: ``[S, C]`` bar heights.
        errors: ``[S, C]`` error-bar half-heights, or ``None``.
        colors: Per-series colors (length ``S``).
        ylabel: Y axis label.
        title: Axis title.
    """
    n_series = len(series_labels)
    n_cat = len(categories)
    x = np.arange(n_cat)
    width = 0.8 / max(n_series, 1)
    for s in range(n_series):
        offset = (s - (n_series - 1) / 2.0) * width
        yerr = errors[s] if errors is not None else None
        ax.bar(
            x + offset, values[s], width, yerr=yerr, capsize=3,
            color=colors[s], edgecolor="black", linewidth=0.5,
            label=series_labels[s],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)


def _save(fig, save_dir: str, name: str) -> None:
    """Save a figure into ``save_dir`` and close it (mirrors play_plots)."""
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _common_velocities(results: list[WorkerResult]) -> np.ndarray:
    """Reference velocity axis (assumes all workers used the same list)."""
    return results[0].velocities


# Position-output group patterns, matching play_plots._print_group_summaries.
_POS_GROUPS = {"Positions": ":pos_", "Orientations": ":ori_", "Joints": "joint:"}


def _group_mean(names: list[str], values: np.ndarray, pattern: str) -> float:
    """Mean of ``values`` over outputs whose name contains ``pattern``."""
    idxs = [i for i, nm in enumerate(names) if pattern in nm]
    if not idxs:
        return float("nan")
    return float(np.nanmean(values[idxs]))


def _frame_pos_mse(result: WorkerResult, vi: int, frame: str) -> float:
    """Mean position-output MSE for one body frame at velocity index ``vi``.

    Averages the ``{frame}:pos_x/y/z`` columns of ``pos_err_mse``.
    """
    idxs = [i for i, nm in enumerate(result.pos_names)
            if nm.startswith(f"{frame}:pos_")]
    if not idxs or result.pos_err_mse.size == 0:
        return float("nan")
    return float(np.nanmean(result.pos_err_mse[vi, idxs]))


def _frame_pos_mse_std(result: WorkerResult, vi: int, frame: str) -> float:
    """Std proxy for one frame's position-output MSE (mean of the per-axis MSE stds)."""
    idxs = [i for i, nm in enumerate(result.pos_names)
            if nm.startswith(f"{frame}:pos_")]
    if not idxs or result.pos_err_mse_std.size == 0:
        return float("nan")
    return float(np.nanmean(result.pos_err_mse_std[vi, idxs]))


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def plot_clf_v_comparison(results: list[WorkerResult], save_dir: str) -> None:
    """Grouped bar of mean CLF Lyapunov V per (velocity, policy)."""
    results = [r for r in results if r.had_clf and r.clf_v_mean.size]
    if not results:
        print("[WARN plot_clf_v_comparison] No CLF data, skipping.")
        return
    vels = _common_velocities(results)
    cats = [_vel_label(v) for v in vels]
    labels = [r.label for r in results]
    values = np.vstack([r.clf_v_mean for r in results])
    errors = np.vstack([r.clf_v_std for r in results])
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(vels) + 3), 5))
    _grouped_bar(ax, cats, labels, values, errors, _policy_colors(len(results)),
                 "Mean Lyapunov V", "CLF Lyapunov V by policy (lower is better)")
    _save(fig, save_dir, "clf_v_comparison.png")


# Curated subgroups: pelvis (highlighted) and the body group ankles live in.
_SUBGROUP_FOCUS = ["pelvis_pos", "pelvis_ori", "other_body_pos", "joint_pos"]


def plot_clf_subgroup_comparison(results: list[WorkerResult], save_dir: str) -> None:
    """Line plots of CLF subgroup V vs velocity, one subplot per subgroup.

    Highlights the pelvis subgroups and ``other_body_pos`` (where the ankle
    position error lives in the CLF decomposition).
    """
    results = [r for r in results if r.had_subgroups and r.clf_subgroup_mean.size]
    if not results:
        print("[WARN plot_clf_subgroup_comparison] No CLF subgroup data, skipping.")
        return
    # Subgroups present in all results, ordered by the focus list then the rest.
    common = set(results[0].clf_subgroup_names)
    for r in results[1:]:
        common &= set(r.clf_subgroup_names)
    if not common:
        print("[WARN plot_clf_subgroup_comparison] No shared subgroups, skipping.")
        return
    ordered = [g for g in _SUBGROUP_FOCUS if g in common]
    ordered += [g for g in sorted(common) if g not in ordered]

    colors = _policy_colors(len(results))
    n = len(ordered)
    ncol = min(2, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4 * nrow), squeeze=False)
    for k, g in enumerate(ordered):
        ax = axes[k // ncol][k % ncol]
        for ri, r in enumerate(results):
            gi = r.clf_subgroup_names.index(g)
            mean = r.clf_subgroup_mean[:, gi]
            std = r.clf_subgroup_std[:, gi]
            ax.plot(r.velocities, mean, marker="o", linewidth=2,
                    color=colors[ri], label=r.label)
            ax.fill_between(r.velocities, mean - std, mean + std, alpha=0.15,
                            color=colors[ri])
        ax.set_title(f"CLF subgroup: {g}")
        ax.set_xlabel("Commanded forward velocity (m/s)")
        ax.set_ylabel("Mean V contribution")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    # Hide any unused axes.
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("CLF V decomposition by body subgroup (pelvis / ankles highlighted)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, save_dir, "clf_subgroup_comparison.png")


def plot_velocity_tracking_comparison(results: list[WorkerResult], save_dir: str) -> None:
    """Grouped bar of forward velocity tracking error per (velocity, policy)."""
    if not results:
        return
    vels = _common_velocities(results)
    cats = [_vel_label(v) for v in vels]
    labels = [r.label for r in results]
    values = np.vstack([r.vel_track_err_mean for r in results])
    errors = np.vstack([r.vel_track_err_std for r in results])
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(vels) + 3), 5))
    _grouped_bar(ax, cats, labels, values, errors, _policy_colors(len(results)),
                 "Mean |v_x cmd - v_x actual| (m/s)",
                 "Forward velocity tracking error (lower is better)")
    _save(fig, save_dir, "velocity_tracking_comparison.png")


def plot_velocity_cmd_vs_actual(results: list[WorkerResult], save_dir: str) -> None:
    """Line plot of actual vs commanded forward velocity, one line per policy."""
    if not results:
        return
    colors = _policy_colors(len(results))
    fig, ax = plt.subplots(figsize=(7, 6))
    # y = x reference over the full commanded range.
    all_v = np.concatenate([r.velocities for r in results])
    lo, hi = float(all_v.min()), float(all_v.max())
    pad = 0.1 * max(hi - lo, 1e-3)
    ref = np.array([lo - pad, hi + pad])
    ax.plot(ref, ref, color="gray", linestyle="--", linewidth=1.5, label="ideal (y=x)")
    for ri, r in enumerate(results):
        ax.errorbar(r.velocities, r.vx_actual_mean, yerr=r.vx_actual_std,
                    marker="o", linewidth=2, capsize=3, color=colors[ri], label=r.label)
    ax.set_xlabel("Commanded forward velocity (m/s)")
    ax.set_ylabel("Actual forward velocity (m/s)")
    ax.set_title("Commanded vs achieved forward velocity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, save_dir, "velocity_cmd_vs_actual.png")


def plot_termination_comparison(results: list[WorkerResult], save_dir: str) -> None:
    """Grouped bar of per-cause termination fraction, one figure per velocity."""
    if not results:
        return
    # Termination terms common to all results (keeps bar clusters aligned).
    common = [t for t in results[0].term_names
              if all(t in r.term_names for r in results)]
    if not common:
        print("[WARN plot_termination_comparison] No shared termination terms, skipping.")
        return
    vels = _common_velocities(results)
    labels = [r.label for r in results]
    colors = _policy_colors(len(results))
    for vi, vx in enumerate(vels):
        values = np.zeros((len(results), len(common)))
        for ri, r in enumerate(results):
            for ci, t in enumerate(common):
                ti = r.term_names.index(t)
                values[ri, ci] = r.term_frac[vi, ti] if r.term_frac.size else np.nan
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(common) + 3), 5))
        _grouped_bar(ax, common, labels, values, None, colors,
                     "Fraction of episode-ends",
                     f"Termination causes at {_vel_label(vx)}")
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        _save(fig, save_dir, f"terminations_comparison_{_vel_label(vx).replace('=', '')}.png")


_HIGHLIGHT_FRAMES = ("pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link")


def plot_frame_tracking_comparison(results: list[WorkerResult], save_dir: str) -> None:
    """Line plots of per-frame position tracking error vs velocity.

    One subplot per tracked frame (pelvis + L/R ankle), one line per policy.
    This is the headline ankle/pelvis position-tracking comparison.
    """
    results = [r for r in results if r.frame_track_err_mean.size and r.frame_names]
    if not results:
        print("[WARN plot_frame_tracking_comparison] No frame-tracking data, skipping.")
        return
    common = [f for f in results[0].frame_names
              if all(f in r.frame_names for r in results)]
    if not common:
        print("[WARN plot_frame_tracking_comparison] No shared frames, skipping.")
        return
    colors = _policy_colors(len(results))
    n = len(common)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
    for k, frame in enumerate(common):
        ax = axes[0][k]
        for ri, r in enumerate(results):
            fi = r.frame_names.index(frame)
            mean = r.frame_track_err_mean[:, fi]
            std = r.frame_track_err_std[:, fi]
            ax.plot(r.velocities, mean, marker="o", linewidth=2,
                    color=colors[ri], label=r.label)
            ax.fill_between(r.velocities, mean - std, mean + std, alpha=0.15,
                            color=colors[ri])
        ax.set_title(frame)
        ax.set_xlabel("Commanded forward velocity (m/s)")
        ax.set_ylabel("Mean ||pos error|| (m)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Per-frame position tracking error (pelvis & ankles)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, save_dir, "frame_tracking_comparison.png")


def plot_position_tracking_groups(results: list[WorkerResult], save_dir: str) -> None:
    """Grouped bar of position-output MSE by group, one subplot per velocity.

    Groups are Positions / Orientations / Joints (matching play_plots), giving a
    broad per-output tracking view to complement the per-frame highlight chart.
    """
    results = [r for r in results if r.pos_err_mse.size and r.pos_names]
    if not results:
        print("[WARN plot_position_tracking_groups] No position-error data, skipping.")
        return
    vels = _common_velocities(results)
    labels = [r.label for r in results]
    colors = _policy_colors(len(results))
    groups = list(_POS_GROUPS.keys())
    nrow = len(vels)
    fig, axes = plt.subplots(nrow, 1, figsize=(8, 3.6 * nrow), squeeze=False)
    for vi, vx in enumerate(vels):
        ax = axes[vi][0]
        values = np.zeros((len(results), len(groups)))
        for ri, r in enumerate(results):
            for gi, g in enumerate(groups):
                values[ri, gi] = _group_mean(r.pos_names, r.pos_err_mse[vi], _POS_GROUPS[g])
        _grouped_bar(ax, groups, labels, values, None, colors,
                     "Mean MSE", f"Position-output MSE by group at {_vel_label(vx)}")
    fig.tight_layout()
    _save(fig, save_dir, "position_tracking_groups.png")


def plot_position_tracking_frames(results: list[WorkerResult], save_dir: str) -> None:
    """Grouped bar of pelvis/ankle position MSE per policy, one subplot per velocity."""
    results = [r for r in results if r.pos_err_mse.size and r.pos_names]
    if not results:
        return
    vels = _common_velocities(results)
    labels = [r.label for r in results]
    colors = _policy_colors(len(results))
    frames = list(_HIGHLIGHT_FRAMES)
    nrow = len(vels)
    fig, axes = plt.subplots(nrow, 1, figsize=(8, 3.6 * nrow), squeeze=False)
    for vi, vx in enumerate(vels):
        ax = axes[vi][0]
        values = np.zeros((len(results), len(frames)))
        for ri, r in enumerate(results):
            for fi, frame in enumerate(frames):
                values[ri, fi] = _frame_pos_mse(r, vi, frame)
        _grouped_bar(ax, frames, labels, values, None, colors,
                     "Mean MSE (m^2)", f"Pelvis & ankle position MSE at {_vel_label(vx)}")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _save(fig, save_dir, "position_tracking_pelvis_ankle.png")


# Registry of comparison plots (name -> function).
PLOT_REGISTRY = {
    "clf_v": plot_clf_v_comparison,
    "clf_subgroup": plot_clf_subgroup_comparison,
    "velocity_tracking": plot_velocity_tracking_comparison,
    "velocity_cmd_vs_actual": plot_velocity_cmd_vs_actual,
    "terminations": plot_termination_comparison,
    "frame_tracking": plot_frame_tracking_comparison,
    "position_groups": plot_position_tracking_groups,
    "position_frames": plot_position_tracking_frames,
}


# ---------------------------------------------------------------------------
# Stats writers
# ---------------------------------------------------------------------------

def _collect_warnings(results: list[WorkerResult]) -> list[str]:
    """Cross-policy sanity warnings for the summary header."""
    warns: list[str] = []
    obs_dims = {r.label: r.meta.get("obs_dim") for r in results}
    distinct = {v for v in obs_dims.values() if v is not None}
    if len(distinct) > 1:
        warns.append(
            f"obs_dim differs across policies {obs_dims} — policies were trained "
            f"on different observation spaces; cross-policy comparison of "
            f"env-state metrics (CLF V, terminations, velocity tracking) is still "
            f"valid, but interpret per-output tracking with care."
        )
    pos_name_sets = {tuple(r.pos_names) for r in results if r.pos_names}
    if len(pos_name_sets) > 1:
        warns.append("pos_names differ across policies — per-output tracking "
                     "comparisons use each policy's own output ordering.")
    traj_paths = {r.label: r.meta.get("traj_path") for r in results}
    if len({p for p in traj_paths.values() if p}) > 1:
        warns.append(
            "policies tracked DIFFERENT reference trajectory libraries "
            f"{traj_paths} — each policy is scored against the references it was "
            "trained on (CLF V / tracking errors are per-policy-relative). To score "
            "all policies on one library instead, pass --shared_traj_path.")
    for r in results:
        if not r.meta.get("velocity_pinned", True):
            warns.append(f"{r.label}: forward velocity could NOT be pinned "
                         f"(no '{r.meta.get('vel_cmd_name')}' command term); "
                         f"commanded velocity is the task default, not the requested vx.")
        if r.meta.get("is_multiskill") and r.meta.get("num_skills", 1) > 1:
            warns.append(f"{r.label}: multiskill task with "
                         f"{r.meta.get('num_skills')} skills — terrain can gate which "
                         f"trajectory variant is eligible per vx; a flat/single-skill "
                         f"task gives the cleanest per-velocity reading.")
    return warns


def write_comparison_stats(results: list[WorkerResult], save_dir: str) -> None:
    """Write ``comparison_summary.txt`` (tables) + ``comparison_stats.csv`` (long).

    Args:
        results: Loaded worker results (one per policy).
        save_dir: Directory to write both files into.
    """
    os.makedirs(save_dir, exist_ok=True)
    vels = _common_velocities(results)
    labels = [r.label for r in results]

    # --- long-format CSV ---------------------------------------------------
    csv_path = os.path.join(save_dir, "comparison_stats.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "velocity", "metric", "value"])
        for r in results:
            for vi, vx in enumerate(r.velocities):
                row = [r.label, f"{float(vx):.4f}"]

                def emit(metric: str, value: float) -> None:
                    w.writerow(row + [metric, f"{float(value):.6f}"])

                if r.clf_v_mean.size:
                    emit("clf_v_mean", r.clf_v_mean[vi])
                    emit("clf_v_std", r.clf_v_std[vi])
                emit("vel_track_err_mean", r.vel_track_err_mean[vi])
                emit("vel_track_err_std", r.vel_track_err_std[vi])
                emit("vx_actual_mean", r.vx_actual_mean[vi])
                emit("vx_actual_std", r.vx_actual_std[vi])
                emit("episode_ends", r.episode_ends[vi])
                for ti, t in enumerate(r.term_names):
                    if r.term_frac.size:
                        emit(f"term_frac:{t}", r.term_frac[vi, ti])
                    if r.term_counts.size:
                        emit(f"term_count:{t}", r.term_counts[vi, ti])
                for gi, g in enumerate(r.clf_subgroup_names):
                    if r.clf_subgroup_mean.size:
                        emit(f"clf_subgroup:{g}", r.clf_subgroup_mean[vi, gi])
                for fi, f in enumerate(r.frame_names):
                    if r.frame_track_err_mean.size:
                        emit(f"frame_track_err:{f}", r.frame_track_err_mean[vi, fi])
                    if r.frame_track_err_std.size:
                        emit(f"frame_track_err_std:{f}", r.frame_track_err_std[vi, fi])
                    if r.pos_err_mse.size:  # per-frame position MSE (+ std) for the m^2 bar
                        emit(f"frame_pos_mse:{f}", _frame_pos_mse(r, vi, f))
                        emit(f"frame_pos_mse_std:{f}", _frame_pos_mse_std(r, vi, f))
                for g, pat in _POS_GROUPS.items():
                    if r.pos_err_mse.size:
                        emit(f"pos_mse_group:{g}", _group_mean(r.pos_names, r.pos_err_mse[vi], pat))

    # --- human summary -----------------------------------------------------
    lines: list[str] = []

    def _log(s: str = "") -> None:
        lines.append(s)

    _log("=" * 72)
    _log("  Policy comparison summary")
    _log("=" * 72)
    m0 = results[0].meta
    _log(f"task: {m0.get('task')}   seed: {m0.get('seed')}   "
         f"num_envs: {m0.get('num_envs')}   steps/vel: {m0.get('steps')}   "
         f"warmup: {m0.get('warmup')}")
    _log(f"velocities (forward vx): {[round(float(v), 3) for v in vels]}")
    _log(f"policies ({len(results)}):")
    for r in results:
        traj = r.meta.get("traj_path") or "(registry default)"
        _log(f"  - {r.label}")
        _log(f"      log_dir: {r.log_dir}")
        _log(f"      ref library: {traj}")
    warns = _collect_warnings(results)
    if warns:
        _log("")
        _log("WARNINGS:")
        for wn in warns:
            _log(f"  ! {wn}")

    def _table(title: str, getter, fmt: str = "{:.4f}") -> None:
        _log("")
        _log(f"{title}  (rows=policy, cols=velocity)")
        header = "  {:<28s}".format("policy") + "".join(
            f"{_vel_label(v):>12s}" for v in vels)
        _log(header)
        _log("  " + "-" * (28 + 12 * len(vels)))
        for r in results:
            cells = "".join(fmt.format(getter(r, vi)).rjust(12) for vi in range(len(vels)))
            _log("  {:<28s}".format(r.label[:28]) + cells)

    if any(r.clf_v_mean.size for r in results):
        _table("Mean CLF Lyapunov V", lambda r, vi: float(r.clf_v_mean[vi]))
    _table("Forward velocity tracking error |cmd-actual| (m/s)",
           lambda r, vi: float(r.vel_track_err_mean[vi]))
    _table("Achieved forward velocity (m/s)",
           lambda r, vi: float(r.vx_actual_mean[vi]))
    for frame in _HIGHLIGHT_FRAMES:
        if any(frame in r.frame_names for r in results):
            def _g(r, vi, _f=frame):
                if _f in r.frame_names and r.frame_track_err_mean.size:
                    return float(r.frame_track_err_mean[vi, r.frame_names.index(_f)])
                return float("nan")
            _table(f"Position tracking error: {frame} (m)", _g)

    # Per-velocity termination breakdown (primary causes).
    common_terms = [t for t in results[0].term_names
                    if all(t in r.term_names for r in results)]
    for vi, vx in enumerate(vels):
        _log("")
        _log(f"Termination fractions at {_vel_label(vx)}  (rows=policy, cols=cause)")
        header = "  {:<28s}".format("policy") + "".join(
            f"{t[:14]:>16s}" for t in common_terms)
        _log(header)
        _log("  " + "-" * (28 + 16 * len(common_terms)))
        for r in results:
            cells = ""
            for t in common_terms:
                val = (r.term_frac[vi, r.term_names.index(t)]
                       if r.term_frac.size else float("nan"))
                cells += "{:.3f}".format(float(val)).rjust(16)
            _log("  {:<28s}".format(r.label[:28]) + cells)

    _log("")
    _log("=" * 72)
    with open(os.path.join(save_dir, "comparison_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Top-level aggregation entry point
# ---------------------------------------------------------------------------

def _select_plot_fns(which: list[str] | str):
    """Resolve a ``which`` selector to a list of plot functions."""
    if which == "all" or which == ["all"]:
        return list(PLOT_REGISTRY.values())
    if isinstance(which, str):
        which = [w.strip() for w in which.split(",")]
    fns = []
    for name in which:
        if name in PLOT_REGISTRY:
            fns.append(PLOT_REGISTRY[name])
        else:
            print(f"[WARN] Unknown comparison plot '{name}', skipping.")
    return fns


def aggregate_and_plot(
    dumps: list[str],
    policy_log_dirs: list[str] | None = None,
    which: list[str] | str = "all",
) -> list[str]:
    """Load worker dumps and write comparison plots + stats into each policy folder.

    Args:
        dumps: Paths to the worker ``.npz`` dumps.
        policy_log_dirs: Log dirs to write a ``policy_comparison/`` folder into.
            Defaults to each loaded result's ``log_dir`` (deduplicated).
        which: Comparison plots to generate (``"all"``, a comma string, or a
            list of names from :data:`PLOT_REGISTRY`).

    Returns:
        The list of ``policy_comparison`` directories written.
    """
    results = [load_worker_dump(d) for d in dumps]
    if not results:
        print("[WARN aggregate_and_plot] No dumps loaded; nothing to do.")
        return []

    if policy_log_dirs is None:
        seen: list[str] = []
        for r in results:
            if r.log_dir and r.log_dir not in seen:
                seen.append(r.log_dir)
        policy_log_dirs = seen

    fns = _select_plot_fns(which)
    written: list[str] = []
    for log_dir in policy_log_dirs:
        save_dir = os.path.join(log_dir, "policy_comparison")
        for fn in fns:
            try:
                fn(results, save_dir)
            except Exception as exc:  # noqa: BLE001 - one bad plot shouldn't abort the rest
                print(f"[WARN] {fn.__name__} failed: {exc}")
        write_comparison_stats(results, save_dir)
        written.append(save_dir)
        print(f"[INFO] Comparison written to: {save_dir}")
    return written
