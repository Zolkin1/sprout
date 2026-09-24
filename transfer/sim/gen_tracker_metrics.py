# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tracking metrics + plots for a generator+tracker MuJoCo eval run.

Parses the CSV (and ``sim_meta.yaml``) written by ``gen_tracker_runner.py`` and
produces the four metric families requested for the eval:

1. Commanded-velocity tracking -- achieved base velocity vs the commanded
   ``(vx, vy, wz)`` (the velocity "joystick" the generator is conditioned on).
2. FM-reference tracking error -- base position / orientation / joint-position
   error between the robot and the generator's predicted reference.
3. Survival -- base height over time + time-to-fall.
4. Joint torques vs effort limits -- per-joint applied torque against the
   actuator force limits, with peak utilisation.

Run standalone on a finished log directory::

    python gen_tracker_metrics.py <log_dir>

or import :func:`create_metrics` (the runner calls it after a logged eval).
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402


def _load(log_dir: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load the run CSV (column name -> array) and the sidecar meta."""
    csv_path = os.path.join(log_dir, "sim_log.csv")
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    header, data = rows[0], np.array(rows[1:], dtype=np.float64)
    cols = {name: data[:, i] for i, name in enumerate(header)}
    meta_path = os.path.join(log_dir, "sim_meta.yaml")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
    return cols, meta


def _mark_handoff(ax, meta) -> None:
    """Draw a vertical line where the FM generator took over the reference.

    Reads ``fm_handoff_time`` from the run meta (written by ``gen_tracker_runner``).
    No-op when absent or ``None`` (e.g. clip-only playback, or an older log).
    """
    t_ho = meta.get("fm_handoff_time") if meta else None
    if t_ho is None:
        return
    ax.axvline(float(t_ho), color="#8a2be2", ls="--", lw=1.4, label="FM handoff")


def _plot_velocity_tracking(cols, out_dir, meta) -> dict:
    """Commanded vs achieved base velocity (vx, vy, wz); returns RMS errors."""
    t = cols["time"]
    pairs = [("vx", "cmd_vx", "vx_b"), ("vy", "cmd_vy", "vy_b"), ("wz", "cmd_wz", "wz_b")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    rms = {}
    for ax, (name, c, a) in zip(axes, pairs):
        ax.plot(t, cols[c], label="commanded", lw=2)
        ax.plot(t, cols[a], label="achieved", lw=1.5, alpha=0.85)
        rms[name] = float(np.sqrt(np.mean((cols[c] - cols[a]) ** 2)))
        ax.set_ylabel(f"{name}  (rms={rms[name]:.3f})")
        ax.grid(alpha=0.3)
        _mark_handoff(ax, meta)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Commanded vs achieved base velocity")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "velocity_tracking.png"), dpi=PLOT_DPI)
    plt.close(fig)
    return rms


def _plot_reference_tracking(cols, out_dir, meta) -> dict:
    """Base pose tracking: FM reference vs actual, overlaid (not the error norm).

    Left column = base position (x, y, z); right column = base orientation
    (roll, pitch, yaw). Returns the per-component RMS error. Falls back to the
    legacy error-norm plot for logs recorded before the per-axis columns existed.
    """
    if not all(k in cols for k in ("ref_pos_x", "act_pos_x", "ref_yaw", "act_yaw")):
        return _plot_reference_tracking_legacy(cols, out_dir, meta)
    t = cols["time"]
    # (ylabel, ref col, act col, summary key)
    pos = [("base x [m]", "ref_pos_x", "act_pos_x", "pos_x"),
           ("base y [m]", "ref_pos_y", "act_pos_y", "pos_y"),
           ("base z [m]", "ref_pos_z", "act_pos_z", "pos_z")]
    ori = [("roll [rad]", "ref_roll", "act_roll", "roll"),
           ("pitch [rad]", "ref_pitch", "act_pitch", "pitch"),
           ("yaw [rad]", "ref_yaw", "act_yaw", "yaw")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    summary = {}
    for r in range(3):
        for col, (label, rk, ak, key) in ((0, pos[r]), (1, ori[r])):
            ax = axes[r][col]
            ax.plot(t, cols[rk], label="FM reference", lw=1.8)
            ax.plot(t, cols[ak], label="actual", lw=1.3, alpha=0.85)
            summary[key] = float(np.sqrt(np.mean((cols[rk] - cols[ak]) ** 2)))
            ax.set_ylabel(f"{label}\n(rms={summary[key]:.3f})")
            ax.grid(alpha=0.3)
            _mark_handoff(ax, meta)
    axes[0][0].legend(loc="upper right", fontsize=8)
    axes[2][0].set_xlabel("time [s]")
    axes[2][1].set_xlabel("time [s]")
    fig.suptitle("Base pose: FM reference vs actual")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_tracking.png"), dpi=PLOT_DPI)
    plt.close(fig)
    return summary


def _plot_reference_tracking_legacy(cols, out_dir, meta) -> dict:
    """Error-norm reference plot for logs without the per-axis ref/act columns."""
    t = cols["time"]
    series = [("base pos err [m]", "pos_err"), ("base ori err [rad]", "ori_err"),
              ("joint pos err [rad]", "joint_pos_err")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    summary = {}
    for ax, (label, key) in zip(axes, series):
        ax.plot(t, cols[key], lw=1.5)
        summary[key] = {"mean": float(np.mean(cols[key])), "max": float(np.max(cols[key]))}
        ax.set_ylabel(f"{label}\n(mean={summary[key]['mean']:.3f})")
        ax.grid(alpha=0.3)
        _mark_handoff(ax, meta)
    if meta and meta.get("fm_handoff_time") is not None:
        axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Tracker-vs-generator reference error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_tracking.png"), dpi=PLOT_DPI)
    plt.close(fig)
    return summary


#: Isaac Lab's ``base_fallen`` thresholds (the tracker task's termination params). Kept here
#: so a MuJoCo run is scored by the SAME predicate the RL side terminates on.
FALL_MIN_HEIGHT = 0.5
FALL_TILT_DEG = 70.0


def _plot_survival(cols, out_dir, meta) -> dict:
    """Base height + fall over time; returns survival summary.

    Survival is recomputed from the trajectory, NOT read from the logged ``fell`` column.
    That column is ``Robot.failed()`` (``qpos[2] < 0.2``) sampled at log time, and it has two
    problems: the run stops the moment it trips, so the failure usually never reaches the
    log at all -- measured 1/100 flagged where the runner itself reported 30/100 fallen --
    and even when it does fire it is height-only at 0.2 m, less than half Isaac Lab's 0.5 m,
    with no orientation check. A robot face-down at 0.21 m and 96 degrees of pitch scored as
    SURVIVED. Anything comparing MuJoCo against Isaac Lab on "success" inherits that.

    The rule here is Isaac's: below :data:`FALL_MIN_HEIGHT`, or tilted past
    :data:`FALL_TILT_DEG` in roll or pitch, at any logged step. Agrees with the runner's own
    live verdict on 98/100 clips, and unlike it, means the same thing the RL side means.
    """
    t = cols["time"]
    # The RUNNER'S verdict wins when it is present, and the fall is DERIVED from state rather
    # than read from the `fell` column -- both branches independently fixed the same bug and
    # each fixed half of it.  `fell` cannot be trusted: it is written before each step's
    # physics and the loop exits on the fall without writing a further row, so a run that fell
    # logs a column of zeros, and inferring from it reported survival for runs the loop had
    # already scored as falls.  Deriving it instead applies ISAAC'S rule (below
    # FALL_MIN_HEIGHT, or past FALL_TILT_DEG in roll or pitch, at any logged step), which is
    # what makes the two simulators score a rollout identically.
    survived_meta = meta.get("survived") if isinstance(meta, dict) else None
    tilt = np.radians(FALL_TILT_DEG)
    fell = cols["base_z"] < FALL_MIN_HEIGHT
    for axis in ("act_roll", "act_pitch"):
        if axis in cols:
            fell = fell | (np.abs(cols[axis]) > tilt)
    fell_idx = int(np.argmax(fell)) if fell.any() else -1
    time_to_fall = float(t[fell_idx]) if fell_idx >= 0 else None
    if survived_meta is False and time_to_fall is None:
        time_to_fall = meta.get("time_to_fall_s")
        if time_to_fall is not None:
            time_to_fall = float(time_to_fall)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, cols["base_z"], lw=1.5, label="base height")
    ax.axhline(FALL_MIN_HEIGHT, color="0.6", ls=":", lw=1.0,
               label=f"fall height {FALL_MIN_HEIGHT} m")
    if time_to_fall is not None:
        ax.axvline(time_to_fall, color="r", ls="--", label=f"fall @ {time_to_fall:.2f}s")
    _mark_handoff(ax, meta)
    ax.set_xlabel("time [s]"); ax.set_ylabel("base z [m]"); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Survival / base height")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "survival.png"), dpi=PLOT_DPI)
    plt.close(fig)
    # survived_meta is the loop's own answer and is authoritative when present. The DERIVED
    # height/tilt rule above is the fallback for a run recorded before the verdict was stamped,
    # and `verdict_source` says which of the two answered. Note what is deliberately NOT used
    # here: the logged `fell` column. That flag is written before the loop's fall check and the
    # loop returns on the next step, so the row where the fall happens is never written and
    # every fall would read as a survival. `time_to_fall_s` is back-filled from the runner
    # above for the same reason -- it is a timestamp, and the column has none on a run that fell.
    survived = (time_to_fall is None) if survived_meta is None else bool(survived_meta)
    return {"survived": survived, "time_to_fall_s": time_to_fall,
            "final_time_s": float(t[-1]),
            "verdict_source": "runner" if survived_meta is not None else "derived",
            "failure_reason": (meta.get("failure_reason", "") if isinstance(meta, dict) else "")}


def _plot_torques(cols, meta, out_dir) -> dict:
    """Per-joint torque vs effort limits; returns peak utilisation."""
    joints = meta.get("joint_names_isaac")
    limits = meta.get("effort_limits")
    tau_keys = [k for k in cols if k.startswith("tau_")]
    if not joints or not tau_keys:
        return {}
    t = cols["time"]
    n = len(joints)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    util = {}
    for i, jn in enumerate(joints):
        ax = axes[i]
        tau = cols.get(f"tau_{jn}")
        if tau is None:
            continue
        ax.plot(t, tau, lw=1.0)
        lim = limits[i] if limits and i < len(limits) else None
        if lim:
            ax.axhline(lim, color="r", ls="--", lw=0.8)
            ax.axhline(-lim, color="r", ls="--", lw=0.8)
            util[jn] = float(np.max(np.abs(tau)) / lim) if lim > 0 else None
        _mark_handoff(ax, meta)
        ax.set_title(jn, fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Joint torque vs effort limit")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(out_dir, "torques.png"), dpi=PLOT_DPI)
    plt.close(fig)
    return {"peak_utilization": util,
            "max_utilization": (max(util.values()) if util else None)}


#: Raster density for every figure here. 120 is legible at a glance and unreadable zoomed in,
#: which is the wrong trade for a 29-subplot grid where the question is usually about ONE joint.
#: Overridable per run (`--dpi`, or `--plot_dpi` on the runner) because the cost is file size:
#: joint_tracking.png is ~12x4 in, so 200 dpi is ~2400x3400 px.
PLOT_DPI: int = 200


def _plot_joint_tracking(cols, meta, out_dir) -> dict:
    """Per-joint FM-reference vs actual joint position; returns per-joint RMS."""
    joints = meta.get("joint_names_isaac")
    if not joints or f"jpref_{joints[0]}" not in cols:
        return {}
    t = cols["time"]
    n = len(joints)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.2 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    rms = {}
    for i, jn in enumerate(joints):
        ax = axes[i]
        r, a = cols.get(f"jpref_{jn}"), cols.get(f"jpact_{jn}")
        if r is None or a is None:
            continue
        ax.plot(t, r, lw=1.3, label="FM ref")
        ax.plot(t, a, lw=1.0, alpha=0.85, label="actual")
        # The PD TARGET -- the policy's action after scaling and the default-pose offset, i.e.
        # what the servo was actually asked for. `ref` and `actual` alone cannot separate
        # "the joint could not get there" from "the policy never asked it to": with the target
        # drawn, a target that tracks the reference while the actual lags is the joint failing
        # to deliver, and a target that ignores the reference is the policy. The column has
        # been in the CSV since it was written; only this plot was not reading it.
        c = cols.get(f"ctrl_{jn}")
        if c is not None:
            ax.plot(t, c, lw=0.9, ls="--", alpha=0.8, label="PD target")
        rms[jn] = float(np.sqrt(np.mean((r - a) ** 2)))
        _mark_handoff(ax, meta)
        ax.set_title(f"{jn}  (rms={rms[jn]:.3f})", fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=6)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-joint position: FM reference vs actual [rad]")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(out_dir, "joint_tracking.png"), dpi=PLOT_DPI)
    plt.close(fig)
    return {"rms": rms, "max_rms": (max(rms.values()) if rms else None)}


def _plot_ee_tracking(cols, meta, out_dir) -> dict:
    """Per-EE FM-reference vs actual position (base frame), x/y/z; returns per-EE RMS."""
    ee = meta.get("ee_frame_names")
    if not ee or f"eeref_{ee[0]}_x" not in cols:
        return {}
    t = cols["time"]
    fig, axes = plt.subplots(len(ee), 3, figsize=(12, 2.4 * len(ee)), sharex=True, squeeze=False)
    summary = {}
    for i, name in enumerate(ee):
        errs = []
        for j, ax_name in enumerate(("x", "y", "z")):
            ax = axes[i][j]
            r, a = cols.get(f"eeref_{name}_{ax_name}"), cols.get(f"eeact_{name}_{ax_name}")
            ax.plot(t, r, lw=1.3, label="FM ref")
            ax.plot(t, a, lw=1.0, alpha=0.85, label="actual")
            errs.append(float(np.sqrt(np.mean((r - a) ** 2))))
            _mark_handoff(ax, meta)
            ax.set_title(f"{name}  {ax_name}", fontsize=7)
            ax.grid(alpha=0.3)
        summary[name] = float(np.linalg.norm(errs))
    axes[0][0].legend(loc="upper right", fontsize=6)
    axes[-1][0].set_xlabel("time [s]")
    fig.suptitle("Per-EE position (base frame): FM reference vs actual [m]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ee_tracking.png"), dpi=PLOT_DPI)
    plt.close(fig)
    return {"rms_norm": summary, "max_rms_norm": (max(summary.values()) if summary else None)}


def create_metrics(log_dir: str, dpi: int | None = None) -> dict:
    """Generate all metric plots + a summary JSON for a finished run.

    Args:
        log_dir: Directory containing ``sim_log.csv`` (and ``sim_meta.yaml``).

    Returns:
        The summary dict (also written to ``metrics_summary.json``).
    """
    global PLOT_DPI
    if dpi:
        PLOT_DPI = int(dpi)
    cols, meta = _load(log_dir)
    out_dir = os.path.join(log_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "command_velocity_rms": _plot_velocity_tracking(cols, out_dir, meta),
        "reference_tracking": _plot_reference_tracking(cols, out_dir, meta),
        "joint_tracking": _plot_joint_tracking(cols, meta, out_dir),
        "ee_tracking": _plot_ee_tracking(cols, meta, out_dir),
        "survival": _plot_survival(cols, out_dir, meta),
        "torques": _plot_torques(cols, meta, out_dir),
    }
    with open(os.path.join(log_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote plots to {out_dir} and metrics_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(f"usage: python gen_tracker_metrics.py <log_dir> [dpi]   (default {PLOT_DPI})")
    create_metrics(sys.argv[1], dpi=int(sys.argv[2]) if len(sys.argv) == 3 else None)
