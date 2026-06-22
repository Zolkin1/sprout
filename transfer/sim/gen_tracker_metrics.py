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


def _plot_velocity_tracking(cols, out_dir) -> dict:
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
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Commanded vs achieved base velocity")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "velocity_tracking.png"), dpi=120)
    plt.close(fig)
    return rms


def _plot_reference_tracking(cols, out_dir) -> dict:
    """Base pose + joint tracking error vs the FM reference; returns means."""
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
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Tracker-vs-generator reference error")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reference_tracking.png"), dpi=120)
    plt.close(fig)
    return summary


def _plot_survival(cols, out_dir) -> dict:
    """Base height + fall flag over time; returns survival summary."""
    t = cols["time"]
    fell = cols["fell"] > 0.5
    fell_idx = np.argmax(fell) if fell.any() else -1
    time_to_fall = float(t[fell_idx]) if fell_idx >= 0 else None
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, cols["base_z"], lw=1.5, label="base height")
    if time_to_fall is not None:
        ax.axvline(time_to_fall, color="r", ls="--", label=f"fall @ {time_to_fall:.2f}s")
    ax.set_xlabel("time [s]"); ax.set_ylabel("base z [m]"); ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Survival / base height")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "survival.png"), dpi=120)
    plt.close(fig)
    return {"survived": time_to_fall is None, "time_to_fall_s": time_to_fall,
            "final_time_s": float(t[-1])}


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
        ax.set_title(jn, fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Joint torque vs effort limit")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "torques.png"), dpi=120)
    plt.close(fig)
    return {"peak_utilization": util,
            "max_utilization": (max(util.values()) if util else None)}


def create_metrics(log_dir: str) -> dict:
    """Generate all metric plots + a summary JSON for a finished run.

    Args:
        log_dir: Directory containing ``sim_log.csv`` (and ``sim_meta.yaml``).

    Returns:
        The summary dict (also written to ``metrics_summary.json``).
    """
    cols, meta = _load(log_dir)
    out_dir = os.path.join(log_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "command_velocity_rms": _plot_velocity_tracking(cols, out_dir),
        "reference_tracking": _plot_reference_tracking(cols, out_dir),
        "survival": _plot_survival(cols, out_dir),
        "torques": _plot_torques(cols, meta, out_dir),
    }
    with open(os.path.join(log_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote plots to {out_dir} and metrics_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python gen_tracker_metrics.py <log_dir>")
    create_metrics(sys.argv[1])
