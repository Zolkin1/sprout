#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Standalone, torch-only diagnostic bench for the flow-matching (FM) generator.

Fixtures a trained FM checkpoint and feeds it *controlled* histories + velocity
commands, then plots the predicted future trajectories. Runs WITHOUT IsaacSim
(imports only the vendored, isaaclab-free ``transfer.fm`` inference stack), so it
is fast and usable locally.

Three experiments:

* ``sweep_command`` -- fix a state history, sweep the velocity command, overlay
  the predicted futures. If the predictions barely move as the command changes,
  the model is ignoring the command / overfitting to the history. By default
  only the decoder ``future_cmd`` is varied (the encoder/history token is kept
  byte-identical); ``--match-past-cmd`` additionally overwrites the history
  token's past command columns.
* ``sweep_history`` -- fix the velocity command, sweep several canonical
  histories (standing / walking / running / turning), overlay the futures. Shows
  how sensitive the output is to the history.
* ``rollout`` -- autoregressively feed the FM's own first predicted step back as
  new history for many steps and check stability (NaN / height collapse /
  velocity tracking).

Canonical histories are pulled from a real ``record_data.py`` recording (the
same data the FM was trained on), selected by matching their recorded
``target_vel`` to a desired command -- realistic walking/running histories
cannot be synthesized by hand.

Plots are shown in an interactive matplotlib window with Left/Right arrow-key
paging when a display is available, and saved as PNGs otherwise (``--save DIR``
forces PNG output).

Example::

    python scripts/flow_matching/fm_test_bench.py \
        --fm_checkpoint logs/flow_matching/2026-06-24_19-00-54_fm-basic-actual-noise/best.pt \
        --device cpu --mode all
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch

# Repo root: this file is scripts/flow_matching/fm_test_bench.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Vendored, isaaclab-free inference stack (pure torch). See transfer/fm/__init__.py.
from transfer.fm import (  # noqa: E402
    BatchedHistory,
    FmOnlineContext,
    build_history_token,
    load_fm_context,
    sample_world,
)

# matplotlib is imported lazily: the backend is chosen in main() BEFORE pyplot is
# imported (so an interactive backend can be selected). cm/colors are
# backend-independent and safe to import eagerly.
import matplotlib  # noqa: E402
import matplotlib.cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402


DEFAULT_CKPT = os.path.join(
    ROOT, "logs", "flow_matching", "2026-06-24_19-00-54_fm-basic-actual-noise", "best.pt"
)

# Canonical command targets (vx, vy, wz) used to fish representative history
# windows out of the recording. "standing" is scarce (recorded mean vx ~1.65),
# so the achieved mean command is logged when a window is selected.
CANONICAL_CMDS: dict[str, list[float]] = {
    "standing": [0.0, 0.0, 0.0],
    "walking": [1.0, 0.0, 0.0],
    "running": [3.0, 0.0, 0.0],
    "turning": [0.8, 0.0, 0.6],
    "side_step": [0.0, 0.6, 0.0],
}


# ======================================================================
# 1. small helpers
# ======================================================================
def quat_to_yaw_np(q: np.ndarray) -> np.ndarray:
    """Yaw (rad) from xyzw quaternions ``(..., 4)``."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Express world-frame vectors ``v`` in the body frame of xyzw quaternions ``q``.

    ``q`` is the body->world rotation; this applies its inverse to ``v``
    (``v_body = R(q)^T v_world``). Broadcasts over leading dims; ``q`` is
    ``(..., 4)`` xyzw and ``v`` is ``(..., 3)``.
    """
    u = q[..., :3]
    w = q[..., 3:4]
    dot = np.sum(u * v, axis=-1, keepdims=True)
    cross = np.cross(u, v)
    return (2.0 * w * w - 1.0) * v - 2.0 * w * cross + 2.0 * dot * u


def to_body_frame_traj(tr: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a copy of ``tr`` with all velocities rotated into the base frame.

    Linear/angular velocities (root and every end-effector) are expressed in the
    instantaneous base frame using ``root_quat`` at the same step, so ``vx`` is
    forward speed (directly comparable to the body-frame velocity command).
    Positions and quaternions are left in the world frame.
    """
    q = tr["root_quat"]  # (T, 4)
    out = dict(tr)
    out["root_lin_vel"] = quat_rotate_inverse_np(q, tr["root_lin_vel"])
    out["root_ang_vel"] = quat_rotate_inverse_np(q, tr["root_ang_vel"])
    if "ee_lin_vel" in tr:
        k = tr["ee_lin_vel"].shape[1]
        qe = np.repeat(q[:, None, :], k, axis=1)  # (T, K, 4): base orientation per EE
        out["ee_lin_vel"] = quat_rotate_inverse_np(qe, tr["ee_lin_vel"])
        out["ee_ang_vel"] = quat_rotate_inverse_np(qe, tr["ee_ang_vel"])
    return out


def disp_traj(tr: dict[str, np.ndarray], vel_frame: str) -> dict[str, np.ndarray]:
    """Apply the display velocity frame (``"body"`` or ``"world"``) to a traj dict."""
    return to_body_frame_traj(tr) if vel_frame == "body" else tr


def term_slice(names: list[str], dims: list[int], target: str) -> tuple[int, int]:
    """Return ``(start, end)`` of a named term in a concatenated vector.

    Local reimplementation of ``robot_rl.flow_matching.dataset.resolve_term_slice``
    (imported here would pull in isaaclab transitively).
    """
    offset = 0
    for name, dim in zip(names, dims):
        if name == target:
            return offset, offset + dim
        offset += dim
    raise KeyError(f"term {target!r} not found in {names}")


def parse_vec3(s: str) -> list[float]:
    """Parse a ``"vx,vy,wz"`` string into a 3-list of floats."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected 3 comma-separated values, got {s!r}")
    return parts


@dataclass
class DuckMeta:
    """Minimal layout metadata read from a recording (duck-typed)."""

    state_term_names: list[str]
    state_term_dims: list[int]
    state_frame_names: list[str]
    state_joint_names: list[str]
    num_envs: int
    dt: float


# ======================================================================
# 2. recording loader (zipfile members -- the .npz is DEFLATE-compressed)
# ======================================================================
# Only these arrays are read; the recording also holds obs (1.6GB), ref_state and
# action which the bench never needs.
_NEEDED_ARRAYS = ("state", "target_vel", "episode_id", "step_in_episode", "done")
_NEEDED_META = (
    "state_term_names", "state_term_dims", "state_frame_names",
    "state_joint_names", "num_envs", "dt",
)


def _read_member(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Read one ``.npy`` member of an npz zip into an array."""
    with zf.open(name + ".npy") as f:
        return np.lib.format.read_array(f, allow_pickle=False)


def load_recording_lite(path: str) -> tuple[dict[str, np.ndarray], DuckMeta]:
    """Load only the arrays the bench needs from a ``record_data.py`` ``.npz``.

    Args:
        path: Path to the recording ``.npz``.

    Returns:
        ``(data, meta)`` where ``data`` holds the numpy arrays in
        :data:`_NEEDED_ARRAYS` and ``meta`` is a :class:`DuckMeta`.
    """
    print(f"[bench] loading recording (state + small arrays only): {path}")
    data: dict[str, np.ndarray] = {}
    raw_meta: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(path) as zf:
        present = {n[:-4] for n in zf.namelist() if n.endswith(".npy")}
        for key in _NEEDED_ARRAYS:
            if key not in present:
                raise KeyError(f"recording {path} is missing required array {key!r}")
            data[key] = _read_member(zf, key)
        for key in _NEEDED_META:
            if key in present:
                raw_meta[key] = _read_member(zf, key)

    meta = DuckMeta(
        state_term_names=[str(s) for s in raw_meta["state_term_names"].tolist()],
        state_term_dims=[int(d) for d in raw_meta["state_term_dims"].tolist()],
        state_frame_names=[str(s) for s in raw_meta["state_frame_names"].tolist()],
        state_joint_names=[str(s) for s in raw_meta["state_joint_names"].tolist()],
        num_envs=int(raw_meta["num_envs"]),
        dt=float(raw_meta["dt"]),
    )
    print(f"[bench]   state {data['state'].shape}, num_envs={meta.num_envs}, dt={meta.dt}")
    return data, meta


def resolve_data_path(fm_checkpoint: str, cli_data: str | None) -> str:
    """Resolve the recording path: ``--data`` > run-dir data.yaml > talos mount.

    Args:
        fm_checkpoint: Path to the FM checkpoint (its run dir may hold data.yaml).
        cli_data: Explicit ``--data`` path, or None.

    Returns:
        An existing path to a recording ``.npz``.

    Raises:
        FileNotFoundError: If no candidate exists.
    """
    if cli_data:
        if not os.path.exists(cli_data):
            raise FileNotFoundError(f"--data {cli_data} does not exist")
        return cli_data

    candidates: list[str] = []
    run_dir = os.path.dirname(os.path.abspath(fm_checkpoint))
    data_yaml = os.path.join(run_dir, "params", "data.yaml")
    if os.path.exists(data_yaml):
        with open(data_yaml) as f:
            for line in f:
                if line.strip().startswith("data:"):
                    d = line.split(":", 1)[1].strip()
                    candidates.append(d)
                    # Map a server-absolute logs path onto the local talos mount.
                    marker = "/logs/"
                    if marker in d:
                        suffix = d.split(marker, 1)[1]
                        candidates.append(
                            os.path.join(ROOT, "talos.caltech.edu_remote_mount", suffix)
                        )
                    break
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Could not resolve a recording. Pass --data explicitly. Tried: "
        + (", ".join(candidates) if candidates else "(no data.yaml next to the checkpoint)")
    )


# ======================================================================
# 3. state slicing
# ======================================================================
class StateLayout:
    """Per-term ``(start, end)`` slices into the recording's flat state vector."""

    def __init__(self, meta: DuckMeta):
        """Resolve slices for root/joints + each end-effector frame block."""
        self.names = meta.state_term_names
        self.dims = meta.state_term_dims
        self.frame_names = meta.state_frame_names
        self.s = {n: term_slice(self.names, self.dims, n) for n in self.names}

    def _col(self, arr: np.ndarray, term: str) -> np.ndarray:
        a, b = self.s[term]
        return arr[..., a:b]

    def parse(self, state_rows: np.ndarray) -> dict[str, np.ndarray]:
        """Parse ``(T, state_dim)`` rows into a common trajectory dict (numpy)."""
        tr = {
            "root_pos": self._col(state_rows, "root_pos_w"),
            "root_quat": self._col(state_rows, "root_quat_w"),
            "root_lin_vel": self._col(state_rows, "root_link_lin_vel_w"),
            "root_ang_vel": self._col(state_rows, "root_link_ang_vel_w"),
            "joint_pos": self._col(state_rows, "joint_pos"),
            "joint_vel": self._col(state_rows, "joint_vel"),
        }
        if self.frame_names:
            tr["ee_pos"] = np.stack([self._col(state_rows, f"{f}:pos_w") for f in self.frame_names], axis=1)
            tr["ee_quat"] = np.stack([self._col(state_rows, f"{f}:quat_w") for f in self.frame_names], axis=1)
            tr["ee_lin_vel"] = np.stack([self._col(state_rows, f"{f}:lin_vel_w") for f in self.frame_names], axis=1)
            tr["ee_ang_vel"] = np.stack([self._col(state_rows, f"{f}:ang_vel_w") for f in self.frame_names], axis=1)
        return tr


def pred_to_traj(pred: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert a ``sample_world`` output (batch 1) into the common traj dict."""
    def f(k: str) -> np.ndarray:
        return pred[k][0].detach().cpu().numpy()

    tr = {
        "root_pos": f("root_pos_w"), "root_quat": f("root_quat_w"),
        "root_lin_vel": f("root_link_lin_vel_w"), "root_ang_vel": f("root_link_ang_vel_w"),
        "joint_pos": f("joint_pos"), "joint_vel": f("joint_vel"),
    }
    if "frame_pos_w" in pred:
        tr["ee_pos"] = f("frame_pos_w")
        tr["ee_quat"] = f("frame_quat_w")
        tr["ee_lin_vel"] = f("frame_lin_vel_w")
        tr["ee_ang_vel"] = f("frame_ang_vel_w")
    return tr


# ======================================================================
# 4. canonical history extraction
# ======================================================================
@dataclass
class Window:
    """A fixed history window pulled from the recording (batch dim = 1)."""

    label: str
    # raw world-frame history inputs for build_history_token, each (1, H, .)
    inputs: dict[str, torch.Tensor]
    past_cmd: torch.Tensor          # (1, H, cmd_dim)
    hist_traj: dict[str, np.ndarray]   # parsed history for plotting (H, .)
    actual_fut_traj: dict[str, np.ndarray]  # recorded next-F steps (F, .)
    actual_fut_cmd: np.ndarray      # recorded next-F command (F, cmd_dim)
    mean_cmd: np.ndarray            # achieved mean command over the window


def find_window(
    data: dict[str, np.ndarray],
    layout: StateLayout,
    ctx: FmOnlineContext,
    target_cmd: list[float],
    label: str,
    n_envs_scan: int,
    device: torch.device,
    cmd_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    steady_weight: float = 1.0,
) -> Window:
    """Find the recording window that best matches ``target_cmd`` and is steady.

    Scans the first ``n_envs_scan`` env streams (the recording is step-major:
    env ``e`` is ``arr[e::num_envs]``). A window is valid iff its
    ``step_in_episode`` is strictly consecutive and no row before the last is a
    ``done`` (mirrors ``dataset.episode_windows``). Validity and scoring are
    vectorized with prefix sums.

    The score is ``||mean(cmd) - target||`` plus ``steady_weight`` times the
    combined std of the command over the window, so a constant-velocity segment
    is preferred over a transient (e.g. decelerating) one with the same mean.
    Set ``steady_weight=0`` for pure mean-matching.

    Args:
        data: Recording arrays.
        layout: State layout for parsing/slicing.
        ctx: Loaded FM context (for H, F, joint/frame order, device).
        target_cmd: Desired ``(vx, vy, wz)``.
        label: Human label for the returned window.
        n_envs_scan: Number of env streams to scan.
        device: Device for the returned input tensors.
        cmd_weights: Per-component weights for the command-match score.
        steady_weight: Penalty weight on the command's within-window std.

    Returns:
        The best-matching :class:`Window`.
    """
    H, F = ctx.H, ctx.F
    W = H + F
    num_envs = layout_num_envs(data)
    state = data["state"]
    tv = data["target_vel"].astype(np.float64)
    sie = data["step_in_episode"]
    done = data["done"].astype(np.int64)
    tgt = np.asarray(target_cmd, dtype=np.float64)
    wts = np.asarray(cmd_weights, dtype=np.float64)

    best = None  # (score, rows)
    for e in range(min(n_envs_scan, num_envs)):
        idx = np.arange(e, state.shape[0], num_envs)
        n_e = len(idx)
        if n_e < W:
            continue
        st_e = sie[idx]
        dn_e = done[idx]
        tv_e = tv[idx]  # (n_e, 3)

        cons = (np.diff(st_e) == 1).astype(np.int64)  # (n_e-1,)
        cons_cs = np.concatenate([[0], np.cumsum(cons)])         # (n_e,)
        done_cs = np.concatenate([[0], np.cumsum(dn_e)])         # (n_e+1,)
        tv_cs = np.concatenate([np.zeros((1, 3)), np.cumsum(tv_e, axis=0)])  # (n_e+1, 3)
        tv_sq_cs = np.concatenate([np.zeros((1, 3)), np.cumsum(tv_e ** 2, axis=0)])

        starts = np.arange(0, n_e - W + 1)
        cons_ok = (cons_cs[starts + (W - 1)] - cons_cs[starts]) == (W - 1)
        done_ok = (done_cs[starts + (W - 1)] - done_cs[starts]) == 0  # no done before last
        valid = cons_ok & done_ok
        if not np.any(valid):
            continue
        vstarts = starts[valid]
        mean_vel = (tv_cs[vstarts + W] - tv_cs[vstarts]) / W       # (M, 3)
        mean_sq = (tv_sq_cs[vstarts + W] - tv_sq_cs[vstarts]) / W  # (M, 3)
        std_tot = np.sqrt(np.maximum(mean_sq - mean_vel ** 2, 0.0).sum(axis=1))  # (M,)
        match = np.sqrt(((mean_vel - tgt) ** 2 * wts).sum(axis=1))
        score = match + steady_weight * std_tot
        j = int(np.argmin(score))
        if best is None or score[j] < best[0]:
            best = (float(score[j]), idx[vstarts[j] : vstarts[j] + W].copy())

    if best is None:
        raise RuntimeError(
            f"no valid {W}-step window found for {label!r} in {n_envs_scan} env streams; "
            f"increase --n-envs-scan"
        )

    rows = best[1]
    ws = state[rows]                # (W, state_dim)
    cmd_w = data["target_vel"][rows]  # (W, cmd_dim)
    hist_rows, fut_rows = ws[:H], ws[H:]

    def t3(term: str, rows_np: np.ndarray) -> torch.Tensor:
        a, b = layout.s[term]
        return torch.from_numpy(np.ascontiguousarray(rows_np[:, a:b])).float().unsqueeze(0).to(device)

    inputs = {
        "root_pos": t3("root_pos_w", hist_rows),
        "root_quat": t3("root_quat_w", hist_rows),
        "root_lin_vel": t3("root_link_lin_vel_w", hist_rows),
        "root_ang_vel": t3("root_link_ang_vel_w", hist_rows),
        "joint_pos": t3("joint_pos", hist_rows),
        "joint_vel": t3("joint_vel", hist_rows),
    }
    past_cmd = torch.from_numpy(np.ascontiguousarray(cmd_w[:H])).float().unsqueeze(0).to(device)
    mean_cmd = cmd_w.mean(axis=0)
    cmd_std = cmd_w.std(axis=0)
    print(f"[bench]   window {label!r}: target={target_cmd} achieved_mean={np.round(mean_cmd, 3).tolist()} "
          f"cmd_std={np.round(cmd_std, 3).tolist()} (score={best[0]:.3f})")
    return Window(
        label=label, inputs=inputs, past_cmd=past_cmd,
        hist_traj=layout.parse(hist_rows), actual_fut_traj=layout.parse(fut_rows),
        actual_fut_cmd=cmd_w[H:], mean_cmd=mean_cmd,
    )


def layout_num_envs(data: dict[str, np.ndarray]) -> int:
    """Number of envs (step-major interleave stride) -- carried in the recording."""
    return int(data["_num_envs"]) if "_num_envs" in data else int(data["num_envs"])


# ======================================================================
# 5. inference + the three modes
# ======================================================================
def predict(
    ctx: FmOnlineContext,
    win: Window,
    future_cmd: torch.Tensor,
    seed: int,
    match_past_cmd: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Run the FM on one window and return the predicted future as a traj dict.

    Args:
        ctx: Loaded FM context.
        win: History window providing the encoder inputs.
        future_cmd: Decoder command ``(1, F, future_cmd_dim)`` (raw, un-normalized).
        seed: Global RNG seed set right before sampling (fair/repro overlays).
        match_past_cmd: If given ``(1, 1, cmd_dim)``, overwrite the history token's
            past command columns with this value (rebuilds the token).

    Returns:
        Predicted future trajectory dict (F, .).
    """
    cmd = win.past_cmd
    if match_past_cmd is not None:
        cmd = match_past_cmd.expand(1, ctx.H, ctx.cmd_dim).contiguous()
    token = build_history_token(
        ctx, win.inputs["root_pos"], win.inputs["root_quat"],
        win.inputs["root_lin_vel"], win.inputs["root_ang_vel"],
        win.inputs["joint_pos"], win.inputs["joint_vel"], cmd, frames=None, scan=None,
    )
    p_ref = win.inputs["root_pos"][:, ctx.H - 1]
    q_ref = win.inputs["root_quat"][:, ctx.H - 1]
    torch.manual_seed(seed)
    fc = future_cmd if ctx.future_cmd_dim > 0 else None
    pred = sample_world(ctx, token, p_ref, q_ref, future_cmd=fc)
    return pred_to_traj(pred)


@dataclass
class Series:
    """One overlay line set: a label/color plus its future (and optional history)."""

    label: str
    color: tuple
    fut: dict[str, np.ndarray]
    fut_cmd: np.ndarray
    hist: dict[str, np.ndarray] | None = None
    past_cmd: np.ndarray | None = None


@dataclass
class Scenario:
    """A plottable comparison: a shared history (maybe) + a list of overlays."""

    name: str
    dt: float
    H: int
    F: int
    series: list[Series]
    shared_hist: dict[str, np.ndarray] | None = None
    shared_past_cmd: np.ndarray | None = None
    actual_fut: dict[str, np.ndarray] | None = None
    actual_fut_cmd: np.ndarray | None = None
    colorbar: tuple | None = None  # (values, label)
    frame_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    vel_frame: str = "body"


def _full_cmd(base: list[float], axis: str, value: float) -> list[float]:
    """Return ``base`` (vx,vy,wz) with the swept ``axis`` set to ``value``."""
    out = list(base)
    out["vx vy wz".split().index(axis)] = value
    return out


def run_sweep_command(ctx, win, args, device) -> Scenario:
    """Fix a history; sweep the command along one axis; overlay predicted futures."""
    values = [float(v) for v in args.values.split(",")]
    base = parse_vec3(args.base_cmd)
    sweep_past = args.match_past_cmd or ctx.future_cmd_dim == 0
    channel = "future+past" if (args.match_past_cmd and ctx.future_cmd_dim > 0) else (
        "past (no decoder cmd)" if ctx.future_cmd_dim == 0 else "future")
    print(f"[bench] sweep_command on {win.label!r}: axis={args.sweep_axis} values={values} "
          f"channel={channel}")
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = matplotlib.colormaps["viridis"]
    vf = args.vel_frame
    series: list[Series] = []
    for v in values:
        cmd_vec = _full_cmd(base, args.sweep_axis, v)
        fut_cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=device).reshape(1, 1, 3).expand(1, ctx.F, 3).contiguous()
        match = None
        if sweep_past:
            match = torch.tensor(cmd_vec, dtype=torch.float32, device=device).reshape(1, 1, 3)
        tr = predict(ctx, win, fut_cmd, args.seed if not args.vary_noise else args.seed + int(v * 1000), match)
        series.append(Series(
            label=f"{args.sweep_axis}={v:g}", color=cmap(norm(v)), fut=disp_traj(tr, vf),
            fut_cmd=np.tile(cmd_vec, (ctx.F, 1)),
        ))
    return Scenario(
        name=f"sweep_command [{win.label}] axis={args.sweep_axis} ({channel})",
        dt=ctx.dt_attr, H=ctx.H, F=ctx.F, series=series,
        shared_hist=disp_traj(win.hist_traj, vf), shared_past_cmd=win.past_cmd[0].cpu().numpy(),
        actual_fut=disp_traj(win.actual_fut_traj, vf), actual_fut_cmd=win.actual_fut_cmd,
        colorbar=(values, f"command {args.sweep_axis}"),
        frame_names=ctx.target_frame_names, joint_names=ctx.joint_names, vel_frame=vf,
    )


def run_sweep_history(ctx, windows, args, device) -> Scenario:
    """Fix the command; sweep canonical histories; overlay predicted futures."""
    cmd_vec = parse_vec3(args.future_cmd)
    print(f"[bench] sweep_history: future_cmd={cmd_vec} histories={[w.label for w in windows]}")
    fut_cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=device).reshape(1, 1, 3).expand(1, ctx.F, 3).contiguous()
    cmap = matplotlib.colormaps["tab10"]
    vf = args.vel_frame
    series: list[Series] = []
    for i, win in enumerate(windows):
        match = torch.tensor(cmd_vec, dtype=torch.float32, device=device).reshape(1, 1, 3) if ctx.future_cmd_dim == 0 else None
        tr = predict(ctx, win, fut_cmd, args.seed, match)
        series.append(Series(
            label=win.label, color=cmap(i % 10), fut=disp_traj(tr, vf), fut_cmd=np.tile(cmd_vec, (ctx.F, 1)),
            hist=disp_traj(win.hist_traj, vf), past_cmd=win.past_cmd[0].cpu().numpy(),
        ))
    return Scenario(
        name=f"sweep_history future_cmd={cmd_vec}", dt=ctx.dt_attr, H=ctx.H, F=ctx.F,
        series=series, frame_names=ctx.target_frame_names, joint_names=ctx.joint_names, vel_frame=vf,
    )


@dataclass
class RolloutResult:
    """Autoregressive rollout output for plotting."""

    traj: dict[str, np.ndarray]   # realized per-step state (T, .)
    cmd: np.ndarray               # commanded vel per step (T, 3)
    dt: float
    diverged_at: int | None
    frame_names: list[str]
    joint_names: list[str]
    vel_frame: str = "body"


def rollout_cmd_schedule(spec: str, n: int) -> np.ndarray:
    """Build a per-step ``(n, 3)`` command schedule from a spec.

    ``"vx,vy,wz"`` -> constant; ``"ramp:a:b"`` -> vx ramps linearly a->b (vy=wz=0).
    """
    if spec.startswith("ramp:"):
        _, a, b = spec.split(":")
        vx = np.linspace(float(a), float(b), n)
        out = np.zeros((n, 3), dtype=np.float32)
        out[:, 0] = vx
        return out
    vec = parse_vec3(spec)
    return np.tile(np.asarray(vec, dtype=np.float32), (n, 1))


def run_rollout(ctx, win, args, device) -> RolloutResult:
    """Autoregressively feed the FM's first predicted step back as history."""
    H, F = ctx.H, ctx.F
    advance = max(1, min(args.advance, F))
    schedule = rollout_cmd_schedule(args.rollout_cmd, args.rollout_steps + F + advance)
    print(f"[bench] rollout from {win.label!r}: steps={args.rollout_steps} advance={advance} "
          f"cmd={args.rollout_cmd}")

    hist = BatchedHistory(1, H, device)
    for k in range(H):
        hist.append({
            "root_pos": win.inputs["root_pos"][:, k],
            "root_quat": win.inputs["root_quat"][:, k],
            "root_lin_vel": win.inputs["root_lin_vel"][:, k],
            "root_ang_vel": win.inputs["root_ang_vel"][:, k],
            "joint_pos": win.inputs["joint_pos"][:, k],
            "joint_vel": win.inputs["joint_vel"][:, k],
            "cmd": win.past_cmd[:, k],
        })

    rec: dict[str, list] = {k: [] for k in
                            ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
                             "joint_pos", "joint_vel", "ee_pos", "ee_quat", "ee_lin_vel", "ee_ang_vel")}
    cmd_rec: list[np.ndarray] = []
    diverged_at = None
    t = 0
    have_ee = ctx.target_num_extra_frames > 0

    while t < args.rollout_steps and diverged_at is None:
        rp = hist.stack("root_pos"); rq = hist.stack("root_quat")
        rlv = hist.stack("root_lin_vel"); rav = hist.stack("root_ang_vel")
        jp = hist.stack("joint_pos"); jv = hist.stack("joint_vel")
        cmd_hist = hist.stack("cmd")
        token = build_history_token(ctx, rp, rq, rlv, rav, jp, jv, cmd_hist, frames=None, scan=None)
        fut_cmd_np = schedule[t : t + F]
        fut_cmd = torch.from_numpy(np.ascontiguousarray(fut_cmd_np)).float().unsqueeze(0).to(device)
        torch.manual_seed(args.seed + t)
        fc = fut_cmd if ctx.future_cmd_dim > 0 else None
        pred = sample_world(ctx, token, rp[:, -1], rq[:, -1], future_cmd=fc)

        for j in range(advance):
            if t >= args.rollout_steps:
                break
            step_cmd = schedule[t]
            new = {
                "root_pos": pred["root_pos_w"][:, j],
                "root_quat": pred["root_quat_w"][:, j],
                "root_lin_vel": pred["root_link_lin_vel_w"][:, j],
                "root_ang_vel": pred["root_link_ang_vel_w"][:, j],
                "joint_pos": pred["joint_pos"][:, j],
                "joint_vel": pred["joint_vel"][:, j],
                "cmd": torch.from_numpy(np.ascontiguousarray(step_cmd)).float().unsqueeze(0).to(device),
            }
            finite = all(torch.isfinite(v).all().item() for v in new.values())
            z = float(new["root_pos"][0, 2].item())
            if (not finite) or not (0.2 < z < 2.0):
                diverged_at = t
                print(f"[bench]   DIVERGED at step {t} (finite={finite}, z={z:.3f})")
                break
            hist.append(new)
            for key in ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel", "joint_pos", "joint_vel"):
                rec[key].append(new[key][0].detach().cpu().numpy())
            if have_ee:
                rec["ee_pos"].append(pred["frame_pos_w"][0, j].detach().cpu().numpy())
                rec["ee_quat"].append(pred["frame_quat_w"][0, j].detach().cpu().numpy())
                rec["ee_lin_vel"].append(pred["frame_lin_vel_w"][0, j].detach().cpu().numpy())
                rec["ee_ang_vel"].append(pred["frame_ang_vel_w"][0, j].detach().cpu().numpy())
            cmd_rec.append(step_cmd.copy())
            t += 1

    traj = {k: np.stack(v) for k, v in rec.items() if v}
    # The feedback above used world-frame predictions; the display copy is rotated
    # into the base frame for body-frame velocity plots (positions stay world).
    if traj:
        traj = disp_traj(traj, args.vel_frame)
    return RolloutResult(
        traj=traj, cmd=np.stack(cmd_rec) if cmd_rec else np.zeros((0, 3)),
        dt=ctx.dt_attr, diverged_at=diverged_at,
        frame_names=ctx.target_frame_names if have_ee else [], joint_names=ctx.joint_names,
        vel_frame=args.vel_frame,
    )


# ======================================================================
# 6. plot builders (each draws into a passed-in, cleared figure)
# ======================================================================
def _time_axes(scn: Scenario) -> tuple[np.ndarray, np.ndarray]:
    """History/future time axes in seconds (history ends at t=0, future starts +dt)."""
    t_hist = (np.arange(scn.H) - (scn.H - 1)) * scn.dt
    t_fut = (np.arange(scn.F) + 1) * scn.dt
    return t_hist, t_fut


def _overlay(ax, getter, scn: Scenario, t_hist, t_fut) -> None:
    """Draw shared history + actual-future reference + every series for one signal."""
    if scn.shared_hist is not None:
        ax.plot(t_hist, getter(scn.shared_hist), color="k", lw=1.8, zorder=6, label="history")
    if scn.actual_fut is not None:
        ax.plot(t_fut, getter(scn.actual_fut), color="0.6", ls="--", lw=1.4, zorder=5, label="actual future")
    for s in scn.series:
        if s.hist is not None:
            ax.plot(t_hist, getter(s.hist), color=s.color, lw=1.4, alpha=0.9)
        ax.plot(t_fut, getter(s.fut), color=s.color, lw=1.6, alpha=0.95, label=s.label)
    ax.axvline(0.0, color="0.3", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)


def _maybe_colorbar(fig, axs, scn: Scenario) -> None:
    """Attach a viridis colorbar for sweep_command, else a compact legend."""
    if scn.colorbar is not None:
        values, label = scn.colorbar
        norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
        sm.set_array([])
        fig.colorbar(sm, ax=list(np.array(axs).ravel()), label=label, fraction=0.02, pad=0.01)
    else:
        handles, labels = axs.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=1)


def draw_root(fig, scn: Scenario) -> None:
    """Root pos (xyz), root lin vel (xyz), root yaw."""
    fig.clear()
    axs = fig.subplots(3, 3)
    t_hist, t_fut = _time_axes(scn)
    labels = ["x", "y", "z"]
    vf = scn.vel_frame
    for i in range(3):
        _overlay(axs[0, i], (lambda a: (lambda tr: tr["root_pos"][:, a]))(i), scn, t_hist, t_fut)
        axs[0, i].set_title(f"root pos {labels[i]} (m, world)", fontsize=9)
        _overlay(axs[1, i], (lambda a: (lambda tr: tr["root_lin_vel"][:, a]))(i), scn, t_hist, t_fut)
        axs[1, i].set_title(f"root lin vel {labels[i]} ({vf}, m/s)", fontsize=9)
    _overlay(axs[2, 0], lambda tr: np.unwrap(quat_to_yaw_np(tr["root_quat"])), scn, t_hist, t_fut)
    axs[2, 0].set_title("root yaw (rad)", fontsize=9)
    axs[2, 1].set_visible(False)
    axs[2, 2].set_visible(False)
    for ax in axs[2, 0:1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(scn.name + "  --  ROOT", fontsize=12)
    _maybe_colorbar(fig, axs, scn)


def draw_ee(fig, scn: Scenario, k: int) -> None:
    """For EE frame ``k``: pos (xyz), lin vel (xyz), yaw."""
    fig.clear()
    axs = fig.subplots(3, 3)
    t_hist, t_fut = _time_axes(scn)
    name = scn.frame_names[k] if k < len(scn.frame_names) else f"ee{k}"
    labels = ["x", "y", "z"]
    vf = scn.vel_frame
    for i in range(3):
        _overlay(axs[0, i], (lambda a: (lambda tr: tr["ee_pos"][:, k, a]))(i), scn, t_hist, t_fut)
        axs[0, i].set_title(f"{name} pos {labels[i]} (m, world)", fontsize=8)
        _overlay(axs[1, i], (lambda a: (lambda tr: tr["ee_lin_vel"][:, k, a]))(i), scn, t_hist, t_fut)
        axs[1, i].set_title(f"{name} lin vel {labels[i]} ({vf}, m/s)", fontsize=8)
    _overlay(axs[2, 0], lambda tr: np.unwrap(quat_to_yaw_np(tr["ee_quat"][:, k])), scn, t_hist, t_fut)
    axs[2, 0].set_title(f"{name} yaw (rad)", fontsize=8)
    axs[2, 1].set_visible(False)
    axs[2, 2].set_visible(False)
    fig.suptitle(scn.name + f"  --  EE {name}", fontsize=11)
    _maybe_colorbar(fig, axs, scn)


def draw_joints(fig, scn: Scenario, vel: bool = False) -> None:
    """Joint positions (or velocities) -- one subplot per joint, 5 columns."""
    fig.clear()
    key = "joint_vel" if vel else "joint_pos"
    nj = len(scn.joint_names)
    n_cols = 5
    n_rows = (nj + n_cols - 1) // n_cols
    axs = fig.subplots(n_rows, n_cols)
    axs = np.array(axs).reshape(n_rows, n_cols)
    t_hist, t_fut = _time_axes(scn)
    for j in range(nj):
        ax = axs.ravel()[j]
        _overlay(ax, (lambda jj: (lambda tr: tr[key][:, jj]))(j), scn, t_hist, t_fut)
        ax.set_title(scn.joint_names[j].replace("_joint", "").replace("_", " "), fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(nj, n_rows * n_cols):
        axs.ravel()[j].set_visible(False)
    unit = "rad/s" if vel else "rad"
    fig.suptitle(scn.name + f"  --  JOINTS ({unit})", fontsize=12)
    _maybe_colorbar(fig, axs, scn)


def draw_cmd(fig, scn: Scenario) -> None:
    """Commanded velocity (vx, vy, wz) over the past + future horizon."""
    fig.clear()
    axs = fig.subplots(1, 3)
    t_hist, t_fut = _time_axes(scn)
    comps = ["vx", "vy", "wz"]
    for i in range(3):
        ax = axs[i]
        if scn.shared_past_cmd is not None:
            ax.plot(t_hist, scn.shared_past_cmd[:, i], color="k", lw=1.8, label="past cmd")
        if scn.actual_fut_cmd is not None:
            ax.plot(t_fut, scn.actual_fut_cmd[:, i], color="0.6", ls="--", lw=1.4, label="actual future cmd")
        for s in scn.series:
            if s.past_cmd is not None:
                ax.plot(t_hist, s.past_cmd[:, i], color=s.color, lw=1.3, alpha=0.8)
            ax.plot(t_fut, s.fut_cmd[:, i], color=s.color, lw=1.8, alpha=0.95, label=s.label)
        ax.axvline(0.0, color="0.3", lw=0.8, ls=":")
        ax.set_title(f"command {comps[i]}", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(scn.name + "  --  COMMAND", fontsize=12)
    _maybe_colorbar(fig, axs, scn)


def draw_rollout(fig, res: RolloutResult) -> None:
    """Stability dashboard for an autoregressive rollout."""
    fig.clear()
    axs = fig.subplots(2, 3)
    T = res.traj["root_pos"].shape[0] if "root_pos" in res.traj else 0
    t = np.arange(T) * res.dt
    if T == 0:
        fig.suptitle("rollout: no steps produced", fontsize=12)
        return
    rp = res.traj["root_pos"]
    rlv = res.traj["root_lin_vel"]
    rav = res.traj["root_ang_vel"]

    axs[0, 0].plot(t, rp[:, 2], color="C0")
    axs[0, 0].axhline(0.2, color="r", ls=":", lw=0.8)
    axs[0, 0].axhline(2.0, color="r", ls=":", lw=0.8)
    axs[0, 0].set_title("root height z (m)", fontsize=9)

    speed = np.linalg.norm(rlv[:, :2], axis=1)
    axs[0, 1].plot(t, rlv[:, 0], label="vx", color="C0")
    axs[0, 1].plot(t, rlv[:, 1], label="vy", color="C1")
    axs[0, 1].plot(t, speed, label="|v|", color="C2", lw=1.0)
    if res.cmd.shape[0] == T:
        axs[0, 1].plot(t, res.cmd[:, 0], ls="--", color="C0", lw=1.0, label="cmd vx")
        axs[0, 1].plot(t, res.cmd[:, 1], ls="--", color="C1", lw=1.0, label="cmd vy")
    axs[0, 1].set_title(f"base lin vel vs cmd ({res.vel_frame}, m/s)", fontsize=9)
    axs[0, 1].legend(fontsize=7)

    axs[0, 2].plot(t, rav[:, 2], label="wz", color="C0")
    if res.cmd.shape[0] == T:
        axs[0, 2].plot(t, res.cmd[:, 2], ls="--", color="C0", lw=1.0, label="cmd wz")
    axs[0, 2].set_title("base yaw rate vs cmd (rad/s)", fontsize=9)
    axs[0, 2].legend(fontsize=7)

    axs[1, 0].plot(rp[:, 0], rp[:, 1], color="C3")
    axs[1, 0].scatter([rp[0, 0]], [rp[0, 1]], c="g", s=20, label="start", zorder=5)
    axs[1, 0].set_title("root xy path (m)", fontsize=9)
    axs[1, 0].set_aspect("equal", adjustable="datalim")
    axs[1, 0].legend(fontsize=7)

    yaw = np.unwrap(quat_to_yaw_np(res.traj["root_quat"]))
    axs[1, 1].plot(t, yaw, color="C4")
    axs[1, 1].set_title("root yaw (rad)", fontsize=9)

    jump = np.concatenate([[0.0], np.linalg.norm(np.diff(rp, axis=0), axis=1) / res.dt])
    axs[1, 2].plot(t, jump, color="C5")
    axs[1, 2].set_title("per-step root speed |dx/dt| (m/s)", fontsize=9)

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        if res.diverged_at is not None:
            ax.axvline(res.diverged_at * res.dt, color="r", lw=1.2)
    for ax in axs[1, :]:
        ax.set_xlabel("time (s)" if ax is not axs[1, 0] else "x (m)")
    status = f"DIVERGED @ step {res.diverged_at}" if res.diverged_at is not None else f"stable for {T} steps"
    fig.suptitle(f"rollout  --  {status}", fontsize=12)


# ======================================================================
# 7. figure browser + PNG save
# ======================================================================
@dataclass
class Page:
    """One pageable figure: a name + a callable that draws into a figure."""

    name: str
    draw: object  # callable(fig) -> None


class FigureBrowser:
    """Page through figures with Left/Right arrows, or save them all as PNGs."""

    def __init__(self, pages: list[Page], plt):
        """Store the pages and a reference to the (backend-bound) pyplot module."""
        self.pages = pages
        self.plt = plt
        self.i = 0

    def _render(self, fig) -> None:
        self.pages[self.i].draw(fig)
        fig.canvas.manager.set_window_title(
            f"[{self.i + 1}/{len(self.pages)}] {self.pages[self.i].name}"
        )
        fig.canvas.draw_idle()

    def _on_key(self, event, fig) -> None:
        if event.key in ("right", "down", " "):
            self.i = (self.i + 1) % len(self.pages)
            self._render(fig)
        elif event.key in ("left", "up"):
            self.i = (self.i - 1) % len(self.pages)
            self._render(fig)
        elif event.key in ("q", "escape"):
            self.plt.close(fig)

    def show(self) -> None:
        """Open one interactive window and bind arrow-key paging."""
        fig = self.plt.figure(figsize=(15, 9))
        fig.canvas.mpl_connect("key_press_event", lambda e: self._on_key(e, fig))
        self._render(fig)
        print(f"[bench] interactive: {len(self.pages)} pages -- Left/Right to page, q to quit")
        self.plt.show()

    def save_all(self, out_dir: str) -> None:
        """Render every page to a PNG under ``out_dir``."""
        os.makedirs(out_dir, exist_ok=True)
        for i, page in enumerate(self.pages):
            fig = self.plt.figure(figsize=(15, 9))
            page.draw(fig)
            safe = page.name.replace(" ", "_").replace("/", "-").replace(":", "-")
            path = os.path.join(out_dir, f"{i:02d}_{safe}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self.plt.close(fig)
            print(f"[bench]   saved {path}")
        print(f"[bench] {len(self.pages)} PNGs in {out_dir}")


def scenario_pages(scn: Scenario) -> list[Page]:
    """Build the per-scenario page list (root, each EE, joints, command)."""
    pages = [Page(f"{scn.name} / root", lambda fig, s=scn: draw_root(fig, s))]
    if scn.frame_names and scn.series and "ee_pos" in scn.series[0].fut:
        for k in range(len(scn.frame_names)):
            pages.append(Page(f"{scn.name} / ee {scn.frame_names[k]}",
                              lambda fig, s=scn, kk=k: draw_ee(fig, s, kk)))
    pages.append(Page(f"{scn.name} / joints", lambda fig, s=scn: draw_joints(fig, s, vel=False)))
    pages.append(Page(f"{scn.name} / command", lambda fig, s=scn: draw_cmd(fig, s)))
    return pages


# ======================================================================
# 8. CLI + main
# ======================================================================
def choose_backend(save: bool):
    """Select a matplotlib backend before pyplot import; return ``(plt, interactive)``."""
    want_interactive = (not save) and bool(os.environ.get("DISPLAY") or sys.platform == "darwin")
    if not want_interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    interactive = want_interactive and plt.get_backend().lower() != "agg"
    return plt, interactive


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fm_checkpoint", default=DEFAULT_CKPT, help="Path to the FM .pt checkpoint.")
    p.add_argument("--data", default=None, help="Recording .npz (default: auto-resolve from the run dir).")
    p.add_argument("--mode", choices=["sweep_command", "sweep_history", "rollout", "all"], default="all")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_sampling_steps", type=int, default=None, help="Override Euler step count.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs-scan", type=int, default=8, help="Env streams to scan for histories.")
    p.add_argument("--steady-weight", type=float, default=1.0,
                   help="Penalty on within-window command std when picking histories: higher prefers "
                        "constant-velocity (steady-state) segments over transients. 0 = pure mean-match.")
    # sweep_command
    p.add_argument("--sweep-axis", choices=["vx", "vy", "wz"], default="vx")
    p.add_argument("--values", default="0,0.5,1.0,1.5,2.0,3.0")
    p.add_argument("--base-cmd", default="0,0,0", help="(vx,vy,wz) base; the swept axis overrides one entry.")
    p.add_argument("--match-past-cmd", action="store_true",
                   help="Also overwrite the history token's past command columns with the swept value.")
    p.add_argument("--sweep-hist", default="walking", help="Canonical history used for sweep_command.")
    # sweep_history
    p.add_argument("--histories", default="standing,walking,running,turning")
    p.add_argument("--future-cmd", default="1.0,0,0", help="Fixed command for sweep_history.")
    # rollout
    p.add_argument("--rollout_steps", type=int, default=200)
    p.add_argument("--advance", type=int, default=1, help="Predicted steps consumed per inference (<=F).")
    p.add_argument("--rollout-cmd", default="1.0,0,0", help="'vx,vy,wz' or 'ramp:a:b'.")
    p.add_argument("--rollout-seed-history", default="walking")
    # plotting / output
    p.add_argument("--vel-frame", choices=["body", "world"], default="body",
                   help="Frame for plotted linear/angular velocities (body = base/heading frame, "
                        "directly comparable to the command; world = raw recorded frame).")
    p.add_argument("--vary-noise", action="store_true", help="Use different sampling noise per overlay.")
    p.add_argument("--save", default=None, help="Force PNG output to this directory.")
    p.add_argument("--out", default=os.path.join(ROOT, "logs", "flow_matching", "test_bench"))
    return p


def main() -> int:
    """Parse args, fixture the FM, run the requested mode(s), and show/save plots."""
    args = build_arg_parser().parse_args()
    plt, interactive = choose_backend(args.save is not None)

    device = torch.device(args.device)
    ctx = load_fm_context(args.fm_checkpoint, device, num_sampling_steps=args.num_sampling_steps)
    # dt is not on the context; the recording carries it -- patched on below.
    print(f"[bench] FM: H={ctx.H} F={ctx.F} nj={ctx.nj} state_dim={ctx.D} "
          f"future_cmd_dim={ctx.future_cmd_dim} history_extra_frames={ctx.history_num_extra_frames} "
          f"target_frames={ctx.target_frame_names} steps={ctx.num_steps}")
    if ctx.history_num_extra_frames != 0:
        raise NotImplementedError(
            "This bench assumes history_num_extra_frames==0 (the deployment design). "
            f"Got {ctx.history_num_extra_frames}; feeding measured EE history is not wired."
        )

    data_path = resolve_data_path(args.fm_checkpoint, args.data)
    data, meta = load_recording_lite(data_path)
    data["_num_envs"] = np.int64(meta.num_envs)
    ctx.dt_attr = meta.dt  # attach dt for plotting (FmOnlineContext has no dt field)

    if meta.state_joint_names != list(ctx.joint_names):
        raise ValueError(
            "joint order mismatch between recording and checkpoint:\n"
            f"  recording: {meta.state_joint_names}\n  checkpoint: {ctx.joint_names}"
        )
    if meta.state_frame_names != list(ctx.target_frame_names):
        raise ValueError(
            "frame order mismatch between recording and checkpoint:\n"
            f"  recording: {meta.state_frame_names}\n  checkpoint: {ctx.target_frame_names}"
        )

    layout = StateLayout(meta)

    def get_window(name: str) -> Window:
        if name not in CANONICAL_CMDS:
            raise ValueError(f"unknown canonical history {name!r}; choose from {list(CANONICAL_CMDS)}")
        return find_window(data, layout, ctx, CANONICAL_CMDS[name], name, args.n_envs_scan, device,
                           steady_weight=args.steady_weight)

    pages: list[Page] = []
    modes = ["sweep_command", "sweep_history", "rollout"] if args.mode == "all" else [args.mode]

    if "sweep_command" in modes:
        win = get_window(args.sweep_hist)
        pages += scenario_pages(run_sweep_command(ctx, win, args, device))
    if "sweep_history" in modes:
        names = [n.strip() for n in args.histories.split(",") if n.strip()]
        windows = [get_window(n) for n in names]
        pages += scenario_pages(run_sweep_history(ctx, windows, args, device))
    if "rollout" in modes:
        win = get_window(args.rollout_seed_history)
        res = run_rollout(ctx, win, args, device)
        pages.append(Page("rollout / stability", lambda fig, r=res: draw_rollout(fig, r)))

    browser = FigureBrowser(pages, plt)
    if interactive:
        browser.show()
    else:
        out_dir = args.save or os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        browser.save_all(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
