#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Standalone, torch-only diagnostic bench for the flow-matching (FM) generator.

Fixtures a trained FM checkpoint and feeds it *controlled* histories + velocity
commands, then plots the predicted future trajectories. Runs WITHOUT IsaacSim
(imports only the vendored, isaaclab-free ``transfer.fm`` inference stack), so it
is fast and usable locally.

The model is the PARC-style single transformer with masked inpainting: a window
of ``F`` state tokens whose first ``N`` (= ``num_history_states``) are the recent
measured states held clean as conditioning (anchor = step N-1), with the novel
prediction occupying tokens ``N..F-1``. The velocity command is a per-step
profile over the window, strided into conditioning tokens.

Four experiments:

* ``sweep_command`` -- fix the N conditioning states, sweep the (window-constant)
  velocity command, overlay the predicted futures. If the predictions barely move
  as the command changes, the model is ignoring the command.
* ``sweep_history`` -- fix the command, sweep several canonical conditioning
  states (standing / walking / running / turning), overlay the futures. Shows how
  sensitive the output is to the conditioning.
* ``sweep_phase`` -- fix the command, shift the conditioning across one gait
  period; tests whether the prediction tracks the conditioning's gait phase.
* ``rollout`` -- autoregressively feed the FM's own novel predicted step back as
  new conditioning for many steps and check stability (NaN / height collapse /
  velocity tracking).

Conditioning states are pulled from a real ``record_data.py`` recording (the same
data the FM was trained on), selected by matching their recorded ``target_vel`` to
a desired command -- realistic walking/running gaits cannot be synthesized by hand.

Plots are shown in an interactive matplotlib window with Left/Right arrow-key
paging when a display is available, and saved as PNGs otherwise (``--save DIR``
forces PNG output).

Example::

    python scripts/flow_matching/fm_test_bench.py \
        --fm_checkpoint logs/flow_matching/<run>/best.pt --device cpu --mode all
"""

from __future__ import annotations

import argparse
import copy
import glob
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
    build_cmd_tokens,
    build_cond_states,
    load_fm_context,
    sample_world,
)

# matplotlib is imported lazily: the backend is chosen in main() BEFORE pyplot is
# imported (so an interactive backend can be selected). cm/colors are
# backend-independent and safe to import eagerly.
import matplotlib  # noqa: E402
import matplotlib.cm  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402


_FALLBACK_CKPT = os.path.join(
    ROOT, "logs", "flow_matching", "2026-06-24_19-00-54_fm-basic-actual-noise", "best.pt"
)


def default_fm_checkpoint() -> str:
    """Most-recently-modified ``logs/flow_matching/*/best.pt`` (fallback constant)."""
    cands = glob.glob(os.path.join(ROOT, "logs", "flow_matching", "*", "best.pt"))
    return max(cands, key=os.path.getmtime) if cands else _FALLBACK_CKPT

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
    # raw world-frame conditioning inputs for build_cond_states, each (1, N, .)
    inputs: dict[str, torch.Tensor]
    past_cmd: torch.Tensor          # (1, H, cmd_dim)
    hist_traj: dict[str, np.ndarray]   # parsed history for plotting (H, .)
    actual_fut_traj: dict[str, np.ndarray]  # recorded next-F steps (F, .)
    actual_fut_cmd: np.ndarray      # recorded next-F command (F, cmd_dim)
    mean_cmd: np.ndarray            # achieved mean command over the window


def find_steady_rows(
    data: dict[str, np.ndarray],
    ctx: FmOnlineContext,
    target_cmd: list[float],
    length: int,
    n_envs_scan: int,
    cmd_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    steady_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Find the steadiest contiguous ``length``-step stretch matching ``target_cmd``.

    Scans the first ``n_envs_scan`` env streams (step-major: env ``e`` is
    ``arr[e::num_envs]``). A stretch is valid iff ``step_in_episode`` is strictly
    consecutive and no row before the last is a ``done`` (mirrors
    ``dataset.episode_windows``). The score is ``||mean(cmd) - target||`` plus
    ``steady_weight`` times the combined within-stretch command std, so a
    constant-velocity segment beats a transient with the same mean. Vectorized
    with prefix sums.

    Returns:
        ``(rows, mean_cmd, cmd_std, score)`` -- absolute row indices of the best
        stretch (length ``length``) and its command mean/std/score.
    """
    L = length
    num_envs = layout_num_envs(data)
    n_rows = data["target_vel"].shape[0]
    tv = data["target_vel"].astype(np.float64)
    sie = data["step_in_episode"]
    done = data["done"].astype(np.int64)
    tgt = np.asarray(target_cmd, dtype=np.float64)
    wts = np.asarray(cmd_weights, dtype=np.float64)

    best = None  # (score, rows)
    for e in range(min(n_envs_scan, num_envs)):
        idx = np.arange(e, n_rows, num_envs)
        n_e = len(idx)
        if n_e < L:
            continue
        st_e = sie[idx]
        dn_e = done[idx]
        tv_e = tv[idx]  # (n_e, 3)

        cons = (np.diff(st_e) == 1).astype(np.int64)
        cons_cs = np.concatenate([[0], np.cumsum(cons)])
        done_cs = np.concatenate([[0], np.cumsum(dn_e)])
        tv_cs = np.concatenate([np.zeros((1, 3)), np.cumsum(tv_e, axis=0)])
        tv_sq_cs = np.concatenate([np.zeros((1, 3)), np.cumsum(tv_e ** 2, axis=0)])

        starts = np.arange(0, n_e - L + 1)
        cons_ok = (cons_cs[starts + (L - 1)] - cons_cs[starts]) == (L - 1)
        done_ok = (done_cs[starts + (L - 1)] - done_cs[starts]) == 0  # no done before last
        valid = cons_ok & done_ok
        if not np.any(valid):
            continue
        vstarts = starts[valid]
        mean_vel = (tv_cs[vstarts + L] - tv_cs[vstarts]) / L
        mean_sq = (tv_sq_cs[vstarts + L] - tv_sq_cs[vstarts]) / L
        std_tot = np.sqrt(np.maximum(mean_sq - mean_vel ** 2, 0.0).sum(axis=1))
        match = np.sqrt(((mean_vel - tgt) ** 2 * wts).sum(axis=1))
        score = match + steady_weight * std_tot
        j = int(np.argmin(score))
        if best is None or score[j] < best[0]:
            best = (float(score[j]), idx[vstarts[j] : vstarts[j] + L].copy())

    if best is None:
        raise RuntimeError(
            f"no valid {L}-step stretch found near {target_cmd} in {n_envs_scan} env streams; "
            f"increase --n-envs-scan or lower --gait-max-period"
        )
    rows = best[1]
    cmd_w = data["target_vel"][rows]
    return rows, cmd_w.mean(axis=0), cmd_w.std(axis=0), best[0]


def _frames_from_rows(
    layout: StateLayout, rows_np: np.ndarray, frame_names: list[str], device: torch.device
) -> torch.Tensor | None:
    """Extract ``(1, L, K, 13)`` world ee-frame states ``[pos, quat(xyzw), lin, ang]``."""
    if not frame_names:
        return None
    parts = []
    for f in frame_names:
        pa, pb = layout.s[f"{f}:pos_w"]
        qa, qb = layout.s[f"{f}:quat_w"]
        la, lb = layout.s[f"{f}:lin_vel_w"]
        aa, ab = layout.s[f"{f}:ang_vel_w"]
        parts.append(np.concatenate(
            [rows_np[:, pa:pb], rows_np[:, qa:qb], rows_np[:, la:lb], rows_np[:, aa:ab]], axis=-1
        ))  # (L, 13)
    fr = np.stack(parts, axis=1)  # (L, K, 13)
    return torch.from_numpy(np.ascontiguousarray(fr)).float().unsqueeze(0).to(device)


def build_window_from_rows(
    data: dict[str, np.ndarray],
    layout: StateLayout,
    ctx: FmOnlineContext,
    rows: np.ndarray,
    device: torch.device,
    label: str,
) -> Window:
    """Build a :class:`Window` from ``F`` contiguous absolute row indices.

    The first ``N`` rows are the clean conditioning states (anchored at ``N-1``); the
    recorded *novel* future (and its command) is rows ``N..F-1`` -- what the model's
    novel prediction is compared against.
    """
    N, F = ctx.N, ctx.F
    ws = data["state"][rows]            # (F, state_dim)
    cmd_w = data["target_vel"][rows]    # (F, cmd_dim)
    hist_rows = ws[:N]

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
        "frames": _frames_from_rows(layout, hist_rows, ctx.target_frame_names, device),  # (1,N,K,13) or None
    }
    past_cmd = torch.from_numpy(np.ascontiguousarray(cmd_w[:N])).float().unsqueeze(0).to(device)
    return Window(
        label=label, inputs=inputs, past_cmd=past_cmd,
        hist_traj=layout.parse(hist_rows), actual_fut_traj=layout.parse(ws[N:]),
        actual_fut_cmd=cmd_w[N:], mean_cmd=cmd_w.mean(axis=0),
    )


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
    """Find the steady ``H+F`` window best matching ``target_cmd`` and build it.

    Thin wrapper over :func:`find_steady_rows` + :func:`build_window_from_rows`;
    ``steady_weight=0`` reverts to pure mean-command matching.
    """
    rows, mean_cmd, cmd_std, score = find_steady_rows(
        data, ctx, target_cmd, ctx.F, n_envs_scan, cmd_weights, steady_weight
    )
    print(f"[bench]   window {label!r}: target={target_cmd} achieved_mean={np.round(mean_cmd, 3).tolist()} "
          f"cmd_std={np.round(cmd_std, 3).tolist()} (score={score:.3f})")
    return build_window_from_rows(data, layout, ctx, rows, device, label)


def layout_num_envs(data: dict[str, np.ndarray]) -> int:
    """Number of envs (step-major interleave stride) -- carried in the recording."""
    return int(data["_num_envs"]) if "_num_envs" in data else int(data["num_envs"])


def cmd_context_window(data: dict[str, np.ndarray], rows: np.ndarray, pad: int) -> tuple[np.ndarray, int]:
    """Recorded command around a contiguous env-stream stretch, padded for context.

    ``rows`` are the absolute (step-major) indices of one env's consecutive steps.
    Returns ``(cmd, left_pad)`` where ``cmd`` is the recorded ``target_vel`` over
    ``[stretch_start - pad, stretch_end + pad]`` (truncated at episode resets and
    array bounds) and ``left_pad`` is how many real steps precede the stretch, so the
    stretch occupies ``cmd[left_pad : left_pad + len(rows)]``.

    Args:
        data: Recording arrays.
        rows: Absolute indices of the stretch (contiguous in one env stream).
        pad: Desired context length (steps) on each side.

    Returns:
        ``(cmd, left_pad)``.
    """
    num_envs = layout_num_envs(data)
    n_total = data["target_vel"].shape[0]
    e = int(rows[0]) % num_envs
    p0 = (int(rows[0]) - e) // num_envs          # stretch-start position in the env stream
    L = len(rows)
    lo = max(0, p0 - pad)
    hi = min((n_total - 1 - e) // num_envs + 1, p0 + L + pad)
    abs_idx = e + np.arange(lo, hi) * num_envs   # candidate window, time-ordered
    sie = data["step_in_episode"][abs_idx]
    dn = data["done"].astype(np.int64)[abs_idx]
    cons = (np.diff(sie) == 1) & (dn[:-1] == 0)  # cons[i]: i -> i+1 stays in one episode
    sp = p0 - lo                                 # stretch start within the candidate window
    ep = sp + L - 1                              # stretch end (inclusive)
    left, right = sp, ep
    while left - 1 >= 0 and cons[left - 1]:
        left -= 1
    while right + 1 < len(abs_idx) and cons[right]:
        right += 1
    sel = abs_idx[left : right + 1]
    return data["target_vel"][sel], sp - left


# ======================================================================
# 5. inference + the three modes
# ======================================================================
def predict(
    ctx: FmOnlineContext,
    win: Window,
    future_cmd: torch.Tensor,
    seed: int,
    include_history: bool = False,
) -> dict[str, np.ndarray]:
    """Run the FM on one window and return the predicted future as a traj dict.

    Args:
        ctx: Loaded FM context.
        win: Conditioning window providing the N clean states (incl. ee frames).
        future_cmd: Command profile over the window ``(1, F, cmd_dim)`` (raw).
        seed: Global RNG seed set right before sampling (fair/repro overlays).
        include_history: If True, return the full window (frames ``0..F-1``): the
            first ``N`` are the conditioning echo (clamped, should equal the input),
            the rest are novel. If False (default), return only the novel frames.

    Returns:
        Trajectory dict -- length ``F`` if ``include_history`` else ``F - N``
        (novel frames ``N..F-1``).
    """
    cond, known = build_cond_states(
        ctx, win.inputs["root_pos"], win.inputs["root_quat"],
        win.inputs["root_lin_vel"], win.inputs["root_ang_vel"],
        win.inputs["joint_pos"], win.inputs["joint_vel"], frames=win.inputs["frames"],
    )
    p_ref = win.inputs["root_pos"][:, ctx.N - 1]
    q_ref = win.inputs["root_quat"][:, ctx.N - 1]
    cmd_tokens = build_cmd_tokens(ctx, future_cmd) if ctx.cmd_dim > 0 else None
    torch.manual_seed(seed)
    pred = sample_world(ctx, cond, known, p_ref, q_ref, cmd_tokens=cmd_tokens)
    tr = pred_to_traj(pred)
    if include_history:
        return tr  # full window: 0..N-1 conditioning echo, N..F-1 novel
    return {k: v[ctx.N:] for k, v in tr.items()}  # novel frames N..F-1


@dataclass
class Series:
    """One overlay line set: a label/color plus its future (and optional history)."""

    label: str
    color: tuple
    fut: dict[str, np.ndarray]
    fut_cmd: np.ndarray
    hist: dict[str, np.ndarray] | None = None
    past_cmd: np.ndarray | None = None
    actual: dict[str, np.ndarray] | None = None  # per-series recorded future (phase mode)


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


def window_cmd_profile(
    win: Window, target_vec: list[float], ctx: FmOnlineContext, dt: float, accel: float
) -> np.ndarray:
    """Per-step command profile ``(F, 3)`` fed over the window.

    With ``accel <= 0`` the target is held constant across the whole window (a step
    change relative to the history). With ``accel > 0`` the profile is in-distribution
    instead: the recorded command on the ``N`` conditioning steps, then a linear ramp
    from the current command toward ``target_vec`` at ``|accel|`` per component
    (m/s^2 for vx/vy, rad/s^2 for wz) over the novel steps -- mirroring how the
    training ``target_vel`` slews (env ``max_acc``, nominally 1.0). With a short
    window the far targets may not be reached, exactly as the model sees them.
    """
    N, F = ctx.N, ctx.F
    target = np.asarray(target_vec, dtype=np.float64)
    if accel <= 0:
        return np.tile(target, (F, 1))
    past = win.past_cmd[0].detach().cpu().numpy().astype(np.float64)  # (N, 3) recorded
    prof = np.empty((F, 3), dtype=np.float64)
    prof[:N] = past
    cur = past[-1].copy()
    step = accel * dt
    for t in range(N, F):
        cur = cur + np.clip(target - cur, -step, step)
        prof[t] = cur
    return prof


def run_sweep_command(ctx, win, args, device) -> Scenario:
    """Fix a history; sweep the command along one axis; overlay predicted futures."""
    values = [float(v) for v in args.values.split(",")]
    base = parse_vec3(args.base_cmd)
    fnov = ctx.F - ctx.N
    accel = args.ramp_accel if args.cmd_ramp else 0.0
    print(f"[bench] sweep_command on {win.label!r}: axis={args.sweep_axis} values={values}"
          f"{f' (ramp @ {accel} /s^2 from history cmd)' if accel > 0 else ''}")
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = matplotlib.colormaps["viridis"]
    vf = args.vel_frame
    series: list[Series] = []
    for v in values:
        cmd_vec = _full_cmd(base, args.sweep_axis, v)
        prof = window_cmd_profile(win, cmd_vec, ctx, ctx.dt_attr, accel)  # (F, 3)
        fut_cmd = torch.from_numpy(prof).float().unsqueeze(0).to(device)  # (1, F, 3)
        tr = predict(ctx, win, fut_cmd, args.seed if not args.vary_noise else args.seed + int(v * 1000))
        series.append(Series(
            label=f"{args.sweep_axis}={v:g}", color=cmap(norm(v)), fut=disp_traj(tr, vf),
            fut_cmd=prof[ctx.N:],  # novel-region command (ramp or constant)
        ))
    return Scenario(
        name=f"sweep_command [{win.label}] axis={args.sweep_axis}",
        dt=ctx.dt_attr, H=ctx.N, F=fnov, series=series,
        shared_hist=disp_traj(win.hist_traj, vf), shared_past_cmd=win.past_cmd[0].cpu().numpy(),
        actual_fut=disp_traj(win.actual_fut_traj, vf), actual_fut_cmd=win.actual_fut_cmd,
        colorbar=(values, f"command {args.sweep_axis}"),
        frame_names=ctx.target_frame_names, joint_names=ctx.joint_names, vel_frame=vf,
    )


def run_sweep_history(ctx, windows, args, device) -> Scenario:
    """Fix the command; sweep canonical histories; overlay predicted futures."""
    cmd_vec = parse_vec3(args.future_cmd)
    fnov = ctx.F - ctx.N
    accel = args.ramp_accel if args.cmd_ramp else 0.0
    print(f"[bench] sweep_history: future_cmd={cmd_vec} histories={[w.label for w in windows]}"
          f"{f' (ramp @ {accel} /s^2 from each history cmd)' if accel > 0 else ''}")
    cmap = matplotlib.colormaps["tab10"]
    vf = args.vel_frame
    series: list[Series] = []
    for i, win in enumerate(windows):
        prof = window_cmd_profile(win, cmd_vec, ctx, ctx.dt_attr, accel)  # (F, 3), ramps from this history's cmd
        fut_cmd = torch.from_numpy(prof).float().unsqueeze(0).to(device)
        tr = predict(ctx, win, fut_cmd, args.seed)
        series.append(Series(
            label=win.label, color=cmap(i % 10), fut=disp_traj(tr, vf), fut_cmd=prof[ctx.N:],
            hist=disp_traj(win.hist_traj, vf), past_cmd=win.past_cmd[0].cpu().numpy(),
        ))
    return Scenario(
        name=f"sweep_history future_cmd={cmd_vec}", dt=ctx.dt_attr, H=ctx.N, F=fnov,
        series=series, frame_names=ctx.target_frame_names, joint_names=ctx.joint_names, vel_frame=vf,
    )


def estimate_gait_period(stretch: dict[str, np.ndarray], dt: float,
                         min_s: float = 0.3, max_s: float = 1.5) -> int | None:
    """Estimate the gait period (in steps) by autocorrelating a foot's height.

    Uses the first end-effector's vertical position (falls back to the first
    joint position) over a steady stretch, picking the first autocorrelation peak
    in the lag range ``[min_s, max_s]`` seconds. Returns ``None`` if undeterminable.
    """
    if "ee_pos" in stretch:
        sig = stretch["ee_pos"][:, 0, 2]
    else:
        sig = stretch["joint_pos"][:, 0]
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    n = len(sig)
    min_lag = max(2, int(round(min_s / dt)))
    max_lag = min(n - 1, int(round(max_s / dt)))
    if max_lag <= min_lag or np.allclose(sig, 0.0):
        return None
    ac = np.correlate(sig, sig, mode="full")[n - 1:]  # autocorr at lags 0..n-1
    return int(min_lag + np.argmax(ac[min_lag : max_lag + 1]))


def run_sweep_phase(ctx, data, layout, args, device) -> Scenario:
    """Fix the command; shift the history across one gait period; overlay futures.

    Pulls one long steady gait stretch, estimates its period, then builds
    ``--n-phases`` history windows whose start is shifted across that period. Each
    gets the SAME command (matched past + future), so the only thing that changes
    is *where in the gait cycle* the history sits. If the model is phase-aware,
    each predicted future continues its own history's phase (and tracks that
    window's recorded continuation, drawn dotted); if not, the futures collapse
    onto one curve regardless of phase.
    """
    N, F = ctx.N, ctx.F
    W = F
    fnov = F - N
    cmd_vec = parse_vec3(args.future_cmd)
    if args.phase_hist not in CANONICAL_CMDS:
        raise ValueError(f"unknown --phase-hist {args.phase_hist!r}; choose from {list(CANONICAL_CMDS)}")
    # A steady stretch long enough to shift the window across one full period.
    length = W + args.gait_max_period
    rows, mean_cmd, cmd_std, score = find_steady_rows(
        data, ctx, CANONICAL_CMDS[args.phase_hist], length, args.n_envs_scan,
        steady_weight=max(args.steady_weight, 1.0),
    )
    stretch = layout.parse(data["state"][rows])  # (length, .)

    period = args.gait_period if args.gait_period > 0 else estimate_gait_period(stretch, ctx.dt_attr)
    if not period or period <= 0:
        period = int(round(0.8 / ctx.dt_attr))
    period = min(period, args.gait_max_period)
    n = max(2, args.n_phases)
    offsets = sorted({int(round(k * period / n)) for k in range(n)})
    print(f"[bench] sweep_phase on {args.phase_hist!r}: cmd={cmd_vec} (matched past+future) "
          f"period={period} steps ({period * ctx.dt_attr:.2f}s) offsets={offsets} "
          f"cmd_std={np.round(cmd_std, 3).tolist()}")

    fut_cmd = torch.tensor(cmd_vec, dtype=torch.float32, device=device).reshape(1, 1, 3).expand(1, F, 3).contiguous()
    cmap = matplotlib.colormaps["viridis"]
    fracs = [o / period for o in offsets]
    norm = mcolors.Normalize(vmin=0.0, vmax=max(fracs) if max(fracs) > 0 else 1.0)
    vf = args.vel_frame
    series: list[Series] = []
    for o, frac in zip(offsets, fracs):
        win = build_window_from_rows(data, layout, ctx, rows[o : o + W], device, f"phase {o}")
        tr = predict(ctx, win, fut_cmd, args.seed)  # same seed -> identical noise across phases
        series.append(Series(
            label=f"phase {o} ({frac * 100:.0f}%)", color=cmap(norm(frac)),
            fut=disp_traj(tr, vf), fut_cmd=np.tile(cmd_vec, (fnov, 1)),
            hist=disp_traj(win.hist_traj, vf), actual=disp_traj(win.actual_fut_traj, vf),
        ))
    return Scenario(
        name=f"sweep_phase [{args.phase_hist}] cmd={cmd_vec} period={period}",
        dt=ctx.dt_attr, H=N, F=fnov, series=series,
        shared_past_cmd=np.tile(cmd_vec, (N, 1)),
        colorbar=(fracs, "gait phase (frac)"),
        frame_names=ctx.target_frame_names, joint_names=ctx.joint_names, vel_frame=vf,
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
    N, F = ctx.N, ctx.F
    advance = max(1, min(args.advance, F - N))
    schedule = rollout_cmd_schedule(args.rollout_cmd, args.rollout_steps + F + advance)
    print(f"[bench] rollout from {win.label!r}: steps={args.rollout_steps} advance={advance} "
          f"cmd={args.rollout_cmd}")

    have_ee = ctx.target_num_extra_frames > 0
    hist = BatchedHistory(1, N, device)
    for k in range(N):
        step = {
            "root_pos": win.inputs["root_pos"][:, k],
            "root_quat": win.inputs["root_quat"][:, k],
            "root_lin_vel": win.inputs["root_lin_vel"][:, k],
            "root_ang_vel": win.inputs["root_ang_vel"][:, k],
            "joint_pos": win.inputs["joint_pos"][:, k],
            "joint_vel": win.inputs["joint_vel"][:, k],
        }
        if have_ee:
            step["frames"] = win.inputs["frames"][:, k]  # (1, K, 13)
        hist.append(step)

    rec: dict[str, list] = {k: [] for k in
                            ("root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
                             "joint_pos", "joint_vel", "ee_pos", "ee_quat", "ee_lin_vel", "ee_ang_vel")}
    cmd_rec: list[np.ndarray] = []
    diverged_at = None
    t = 0

    while t < args.rollout_steps and diverged_at is None:
        rp = hist.stack("root_pos"); rq = hist.stack("root_quat")
        rlv = hist.stack("root_lin_vel"); rav = hist.stack("root_ang_vel")
        jp = hist.stack("joint_pos"); jv = hist.stack("joint_vel")
        frames = hist.stack("frames") if have_ee else None
        cond, known = build_cond_states(ctx, rp, rq, rlv, rav, jp, jv, frames=frames)
        fut_cmd_np = schedule[t : t + F]
        fut_cmd = torch.from_numpy(np.ascontiguousarray(fut_cmd_np)).float().unsqueeze(0).to(device)
        cmd_tokens = build_cmd_tokens(ctx, fut_cmd) if ctx.cmd_dim > 0 else None
        torch.manual_seed(args.seed + t)
        pred = sample_world(ctx, cond, known, rp[:, -1], rq[:, -1], cmd_tokens=cmd_tokens)

        for j in range(advance):
            if t >= args.rollout_steps:
                break
            pj = N + j  # novel frame index (frames 0..N-1 reproduce the conditioning)
            step_cmd = schedule[t]
            new = {
                "root_pos": pred["root_pos_w"][:, pj],
                "root_quat": pred["root_quat_w"][:, pj],
                "root_lin_vel": pred["root_link_lin_vel_w"][:, pj],
                "root_ang_vel": pred["root_link_ang_vel_w"][:, pj],
                "joint_pos": pred["joint_pos"][:, pj],
                "joint_vel": pred["joint_vel"][:, pj],
            }
            if have_ee:
                new["frames"] = torch.cat(
                    [pred["frame_pos_w"][:, pj], pred["frame_quat_w"][:, pj],
                     pred["frame_lin_vel_w"][:, pj], pred["frame_ang_vel_w"][:, pj]], dim=-1
                )  # (1, K, 13)
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
                rec["ee_pos"].append(pred["frame_pos_w"][0, pj].detach().cpu().numpy())
                rec["ee_quat"].append(pred["frame_quat_w"][0, pj].detach().cpu().numpy())
                rec["ee_lin_vel"].append(pred["frame_lin_vel_w"][0, pj].detach().cpu().numpy())
                rec["ee_ang_vel"].append(pred["frame_ang_vel_w"][0, pj].detach().cpu().numpy())
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


class ForwardKinematics:
    """MuJoCo forward kinematics: generated root pose + joints -> EE world positions.

    Used to check whether the FM's predicted end-effector frames are consistent with
    the FK of its predicted joints. Lazily imports ``mujoco`` and loads the G1 model;
    raises if either is unavailable so the caller can degrade gracefully.
    """

    def __init__(self, model_path: str, joint_names: list[str], frame_names: list[str]):
        """Load the model and map FM joint names -> qpos addresses, frames -> body ids."""
        import mujoco  # lazy: keeps the rest of the bench import-light / mujoco-optional

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        if int(self.model.jnt_type[0]) != 0:  # 0 == mjJNT_FREE (floating base at qpos[0:7])
            raise ValueError(f"{model_path}: expected a floating-base (free) joint at index 0.")
        self.jadr = np.array([int(self.model.joint(n).qposadr[0]) for n in joint_names], dtype=np.int64)
        self.jdof = np.array([int(self.model.joint(n).dofadr[0]) for n in joint_names], dtype=np.int64)
        self.bids = [int(self.model.body(f).id) for f in frame_names]
        self.base_bid = int(self.model.jnt_bodyid[0])      # floating-base child body (its world rotation)

    def positions(self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        """FK EE world positions for a ``(T, .)`` batch -> ``(T, K, 3)``."""
        t_n, k = root_pos.shape[0], len(self.bids)
        out = np.empty((t_n, k, 3), dtype=np.float32)
        m, d = self.model, self.data
        for t in range(t_n):
            d.qpos[:] = 0.0
            d.qpos[0:3] = root_pos[t]
            q = root_quat_xyzw[t]
            d.qpos[3:7] = (q[3], q[0], q[1], q[2])  # xyzw -> wxyz
            d.qpos[self.jadr] = joint_pos[t]
            self._mj.mj_kinematics(m, d)
            for j, bid in enumerate(self.bids):
                out[t, j] = d.xpos[bid]
        return out

    def velocities(
        self, root_pos: np.ndarray, root_quat_xyzw: np.ndarray,
        root_lin_vel_w: np.ndarray, root_ang_vel_w: np.ndarray,
        joint_pos: np.ndarray, joint_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """FK EE world-frame link velocities -> ``(lin (T,K,3), ang (T,K,3))``.

        Maps the recorded world link twist onto MuJoCo's free-joint qvel: linear is
        the world frame-origin velocity (direct), angular is body-local (``R^T omega``);
        joint dof velocities map by name. Reads each EE frame-origin velocity via the
        point Jacobian (``v = Jp @ qvel``, ``w = Jr @ qvel``) -- ``mj_objectVelocity``
        returns a COM-based velocity, so the Jacobian is used instead. Validated exact
        against the recording's ``body_link_vel_w``.
        """
        mj, m, d = self._mj, self.model, self.data
        t_n, k, nv = root_pos.shape[0], len(self.bids), self.model.nv
        lin = np.empty((t_n, k, 3), dtype=np.float32)
        ang = np.empty((t_n, k, 3), dtype=np.float32)
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        for t in range(t_n):
            d.qpos[:] = 0.0
            d.qvel[:] = 0.0
            d.qpos[0:3] = root_pos[t]
            q = root_quat_xyzw[t]
            d.qpos[3:7] = (q[3], q[0], q[1], q[2])
            d.qpos[self.jadr] = joint_pos[t]
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)                                      # required by mj_jac
            rot = d.xmat[self.base_bid].reshape(3, 3)              # base world rotation
            d.qvel[0:3] = root_lin_vel_w[t]                        # free-joint linear = world frame-origin vel
            d.qvel[3:6] = rot.T @ root_ang_vel_w[t]               # angular in body-local
            d.qvel[self.jdof] = joint_vel[t]
            for j, bid in enumerate(self.bids):
                mj.mj_jac(m, d, jacp, jacr, d.xpos[bid], bid)      # Jacobian of the EE frame origin
                lin[t, j] = jacp @ d.qvel
                ang[t, j] = jacr @ d.qvel
        return lin, ang


def try_build_fk(args, ctx: FmOnlineContext) -> ForwardKinematics | None:
    """Build the FK helper, or return None (with a warning) if disabled/unavailable."""
    if args.no_fk:
        return None
    try:
        return ForwardKinematics(args.fk_model, ctx.joint_names, ctx.target_frame_names)
    except Exception as ex:  # mujoco missing, model not found, name mismatch, ...
        print(f"[bench] FK disabled ({type(ex).__name__}: {str(ex)[:90]}); skipping ee-vs-fk page")
        return None


@dataclass
class ForecastResult:
    """Open-loop forecasts overlaid on a ground-truth recording."""

    gt: dict[str, np.ndarray]            # ground-truth trajectory (L, .)
    preds: list[tuple[int, dict[str, np.ndarray]]]  # (anchor_step, FULL-window traj (F, .))
    cmd_context: np.ndarray              # recorded command incl. before/after context (M, 3)
    cmd_context_offset: int              # steps of context before the FM region (stretch start)
    N: int                               # conditioning states (first N frames are the echo)
    dt: float
    frame_names: list[str]
    joint_names: list[str]
    vel_frame: str = "body"
    label: str = "forecast"              # source label (e.g. gait, or teacher/tracker for parity)


def forecast_along_rows(ctx, data, layout, rows, args, device, vf, label) -> ForecastResult:
    """Forecast the FM along a given contiguous ``rows`` stretch of a recording.

    At every ``--forecast-stride`` steps, condition on the N real states ending there
    and the *recorded* command over the window, predict the full window (echo + novel),
    and place it on the absolute timeline so it branches off the ground-truth curve.
    Shared by :func:`run_forecast` (mined steady stretch) and :func:`run_parity_forecast`
    (a recording's parity episode).
    """
    N, F = ctx.N, ctx.F
    fnov = F - N
    L = len(rows)
    gt = layout.parse(data["state"][rows])
    cmd_w = data["target_vel"][rows]
    stride = max(1, args.forecast_stride)
    fk = try_build_fk(args, ctx)

    def attach_fk(tr: dict[str, np.ndarray]) -> None:
        """Add FK consistency fields to a RAW (world-frame) traj, before disp_traj."""
        if fk is None or "ee_pos" not in tr:
            return
        tr["fk_ee_pos"] = fk.positions(tr["root_pos"], tr["root_quat"], tr["joint_pos"])
        lin, ang = fk.velocities(tr["root_pos"], tr["root_quat"], tr["root_lin_vel"],
                                 tr["root_ang_vel"], tr["joint_pos"], tr["joint_vel"])
        tr["fk_lin_err"] = np.linalg.norm(tr["ee_lin_vel"] - lin, axis=-1).astype(np.float32)
        tr["fk_ang_err"] = np.linalg.norm(tr["ee_ang_vel"] - ang, axis=-1).astype(np.float32)

    attach_fk(gt)
    preds: list[tuple[int, dict[str, np.ndarray]]] = []
    for a in range(N - 1, L - fnov, stride):
        s0 = a - (N - 1)
        win = build_window_from_rows(data, layout, ctx, rows[s0 : s0 + F], device, f"a{a}")
        prof = np.ascontiguousarray(cmd_w[s0 : s0 + F])
        fut_cmd = torch.from_numpy(prof).float().unsqueeze(0).to(device)
        tr = predict(ctx, win, fut_cmd, args.seed, include_history=True)
        attach_fk(tr)
        preds.append((a, disp_traj(tr, vf)))
    cmd_ctx, left_pad = cmd_context_window(data, rows, args.forecast_pad)
    return ForecastResult(
        gt=disp_traj(gt, vf), preds=preds, cmd_context=np.ascontiguousarray(cmd_ctx),
        cmd_context_offset=int(left_pad), N=ctx.N, dt=ctx.dt_attr,
        frame_names=ctx.target_frame_names, joint_names=ctx.joint_names, vel_frame=vf, label=label,
    )


def run_forecast(ctx, data, layout, args, device) -> ForecastResult:
    """Forecast the FM along a mined steady stretch of the (default) recording."""
    length = max(ctx.F, args.forecast_len)
    rows, mean_cmd, cmd_std, score = find_steady_rows(
        data, ctx, CANONICAL_CMDS[args.forecast_hist], length, args.n_envs_scan,
        steady_weight=args.steady_weight,
    )
    res = forecast_along_rows(ctx, data, layout, rows, args, device, args.vel_frame, args.forecast_hist)
    print(f"[bench] forecast on {args.forecast_hist!r}: L={len(rows)} steps ({len(rows) * ctx.dt_attr:.1f}s) "
          f"stride={max(1, args.forecast_stride)} -> {len(res.preds)} forecasts "
          f"(horizon {ctx.F - ctx.N} steps) cmd_std={np.round(cmd_std, 3).tolist()}")
    return res


def episode_rows(data: dict[str, np.ndarray], n_envs_scan: int | None = None) -> np.ndarray:
    """Absolute row indices of the LONGEST contiguous, reset-free episode in a recording.

    Scans env streams (step-major) and returns the rows of the single longest run of
    consecutive ``step_in_episode`` with no intervening ``done`` -- used as the parity
    episode to forecast along.
    """
    num_envs = layout_num_envs(data)
    n_rows = data["target_vel"].shape[0]
    sie = data["step_in_episode"]
    done = data["done"].astype(np.int64)
    n_scan = num_envs if n_envs_scan is None else min(n_envs_scan, num_envs)
    best_rows, best_len = None, 0
    for e in range(n_scan):
        idx = np.arange(e, n_rows, num_envs)
        st, dn = sie[idx], done[idx]
        boundary = np.ones(len(idx), dtype=bool)          # new segment where step jumps or prev was done
        boundary[1:] = (np.diff(st) != 1) | (dn[:-1] != 0)
        seg_id = np.cumsum(boundary)
        vals, counts = np.unique(seg_id, return_counts=True)
        j = int(np.argmax(counts))
        if counts[j] > best_len:
            best_len = int(counts[j])
            best_rows = idx[seg_id == vals[j]].copy()
    if best_rows is None:
        raise RuntimeError("no contiguous episode found in recording")
    return best_rows


def latest_npz(path: str) -> str:
    """Resolve a path-or-directory to a concrete ``.npz`` (newest in a directory)."""
    if os.path.isdir(path):
        cands = glob.glob(os.path.join(path, "*.npz"))
        if not cands:
            raise SystemExit(f"[bench] no .npz recording in {path}")
        return max(cands, key=os.path.getmtime)
    if not os.path.exists(path):
        raise SystemExit(f"[bench] recording not found: {path}")
    return path


def reanchor_xy(res: ForecastResult) -> ForecastResult:
    """Translate the episode + its forecasts so it starts at world (0,0) in xy (z kept).

    Each parity recording's longest episode comes from a different env (different world
    origin); shifting both to a common start makes the policies' positions/xy paths
    directly comparable. Velocities/yaw/foot-z/EE-error are translation-invariant.
    """
    off = res.gt["root_pos"][0, :2].copy()
    res.gt["root_pos"][:, :2] -= off
    for _, pred in res.preds:
        pred["root_pos"][:, :2] -= off
    return res


def run_parity_forecast(ctx, path: str, label: str, args, device) -> ForecastResult:
    """Load a policy-on-parity recording and forecast along its longest episode."""
    data, meta = load_recording_lite(path)
    data["_num_envs"] = np.int64(meta.num_envs)
    ctx.dt_attr = meta.dt
    if meta.state_joint_names != list(ctx.joint_names) or meta.state_frame_names != list(ctx.target_frame_names):
        raise SystemExit(f"[bench] {label} recording joint/frame names != checkpoint; re-record with the FM's layout.")
    layout = StateLayout(meta)
    rows = episode_rows(data, args.n_envs_scan)
    res = reanchor_xy(forecast_along_rows(ctx, data, layout, rows, args, device, args.vel_frame, label))
    print(f"[bench] parity[{label}]: episode {len(rows)} steps ({len(rows) * ctx.dt_attr:.1f}s) "
          f"-> {len(res.preds)} forecasts (re-anchored to xy origin)")
    return res


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
        if s.actual is not None:
            ax.plot(t_fut, getter(s.actual), color=s.color, lw=1.0, ls=":", alpha=0.6)
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
    axs[0, 1].plot(t, rlv[:, 2], label="vz", color="C3", lw=1.0)
    axs[0, 1].plot(t, speed, label="|vxy|", color="C2", lw=1.0)
    if res.cmd.shape[0] == T:
        axs[0, 1].plot(t, res.cmd[:, 0], ls="--", color="C0", lw=1.0, label="cmd vx")
        axs[0, 1].plot(t, res.cmd[:, 1], ls="--", color="C1", lw=1.0, label="cmd vy")
    axs[0, 1].set_title(f"base lin vel (vx/vy/vz) vs cmd ({res.vel_frame}, m/s)", fontsize=9)
    axs[0, 1].legend(fontsize=7, ncol=2)

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


def _forecast_color(res: ForecastResult):
    """Viridis normalizer over anchor time (s) + colormap for the forecast fans."""
    L = res.gt["root_pos"].shape[0]
    return mcolors.Normalize(vmin=0.0, vmax=max(1e-6, (L - 1) * res.dt)), matplotlib.colormaps["viridis"]


def draw_forecast(fig, res: ForecastResult) -> None:
    """Ground-truth spine (black) with each forecast branching off it.

    Each branch is the full predicted window: the first N frames are the FM's
    history echo (dotted -- should lie on the ground truth) and the rest are the
    novel prediction (solid). Colored by anchor time.
    """
    fig.clear()
    axs = fig.subplots(3, 3)
    L = res.gt["root_pos"].shape[0]
    n = res.N
    fwin = res.preds[0][1]["root_pos"].shape[0] if res.preds else 0  # full window length F
    fnov = max(0, fwin - n)
    dt, vf = res.dt, res.vel_frame
    norm, cmap = _forecast_color(res)
    t_gt = np.arange(L) * dt

    def branch(ax, ys, col) -> None:
        """Plot one full-window signal: echo (dotted) then novel (solid), in absolute time."""
        for a, pred in res.preds:
            tw = (np.arange(fwin) + (a - (n - 1))) * dt   # window starts n-1 steps before anchor
            y = ys(pred)
            ax.plot(tw[:n], y[:n], color=col(a), lw=0.9, ls=":", alpha=0.8, zorder=3)        # FM history echo
            ax.plot(tw[n - 1:], y[n - 1:], color=col(a), lw=1.0, alpha=0.85, zorder=4)        # novel

    def fan(ax, getter) -> None:
        ax.plot(t_gt, getter(res.gt), color="k", lw=1.6, zorder=5)
        branch(ax, getter, lambda a: cmap(norm(a * dt)))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time (s)")

    lab = ["x", "y", "z"]
    for i in range(3):
        fan(axs[0, i], (lambda c: (lambda tr: tr["root_pos"][:, c]))(i))
        axs[0, i].set_title(f"root pos {lab[i]} (m, world)", fontsize=9)
        fan(axs[1, i], (lambda c: (lambda tr: tr["root_lin_vel"][:, c]))(i))
        axs[1, i].set_title(f"root lin vel {lab[i]} ({vf}, m/s)", fontsize=9)

    ax = axs[2, 0]
    ax.plot(res.gt["root_pos"][:, 0], res.gt["root_pos"][:, 1], color="k", lw=1.6, zorder=5)
    for a, pred in res.preds:
        col = cmap(norm(a * dt))
        px, py = pred["root_pos"][:, 0], pred["root_pos"][:, 1]
        ax.plot(px[:n], py[:n], color=col, ls=":", lw=0.9, alpha=0.8, zorder=3)
        ax.plot(px[n - 1:], py[n - 1:], color=col, lw=1.0, alpha=0.85, zorder=4)
    ax.set_title("root xy path (m, world)", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")

    fan(axs[2, 1], lambda tr: np.unwrap(quat_to_yaw_np(tr["root_quat"])))
    axs[2, 1].set_title("root yaw (rad)", fontsize=9)
    if "ee_pos" in res.gt:
        fan(axs[2, 2], lambda tr: tr["ee_pos"][:, 0, 2])
        axs[2, 2].set_title(f"{res.frame_names[0]} pos z (m, world)", fontsize=8)
    else:
        axs[2, 2].set_visible(False)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, ax=list(axs.ravel()), label="forecast anchor time (s)", fraction=0.02, pad=0.01)
    fig.suptitle(f"forecast vs ground truth  ({len(res.preds)} forecasts, horizon {fnov * dt:.2f}s; "
                 f"dotted = FM history echo, solid = novel)  --  ROOT", fontsize=12)


def draw_forecast_path(fig, res: ForecastResult) -> None:
    """Dedicated top-down xy view: ground-truth path with the forecast branches."""
    fig.clear()
    ax = fig.subplots(1, 1)
    dt, n = res.dt, res.N
    norm, cmap = _forecast_color(res)
    ax.plot(res.gt["root_pos"][:, 0], res.gt["root_pos"][:, 1], color="k", lw=2.2, zorder=5, label="ground truth")
    for a, pred in res.preds:
        col = cmap(norm(a * dt))
        px, py = pred["root_pos"][:, 0], pred["root_pos"][:, 1]
        ax.plot(px[:n], py[:n], color=col, ls=":", lw=1.0, alpha=0.8, zorder=3)       # FM history echo
        ax.plot(px[n - 1:], py[n - 1:], color=col, lw=1.3, alpha=0.85, zorder=4)      # novel
    ax.scatter([res.gt["root_pos"][0, 0]], [res.gt["root_pos"][0, 1]], c="g", s=30, zorder=6, label="start")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=9)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="forecast anchor time (s)")
    fig.suptitle("forecast vs ground truth  --  root xy path  (dotted = FM history echo, solid = novel)",
                 fontsize=12)


def draw_forecast_cmd(fig, res: ForecastResult) -> None:
    """Recorded commanded velocity (vx, vy, wz) with before/after context.

    The shaded band + dashed lines mark the FM region (where forecasts condition on
    and predict the recording); the command outside it shows whether that region is
    coming out of / heading into a ramp. Time 0 = FM region start.
    """
    fig.clear()
    ax = fig.subplots(1, 1)
    cmd, off = res.cmd_context, res.cmd_context_offset
    L = res.gt["root_pos"].shape[0]
    M = cmd.shape[0]
    t = (np.arange(M) - off) * res.dt            # stretch (FM region) starts at t=0
    t_start, t_stop = 0.0, (L - 1) * res.dt
    ax.axvspan(t_start, t_stop, color="0.85", alpha=0.5, zorder=0, label="FM region")
    ax.axvline(t_start, color="0.3", ls="--", lw=1.1)
    ax.axvline(t_stop, color="0.3", ls="--", lw=1.1)
    for i, (name, c) in enumerate(zip(["vx", "vy", "wz"], ["C0", "C1", "C2"])):
        ax.plot(t, cmd[:, i], color=c, lw=1.6, label=name)
    ax.set_xlabel("time (s)   (0 = FM region start; shaded = where the FM is used)")
    ax.set_ylabel("command (m/s, rad/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("forecast vs ground truth  --  recorded commanded velocity (+ context)", fontsize=12)


def draw_forecast_fk(fig, res: ForecastResult) -> None:
    """Consistency: generated EE position vs FK of the generated joints, per EE.

    Per end-effector, the Cartesian error ``||generated EE - FK(generated root+joints)||``
    over the forecast windows (colored by anchor time), with the ground-truth baseline
    (black, ~0 since FK matches the recording exactly). Branch values near the baseline
    mean the FM's predicted joints and EE frames agree; values above it mean the
    generated joints don't place the EE where the generated EE frame says.
    """
    fig.clear()
    k = len(res.frame_names)
    ncol = 2
    nrow = (k + ncol - 1) // ncol
    axs = np.array(fig.subplots(nrow, ncol)).reshape(nrow, ncol)
    n = res.N
    fwin = res.preds[0][1]["root_pos"].shape[0] if res.preds else 0
    dt = res.dt
    norm, cmap = _forecast_color(res)
    L = res.gt["root_pos"].shape[0]
    t_gt = np.arange(L) * dt
    gt_err = np.linalg.norm(res.gt["ee_pos"] - res.gt["fk_ee_pos"], axis=-1)  # (L, K)
    novel_errs: list[np.ndarray] = []
    for j in range(k):
        ax = axs.ravel()[j]
        ax.plot(t_gt, gt_err[:, j], color="k", lw=1.4, zorder=5)
        for a, pred in res.preds:
            e = np.linalg.norm(pred["ee_pos"][:, j] - pred["fk_ee_pos"][:, j], axis=-1)  # (F,)
            ax.plot((np.arange(fwin) + (a - (n - 1))) * dt, e, color=cmap(norm(a * dt)), lw=1.0, alpha=0.85)
            novel_errs.append(e[n:])
        ax.set_title(res.frame_names[j], fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("|gen EE - FK| (m)", fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(k, nrow * ncol):
        axs.ravel()[j].set_visible(False)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, ax=list(axs.ravel()), label="forecast anchor time (s)", fraction=0.02, pad=0.01)
    mean_nov = float(np.concatenate(novel_errs).mean()) if novel_errs else 0.0
    fig.suptitle(f"generated EE position vs FK(generated joints)  --  mean novel error {mean_nov * 1000:.1f} mm "
                 f"(black = GT baseline)", fontsize=12)


def draw_forecast_fk_vel(fig, res: ForecastResult) -> None:
    """Consistency of generated EE *velocities* vs FK of the generated joint velocities.

    Per EE, the world-frame error magnitude ``||generated EE vel - FK vel||`` over the
    forecasts (colored by anchor), with the GT baseline (black, ~0): left column linear
    (m/s), right column angular (rad/s). Lines near the baseline mean the generated EE
    velocities are consistent with the differential FK of the generated joint motion.
    """
    fig.clear()
    k = len(res.frame_names)
    axs = np.array(fig.subplots(k, 2)).reshape(k, 2)
    n = res.N
    fwin = res.preds[0][1]["root_pos"].shape[0] if res.preds else 0
    dt = res.dt
    norm, cmap = _forecast_color(res)
    L = res.gt["root_pos"].shape[0]
    t_gt = np.arange(L) * dt
    cols = [("fk_lin_err", "lin vel", "m/s"), ("fk_ang_err", "ang vel", "rad/s")]
    novel: dict[str, list] = {c[0]: [] for c in cols}
    for j in range(k):
        for c, (key, title, unit) in enumerate(cols):
            ax = axs[j, c]
            ax.plot(t_gt, res.gt[key][:, j], color="k", lw=1.4, zorder=5)
            for a, pred in res.preds:
                ax.plot((np.arange(fwin) + (a - (n - 1))) * dt, pred[key][:, j],
                        color=cmap(norm(a * dt)), lw=1.0, alpha=0.85)
                novel[key].append(pred[key][n:, j])
            ax.set_title(f"{res.frame_names[j]} {title}", fontsize=8)
            ax.set_ylabel(f"|gen - FK| ({unit})", fontsize=7)
            ax.grid(True, alpha=0.3)
            if j == k - 1:
                ax.set_xlabel("time (s)")
            else:
                ax.tick_params(labelbottom=False)  # declutter: ticks only on the bottom row
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    fig.colorbar(sm, ax=list(axs.ravel()), label="forecast anchor time (s)", fraction=0.02, pad=0.01)
    ml = float(np.concatenate(novel["fk_lin_err"]).mean()) if novel["fk_lin_err"] else 0.0
    ma = float(np.concatenate(novel["fk_ang_err"]).mean()) if novel["fk_ang_err"] else 0.0
    fig.suptitle(f"generated EE velocity vs FK(generated joints)  --  mean novel lin {ml * 1000:.1f} mm/s, "
                 f"ang {ma:.3f} rad/s (black = GT baseline)", fontsize=12)


# ======================================================================
# 6b. parity comparison plots (overlay multiple policy rollouts)
# ======================================================================
def _parity_track(ax, getter, res: ForecastResult, color, n, dt) -> None:
    """Plot one source on an axis: rollout spine (solid) + its FM forecasts (thin)."""
    L = res.gt["root_pos"].shape[0]
    ax.plot(np.arange(L) * dt, getter(res.gt), color=color, lw=1.8, zorder=5, label=res.label)
    fwin = res.preds[0][1]["root_pos"].shape[0] if res.preds else n
    for a, pred in res.preds:
        tw = (np.arange(fwin) + (a - (n - 1))) * dt
        y = getter(pred)
        ax.plot(tw[n - 1:], y[n - 1:], color=color, lw=0.7, alpha=0.25, zorder=3)  # novel forecast
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("time (s)")


def draw_parity(fig, results: list[ForecastResult]) -> None:
    """Overlay each policy's parity rollout + FM forecasts, colored by source."""
    fig.clear()
    axs = fig.subplots(3, 3)
    n, dt, vf = results[0].N, results[0].dt, results[0].vel_frame
    colors = [f"C{i}" for i in range(len(results))]
    lab = ["x", "y", "z"]
    for i in range(3):
        for r, c in zip(results, colors):
            _parity_track(axs[0, i], (lambda a: (lambda tr: tr["root_pos"][:, a]))(i), r, c, n, dt)
        axs[0, i].set_title(f"root pos {lab[i]} (m, world)", fontsize=9)
        for r, c in zip(results, colors):
            _parity_track(axs[1, i], (lambda a: (lambda tr: tr["root_lin_vel"][:, a]))(i), r, c, n, dt)
        axs[1, i].set_title(f"root lin vel {lab[i]} ({vf}, m/s)", fontsize=9)
    ax = axs[2, 0]
    for r, c in zip(results, colors):
        ax.plot(r.gt["root_pos"][:, 0], r.gt["root_pos"][:, 1], color=c, lw=1.8, zorder=5, label=r.label)
        for a, pred in r.preds:
            ax.plot(pred["root_pos"][n - 1:, 0], pred["root_pos"][n - 1:, 1], color=c, lw=0.7, alpha=0.25, zorder=3)
    ax.set_title("root xy path (m, world)", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    for r, c in zip(results, colors):
        _parity_track(axs[2, 1], lambda tr: np.unwrap(quat_to_yaw_np(tr["root_quat"])), r, c, n, dt)
    axs[2, 1].set_title("root yaw (rad)", fontsize=9)
    if "ee_pos" in results[0].gt:
        for r, c in zip(results, colors):
            _parity_track(axs[2, 2], lambda tr: tr["ee_pos"][:, 0, 2], r, c, n, dt)
        axs[2, 2].set_title(f"{results[0].frame_names[0]} pos z (m, world)", fontsize=8)
    else:
        axs[2, 2].set_visible(False)
    h, lbls = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lbls, loc="upper right", fontsize=10)
    fig.suptitle("parity: " + " vs ".join(r.label for r in results)
                 + "  (solid = rollout, thin = FM forecasts)  --  ROOT", fontsize=12)


def draw_parity_cmd(fig, results: list[ForecastResult]) -> None:
    """Recorded commanded velocity per source over the parity episode (should ~match)."""
    fig.clear()
    axs = fig.subplots(1, 3)
    dt = results[0].dt
    colors = [f"C{i}" for i in range(len(results))]
    comps = ["vx", "vy", "wz"]
    for i in range(3):
        for r, c in zip(results, colors):
            L = r.gt["root_pos"].shape[0]
            off = r.cmd_context_offset
            cmd = r.cmd_context[off:off + L]
            axs[i].plot(np.arange(cmd.shape[0]) * dt, cmd[:, i], color=c, lw=1.6, label=r.label)
        axs[i].set_title(f"command {comps[i]}", fontsize=10)
        axs[i].set_xlabel("time (s)")
        axs[i].grid(True, alpha=0.3)
    h, lbls = axs[0].get_legend_handles_labels()
    fig.legend(h, lbls, loc="upper right", fontsize=9)
    fig.suptitle("parity: recorded commanded velocity (" + " vs ".join(r.label for r in results) + ")", fontsize=12)


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
# 7b. data dump (.npz) for offline analysis
# ======================================================================
def dump_forecast(res: ForecastResult, path: str) -> None:
    """Write the forecast ground truth + per-anchor predictions + command to ``.npz``.

    Keys: ``gt_<field>`` (L, .) ground truth; ``anchors`` (n,) anchor steps and
    ``pred_<field>`` (n, F, .) full-window forecasts (frames 0..N-1 are the FM history
    echo, N..F-1 novel); ``cmd_context`` (M,3), ``cmd_context_offset``, ``N``, ``dt``,
    ``vel_frame``, ``frame_names``, ``joint_names``.
    """
    out: dict[str, np.ndarray] = {f"gt_{k}": v for k, v in res.gt.items()}
    if res.preds:
        out["anchors"] = np.array([a for a, _ in res.preds], dtype=np.int64)
        for k in res.preds[0][1]:
            out[f"pred_{k}"] = np.stack([p[k] for _, p in res.preds], axis=0)
    out["cmd_context"] = res.cmd_context
    out["cmd_context_offset"] = np.int64(res.cmd_context_offset)
    out["N"] = np.int64(res.N)
    out["dt"] = np.float64(res.dt)
    out["vel_frame"] = np.array(res.vel_frame)
    out["frame_names"] = np.array(res.frame_names)
    out["joint_names"] = np.array(res.joint_names)
    np.savez_compressed(path, **out)
    print(f"[bench]   dumped {path}")


def dump_rollout(res: RolloutResult, path: str) -> None:
    """Write the autoregressive rollout's realized trajectory + command to ``.npz``.

    Keys: ``traj_<field>`` (T, .) realized states; ``cmd`` (T,3); ``dt``;
    ``diverged_at`` (-1 if stable); ``vel_frame``, ``frame_names``, ``joint_names``.
    """
    out: dict[str, np.ndarray] = {f"traj_{k}": v for k, v in res.traj.items()}
    out["cmd"] = res.cmd
    out["dt"] = np.float64(res.dt)
    out["diverged_at"] = np.int64(-1 if res.diverged_at is None else res.diverged_at)
    out["vel_frame"] = np.array(res.vel_frame)
    out["frame_names"] = np.array(res.frame_names)
    out["joint_names"] = np.array(res.joint_names)
    np.savez_compressed(path, **out)
    print(f"[bench]   dumped {path}")


# ======================================================================
# 7c. viser visualization (reuses robot_rl.flow_matching.viz, no IsaacSim)
# ======================================================================
def _flat_terrain(center_xy: tuple[float, float], half: float = 15.0, z: float = 0.0):
    """A simple ground-plane mesh (two triangles) centered at ``center_xy``."""
    cx, cy = center_xy
    v = np.array([[cx - half, cy - half, z], [cx + half, cy - half, z],
                  [cx + half, cy + half, z], [cx - half, cy + half, z]], dtype=np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return v, f


def forecast_multi_to_viz(results: list[ForecastResult], ctx: FmOnlineContext, urdf_path: str, args):
    """Build a multi-robot viz recording: one robot per forecast (GT robot + ghosts).

    Each robot is re-anchored to its own lane (start x=0, y spread across robots) so
    several gaits can be watched side-by-side; predictions are translated by the same
    per-robot offset and flattened with a per-robot ``pred_index``.
    """
    from robot_rl.flow_matching.viz import VizRecordingMeta  # lazy (viser/yourdfpy)

    r_n = len(results)
    L = min(res.gt["root_pos"].shape[0] for res in results)  # equal in practice; align defensively
    n, nj = ctx.N, ctx.nj
    fwin = results[0].preds[0][1]["root_pos"].shape[0] if results[0].preds else n
    fnov = max(1, fwin - n)
    spacing = args.viz_lane_spacing

    root_pos = np.zeros((L, r_n, 3), np.float32)
    root_quat = np.zeros((L, r_n, 4), np.float32)
    joint = np.zeros((L, r_n, nj), np.float32)
    vel_cmd = np.zeros((L, r_n, 3), np.float32)
    pidx = np.full((L, r_n), -1, np.int32)
    pred_rp_l: list[np.ndarray] = []
    pred_rq_l: list[np.ndarray] = []
    pred_jp_l: list[np.ndarray] = []
    pred_base = 0
    for r, res in enumerate(results):
        rp = res.gt["root_pos"][:L].astype(np.float32).copy()
        lane_y = (r - (r_n - 1) / 2.0) * spacing
        off = np.array([-rp[0, 0], lane_y - rp[0, 1], 0.0], np.float32)  # start at (0, lane), keep z
        rp += off
        root_pos[:, r] = rp
        root_quat[:, r] = res.gt["root_quat"][:L]
        joint[:, r] = res.gt["joint_pos"][:L]
        o = res.cmd_context_offset
        cmd = res.cmd_context[o:o + L].astype(np.float32)
        if cmd.shape[0] < L:
            cmd = np.concatenate([cmd, np.tile(cmd[-1:], (L - cmd.shape[0], 1))], axis=0)
        vel_cmd[:, r] = cmd[:L]
        anchors = [a for a, _ in res.preds]
        for a, pred in res.preds:
            prp = pred["root_pos"][n:].astype(np.float32).copy()
            prp += off
            pred_rp_l.append(prp)
            pred_rq_l.append(pred["root_quat"][n:].astype(np.float32))
            pred_jp_l.append(pred["joint_pos"][n:].astype(np.float32))
        for p, a in enumerate(anchors):
            if a >= L:
                continue
            nxt = anchors[p + 1] if p + 1 < len(anchors) else L
            pidx[a:min(nxt, L), r] = pred_base + p
        pred_base += len(res.preds)

    pred_rp = np.stack(pred_rp_l) if pred_rp_l else np.zeros((1, fnov, 3), np.float32)
    pred_rq = np.stack(pred_rq_l) if pred_rq_l else np.tile(np.array([0, 0, 0, 1], np.float32), (1, fnov, 1))
    pred_jp = np.stack(pred_jp_l) if pred_jp_l else np.zeros((1, fnov, nj), np.float32)

    allxy = root_pos[..., :2].reshape(-1, 2)
    cx, cy = float(allxy[:, 0].mean()), float(allxy[:, 1].mean())
    half = max(5.0, float((allxy.max(0) - allxy.min(0)).max()) / 2.0 + 3.0)
    tv, tf = _flat_terrain((cx, cy), half=half)
    data = {
        "root_pos": root_pos, "root_quat_xyzw": root_quat, "joint_pos": joint,
        "vel_cmd": vel_cmd, "done": np.zeros((L, r_n), np.uint8), "pred_index": pidx,
        "pred_root_pos": pred_rp, "pred_root_quat_xyzw": pred_rq, "pred_joint_pos": pred_jp,
        "terrain_vertices": tv, "terrain_faces": tf,
    }
    meta = VizRecordingMeta(
        urdf_path=urdf_path, task_name="fm_test_bench/forecast", fm_checkpoint=args.fm_checkpoint,
        recorded_at=datetime.now().isoformat(timespec="seconds"), quat_format="xyzw", dry_run=False,
        dt=results[0].dt, update_every=max(1, args.forecast_stride), num_envs=r_n, num_steps=L,
        history_len=n, future_len=fnov, num_joints=nj, num_scan_rays=0,
        joint_names=list(ctx.joint_names), terrain_source="plane_fallback",
    )
    return data, meta


def rollout_to_viz(res: RolloutResult, ctx: FmOnlineContext, urdf_path: str, args):
    """Build a viz recording (data, meta) from an autoregressive rollout (robot only)."""
    from robot_rl.flow_matching.viz import VizRecordingMeta  # lazy

    if "root_pos" not in res.traj:
        raise SystemExit("[bench] rollout produced no steps to visualize.")
    rp = res.traj["root_pos"].astype(np.float32)
    rq = res.traj["root_quat"].astype(np.float32)
    jp = res.traj["joint_pos"].astype(np.float32)
    t_n, nj = rp.shape[0], jp.shape[1]
    cmd = res.cmd.astype(np.float32) if res.cmd.shape[0] == t_n else np.zeros((t_n, 3), np.float32)
    tv, tf = _flat_terrain((float(rp[:, 0].mean()), float(rp[:, 1].mean())))
    data = {
        "root_pos": rp[:, None, :], "root_quat_xyzw": rq[:, None, :], "joint_pos": jp[:, None, :],
        "vel_cmd": cmd[:, None, :], "done": np.zeros((t_n, 1), np.uint8),
        "pred_index": np.full((t_n, 1), -1, np.int32),  # no prediction ghosts for the rollout
        "pred_root_pos": np.zeros((1, 1, 3), np.float32),
        "pred_root_quat_xyzw": np.tile(np.array([0, 0, 0, 1], np.float32), (1, 1, 1)),
        "pred_joint_pos": np.zeros((1, 1, nj), np.float32),
        "terrain_vertices": tv, "terrain_faces": tf,
    }
    meta = VizRecordingMeta(
        urdf_path=urdf_path, task_name="fm_test_bench/rollout", fm_checkpoint=args.fm_checkpoint,
        recorded_at=datetime.now().isoformat(timespec="seconds"), quat_format="xyzw", dry_run=False,
        dt=res.dt, update_every=max(1, args.advance), num_envs=1, num_steps=t_n,
        history_len=ctx.N, future_len=1, num_joints=nj, num_scan_rays=0,
        joint_names=list(ctx.joint_names), terrain_source="plane_fallback",
    )
    return data, meta


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
    p.add_argument("--fm_checkpoint", default=None,
                   help="Path to the FM .pt checkpoint (default: latest logs/flow_matching/*/best.pt).")
    p.add_argument("--data", default=None, help="Recording .npz (default: auto-resolve from the run dir).")
    p.add_argument("--mode",
                   choices=["sweep_command", "sweep_history", "sweep_phase", "forecast", "rollout",
                            "parity", "viz", "all"],
                   default="all")
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
    p.add_argument("--sweep-hist", default="walking", help="Canonical history used for sweep_command.")
    # sweep_history
    p.add_argument("--histories", default="standing,walking,running,turning")
    p.add_argument("--future-cmd", default="1.0,0,0", help="Fixed command for sweep_history and sweep_phase.")
    # command profile (sweep_command / sweep_history): ramp toward the target instead
    # of stepping to it, so the swept command stays in-distribution vs the history.
    p.add_argument("--cmd-ramp", action="store_true",
                   help="Ramp the swept command from each history's current command toward the target "
                        "(in-distribution; training target_vel slews) instead of a constant step.")
    p.add_argument("--ramp-accel", type=float, default=1.0,
                   help="Ramp rate for --cmd-ramp in m/s^2 (vx,vy) and rad/s^2 (wz); matches the training "
                        "env max_acc (1.0). Over a short window far targets may not be reached.")
    # sweep_phase (does the prediction track the history's gait phase?)
    p.add_argument("--phase-hist", default="walking", help="Canonical gait used for the phase sweep.")
    p.add_argument("--n-phases", type=int, default=8, help="Number of phase offsets across one gait period.")
    p.add_argument("--gait-period", type=int, default=0,
                   help="Gait period in steps (0 = auto-estimate via foot-height autocorrelation).")
    p.add_argument("--gait-max-period", type=int, default=80,
                   help="Upper bound (steps) for the period / extra steady stretch length to scan.")
    # forecast (open-loop predictions overlaid on a ground-truth recording)
    p.add_argument("--forecast-hist", default="walking",
                   help="Canonical command used to locate the ground-truth stretch for the forecast.")
    p.add_argument("--forecast-len", type=int, default=150,
                   help="Ground-truth stretch length (steps) for the forecast overlay.")
    p.add_argument("--forecast-stride", type=int, default=5,
                   help="Re-forecast every this many ground-truth steps.")
    p.add_argument("--forecast-pad", type=int, default=60,
                   help="Steps of recorded command context to show before/after the FM region "
                        "(trimmed at episode resets) so you can see ramps into/out of it.")
    p.add_argument("--fk-model", default=os.path.join(ROOT, "transfer", "sim", "robots", "g1", "g1_21j.xml"),
                   help="MuJoCo G1 model for the EE-vs-FK consistency page (FK of generated joints).")
    p.add_argument("--no-fk", action="store_true",
                   help="Skip the generated-EE-vs-FK(generated-joints) consistency page.")
    # parity compare (teacher vs tracker, both tracking the parity clip)
    p.add_argument("--parity-teacher",
                   default=os.path.join(ROOT, "logs", "flow_matching", "test_bench", "parity_teacher"),
                   help="Teacher-on-parity record_data .npz (or a dir; newest .npz is used).")
    p.add_argument("--parity-tracker",
                   default=os.path.join(ROOT, "logs", "flow_matching", "test_bench", "parity_tracker"),
                   help="Tracker-on-parity record_data .npz (or a dir; newest .npz is used). Optional.")
    # rollout
    p.add_argument("--rollout_steps", type=int, default=200)
    p.add_argument("--advance", type=int, default=1, help="Predicted steps consumed per inference (<=F).")
    p.add_argument("--rollout-cmd", default="1.0,0,0", help="'vx,vy,wz' or 'ramp:a:b'.")
    p.add_argument("--rollout-seed-history", default="walking")
    # viser visualization (--mode viz): play the generated motion as a 3D G1 robot
    p.add_argument("--viz-source", choices=["forecast", "rollout", "parity"], default="forecast",
                   help="What --mode viz visualizes: forecast (GT robots + ghosts), rollout, or parity "
                        "(robot 0 = teacher, robot 1 = tracker, each tracking the parity clip with FM ghosts).")
    p.add_argument("--urdf", default=os.path.join(ROOT, "robot_assets", "g1", "g1_21j", "g1_21j.urdf"),
                   help="Robot URDF rendered by the viser viz.")
    p.add_argument("--viz-port", type=int, default=8080, help="Viser server port for --mode viz.")
    p.add_argument("--viz-ghosts", type=int, default=6, help="Initially shown prediction ghosts (forecast viz).")
    p.add_argument("--viz-max-ghosts", type=int, default=10, help="Ghost slot-pool size (forecast viz).")
    p.add_argument("--viz-robots", type=int, default=4,
                   help="Forecast viz: number of robots (one per gait from --viz-histories), shown in lanes.")
    p.add_argument("--viz-histories", default="standing,walking,running,turning",
                   help="Forecast viz: comma-separated canonical gaits, one robot each (capped at --viz-robots).")
    p.add_argument("--viz-len", type=int, default=300,
                   help="Forecast viz: ground-truth steps per robot (default ~2x the forecast-plot default).")
    p.add_argument("--viz-lane-spacing", type=float, default=2.0,
                   help="Forecast viz: lateral spacing (m) between the lane-aligned robots.")
    # plotting / output
    p.add_argument("--vel-frame", choices=["body", "world"], default="body",
                   help="Frame for plotted linear/angular velocities (body = base/heading frame, "
                        "directly comparable to the command; world = raw recorded frame).")
    p.add_argument("--vary-noise", action="store_true", help="Use different sampling noise per overlay.")
    p.add_argument("--dump-data", action="store_true",
                   help="Also write the forecast (ground truth + forecasts + command) and rollout "
                        "trajectories as .npz alongside the plots, for offline analysis.")
    p.add_argument("--save", default=None, help="Force PNG output to this directory.")
    p.add_argument("--out", default=os.path.join(ROOT, "logs", "flow_matching", "test_bench"))
    return p


def main() -> int:
    """Parse args, fixture the FM, run the requested mode(s), and show/save plots."""
    args = build_arg_parser().parse_args()
    plt, interactive = choose_backend(args.save is not None)

    device = torch.device(args.device)
    auto = args.fm_checkpoint is None
    args.fm_checkpoint = args.fm_checkpoint or default_fm_checkpoint()
    print(f"[bench] FM checkpoint: {os.path.abspath(args.fm_checkpoint)}"
          f"{'  (auto: latest run -- pass --fm_checkpoint to pin)' if auto else ''}")
    # Guard against pre-PARC checkpoints: the new loader needs ``state_normalizer``;
    # old encoder-decoder checkpoints (target/history split) would die with a cryptic
    # KeyError. Load once on CPU and pass the dict through to avoid a double load.
    ckpt_raw = torch.load(args.fm_checkpoint, map_location="cpu", weights_only=False)
    if "state_normalizer" not in ckpt_raw:
        norms = sorted(k for k in ckpt_raw if "norm" in k.lower())
        raise SystemExit(
            f"[bench] {args.fm_checkpoint} looks like a pre-PARC FM checkpoint "
            f"(no 'state_normalizer'; found {norms}). This bench targets the PARC "
            f"single-transformer arch -- train or pull a new-arch checkpoint, or pass "
            f"--fm_checkpoint to a PARC run."
        )
    ctx = load_fm_context(ckpt_raw, device, num_sampling_steps=args.num_sampling_steps)
    # dt is not on the context; the recording carries it -- patched on below.
    print(f"[bench] FM: N={ctx.N} F={ctx.F} nj={ctx.nj} state_dim={ctx.D} "
          f"cmd_dim={ctx.cmd_dim} num_cmd_tokens={ctx.num_cmd_tokens} "
          f"target_frames={ctx.target_frame_names} steps={ctx.num_steps}")

    # Parity modes load their own (teacher/tracker) recordings; everything else needs
    # the default recording (mined for stretches / canonical histories).
    parity_run = args.mode == "parity" or (args.mode == "viz" and args.viz_source == "parity")
    data = layout = None
    if not parity_run:
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

    def parity_results() -> list[ForecastResult]:
        """Forecast teacher- and (if available) tracker-on-parity recordings."""
        results = [run_parity_forecast(ctx, latest_npz(args.parity_teacher), "teacher", args, device)]
        try:
            results.append(run_parity_forecast(ctx, latest_npz(args.parity_tracker), "tracker", args, device))
        except SystemExit as ex:
            print(f"[bench] parity: tracker recording unavailable ({ex}); teacher only.")
        return results

    # -- viser visualization mode: build a recording from a forecast/rollout and serve it --
    if args.mode == "viz":
        try:
            from robot_rl.flow_matching.viz import FlowMatchingViewer, save_viz_recording, unique_output_path
        except Exception as ex:  # viser / yourdfpy / robot_rl viz unavailable
            raise SystemExit(f"[bench] viz unavailable ({type(ex).__name__}: {str(ex)[:120]}); "
                             f"needs viser + yourdfpy (and the robot_rl.flow_matching.viz package).")
        if not os.path.exists(args.urdf):
            raise SystemExit(f"[bench] --urdf not found: {args.urdf}")
        if args.viz_source == "parity":
            results = parity_results()  # robot 0 = teacher, robot 1 = tracker (if available)
            vdata, vmeta = forecast_multi_to_viz(results, ctx, args.urdf, args)
            print(f"[bench] viz parity: {len(results)} robots ({', '.join(r.label for r in results)})")
        elif args.viz_source == "rollout":
            vdata, vmeta = rollout_to_viz(run_rollout(ctx, get_window(args.rollout_seed_history), args, device),
                                          ctx, args.urdf, args)
        else:
            gaits = [g.strip() for g in args.viz_histories.split(",") if g.strip()][:max(1, args.viz_robots)]
            results: list[ForecastResult] = []
            for g in gaits:
                a = copy.copy(args)
                a.forecast_hist, a.forecast_len, a.no_fk = g, args.viz_len, True  # FK not needed for viz
                try:
                    results.append(run_forecast(ctx, data, layout, a, device))
                except Exception as ex:  # e.g. no steady stretch of that gait/length
                    print(f"[bench] viz: skipping gait {g!r} ({type(ex).__name__}: {str(ex)[:80]})")
            if not results:
                raise SystemExit("[bench] no viz robots could be built; try other --viz-histories / lower --viz-len.")
            vdata, vmeta = forecast_multi_to_viz(results, ctx, args.urdf, args)
            print(f"[bench] viz: {len(results)} robots ({', '.join(gaits[:len(results)])}), "
                  f"{vmeta.num_steps} steps each")
        out_dir = args.save or os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        rec_path = unique_output_path(out_dir, prefix=f"viz_{args.viz_source}")
        save_viz_recording(rec_path, vdata, vmeta)
        print(f"[bench] saved viz recording: {rec_path}")
        print(f"[bench] launching viser on port {args.viz_port} -- open http://localhost:{args.viz_port} "
              f"(Ctrl-C to quit; re-view later with scripts/flow_matching/visualize.py --data <rec>)")
        viewer = FlowMatchingViewer(vdata, vmeta, args.urdf, port=args.viz_port,
                                    max_ghosts=args.viz_max_ghosts,
                                    initial_ghosts=min(args.viz_ghosts, args.viz_max_ghosts))
        try:
            viewer.run()
        except KeyboardInterrupt:
            pass
        return 0

    # -- parity compare mode: teacher vs tracker on the parity clip (overlay plots) --
    if args.mode == "parity":
        results = parity_results()
        pages = [
            Page("parity / root", lambda fig, r=results: draw_parity(fig, r)),
            Page("parity / command", lambda fig, r=results: draw_parity_cmd(fig, r)),
        ]
        for r in results:  # per-source detail (the standard forecast pages)
            pages.append(Page(f"parity / {r.label} / root", lambda fig, rr=r: draw_forecast(fig, rr)))
            pages.append(Page(f"parity / {r.label} / xy", lambda fig, rr=r: draw_forecast_path(fig, rr)))
            if "fk_ee_pos" in r.gt:
                pages.append(Page(f"parity / {r.label} / ee vs fk", lambda fig, rr=r: draw_forecast_fk(fig, rr)))
        browser = FigureBrowser(pages, plt)
        out_dir = args.save or os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if args.dump_data:
            os.makedirs(out_dir, exist_ok=True)
            for r in results:
                dump_forecast(r, os.path.join(out_dir, f"parity_{r.label}.npz"))
        if interactive:
            browser.show()
        else:
            browser.save_all(out_dir)
        return 0

    pages: list[Page] = []
    modes = (["sweep_command", "sweep_history", "sweep_phase", "forecast", "rollout"]
             if args.mode == "all" else [args.mode])
    forecast_result: ForecastResult | None = None
    rollout_result: RolloutResult | None = None

    if "sweep_command" in modes:
        win = get_window(args.sweep_hist)
        pages += scenario_pages(run_sweep_command(ctx, win, args, device))
    if "sweep_history" in modes:
        names = [n.strip() for n in args.histories.split(",") if n.strip()]
        windows = [get_window(n) for n in names]
        pages += scenario_pages(run_sweep_history(ctx, windows, args, device))
    if "sweep_phase" in modes:
        pages += scenario_pages(run_sweep_phase(ctx, data, layout, args, device))
    if "forecast" in modes:
        forecast_result = run_forecast(ctx, data, layout, args, device)
        pages.append(Page("forecast / root", lambda fig, r=forecast_result: draw_forecast(fig, r)))
        pages.append(Page("forecast / xy path", lambda fig, r=forecast_result: draw_forecast_path(fig, r)))
        pages.append(Page("forecast / command", lambda fig, r=forecast_result: draw_forecast_cmd(fig, r)))
        if "fk_ee_pos" in forecast_result.gt:
            pages.append(Page("forecast / ee vs fk (pos)", lambda fig, r=forecast_result: draw_forecast_fk(fig, r)))
            pages.append(Page("forecast / ee vs fk (vel)", lambda fig, r=forecast_result: draw_forecast_fk_vel(fig, r)))
    if "rollout" in modes:
        win = get_window(args.rollout_seed_history)
        rollout_result = run_rollout(ctx, win, args, device)
        pages.append(Page("rollout / stability", lambda fig, r=rollout_result: draw_rollout(fig, r)))

    browser = FigureBrowser(pages, plt)
    out_dir = args.save or os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if args.dump_data:
        os.makedirs(out_dir, exist_ok=True)
        if forecast_result is not None:
            dump_forecast(forecast_result, os.path.join(out_dir, "forecast_data.npz"))
        if rollout_result is not None:
            dump_rollout(rollout_result, os.path.join(out_dir, "rollout_data.npz"))
    if interactive:
        browser.show()
    else:
        browser.save_all(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
