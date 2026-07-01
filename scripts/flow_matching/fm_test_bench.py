#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Standalone, torch-only diagnostic bench for the flow-matching (FM) generator.

Fixtures a trained FM checkpoint and replays a recorded **teacher rollout** through
it, then plots the predicted future trajectories. Runs WITHOUT IsaacSim (imports
only the vendored, isaaclab-free ``transfer.fm`` inference stack), so it is fast and
usable locally.

The model is the robot-state-conditioned PARC-style single transformer: a window of
``F`` state tokens whose first ``N`` (= ``num_history_states``) are the recent
measured *robot* states held clean as conditioning (anchor = step N-1), plus a height
token and a strided velocity-command profile, with the novel prediction occupying
tokens ``N..F-1``.

This bench operates ONLY on robot-state-conditioned teacher recordings
(``record_data.py`` with ``meta.ref_source == "ego_anchored_clip_future"``): step-major
env-interleaved rollouts that also carry ``ref_future`` (the ego-anchored clip future
the FM is trained to predict). Clean clip recordings and other robot rollouts are
rejected at load.

The single ``forecast_step`` mode is an arrow-key step scrubber: at each FM
re-inference (snap) step it conditions the FM on the recorded robot-state history +
height + future command, samples the F-step future, and overlays it against the real
ego-anchored clip future (``ref_future``) and the robot's actual recorded future.
Left/Right focus a snap; Up/Down toggle the overlay/detail views.

Plots are shown in an interactive matplotlib window when a display is available, and
saved as PNGs otherwise (``--save DIR`` forces PNG output).

Example::

    python scripts/flow_matching/fm_test_bench.py \
        --fm_checkpoint logs/flow_matching/<run>/best.pt --device cpu
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

# Repo root: this file is scripts/flow_matching/fm_test_bench.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Vendored, isaaclab-free inference stack (pure torch). See transfer/fm/__init__.py.
from transfer.fm import (  # noqa: E402
    build_cmd_tokens,
    build_height_token,
    build_history_tokens,
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


def _quat_mul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of xyzw quaternions ``(..., 4)`` (a ⊗ b)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def _quat_conj_t(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of xyzw quaternions ``(..., 4)``."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([-x, -y, -z, w], dim=-1)


def _yaw_quat_t(q: torch.Tensor) -> torch.Tensor:
    """Yaw-only (z-rotation) xyzw quaternion of ``q`` ``(..., 4)`` -- mirrors isaaclab yaw_quat."""
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    zr = torch.zeros_like(yaw)
    return torch.stack([zr, zr, torch.sin(half), torch.cos(half)], dim=-1)


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
    ref_source: str = ""
    ref_future_len: int = 0


# ======================================================================
# 2. recording loader (zipfile members -- the .npz is DEFLATE-compressed)
# ======================================================================
# Only these arrays are read; the recording also holds obs (1.6GB), ref_state and
# action which the bench never needs.
_NEEDED_ARRAYS = ("state", "target_vel", "episode_id", "step_in_episode", "done")
# Robot-state-conditioned (ego_anchored_clip_future) recordings carry these too; loaded
# when present so the teacher-data forecast_step can overlay the real clip future.
_OPTIONAL_ARRAYS = ("ref_future", "cmd_future", "reanchor_active")
_NEEDED_META = (
    "state_term_names", "state_term_dims", "state_frame_names",
    "state_joint_names", "num_envs", "dt", "ref_source", "ref_future_len",
)


def _read_member(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Read one ``.npy`` member of an npz zip into an array."""
    with zf.open(name + ".npy") as f:
        return np.lib.format.read_array(f, allow_pickle=False)


def load_recording_lite(path: str) -> tuple[dict[str, np.ndarray], DuckMeta]:
    """Load only the arrays the bench needs from a ``clip_to_fm_data.py`` ``.npz``.

    Args:
        path: Path to the clean clip recording ``.npz`` (episode-contiguous,
            ``num_envs == 0``).

    Returns:
        ``(data, meta)`` where ``data`` holds the numpy arrays in
        :data:`_NEEDED_ARRAYS` and ``meta`` is a :class:`DuckMeta`.

    Raises:
        ValueError: If the recording is a robot rollout (``num_envs > 0``); the
            clip-conditioned FM (and this bench) never use robot state.
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
        for key in _OPTIONAL_ARRAYS:
            if key in present:
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
        ref_source=str(raw_meta["ref_source"]) if "ref_source" in raw_meta else "",
        ref_future_len=int(raw_meta["ref_future_len"]) if "ref_future_len" in raw_meta else 0,
    )
    print(f"[bench]   state {data['state'].shape}, num_envs={meta.num_envs}, dt={meta.dt}, "
          f"ref_source={meta.ref_source or 'none'}")
    # Robot-state-conditioned recordings (ego_anchored_clip_future) ARE robot rollouts
    # (num_envs>0, step-major) and carry ref_future -- the teacher-data forecast_step uses
    # them. Only reject robot rollouts that are NOT this kind.
    is_teacher = meta.ref_source == "ego_anchored_clip_future" and "ref_future" in data
    if meta.num_envs > 0 and not is_teacher:
        raise ValueError(
            f"fm_test_bench operates on clean clip recordings (num_envs=0) or robot-state-"
            f"conditioned teacher recordings (ref_source='ego_anchored_clip_future' + ref_future) "
            f"-- {path} has num_envs={meta.num_envs}, ref_source={meta.ref_source!r}."
        )
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
# 4. Teacher-data forecast_step (robot-state-conditioned FM)
# ======================================================================
@dataclass
class TeacherStepResult:
    """Per-snap comparison for a recorded teacher rollout: at each FM re-inference
    (snap) step, the real ego-anchored clip future (the target) vs the FM prediction
    conditioned on the recorded robot-state history vs the robot's actual future."""

    anchors: list[int]                         # step_in_episode of each snap
    real: list[dict[str, np.ndarray]]          # (F, .) ego-anchored clip future (ref_future)
    fm: list[dict[str, np.ndarray]]            # (F, .) FM prediction (history-conditioned)
    robot: list[dict[str, np.ndarray]]         # (<=F, .) robot actual future (recorded state)
    spine: dict[str, np.ndarray]               # (L, .) robot actual over the WHOLE episode
    spine_steps: np.ndarray                    # (L,) step_in_episode of each spine row
    episode_id: int
    split: str
    N: int
    F: int
    dt: float
    frame_names: list[str]
    joint_names: list[str]
    foot_idxs: list[int]


def _episode_rows_sorted(data: dict[str, np.ndarray], ep: int) -> np.ndarray:
    """Absolute rows of episode ``ep``, time-ordered by ``step_in_episode``.

    Robot rollouts are step-major env-interleaved, so a single ``episode_id``'s rows
    are scattered; sort them by step to recover the time order.
    """
    rows = np.nonzero(data["episode_id"] == ep)[0]
    return rows[np.argsort(data["step_in_episode"][rows], kind="stable")]


def _val_episode_ids(fm_checkpoint: str) -> set[int]:
    """Read the held-out ``val_episode_ids`` from the checkpoint run's data.yaml (or empty)."""
    run_dir = os.path.dirname(os.path.abspath(fm_checkpoint))
    data_yaml = os.path.join(run_dir, "params", "data.yaml")
    if not os.path.exists(data_yaml):
        return set()
    try:
        import yaml  # noqa: PLC0415
        with open(data_yaml) as f:
            d = yaml.safe_load(f)
        return {int(e) for e in (d.get("val_episode_ids") or [])}
    except Exception as exc:
        print(f"[bench] could not read val_episode_ids from {data_yaml}: {exc}")
        return set()


def _select_teacher_episode(data, args) -> tuple[int, str]:
    """Pick the episode_id to scrub, honoring --forecast-clip and --forecast-split."""
    val_eps = _val_episode_ids(args.fm_checkpoint)
    all_eps = [int(e) for e in np.unique(data["episode_id"])]
    if args.forecast_split == "eval":
        pool = [e for e in all_eps if e in val_eps]
        if not pool:
            raise SystemExit("[bench] --forecast-split eval: no val_episode_ids found in data.yaml.")
    elif args.forecast_split == "train":
        pool = [e for e in all_eps if e not in val_eps]
    else:
        pool = all_eps
    if args.forecast_clip >= 0:
        if args.forecast_clip not in pool:
            raise SystemExit(f"[bench] episode {args.forecast_clip} not in the '{args.forecast_split}' split.")
        return args.forecast_clip, args.forecast_split
    return pool[0], args.forecast_split


def _world_pred_to_traj(world: dict[str, "torch.Tensor"]) -> dict[str, np.ndarray]:
    """Convert a :func:`sample_world` output (batch 0) to the bench traj-dict format."""
    out = {
        "root_pos": world["root_pos_w"][0].cpu().numpy(),
        "root_quat": world["root_quat_w"][0].cpu().numpy(),
        "root_lin_vel": world["root_link_lin_vel_w"][0].cpu().numpy(),
        "root_ang_vel": world["root_link_ang_vel_w"][0].cpu().numpy(),
        "joint_pos": world["joint_pos"][0].cpu().numpy(),
        "joint_vel": world["joint_vel"][0].cpu().numpy(),
    }
    if "frame_pos_w" in world:
        # sample_world frames are (F, K, .) for batch 0; bench traj uses (F, K, .) too.
        out["ee_pos"] = world["frame_pos_w"][0].cpu().numpy()
        out["ee_quat"] = world["frame_quat_w"][0].cpu().numpy()
        out["ee_lin_vel"] = world["frame_lin_vel_w"][0].cpu().numpy()
        out["ee_ang_vel"] = world["frame_ang_vel_w"][0].cpu().numpy()
    return out


@torch.no_grad()
def run_forecast_step_teacher(ctx, data, meta, layout, args, device) -> TeacherStepResult:
    """Scrub a recorded teacher rollout at its snap (FM re-inference) steps.

    At each snap with N prior consecutive steps: condition the FM on the recorded
    robot-state history + height + future command, sample the F-step future, and pair it
    with the real ego-anchored clip future (``ref_future``) and the robot's actual future.
    """
    N, F = ctx.N, ctx.F
    ep, split = _select_teacher_episode(data, args)
    rows = _episode_rows_sorted(data, ep)
    steps = data["step_in_episode"][rows]
    snap = data["reanchor_active"][rows].astype(bool)
    state, ref_future, cmd_future = data["state"], data["ref_future"], data["cmd_future"]
    foot_idxs = [i for i, f in enumerate(meta.state_frame_names) if "ankle" in f.lower() or "foot" in f.lower()]

    def _t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)

    anchors, real, fm, robot = [], [], [], []
    for j in range(N - 1, len(rows)):
        if not snap[j] or not np.all(np.diff(steps[j - (N - 1):j + 1]) == 1):
            continue
        hist_rows, snap_row = rows[j - (N - 1):j + 1], rows[j]
        h = layout.parse(state[hist_rows])  # (N, .) robot-state history
        frames = None
        if meta.state_frame_names:
            frames = _t(np.concatenate([h["ee_pos"], h["ee_quat"], h["ee_lin_vel"], h["ee_ang_vel"]],
                                       axis=-1))[None]  # (1, N, K, 13)
        hist = build_history_tokens(
            ctx, _t(h["root_pos"])[None], _t(h["root_quat"])[None],
            _t(h["root_lin_vel"])[None], _t(h["root_ang_vel"])[None],
            _t(h["joint_pos"])[None], _t(h["joint_vel"])[None], frames=frames,
        )
        height = build_height_token(ctx, _t(h["root_pos"][-1, 2]))
        cmd = build_cmd_tokens(ctx, _t(cmd_future[snap_row])[None]) if ctx.cmd_dim > 0 else None
        p_ref, q_ref = _t(h["root_pos"][-1])[None], _t(h["root_quat"][-1])[None]
        world = sample_world(ctx, hist, height, cmd, p_ref, q_ref)
        fm.append(_world_pred_to_traj(world))
        real.append(layout.parse(ref_future[snap_row]))          # (F, .) ego-anchored clip future
        robot.append(layout.parse(state[rows[j:min(j + F, len(rows))]]))  # (<=F, .) robot actual
        anchors.append(int(steps[j]))

    if not anchors:
        raise SystemExit(f"[bench] episode {ep} has no usable snaps (need {N} prior consecutive steps).")
    spine = layout.parse(state[rows])  # robot-actual over the whole episode (the rollout spine)
    print(f"[bench] teacher forecast_step: episode {ep} ({split}), {len(anchors)} snaps, F={F}, "
          f"spine L={len(rows)}.")
    return TeacherStepResult(
        anchors=anchors, real=real, fm=fm, robot=robot, spine=spine, spine_steps=steps,
        episode_id=ep, split=split, N=N, F=F, dt=ctx.dt_attr, frame_names=meta.state_frame_names,
        joint_names=meta.state_joint_names, foot_idxs=foot_idxs,
    )


def draw_teacher_overlay(fig, res: TeacherStepResult, k: int) -> None:
    """Full-episode robot rollout (spine) with the FM prediction from every snap
    branching off it (colored by snap time), the focused snap ``k`` highlighted.

    Black = the robot's actual trajectory over the whole clip; faint viridis = each
    snap's FM-predicted future placed at its clip-step; crimson = the focused snap
    (solid = FM prediction, dashed = the real ego-anchored clip future it targets).
    Left/Right re-focus a snap; the spine + all branches stay drawn.
    """
    dt = res.dt
    sp_t = res.spine_steps.astype(float) * dt
    norm = mcolors.Normalize(0.0, max(1e-6, max(res.anchors) * dt))
    cmap = matplotlib.colormaps["viridis"]
    fig.clear()
    axs = fig.subplots(2, 2)

    def btime(ki):
        return (res.anchors[ki] + np.arange(res.F)) * dt

    def panel(ax, getter, title, ylabel):
        ax.plot(sp_t, getter(res.spine), color="k", lw=1.8, zorder=5, label="robot actual (rollout)")
        for ki in range(len(res.anchors)):
            ax.plot(btime(ki), getter(res.fm[ki]), color=cmap(norm(res.anchors[ki] * dt)),
                    lw=0.8, alpha=0.30, zorder=3)
        ax.plot(btime(k), getter(res.fm[k]), color="crimson", lw=2.2, zorder=7, label="FM pred @ snap")
        ax.plot(btime(k), getter(res.real[k]), color="crimson", lw=1.5, ls="--", zorder=7, label="real clip @ snap")
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("clip time (s)"); ax.grid(alpha=0.3)

    panel(axs[0, 0], lambda tr: tr["root_pos"][:, 0], "root x", "m")
    panel(axs[0, 1], lambda tr: tr["root_pos"][:, 2], "root height (z)", "m")
    fi = res.foot_idxs[0] if res.foot_idxs else 0
    if res.frame_names:
        panel(axs[1, 1], lambda tr: tr["ee_pos"][:, fi, 2], f"foot z ({res.frame_names[fi]})", "m")
    else:
        axs[1, 1].set_visible(False)
    axs[0, 0].legend(fontsize=7, loc="best")

    # top-down xy path
    ax = axs[1, 0]
    ax.plot(res.spine["root_pos"][:, 0], res.spine["root_pos"][:, 1], color="k", lw=1.8, zorder=5)
    for ki in range(len(res.anchors)):
        p = res.fm[ki]["root_pos"]
        ax.plot(p[:, 0], p[:, 1], color=cmap(norm(res.anchors[ki] * dt)), lw=0.8, alpha=0.30, zorder=3)
    pk, rk = res.fm[k]["root_pos"], res.real[k]["root_pos"]
    ax.plot(pk[:, 0], pk[:, 1], color="crimson", lw=2.2, zorder=7)
    ax.plot(rk[:, 0], rk[:, 1], color="crimson", lw=1.5, ls="--", zorder=7)
    ax.scatter([res.spine["root_pos"][0, 0]], [res.spine["root_pos"][0, 1]], c="g", s=30, zorder=8)
    ax.set_title("root xy path (world)"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.3)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis"); sm.set_array([])
    fig.colorbar(sm, ax=list(axs.ravel()), label="snap time (s)", fraction=0.02, pad=0.01)
    fig.suptitle(f"teacher forecast_step [overlay]  ep={res.episode_id} ({res.split})  "
                 f"snap {k + 1}/{len(res.anchors)} @ clip-step {res.anchors[k]}  "
                 f"(black=rollout, faint=FM branches, crimson=focused: solid FM / dashed real clip)")


def draw_teacher_detail(fig, res: TeacherStepResult, k: int) -> None:
    """Per-node detail at snap ``k``: real clip future / FM prediction / robot actual."""
    real, fm, robot = res.real[k], res.fm[k], res.robot[k]
    t = np.arange(res.F) * res.dt
    fig.clear()
    axes = fig.subplots(2, 3).ravel()
    series = [("real clip", real, "-", 2.0), ("FM pred", fm, "--", 1.8), ("robot", robot, ":", 1.6)]

    def _plot(ax, getter, title, ylabel):
        for label, tr, ls, lw in series:
            y = getter(tr)
            if y is None:
                continue
            ax.plot(t[:len(y)], y, ls, lw=lw, label=label)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.set_xlabel("future time (s)"); ax.grid(alpha=0.3)

    _plot(axes[0], lambda tr: tr["root_pos"][:, 0], "root x", "m")
    _plot(axes[1], lambda tr: tr["root_pos"][:, 1], "root y", "m")
    _plot(axes[2], lambda tr: tr["root_pos"][:, 2], "root height (z)", "m")
    # two representative joints
    _plot(axes[3], lambda tr: tr["joint_pos"][:, 0], f"joint[0] {res.joint_names[0]}", "rad")
    nj = real["joint_pos"].shape[1]
    ji = min(3, nj - 1)
    _plot(axes[4], lambda tr: tr["joint_pos"][:, ji], f"joint[{ji}] {res.joint_names[ji]}", "rad")
    # foot height (first foot frame), else first ee frame z
    fi = res.foot_idxs[0] if res.foot_idxs else 0
    if res.frame_names:
        _plot(axes[5], lambda tr: tr["ee_pos"][:, fi, 2], f"foot z ({res.frame_names[fi]})", "m")
    else:
        axes[5].set_visible(False)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"teacher forecast_step  ep={res.episode_id} ({res.split})  "
                 f"snap {k + 1}/{len(res.anchors)}  clip-step {res.anchors[k]}")
    fig.tight_layout()


class TeacherForecastBrowser:
    """Scrub a :class:`TeacherStepResult`: Left/Right focus a snap, Up/Down toggle view.

    Two views: ``overlay`` (full-episode rollout spine with every snap's FM branch
    fanning off it, focused snap highlighted) and ``detail`` (per-node real / FM /
    robot at the focused snap)."""

    _VIEWS = (("overlay", draw_teacher_overlay), ("detail", draw_teacher_detail))

    def __init__(self, res: TeacherStepResult, plt):
        self.res, self.plt, self.k, self.view, self.n = res, plt, 0, 0, len(res.anchors)

    def _render(self, fig):
        name, draw = self._VIEWS[self.view]
        draw(fig, self.res, self.k)
        fig.canvas.manager.set_window_title(
            f"[snap {self.k + 1}/{self.n} | {name}] teacher forecast_step  clip-step {self.res.anchors[self.k]}"
        )
        fig.canvas.draw_idle()

    def _on_key(self, event, fig):
        if event.key in ("right", " "):
            self.k = (self.k + 1) % self.n; self._render(fig)
        elif event.key == "left":
            self.k = (self.k - 1) % self.n; self._render(fig)
        elif event.key in ("up", "down"):
            self.view = (self.view + 1) % len(self._VIEWS); self._render(fig)
        elif event.key in ("q", "escape"):
            self.plt.close(fig)

    def show(self):
        fig = self.plt.figure(figsize=(15, 9))
        fig.canvas.mpl_connect("key_press_event", lambda e: self._on_key(e, fig))
        self._render(fig)
        print(f"[bench] teacher forecast_step: {self.n} snaps -- Left/Right focus a snap, "
              f"Up/Down toggle overlay/detail, q to quit")
        self.plt.show()

    def save_all(self, out_dir: str, stride: int):
        os.makedirs(out_dir, exist_ok=True)
        # One overlay per strided snap + a detail for each strided snap.
        for name, draw in self._VIEWS:
            for k in range(0, self.n, max(1, stride)):
                fig = self.plt.figure(figsize=(15, 9))
                draw(fig, self.res, k)
                path = os.path.join(out_dir, f"teacher_{name}_{self.res.anchors[k]:04d}.png")
                fig.savefig(path, dpi=150, bbox_inches="tight"); self.plt.close(fig)
                print(f"[bench]   saved {path}")


# ======================================================================
# 5. CLI + main
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
    p.add_argument("--mode", choices=["forecast_step"], default="forecast_step",
                   help="Only 'forecast_step' is supported: a standalone, arrow-key step scrubber over a "
                        "recorded teacher rollout (real ego-anchored clip future vs history-conditioned FM "
                        "prediction vs robot actual).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num_sampling_steps", type=int, default=None, help="Override Euler step count.")
    # forecast_step (teacher rollout scrubber)
    p.add_argument("--forecast-clip", type=int, default=-1,
                   help="forecast_step: episode_id to scrub (-1 = first in the chosen --forecast-split).")
    p.add_argument("--forecast-split", choices=["any", "train", "eval"], default="any",
                   help="forecast_step: restrict episode selection to the train or held-out eval set "
                        "(read from the checkpoint's data.yaml val_episode_ids). Default any.")
    p.add_argument("--forecast-stride", type=int, default=5,
                   help="forecast_step (PNG save): render every this many snaps.")
    # output
    p.add_argument("--save", default=None, help="Force PNG output to this directory.")
    p.add_argument("--out", default=os.path.join(ROOT, "logs", "flow_matching", "test_bench"))
    return p


def main() -> int:
    """Parse args, fixture the FM, run the teacher forecast_step scrubber, and show/save plots."""
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

    data_path = resolve_data_path(args.fm_checkpoint, args.data)
    data, meta = load_recording_lite(data_path)
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

    teacher = meta.ref_source == "ego_anchored_clip_future"
    if not teacher:
        raise SystemExit(
            "[bench] this bench now only supports robot-state teacher recordings "
            "(ref_source=ego_anchored_clip_future)."
        )

    # The only mode is the standalone arrow-key step scrubber over a recorded teacher
    # rollout: at each FM re-inference (snap) it compares the real ego-anchored clip
    # future vs the history-conditioned FM prediction vs the robot's actual future.
    step_res = run_forecast_step_teacher(ctx, data, meta, layout, args, device)
    scrubber = TeacherForecastBrowser(step_res, plt)
    out_dir = args.save or os.path.join(args.out, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if interactive:
        scrubber.show()
    else:
        scrubber.save_all(out_dir, args.forecast_stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
