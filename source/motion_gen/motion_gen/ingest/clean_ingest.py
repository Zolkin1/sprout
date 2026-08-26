# Copyright 2026 Zachary Olkin. All rights reserved.

"""Ingest the clean steady-state periodic trajectory library into motion-matching sources.

The library (e.g. HuggingFace ``trajectories/skills/library_v1``) is a set of traj-opt
*periodic strides* -- one ``.npz`` per trajectory, in per-skill folders, plus a
``manifest.csv`` with columns ``skill, vel_x, vel_y, vel_yaw, periodic, status, clip, file``.
Each trajectory tracks ONE steady ``(vx, vy, vyaw)`` "perfectly" (traj-opt closure), so they
ingest as the STEADY tier (``steady=True``) and, being single closed cycles, as
``wrap="periodic"`` (auto-loop in the matcher).

Each ``.npz`` holds a MuJoCo ``g1_29dof_feet.xml`` trajectory::

    state[T, 71] = qpos(36) + qvel(35)
                 = [root_pos(3), root_quat(4, wxyz), 29 joints] + [lin(3), ang(3), 29 qd]
    time[T]      = per-sample timestamps (sim_dt ~ 0.005 s -- NOT the coarse `node_dt`)
    meta         = json string (skill / conditioner / achieved_twist / ...)

We take ``root_pos``/``root_quat`` + the 29 joints, map the 29 (canonical Unitree G1 order,
identical to the bones seed-CSV order) to the G1-21 set via :data:`bones_ingest._G1_FROM_CSV`,
resample to ``1/dt`` Hz, and finite-difference velocities -- the SAME conventions as
:mod:`.bones_ingest` / :mod:`.trajopt_ingest` so descriptors are consistent across sources.
The stored ``qvel`` columns are ignored (MuJoCo free-joint frame conventions differ); all
velocities are recomputed by finite difference.
"""

from __future__ import annotations

import csv as _csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import ingest_common as IC
from .bones_ingest import _G1_FROM_CSV, _resample
from .motion_db import BONES, MotionSource

_QPOS = 36  # root_pos(3) + root_quat(4) + 29 joints


def _resolve_dir(path: str, hf_repo: str | None) -> Path:
    """Return the local library directory, downloading from HuggingFace if ``hf_repo`` is set.

    Caches under ``./hf/<path>`` (see :mod:`motion_gen.hf_cache`).
    """
    from ..hf_cache import resolve_library_path

    return resolve_library_path(path, hf_repo)


def _read_manifest(lib_dir: Path) -> list[dict]:
    """Read ``manifest.csv`` -> list of row dicts (one per trajectory)."""
    mpath = lib_dir / "manifest.csv"
    if not mpath.is_file():
        raise FileNotFoundError(f"manifest.csv not found under {lib_dir}")
    with open(mpath, newline="") as f:
        return list(_csv.DictReader(f))



#: Substrings IPOPT reports on a run that actually converged. Everything else -- "Maximum number
#: of iterations exceeded", "Restoration phase failed" -- is a solve that did NOT.
_CONVERGED_MARKERS = ("locally optimal", "optimal solution found")


def solve_converged(row: dict) -> bool:
    """Did the solver converge for this manifest row?

    NOTHING in this repo read the ``status`` column until now, so every non-converged solve was
    ingested identically to a converged one. Measured on ``library_v2``: 60 of 302 rows (19.9%)
    did not converge, and the failures are concentrated rather than spread -- ``run_turn`` 28/42,
    ``stairs_down`` 6/8, ``jump_off_front`` 4/7, ``jog_turn`` 10/35.

    What to DO about it differs by consumer, which is why this returns a fact rather than
    filtering here:

    * stair TILE geometry must use converged cells only -- a tile authored around a reference
      whose solve failed is a staircase nothing in the library can walk;
    * the flat families must NOT be filtered -- dropping 28 of 42 ``run_turn`` cells would open a
      hole across the whole fast-turn corner, which is worse than a locally-suboptimal solve, and
      nearest-neighbour selection copes with the latter.

    An absent or empty ``status`` counts as converged: older manifests do not carry the column,
    and treating "unknown" as failure would silently empty them.
    """
    st = (row.get("status") or "").strip().lower()
    return (not st) or any(m in st for m in _CONVERGED_MARKERS)


def summarize_convergence(rows: list[dict]) -> dict[str, tuple[int, int]]:
    """``skill -> (converged, total)`` over manifest rows, for reporting."""
    out: dict[str, list[int]] = {}
    for r in rows:
        k = (r.get("skill") or "?").strip()
        c = out.setdefault(k, [0, 0])
        c[0] += int(solve_converged(r))
        c[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def _clip_to_state(npz_path: str, dt: float, device):
    """Load one library ``.npz`` -> per-frame world state, resampled + finite-differenced.

    Returns ``(joint_pos[F,21], root_pos[F,3], root_quat[F,4] wxyz, root_lin_vel[F,3],
    root_ang_vel[F,3], contact[F,2])`` or ``None`` if the file is missing / malformed / too short.

    ``contact`` is the SOLVER's own per-foot contact schedule, not a geometric guess.  Carrying it is
    what lets the clip say when a foot is actually planted: the geometric flag derived downstream
    reports a per-foot duty factor of 0.86 on these very frames, against the solver's 0.32, so it
    calls most of the swing phase "contact".
    """
    if not os.path.isfile(npz_path):
        return None
    try:
        d = np.load(npz_path, allow_pickle=True)
        st = np.asarray(d["state"], dtype=np.float64)
        tm = np.asarray(d["time"], dtype=np.float64)
    except Exception:
        return None
    if st.ndim != 2 or st.shape[1] < _QPOS or st.shape[0] < 4 or tm.shape[0] != st.shape[0]:
        return None
    # sample rate from the timestamps (the trajectory is stored at sim_dt, not node_dt)
    diffs = np.diff(tm)
    src_dt = float(np.median(diffs))
    if not np.isfinite(src_dt) or src_dt <= 0:
        return None
    src_fps = 1.0 / src_dt

    root_pos = st[:, 0:3]
    root_quat = st[:, 3:7]                       # wxyz
    joints29 = st[:, 7:_QPOS]                    # canonical G1 29-dof order (== seed-CSV order)

    # Contact is BOOLEAN, so it is sampled nearest-neighbour onto the same grid _resample builds --
    # interpolating a flag would invent fractional contact at every touchdown and lift-off.
    contact = None
    try:
        cf = np.asarray(d["contact_flags"])
        bodies = [str(b) for b in np.asarray(d["contact_bodies"])]
        feet = [bodies.index(n) for n in bodies if n.endswith(("left_ankle_roll_link",
                                                              "right_ankle_roll_link"))]
        if len(feet) == 2 and cf.shape[0] == st.shape[0]:
            t_new = np.arange(0.0, tm[-1] - tm[0] + 1e-9, dt)
            idx = np.clip(np.round(t_new * src_fps).astype(int), 0, cf.shape[0] - 1)
            contact = cf[idx][:, feet].astype(np.uint8)
    except Exception:
        contact = None

    rp, qq, jj = _resample(root_pos, root_quat, joints29, src_fps, dt)
    joint_pos = torch.as_tensor(jj[:, _G1_FROM_CSV], dtype=torch.float32, device=device)
    root_pos_t = torch.as_tensor(rp, dtype=torch.float32, device=device)
    root_quat_t = torch.as_tensor(qq, dtype=torch.float32, device=device)
    joint_vel = IC.central_diff(joint_pos, dt)
    root_lin_vel = IC.central_diff(root_pos_t, dt)
    root_ang_vel = IC.angular_velocity_from_quat(root_quat_t, dt)
    if contact is not None and contact.shape[0] != joint_pos.shape[0]:
        contact = None                       # grids disagreed; better absent than misaligned
    return joint_pos, root_pos_t, root_quat_t, root_lin_vel, root_ang_vel, contact


def ingest_clean_library(
    path: str,
    hf_repo: str | None = None,
    dt: float = 0.02,
    device: str | torch.device = "cpu",
    skills: list[str] | None = None,
    limit: int | None = None,
    min_frames: int = 8,
) -> list[MotionSource]:
    """Load the clean periodic-trajectory library into STEADY-tier :class:`MotionSource` list.

    Args:
        path: Library folder (local, or relative to the HF repo root when ``hf_repo`` set).
        hf_repo: Optional HuggingFace repo id (e.g. ``"zolkin/robot_rl"``) to download from.
        dt: Target timestep (matcher runs at ``1/dt`` Hz).
        device: Torch device for the returned tensors.
        skills: Optional whitelist of skill names (manifest ``skill`` column).
        limit: Optional cap on the number of clips.
        min_frames: Skip clips shorter than this (after resampling).

    Every clip is tagged ``steady=True`` with its manifest ``(vel_x, vel_y, vel_yaw)`` as
    ``steady_vel``, and ``wrap`` from the manifest ``periodic`` flag ("periodic" -> auto-loop).
    """
    lib_dir = _resolve_dir(path, hf_repo)
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"clean library directory not found: {lib_dir}")
    allow = set(skills) if skills is not None else None

    sources: list[MotionSource] = []
    for r in _read_manifest(lib_dir):
        skill = (r.get("skill") or "").strip()
        if allow is not None and skill not in allow:
            continue
        # Combined libraries (e.g. library_v2) hold BOTH flat locomotion and terrain interactions in
        # one manifest; the terrain rows carry a non-empty ``terrain`` column and belong to
        # terrain_ingest, not the clean/steady tier -- skip them here.  Older clean-only libraries
        # (library_v1) have no ``terrain`` column, so nothing is dropped.
        if (r.get("terrain") or "").strip():
            continue
        rel = (r.get("file") or "").strip()
        if not rel:
            continue
        st = _clip_to_state(str(lib_dir / rel), dt, device)
        if st is None or st[0].shape[0] < min_frames:
            continue
        joint_pos, root_pos_t, root_quat_t, root_lin_vel, root_ang_vel, contact = st
        try:
            steady_vel = torch.tensor(
                [float(r["vel_x"]), float(r["vel_y"]), float(r["vel_yaw"])],
                dtype=torch.float32, device=device)
        except (KeyError, ValueError, TypeError):
            steady_vel = None
        wrap = "periodic" if _truthy(r.get("periodic", "true")) else "episodic"
        stem = os.path.splitext(os.path.basename(rel))[0]
        sources.append(MotionSource(
            joint_pos=joint_pos, joint_vel=IC.central_diff(joint_pos, dt),
            root_pos=root_pos_t, root_quat=root_quat_t,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            source_kind=BONES, skill_id=-1, wrap=wrap,
            steady=True, steady_vel=steady_vel,
            contact=(torch.as_tensor(contact, dtype=torch.uint8, device=device)
                     if contact is not None else None),
            name=f"clean:{skill}:{stem}",
        ))
        if limit is not None and len(sources) >= limit:
            break
    return sources
