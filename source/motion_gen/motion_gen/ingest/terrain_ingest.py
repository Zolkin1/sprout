# Copyright 2026 Zachary Olkin. All rights reserved.

"""Ingest the terrain traj-opt library (box / stair interactions) into motion sources.

The terrain library (e.g. HuggingFace ``trajectories/skills/terrain_v1``) is the same
clean/traj-opt ``.npz`` format as :mod:`.clean_ingest` (``state[T,71]`` + ``time`` +
``meta``) but each clip is a **terrain interaction** the box-clip generator will PLAY
(a box jump once, a stair stride looped), not a flat stride to stitch.  On top of the
clean fields these clips carry the extra info the terrain pipeline needs, which the
plain :func:`clean_ingest.ingest_clean_library` drops:

* ``contact_flags[T, C]`` + ``contact_bodies`` -- per-body in-contact schedule (the two
  ankle bodies are the feet; wrists carry hand support for vault clips).  Resampled onto
  the clip's ``dt`` frames and attached as :attr:`MotionSource.contact` (foot columns).
* ``meta.terrain`` -- the geometry the clip was solved on (box ``{pos, size, yaw}`` /
  stair ``riser, tread, n_steps, boxes``).  MuJoCo box ``size`` is **half-extents**; we
  convert to full extents so it drops onto the full-extent ``OverlayVolumeCfg``.
* ``meta.provenance`` -- the bones source clip + crop, so the approach DP knows which
  bones lead-up (frames before the crop) hands off into the played interior.

Velocity is body/heading frame (matcher descriptors are yaw-aligned), so the clip's
stored world heading is a non-issue for matching; ``heading_yaw0`` is recorded so the
geometry can be yaw-normalized WITH the motion when the segment is spliced.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from . import ingest_common as IC
from .clean_ingest import _clip_to_state, _read_manifest, _resolve_dir, _truthy
from .motion_db import BONES, MotionSource

_ANKLE_KEY = "ankle_roll_link"   # contact body suffix identifying a foot


def _yaw_from_quat_wxyz(q: torch.Tensor) -> float:
    """Yaw angle (rad) about world z from a single ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _resample_flags_to(n_out: int, flags: np.ndarray) -> np.ndarray:
    """Nearest-resample a ``[T_src, C]`` flag array to ``n_out`` frames.

    Binary contact -> nearest (not linear).  ``linspace`` over source indices matches the
    uniform state resample :func:`clean_ingest._clip_to_state` applies (both map target
    frame ``k`` to source time ``k * dt``).
    """
    ts = flags.shape[0]
    idx = np.clip(np.round(np.linspace(0, ts - 1, n_out)).astype(int), 0, ts - 1)
    return flags[idx]


def _norm_terrain_geom(terr: dict) -> dict:
    """Copy ``meta.terrain``, converting MuJoCo half-extent box sizes to FULL extents."""
    out = dict(terr)
    out["boxes"] = [
        {"pos": [float(v) for v in b["pos"]],
         "size": [2.0 * float(v) for v in b["size"]],   # half-extent -> full extent
         "yaw": float(b.get("yaw", 0.0))}
        for b in terr.get("boxes", [])
    ]
    return out


def ingest_terrain_library(
    path: str,
    hf_repo: str | None = None,
    dt: float = 0.02,
    device: str | torch.device = "cpu",
    skills: list[str] | None = None,
    limit: int | None = None,
    min_frames: int = 8,
) -> list[MotionSource]:
    """Load the terrain interaction library into STEADY-tier :class:`MotionSource` list.

    Each source carries :attr:`~MotionSource.contact` (foot in-contact flags ``[Fi, 2]``),
    :attr:`~MotionSource.terrain_meta` (normalized geometry + ``contact_bodies`` +
    ``heading_yaw0`` + command vel), and :attr:`~MotionSource.provenance` (bones link).

    Args mirror :func:`clean_ingest.ingest_clean_library`.
    """
    lib_dir = _resolve_dir(path, hf_repo)
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"terrain library directory not found: {lib_dir}")
    allow = set(skills) if skills is not None else None

    sources: list[MotionSource] = []
    for r in _read_manifest(lib_dir):
        skill = (r.get("skill") or "").strip()
        if allow is not None and skill not in allow:
            continue
        rel = (r.get("file") or "").strip()
        if not rel:
            continue
        npz_path = str(lib_dir / rel)
        st = _clip_to_state(npz_path, dt, device)
        if st is None or st[0].shape[0] < min_frames:
            continue
        joint_pos, root_pos_t, root_quat_t, root_lin_vel, root_ang_vel = st
        F = int(joint_pos.shape[0])

        z = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(z["meta"])) if "meta" in z else {}

        contact = None
        bodies = None
        if "contact_flags" in z and "contact_bodies" in z:
            flags = np.asarray(z["contact_flags"])
            bodies = [str(x).split("/")[-1] for x in np.atleast_1d(z["contact_bodies"])]
            foot_cols = [i for i, b in enumerate(bodies) if b.endswith(_ANKLE_KEY)]
            if foot_cols:
                fr = _resample_flags_to(F, flags[:, foot_cols])
                contact = torch.as_tensor((fr > 0).astype(np.uint8), dtype=torch.uint8, device=device)

        try:
            steady_vel = torch.tensor(
                [float(r["vel_x"]), float(r["vel_y"]), float(r["vel_yaw"])],
                dtype=torch.float32, device=device)
        except (KeyError, ValueError, TypeError):
            steady_vel = None

        terrain_meta = {
            "terrain": _norm_terrain_geom(meta.get("terrain", {})),
            "contact_bodies": bodies,
            "heading_yaw0": _yaw_from_quat_wxyz(root_quat_t[0]),
            "vel_cmd": None if steady_vel is None else steady_vel.tolist(),
            "skill": skill,
        }
        wrap = "periodic" if _truthy(r.get("periodic", "true")) else "episodic"
        stem = os.path.splitext(os.path.basename(rel))[0]
        sources.append(MotionSource(
            joint_pos=joint_pos, joint_vel=IC.central_diff(joint_pos, dt),
            root_pos=root_pos_t, root_quat=root_quat_t,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            source_kind=BONES, skill_id=-1, wrap=wrap,
            steady=True, steady_vel=steady_vel,
            contact=contact, terrain_meta=terrain_meta, provenance=meta.get("provenance"),
            name=f"terrain:{skill}:{stem}",
        ))
        if limit is not None and len(sources) >= limit:
            break
    return sources
