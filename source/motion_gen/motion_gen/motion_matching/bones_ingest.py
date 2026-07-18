# Copyright 2026 Zachary Olkin. All rights reserved.

"""Ingest the bones-seed G1 mocap dataset into motion-matching sources.

Ports the metadata filter chain + CSV parse from ``traj_opt/human_data`` (no pinocchio,
no IsaacLab) so ``motion_gen`` stays self-contained:

* :func:`load_metadata` / :func:`filter_rows` -- pick clips by package -> category ->
  style (+ movement type / body position), dropping mirrors.
* :func:`parse_csv_g1` -- seed CSV (root xyz cm, root Euler XYZ deg, 29 joint deg) ->
  root xyz (m), root quaternion **wxyz**, 29 joint angles (rad).  The Euler->matrix uses
  ``R = Rz(z) Ry(y) Rx(x)`` -- exactly pinocchio's ``rpy.rpyToMatrix(x, y, z)``, which
  traj_opt validated against foot-ground contact + pelvis uprightness.

Clips are resampled 120fps -> ``1/dt`` Hz, mapped from the 29 CSV joints to the G1-21
set, and given finite-difference velocities (same method as traj-opt ingest).  bones
clips are finite recordings, so they ingest as ``wrap="episodic"`` (edge-repeat).
"""

from __future__ import annotations

import csv as _csv
import json
import os

import h5py
import numpy as np
import torch

from . import features as F
from . import ingest_common as IC
from .motion_db import BONES, MotionSource

# 29-column seed CSV joint order (after Frame + 3 translation + 3 Euler).
CSV_JOINT_NAMES: list[str] = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
# Column index (within the 29) for each G1-21 joint.
_G1_FROM_CSV = [CSV_JOINT_NAMES.index(j) for j in F.G1_JOINT_NAMES]


# ---- metadata filtering (ported from bone_seed_selector) --------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()


def load_metadata(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(_csv.DictReader(f))


def _match(row_val: str, want) -> bool:
    """Case-insensitive match; ``want`` may be a scalar (exact) or a list/set (membership)."""
    if isinstance(want, (list, set, tuple)):
        return _norm(row_val) in {_norm(v) for v in want}
    return _norm(row_val) == _norm(want)


def filter_rows(rows, filters=None, styles=None, exclude_mirror=True,
                exclude_name_substrings=None, include_name_substrings=None) -> list[dict]:
    """Apply the package/category/... filter chain (case-insensitive).

    ``filters`` maps a metadata column to a required value; the value may be a scalar
    (exact match) or a list/set (membership, e.g. ``content_type_of_movement`` in
    ``{walking, jogging}``).

    ``include_name_substrings`` (if given) is an allow-list on the filename: a row is
    kept only if its filename contains at least one of the substrings.  This selects
    motion *families* by name (e.g. ``["jump_over_obstacle", "jump_on", "jump_off"]``
    for the terrain lead-ins) where a column filter is too coarse -- the ``Stunts``
    category is a junk-mixed bag and ``jumping`` movement is too broad.  ``exclude_
    name_substrings`` is applied after and wins on conflict.
    """
    filters = filters or {}
    allowed_styles = {_norm(s) for s in (styles or [])}
    excl_sub = [_norm(s) for s in (exclude_name_substrings or [])]
    incl_sub = [_norm(s) for s in (include_name_substrings or [])]
    out = []
    for r in rows:
        if exclude_mirror and str(r.get("is_mirror", "")).strip() == "True":
            continue
        if allowed_styles and _norm(r.get("content_uniform_style", "")) not in allowed_styles:
            continue
        if any(not _match(r.get(col, ""), val) for col, val in filters.items()):
            continue
        name = _norm(r.get("filename", ""))
        if incl_sub and not any(sub in name for sub in incl_sub):
            continue
        if any(sub in name for sub in excl_sub):
            continue
        out.append(r)
    return out


def clip_to_state(file_path: str, src_fps: float = 120.0, dt: float = 0.02,
                  device: str | torch.device = "cpu"):
    """Parse + resample + finite-difference one seed CSV -> per-frame world state tensors.

    Returns ``(joint_pos[F,21], root_pos[F,3], root_quat[F,4] wxyz, root_lin_vel[F,3],
    root_ang_vel[F,3])`` or ``None`` if the file is missing/unparseable/too short.
    """
    if not os.path.isfile(file_path):
        return None
    try:
        root_pos, root_quat, joints29 = parse_csv_g1(file_path)
    except Exception:
        return None
    rp, qq, jj = _resample(root_pos, root_quat, joints29, src_fps, dt)
    joint_pos = torch.as_tensor(jj[:, _G1_FROM_CSV], dtype=torch.float32, device=device)
    root_pos_t = torch.as_tensor(rp, dtype=torch.float32, device=device)
    root_quat_t = torch.as_tensor(qq, dtype=torch.float32, device=device)
    joint_vel = IC.central_diff(joint_pos, dt)
    root_lin_vel = IC.central_diff(root_pos_t, dt)
    root_ang_vel = IC.angular_velocity_from_quat(root_quat_t, dt)
    return joint_pos, root_pos_t, root_quat_t, root_lin_vel, root_ang_vel


# ---- CSV parse (no pinocchio) ----------------------------------------------
def _rpy_to_matrix(eul: np.ndarray) -> np.ndarray:
    """``[N,3]`` Euler XYZ (rad) -> ``[N,3,3]`` R = Rz(z) Ry(y) Rx(x) (pinocchio rpyToMatrix)."""
    x, y, z = eul[:, 0], eul[:, 1], eul[:, 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    n = eul.shape[0]
    Rx = np.zeros((n, 3, 3)); Rx[:, 0, 0] = 1; Rx[:, 1, 1] = cx; Rx[:, 1, 2] = -sx; Rx[:, 2, 1] = sx; Rx[:, 2, 2] = cx
    Ry = np.zeros((n, 3, 3)); Ry[:, 1, 1] = 1; Ry[:, 0, 0] = cy; Ry[:, 0, 2] = sy; Ry[:, 2, 0] = -sy; Ry[:, 2, 2] = cy
    Rz = np.zeros((n, 3, 3)); Rz[:, 2, 2] = 1; Rz[:, 0, 0] = cz; Rz[:, 0, 1] = -sz; Rz[:, 1, 0] = sz; Rz[:, 1, 1] = cz
    return Rz @ Ry @ Rx


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """``[N,3,3]`` rotation matrices -> ``[N,4]`` wxyz unit quaternions."""
    n = R.shape[0]
    q = np.zeros((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(n):
        m = R[i]; t = tr[i]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
        q[i] = [w, x, y, z]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def parse_csv_g1(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seed CSV -> ``(root_pos[N,3] m, root_quat_wxyz[N,4], joints29[N,29] rad)``."""
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 36:
        raise ValueError(f"expected 36 CSV columns, got {data.shape[1]} in {file_path}")
    root_pos = data[:, 1:4] / 100.0                       # cm -> m
    R = _rpy_to_matrix(np.deg2rad(data[:, 4:7]))          # root Euler XYZ deg -> R
    root_quat = _matrix_to_quat_wxyz(R)                   # wxyz
    joints29 = np.deg2rad(data[:, 7:36])
    return root_pos, root_quat, joints29


# ---- resampling ------------------------------------------------------------
def _resample(root_pos, root_quat, joints, src_fps: float, dt: float):
    """Resample to ``1/dt`` Hz (linear for pos/joints, hemisphere-fixed nlerp for quat)."""
    n = root_pos.shape[0]
    t_src = np.arange(n) / src_fps
    t_new = np.arange(0.0, t_src[-1] + 1e-9, dt)
    # hemisphere-consistent quats before interpolation (kill double-cover sign flips)
    q = root_quat.copy()
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]

    def interp_cols(a):
        return np.stack([np.interp(t_new, t_src, a[:, c]) for c in range(a.shape[1])], axis=1)

    rp = interp_cols(root_pos)
    jj = interp_cols(joints)
    qq = interp_cols(q)
    qq = qq / np.linalg.norm(qq, axis=1, keepdims=True)
    return rp, qq, jj


# ---- top-level -------------------------------------------------------------
def ingest_bones(
    dataset_dir: str,
    metadata_csv: str | None = None,
    filters: dict | None = None,
    styles: list[str] | None = None,
    exclude_mirror: bool = True,
    exclude_name_substrings: list[str] | None = None,
    src_fps: float = 120.0,
    dt: float = 0.02,
    min_frames: int = 8,
    limit: int | None = None,
    device: str | torch.device = "cpu",
) -> list[MotionSource]:
    """Filter + parse + resample bones-seed clips into :class:`MotionSource` list.

    Default filter (if ``filters`` is None) is Locomotion / neutral walking+running.
    """
    meta_path = metadata_csv or os.path.join(dataset_dir, "metadata", "seed_metadata_v004.csv")
    if filters is None:
        filters = {"package": "Locomotion"}
    if styles is None:
        styles = ["neutral"]
    rows = filter_rows(load_metadata(meta_path), filters, styles, exclude_mirror,
                       exclude_name_substrings)

    sources: list[MotionSource] = []
    for r in rows:
        rel = r.get("move_g1_path", "").strip()
        if not rel:
            continue
        st = clip_to_state(os.path.join(dataset_dir, rel), src_fps, dt, device)
        if st is None or st[0].shape[0] < min_frames:
            continue
        joint_pos, root_pos_t, root_quat_t, root_lin_vel, root_ang_vel = st
        sources.append(MotionSource(
            joint_pos=joint_pos, joint_vel=IC.central_diff(joint_pos, dt),
            root_pos=root_pos_t, root_quat=root_quat_t,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            source_kind=BONES, skill_id=-1, wrap="episodic",
            name=f"bones:{r.get('filename', '')}",
        ))
        if limit is not None and len(sources) >= limit:
            break
    return sources


# meta_<shortkey> dataset  ->  the metadata column name filter_rows expects
_CACHE_META = {
    "package": "package", "category": "category", "style": "content_uniform_style",
    "movement": "content_type_of_movement", "body": "content_body_position",
    "is_mirror": "is_mirror", "filename": "filename",
}


def load_cache(
    cache_path: str,
    filters: dict | None = None,
    styles: list[str] | None = None,
    exclude_mirror: bool = True,
    exclude_name_substrings: list[str] | None = None,
    include_name_substrings: list[str] | None = None,
    limit: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[list[MotionSource], dict]:
    """Load a precomputed bones cache, filter clips, return (sources, cache-attrs).

    Sources carry the cached per-frame features (``MotionSource.feat``) so the DB build
    skips FK entirely.  ``filters`` values may be scalars or lists (see :func:`filter_rows`),
    e.g. ``{"content_type_of_movement": ["walking", "jogging"], "content_body_position": "standing"}``.
    """
    def _maybe_json(v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v

    f = h5py.File(cache_path, "r")
    attrs = {k: _maybe_json(v) for k, v in f.attrs.items()}
    C = int(f.attrs["n_clips"])
    cols = {orig: [(s.decode() if isinstance(s, bytes) else s) for s in f[f"meta_{sk}"][:]]
            for sk, orig in _CACHE_META.items()}
    rows = [{**{orig: cols[orig][i] for orig in cols}, "_idx": i} for i in range(C)]
    kept = filter_rows(rows, filters=filters, styles=styles, exclude_mirror=exclude_mirror,
                       exclude_name_substrings=exclude_name_substrings,
                       include_name_substrings=include_name_substrings)
    if limit is not None:
        kept = kept[:limit]

    dt = json.loads(f.attrs["dt"])
    cs = f["clip_start"][:]
    cl = f["clip_len"][:]
    # Read the arrays we need once (cache fits RAM), then slice per clip.
    arrs = {k: f[k][:] for k in ("joint_pos", "root_pos", "root_quat", "root_lin_vel",
                                 "root_ang_vel", "feat")}
    f.close()

    def T(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    sources: list[MotionSource] = []
    for r in kept:
        i = r["_idx"]; a = int(cs[i]); b = a + int(cl[i])
        rq = T(arrs["root_quat"][a:b])
        rq = rq / rq.norm(dim=-1, keepdim=True).clamp_min(1e-9)   # fp16 renorm
        jp = T(arrs["joint_pos"][a:b])
        sources.append(MotionSource(
            joint_pos=jp, joint_vel=IC.central_diff(jp, dt),
            root_pos=T(arrs["root_pos"][a:b]), root_quat=rq,
            root_lin_vel=T(arrs["root_lin_vel"][a:b]), root_ang_vel=T(arrs["root_ang_vel"][a:b]),
            source_kind=BONES, skill_id=-1, wrap="episodic",
            feat=T(arrs["feat"][a:b]), name=f"bones:{r.get('filename', '')}",
        ))
    return sources, attrs
