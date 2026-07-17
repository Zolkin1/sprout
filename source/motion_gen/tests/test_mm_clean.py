# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the clean steady-state trajectory ingest (``clean_ingest.py``).

Self-contained: builds a tiny synthetic library (npz + manifest.csv) mirroring the real
``trajectories/skills/library_v1`` schema (MuJoCo g1_29dof ``state[T,71]`` at sim_dt, with a
``time`` array and json ``meta``), then checks the adapter: STEADY tagging, periodic wrap,
manifest steady velocity, sim_dt inferred from ``time`` (not node_dt), the 29->21 joint map,
and finite-difference body velocity matching the label.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from motion_gen.motion_matching.bones_ingest import _G1_FROM_CSV
from motion_gen.motion_matching.clean_ingest import ingest_clean_library

_SIM_DT = 0.005
_Z = 0.78


def _make_npz(path, *, T, vx, vy, wz):
    """Write a synthetic library .npz: state[T,71] qpos+qvel, time[T], meta json.

    Root advances at body velocity (vx, vy, wz) in the world; joints are held at a
    recognizable per-column pattern (column k == k) so the 29->21 extraction is exact.
    """
    t = np.arange(T) * _SIM_DT
    yaw = wz * t
    # integrate body velocity into world position
    x = np.zeros(T); y = np.zeros(T)
    for i in range(1, T):
        c, s = np.cos(yaw[i - 1]), np.sin(yaw[i - 1])
        x[i] = x[i - 1] + (c * vx - s * vy) * _SIM_DT
        y[i] = y[i - 1] + (s * vx + c * vy) * _SIM_DT
    root_pos = np.stack([x, y, np.full(T, _Z)], axis=1)
    root_quat = np.stack([np.cos(yaw / 2), np.zeros(T), np.zeros(T), np.sin(yaw / 2)], axis=1)
    joints29 = np.tile(np.arange(29, dtype=np.float64) * 0.01, (T, 1))   # col k == 0.01*k
    qvel = np.zeros((T, 35))                                             # ignored by the adapter
    state = np.concatenate([root_pos, root_quat, joints29, qvel], axis=1)
    assert state.shape == (T, 71)
    meta = json.dumps({"skill": "synthetic", "achieved_twist": [vx, vy, wz], "periodic": True})
    np.savez(path, state=state, time=t, meta=meta, node_dt=np.float64(0.04))


def _make_library(root):
    root = str(root)
    import os
    rows = [
        # skill, vel_x, vel_y, vel_yaw, periodic, T
        ("walk_forward", 0.8, 0.0, 0.0, True, 417),
        ("walk_turn", 0.4, 0.0, 0.5, True, 401),
        ("run_forward", 2.4, 0.0, 0.0, True, 137),      # short stride
    ]
    manifest = []
    for skill, vx, vy, wz, per, T in rows:
        d = os.path.join(root, skill)
        os.makedirs(d, exist_ok=True)
        fname = f"{skill}/{skill}_x{int(vx*1000):04d}.npz"
        _make_npz(os.path.join(root, fname), T=T, vx=vx, vy=vy, wz=wz)
        manifest.append(dict(skill=skill, vel_x=vx, vel_y=vy, vel_yaw=wz,
                             periodic=per, status="ok", clip="synthetic.csv", file=fname))
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        for m in manifest:
            w.writerow(m)
    return root, {m["skill"]: m for m in manifest}


def test_ingest_basic(tmp_path):
    root, man = _make_library(tmp_path)
    dt = 0.02
    srcs = ingest_clean_library(root, hf_repo=None, dt=dt, device="cpu")
    assert len(srcs) == 3
    by = {s.name.split(":")[1]: s for s in srcs}          # skill from "clean:<skill>:<stem>"
    assert set(by) == {"walk_forward", "walk_turn", "run_forward"}

    for skill, s in by.items():
        m = man[skill]
        assert s.wrap == "periodic"
        assert s.steady is True
        assert s.steady_vel is not None
        assert torch.allclose(
            s.steady_vel, torch.tensor([m["vel_x"], m["vel_y"], m["vel_yaw"]]), atol=1e-5)
        # resampled length ~ span * (1/dt); span = (T-1)*sim_dt
        span = (int({"walk_forward": 417, "walk_turn": 401, "run_forward": 137}[skill]) - 1) * _SIM_DT
        assert abs(s.n - round(span / dt)) <= 1
        assert s.joint_pos.shape[1] == 21 and s.root_quat.shape[1] == 4


def test_joint_map_and_velocity(tmp_path):
    root, _ = _make_library(tmp_path)
    srcs = ingest_clean_library(root, hf_repo=None, dt=0.02, device="cpu")
    wf = next(s for s in srcs if "walk_forward" in s.name)
    # joints held at col k == 0.01*k -> extracted 21 == 0.01 * _G1_FROM_CSV (exact after resample)
    expect = torch.tensor(_G1_FROM_CSV, dtype=torch.float32) * 0.01
    assert torch.allclose(wf.joint_pos[wf.n // 2], expect, atol=1e-5)
    # finite-difference body velocity (yaw-planar) matches the manifest label (0.8, 0, 0)
    from motion_gen.motion_matching.motion_db import _yaw_planar_vel
    pv = _yaw_planar_vel(wf.root_quat, wf.root_lin_vel, wf.root_ang_vel)
    interior = pv[2:-2].mean(0)                            # drop FD edge frames
    assert torch.allclose(interior, torch.tensor([0.8, 0.0, 0.0]), atol=2e-2)


def test_skills_filter_and_limit(tmp_path):
    root, _ = _make_library(tmp_path)
    only = ingest_clean_library(root, hf_repo=None, dt=0.02, device="cpu", skills=["walk_turn"])
    assert len(only) == 1 and "walk_turn" in only[0].name
    capped = ingest_clean_library(root, hf_repo=None, dt=0.02, device="cpu", limit=2)
    assert len(capped) == 2
