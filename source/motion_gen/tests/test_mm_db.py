# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the motion database: self-match, velocity-bias steering, candidate mask."""

from __future__ import annotations

import os

import pytest
import torch

from motion_gen import features as F
from motion_gen.ingest.motion_db import BONES, TRAJOPT, MotionDB, MotionSource

_HAVE_MJCF = os.path.isfile(F.DEFAULT_MJCF)
pytestmark = pytest.mark.skipif(not _HAVE_MJCF, reason="G1 MJCF not found")

DT = 0.02


def _make_source(Fi: int, vx: float, seed: int, kind=TRAJOPT, skill_id=0, name="") -> MotionSource:
    """Synthetic motion: smooth sinusoidal joints + constant forward root velocity."""
    g = torch.Generator().manual_seed(seed)
    nj = len(F.G1_JOINT_NAMES)
    tt = (torch.arange(Fi).float() * DT).unsqueeze(1)          # [Fi,1] seconds
    phase = (torch.rand(nj, generator=g) * 6.283).unsqueeze(0)  # [1,nj]
    w = (0.5 + torch.rand(nj, generator=g)).unsqueeze(0) * 6.283  # [1,nj] rad/s
    amp = 0.2
    joint_pos = amp * torch.sin(w * tt + phase)                # [Fi,nj]
    joint_vel = amp * w * torch.cos(w * tt + phase)
    root_pos = torch.zeros(Fi, 3); root_pos[:, 0] = vx * tt.squeeze(1); root_pos[:, 2] = 0.8
    root_quat = torch.zeros(Fi, 4); root_quat[:, 0] = 1.0      # identity wxyz
    root_lin_vel = torch.zeros(Fi, 3); root_lin_vel[:, 0] = vx
    root_ang_vel = torch.zeros(Fi, 3)
    return MotionSource(joint_pos, joint_vel, root_pos, root_quat, root_lin_vel, root_ang_vel,
                        source_kind=kind, skill_id=skill_id, name=name)


def _build_db():
    cfg = F.FeatureConfig(dt=DT)
    fk = F.make_fk(cfg, dtype=torch.float32, device="cpu")
    a = _make_source(60, vx=1.0, seed=1, skill_id=0, name="A")   # slow
    b = _make_source(60, vx=3.0, seed=1, skill_id=1, name="B")   # fast (same joint pattern)
    db = MotionDB(cfg, device="cpu").build([a, b], fk)
    return cfg, fk, db, a


def _raw_desc(cfg, fk, src: MotionSource):
    feat = F.compute_frame_features(src.joint_pos, src.joint_vel, src.root_quat,
                                    src.root_lin_vel, src.root_ang_vel, fk, cfg)
    return F.assemble_descriptors(feat, cfg, wrap=src.wrap)


def test_self_match_zero_cost():
    cfg, fk, db, a = _build_db()
    desc_a = _raw_desc(cfg, fk, a)
    f = 20  # a mid frame (full window; no edge clamping)
    idx, cost, _ = db.nn_search(desc_a[f:f + 1])
    gi = int(idx[0])
    assert int(db.owner[gi]) == 0 and int(db.local[gi]) == f      # source A, frame f
    assert float(cost[0]) < 1e-4


def test_velocity_bias_steers_winner():
    cfg, fk, db, a = _build_db()
    desc_a = _raw_desc(cfg, fk, a)
    q = desc_a[20:21]
    # No bias: closest is A (its own frame).
    idx0, _, _ = db.nn_search(q, w_vel=0.0)
    assert int(db.owner[int(idx0[0])]) == 0
    # Strong bias toward B's velocity (3,0,0): winner flips to source B.
    vel_bias = torch.tensor([[3.0, 0.0, 0.0]])
    idx1, _, _ = db.nn_search(q, vel_bias=vel_bias, w_vel=100.0)
    assert int(db.owner[int(idx1[0])]) == 1


def test_candidate_mask_restricts():
    cfg, fk, db, a = _build_db()
    desc_a = _raw_desc(cfg, fk, a)
    q = desc_a[20:21]
    mask = db.skill_id == 1                # only source B (skill 1)
    idx, _, _ = db.nn_search(q, candidate_mask=mask)
    assert int(db.owner[int(idx[0])]) == 1


def test_save_load_roundtrip(tmp_path):
    cfg, fk, db, a = _build_db()
    p = str(tmp_path / "db.npz")
    db.save(p)
    db2 = MotionDB.load(p, cfg, device="cpu")
    desc_a = _raw_desc(cfg, fk, a)
    i1, _, _ = db.nn_search(desc_a[20:21])
    i2, _, _ = db2.nn_search(desc_a[20:21])
    assert int(i1[0]) == int(i2[0])
    assert torch.allclose(db.desc_norm, db2.desc_norm, atol=1e-5)
