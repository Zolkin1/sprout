# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for motion-matching features (FK parity, Jacobian, invariances, layout)."""

from __future__ import annotations

import os

import pytest
import torch

from motion_gen.math_utils import quat_apply, quat_mul
from motion_gen import features as F

_MJCF = F.DEFAULT_MJCF
_HAVE_MJCF = os.path.isfile(_MJCF)
pytestmark = pytest.mark.skipif(not _HAVE_MJCF, reason="G1 MJCF not found")


def _fk(dtype=torch.float64):
    cfg = F.FeatureConfig()
    return F.make_fk(cfg, dtype=dtype), cfg


def test_fk_matches_mujoco_feet_and_palms():
    """FKChain EE positions match mujoco.mj_kinematics for feet + palms."""
    mujoco = pytest.importorskip("mujoco")
    fk, cfg = _fk()
    frames = cfg.frames()
    m = mujoco.MjModel.from_xml_path(_MJCF)
    d = mujoco.MjData(m)
    jadr = [int(m.joint(n).qposadr[0]) for n in F.G1_JOINT_NAMES]
    pelvis = int(m.body("pelvis_link").id)
    bids = [int(m.body(f).id) for f in frames]

    torch.manual_seed(0)
    B, nj = 5, len(F.G1_JOINT_NAMES)
    q = (torch.rand(B, nj, dtype=torch.float64) - 0.5) * 1.5
    pos, rot, _, _ = fk.pose_and_jacobian(q)

    for b in range(B):
        d.qpos[:] = 0.0
        for k, a in enumerate(jadr):
            d.qpos[a] = float(q[b, k])
        mujoco.mj_kinematics(m, d)
        p_pel = d.xpos[pelvis].copy()
        R_pel = d.xmat[pelvis].reshape(3, 3).copy()
        for f_i, bid in enumerate(bids):
            # mujoco world EE -> pelvis frame
            ee_pel = R_pel.T @ (d.xpos[bid] - p_pel)
            err = (pos[b, f_i].numpy() - ee_pel)
            assert (abs(err) < 1e-5).all(), f"frame {frames[f_i]} b{b} err {err}"


def test_jacobian_matches_finite_difference():
    """FK linear Jacobian jv matches finite-difference d(pos)/d(q)."""
    fk, _ = _fk()
    torch.manual_seed(1)
    nj = len(F.G1_JOINT_NAMES)
    q0 = (torch.rand(1, nj, dtype=torch.float64) - 0.5)
    pos0, _, jv, _ = fk.pose_and_jacobian(q0)
    eps = 1e-6
    for j in range(nj):
        dq = torch.zeros(1, nj, dtype=torch.float64)
        dq[0, j] = eps
        pos_p, *_ = fk.pose_and_jacobian(q0 + dq)
        fd = (pos_p - pos0) / eps          # [1,K,3]
        assert (abs(fd[0] - jv[0, :, :, j]) < 1e-4).all(), f"jv mismatch joint {j}"


def test_feature_dims_and_determinism():
    fk, cfg = _fk(dtype=torch.float32)
    nj = len(F.G1_JOINT_NAMES)
    torch.manual_seed(2)
    q = torch.rand(7, nj)
    dq = torch.rand(7, nj)
    quat = torch.randn(7, 4)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    vlin = torch.randn(7, 3)
    vang = torch.randn(7, 3)
    feat = F.compute_frame_features(q, dq, quat, vlin, vang, fk, cfg)
    assert feat.shape == (7, F.per_frame_feature_dim(cfg))
    feat2 = F.compute_frame_features(q, dq, quat, vlin, vang, fk, cfg)
    assert torch.allclose(feat, feat2)
    # descriptor assembly: every frame gets a full-window descriptor
    desc = F.assemble_descriptors(feat, cfg, wrap="episodic")
    assert desc.shape == (7, F.descriptor_dim(cfg))


def test_offset_index_wrap_vs_clamp():
    """Periodic wraps modulo n; episodic clamps to the ends."""
    offs = [-2, -1, 0, 1, 2]
    n = 5
    wrap = F.offset_index(n, offs, "periodic")
    clamp = F.offset_index(n, offs, "episodic")
    # frame 0: periodic sees [-2,-1] -> [3,4]; episodic clamps -> [0,0]
    assert wrap[0].tolist() == [3, 4, 0, 1, 2]
    assert clamp[0].tolist() == [0, 0, 0, 1, 2]
    # last frame 4: periodic [+1,+2] -> [0,1]; episodic clamps -> [4,4]
    assert wrap[4].tolist() == [2, 3, 4, 0, 1]
    assert clamp[4].tolist() == [2, 3, 4, 4, 4]


def test_heading_and_translation_invariance():
    """Per-frame feature is invariant to world heading (yaw) and world position."""
    fk, cfg = _fk(dtype=torch.float64)
    nj = len(F.G1_JOINT_NAMES)
    torch.manual_seed(3)
    q = torch.rand(4, nj, dtype=torch.float64)
    dq = torch.rand(4, nj, dtype=torch.float64)
    quat = torch.randn(4, 4, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    vlin = torch.randn(4, 3, dtype=torch.float64)
    vang = torch.randn(4, 3, dtype=torch.float64)
    feat1 = F.compute_frame_features(q, dq, quat, vlin, vang, fk, cfg)

    # Apply an arbitrary world-frame yaw (about +z): premultiply root quat, rotate
    # world velocities the same way.  The yaw-invariant feature must not change.
    ang = torch.tensor([0.7, -1.3, 2.5, 0.2], dtype=torch.float64)
    dyaw = torch.stack([torch.cos(ang / 2), torch.zeros_like(ang), torch.zeros_like(ang), torch.sin(ang / 2)], dim=-1)
    quat2 = quat_mul(dyaw, quat)
    vlin2 = quat_apply(dyaw, vlin)
    vang2 = quat_apply(dyaw, vang)
    feat2 = F.compute_frame_features(q, dq, quat2, vlin2, vang2, fk, cfg)
    assert torch.allclose(feat1, feat2, atol=1e-9), (feat1 - feat2).abs().max()
