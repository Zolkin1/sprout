# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator-free unit tests for the flow-matching subpackage.

Run with: ``pytest source/robot_rl/robot_rl/flow_matching/tests`` inside the
``env_isaaclab`` venv. All tests use synthetic tensors; no Isaac Sim app needed.

Covers the robot-state-conditioned paradigm: an **xyz+yaw** relative frame, a single
transformer over ``[history(N) | height(1) | cmd(Ncmd) | future(F)]`` (no inpainting /
known-mask; a learned null-history embedding), snap-only dataset windows, and the online
inference round-trip.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_rl.flow_matching import (
    FKChain,
    FlatTerrain,
    FlowMatchingConfig,
    FlowMatchingTransformer,
    Normalizer,
    box_sole_points,
    build_train_val,
    fk_consistency_terms,
    flow_matching_loss,
    frame_block_start,
    frame_consistency_terms,
    joint_vel_consistency_term,
    matrix_from_quat,
    matrix_to_rot6d,
    orientation_slice,
    orientation_slices,
    quat_to_rot6d,
    relative_state,
    relative_state_from_window,
    rot6d_to_matrix,
    rot6d_to_quat,
    sample_flow_batch,
    sample_trajectory,
    sample_trajectory_grad,
    snap_windows,
    so3_log,
    state_to_absolute,
    target_state_dim,
    terrain_penetration_term,
    yaw_quat,
)
from robot_rl.flow_matching.online import (
    build_cmd_tokens,
    build_height_token,
    build_history_tokens,
    load_fm_context,
    sample_world,
)


def _rand_unit_quat(n: int) -> torch.Tensor:
    """Random unit quaternions (xyzw), shape (n, 4)."""
    q = torch.randn(n, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _norm(x: torch.Tensor) -> Normalizer:
    """Standardize-everything normalizer fit on ``x`` (last-dim width)."""
    return Normalizer.fit(x.reshape(-1, x.shape[-1]), torch.ones(x.shape[-1], dtype=torch.bool))


# ---------------------------------------------------------------------------
# 1. 6D rotation round-trip + Gram-Schmidt orthonormality
# ---------------------------------------------------------------------------
def test_rot6d_roundtrip_matrices():
    torch.manual_seed(0)
    q = _rand_unit_quat(64)
    r = matrix_from_quat(q)
    assert torch.allclose(r, rot6d_to_matrix(matrix_to_rot6d(r)), atol=1e-6)
    r3 = matrix_from_quat(rot6d_to_quat(quat_to_rot6d(q)))
    assert torch.allclose(r, r3, atol=1e-5)


def test_rot6d_orthonormal_from_garbage():
    torch.manual_seed(1)
    d6 = torch.randn(100, 6, dtype=torch.float64)
    r = rot6d_to_matrix(d6)
    eye = torch.eye(3, dtype=torch.float64).expand_as(r)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-6)
    assert torch.allclose(torch.linalg.det(r), torch.ones(100, dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. xyz+yaw relative-frame invertibility
# ---------------------------------------------------------------------------
def test_relative_state_invertible():
    torch.manual_seed(2)
    nj, w = 21, 50
    p, q = torch.randn(w, 3, dtype=torch.float64), _rand_unit_quat(w)
    v, om = torch.randn(w, 3, dtype=torch.float64), torch.randn(w, 3, dtype=torch.float64)
    jp, jv = torch.randn(w, nj, dtype=torch.float64), torch.randn(w, nj, dtype=torch.float64)
    ref = w // 2
    rel = relative_state_from_window(p, q, v, om, jp, jv, ref_index=ref)
    assert rel.shape == (w, target_state_dim(nj))
    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rec = state_to_absolute(rel, p_ref, q_ref, nj)
    assert torch.allclose(rec["joint_pos"], jp, atol=1e-6)
    assert torch.allclose(rec["root_pos_w"], p, atol=1e-5)
    assert torch.allclose(rec["root_link_lin_vel_w"], v, atol=1e-5)
    assert torch.allclose(rec["root_link_ang_vel_w"], om, atol=1e-5)
    assert torch.allclose(matrix_from_quat(rec["root_quat_w"]), matrix_from_quat(q), atol=1e-5)


def test_relative_state_invertible_with_frames():
    torch.manual_seed(12)
    nj, w, k = 21, 30, 4
    p, q = torch.randn(w, 3, dtype=torch.float64), _rand_unit_quat(w)
    v, om = torch.randn(w, 3, dtype=torch.float64), torch.randn(w, 3, dtype=torch.float64)
    jp, jv = torch.randn(w, nj, dtype=torch.float64), torch.randn(w, nj, dtype=torch.float64)
    fpos = torch.randn(w, k, 3, dtype=torch.float64)
    fquat = _rand_unit_quat(w * k).reshape(w, k, 4)
    flin, fang = torch.randn(w, k, 3, dtype=torch.float64), torch.randn(w, k, 3, dtype=torch.float64)
    ref = w // 2
    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rel = relative_state(p, q, v, om, jp, jv, p_ref, q_ref,
                         frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang)
    assert rel.shape == (w, target_state_dim(nj, k))
    assert len(orientation_slices(nj, k)) == k + 1
    rec = state_to_absolute(rel, p_ref, q_ref, nj, num_extra_frames=k)
    assert torch.allclose(rec["frame_pos_w"], fpos, atol=1e-5)
    assert torch.allclose(rec["frame_lin_vel_w"], flin, atol=1e-5)
    assert torch.allclose(
        matrix_from_quat(rec["frame_quat_w"].reshape(w * k, 4)),
        matrix_from_quat(fquat.reshape(w * k, 4)), atol=1e-5,
    )


def test_relative_frame_is_yaw_only():
    """At the anchor, the root pos is ~0 and only the anchor YAW is removed (roll/pitch kept)."""
    torch.manual_seed(3)
    nj, w = 21, 10
    p, q = torch.randn(w, 3, dtype=torch.float64), _rand_unit_quat(w)
    v, om = torch.randn(w, 3, dtype=torch.float64), torch.randn(w, 3, dtype=torch.float64)
    jp, jv = torch.randn(w, nj, dtype=torch.float64), torch.randn(w, nj, dtype=torch.float64)
    ref = 7
    rel = relative_state_from_window(p, q, v, om, jp, jv, ref_index=ref)
    o0, o1 = orientation_slice(nj)
    # anchor root position is exactly 0 (xyz, incl. z, subtracted)
    assert torch.allclose(rel[ref, 2 * nj : 2 * nj + 3], torch.zeros(3, dtype=torch.float64), atol=1e-6)
    # the relative ori at the anchor = yaw_quat(q_ref)^-1 * q_ref -> keeps roll/pitch, removes yaw.
    q_rel_anchor = rot6d_to_quat(rel[ref, o0:o1].unsqueeze(0))
    expected = matrix_from_quat(yaw_quat(q[ref : ref + 1]).transpose(0, 0) if False else yaw_quat(q[ref : ref + 1]))
    # compare as matrices: R_rel == R_yaw^T R_anchor
    R_yaw = matrix_from_quat(yaw_quat(q[ref : ref + 1]))
    R_anchor = matrix_from_quat(q[ref : ref + 1])
    R_expected = R_yaw.transpose(-1, -2) @ R_anchor
    assert torch.allclose(matrix_from_quat(q_rel_anchor), R_expected, atol=1e-6)


def test_velocity_frame_pure_yaw_rotation():
    torch.manual_seed(4)
    nj, w = 5, 4
    p = torch.zeros(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    v = torch.randn(w, 3, dtype=torch.float64)
    om = torch.zeros(w, 3, dtype=torch.float64)
    jp = jv = torch.zeros(w, nj, dtype=torch.float64)
    ref = 0
    rel = relative_state_from_window(p, q, v, om, jp, jv, ref_index=ref)
    v_rel = rel[:, 2 * nj + 3 : 2 * nj + 6]
    r_yaw = matrix_from_quat(yaw_quat(q[ref : ref + 1]).expand(w, -1))
    expected = (r_yaw.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(v_rel, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Normalizer
# ---------------------------------------------------------------------------
def test_normalizer_mask_and_inverse():
    torch.manual_seed(5)
    d = 10
    x = torch.randn(2000, d) * 5.0 + 3.0
    x[:, 4] = 2.0
    mask = torch.ones(d, dtype=torch.bool)
    mask[7:9] = False
    norm = Normalizer.fit(x, mask)
    z = norm.normalize(x)
    std_dims = [i for i in range(d) if mask[i] and i != 4]
    assert torch.allclose(z[:, std_dims].mean(0), torch.zeros(len(std_dims)), atol=1e-5)
    assert torch.allclose(z[:, std_dims].std(0), torch.ones(len(std_dims)), atol=1e-2)
    assert torch.allclose(z[:, 7:9], x[:, 7:9], atol=1e-6)
    assert torch.allclose(norm.denormalize(z), x, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Flow batch math (no mask -- the whole F window follows noise->data)
# ---------------------------------------------------------------------------
def test_flow_batch_no_mask():
    torch.manual_seed(6)
    b, f, d = 8, 10, 57
    x1 = torch.randn(b, f, d)
    x_tau, tau, target_v = sample_flow_batch(x1)
    assert x_tau.shape == x1.shape and target_v.shape == x1.shape and tau.shape == (b,)
    x0 = x1 - target_v
    t = tau.view(-1, 1, 1)
    assert torch.allclose(x_tau, (1 - t) * x0 + t * x1, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Model: token layout, conditioning, learned null-history
# ---------------------------------------------------------------------------
def _model(d=57, f=8, n=2, cmd_dim=3, stride=5, **kw):
    return FlowMatchingTransformer(
        state_dim=d, future_len=f, num_history_states=n, history_dim=d, height_dim=1,
        cmd_dim=cmd_dim, cmd_stride=stride, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, **kw
    ).eval()


def test_model_shapes_and_conditioning():
    torch.manual_seed(7)
    b, f, d, n, cmd_dim, stride = 3, 8, 57, 2, 3, 5
    model = _model(d, f, n, cmd_dim, stride)
    x, tau = torch.randn(b, f, d), torch.rand(b)
    hist, height = torch.randn(b, n, d), torch.randn(b, 1)
    cmd = torch.randn(b, len(range(0, f, stride)), cmd_dim)
    with torch.no_grad():
        v0 = model(x, tau, hist, height, cmd)
        assert v0.shape == (b, f, d)
        # history conditions the output
        v_h = model(x, tau, hist + 3.0, height, cmd)
    assert not torch.allclose(v0, v_h, atol=1e-5)


def test_model_null_history():
    torch.manual_seed(8)
    b, f, d, n = 4, 8, 57, 2
    model = _model(d, f, n, cmd_dim=0)
    x, tau, height = torch.randn(b, f, d), torch.rand(b), torch.randn(b, 1)
    hist = torch.randn(b, n, d)
    mask = torch.ones(b, dtype=torch.bool)
    with torch.no_grad():
        a = model(x, tau, hist, height, None, history_null_mask=mask)
        bb = model(x, tau, hist + 5.0, height, None, history_null_mask=mask)  # history nulled -> ignored
        cond = model(x, tau, hist, height, None)
    assert torch.allclose(a, bb, atol=1e-6)        # null ignores history content
    assert not torch.allclose(a, cond, atol=1e-5)  # null differs from conditioned


def test_model_no_command():
    model = _model(d=8, f=6, n=2, cmd_dim=0)
    with torch.no_grad():
        out = model(torch.randn(2, 6, 8), torch.rand(2), torch.randn(2, 2, 8), torch.randn(2, 1), None)
    assert out.shape == (2, 6, 8)


# ---------------------------------------------------------------------------
# 6. Flow loss + samplers (no inpainting)
# ---------------------------------------------------------------------------
def test_flow_loss_and_samplers():
    torch.manual_seed(9)
    b, f, d, n, cmd_dim, stride = 8, 8, 57, 2, 3, 5
    model = _model(d, f, n, cmd_dim, stride)
    x1, hist, height = torch.randn(b, f, d), torch.randn(b, n, d), torch.randn(b, 1)
    cmd = torch.randn(b, len(range(0, f, stride)), cmd_dim)
    null = torch.rand(b) < 0.25
    out = flow_matching_loss(model, x1, hist, height, cmd, history_null_mask=null)
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()
    assert model.null_history.grad is not None
    with torch.no_grad():
        traj = sample_trajectory(model, hist, height, cmd, f, d, num_steps=4)
        assert traj.shape == (b, f, d) and torch.isfinite(traj).all()
    g = sample_trajectory_grad(model, hist, height, cmd, f, d, num_steps=4, grad_steps=1)
    assert g.shape == (b, f, d) and g.requires_grad


# ---------------------------------------------------------------------------
# 7. SO(3) log + finite-difference consistency terms
# ---------------------------------------------------------------------------
def _rodrigues(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(B, 3, 3)`` from rotation vectors ``w`` (B, 3) (Rodrigues)."""
    theta = w.norm(dim=-1, keepdim=True)
    k = w / theta.clamp_min(1e-12)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2] = -kz, ky
    K[:, 1, 0], K[:, 1, 2] = kz, -kx
    K[:, 2, 0], K[:, 2, 1] = -ky, kx
    eye = torch.eye(3, dtype=w.dtype).expand(w.shape[0], 3, 3)
    th = theta[:, :, None]
    return eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def test_so3_log_recovers_rotation_vector():
    torch.manual_seed(20)
    dirs = torch.randn(200, 3, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    angles = torch.cat([torch.full((100,), 1e-7, dtype=torch.float64),
                        torch.rand(100, dtype=torch.float64) * 2.5])
    w = dirs * angles.unsqueeze(-1)
    rec = so3_log(_rodrigues(w))
    assert torch.allclose(rec, w, atol=1e-7)


def _consistent_frame_traj(nf: int, nj: int, k: int, dt: float, seed: int) -> torch.Tensor:
    """Self-consistent ``(1, F, D)`` traj: pos integrates lin vel, ori integrates ang vel."""
    g = torch.Generator().manual_seed(seed)
    d = target_state_dim(nj, k)
    x = torch.zeros(1, nf, d, dtype=torch.float64)
    for fi in range(1 + k):
        b = frame_block_start(nj, fi)
        pos0 = torch.randn(3, generator=g, dtype=torch.float64)
        vel = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
        omega = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
        for t in range(nf):
            r = _rodrigues((omega * (t * dt)).unsqueeze(0))[0]
            x[0, t, b:b + 3] = pos0 + vel * (t * dt)
            x[0, t, b + 3:b + 6] = vel
            x[0, t, b + 6:b + 12] = matrix_to_rot6d(r)
            x[0, t, b + 12:b + 15] = omega
    return x


def test_frame_consistency_zero_for_consistent_traj():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=1)
    vel, accel = frame_consistency_terms(x, x, dt, nj, k)
    jvel = joint_vel_consistency_term(x, dt, nj)
    assert vel.max().item() < 1e-10 and accel.max().item() < 1e-10 and jvel.max().item() < 1e-10


def test_frame_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=2)
    pred = x.clone()
    pred[0, 3, frame_block_start(nj, 0):frame_block_start(nj, 0) + 3] += 1.0
    vel, _ = frame_consistency_terms(pred, x, dt, nj, k)
    assert vel.max().item() > 1e-2


def test_joint_vel_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=3)
    pred = x.clone()
    pred[0, 3, 0:nj] += 0.5
    assert joint_vel_consistency_term(pred, dt, nj).max().item() > 1e-2


# ---------------------------------------------------------------------------
# 8. Forward kinematics (skipif no MJCF)
# ---------------------------------------------------------------------------
_FK_MJCF = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    "transfer", "sim", "robots", "g1", "g1_21j.xml",
)
_FK_JOINTS = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
)
_FK_FRAMES = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]


def _have_fk_model() -> bool:
    return os.path.isfile(_FK_MJCF)


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _FK_FRAMES, base_link="pelvis_link", dtype=torch.float64)
    m = mujoco.MjModel.from_xml_path(_FK_MJCF)
    d = mujoco.MjData(m)
    jadr = [int(m.joint(n).qposadr[0]) for n in _FK_JOINTS]
    bids = [int(m.body(f).id) for f in _FK_FRAMES]
    rng = np.random.default_rng(0)
    b, nj = 6, len(_FK_JOINTS)
    qs = rng.uniform(-0.6, 0.6, size=(b, nj))
    mj_pos = np.zeros((b, len(_FK_FRAMES), 3))
    for i in range(b):
        d.qpos[:] = 0.0
        d.qpos[3:7] = (1, 0, 0, 0)
        for a, j in zip(jadr, qs[i]):
            d.qpos[a] = j
        mujoco.mj_kinematics(m, d)
        for kf, bid in enumerate(bids):
            mj_pos[i, kf] = d.xpos[bid]
    pos, _, _, _ = chain.pose_and_jacobian(torch.tensor(qs, dtype=torch.float64))
    assert np.abs(pos.numpy() - mj_pos).max() < 1e-6


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_fk_consistency_zero_for_fk_consistent_state():
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _FK_FRAMES, base_link="pelvis_link", dtype=torch.float64)
    nj, k = len(_FK_JOINTS), len(_FK_FRAMES)
    d, nf = target_state_dim(nj, k), 4
    rng = np.random.default_rng(1)
    q = torch.tensor(rng.uniform(-0.5, 0.5, size=(1, nf, nj)), dtype=torch.float64)
    qd = torch.tensor(rng.standard_normal((1, nf, nj)) * 0.3, dtype=torch.float64)
    fk_pos, fk_rot, jv, jw = chain.pose_and_jacobian(q.reshape(nf, nj))
    fk_pos, fk_rot = fk_pos.reshape(1, nf, k, 3), fk_rot.reshape(1, nf, k, 3, 3)
    v_fk = torch.einsum("mkij,mj->mki", jv, qd.reshape(nf, nj)).reshape(1, nf, k, 3)
    w_fk = torch.einsum("mkij,mj->mki", jw, qd.reshape(nf, nj)).reshape(1, nf, k, 3)
    x = torch.zeros(1, nf, d, dtype=torch.float64)
    x[..., 0:nj], x[..., nj:2 * nj] = q, qd
    rb = frame_block_start(nj, 0)
    x[..., rb + 6:rb + 12] = torch.tensor([1.0, 0, 0, 0, 1, 0], dtype=torch.float64)
    for kf in range(k):
        eb = frame_block_start(nj, 1 + kf)
        x[..., eb:eb + 3] = fk_pos[:, :, kf]
        x[..., eb + 3:eb + 6] = v_fk[:, :, kf]
        x[..., eb + 6:eb + 12] = matrix_to_rot6d(fk_rot[:, :, kf])
        x[..., eb + 12:eb + 15] = w_fk[:, :, kf]
    pose_ps, vel_ps = fk_consistency_terms(x, 0.05, nj, k, chain)
    assert pose_ps.max().item() < 1e-8 and vel_ps.max().item() < 1e-8
    xb = x.clone(); xb[0, 1, frame_block_start(nj, 1):frame_block_start(nj, 1) + 3] += 0.1
    pose_b, _ = fk_consistency_terms(xb, 0.05, nj, k, chain)
    assert pose_b.max().item() > 1e-4


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_flow_loss_fk_terms_and_tau_gate():
    torch.manual_seed(30)
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _FK_FRAMES, base_link="pelvis_link")
    nj, k = len(_FK_JOINTS), len(_FK_FRAMES)
    d, f, n = target_state_dim(nj, k), 8, 2
    model = _model(d, f, n, cmd_dim=0)
    x1, hist, height = torch.randn(16, f, d), torch.randn(16, n, d), torch.randn(16, 1)
    norm = _norm(x1)
    ckw = dict(state_norm=norm, dt=0.05, num_joints=nj, num_extra_frames=k,
               vel_weight=0.1, accel_weight=0.01, joint_vel_weight=0.1,
               fk=chain, fk_pose_weight=0.1, fk_vel_weight=0.01)
    out = flow_matching_loss(model, x1, hist, height, None, tau_min=0.0, **ckw)
    for key in ("vel_consistency", "accel_consistency", "joint_vel_consistency", "fk_pose", "fk_vel"):
        assert key in out
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out_g = flow_matching_loss(model, x1, hist, height, None, tau_min=2.0, **ckw)
    assert out_g["accel_consistency"].item() == 0.0
    assert out_g["fk_pose"].item() == 0.0 and out_g["fk_vel"].item() == 0.0


# ---------------------------------------------------------------------------
# 9. Terrain-penetration loss (new signature: no known mask, yaw-anchor lift)
# ---------------------------------------------------------------------------
_IDENT_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0])
_IDENT_ORI6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def _foot_state(nj, k, nf, foot_pos, foot_R=None):
    d = target_state_dim(nj, k)
    x = torch.zeros(1, nf, d, dtype=torch.float64)
    rb = frame_block_start(nj, 0)
    x[..., rb + 6:rb + 12] = _IDENT_ORI6D.to(torch.float64)
    eb = frame_block_start(nj, 1)
    x[..., eb:eb + 3] = foot_pos.to(torch.float64)
    x[..., eb + 6:eb + 12] = (_IDENT_ORI6D if foot_R is None else matrix_to_rot6d(foot_R)).to(torch.float64)
    return x


def test_box_sole_points():
    pts = box_sole_points(grid=(3, 2))
    assert pts.shape == (6, 3)
    assert torch.allclose(pts[:, 2], torch.full((6,), -0.02517 - 0.009255), atol=1e-6)


def test_flat_terrain_penetration():
    d = FlatTerrain(0.0).penetration(torch.tensor([[0.0, 0.0, -0.05], [0.0, 0.0, 0.1]]))
    assert torch.allclose(d, torch.tensor([0.05, 0.0]))


def test_terrain_ankle_mode_identity_anchor():
    nj, k, nf = 2, 1, 3
    p_ref, q_ref, terrain = torch.zeros(1, 3, dtype=torch.float64), _IDENT_QUAT.to(torch.float64)[None], FlatTerrain(0.0)
    x_lo = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, -0.05]))
    assert abs(terrain_penetration_term(x_lo, p_ref, q_ref, nj, [0], None, terrain).item() - 0.05 ** 2) < 1e-9
    x_hi = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, 0.05]))
    assert terrain_penetration_term(x_hi, p_ref, q_ref, nj, [0], None, terrain).item() < 1e-12


def test_terrain_contact_mode_and_tilt():
    nj, k, nf = 2, 1, 2
    p_ref, q_ref, terrain = torch.zeros(1, 3, dtype=torch.float64), _IDENT_QUAT.to(torch.float64)[None], FlatTerrain(0.0)
    cpts = box_sole_points(grid=(3, 2)).to(torch.float64)
    sole_z = -0.02517 - 0.009255
    x0 = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, -sole_z]))
    assert terrain_penetration_term(x0, p_ref, q_ref, nj, [0], cpts, terrain).item() < 1e-12
    th = torch.tensor(0.5, dtype=torch.float64)
    R_pitch = torch.tensor([[torch.cos(th), 0, torch.sin(th)], [0.0, 1.0, 0.0], [-torch.sin(th), 0, torch.cos(th)]],
                           dtype=torch.float64)
    x_tilt = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, 0.04]), foot_R=R_pitch)
    assert terrain_penetration_term(x_tilt, p_ref, q_ref, nj, [0], None, terrain).item() < 1e-12  # ankle above
    assert terrain_penetration_term(x_tilt, p_ref, q_ref, nj, [0], cpts, terrain).item() > 1e-4   # toe dips


def test_terrain_anchor_height_shifts_world_z():
    nj, k, nf = 2, 1, 2
    q_ref, terrain = _IDENT_QUAT.to(torch.float64)[None], FlatTerrain(0.0)
    x = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, 0.10]))  # foot 10 cm above the anchor
    hi = terrain_penetration_term(x, torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float64), q_ref, nj, [0], None, terrain)
    lo = terrain_penetration_term(x, torch.tensor([[0.0, 0.0, -0.2]], dtype=torch.float64), q_ref, nj, [0], None, terrain)
    assert hi.item() < 1e-12 and lo.item() > 1e-4  # anchor z shifts the foot world z


# ---------------------------------------------------------------------------
# 10. snap_windows + dataset + online round-trip (the new pipeline)
# ---------------------------------------------------------------------------
def test_snap_windows():
    # one episode, snap every 4 steps; N=2 history.
    L, N, K = 12, 2, 4
    eid = np.zeros(L, np.int64)
    step = np.arange(L, dtype=np.int64)
    done = np.zeros(L, np.uint8); done[-1] = 1
    snap = (np.arange(L) % K == 0).astype(np.uint8)  # snaps at 0,4,8
    wins = snap_windows(eid, step, done, snap, num_history=N, stride=1)
    # snap at step 0 lacks N-1 prior steps -> dropped; 4 and 8 kept.
    assert [w[-1] for w in wins] == [4, 8]
    assert all(len(w) == N for w in wins)
    assert wins[0].tolist() == [3, 4]


def _synthetic_recording(n_ep=6, L=40, nj=4, K=2, F=8, Kstep=4, seed=0):
    rng = np.random.default_rng(seed)
    node = target_state_dim(nj, K)
    parts = []
    for e in range(n_ep):
        rows = [np.concatenate([rng.standard_normal(3), [0, 0, 0, 1.0], rng.standard_normal(3),
                                rng.standard_normal(3), rng.standard_normal(2 * nj), rng.standard_normal(K * 13)])
                for _ in range(L)]
        s = np.stack(rows).astype(np.float32)
        rfu = rng.standard_normal((L, F, node)).astype(np.float32)
        rfu[:, 0, 0:3] = s[:, 0:3]; rfu[:, 0, 3:7] = s[:, 3:7]  # node-0 ego-anchored on the robot
        cf = rng.standard_normal((L, F, 3)).astype(np.float32)
        ra = (np.arange(L) % Kstep == 0).astype(np.uint8)
        done = np.zeros(L, np.uint8); done[-1] = 1
        parts.append((s, rfu, cf, ra, np.full(L, e, np.int64), np.arange(L, dtype=np.int64), done))
    keys = ["state", "ref_future", "cmd_future", "reanchor_active", "episode_id", "step_in_episode", "done"]
    data = {k: np.concatenate([p[i] for p in parts]) for i, k in enumerate(keys)}
    stn = ["root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_link_ang_vel_w", "joint_pos", "joint_vel"]
    std = [3, 4, 3, 3, nj, nj]
    frames = [f"f{i}" for i in range(K)]
    for fr in frames:
        stn += [f"{fr}:pos_w", f"{fr}:quat_w", f"{fr}:lin_vel_w", f"{fr}:ang_vel_w"]
        std += [3, 4, 3, 3]
    meta = SimpleNamespace(state_term_names=stn, state_term_dims=std, state_frame_names=frames,
                           state_joint_names=[f"j{i}" for i in range(nj)], ref_future_len=F)
    return data, meta, nj, K, F, node


def test_dataset_snap_windows_build():
    torch.manual_seed(0)
    data, meta, nj, K, F, node = _synthetic_recording()
    cfg = FlowMatchingConfig(num_history_states=2, future_len=F, cmd_array="target_vel", cmd_stride=4,
                             target_source="ego_anchored_clip_future", val_fraction=0.34, seed=0)
    train, val = build_train_val(data, meta, cfg)
    it = train[0]
    assert it["history"].shape == (2, node)
    assert it["state"].shape == (F, node)
    assert it["height"].shape == (1,)
    assert it["cmd_tokens"].shape == (cfg.num_cmd_tokens, 3)
    assert cfg.state_dim == node and cfg.history_dim == node and cfg.cmd_dim == 3
    assert val is not None and len(val) > 0
    assert isinstance(train.val_episode_ids, list)
    # node-0 target offset ~ 0 in the robot frame (ego-anchored), so the raw target root pos is small.
    tgt = train.state_normalizer.denormalize(train.state)  # (Nw, F, D)
    rb = frame_block_start(nj, K if False else 0)
    assert tgt[:, 0, rb:rb + 3].norm(dim=-1).mean().item() < 0.15


def test_online_roundtrip():
    torch.manual_seed(0)
    data, meta, nj, K, F, node = _synthetic_recording()
    N = 2
    cfg = FlowMatchingConfig(num_history_states=N, future_len=F, cmd_array="target_vel", cmd_stride=4,
                             target_source="ego_anchored_clip_future", val_fraction=0.2, seed=0,
                             d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    train, _ = build_train_val(data, meta, cfg)
    # Model must match the ctor load_fm_context will rebuild from cfg.
    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=F, num_history_states=N, history_dim=cfg.state_dim,
        height_dim=1, cmd_dim=cfg.cmd_dim, cmd_stride=cfg.cmd_stride,
        d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
    ).eval()
    ckpt = {
        "config": cfg.to_dict(),
        "meta": {"state_joint_names": meta.state_joint_names, "target_frame_names": meta.state_frame_names},
        "num_joints": nj,
        "state_normalizer": train.state_normalizer.state_dict(),
        "cmd_normalizer": train.cmd_normalizer.state_dict(),
        "height_normalizer": train.height_normalizer.state_dict(),
        "model": model.state_dict(),
    }
    ctx = load_fm_context(ckpt, device="cpu", num_sampling_steps=4)
    assert ctx.N == N and ctx.F == F and ctx.D == node and ctx.num_extra_frames == K
    V = 5
    rp = torch.randn(V, N, 3)
    rq = torch.nn.functional.normalize(torch.randn(V, N, 4), dim=-1)
    hist = build_history_tokens(ctx, rp, rq, torch.randn(V, N, 3), torch.randn(V, N, 3),
                                torch.randn(V, N, nj), torch.randn(V, N, nj), frames=torch.randn(V, N, K, 13))
    height = build_height_token(ctx, rp[:, -1, 2])
    cmd = build_cmd_tokens(ctx, torch.randn(V, F, 3))
    assert hist.shape == (V, N, node) and height.shape == (V, 1) and cmd.shape == (V, ctx.num_cmd_tokens, 3)
    world = sample_world(ctx, hist, height, cmd, p_ref=torch.randn(V, 3),
                         q_ref=torch.nn.functional.normalize(torch.randn(V, 4), dim=-1))
    assert world["root_pos_w"].shape == (V, F, 3)
    assert world["frame_pos_w"].shape == (V, F, K, 3)
    assert torch.isfinite(world["root_pos_w"]).all() and torch.isfinite(world["joint_pos"]).all()
