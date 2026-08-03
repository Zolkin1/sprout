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
    FlowMatchingConfig,
    FlowMatchingTransformer,
    FutureStateDataset,
    Normalizer,
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
    window_block_slices,
    window_dim,
    yaw_quat,
)
from robot_rl.flow_matching.tests.recording_fixtures import per_step_recording
from robot_rl.flow_matching.online import (
    build_cmd_tokens,
    build_extero_tokens,
    build_history_tokens,
    load_fm_context,
    sample_window,
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
    """Extra frames are position-only: 3 channels each, and only the root has an ori6d."""
    torch.manual_seed(12)
    nj, w, k = 21, 30, 4
    p, q = torch.randn(w, 3, dtype=torch.float64), _rand_unit_quat(w)
    v, om = torch.randn(w, 3, dtype=torch.float64), torch.randn(w, 3, dtype=torch.float64)
    jp, jv = torch.randn(w, nj, dtype=torch.float64), torch.randn(w, nj, dtype=torch.float64)
    fpos = torch.randn(w, k, 3, dtype=torch.float64)
    ref = w // 2
    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rel = relative_state(p, q, v, om, jp, jv, p_ref, q_ref, frame_pos_w=fpos)
    assert rel.shape == (w, target_state_dim(nj, k))
    assert rel.shape[1] == 2 * nj + 15 + 3 * k
    # Only the root carries an orientation now.
    assert orientation_slices(nj) == [orientation_slice(nj)]
    rec = state_to_absolute(rel, p_ref, q_ref, nj, num_extra_frames=k)
    assert torch.allclose(rec["frame_pos_w"], fpos, atol=1e-5)
    assert "frame_quat_w" not in rec and "frame_lin_vel_w" not in rec


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
# Height token is now a scan vector (K rays) embedded into ONE token; use a small K in tests.
_HK = 8


def _model(d=57, f=8, n=2, cmd_dim=3, stride=5, height_dim=_HK, **kw):
    return FlowMatchingTransformer(
        state_dim=d, future_len=f, num_history_states=n, history_dim=d, height_dim=height_dim,
        cmd_dim=cmd_dim, cmd_stride=stride, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, **kw
    ).eval()


def test_model_shapes_and_conditioning():
    torch.manual_seed(7)
    b, f, d, n, cmd_dim, stride = 3, 8, 57, 2, 3, 5
    model = _model(d, f, n, cmd_dim, stride)
    x, tau = torch.randn(b, f, d), torch.rand(b)
    hist, height = torch.randn(b, n, d), torch.randn(b, _HK)
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
    x, tau, height = torch.randn(b, f, d), torch.rand(b), torch.randn(b, _HK)
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
        out = model(torch.randn(2, 6, 8), torch.rand(2), torch.randn(2, 2, 8), torch.randn(2, _HK), None)
    assert out.shape == (2, 6, 8)


# ---------------------------------------------------------------------------
# 6. Flow loss + samplers (no inpainting)
# ---------------------------------------------------------------------------
def test_flow_loss_and_samplers():
    torch.manual_seed(9)
    b, f, d, n, cmd_dim, stride = 8, 8, 57, 2, 3, 5
    model = _model(d, f, n, cmd_dim, stride)
    x1, hist, height = torch.randn(b, f, d), torch.randn(b, n, d), torch.randn(b, _HK)
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
    """Self-consistent ``(1, F, D)`` window: base_pose integrates base_vel.

    ``base_pose[0:3]`` ramps at ``base_vel[0:3]``, and ``base_pose[3:6]`` is the rotation
    vector of a constant-``omega`` rotation, so
    ``Log(Exp(e_{t+1})T Exp(e_t)) / dt == omega == base_vel[3:6]`` exactly. Both hold
    because every node shares one anchor -- that is the property the consistency loss
    relies on.

    The orientation ramp is NEGATIVE because ``base_pose[3:6]`` is currently
    ``Log(R_refT R_act)`` (see ``window.BASE_POSE_CONVENTION``): a reference turning at
    ``+omega`` away from the anchor makes that log map run the other way. Unifying the
    base_pose sign flips this line and the transpose in ``frame_consistency_terms``
    together.

    The ee block is position-only, so it carries nothing to tie and is a plain ramp.
    """
    g = torch.Generator().manual_seed(seed)
    blocks = window_block_slices(nj, k)
    x = torch.zeros(1, nf, window_dim(nj, k), dtype=torch.float64)

    pos0 = torch.randn(3, generator=g, dtype=torch.float64)
    vel = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
    omega = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
    for t in range(nf):
        x[0, t, 0:3] = pos0 + vel * (t * dt)
        x[0, t, 3:6] = -omega * (t * dt)    # rotation VECTOR; so3_exp turns it into R
        x[0, t, 6:9] = vel
        x[0, t, 9:12] = omega

    ee = blocks["ee_pos"]
    for kf in range(k):
        e0 = ee.start + 3 * kf
        base = torch.randn(3, generator=g, dtype=torch.float64)
        rate = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
        for t in range(nf):
            x[0, t, e0:e0 + 3] = base + rate * (t * dt)
    return x


def test_frame_consistency_zero_for_consistent_traj():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=1)
    vel, accel = frame_consistency_terms(x, x, dt, nj)
    jvel = joint_vel_consistency_term(x, dt, nj)
    assert vel.max().item() < 1e-10 and accel.max().item() < 1e-10 and jvel.max().item() < 1e-10


def test_frame_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=2)
    pred = x.clone()
    pred[0, 3, 0:3] += 1.0
    vel, _ = frame_consistency_terms(pred, x, dt, nj)
    assert vel.max().item() > 1e-2


def test_joint_vel_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=3)
    pred = x.clone()
    pred[0, 3, window_block_slices(nj, k)["joint_pos"]] += 0.5
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
    d, nf = window_dim(nj, k), 4
    blocks = window_block_slices(nj, k)
    rng = np.random.default_rng(1)
    q = torch.tensor(rng.uniform(-0.5, 0.5, size=(1, nf, nj)), dtype=torch.float64)
    qd = torch.tensor(rng.standard_normal((1, nf, nj)) * 0.3, dtype=torch.float64)
    fk_pos, _fk_rot, _jv, _jw = chain.pose_and_jacobian(q.reshape(nf, nj))
    x = torch.zeros(1, nf, d, dtype=torch.float64)
    x[..., blocks["joint_pos"]], x[..., blocks["joint_vel"]] = q, qd
    # The window's ee block is already base-relative, which is the frame FKChain reports
    # in -- so an FK-consistent state is a straight copy, with no root pose involved.
    x[..., blocks["ee_pos"]] = fk_pos.reshape(1, nf, k * 3)
    pose_ps = fk_consistency_terms(x, nj, k, chain)
    assert pose_ps.max().item() < 1e-8
    xb = x.clone(); xb[0, 1, blocks["ee_pos"].start:blocks["ee_pos"].start + 3] += 0.1
    assert fk_consistency_terms(xb, nj, k, chain).max().item() > 1e-4


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_flow_loss_fk_terms_and_tau_gate():
    torch.manual_seed(30)
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _FK_FRAMES, base_link="pelvis_link")
    nj, k = len(_FK_JOINTS), len(_FK_FRAMES)
    d, f, n = target_state_dim(nj, k), 8, 2
    model = _model(d, f, n, cmd_dim=0)
    x1, hist, height = torch.randn(16, f, d), torch.randn(16, n, d), torch.randn(16, _HK)
    norm = _norm(x1)
    ckw = dict(state_norm=norm, dt=0.05, num_joints=nj, num_extra_frames=k,
               vel_weight=0.1, accel_weight=0.01, joint_vel_weight=0.1,
               fk=chain, fk_pose_weight=0.1)
    out = flow_matching_loss(model, x1, hist, height, None, tau_min=0.0, **ckw)
    for key in ("vel_consistency", "accel_consistency", "joint_vel_consistency", "fk_pose"):
        assert key in out
    assert "fk_vel" not in out
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out_g = flow_matching_loss(model, x1, hist, height, None, tau_min=2.0, **ckw)
    assert out_g["accel_consistency"].item() == 0.0
    assert out_g["fk_pose"].item() == 0.0


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


def _synthetic_recording(n_ep=6, L=40, nj=4, K=2, F=8, Kstep=4, Ksc=6, seed=0):
    """Synthesize a per-step recording plus its layout meta.

    Returns ``(data, meta, nj, K, F, state_dim)``. Note the two DIFFERENT widths:
    ``raw_node`` = ``sum(state_term_dims)`` is what ``state``/``ref_state``/``ref_future``
    rows are (world-frame, full 13 per extra frame), while ``state_dim`` is the narrower
    reference-window node the dataset builds from them (ee blocks position-only).

    The trajectory is smooth, tracked and near-upright -- see
    :mod:`~robot_rl.flow_matching.tests.recording_fixtures` for why the dataset now
    rejects white noise.
    """
    data, stn, std, frames = per_step_recording(
        n_ep=n_ep, length=L, nj=nj, k=K, future_len=F, snap_every=Kstep,
        num_scan_rays=Ksc, seed=seed,
    )
    meta = SimpleNamespace(state_term_names=stn, state_term_dims=std, state_frame_names=frames,
                           state_joint_names=[f"j{i}" for i in range(nj)], ref_future_len=F)
    return data, meta, nj, K, F, window_dim(nj, K)


def test_dataset_snap_windows_build():
    torch.manual_seed(0)
    data, meta, nj, K, F, state_dim = _synthetic_recording()
    cfg = FlowMatchingConfig(num_history_states=2, future_len=F, cmd_array="target_vel", cmd_stride=4,
                             extero_source="height_scan", target_source="ego_anchored_clip_future", val_fraction=0.34, seed=0)
    train, val = build_train_val(data, meta, cfg)
    it = train[0]
    ksc = data["height_scan"].shape[1]
    # The built vector is NARROWER than the recorded row: the extra frames' quat/lin/ang
    # columns are recorded but not read (position-only ee blocks).
    assert state_dim < data["state"].shape[1]
    assert it["history"].shape == (2, state_dim)
    assert it["state"].shape == (F, state_dim)
    assert it["height"].shape == (ksc,)
    assert it["cmd_tokens"].shape == (cfg.num_cmd_tokens, 3)
    assert cfg.state_dim == state_dim and cfg.history_dim == state_dim and cfg.cmd_dim == 3
    assert cfg.height_dim == ksc
    assert val is not None and len(val) > 0
    assert isinstance(train.val_episode_ids, list)
    # node-0 base offset ~ 0 in the robot frame (ego-anchored): the target's base_pose
    # position at node 0 is just the live tracking error.
    tgt = train.state_normalizer.denormalize(train.state)  # (Nw, F, D)
    bp = window_block_slices(nj, K)["base_pose"]
    assert tgt[:, 0, bp.start:bp.start + 3].norm(dim=-1).mean().item() < 0.15


def _slim_cfg(F, **kw):
    """A config for the three-block window the g1 teacher packs today."""
    return FlowMatchingConfig(
        num_history_states=2, future_len=F, cmd_array="target_vel", cmd_stride=4,
        extero_source="height_scan", target_source="ego_anchored_clip_future",
        val_fraction=0.34, seed=0, **kw,
    )


def test_dataset_defaults_the_selection_to_the_recording_teacher():
    """An FM trained on a recording predicts what THAT teacher observed, by default.

    The recorded arrays are world-frame rows, so any selection could be packed from them --
    which is exactly why the width has to be pinned to something rather than left at "all
    five". The generator's output is the policy's observation, so the only right answer is
    the teacher's own window.
    """
    torch.manual_seed(0)
    data, meta, nj, K, F, _ = _synthetic_recording()
    slim = ["base_pose", "base_vel", "joint_pos"]
    meta.reference_blocks = slim

    cfg = _slim_cfg(F)
    train, _ = build_train_val(data, meta, cfg)
    d_slim = window_dim(nj, K, slim)
    assert d_slim == 12 + nj < window_dim(nj, K)
    assert cfg.reference_blocks == slim and cfg.state_dim == d_slim
    assert train.state_dim == d_slim and tuple(train.blocks) == tuple(slim)
    assert train[0]["state"].shape == (F, d_slim)
    assert train[0]["history"].shape == (2, d_slim)
    # The blocks that ARE packed carry the same numbers they would in the full window --
    # dropping one changes the width, never the content of its neighbours.
    meta.reference_blocks = None
    full, _ = build_train_val(data, meta, _slim_cfg(F))
    a = train.state_normalizer.denormalize(train.state)
    b = full.state_normalizer.denormalize(full.state)
    torch.testing.assert_close(a[..., :d_slim], b[..., :d_slim], atol=1e-5, rtol=1e-5)


def test_dataset_disables_the_loss_terms_the_selection_cannot_support():
    """A consistency weight whose blocks are gone is zeroed, not left to index a neighbour.

    ``joint_vel_consistency`` ties joint_pos to joint_vel and ``fk_pose`` ties joint_pos to
    ee_pos; with neither second block present there is nothing to relate, and the column
    range each used to read now belongs to a different block entirely.
    """
    torch.manual_seed(0)
    data, meta, nj, K, F, _ = _synthetic_recording()
    meta.reference_blocks = ["base_pose", "base_vel", "joint_pos"]
    cfg = _slim_cfg(F, joint_vel_consistency_weight=0.01, fk_pose_weight=0.01,
                    vel_consistency_weight=0.01)
    build_train_val(data, meta, cfg)
    assert cfg.joint_vel_consistency_weight == 0.0
    assert cfg.fk_pose_weight == 0.0
    # base_pose + base_vel both survive, so the velocity term is still live.
    assert cfg.vel_consistency_weight == 0.01


def test_flow_loss_skips_the_terms_whose_blocks_are_absent():
    """The second line of the same defence, for a caller assembling the kwargs itself."""
    torch.manual_seed(0)
    b, f, nj, k = 3, 5, 4, 2
    slim = ("base_pose", "base_vel", "joint_pos")
    d = window_dim(nj, k, slim)
    model = FlowMatchingTransformer(
        state_dim=d, future_len=f, num_history_states=2, history_dim=d, height_dim=6,
        cmd_dim=0, d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
    )
    norm = Normalizer(torch.zeros(d), torch.ones(d), torch.ones(d, dtype=torch.bool))
    out = flow_matching_loss(
        model, torch.randn(b, f, d), torch.randn(b, 2, d), torch.randn(b, 6),
        state_norm=norm, num_joints=nj, num_extra_frames=k, blocks=slim,
        vel_weight=0.01, joint_vel_weight=0.01, fk_pose_weight=0.01, fk=object(),
    )
    assert "vel_consistency" in out                 # base_pose + base_vel are both there
    assert "joint_vel_consistency" not in out       # no joint_vel block
    assert "fk_pose" not in out                     # no ee_pos block


def test_online_roundtrip():
    torch.manual_seed(0)
    data, meta, nj, K, F, state_dim = _synthetic_recording()
    N = 2
    cfg = FlowMatchingConfig(num_history_states=N, future_len=F, cmd_array="target_vel", cmd_stride=4,
                             extero_source="height_scan", target_source="ego_anchored_clip_future", val_fraction=0.2, seed=0,
                             d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    train, _ = build_train_val(data, meta, cfg)
    # Model must match the ctor load_fm_context will rebuild from cfg.
    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=F, num_history_states=N, history_dim=cfg.state_dim,
        height_dim=cfg.height_dim, cmd_dim=cfg.cmd_dim, cmd_stride=cfg.cmd_stride,
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
    assert ctx.N == N and ctx.F == F and ctx.D == state_dim and ctx.num_extra_frames == K
    V = 5
    # The history is window nodes the generator emitted, not robot state -- so the online
    # path takes the same (V, N, D) it produces, closing the autoregressive loop.
    hist = build_history_tokens(ctx, torch.randn(V, N, state_dim))
    height = build_extero_tokens(ctx, torch.randn(V, cfg.height_dim))
    cmd = build_cmd_tokens(ctx, torch.randn(V, F, 3))
    assert (hist.shape == (V, N, state_dim) and height.shape == (V, cfg.height_dim)
            and cmd.shape == (V, ctx.num_cmd_tokens, 3))
    pred = sample_window(ctx, hist, height, cmd)
    assert pred.shape == (V, F, state_dim)
    assert torch.isfinite(pred).all()
    # A node the model just emitted is directly re-usable as a context token.
    assert build_history_tokens(ctx, pred[:, :N]).shape == (V, N, state_dim)
    with pytest.raises(ValueError, match="window nodes"):
        build_history_tokens(ctx, torch.randn(V, N, state_dim + 1))




# ---------------------------------------------------------------------------
# 11. Autoregressive reference-window history
# ---------------------------------------------------------------------------
def _built(data, meta, F, N=2, **over):
    """Build a dataset over a synthetic recording and return ``(raw_history, dataset)``."""
    cfg = FlowMatchingConfig(
        num_history_states=N, future_len=F, cmd_array="target_vel", cmd_stride=4,
        extero_source="height_scan", target_source="ego_anchored_clip_future",
        val_fraction=0.34, seed=0, obs_noise=False, **over,
    )
    train, _ = build_train_val(data, meta, cfg)
    return train.state_normalizer.denormalize(train.history), train


def test_history_is_the_applied_reference_not_the_measured_state():
    """The context tokens come from ``ref_state``, which is the whole point.

    Conditioning on the measured robot instead would still produce a correctly-shaped
    history and a converging loss -- it would just not be the autoregressive model. Offset
    the reference's joints so the two sources are distinguishable by inspection.
    """
    torch.manual_seed(0)
    data, meta, nj, K, F, _ = _synthetic_recording()
    blocks = window_block_slices(nj, K)
    jp = blocks["joint_pos"]

    base, _ = _built(data, meta, F)
    shifted = {k: v.copy() for k, v in data.items()}
    a = 13                                        # joint_pos start in the raw node layout
    shifted["ref_state"][:, a:a + nj] += 0.3      # the APPLIED REFERENCE moves...
    shifted["ref_future"][:, :, a:a + nj] += 0.3  # ...consistently, since node k IS ref[t+k]
    moved, _ = _built(shifted, meta, F)

    assert torch.allclose(moved[..., jp], base[..., jp] + 0.3, atol=1e-4), \
        "history joints did not follow ref_state -- the history is not the applied reference"


def test_history_rows_are_anchored_on_their_own_step():
    """Each context token uses the robot pose from ITS step, not the snap's.

    This is what lets deploy push back the node it just emitted with no odometry: a row is
    the window node as it was actually served. Perturbing the measured robot at the snap
    must therefore leave the earlier history rows untouched while moving the target.
    """
    torch.manual_seed(0)
    data, meta, nj, K, F, _ = _synthetic_recording()
    bp = window_block_slices(nj, K)["base_pose"]
    base_h, ds = _built(data, meta, F, N=2)
    base_t = ds.state

    moved = {k: v.copy() for k, v in data.items()}
    snaps = data["reanchor_active"].astype(bool)
    moved["state"] = moved["state"].copy()
    moved["state"][snaps, 0:3] += 0.05        # only the SNAP rows' measured robot moves
    moved_h, ds2 = _built(moved, meta, F, N=2)

    # The target is anchored on the snap robot, so it must move...
    assert not torch.allclose(ds2.state[..., bp], base_t[..., bp], atol=1e-4)
    # ...while the history rows, anchored on their own (untouched) steps, must not.
    torch.testing.assert_close(moved_h, base_h, atol=1e-5, rtol=1e-5)


def test_history_excludes_the_snap_and_abuts_the_target():
    """History row -1 is the step BEFORE node 0, so they are one control step apart."""
    torch.manual_seed(0)
    data, meta, nj, K, F, _ = _synthetic_recording()
    jp = window_block_slices(nj, K)["joint_pos"]
    hist, ds = _built(data, meta, F, N=3)
    tgt = ds.state_normalizer.denormalize(ds.state)
    gap = (hist[:, -1, jp] - tgt[:, 0, jp]).abs().max().item()
    # One step of the fixture's random walk is ~0.01 rad; two would be visibly larger.
    assert gap < 0.1, f"history[-1] is not one step before target[0] (gap {gap:.3f} rad)"


def test_history_needs_one_more_row_than_it_uses():
    """The snap row is excluded from the history, so a window spans N+1 recorded steps.

    (The windowed ``.h5`` path states this as ``H >= N + 1``; the per-step path simply runs
    out of consecutive rows first -- see ``test_record_windows`` for the ``.h5`` guard.)
    """
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording(L=40, Kstep=4)
    with pytest.raises(ValueError, match="No valid snap windows"):
        _built(data, meta, F, N=64)


def test_misaligned_reference_ring_is_rejected():
    """``ref_state[t] != ref_future[t, 0]`` fails loudly rather than skewing the history.

    An off-by-one between the recorder's applied-reference ring and its look-ahead gather
    is invisible downstream: the shapes are right and the model trains.
    """
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording()
    bad = {k: v.copy() for k, v in data.items()}
    bad["ref_state"] = np.roll(bad["ref_state"], 1, axis=0)
    with pytest.raises(ValueError, match="history_ref\\[H-1\\] != ref_future\\[0\\]"):
        _built(bad, meta, F)


def test_orientation_past_the_branch_cut_is_rejected():
    """base_pose's orientation half saturates at pi; a wrap has to be caught at build.

    It would otherwise put a jump discontinuity in both the regression target and the
    policy's observation.
    """
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording()
    bad = {k: v.copy() for k, v in data.items()}
    # Turn the measured robot 180 deg away from the reference: quat (x,y,z,w) = (0,0,1,0).
    bad["state"] = bad["state"].copy()
    bad["state"][:, 3:7] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="branch cut"):
        _built(bad, meta, F)


def _wrap_stub(nj=4, k=2, dt=0.02):
    """A bare dataset carrying only what ``_wrap_keep_mask`` reads."""
    ds = object.__new__(FutureStateDataset)
    ds.block_slices = window_block_slices(nj, k)
    ds.cfg = FlowMatchingConfig(dt=dt)
    return ds


def _target_with_wraps(n_win, n_bad, f=8, nj=4, k=2):
    """``(n_win, f, D)`` targets, ``n_bad`` of which cross the branch cut mid-horizon."""
    d = window_dim(nj, k)
    t = torch.zeros(n_win, f, d)
    ori = window_block_slices(nj, k)["base_pose"].start + 3
    t[:, :, ori] = 0.2                                   # everything else well inside
    t[:n_bad, f // 2, ori] = 3.12                        # a wrap partway through
    return t


def test_a_few_wrapped_windows_are_dropped_not_fatal():
    """One fast-spinning clip must not take the whole build down with it.

    Measured on the depth_v5 recording: 4 windows in 181426 (0.002%), all from one clip,
    wrapping at nodes 9-44. Shortening the horizon cannot help -- a window that wraps at
    node 9 wraps at every future_len worth training -- so dropping is the only proportionate
    response, and it has to be a per-window decision rather than a per-build one.
    """
    ds = _wrap_stub()
    keep = ds._wrap_keep_mask(_target_with_wraps(1000, 4))
    assert keep is not None
    assert int((~keep).sum()) == 4 and keep[:4].sum() == 0 and keep[4:].all()


def test_a_clean_recording_skips_the_filter_entirely():
    """No wraps -> ``None``, so the common path does not copy every array."""
    assert _wrap_stub()._wrap_keep_mask(_target_with_wraps(50, 0)) is None


def test_widespread_wrapping_is_fatal_rather_than_filtered():
    """Past the ceiling it is the horizon that is wrong, not the windows.

    Dropping at that scale would quietly bias the dataset away from exactly the turns the
    generator most needs to learn, and leave a model that cannot represent them.
    """
    ds = _wrap_stub()
    with pytest.raises(ValueError, match="incompatible"):
        ds._wrap_keep_mask(_target_with_wraps(100, 5))   # 5% > the 1% ceiling


def test_unanchored_future_is_rejected():
    """Node 0 must sit on the robot -- the ego-anchor invariant the paradigm assumes."""
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording()
    bad = {k: v.copy() for k, v in data.items()}
    bad["ref_state"] = bad["ref_state"] + 0.0
    bad["ref_future"] = bad["ref_future"].copy()
    bad["ref_state"][:, 0:3] += 5.0
    bad["ref_future"][:, :, 0:3] += 5.0
    with pytest.raises(ValueError, match="ego-anchored"):
        _built(bad, meta, F)
