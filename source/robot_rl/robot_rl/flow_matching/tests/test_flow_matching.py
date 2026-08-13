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
    body_pose_terms,
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
    project_window_rotations,
    quat_to_rot6d,
    relative_state,
    relative_state_from_window,
    rot6d_to_matrix,
    reanchor_window,
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
    window_to_world,
    world_to_window,
    yaw_quat,
)
from isaaclab.utils.math import quat_conjugate, quat_mul, yaw_quat
from robot_rl.flow_matching.tests.recording_fixtures import (
    camera_spec,
    per_step_recording,
    windowed_recording,
)
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


def _window_case(seed: int, nf: int = 5, nj: int = 4, k: int = 2):
    """A random anchor + world reference path, and the window that encodes it."""
    g = torch.Generator().manual_seed(seed)

    def rq(n: int) -> torch.Tensor:
        """``n`` random unit quaternions (xyzw)."""
        q = torch.randn(n, 4, generator=g, dtype=torch.float64)
        return q / q.norm(dim=-1, keepdim=True)

    n = 3
    ap, aq = torch.randn(n, 3, generator=g, dtype=torch.float64), rq(n)
    rp, rqq = torch.randn(n, nf, 3, generator=g, dtype=torch.float64), rq(n * nf).view(n, nf, 4)
    win = world_to_window(
        ap[:, None, :], aq[:, None, :], rp, rqq,
        torch.randn(n, nf, 3, generator=g, dtype=torch.float64),
        torch.randn(n, nf, 3, generator=g, dtype=torch.float64),
        torch.randn(n, nf, nj, generator=g, dtype=torch.float64),
        torch.randn(n, nf, nj, generator=g, dtype=torch.float64),
        torch.randn(n, nf, k, 3, generator=g, dtype=torch.float64),
    )
    return (ap, aq, rp, rqq, win, nj, k)


def test_base_pose_encodes_the_reference_pose_in_the_anchor_yaw_frame():
    """``base_pose[3:9]`` decodes to ``R_yawT R_ref`` -- SIGN and FRAME, pinned independently.

    Asserted against a directly-built relative rotation rather than through the window's own
    inverse, which would agree with any consistent sign or frame. Together with
    ``BASE_POSE_CONVENTION`` this is what stops either being changed back by accident: the
    width would not move, so a stale policy would load clean and steer wrong.

    The anchor is YAW-ONLY, so the robot's roll and pitch are absent from the frame while the
    reference's are retained in full.
    """
    ap, aq, rp, rqq, win, nj, k = _window_case(0)
    n, nf = rqq.shape[:2]
    slc = window_block_slices(nj, k)["base_pose"]
    want = matrix_from_quat(
        quat_mul(quat_conjugate(yaw_quat(aq)[:, None, :].expand(n, nf, 4).reshape(-1, 4)),
                 rqq.reshape(-1, 4))
    )
    got = rot6d_to_matrix(win[..., slc.start + 3:slc.start + 9].reshape(-1, 6))
    assert torch.allclose(got, want, atol=1e-10)


def test_window_to_world_inverts_world_to_window():
    """Forward then inverse returns the world reference it started from."""
    ap, aq, rp, rqq, win, nj, k = _window_case(1)
    back = window_to_world(win, ap[:, None, :], aq[:, None, :], nj, k)
    assert torch.allclose(back["base_pos_w"], rp, atol=1e-10)
    # Compare ROTATIONS: the inverse goes through quat_from_matrix, which fixes the sign, so
    # a quaternion may come back negated -- the same rotation.
    assert torch.allclose(matrix_from_quat(back["base_quat_w"].reshape(-1, 4)),
                          matrix_from_quat(rqq.reshape(-1, 4)), atol=1e-10)


def test_reanchor_preserves_the_world_reference():
    """Re-anchoring is a change of frame, not of content -- for BOTH halves of base_pose.

    The orientation half is where this could go wrong quietly: it does not round-trip
    through the world like the position half does, but rotates its two stored columns by the
    anchor delta (``R_newT R_ref = (R_newT R_old)(R_oldT R_ref)``). That identity holds only
    because the block IS those columns, so it is worth pinning against an independent lift.
    Nothing else covers it -- the tracker's own re-anchor test checks the position half only.
    """
    ap, aq, rp, rqq, win, nj, k = _window_case(2)
    g = torch.Generator().manual_seed(99)
    n = ap.shape[0]
    np_, nq = torch.randn(n, 3, generator=g, dtype=torch.float64), torch.randn(
        n, 4, generator=g, dtype=torch.float64)
    nq = nq / nq.norm(dim=-1, keepdim=True)

    moved = reanchor_window(win, ap[:, None, :], aq[:, None, :],
                            np_[:, None, :], nq[:, None, :], nj, k)
    back = window_to_world(moved, np_[:, None, :], nq[:, None, :], nj, k)
    assert torch.allclose(back["base_pos_w"], rp, atol=1e-10)
    assert torch.allclose(matrix_from_quat(back["base_quat_w"].reshape(-1, 4)),
                          matrix_from_quat(rqq.reshape(-1, 4)), atol=1e-10)


def test_project_window_rotations_is_idempotent_and_fixes_garbage():
    """The generator's output projection: exact on rotations, valid on anything else."""
    ap, aq, rp, rqq, win, nj, k = _window_case(3)
    slc = window_block_slices(nj, k)["base_pose"]
    ori = slice(slc.start + 3, slc.start + 9)

    # A window built from real quaternions is already on the manifold, so projecting is a
    # no-op -- otherwise the projection would be silently perturbing honest references.
    assert torch.allclose(project_window_rotations(win, nj, k), win, atol=1e-10)

    # A raw 9-vector is not a rotation; after projecting, it is.
    garbage = win.clone()
    garbage[..., ori] = torch.randn_like(garbage[..., ori])
    fixed = project_window_rotations(garbage, nj, k)
    r = rot6d_to_matrix(fixed[..., ori].reshape(-1, 6))
    eye = torch.eye(3, dtype=r.dtype).expand_as(r)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-10)
    # ...and every other channel is untouched.
    assert torch.equal(fixed[..., :ori.start], garbage[..., :ori.start])
    assert torch.equal(fixed[..., ori.stop:], garbage[..., ori.stop:])


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


def _model(d=57, f=8, n=2, cmd_dim=3, stride=5, extero_sensor_dim=_HK, **kw):
    return FlowMatchingTransformer(
        state_dim=d, future_len=f, num_history_states=n, history_dim=d, extero_sensor_dim=extero_sensor_dim,
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

    ``base_pose[0:3]`` ramps at ``base_vel[0:3]``, and ``base_pose[3:9]`` is the 6D
    representation of a constant-``omega`` rotation ``R_t = Exp(omega * t * dt)``, so
    ``Log(R_{t+1} R_tT) / dt == omega == base_vel[3:6]`` exactly. Both hold because every
    node shares one anchor -- that is the property the consistency loss relies on.

    The orientation ramp is POSITIVE: ``base_pose[3:9]`` encodes ``R_actT R_ref``, so a
    reference turning at ``+omega`` relative to the anchor turns the block the same way. It
    used to be negated, back when the block was ``Log(R_refT R_act)``; that sign and the
    transpose in ``frame_consistency_terms`` moved together.

    Offsets come from the layout rather than being written out, since ``base_pose`` is now 9
    wide and ``base_vel`` no longer starts at 6.

    The ee block is position-only, so it carries nothing to tie and is a plain ramp.
    """
    g = torch.Generator().manual_seed(seed)
    blocks = window_block_slices(nj, k)
    x = torch.zeros(1, nf, window_dim(nj, k), dtype=torch.float64)

    bp, bv = blocks["base_pose"].start, blocks["base_vel"].start
    pos0 = torch.randn(3, generator=g, dtype=torch.float64)
    vel = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
    omega = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
    for t in range(nf):
        x[0, t, bp:bp + 3] = pos0 + vel * (t * dt)
        x[0, t, bp + 3:bp + 9] = matrix_to_rot6d(_rodrigues((omega * (t * dt))[None, :]))[0]
        x[0, t, bv:bv + 3] = vel
        x[0, t, bv + 3:bv + 6] = omega

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
    vel = frame_consistency_terms(x, x, dt, nj)
    jvel = joint_vel_consistency_term(x, dt, nj)
    assert vel.max().item() < 1e-10 and jvel.max().item() < 1e-10


def test_frame_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=2)
    pred = x.clone()
    pred[0, 3, 0:3] += 1.0
    vel = frame_consistency_terms(pred, x, dt, nj)
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
               vel_weight=0.1, joint_vel_weight=0.1,
               fk=chain, fk_pose_weight=0.1)
    out = flow_matching_loss(model, x1, hist, height, None, tau_min=0.0, **ckw)
    for key in ("vel_consistency", "joint_vel_consistency", "fk_pose"):
        assert key in out
    assert "fk_vel" not in out
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out_g = flow_matching_loss(model, x1, hist, height, None, tau_min=2.0, **ckw)
    assert out_g["fk_pose"].item() == 0.0


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_flow_loss_body_pose_terms_are_wired_and_gated():
    """The body-pose terms reach the objective, are τ-gated, and are OFF by default."""
    torch.manual_seed(31)
    body = FKChain(_FK_MJCF, _FK_JOINTS, _BODY_FRAMES, base_link="pelvis_link")
    nj, k = len(_FK_JOINTS), 2
    d, f, n = window_dim(nj, k), 8, 2
    model = _model(d, f, n, cmd_dim=0)
    x1, hist, height = torch.randn(16, f, d), torch.randn(16, n, d), torch.randn(16, _HK)
    norm = _norm(x1)
    base = dict(state_norm=norm, dt=0.05, num_joints=nj, num_extra_frames=k)

    def loss(**kw):
        """One loss evaluation on a FIXED flow-batch draw.

        ``sample_flow_batch`` draws ``x0`` and ``tau`` internally, so two calls are not
        otherwise comparable -- and comparing them is the point of this test.
        """
        torch.manual_seed(1234)
        return flow_matching_loss(model, x1, hist, height, None, **base, **kw)

    # Off by default: no body_fk, no keys, and the loss is the flow MSE alone.
    off = loss(tau_min=0.0)
    assert "body_pose_root" not in off and "body_pose_world" not in off

    on = loss(tau_min=0.0, body_fk=body,
              body_pose_root_weight=0.1, body_pose_world_weight=0.1)
    for key in ("body_pose_root", "body_pose_world"):
        assert key in on and torch.isfinite(on[key])
    assert torch.isfinite(on["loss"]) and on["loss"].requires_grad
    assert on["loss"].item() > off["loss"].item()

    # τ-gated with the FK term: nothing survives an unreachable gate.
    gated = loss(tau_min=2.0, body_fk=body,
                 body_pose_root_weight=0.1, body_pose_world_weight=0.1)
    assert gated["body_pose_root"].item() == 0.0
    assert gated["body_pose_world"].item() == 0.0

    # A weight of zero still REPORTS its term (free to log, since FK ran anyway) but must
    # not change the objective -- that is what makes the ablation a single flag.
    root_only = loss(tau_min=0.0, body_fk=body,
                     body_pose_root_weight=0.1, body_pose_world_weight=0.0)
    assert "body_pose_world" in root_only
    torch.testing.assert_close(root_only["body_pose_root"], on["body_pose_root"])
    assert root_only["loss"].item() < on["loss"].item()


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


def _synthetic_recording(n_ep=6, W_ep=6, nj=4, K=2, F=8, H=4, Ksc=6, depth=None, seed=0):
    """Synthesize a windowed recording plus its layout meta.

    Returns ``(data, meta, nj, K, F, state_dim)``. Note the two DIFFERENT widths:
    ``raw_node`` = ``sum(state_term_dims)`` is what ``history_state``/``history_ref``/
    ``ref_future`` rows are (world-frame, full 13 per extra frame), while ``state_dim`` is
    the narrower reference-window node the dataset builds from them (ee blocks
    position-only).

    The trajectory is smooth, tracked and near-upright -- see
    :mod:`~robot_rl.flow_matching.tests.recording_fixtures` for why the dataset now
    rejects white noise, and why the history rows are PLAN-anchored.
    """
    data, stn, std, frames = windowed_recording(
        n_ep=n_ep, windows_per_ep=W_ep, history_len=H, future_len=F, nj=nj, k=K,
        num_scan_rays=Ksc, depth=depth, seed=seed,
    )
    meta = SimpleNamespace(state_term_names=stn, state_term_dims=std, state_frame_names=frames,
                           state_joint_names=[f"j{i}" for i in range(nj)], ref_future_len=F,
                           history_anchor="plan", history_len=H,
                           # Only alongside a depth array, exactly as the recorder behaves.
                           depth_camera_spec=camera_spec(*depth) if depth else None)
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
    assert state_dim < data["history_state"].shape[-1]
    assert it["history"].shape == (2, state_dim)
    assert it["state"].shape == (F, state_dim)
    assert it["extero"].shape == (ksc,)
    assert it["cmd_tokens"].shape == (cfg.num_cmd_tokens, 3)
    assert cfg.state_dim == state_dim and cfg.history_dim == state_dim and cfg.cmd_dim == 3
    assert cfg.extero_sensor_dim == ksc
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
    assert d_slim == 15 + nj < window_dim(nj, K)   # base_pose(9) + base_vel(6) + nj
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
        state_dim=d, future_len=f, num_history_states=2, history_dim=d, extero_sensor_dim=6,
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
        extero_sensor_dim=cfg.extero_sensor_dim, cmd_dim=cfg.cmd_dim, cmd_stride=cfg.cmd_stride,
        d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward,
        attention_mode=cfg.attention_mode,
    ).eval()
    ckpt = {
        "config": cfg.to_dict(),
        "meta": {"state_joint_names": meta.state_joint_names, "target_frame_names": meta.state_frame_names},
        "num_joints": nj,
        "state_normalizer": train.state_normalizer.state_dict(),
        "cmd_normalizer": train.cmd_normalizer.state_dict(),
        "extero_normalizer": train.extero_normalizer.state_dict(),
        "model": model.state_dict(),
    }
    ctx = load_fm_context(ckpt, device="cpu", num_sampling_steps=4)
    assert ctx.N == N and ctx.F == F and ctx.D == state_dim and ctx.num_extra_frames == K
    V = 5
    # The history is window nodes the generator emitted, not robot state -- so the online
    # path takes the same (V, N, D) it produces, closing the autoregressive loop.
    hist = build_history_tokens(ctx, torch.randn(V, N, state_dim))
    height = build_extero_tokens(ctx, torch.randn(V, cfg.extero_sensor_dim))
    cmd = build_cmd_tokens(ctx, torch.randn(V, F, 3))
    assert (hist.shape == (V, N, state_dim) and height.shape == (V, cfg.extero_sensor_dim)
            and cmd.shape == (V, ctx.num_cmd_tokens, 3))
    pred = sample_window(ctx, hist, height, cmd)
    assert pred.shape == (V, F, state_dim)
    assert torch.isfinite(pred).all()
    # A node the model just emitted is directly re-usable as a context token.
    assert build_history_tokens(ctx, pred[:, :N]).shape == (V, N, state_dim)
    with pytest.raises(ValueError, match="window nodes"):
        build_history_tokens(ctx, torch.randn(V, N, state_dim + 1))


def test_online_roundtrip_with_the_cnn_encoder():
    """Same loop for the CONVOLUTIONAL encoder: dataset -> ckpt -> context -> sample.

    The whole chain in one test, because each link carries its own contract and they only
    have to agree end to end: the dataset derives ``extero_image_hw`` from the camera
    contract, the config carries it into the checkpoint, ``load_fm_context`` rebuilds the
    per-camera trunks from it and re-checks it against the recorded spec, and the flat
    ``(V, T*K)`` reading the deploy runtime feeds still folds correctly inside.
    """
    torch.manual_seed(0)
    H, W = 26, 30
    data, meta, nj, K, F, state_dim = _synthetic_recording(depth=(2, H, W))
    N = 2
    cfg = FlowMatchingConfig(num_history_states=N, future_len=F, cmd_array="target_vel",
                             cmd_stride=4, extero_source="depth", extero_encoder="cnn",
                             extero_norm_mode="per_sensor",
                             target_source="ego_anchored_clip_future", val_fraction=0.2,
                             seed=0, d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    train, _ = build_train_val(data, meta, cfg)
    # The dataset must have DERIVED the geometry -- nothing above stated it.
    assert cfg.extero_image_hw == [[H, W], [H, W]]
    assert cfg.extero_sensor_count == 2 and cfg.extero_sensor_dim == H * W
    assert cfg.tokens_per_sensor == 16 and cfg.extero_seq_tokens == 32
    # Through from_config: the reload under test goes through it, so building the saved model
    # any other way compares two different architectures and reports it as missing keys.
    model = FlowMatchingTransformer.from_config(cfg).eval()
    ckpt = {
        "config": cfg.to_dict(),
        "meta": {"state_joint_names": meta.state_joint_names,
                 "target_frame_names": meta.state_frame_names,
                 "extero_spec": train.extero_spec},
        "num_joints": nj,
        "state_normalizer": train.state_normalizer.state_dict(),
        "cmd_normalizer": train.cmd_normalizer.state_dict(),
        "extero_normalizer": train.extero_normalizer.state_dict(),
        "model": model.state_dict(),
    }
    ctx = load_fm_context(ckpt, device="cpu", num_sampling_steps=4)
    assert ctx.extero_image_hw == [(H, W), (H, W)]
    # The width contract a tracker checks its LIVE cameras against
    # (``FmTrackingCommand._check_extero_shape``) has to be on the context: it is the only
    # description of the cameras a consumer gets without re-opening the checkpoint.
    assert (ctx.extero_sensor_count, ctx.extero_sensor_dim) == (2, H * W)
    V = 4
    hist = build_history_tokens(ctx, torch.randn(V, N, state_dim))
    # FLAT, exactly as the deploy runtime feeds it -- the fold happens inside the model.
    extero = build_extero_tokens(ctx, torch.randn(V, 2, H * W))
    assert extero.shape == (V, 2 * H * W)
    pred = sample_window(ctx, hist, extero, build_cmd_tokens(ctx, torch.randn(V, F, 3)))
    assert pred.shape == (V, F, state_dim) and torch.isfinite(pred).all()

    # A config that misdescribes the image shape must be refused, not folded anyway --
    # transposing it keeps every tensor the right size and scrambles every kernel.
    bad = {**ckpt, "config": {**cfg.to_dict(), "extero_image_hw": [[W, H], [W, H]]}}
    with pytest.raises(ValueError, match="disagrees with the recorded camera contract"):
        load_fm_context(bad, device="cpu")
    # ...and so must a config claiming the other encoder over these weights.
    stale = {**ckpt, "config": {**cfg.to_dict(), "extero_encoder": "mlp"}}
    with pytest.raises(ValueError, match="extero_cnn"):
        load_fm_context(stale, device="cpu")




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
    """The context tokens come from ``history_ref``, which is the whole point.

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
    a = 13                                           # joint_pos start in the raw node layout
    shifted["history_ref"][:, :, a:a + nj] += 0.3     # the APPLIED REFERENCE moves...
    shifted["ref_future"][:, :, a:a + nj] += 0.3     # ...consistently: node 0 IS row H-1
    moved, _ = _built(shifted, meta, F)

    assert torch.allclose(moved[..., jp], base[..., jp] + 0.3, atol=1e-4), \
        "history joints did not follow history_ref -- the history is not the applied reference"


def test_history_rows_are_anchored_on_the_plan_not_the_snap():
    """The context tokens use the PLAN's anchor pose, not the robot at the snap.

    This is what lets deploy push back the node it just emitted with no odometry: a row is
    the window node in the frame of the plan that produced it. Perturbing the measured
    robot at the snap row must therefore move the target -- which IS anchored there -- and
    leave the history rows untouched.
    """
    torch.manual_seed(0)
    data, meta, nj, K, F, _ = _synthetic_recording()
    bp = window_block_slices(nj, K)["base_pose"]
    base_h, ds = _built(data, meta, F, N=2)
    base_t = ds.state

    moved = {k: v.copy() for k, v in data.items()}
    moved["history_state"][:, -1, 0:3] += 0.05   # only the SNAP row's measured robot moves
    moved_h, ds2 = _built(moved, meta, F, N=2)

    # The target is anchored on the snap robot, so it must move...
    assert not torch.allclose(ds2.state[..., bp], base_t[..., bp], atol=1e-4)
    # ...while the history rows, anchored on the (untouched) plan pose, must not.
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
    """The snap row is excluded from the history, so a window spans N+1 recorded rows.

    ``H >= N + 1``: N context rows plus the snap row they precede. Asking for more context
    than the recording stored has to name the ``--history_len`` to re-record with rather
    than silently serving a shorter history.
    """
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording(H=4)
    with pytest.raises(ValueError, match="needs history_len"):
        _built(data, meta, F, N=64)


def test_misaligned_reference_ring_is_rejected():
    """``history_ref[H-1] != ref_future[0]`` fails loudly rather than skewing the history.

    An off-by-one between the recorder's applied-reference ring and its look-ahead gather
    is invisible downstream: the shapes are right and the model trains.
    """
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording()
    bad = {k: v.copy() for k, v in data.items()}
    # Roll WITHIN each window, so row H-1 is some other step of the same reference -- the
    # shape stays right and only the alignment breaks, which is the whole point.
    bad["history_ref"] = np.roll(bad["history_ref"], 1, axis=1)
    with pytest.raises(ValueError, match="history_ref\\[H-1\\] != ref_future\\[0\\]"):
        _built(bad, meta, F)


def test_per_step_recording_is_refused():
    """A per-step (``.npz``) recording cannot build a dataset at all, and must say so.

    The generator conditions on a PLAN-frame history, which only the windowed writer
    produces, so an absent ``history_anchor`` means the rows are anchored per step --
    correctly shaped, and describing something the generator is never served. Worth pinning
    because the guard was added without updating these fixtures, which left every
    dataset-building test in this package red: a fixture that cannot build is
    indistinguishable from a fixture that builds the wrong thing unless something asserts
    which one it is.
    """
    torch.manual_seed(0)
    data, stn, std, frames = per_step_recording(n_ep=3, length=24, nj=4, k=2, future_len=6)
    meta = SimpleNamespace(state_term_names=stn, state_term_dims=std, state_frame_names=frames,
                           state_joint_names=[f"j{i}" for i in range(4)], ref_future_len=6)
    with pytest.raises(ValueError, match="history_anchor"):
        _built(data, meta, 6)


def test_unanchored_future_is_rejected():
    """Node 0 must sit on the robot -- the ego-anchor invariant the paradigm assumes."""
    torch.manual_seed(0)
    data, meta, _, _, F, _ = _synthetic_recording()
    bad = {k: v.copy() for k, v in data.items()}
    # Move the reference away from the robot while keeping the ring aligned with the
    # look-ahead, so the ego-anchor guard is the one that fires rather than guard 3.
    bad["history_ref"][:, :, 0:3] += 5.0
    bad["ref_future"][:, :, 0:3] += 5.0
    with pytest.raises(ValueError, match="ego-anchored"):
        _built(bad, meta, F)


# ---------------------------------------------------------------------------
# Weight EMA.
# ---------------------------------------------------------------------------
def _tiny_model(d=8):
    """A minimal module with a float parameter and an integer buffer."""
    m = torch.nn.Linear(d, d)
    m.register_buffer("counter", torch.zeros(1, dtype=torch.long))
    return m


def test_ema_tracks_a_lagged_average_not_the_latest_weights():
    """The whole point: the shadow is an AVERAGE of the trajectory, not a copy of it.

    An EMA refreshed only when a checkpoint is written would equal the live weights, which
    buys nothing -- so this pins that it lags, and that repeated updates pull it toward the
    current value rather than snapping to it.
    """
    from robot_rl.flow_matching import WeightEMA
    torch.manual_seed(0)
    m = _tiny_model()
    ema = WeightEMA(m, decay=0.9, warmup=False)
    start = m.weight.detach().clone()

    with torch.no_grad():
        m.weight.add_(1.0)                      # a big step away from the start
    ema.update(m)
    # One update at decay 0.9 moves a tenth of the way, so it is neither the old nor the new.
    assert torch.allclose(ema.shadow["weight"], start + 0.1, atol=1e-6)
    for _ in range(80):
        ema.update(m)
    # Many updates with the weights held still converge onto them.
    torch.testing.assert_close(ema.shadow["weight"], m.weight.detach(), atol=1e-3, rtol=0)


def test_ema_warmup_avoids_being_dominated_by_the_initialisation():
    """Without warmup the first ~1/(1-decay) steps are mostly the random init.

    At decay 0.999 that is 1000 steps -- most of a short run -- so the ramp matters rather
    than being a nicety.
    """
    from robot_rl.flow_matching import WeightEMA
    m = _tiny_model()
    warm, cold = WeightEMA(m, decay=0.999), WeightEMA(m, decay=0.999, warmup=False)
    with torch.no_grad():
        m.weight.add_(1.0)
    warm.update(m)
    cold.update(m)
    moved_warm = (warm.shadow["weight"] - cold.shadow["weight"]).abs().mean()
    assert moved_warm > 0.05, "warmup should track the live weights far faster at step 1"
    assert warm.effective_decay() < warm.decay


def test_ema_swapped_in_restores_the_raw_weights():
    """Validation must not leave the trainer optimising the shadow copy."""
    from robot_rl.flow_matching import WeightEMA
    m = _tiny_model()
    ema = WeightEMA(m, decay=0.5, warmup=False)
    with torch.no_grad():
        m.weight.add_(2.0)
    ema.update(m)
    raw = m.weight.detach().clone()
    with ema.swapped_in(m):
        assert not torch.allclose(m.weight, raw)          # EMA weights are installed
    torch.testing.assert_close(m.weight.detach(), raw)     # and the raw ones come back


def test_ema_copies_integer_buffers_rather_than_averaging_them():
    """Averaging an index or a counter is meaningless, so those are copied through."""
    from robot_rl.flow_matching import WeightEMA
    m = _tiny_model()
    ema = WeightEMA(m, decay=0.9, warmup=False)
    with torch.no_grad():
        m.counter.fill_(7)
    ema.update(m)
    assert ema.shadow["counter"].item() == 7
    assert ema.shadow["counter"].dtype == torch.long


# ---------------------------------------------------------------------------
# Clip-family filtering (--skip_families).
# ---------------------------------------------------------------------------
def _windowed_with_families(codes, names):
    """A minimal windowed recording carrying a per-window clip_family tag."""
    from robot_rl.flow_matching.dataset import _family_keep_mask
    data = {"clip_family": np.asarray(codes, dtype=np.uint8)}
    meta = SimpleNamespace(family_names=list(names))
    return _family_keep_mask, data, meta


def test_skip_families_drops_only_the_named_family():
    """The point of recording a family CODE: any family becomes a train-time switch.

    ``family_jump_off`` is the backwards jump-off-a-box family, which the forward-facing
    depth cameras cannot observe -- so those windows ask the generator to predict something
    invisible. Nothing else should move.
    """
    keep_mask, data, meta = _windowed_with_families(
        [0, 1, 2, 1, 2, 0], ["unknown", "family_jump_off", "family_stairs_up"])
    cfg = FlowMatchingConfig(skip_families=["family_jump_off"])
    keep = keep_mask(data, meta, cfg, 6)
    assert keep.tolist() == [True, False, True, False, True, True]


def test_skip_families_is_a_noop_when_nothing_is_requested():
    """No flag, no copy: the common path must not pay for the filter."""
    keep_mask, data, meta = _windowed_with_families([0, 1], ["unknown", "family_jump_off"])
    assert keep_mask(data, meta, FlowMatchingConfig(), 2) is None


def test_skip_families_refuses_a_recording_without_the_tag():
    """Silently training on the family you asked to exclude is the failure to avoid."""
    from robot_rl.flow_matching.dataset import _family_keep_mask
    cfg = FlowMatchingConfig(skip_families=["family_jump_off"])
    with pytest.raises(ValueError, match="predates them"):
        _family_keep_mask({}, SimpleNamespace(family_names=[]), cfg, 4)


def test_skip_families_refuses_a_name_the_recording_does_not_have():
    """A typo would otherwise read as 'nothing to drop' -- i.e. as success."""
    keep_mask, data, meta = _windowed_with_families([0, 1], ["unknown", "family_jump_off"])
    cfg = FlowMatchingConfig(skip_families=["family_jumpoff"])     # typo
    with pytest.raises(ValueError, match="does not contain"):
        keep_mask(data, meta, cfg, 2)


def test_skip_families_refuses_to_drop_everything():
    """Nothing left to train on is a configuration error, not an empty dataset."""
    keep_mask, data, meta = _windowed_with_families([1, 1, 1], ["unknown", "family_jump_off"])
    cfg = FlowMatchingConfig(skip_families=["family_jump_off"])
    with pytest.raises(ValueError, match="drop ALL"):
        keep_mask(data, meta, cfg, 3)


# ---------------------------------------------------------------------------
# Transition oversampling (--transition_fraction).
# ---------------------------------------------------------------------------
def _patched_transition_mask(monkeypatch, mask):
    """Stub the library join so the weight math is testable without a clip .h5."""
    from robot_rl.flow_matching import dataset as ds
    monkeypatch.setattr(ds, "transition_window_mask",
                        lambda data, meta, cfg, w: np.asarray(mask, dtype=bool))
    return ds.transition_sampler_weights


def test_transition_weights_hit_the_requested_fraction(monkeypatch):
    """The weight makes the EXPECTED per-epoch transition share equal the target.

    Checked as a probability, not as a raw weight: the whole point of the knob is the
    resulting mix, and w = p*n_f/((1-p)*n_t) is easy to get subtly wrong (e.g. by using
    the total instead of the non-transition count).
    """
    mask = np.array([True] * 2 + [False] * 8)          # 20% natural, like the real 17.8%
    weights_fn = _patched_transition_mask(monkeypatch, mask)
    cfg = FlowMatchingConfig(transition_fraction=0.5)
    w = weights_fn({}, None, cfg, mask.size, np.arange(mask.size))
    share = w[mask].sum() / w.sum()
    assert share == pytest.approx(0.5)
    assert w[~mask].tolist() == [1.0] * 8              # non-transitions stay at unit weight


def test_transition_fraction_none_disables_weighting(monkeypatch):
    """``None`` must yield no sampler at all, not a uniform weight vector."""
    weights_fn = _patched_transition_mask(monkeypatch, [True, False])
    assert weights_fn({}, None, FlowMatchingConfig(transition_fraction=None), 2,
                      np.arange(2)) is None


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
def test_transition_fraction_rejects_degenerate_targets(monkeypatch, p):
    """0 and 1 are unreachable by reweighting and would divide by zero."""
    weights_fn = _patched_transition_mask(monkeypatch, [True, False])
    with pytest.raises(ValueError, match="must be in"):
        weights_fn({}, None, FlowMatchingConfig(transition_fraction=p), 2, np.arange(2))


def test_transition_weighting_refuses_a_single_class_split(monkeypatch):
    """No reweighting can manufacture a class the train split does not contain."""
    weights_fn = _patched_transition_mask(monkeypatch, [False, False, False])
    with pytest.raises(ValueError, match="needs both kinds"):
        weights_fn({}, None, FlowMatchingConfig(transition_fraction=0.5), 3, np.arange(3))


def test_transition_mask_refuses_a_recording_without_the_clip_index():
    """A recording predating clip_id/clip_step must error, not sample uniformly.

    A silent fallback would report a transition-weighted run that never weighted anything.
    """
    from robot_rl.flow_matching.dataset import transition_window_mask
    cfg = FlowMatchingConfig(transition_fraction=0.5)
    with pytest.raises(ValueError, match="clip_id"):
        transition_window_mask({"episode_id": np.zeros(3)}, SimpleNamespace(), cfg, 3)


# ---------------------------------------------------------------------------
# Body-link pose terms: the CLF's metric, as a loss
# ---------------------------------------------------------------------------
_BODY_FRAMES = ["left_knee_link", "right_knee_link", "left_ankle_roll_link", "torso_link"]


def _body_case(seed: int, nf: int = 4, b: int = 3, jitter: float = 0.0):
    """A random window pair (pred, target) over the FK joint set, plus the FK chain.

    ``jitter=0`` makes the prediction identical to the target, which is the case every
    assertion about a pose error vanishing rests on.
    """
    nj, k = len(_FK_JOINTS), 2
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _BODY_FRAMES, base_link="pelvis_link",
                    dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    d = window_dim(nj, k)
    tgt = torch.randn(b, nf, d, generator=g, dtype=torch.float64) * 0.2
    slc = window_block_slices(nj, k)
    bp = slc["base_pose"].start
    # base_pose's orientation half must be a real rotation for the composition to mean
    # anything, so build it from random quaternions rather than from the noise above.
    q = torch.randn(b * nf, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    tgt[..., bp + 3:bp + 9] = quat_to_rot6d(q).reshape(b, nf, 6)
    pred = tgt.clone()
    if jitter:
        pred[..., slc["joint_pos"]] += jitter * torch.randn(
            b, nf, nj, generator=g, dtype=torch.float64)
    return pred, tgt, chain, nj, k


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_body_pose_terms_vanish_on_an_exact_prediction():
    """Both terms are zero when the prediction IS the target -- and finite-gradient there.

    The second half matters as much as the first: a pose loss spends its life near
    coincidence, so an ``arccos``-style orientation metric would have an infinite gradient
    exactly where the term is working. ``so3_log`` is built to avoid that.
    """
    pred, tgt, chain, nj, k = _body_case(0)
    pred = pred.requires_grad_(True)
    out = body_pose_terms(pred, tgt, nj, k, chain)
    assert set(out) == {f"body_pose_{s}{h}" for s in ("root", "world")
                        for h in ("", "_pos", "_ori")}
    for name, v in out.items():
        assert v.shape == (pred.shape[0],)
        assert float(v.max()) < 1e-20, name
    out["body_pose_world"].sum().backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_body_pose_terms_grow_with_a_joint_error():
    """A joint perturbation moves both terms off zero -- the joints do drive the loss."""
    _, tgt, chain, nj, k = _body_case(1)
    pred, _, _, _, _ = _body_case(1, jitter=0.05)
    out = body_pose_terms(pred, tgt, nj, k, chain)
    assert float(out["body_pose_root"].min()) > 0.0
    assert float(out["body_pose_world"].min()) > 0.0


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_body_pose_world_equals_the_world_frame_error():
    """The composed term IS the world-frame error, which is the whole point of the design.

    It is computed in the ANCHOR frame so that no world coordinate enters the
    representation. That is only acceptable because the two are the same number:
    prediction and target share one anchor, and a rigid transform preserves distances and
    angles. Verified here against an explicit world-frame computation through a random
    anchor -- if this ever fails, the term has stopped meaning what its name says.
    """
    pred, tgt, chain, nj, k = _body_case(2, jitter=0.05)
    got = body_pose_terms(pred, tgt, nj, k, chain, ori_scale=0.0)["body_pose_world"]

    # Lift both windows to a random WORLD anchor and difference there instead.
    g = torch.Generator().manual_seed(7)
    b, nf = pred.shape[:2]
    ap = torch.randn(b, 3, generator=g, dtype=torch.float64)
    aq = torch.randn(b, 4, generator=g, dtype=torch.float64)
    aq = aq / aq.norm(dim=-1, keepdim=True)

    def world_body_pos(x):
        """Body positions in the WORLD, through the anchor and the window's base_pose."""
        w = window_to_world(x, ap[:, None, :], aq[:, None, :], nj, k)
        q = x[..., window_block_slices(nj, k)["joint_pos"]].reshape(b * nf, nj)
        p_root, _r, _jv, _jw = chain.pose_and_jacobian(q)
        p_root = p_root.reshape(b, nf, -1, 3)
        base_r = matrix_from_quat(w["base_quat_w"].reshape(-1, 4)).reshape(b, nf, 3, 3)
        return w["base_pos_w"][..., None, :] + torch.einsum("bfij,bfkj->bfki", base_r, p_root)

    want = ((world_body_pos(pred) - world_body_pos(tgt)) ** 2).sum(dim=-1).mean(dim=(1, 2))
    torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-10)


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_body_pose_root_ignores_the_base_while_world_does_not():
    """The reason both terms exist: only the root term is ungameable by moving the base.

    ``body_pose_world`` folds the base error in -- that is what makes it the CLF's metric
    and what lets a base error partially cancel a joint one. ``body_pose_root`` is blind to
    the base, so it pins posture on its own. ``fk_consistency_terms`` was rewritten to stop
    composing through the root for exactly this reason.
    """
    pred, tgt, chain, nj, k = _body_case(3)
    bp = window_block_slices(nj, k)["base_pose"].start
    moved = pred.clone()
    moved[..., bp:bp + 3] += 0.25                      # shift the predicted base only

    base = body_pose_terms(pred, tgt, nj, k, chain)
    shifted = body_pose_terms(moved, tgt, nj, k, chain)
    torch.testing.assert_close(shifted["body_pose_root"], base["body_pose_root"])
    assert float(shifted["body_pose_world"].min()) > float(base["body_pose_world"].max())


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_body_pose_world_absent_without_base_pose():
    """No base_pose block, nothing to compose with -- reported as absent, not as zero."""
    pred, tgt, chain, nj, k = _body_case(4)
    slim = ("joint_pos",)
    d = window_dim(nj, k, slim)
    out = body_pose_terms(pred[..., :d], tgt[..., :d], nj, k, chain, blocks=slim)
    assert set(out) == {"body_pose_root", "body_pose_root_pos", "body_pose_root_ori"}


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_body_pose_terms_need_joint_pos():
    """Without joints there is no posture to grade, and the weight should have been zeroed."""
    pred, tgt, chain, nj, k = _body_case(5)
    with pytest.raises(ValueError, match="body_pose_terms"):
        body_pose_terms(pred[..., :9], tgt[..., :9], nj, k, chain, blocks=("base_pose",))


def test_window_is_invariant_to_the_anchor_s_roll_and_pitch():
    """The property the yaw frame exists for: tilting the robot changes NOTHING in the window.

    This is what makes a roll/pitch domain randomization camera-only by construction -- it
    perturbs what the depth cameras see and leaves every window node bit-identical, with no
    bookkeeping on the reference side. Under the previous full-rotation anchor the same
    perturbation rotated the whole base block.

    Held at 1e-13 rather than exact equality. The anchor reaches the base blocks only through
    ``yaw_quat``, so the invariance is exact in real arithmetic -- but ``yaw_quat`` recovers the
    heading through ``atan2`` then ``sin``/``cos``, so two quaternions with mathematically
    equal yaw give the same heading only to rounding. Measured residual: 1.3e-15, about 6
    float64 eps, against a signal of order 1.
    """
    from isaaclab.utils.math import quat_from_euler_xyz
    nj, k, nf, n = 5, 2, 3, 4
    g = torch.Generator().manual_seed(11)

    def rq(m):
        """``m`` random unit quaternions (xyzw)."""
        q = torch.randn(m, 4, generator=g, dtype=torch.float64)
        return q / q.norm(dim=-1, keepdim=True)

    ap = torch.randn(n, 3, generator=g, dtype=torch.float64)
    args = (torch.randn(n, nf, 3, generator=g, dtype=torch.float64), rq(n * nf).view(n, nf, 4),
            torch.randn(n, nf, 3, generator=g, dtype=torch.float64),
            torch.randn(n, nf, 3, generator=g, dtype=torch.float64),
            torch.randn(n, nf, nj, generator=g, dtype=torch.float64),
            torch.randn(n, nf, nj, generator=g, dtype=torch.float64),
            torch.randn(n, nf, k, 3, generator=g, dtype=torch.float64))

    yaw = torch.rand(n, generator=g, dtype=torch.float64) * 6.0 - 3.0
    zero = torch.zeros(n, dtype=torch.float64)
    flat = quat_from_euler_xyz(zero, zero, yaw)
    tilted = quat_from_euler_xyz(zero + 0.35, zero - 0.22, yaw)      # same yaw, big tilt
    assert torch.allclose(yaw_quat(flat), yaw_quat(tilted), atol=1e-12)

    w_flat = world_to_window(ap[:, None, :], flat[:, None, :], *args)
    w_tilted = world_to_window(ap[:, None, :], tilted[:, None, :], *args)
    torch.testing.assert_close(w_flat, w_tilted, atol=1e-13, rtol=0.0)
    # ...and it is not vacuous: the block genuinely varies across nodes.
    assert w_flat[..., 0:9].std() > 0.1
