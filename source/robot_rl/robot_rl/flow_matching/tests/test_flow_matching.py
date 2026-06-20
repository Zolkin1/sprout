# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator-free unit tests for the flow-matching subpackage.

Run with: ``pytest source/robot_rl/robot_rl/flow_matching/tests`` inside the
``env_isaaclab`` venv. All tests use synthetic tensors; no Isaac Sim app needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab.utils.math import matrix_from_quat

from robot_rl.flow_matching import (
    FlowMatchingConfig,
    FlowMatchingTransformer,
    FutureStateDataset,
    Normalizer,
    build_train_val,
    episode_windows,
    flow_matching_loss,
    matrix_to_rot6d,
    orientation_slice,
    orientation_slices,
    quat_to_rot6d,
    relative_state,
    relative_state_from_window,
    resolve_term_slice,
    rot6d_to_matrix,
    rot6d_to_quat,
    sample_flow_batch,
    sample_trajectory,
    state_to_absolute,
    target_state_dim,
)


def _rand_unit_quat(n: int) -> torch.Tensor:
    """Random unit quaternions (xyzw), shape (n, 4)."""
    q = torch.randn(n, 4, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 1. 6D rotation round-trip + Gram-Schmidt orthonormality
# ---------------------------------------------------------------------------
def test_rot6d_roundtrip_matrices():
    torch.manual_seed(0)
    q = _rand_unit_quat(64)
    r = matrix_from_quat(q)
    r2 = rot6d_to_matrix(matrix_to_rot6d(r))
    assert torch.allclose(r, r2, atol=1e-6)
    # full quat->6d->quat round trip compared as matrices (avoids double-cover)
    r3 = matrix_from_quat(rot6d_to_quat(quat_to_rot6d(q)))
    assert torch.allclose(r, r3, atol=1e-5)


def test_rot6d_orthonormal_from_garbage():
    torch.manual_seed(1)
    d6 = torch.randn(100, 6, dtype=torch.float64)  # not a valid rotation
    r = rot6d_to_matrix(d6)
    eye = torch.eye(3, dtype=torch.float64).expand_as(r)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-6)
    det = torch.linalg.det(r)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Relative-transform invertibility
# ---------------------------------------------------------------------------
def test_relative_state_invertible():
    torch.manual_seed(2)
    nj, w = 21, 50
    p = torch.randn(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    v = torch.randn(w, 3, dtype=torch.float64)
    om = torch.randn(w, 3, dtype=torch.float64)
    jp = torch.randn(w, nj, dtype=torch.float64)
    jv = torch.randn(w, nj, dtype=torch.float64)
    ref = w // 2

    rel = relative_state_from_window(p, q, v, om, jp, jv, ref_index=ref)
    assert rel.shape == (w, target_state_dim(nj))

    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rec = state_to_absolute(rel, p_ref, q_ref, nj)

    assert torch.allclose(rec["joint_pos"], jp, atol=1e-6)
    assert torch.allclose(rec["joint_vel"], jv, atol=1e-6)
    assert torch.allclose(rec["root_pos_w"], p, atol=1e-5)
    assert torch.allclose(rec["root_lin_vel_w"], v, atol=1e-5)
    assert torch.allclose(rec["root_ang_vel_w"], om, atol=1e-5)
    # quaternions compared as matrices (double cover)
    assert torch.allclose(matrix_from_quat(rec["root_quat_w"]), matrix_from_quat(q), atol=1e-5)


def test_relative_state_invertible_with_frames():
    """Round-trip relative_state <-> state_to_absolute with extra body frames."""
    torch.manual_seed(12)
    nj, w, k = 21, 30, 4
    p = torch.randn(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    v = torch.randn(w, 3, dtype=torch.float64)
    om = torch.randn(w, 3, dtype=torch.float64)
    jp = torch.randn(w, nj, dtype=torch.float64)
    jv = torch.randn(w, nj, dtype=torch.float64)
    fpos = torch.randn(w, k, 3, dtype=torch.float64)
    fquat = _rand_unit_quat(w * k).reshape(w, k, 4)
    flin = torch.randn(w, k, 3, dtype=torch.float64)
    fang = torch.randn(w, k, 3, dtype=torch.float64)
    ref = w // 2

    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rel = relative_state(
        p, q, v, om, jp, jv, p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat, frame_lin_vel_w=flin, frame_ang_vel_w=fang,
    )
    assert rel.shape == (w, target_state_dim(nj, k))
    # every ori block (base + frames) is left raw (not standardized) -> there are k+1
    assert len(orientation_slices(nj, k)) == k + 1

    rec = state_to_absolute(rel, p_ref, q_ref, nj, num_extra_frames=k)
    assert torch.allclose(rec["root_pos_w"], p, atol=1e-5)
    assert rec["frame_pos_w"].shape == (w, k, 3)
    assert torch.allclose(rec["frame_pos_w"], fpos, atol=1e-5)
    assert torch.allclose(rec["frame_lin_vel_w"], flin, atol=1e-5)
    assert torch.allclose(rec["frame_ang_vel_w"], fang, atol=1e-5)
    assert torch.allclose(
        matrix_from_quat(rec["frame_quat_w"].reshape(w * k, 4)),
        matrix_from_quat(fquat.reshape(w * k, 4)),
        atol=1e-5,
    )


def test_relative_state_reference_is_identity():
    torch.manual_seed(3)
    nj, w = 21, 10
    p = torch.randn(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    v = torch.randn(w, 3, dtype=torch.float64)
    om = torch.randn(w, 3, dtype=torch.float64)
    jp = torch.randn(w, nj, dtype=torch.float64)
    jv = torch.randn(w, nj, dtype=torch.float64)
    ref = 7
    rel = relative_state_from_window(p, q, v, om, jp, jv, ref_index=ref)
    o0, o1 = orientation_slice(nj)
    p_rel = rel[ref, 2 * nj : 2 * nj + 3]
    ori6d = rel[ref, o0:o1]
    assert torch.allclose(p_rel, torch.zeros(3, dtype=torch.float64), atol=1e-6)
    expected_ori = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    assert torch.allclose(ori6d, expected_ori, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Velocity reference transform is a pure rotation R_ref^T v_w
# ---------------------------------------------------------------------------
def test_velocity_frame_pure_rotation():
    torch.manual_seed(4)
    nj, w = 5, 4
    p = torch.zeros(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    v = torch.randn(w, 3, dtype=torch.float64)
    om = torch.zeros(w, 3, dtype=torch.float64)
    jp = torch.zeros(w, nj, dtype=torch.float64)
    jv = torch.zeros(w, nj, dtype=torch.float64)
    ref = 0
    rel = relative_state_from_window(p, q, v, om, jp, jv, ref_index=ref)
    v_rel = rel[:, 2 * nj + 3 : 2 * nj + 6]
    r_ref = matrix_from_quat(q[ref : ref + 1].expand(w, -1))  # (w,3,3)
    expected = (r_ref.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(v_rel, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Normalizer
# ---------------------------------------------------------------------------
def test_normalizer_mask_and_inverse():
    torch.manual_seed(5)
    d = 10
    x = torch.randn(2000, d) * 5.0 + 3.0
    x[:, 4] = 2.0  # constant column -> std clamp, no NaN
    mask = torch.ones(d, dtype=torch.bool)
    mask[7:9] = False  # identity dims (pretend orientation block)
    norm = Normalizer.fit(x, mask)
    z = norm.normalize(x)
    # standardized dims ~ mean 0 std 1 (skip the constant col 4)
    std_dims = [i for i in range(d) if mask[i] and i != 4]
    assert torch.allclose(z[:, std_dims].mean(0), torch.zeros(len(std_dims)), atol=1e-5)
    assert torch.allclose(z[:, std_dims].std(0), torch.ones(len(std_dims)), atol=1e-2)
    # identity dims unchanged
    assert torch.allclose(z[:, 7:9], x[:, 7:9], atol=1e-6)
    # invertible, finite
    assert torch.allclose(norm.denormalize(z), x, atol=1e-4)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# 5. Flow batch math
# ---------------------------------------------------------------------------
def test_flow_batch_math():
    torch.manual_seed(6)
    x1 = torch.randn(8, 25, 57)
    x_tau, tau, target_v = sample_flow_batch(x1)
    assert x_tau.shape == x1.shape and target_v.shape == x1.shape and tau.shape == (8,)
    # recover x0 and verify the interpolation identity
    x0 = x1 - target_v
    t = tau.view(-1, 1, 1)
    assert torch.allclose(x_tau, (1 - t) * x0 + t * x1, atol=1e-6)
    assert torch.allclose(target_v, x1 - x0, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Model shapes + full (non-causal) decoder self-attention
# ---------------------------------------------------------------------------
def test_model_shapes_and_noncausal():
    torch.manual_seed(7)
    b, h, f, din, d = 3, 25, 25, 60, 57
    model = FlowMatchingTransformer(din, d, h, f, d_model=32, nhead=4,
                                    num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64)
    model.eval()
    history = torch.randn(b, h, din)
    memory = model.encode(history)
    assert memory.shape == (b, h, 32)
    x_tau = torch.randn(b, f, d)
    tau = torch.rand(b)
    with torch.no_grad():
        v0 = model(x_tau, tau, memory)
        assert v0.shape == (b, f, d)
        # perturb future token 0 -> output at token 1 must change (full self-attn)
        x_pert = x_tau.clone()
        x_pert[:, 0, :] += 5.0
        v1 = model(x_pert, tau, memory)
    assert not torch.allclose(v0[:, 1, :], v1[:, 1, :], atol=1e-5)


# ---------------------------------------------------------------------------
# 7. Tiny synthetic training drives the loss down
# ---------------------------------------------------------------------------
def test_tiny_training_reduces_loss():
    torch.manual_seed(8)
    n, h, f, din, d = 256, 6, 6, 12, 8
    history = torch.randn(n, h, din)
    # future is a deterministic linear function of the flattened history
    w_map = torch.randn(h * din, f * d) * 0.1
    future = (history.reshape(n, -1) @ w_map).reshape(n, f, d)
    # standardize future so the N(0,I) prior is well scaled
    future = (future - future.mean(0)) / future.std(0).clamp_min(1e-6)

    model = FlowMatchingTransformer(din, d, h, f, d_model=64, nhead=4,
                                    num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(8)
    init_loss = flow_matching_loss(model, history, future)["loss"].item()
    for _ in range(300):
        opt.zero_grad()
        loss = flow_matching_loss(model, history, future)["loss"]
        loss.backward()
        opt.step()
    final_loss = flow_matching_loss(model, history, future)["loss"].item()
    assert final_loss < 0.5 * init_loss, f"init={init_loss:.3f} final={final_loss:.3f}"


# ---------------------------------------------------------------------------
# 8. Sampling shape + SO(3) validity
# ---------------------------------------------------------------------------
def test_sampling_shape_and_so3():
    torch.manual_seed(9)
    b, h, f, din, nj = 4, 25, 25, 60, 21
    d = target_state_dim(nj)
    model = FlowMatchingTransformer(din, d, h, f, d_model=32, nhead=4,
                                    num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64)
    model.eval()
    history = torch.randn(b, h, din)
    traj = sample_trajectory(model, history, f, d, num_steps=8)
    assert traj.shape == (b, f, d)
    o0, o1 = orientation_slice(nj)
    r = rot6d_to_matrix(traj[..., o0:o1])
    eye = torch.eye(3).expand_as(r)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. Windowing logic
# ---------------------------------------------------------------------------
def test_episode_windows_respect_boundaries():
    # episode 0: steps 0..4 (done on last). episode 1: steps 0..2 (done on last).
    episode_id = np.array([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    step = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
    done = np.array([0, 0, 0, 0, 1, 0, 0, 1], dtype=np.int64)
    wins = episode_windows(episode_id, step, done, window_len=3, stride=1)
    for w in wins:
        assert len(w) == 3
        assert len(set(episode_id[w])) == 1            # one episode
        assert np.all(np.diff(step[w]) == 1)           # consecutive
        assert np.all(done[w][:-1] == 0)               # no internal done
    # episode 0 -> 3 windows (starts 0,1,2), episode 1 -> 1 window. Total 4.
    assert len(wins) == 4


# ---------------------------------------------------------------------------
# Dataset end-to-end build with synthetic recorded-style data
# ---------------------------------------------------------------------------
def _synthetic_dataset(num_envs=4, steps=40, nj=21, scan_dim=150, with_ref=False, frame_names=None):
    """Build a synthetic (data, meta) pair shaped like a recorded .npz.

    If ``with_ref`` is set, a distinct ``ref_state`` array (same layout as
    ``state``, unit quaternions) is added so the reference target source can be
    exercised. ``frame_names`` appends extra body frames (each
    ``pos_w(3), quat_w(4), lin_vel_w(3), ang_vel_w(3)``) to the state layout.
    """
    rng = np.random.default_rng(0)
    n = num_envs * steps
    frame_names = list(frame_names or [])
    # obs layout: velocity_commands(3), height_scan(scan_dim), joint_pos(nj)
    obs_term_names = ["velocity_commands", "height_scan", "joint_pos"]
    obs_term_dims = [3, scan_dim, nj]
    obs = rng.standard_normal((n, sum(obs_term_dims))).astype(np.float32)
    # state layout matches the recorder (root + joints, then any extra frames)
    state_term_names = ["root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"]
    state_term_dims = [3, 4, 3, 3, nj, nj]
    for f in frame_names:
        state_term_names += [f"{f}:pos_w", f"{f}:quat_w", f"{f}:lin_vel_w", f"{f}:ang_vel_w"]
        state_term_dims += [3, 4, 3, 3]

    # Offsets of every quaternion block (root + per-frame), to unit-normalize them.
    quat_blocks: list[tuple[int, int]] = []
    off = 0
    for name, dim in zip(state_term_names, state_term_dims):
        if name == "root_quat_w" or name.endswith(":quat_w"):
            quat_blocks.append((off, off + dim))
        off += dim

    def _make_state(seed: int) -> np.ndarray:
        s = np.random.default_rng(seed).standard_normal((n, sum(state_term_dims))).astype(np.float32)
        for a, b in quat_blocks:
            qb = s[:, a:b]
            s[:, a:b] = qb / np.linalg.norm(qb, axis=1, keepdims=True)  # unit quaternion block
        return s

    state = _make_state(0)
    # episode_id / step_in_episode: one episode per env, steps 0..steps-1
    episode_id = np.repeat(np.arange(num_envs), steps).astype(np.int64)
    step_in_episode = np.tile(np.arange(steps), num_envs).astype(np.int64)
    done = (step_in_episode == steps - 1).astype(np.uint8)
    data = {
        "obs": obs, "state": state, "episode_id": episode_id,
        "step_in_episode": step_in_episode, "done": done,
    }
    if with_ref:
        data["ref_state"] = _make_state(1)  # distinct from the actual state
    meta = SimpleNamespace(
        obs_term_names=obs_term_names, obs_term_dims=obs_term_dims,
        state_term_names=state_term_names, state_term_dims=state_term_dims,
        state_frame_names=frame_names,
    )
    return data, meta


def test_dataset_build_shapes():
    # episodes must be at least history_len + future_len (= 50) long to yield windows
    data, meta = _synthetic_dataset(num_envs=6, steps=60, nj=21, scan_dim=150)
    cfg = FlowMatchingConfig(history_len=25, future_len=25, val_fraction=0.34, seed=0)
    train_ds, val_ds = build_train_val(data, meta, cfg)
    assert cfg.state_dim == target_state_dim(21)         # 57
    assert cfg.history_in_dim == 57 + 3 + 150            # state + vel_cmd + height_scan
    sample = train_ds[0]
    assert sample["history"].shape == (25, cfg.history_in_dim)
    assert sample["future"].shape == (25, 57)
    assert sample["p_ref"].shape == (3,) and sample["q_ref"].shape == (4,)
    assert val_ds is not None and len(val_ds) > 0
    # episode-disjoint split
    # (rebuild windows to map back is overkill; rely on build_train_val's split-by-episode)


def test_dataset_build_with_extra_frames():
    """Dataset resolves extra-frame state terms and widens state_dim accordingly."""
    frames = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    k = len(frames)
    data, meta = _synthetic_dataset(num_envs=6, steps=60, nj=21, scan_dim=8, frame_names=frames)
    cfg = FlowMatchingConfig(history_len=25, future_len=25, val_fraction=0.34, seed=0)
    train_ds, val_ds = build_train_val(data, meta, cfg)
    assert train_ds.num_extra_frames == k
    assert cfg.state_dim == target_state_dim(21, k)            # 57 + 4*15 = 117
    assert cfg.history_in_dim == target_state_dim(21, k) + 3 + 8  # state + vel_cmd + scan
    sample = train_ds[0]
    assert sample["history"].shape == (25, cfg.history_in_dim)
    assert sample["future"].shape == (25, target_state_dim(21, k))


def _split_state(rw, s_slc, name):
    """Slice a named state term out of a windowed state tensor."""
    a, b = s_slc[name]
    return rw[..., a:b]


def test_target_source_reference_uses_ref_state():
    """`target_source='reference'` predicts ref_state while keeping the actual
    history/anchor; `'actual'` predicts state. Both modes share history/p_ref/q_ref."""
    nj = 21
    data, meta = _synthetic_dataset(num_envs=4, steps=60, nj=nj, scan_dim=8, with_ref=True)
    h, f = 25, 25
    windows = episode_windows(
        data["episode_id"], data["step_in_episode"], data["done"], h + f, stride=1
    )
    assert len(windows) > 0

    cfg_a = FlowMatchingConfig(history_len=h, future_len=f, target_source="actual", seed=0)
    cfg_r = FlowMatchingConfig(history_len=h, future_len=f, target_source="reference", seed=0)
    ds_a = FutureStateDataset(data, meta, cfg_a, windows)
    ds_r = FutureStateDataset(data, meta, cfg_r, windows)

    # History conditioning and the relative-frame anchor are identical across modes.
    torch.testing.assert_close(ds_a.history, ds_r.history)
    torch.testing.assert_close(ds_a.p_ref, ds_r.p_ref)
    torch.testing.assert_close(ds_a.q_ref, ds_r.q_ref)

    # Targets differ because ref_state != state.
    fut_a = ds_a.target_normalizer.denormalize(ds_a.future)
    fut_r = ds_r.target_normalizer.denormalize(ds_r.future)
    assert not torch.allclose(fut_a, fut_r)

    # The reference future is relative_state(ref_state future rows) anchored at the
    # *actual* pose at step h-1. Verify directly on window 0.
    s_names = list(meta.state_term_names)
    s_dims = list(meta.state_term_dims)
    s_slc = {name: resolve_term_slice(s_names, s_dims, name) for name in s_names}
    w0 = windows[0]
    ref_w = torch.from_numpy(data["ref_state"][w0]).float()       # (W, raw)
    state_w = torch.from_numpy(data["state"][w0]).float()
    p_ref = state_w[h - 1 : h, s_slc["root_pos_w"][0]:s_slc["root_pos_w"][1]].expand(f, 3)
    q_ref = state_w[h - 1 : h, s_slc["root_quat_w"][0]:s_slc["root_quat_w"][1]].expand(f, 4)
    fut = ref_w[h:]
    expected = relative_state(
        _split_state(fut, s_slc, "root_pos_w"), _split_state(fut, s_slc, "root_quat_w"),
        _split_state(fut, s_slc, "root_lin_vel_w"), _split_state(fut, s_slc, "root_ang_vel_w"),
        _split_state(fut, s_slc, "joint_pos"), _split_state(fut, s_slc, "joint_vel"),
        p_ref, q_ref,
    )
    torch.testing.assert_close(fut_r[0], expected, rtol=1e-4, atol=1e-4)


def test_target_source_reference_requires_ref_state():
    """Selecting the reference target without a recorded ref_state raises."""
    data, meta = _synthetic_dataset(num_envs=4, steps=60, nj=21, scan_dim=8, with_ref=False)
    cfg = FlowMatchingConfig(history_len=25, future_len=25, target_source="reference", seed=0)
    windows = episode_windows(
        data["episode_id"], data["step_in_episode"], data["done"], 50, stride=1
    )
    with pytest.raises(ValueError, match="ref_state"):
        FutureStateDataset(data, meta, cfg, windows)
