# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator-free unit tests for the flow-matching subpackage.

Run with: ``pytest source/robot_rl/robot_rl/flow_matching/tests`` inside the
``env_isaaclab`` venv. All tests use synthetic tensors; no Isaac Sim app needed.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab.utils.math import matrix_from_quat

from robot_rl.flow_matching import (
    FKChain,
    FlatTerrain,
    FlowMatchingConfig,
    FlowMatchingTransformer,
    FutureStateDataset,
    Normalizer,
    box_sole_points,
    build_train_val,
    episode_windows,
    filter_windows_by_recovery,
    fk_consistency_terms,
    flow_matching_loss,
    frame_block_start,
    frame_consistency_terms,
    joint_vel_consistency_term,
    matrix_from_quat,
    matrix_to_rot6d,
    terrain_penetration_term,
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
    sample_trajectory_grad,
    so3_log,
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
    jp = torch.randn(w, nj, dtype=torch.float64)
    ref = w // 2

    rel = relative_state_from_window(p, q, jp, ref_index=ref)
    assert rel.shape == (w, target_state_dim(nj))

    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rec = state_to_absolute(rel, p_ref, q_ref, nj)

    assert torch.allclose(rec["joint_pos"], jp, atol=1e-6)
    assert torch.allclose(rec["root_pos_w"], p, atol=1e-5)
    # quaternions compared as matrices (double cover)
    assert torch.allclose(matrix_from_quat(rec["root_quat_w"]), matrix_from_quat(q), atol=1e-5)
    # position-only state: no velocity keys are decoded
    assert "joint_vel" not in rec and "root_link_lin_vel_w" not in rec


def test_relative_state_invertible_with_frames():
    """Round-trip relative_state <-> state_to_absolute with extra body frames."""
    torch.manual_seed(12)
    nj, w, k = 21, 30, 4
    p = torch.randn(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    jp = torch.randn(w, nj, dtype=torch.float64)
    fpos = torch.randn(w, k, 3, dtype=torch.float64)
    fquat = _rand_unit_quat(w * k).reshape(w, k, 4)
    ref = w // 2

    p_ref = p[ref : ref + 1].expand(w, -1)
    q_ref = q[ref : ref + 1].expand(w, -1)
    rel = relative_state(
        p, q, jp, p_ref, q_ref,
        frame_pos_w=fpos, frame_quat_w=fquat,
    )
    assert rel.shape == (w, target_state_dim(nj, k))
    # every ori block (base + frames) is left raw (not standardized) -> there are k+1
    assert len(orientation_slices(nj, k)) == k + 1

    rec = state_to_absolute(rel, p_ref, q_ref, nj, num_extra_frames=k)
    assert torch.allclose(rec["root_pos_w"], p, atol=1e-5)
    assert rec["frame_pos_w"].shape == (w, k, 3)
    assert torch.allclose(rec["frame_pos_w"], fpos, atol=1e-5)
    assert torch.allclose(
        matrix_from_quat(rec["frame_quat_w"].reshape(w * k, 4)),
        matrix_from_quat(fquat.reshape(w * k, 4)),
        atol=1e-5,
    )
    assert "frame_lin_vel_w" not in rec and "frame_ang_vel_w" not in rec


def test_relative_state_reference_is_identity():
    torch.manual_seed(3)
    nj, w = 21, 10
    p = torch.randn(w, 3, dtype=torch.float64)
    q = _rand_unit_quat(w)
    jp = torch.randn(w, nj, dtype=torch.float64)
    ref = 7
    rel = relative_state_from_window(p, q, jp, ref_index=ref)
    o0, o1 = orientation_slice(nj)
    p_rel = rel[ref, nj : nj + 3]
    ori6d = rel[ref, o0:o1]
    assert torch.allclose(p_rel, torch.zeros(3, dtype=torch.float64), atol=1e-6)
    expected_ori = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    assert torch.allclose(ori6d, expected_ori, atol=1e-6)


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
# 5. Masked flow batch math
# ---------------------------------------------------------------------------
def test_flow_batch_masked():
    torch.manual_seed(6)
    b, f, d, n = 8, 10, 57, 2
    x1 = torch.randn(b, f, d)
    cond = torch.randn(b, f, d)
    known = torch.zeros(b, f, dtype=torch.bool)
    known[:, :n] = True
    x_tau, tau, target_v = sample_flow_batch(x1, cond, known)
    assert x_tau.shape == x1.shape and target_v.shape == x1.shape and tau.shape == (b,)
    # Known tokens are held at the observed cond values.
    assert torch.allclose(x_tau[:, :n], cond[:, :n])
    # Unknown tokens follow the linear noise->data path.
    x0 = x1 - target_v
    t = tau.view(-1, 1, 1)
    interp = (1 - t) * x0 + t * x1
    assert torch.allclose(x_tau[:, n:], interp[:, n:], atol=1e-6)
    assert torch.allclose(target_v, x1 - x0, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Model shapes + full (non-causal) self-attention over the token sequence
# ---------------------------------------------------------------------------
def test_model_shapes_and_full_attention():
    torch.manual_seed(7)
    b, f, d, cmd_dim, stride = 3, 8, 57, 3, 5
    model = FlowMatchingTransformer(
        state_dim=d, future_len=f, cmd_dim=cmd_dim, cmd_stride=stride,
        d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
    )
    model.eval()
    n_cmd = len(range(0, f, stride))  # 2
    x_tau = torch.randn(b, f, d)
    tau = torch.rand(b)
    cmd_tokens = torch.randn(b, n_cmd, cmd_dim)
    with torch.no_grad():
        v0 = model(x_tau, tau, cmd_tokens)
        assert v0.shape == (b, f, d)
        # perturb state token 0 -> output at token 1 must change (full self-attn)
        x_pert = x_tau.clone()
        x_pert[:, 0, :] += 5.0
        v1 = model(x_pert, tau, cmd_tokens)
    assert not torch.allclose(v0[:, 1, :], v1[:, 1, :], atol=1e-5)


def test_model_no_command():
    """A model built with cmd_dim=0 runs with cmd_tokens=None."""
    torch.manual_seed(7)
    b, f, d = 2, 6, 8
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    model.eval()
    with torch.no_grad():
        v = model(torch.randn(b, f, d), torch.rand(b), None)
    assert v.shape == (b, f, d)


# ---------------------------------------------------------------------------
# 7. Masked inpainting: known tokens preserved; loss ignores them
# ---------------------------------------------------------------------------
def test_sample_trajectory_preserves_known():
    torch.manual_seed(9)
    b, f, n, nj = 4, 10, 2, 21
    d = target_state_dim(nj)
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=3, cmd_stride=5,
                                    d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    model.eval()
    cond = torch.randn(b, f, d)
    known = torch.zeros(b, f, dtype=torch.bool)
    known[:, :n] = True
    cmd_tokens = torch.randn(b, len(range(0, f, 5)), 3)
    traj = sample_trajectory(model, cond, known, f, d, num_steps=8, cmd_tokens=cmd_tokens)
    assert traj.shape == (b, f, d)
    # Known tokens are reproduced bit-exactly (re-clamped every Euler step).
    assert torch.equal(traj[:, :n], cond[:, :n])
    # ori6d block is Gram-Schmidt-able into a valid rotation for every token.
    o0, o1 = orientation_slice(nj)
    r = rot6d_to_matrix(traj[..., o0:o1])
    eye = torch.eye(3).expand_as(r)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-5)


def test_loss_ignores_known_tokens():
    torch.manual_seed(10)
    b, f, d, n = 4, 10, 12, 3
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    model.eval()
    x1 = torch.randn(b, f, d)
    cond = torch.randn(b, f, d)
    known = torch.zeros(b, f, dtype=torch.bool)
    known[:, :n] = True
    torch.manual_seed(123)
    l0 = flow_matching_loss(model, x1, cond, known)["flow_mse"].item()
    x1b = x1.clone()
    x1b[:, :n] += 10.0  # perturb the *known* tokens' targets only
    torch.manual_seed(123)
    l1 = flow_matching_loss(model, x1b, cond, known)["flow_mse"].item()
    assert abs(l0 - l1) < 1e-6


def test_unconditional_all_false_mask():
    torch.manual_seed(11)
    b, f, nj = 2, 8, 21
    d = target_state_dim(nj)
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=32, nhead=4, num_layers=2, dim_feedforward=64)
    model.eval()
    cond = torch.randn(b, f, d)
    known = torch.zeros(b, f, dtype=torch.bool)  # drop history entirely
    traj = sample_trajectory(model, cond, known, f, d, num_steps=4)
    assert traj.shape == (b, f, d)
    # Nothing is clamped -> cond is ignored.
    assert not torch.equal(traj[:, 0], cond[:, 0])


# ---------------------------------------------------------------------------
# 8. Tiny synthetic training drives the (masked) loss down
# ---------------------------------------------------------------------------
def test_tiny_training_reduces_loss():
    torch.manual_seed(8)
    num, f, d, n = 256, 6, 8, 2
    # Future tokens are a deterministic affine function of the conditioning tokens,
    # so conditioning carries real signal the model can exploit.
    cond_part = torch.randn(num, n, d)
    w_map = torch.randn(n * d, (f - n) * d) * 0.1
    future_part = (cond_part.reshape(num, -1) @ w_map).reshape(num, f - n, d)
    state = torch.cat([cond_part, future_part], dim=1)
    state = (state - state.mean(0)) / state.std(0).clamp_min(1e-6)
    cond = state.clone()
    known = torch.zeros(num, f, dtype=torch.bool)
    known[:, :n] = True

    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(8)
    init_loss = flow_matching_loss(model, state, cond, known)["loss"].item()
    for _ in range(300):
        opt.zero_grad()
        loss = flow_matching_loss(model, state, cond, known)["loss"]
        loss.backward()
        opt.step()
    final_loss = flow_matching_loss(model, state, cond, known)["loss"].item()
    assert final_loss < 0.5 * init_loss, f"init={init_loss:.3f} final={final_loss:.3f}"


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


def test_filter_windows_by_recovery():
    """clean_prediction keeps clean + bad-history windows, drops perturbed predictions."""
    n_history = 2
    # 8 contiguous steps; perturbation active on steps 2 and 3.
    perturb = np.zeros(8, dtype=np.uint8)
    perturb[2:4] = 1
    windows = [np.arange(s, s + 4) for s in range(0, 5)]  # length-4 windows, starts 0..4

    # mode='all' is a no-op.
    assert filter_windows_by_recovery(windows, perturb, n_history, "all") == windows
    # No flag array -> no-op regardless of mode.
    assert filter_windows_by_recovery(windows, None, n_history, "clean_prediction") == windows

    kept = filter_windows_by_recovery(windows, perturb, n_history, "clean_prediction")
    kept_starts = {int(w[0]) for w in kept}
    # Prediction region is the window's steps [2,3] (absolute [start+2, start+3]).
    #   start 0: prediction {2,3} -> overlaps perturb -> dropped.
    #   start 1: prediction {3,4} -> overlaps 3       -> dropped.
    #   start 2: prediction {4,5} -> clean            -> kept (history {2,3} IS perturbed).
    #   start 3: prediction {5,6} -> clean            -> kept.
    #   start 4: prediction {6,7} -> clean            -> kept.
    assert kept_starts == {2, 3, 4}, kept_starts
    # The bad-state-history -> clean-recovery window (start 2) is exactly what we want kept.
    assert any(int(w[0]) == 2 for w in kept)


def test_filter_windows_by_recovery_bad_mode():
    """An unrecognized mode raises (with a flag array present)."""
    with pytest.raises(ValueError):
        filter_windows_by_recovery([np.arange(4)], np.zeros(8, np.uint8), 2, "bogus")


def test_build_train_val_recovery_mode():
    """build_train_val honors recovery_mode: clean_prediction drops perturbed-prediction windows."""
    data, meta = _synthetic_dataset(num_envs=6, steps=30, nj=21)
    # Mark a mid-episode perturbation burst in every env (steps 10..14 of each).
    perturb = (data["step_in_episode"] >= 10) & (data["step_in_episode"] < 15)
    data["perturb_active"] = perturb.astype(np.uint8)

    cfg_all = FlowMatchingConfig(future_len=10, num_history_states=2, recovery_mode="all", seed=0)
    train_all, _ = build_train_val(data, meta, cfg_all)

    cfg_clean = FlowMatchingConfig(future_len=10, num_history_states=2,
                                   recovery_mode="clean_prediction", seed=0)
    train_clean, _ = build_train_val(data, meta, cfg_clean)

    # Filtering can only remove windows.
    assert len(train_clean) < len(train_all)
    assert len(train_clean) > 0


# ---------------------------------------------------------------------------
# Dataset end-to-end build with synthetic recorded-style data
# ---------------------------------------------------------------------------
def _synthetic_dataset(num_envs=4, steps=40, nj=21, scan_dim=8, with_ref=False, frame_names=None):
    """Build a synthetic (data, meta) pair shaped like a recorded .npz.

    If ``with_ref`` is set, a distinct ``ref_state`` array (same layout as
    ``state``, unit quaternions) is added so the reference target source can be
    exercised. ``frame_names`` appends extra body frames (each
    ``pos_w(3), quat_w(4), lin_vel_w(3), ang_vel_w(3)``) to the state layout.
    """
    rng = np.random.default_rng(0)
    n = num_envs * steps
    frame_names = list(frame_names or [])
    # obs layout: velocity_commands(3), height_scan(scan_dim), joint_pos(nj). The obs
    # is unused by the dataset now (perception is not wired into the token layout).
    obs_term_names = ["velocity_commands", "height_scan", "joint_pos"]
    obs_term_dims = [3, scan_dim, nj]
    obs = rng.standard_normal((n, sum(obs_term_dims))).astype(np.float32)
    # state layout matches the recorder (root + joints, then any extra frames)
    state_term_names = ["root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_link_ang_vel_w", "joint_pos", "joint_vel"]
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
        # Recorded joystick command array (width 3); only used when a test sets
        # cfg.cmd_array="target_vel" (otherwise ignored, cmd_dim == 0).
        "target_vel": rng.standard_normal((n, 3)).astype(np.float32),
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
    data, meta = _synthetic_dataset(num_envs=6, steps=30, nj=21)
    cfg = FlowMatchingConfig(future_len=10, num_history_states=2, cmd_stride=5,
                             val_fraction=0.34, seed=0, cmd_array="target_vel")
    train_ds, val_ds = build_train_val(data, meta, cfg)
    assert cfg.state_dim == target_state_dim(21)  # 30 (= nj + 9, position-only, K=0)
    assert cfg.cmd_dim == 3
    assert cfg.num_cmd_tokens == 2  # ceil(10/5)
    s = train_ds[0]
    assert s["state"].shape == (10, target_state_dim(21))
    assert s["cond"].shape == (10, target_state_dim(21))
    assert s["cmd_tokens"].shape == (2, 3)
    assert s["p_ref"].shape == (3,) and s["q_ref"].shape == (4,)
    assert val_ds is not None and len(val_ds) > 0


def test_dataset_build_with_extra_frames():
    """Dataset resolves extra-frame state terms and widens state_dim accordingly.

    History and target share the single full-state representation (every recorded
    frame), so both ``state`` and ``cond`` carry the extra frames.
    """
    frames = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    k = len(frames)
    data, meta = _synthetic_dataset(num_envs=6, steps=30, nj=21, frame_names=frames)
    cfg = FlowMatchingConfig(future_len=10, num_history_states=2, val_fraction=0.34, seed=0,
                             cmd_array="target_vel")
    train_ds, _ = build_train_val(data, meta, cfg)
    assert train_ds.num_extra_frames == k
    assert cfg.state_dim == target_state_dim(21, k)  # 30 + 4*9 = 66 (position-only)
    s = train_ds[0]
    assert s["state"].shape == (10, target_state_dim(21, k))
    assert s["cond"].shape == (10, target_state_dim(21, k))
    assert s["cmd_tokens"].shape == (2, 3)


def test_online_roundtrip_with_frames():
    """Build a checkpoint and confirm the online context round-trips conditioning
    -> masked sampling -> world-frame prediction (history carries the EE frames the
    model predicts)."""
    from robot_rl.flow_matching.online import build_cmd_tokens, build_cond_states, load_fm_context, sample_world

    nj = 21
    frames = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    k = len(frames)
    data, meta = _synthetic_dataset(num_envs=6, steps=30, nj=nj, frame_names=frames)
    cfg = FlowMatchingConfig(
        future_len=10, num_history_states=2, cmd_stride=5, val_fraction=0.34, seed=0,
        cmd_array="target_vel", d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
    )
    train_ds, _ = build_train_val(data, meta, cfg)
    assert cfg.state_dim == target_state_dim(nj, k)

    model = FlowMatchingTransformer(
        state_dim=cfg.state_dim, future_len=cfg.future_len, cmd_dim=cfg.cmd_dim,
        cmd_stride=cfg.cmd_stride, d_model=cfg.d_model, nhead=cfg.nhead,
        num_layers=cfg.num_layers, dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
    )
    ckpt = {
        "config": cfg.to_dict(),
        "meta": {"state_joint_names": [f"j{i}" for i in range(nj)], "target_frame_names": frames},
        "num_joints": nj,
        "model": model.state_dict(),
        "state_normalizer": train_ds.state_normalizer.state_dict(),
        "cmd_normalizer": train_ds.cmd_normalizer.state_dict(),
    }
    ctx = load_fm_context(ckpt, device="cpu")
    assert ctx.N == 2 and ctx.F == 10 and ctx.D == target_state_dim(nj, k)
    assert ctx.target_num_extra_frames == k and ctx.frame_names == frames
    assert ctx.cmd_dim == 3 and ctx.num_cmd_tokens == 2

    v, nstate = 2, ctx.N
    fr = torch.randn(v, nstate, k, 13)
    fr[..., 3:7] = fr[..., 3:7] / fr[..., 3:7].norm(dim=-1, keepdim=True)  # unit quats
    cond, known = build_cond_states(
        ctx,
        root_pos=torch.randn(v, nstate, 3),
        root_quat=_rand_unit_quat(v * nstate).reshape(v, nstate, 4).float(),
        root_lin_vel=torch.randn(v, nstate, 3), root_ang_vel=torch.randn(v, nstate, 3),
        joint_pos=torch.randn(v, nstate, nj), joint_vel=torch.randn(v, nstate, nj),
        frames=fr,
    )
    assert cond.shape == (v, ctx.F, ctx.D) and known.shape == (v, ctx.F)
    assert known[:, :nstate].all() and not known[:, nstate:].any()

    cmd_tokens = build_cmd_tokens(ctx, torch.randn(v, ctx.F, 3))
    assert cmd_tokens.shape == (v, ctx.num_cmd_tokens, 3)

    world = sample_world(
        ctx, cond, known, p_ref=torch.randn(v, 3), q_ref=_rand_unit_quat(v).float(),
        cmd_tokens=cmd_tokens,
    )
    assert world["root_pos_w"].shape == (v, ctx.F, 3)
    assert world["frame_pos_w"].shape == (v, ctx.F, k, 3)  # target frames are predicted


def _split_state(rw, s_slc, name):
    """Slice a named state term out of a windowed state tensor."""
    a, b = s_slc[name]
    return rw[..., a:b]


def test_target_source_reference_uses_ref_state():
    """`target_source='reference'` predicts ref_state while conditioning on the
    actual state; `'actual'` predicts the actual state. Both share the conditioning."""
    nj = 21
    data, meta = _synthetic_dataset(num_envs=4, steps=30, nj=nj, with_ref=True)
    f, n = 10, 2
    windows = episode_windows(
        data["episode_id"], data["step_in_episode"], data["done"], f, stride=1
    )
    assert len(windows) > 0

    cfg_a = FlowMatchingConfig(future_len=f, num_history_states=n, target_source="actual", seed=0)
    cfg_r = FlowMatchingConfig(future_len=f, num_history_states=n, target_source="reference", seed=0)
    ds_a = FutureStateDataset(data, meta, cfg_a, windows)
    ds_r = FutureStateDataset(data, meta, cfg_r, windows)

    # The actual-state conditioning (raw) and the relative-frame anchor are identical.
    cond_a_raw = ds_a.state_normalizer.denormalize(ds_a.cond)
    cond_r_raw = ds_r.state_normalizer.denormalize(ds_r.cond)
    torch.testing.assert_close(cond_a_raw, cond_r_raw, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ds_a.p_ref, ds_r.p_ref)
    torch.testing.assert_close(ds_a.q_ref, ds_r.q_ref)

    # Targets differ because ref_state != state.
    fut_a = ds_a.state_normalizer.denormalize(ds_a.state)
    fut_r = ds_r.state_normalizer.denormalize(ds_r.state)
    assert not torch.allclose(fut_a, fut_r)

    # The reference target is relative_state(ref over all F) anchored at the *actual*
    # pose at step N-1. Verify directly on window 0.
    s_names = list(meta.state_term_names)
    s_dims = list(meta.state_term_dims)
    s_slc = {name: resolve_term_slice(s_names, s_dims, name) for name in s_names}
    w0 = windows[0]
    ref_w = torch.from_numpy(data["ref_state"][w0]).float()       # (F, raw)
    state_w = torch.from_numpy(data["state"][w0]).float()
    anchor = n - 1
    p_ref = state_w[anchor : anchor + 1, s_slc["root_pos_w"][0]:s_slc["root_pos_w"][1]].expand(f, 3)
    q_ref = state_w[anchor : anchor + 1, s_slc["root_quat_w"][0]:s_slc["root_quat_w"][1]].expand(f, 4)
    expected = relative_state(
        _split_state(ref_w, s_slc, "root_pos_w"), _split_state(ref_w, s_slc, "root_quat_w"),
        _split_state(ref_w, s_slc, "joint_pos"),
        p_ref, q_ref,
    )
    torch.testing.assert_close(fut_r[0], expected, rtol=1e-4, atol=1e-4)


def test_conditioning_observation_noise():
    """Train-time obs noise perturbs the conditioning tokens (first N) only,
    resampled per access; the target, anchors, command, and later tokens stay clean."""
    n = 2
    data, meta = _synthetic_dataset(num_envs=6, steps=30, nj=21)
    cfg = FlowMatchingConfig(future_len=10, num_history_states=n, val_fraction=0.34, seed=0,
                             cmd_array="target_vel")  # obs_noise defaults on
    train_ds, val_ds = build_train_val(data, meta, cfg)
    assert train_ds.add_noise is True
    assert val_ds is not None and val_ds.add_noise is False

    torch.manual_seed(0)
    s1, s2 = train_ds[0], train_ds[0]
    # cond is resampled each access -> the conditioning tokens differ.
    assert not torch.allclose(s1["cond"], s2["cond"])
    # Only the first N tokens are noised; later tokens and the target stay identical.
    torch.testing.assert_close(s1["cond"][n:], s2["cond"][n:])
    torch.testing.assert_close(s1["state"], s2["state"])
    torch.testing.assert_close(s1["cmd_tokens"], s2["cmd_tokens"])

    # Amplitude-vector mapping: root ori6d block is raw (std==1) so it equals the
    # configured amplitude exactly; joint channels are noised.
    o0, o1 = orientation_slice(train_ds.num_joints)
    torch.testing.assert_close(train_ds.state_noise[o0:o1],
                               torch.full((6,), cfg.noise_ori, dtype=train_ds.state_noise.dtype))
    assert torch.all(train_ds.state_noise[:train_ds.num_joints] > 0)  # joint_pos

    # The val split (and any add_noise=False dataset) is deterministic.
    torch.testing.assert_close(val_ds[0]["cond"], val_ds[0]["cond"])

    # Master flag off -> no noise even though build_train_val requests add_noise.
    cfg_off = FlowMatchingConfig(future_len=10, num_history_states=n, val_fraction=0.34, seed=0,
                                 cmd_array="target_vel", obs_noise=False)
    ds_off, _ = build_train_val(data, meta, cfg_off)
    assert ds_off.add_noise is False and ds_off.state_noise is None
    torch.testing.assert_close(ds_off[0]["cond"], ds_off[0]["cond"])


def test_target_source_reference_requires_ref_state():
    """Selecting the reference target without a recorded ref_state raises."""
    data, meta = _synthetic_dataset(num_envs=4, steps=30, nj=21, with_ref=False)
    cfg = FlowMatchingConfig(future_len=10, num_history_states=2, target_source="reference", seed=0)
    windows = episode_windows(
        data["episode_id"], data["step_in_episode"], data["done"], 10, stride=1
    )
    with pytest.raises(ValueError, match="ref_state"):
        FutureStateDataset(data, meta, cfg, windows)


# ---------------------------------------------------------------------------
# Consistency regularization: SO(3) log + frame velocity/acceleration losses
# ---------------------------------------------------------------------------
def _rodrigues(w: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(B, 3, 3)`` from rotation vectors ``w`` (B, 3) (Rodrigues)."""
    theta = w.norm(dim=-1, keepdim=True)               # (B, 1)
    k = w / theta.clamp_min(1e-12)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    K = torch.zeros(w.shape[0], 3, 3, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2] = -kz, ky
    K[:, 1, 0], K[:, 1, 2] = kz, -kx
    K[:, 2, 0], K[:, 2, 1] = -ky, kx
    eye = torch.eye(3, dtype=w.dtype).expand(w.shape[0], 3, 3)
    th = theta[:, :, None]                              # (B, 1, 1)
    return eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)


def test_so3_log_recovers_rotation_vector():
    torch.manual_seed(20)
    # mix of tiny (near-identity, the gradient-singular regime), moderate, and large angles
    dirs = torch.randn(200, 3, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    angles = torch.cat([
        torch.full((100,), 1e-7, dtype=torch.float64),   # essentially identity
        torch.rand(100, dtype=torch.float64) * 2.5,       # up to ~143 deg
    ])
    w = dirs * angles.unsqueeze(-1)
    rec = so3_log(_rodrigues(w))
    assert torch.allclose(rec, w, atol=1e-7), float((rec - w).abs().max())
    assert torch.isfinite(so3_log(_rodrigues(w))).all()


def _consistent_frame_traj(nf: int, nj: int, k: int, dt: float, seed: int) -> torch.Tensor:
    """A smooth position-only ``(1, F, D)`` trajectory: each frame's position ramps at a
    constant linear velocity, its orientation integrates a constant angular velocity, and
    the joints ramp linearly. The FD-match consistency terms compare FD(prediction) to
    FD(data), so passing this same trajectory as both gives ~0 residual."""
    g = torch.Generator().manual_seed(seed)
    d = target_state_dim(nj, k)
    x = torch.zeros(1, nf, d, dtype=torch.float64)
    jp0 = torch.randn(nj, generator=g, dtype=torch.float64)
    jvel = torch.randn(nj, generator=g, dtype=torch.float64) * 0.5
    for t in range(nf):
        x[0, t, 0:nj] = jp0 + jvel * (t * dt)
    for fi in range(1 + k):
        b = frame_block_start(nj, fi)
        pos0 = torch.randn(3, generator=g, dtype=torch.float64)
        vel = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
        omega = torch.randn(3, generator=g, dtype=torch.float64) * 0.5
        for t in range(nf):
            r = _rodrigues((omega * (t * dt)).unsqueeze(0))[0]   # (3, 3)
            x[0, t, b:b + 3] = pos0 + vel * (t * dt)
            x[0, t, b + 3:b + 9] = matrix_to_rot6d(r)
    return x


def test_frame_consistency_zero_for_consistent_traj():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=1)
    known = torch.zeros(1, nf, dtype=torch.bool)  # all unknown -> consistency on the prediction itself
    vel, accel = frame_consistency_terms(x, x, known, dt, nj, k)  # per-sample (B,)
    jvel = joint_vel_consistency_term(x, x, known, dt, nj)
    # A self-consistent constant-velocity trajectory has ~zero finite-difference residual.
    assert vel.max().item() < 1e-10, vel.max().item()
    assert accel.max().item() < 1e-10, accel.max().item()
    assert jvel.max().item() < 1e-10, jvel.max().item()


def test_frame_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=2)
    known = torch.zeros(1, nf, dtype=torch.bool)
    pred = x.clone()
    pred[0, 3, frame_block_start(nj, 0):frame_block_start(nj, 0) + 3] += 1.0  # jump the root position
    vel, _ = frame_consistency_terms(pred, x, known, dt, nj, k)
    assert vel.max().item() > 1e-2, vel.max().item()  # the position jump breaks pos-FD == velocity


def test_joint_vel_consistency_penalizes_jump():
    nf, nj, k, dt = 6, 2, 1, 0.05
    x = _consistent_frame_traj(nf, nj, k, dt, seed=3)
    known = torch.zeros(1, nf, dtype=torch.bool)
    pred = x.clone()
    pred[0, 3, 0:nj] += 0.5  # jump the joint positions
    assert joint_vel_consistency_term(pred, x, known, dt, nj).max().item() > 1e-2


# ---------------------------------------------------------------------------
# Forward kinematics (torch FK) vs MuJoCo, Jacobian vs FD, and the FK loss
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
def test_fk_matches_mujoco_and_jacobian_vs_fd():
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
    mj_rot = np.zeros((b, len(_FK_FRAMES), 3, 3))
    for i in range(b):
        d.qpos[:] = 0.0
        d.qpos[3:7] = (1, 0, 0, 0)  # pelvis at identity -> xpos/xmat are in the base frame
        for a, j in zip(jadr, qs[i]):
            d.qpos[a] = j
        mujoco.mj_kinematics(m, d)
        for kf, bid in enumerate(bids):
            mj_pos[i, kf] = d.xpos[bid]
            mj_rot[i, kf] = d.xmat[bid].reshape(3, 3)

    q = torch.tensor(qs, dtype=torch.float64)
    pos, rot, jv, jw = chain.pose_and_jacobian(q)
    assert np.abs(pos.numpy() - mj_pos).max() < 1e-6
    assert np.abs(rot.numpy() - mj_rot).max() < 1e-6

    # analytic Jacobian vs central finite difference of the FK pose
    eps = 1e-6
    _, rot0, _, _ = chain.pose_and_jacobian(q)
    jv_fd = torch.zeros_like(jv)
    jw_fd = torch.zeros_like(jw)
    for j in range(nj):
        dq = torch.zeros_like(q)
        dq[:, j] = eps
        pp, rp, _, _ = chain.pose_and_jacobian(q + dq)
        pm, rm, _, _ = chain.pose_and_jacobian(q - dq)
        jv_fd[:, :, :, j] = (pp - pm) / (2 * eps)
        dR = (rp @ rot0.transpose(-1, -2) - rm @ rot0.transpose(-1, -2)) / (2 * eps)
        jw_fd[:, :, :, j] = torch.stack(
            [dR[..., 2, 1] - dR[..., 1, 2], dR[..., 0, 2] - dR[..., 2, 0], dR[..., 1, 0] - dR[..., 0, 1]], -1
        ) * 0.5
    assert (jv - jv_fd).abs().max() < 1e-4
    assert (jw - jw_fd).abs().max() < 1e-4


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_fk_consistency_zero_for_fk_consistent_state():
    """A state whose ee blocks equal FK(joints)/J·qdot (root at identity) has ~0 FK loss."""
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _FK_FRAMES, base_link="pelvis_link", dtype=torch.float64)
    nj, k = len(_FK_JOINTS), len(_FK_FRAMES)
    d = target_state_dim(nj, k)
    nf = 4
    rng = np.random.default_rng(1)
    q = torch.tensor(rng.uniform(-0.5, 0.5, size=(1, nf, nj)), dtype=torch.float64)
    fk_pos, fk_rot, _jv, _jw = chain.pose_and_jacobian(q.reshape(nf, nj))
    fk_pos = fk_pos.reshape(1, nf, k, 3); fk_rot = fk_rot.reshape(1, nf, k, 3, 3)

    x = torch.zeros(1, nf, d, dtype=torch.float64)
    x[..., 0:nj] = q
    # root block: identity rotation, zero position -> base frame == ref frame.
    rb = frame_block_start(nj, 0)
    x[..., rb + 3:rb + 9] = torch.tensor([1.0, 0, 0, 0, 1, 0], dtype=torch.float64)  # ori6d of I
    for kf in range(k):
        eb = frame_block_start(nj, 1 + kf)
        x[..., eb:eb + 3] = fk_pos[:, :, kf]
        x[..., eb + 3:eb + 9] = matrix_to_rot6d(fk_rot[:, :, kf])

    known = torch.zeros(1, nf, dtype=torch.bool)
    pose_ps = fk_consistency_terms(x, x, known, nj, k, chain)
    assert pose_ps.max().item() < 1e-8, pose_ps.max().item()
    # perturb an ee position -> pose loss rises
    xb = x.clone(); xb[0, 1, frame_block_start(nj, 1):frame_block_start(nj, 1) + 3] += 0.1
    pose_b = fk_consistency_terms(xb, xb, known, nj, k, chain)
    assert pose_b.max().item() > 1e-4


@pytest.mark.skipif(not _have_fk_model(), reason="G1 MJCF not found")
def test_flow_loss_fk_terms_and_tau_gate():
    """flow_matching_loss reports fk_pose/joint_vel; tau_min>1 gates accel+FK to 0."""
    torch.manual_seed(30)
    chain = FKChain(_FK_MJCF, _FK_JOINTS, _FK_FRAMES, base_link="pelvis_link")
    nj, k = len(_FK_JOINTS), len(_FK_FRAMES)
    d = target_state_dim(nj, k)
    f, n = 8, 2
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32).eval()
    x1 = torch.randn(16, f, d)
    cond = torch.randn(16, f, d)
    known = torch.zeros(16, f, dtype=torch.bool); known[:, :n] = True
    norm = Normalizer.fit(x1.reshape(-1, d), torch.ones(d, dtype=torch.bool))
    ckw = dict(state_norm=norm, dt=0.05, num_joints=nj, num_extra_frames=k,
               vel_weight=0.1, accel_weight=0.01, joint_vel_weight=0.1,
               fk=chain, fk_pose_weight=0.1)
    out = flow_matching_loss(model, x1, cond, known, None, tau_min=0.0, **ckw)
    for key in ("vel_consistency", "accel_consistency", "joint_vel_consistency", "fk_pose"):
        assert key in out
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    # tau_min above 1 -> no sample passes the gate -> accel + FK contribute 0.
    out_g = flow_matching_loss(model, x1, cond, known, None, tau_min=2.0, **ckw)
    assert out_g["accel_consistency"].item() == 0.0
    assert out_g["fk_pose"].item() == 0.0


# ---------------------------------------------------------------------------
# Terrain-penetration loss
# ---------------------------------------------------------------------------
_IDENT_QUAT = torch.tensor([0.0, 0.0, 0.0, 1.0])           # xyzw
_IDENT_ORI6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # rot6d of identity


def _foot_state(nj: int, k: int, nf: int, foot_pos: torch.Tensor, foot_R: torch.Tensor | None = None) -> torch.Tensor:
    """A (1,nf,D) state with foot frame 0 at ``foot_pos`` (and optional rotation)."""
    d = target_state_dim(nj, k)
    x = torch.zeros(1, nf, d, dtype=torch.float64)
    rb = frame_block_start(nj, 0)
    x[..., rb + 3:rb + 9] = _IDENT_ORI6D.to(torch.float64)         # root ori = identity
    eb = frame_block_start(nj, 1)                                   # first extra (foot) frame
    x[..., eb:eb + 3] = foot_pos.to(torch.float64)
    ori = _IDENT_ORI6D if foot_R is None else matrix_to_rot6d(foot_R)
    x[..., eb + 3:eb + 9] = ori.to(torch.float64)
    return x


def test_box_sole_points():
    pts = box_sole_points(grid=(3, 2))
    assert pts.shape == (6, 3)
    sole_z = -0.02517 - 0.009255  # box center_z - half_z
    assert torch.allclose(pts[:, 2], torch.full((6,), sole_z), atol=1e-6)


def test_flat_terrain_penetration():
    t = FlatTerrain(ground_z=0.0)
    p = torch.tensor([[0.0, 0.0, -0.05], [0.0, 0.0, 0.1]])
    d = t.penetration(p)
    assert torch.allclose(d, torch.tensor([0.05, 0.0]))


def test_terrain_ankle_mode_identity_anchor():
    nj, k, nf = 2, 1, 3
    known = torch.zeros(1, nf, dtype=torch.bool)
    p_ref = torch.zeros(1, 3, dtype=torch.float64)
    q_ref = _IDENT_QUAT.to(torch.float64)[None]
    terrain = FlatTerrain(0.0)
    # foot ankle origin 5 cm below ground -> penetration 0.05 (world z == rel z under identity anchor)
    x_lo = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, -0.05]))
    pen_lo = terrain_penetration_term(x_lo, x_lo, known, p_ref, q_ref, nj, [0], None, terrain)
    assert abs(pen_lo.item() - 0.05 ** 2) < 1e-9
    # foot above ground -> ~0
    x_hi = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, 0.05]))
    assert terrain_penetration_term(x_hi, x_hi, known, p_ref, q_ref, nj, [0], None, terrain).item() < 1e-12


def test_terrain_contact_mode_sole_and_tilt():
    nj, k, nf = 2, 1, 2
    known = torch.zeros(1, nf, dtype=torch.bool)
    p_ref = torch.zeros(1, 3, dtype=torch.float64)
    q_ref = _IDENT_QUAT.to(torch.float64)[None]
    terrain = FlatTerrain(0.0)
    cpts = box_sole_points(grid=(3, 2)).to(torch.float64)
    sole_z = -0.02517 - 0.009255  # box sole offset in the foot frame

    # Flat foot with the ankle origin lifted exactly to +|sole_z| -> sole at ground -> ~0.
    x0 = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, -sole_z]))
    assert terrain_penetration_term(x0, x0, known, p_ref, q_ref, nj, [0], cpts, terrain).item() < 1e-12
    # Same, lowered 3 cm -> sole 3 cm below ground -> positive.
    x1 = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, -sole_z - 0.03]))
    assert terrain_penetration_term(x1, x1, known, p_ref, q_ref, nj, [0], cpts, terrain).item() > 1e-4

    # Tilt value-add: ankle ORIGIN above ground (ankle mode = 0), but a pitched foot's toe
    # dips below ground -> contact mode catches it.
    th = torch.tensor(0.5, dtype=torch.float64)
    R_pitch = torch.tensor([[torch.cos(th), 0, torch.sin(th)],
                            [0.0, 1.0, 0.0],
                            [-torch.sin(th), 0, torch.cos(th)]], dtype=torch.float64)
    x_tilt = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, 0.04]), foot_R=R_pitch)
    ankle_pen = terrain_penetration_term(x_tilt, x_tilt, known, p_ref, q_ref, nj, [0], None, terrain)
    contact_pen = terrain_penetration_term(x_tilt, x_tilt, known, p_ref, q_ref, nj, [0], cpts, terrain)
    assert ankle_pen.item() < 1e-12          # ankle origin (z=+0.04) is above ground
    assert contact_pen.item() > 1e-4         # the tilted toe penetrates


def test_terrain_nonidentity_anchor_lift():
    """Penetration is computed in WORLD z: the anchor height shifts the foot world z."""
    nj, k, nf = 2, 1, 2
    known = torch.zeros(1, nf, dtype=torch.bool)
    q_ref = _IDENT_QUAT.to(torch.float64)[None]
    terrain = FlatTerrain(0.0)
    x = _foot_state(nj, k, nf, torch.tensor([0.0, 0.0, 0.10]))  # foot 10 cm above the anchor
    # anchor 0.2 m up -> foot world z = 0.30 -> no penetration
    assert terrain_penetration_term(x, x, known, torch.tensor([[0.0, 0.0, 0.2]], dtype=torch.float64),
                                    q_ref, nj, [0], None, terrain).item() < 1e-12
    # anchor 0.2 m below ground -> foot world z = -0.10 -> penetration 0.10
    pen = terrain_penetration_term(x, x, known, torch.tensor([[0.0, 0.0, -0.2]], dtype=torch.float64),
                                   q_ref, nj, [0], None, terrain)
    assert abs(pen.item() - 0.10 ** 2) < 1e-9


def test_flow_loss_terrain_term_and_tau_gate():
    torch.manual_seed(40)
    nj, k, f, n = 3, 1, 8, 2
    d = target_state_dim(nj, k)
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32).eval()
    x1 = torch.randn(16, f, d)
    cond = torch.randn(16, f, d)
    known = torch.zeros(16, f, dtype=torch.bool); known[:, :n] = True
    p_ref = torch.randn(16, 3)
    q_ref = torch.randn(16, 4); q_ref = q_ref / q_ref.norm(dim=-1, keepdim=True)
    norm = Normalizer.fit(x1.reshape(-1, d), torch.ones(d, dtype=torch.bool))
    ckw = dict(state_norm=norm, num_joints=nj, num_extra_frames=k, p_ref=p_ref, q_ref=q_ref,
               terrain=FlatTerrain(0.0), foot_idxs=[0], contact_pts=None, terrain_weight=0.1)
    out = flow_matching_loss(model, x1, cond, known, None, tau_min=0.0, **ckw)
    assert "terrain_penetration" in out and torch.isfinite(out["loss"]) and out["loss"].requires_grad
    # Ungated by default: the term survives even when no sample passes the τ-gate.
    out_u = flow_matching_loss(model, x1, cond, known, None, tau_min=2.0, **ckw)
    assert out_u["terrain_penetration"].item() > 0.0
    # Opt-in τ-gating restores the old behavior: tau_min>1 -> nothing contributes.
    out_g = flow_matching_loss(model, x1, cond, known, None, tau_min=2.0, terrain_tau_gated=True, **ckw)
    assert out_g["terrain_penetration"].item() == 0.0  # τ-gated out


def test_sample_trajectory_grad_shape_and_truncated_backprop():
    """sample_trajectory_grad matches sample_trajectory's shape/clamping and only the last
    grad_steps carry gradient (truncated backprop), with grad reaching the model params."""
    torch.manual_seed(41)
    nj, k, f, n = 3, 1, 6, 2
    d = target_state_dim(nj, k)
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32).eval()
    cond = torch.randn(4, f, d)
    known = torch.zeros(4, f, dtype=torch.bool); known[:, :n] = True
    out = sample_trajectory_grad(model, cond, known, f, d, num_steps=4, grad_steps=1)
    assert out.shape == (4, f, d)
    # Known (history) tokens are re-clamped to cond, exactly like sample_trajectory.
    assert torch.allclose(out[:, :n], cond[:, :n])
    # Gradient of the generated sample reaches the model parameters (graph is alive).
    model.zero_grad(set_to_none=True)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    # grad_steps is clamped to >=1: grad_steps=0 still builds a graph (no silent detach-all).
    out0 = sample_trajectory_grad(model, cond, known, f, d, num_steps=4, grad_steps=0)
    assert out0.requires_grad


def test_flow_loss_sampled_rollout_terrain_penalty():
    """terrain_sample_weight>0 adds a differentiable terrain_penetration_sample term; the
    sub-batch caps it and the key is absent when the weight is 0."""
    torch.manual_seed(42)
    nj, k, f, n = 3, 1, 8, 2
    d = target_state_dim(nj, k)
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32).eval()
    x1 = torch.randn(16, f, d)
    cond = torch.randn(16, f, d)
    known = torch.zeros(16, f, dtype=torch.bool); known[:, :n] = True
    p_ref = torch.randn(16, 3)
    q_ref = torch.randn(16, 4); q_ref = q_ref / q_ref.norm(dim=-1, keepdim=True)
    norm = Normalizer.fit(x1.reshape(-1, d), torch.ones(d, dtype=torch.bool))
    base = dict(state_norm=norm, num_joints=nj, num_extra_frames=k, p_ref=p_ref, q_ref=q_ref,
                terrain=FlatTerrain(0.0), foot_idxs=[0], contact_pts=None)
    # Off by default (no x1_hat term, no sample term) -> key absent.
    out_off = flow_matching_loss(model, x1, cond, known, None, **base)
    assert "terrain_penetration_sample" not in out_off
    # Enabled: differentiable, reported, and contributes to the loss.
    out = flow_matching_loss(model, x1, cond, known, None, terrain_sample_weight=10.0,
                             terrain_sample_size=8, terrain_sample_steps=4,
                             terrain_sample_grad_steps=1, **base)
    assert "terrain_penetration_sample" in out
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    assert out["terrain_penetration_sample"].item() >= 0.0
    model.zero_grad(set_to_none=True)
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_flow_matching_loss_consistency_terms_present():
    """flow_matching_loss reports + adds the consistency terms when weighted."""
    torch.manual_seed(21)
    nj, k, f, n = 3, 1, 8, 2
    d = target_state_dim(nj, k)
    model = FlowMatchingTransformer(state_dim=d, future_len=f, cmd_dim=0,
                                    d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    model.eval()
    x1 = torch.randn(4, f, d)
    cond = torch.randn(4, f, d)
    known = torch.zeros(4, f, dtype=torch.bool)
    known[:, :n] = True
    norm = Normalizer.fit(x1.reshape(-1, d), torch.ones(d, dtype=torch.bool))
    out = flow_matching_loss(
        model, x1, cond, known, None, state_norm=norm, dt=0.05,
        num_joints=nj, num_extra_frames=k, vel_weight=0.1, accel_weight=0.01,
    )
    assert "vel_consistency" in out and "accel_consistency" in out
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    # Disabled (zero weights) -> no consistency terms, loss == flow_mse.
    out0 = flow_matching_loss(model, x1, cond, known, None, state_norm=norm, dt=0.05,
                              num_joints=nj, num_extra_frames=k)
    assert "vel_consistency" not in out0
