# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the snap-window streaming recorder's correctness.

Two guarantees, both pure (numpy/torch, no IsaacLab):

1. The online ring-buffer window formation used by ``record_data.collect_windows``
   (a per-env ``valid_count`` reset on episode end, emit when the reanchor flag fires
   and the ring holds ``H`` consecutive valid rows) selects EXACTLY the same snaps as
   ``flow_matching.dataset.snap_windows(num_history=H, stride=1)`` over the equivalent
   per-step recording -- so the two recorders are interchangeable.

2. The windowed FM dataset path (``build_train_val`` on pre-formed windows) produces the
   SAME normalized training tensors as the legacy per-step path over the same rollout.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from robot_rl.flow_matching.config import FlowMatchingConfig
from robot_rl.flow_matching.dataset import build_train_val, snap_windows
from robot_rl.flow_matching.tests.recording_fixtures import episode_arrays


# ---------------------------------------------------------------------------
# 1. Online ring formation == snap_windows
# ---------------------------------------------------------------------------
def form_snap_windows_online(reanchor: np.ndarray, done: np.ndarray, H: int) -> set:
    """Reference of ``collect_windows``' ring: given per-step ``reanchor``/``done``
    ``[T, N]`` (reanchor sampled PRE-step, done applied POST-step), return the set of
    ``(env, t)`` that emit a window (a full ``H``-row ring with no reset)."""
    T, N = reanchor.shape
    valid = np.zeros(N, dtype=np.int64)
    emitted: set[tuple[int, int]] = set()
    for t in range(T):
        valid = np.minimum(valid + 1, H)                 # push step t's row into the ring
        emit = reanchor[t].astype(bool) & (valid >= H)
        for e in np.nonzero(emit)[0]:
            emitted.add((int(e), int(t)))
        valid[done[t].astype(bool)] = 0                  # reset after the step
    return emitted


def _flat_episode_arrays(done: np.ndarray):
    """Build per-step episode_id/step_in_episode ``[T, N]`` from a done mask, matching
    record_data's bookkeeping (fresh id + zeroed step counter on reset)."""
    T, N = done.shape
    episode_id = np.zeros((T, N), np.int64)
    step_in_ep = np.zeros((T, N), np.int64)
    cur_ep = np.arange(N, dtype=np.int64)
    cur_step = np.zeros(N, np.int64)
    next_ep = N
    for t in range(T):
        episode_id[t] = cur_ep
        step_in_ep[t] = cur_step
        d = done[t].astype(bool)
        nd = int(d.sum())
        cur_ep = cur_ep.copy()
        cur_ep[d] = np.arange(next_ep, next_ep + nd, dtype=np.int64)
        next_ep += nd
        cur_step = cur_step + 1
        cur_step[d] = 0
    return episode_id, step_in_ep


def test_online_windows_match_snap_windows():
    """Randomized multi-env rollouts (with mid-episode resets + varied snaps): the online
    ring emits exactly the snaps snap_windows would window, for several H."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        T, N = int(rng.integers(6, 40)), int(rng.integers(1, 6))
        H = int(rng.integers(1, 5))
        reanchor = rng.random((T, N)) < 0.25            # ~25% snap density
        done = rng.random((T, N)) < 0.12                # ~12% reset density
        episode_id, step_in_ep = _flat_episode_arrays(done)

        # snap_windows over the flattened per-step arrays -> keyed by (episode, step).
        epf, stpf = episode_id.reshape(-1), step_in_ep.reshape(-1)
        wins = snap_windows(epf, stpf, done.reshape(-1), reanchor.reshape(-1), H, 1)
        sw_keys = {(int(epf[w[-1]]), int(stpf[w[-1]])) for w in wins}

        online = form_snap_windows_online(reanchor, done, H)
        online_keys = {(int(episode_id[t, e]), int(step_in_ep[t, e])) for (e, t) in online}

        assert sw_keys == online_keys, f"trial {trial}: T={T} N={N} H={H}"


# ---------------------------------------------------------------------------
# 2. Windowed dataset path == legacy per-step path
# ---------------------------------------------------------------------------
_NJ = 3
_STATE_TERMS = ["root_pos_w", "root_quat_w", "root_link_lin_vel_w",
                "root_link_ang_vel_w", "joint_pos", "joint_vel"]
_STATE_DIMS = [3, 4, 3, 3, _NJ, _NJ]
_NODE = sum(_STATE_DIMS)  # 19


def _meta():
    return SimpleNamespace(
        state_term_names=list(_STATE_TERMS),
        state_term_dims=list(_STATE_DIMS),
        state_frame_names=[],
    )


def _cfg():
    # extero_source is explicit because these fixtures emit a height scan and no depth
    # array; the config default is "depth" (the deployable signal). This file is about the
    # windowed writer matching the per-step path, not about exteroception.
    return FlowMatchingConfig(
        future_len=4, num_history_states=2, cmd_array="target_vel", cmd_stride=2,
        stride=1, val_fraction=0.25, seed=0, obs_noise=False,
        extero_source="height_scan",
    )


def _synthetic_per_step(seed=0):
    """A per-step recording: 4 episodes x 20 consecutive steps, snaps at steps 5/10/15.

    Built from the shared plausible-rollout fixture rather than from noise: the dataset
    validates the recording (smooth, tracked, and
    ``ref_state[t] == ref_future[t, 0]``), so white noise no longer builds. This file is
    about the windowed writer matching the per-step path, which those properties do not
    affect.
    """
    rng = np.random.default_rng(seed)
    n_ep, L, F, K = 4, 20, 4, 5
    P = n_ep * L
    step_in_episode = np.tile(np.arange(L), n_ep).astype(np.int64)
    per_ep = [episode_arrays(rng, L, F, _NJ, 0, snap_every=1) for _ in range(n_ep)]
    data = {k: np.concatenate([ep[k] for ep in per_ep]) for k in per_ep[0]}
    data["reanchor_active"] = np.isin(step_in_episode, [5, 10, 15]).astype(np.uint8)
    data["done"] = (step_in_episode == L - 1).astype(np.uint8)
    data["height_scan"] = rng.standard_normal((P, K)).astype(np.float32)
    data["cmd_future"] = rng.standard_normal((P, F, 3)).astype(np.float32)
    data["episode_id"] = np.repeat(np.arange(n_ep), L).astype(np.int64)
    data["step_in_episode"] = step_in_episode
    return data


def _to_windowed(data, N):
    """Pre-form the windows snap_windows would build, in the same order.

    ``H == N + 1``: the N autoregressive context rows plus the snap row they precede (the
    snap is node 0 of the future, so it is not part of the history block).
    """
    wins = snap_windows(data["episode_id"], data["step_in_episode"], data["done"],
                        data["reanchor_active"], N + 1, 1)
    idx = np.stack(wins).astype(np.int64)   # (W, N+1)
    snap = idx[:, -1]
    return {
        "history_state": data["state"][idx],         # (W, H, node)  -> H == N + 1
        # The applied-reference ring, gathered at the SAME rows as the measured one. That
        # alignment is what the dataset's history/target guard checks.
        "history_ref": data["ref_state"][idx],       # (W, H, node)
        "ref_future": data["ref_future"][snap],      # (W, F, node)
        "height_scan": data["height_scan"][snap],    # (W, K)
        "cmd_future": data["cmd_future"][snap],      # (W, F, 3)
        "episode_id": data["episode_id"][snap],      # (W,)
    }


def test_windowed_dataset_matches_per_step():
    """build_train_val on the pre-formed windowed data yields the SAME normalized
    train/val tensors (state, history, height, cmd, anchor) + val split as the legacy path."""
    data = _synthetic_per_step()
    meta = _meta()
    N = 2

    tl, vl = build_train_val(data, meta, _cfg())          # legacy per-step
    tw, vw = build_train_val(_to_windowed(data, N), meta, _cfg())  # windowed

    assert len(tl) == len(tw) > 0 and (vl is None) == (vw is None)
    assert tl.val_episode_ids == tw.val_episode_ids

    def _same(ds_a, ds_b):
        for attr in ("state", "history", "height_norm_tokens", "cmd_tokens", "p_ref", "q_ref"):
            a, b = getattr(ds_a, attr), getattr(ds_b, attr)
            assert torch.allclose(a, b, atol=1e-5), f"{attr} differs (max {(a - b).abs().max():.2e})"

    _same(tl, tw)
    if vl is not None:
        assert len(vl) == len(vw)
        _same(vl, vw)


def test_windowed_rejects_too_short_history():
    """Training with num_history_states > recorded H raises a clear error."""
    import pytest

    data = _synthetic_per_step()
    windowed = _to_windowed(data, 2)  # H == 3
    cfg = _cfg()
    cfg.num_history_states = 4        # needs H >= 5
    with pytest.raises(ValueError, match="history_len"):
        build_train_val(windowed, _meta(), cfg)
