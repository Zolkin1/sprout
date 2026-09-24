# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the exact backward-DP motion matcher (``dp.py``).

Synthetic (no bones cache / IsaacLab).  Key checks: the all-frames ``_match_cost_all`` equals
beam's per-frame ``_match_cost`` (the A/B/C linchpin), the DP recovers the brute-force optimum
on a tiny graph, and its objective never exceeds an (unconstrained) beam's.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_gen.motion_matching.beam import BeamConfig, BeamMatcher
from motion_gen.motion_matching.dp import DPConfig, DPMatcher
from motion_gen.motion_matching.graph import VARIED, GraphConfig, build_candidate_graph
from motion_gen.features import G1_JOINT_NAMES as _G1_JN
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig, VanillaMatcher

_NJ = len(_G1_JN)                                    # joint-DOF count (29); synthetic clips match the DB width


def _clip(n, *, seed, vx=0.6, wz=0.0, z=0.78):
    g = torch.Generator().manual_seed(seed)
    dt = 0.02
    yaw = torch.arange(n) * wz * dt
    step = vx * dt
    rp = torch.zeros(n, 3); rp[:, 2] = z
    dx = step * torch.cos(yaw); dy = step * torch.sin(yaw)
    rp[:, 0] = torch.cumsum(dx, 0) - dx[0]; rp[:, 1] = torch.cumsum(dy, 0) - dy[0]
    rq = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], -1)
    jp = 0.15 * torch.randn(n, _NJ, generator=g); jv = 0.15 * torch.randn(n, _NJ, generator=g)
    return SimpleNamespace(name=f"clip{seed}", n=n, root_pos=rp, root_quat=rq, joint_pos=jp, joint_vel=jv)


def _const_cmd(vx, vy=0.0, wz=0.0):
    return lambda t: (vx, vy, wz)


def _graph_for(m, gcfg):
    cost = m._norm[:, m._traj_dim:]
    return build_candidate_graph(cost[:, POSE_POSTURAL_IDX], cost, m.feat_raw, m.owner,
                                 m.src_start, m.src_len, m.yaw_all, gcfg)


def _dp(sources, dcfg=None, gcfg=None, dt=0.02):
    dm = DPMatcher(sources, VanillaConfig(dt=dt), dcfg or DPConfig(dt=dt), device="cpu")
    dm.attach_graph(_graph_for(dm, gcfg or GraphConfig(k_min=2, k_max=6, k_target=5, dt=dt)))
    return dm


def _path_objective(dm, path, jumped, T):
    """Objective of a path under the DP cost: sum mc(path[t],t) + sum_cuts (jump_cost+cut_penalty)."""
    mc = [dm._match_cost_all(t) for t in range(T)]
    obj = float(sum(mc[t][int(path[t])] for t in range(T)))
    for t in range(1, T):
        if jumped[t]:
            src, dst = int(path[t - 1]), int(path[t])
            slots = np.where(dm._cand[src] == dst)[0]
            jc = min(float(dm._jcost[src, k]) for k in slots)   # the finite matching slot
            obj += jc + dm.dcfg.cut_penalty
    return obj


# ----------------------------------------------------------------------------------------
def test_dp_match_cost_all_equals_beam_matchcost():
    # The A/B/C linchpin: the all-frames cost must equal the exact per-frame velocity window,
    # both for constant-window interior t and for a changing-command boundary t.
    srcs = [_clip(90, seed=1, vx=0.5), _clip(80, seed=2, vx=1.2, wz=0.3), _clip(70, seed=3, vx=0.0)]
    dm = _dp(srcs)
    T = 120
    cmd = lambda t: (1.0, 0.0, 0.0) if t < 1.2 else (0.0, 1.0, 0.0)   # a command change at t=60
    dm._prepare(cmd, T)
    for t in (5, 30, 59, 60, 61, 90, T - 3):            # interior, right at the change, ends
        allc = dm._match_cost_all(t)
        fr = np.array([0, 7, 40, 100, dm._bvel.shape[0] - 5])
        sub = dm._match_cost(fr, t)
        assert np.allclose(allc[fr], sub, rtol=1e-9, atol=1e-9), f"mismatch at t={t}"


def test_dp_matches_brute_force_tiny():
    # DP must equal the true optimum found by exhaustive path enumeration on a small graph.
    srcs = [_clip(60, seed=1, vx=0.2), _clip(60, seed=2, vx=1.0), _clip(60, seed=3, vx=0.0, wz=0.5)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, cut_penalty=1.5), gcfg=GraphConfig(k_min=2, k_max=4, k_target=3))
    T = 5
    cmd = lambda t: (1.0, 0.0, 0.0) if t < 0.06 else (0.0, 0.0, 0.5)
    dm._prepare(cmd, T)
    action = dm._dp_backward(T)
    path, jumped = dm._reconstruct(action, T)
    dp_obj = _path_objective(dm, path, jumped, T)

    mc = [dm._match_cost_all(t) for t in range(T)]
    best = [math.inf]
    def rec(f, t, cost):
        cost += float(mc[t][f])
        if cost >= best[0]:
            return
        if t == T - 1:
            best[0] = cost; return
        nf = int(dm._next[f])
        if nf != -1:
            rec(nf, t + 1, cost)
        for k in range(dm._cand.shape[1]):
            jc = float(dm._jcost[f, k])
            if np.isfinite(jc):
                rec(int(dm._cand[f, k]), t + 1, cost + jc + dm.dcfg.cut_penalty)
    rec(int(dm._stationary), 0, 0.0)
    assert math.isclose(dp_obj, best[0], rel_tol=1e-6, abs_tol=1e-6), (dp_obj, best[0])


def test_dp_runs_and_plays():
    dm = _dp([_clip(120, seed=1, vx=0.0), _clip(120, seed=2, vx=1.0), _clip(120, seed=3, vx=0.6, wz=0.4)])
    T = 200
    rp, rq, jj = dm.generate(_const_cmd(1.0), T, debug=True)
    assert rp.shape == (T, 3) and rq.shape == (T, 4) and jj.shape == (T, _NJ)
    assert torch.isfinite(rp).all() and torch.isfinite(rq).all() and torch.isfinite(jj).all()
    assert dm.dbg is not None and len(dm.dbg) == T and dm.dbg[0][1] is False


def test_dp_rides_matching_clip_via_continue():
    # forward command + a forward clip -> the DP should COMMIT to the forward clip and ride it
    # via the continue edge (dominant fraction, few cuts).  (The fixed stationary start is on
    # the standing clip, so >=1 cut to get onto it + maybe a forced jump at its end.)
    srcs = [_clip(220, seed=2, vx=1.0), _clip(120, seed=1, vx=0.0)]     # forward=owner0, stand=owner1
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, cut_penalty=3.0))
    T = 180
    dm._prepare(_const_cmd(1.0), T)
    path, _ = dm._reconstruct(dm._dp_backward(T), T)
    frac_fwd = float((dm.owner.numpy()[path] == 0).mean())
    assert frac_fwd > 0.7, f"forward command should ride the forward clip: frac={frac_fwd:.2f}"
    assert dm.n_cuts <= 4, f"should ride via continue, not thrash: {dm.n_cuts} cuts"


def test_dp_steering_switches_clips():
    srcs = [_clip(150, seed=1, vx=0.0), _clip(150, seed=2, vx=1.0)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, cut_penalty=1.0))
    T = 300
    cmd = lambda t: (1.0, 0.0, 0.0) if t < 3.0 else (0.0, 0.0, 0.0)
    dm._prepare(cmd, T)
    path, _ = dm._reconstruct(dm._dp_backward(T), T)
    owner = dm.owner.numpy()
    assert (owner[path[:120]] == 1).mean() > 0.6          # forward clip in the first half
    assert (owner[path[180:]] == 0).mean() > 0.6          # standing clip in the second
    assert dm.n_cuts >= 1


def test_dp_cut_penalty_monotonic():
    srcs = [_clip(120, seed=1, vx=0.0), _clip(120, seed=2, vx=1.0), _clip(120, seed=3, vx=0.5, wz=0.6)]
    T = 250
    cmd = lambda t: (1.0, 0.0, 0.0) if (t % 1.0) < 0.5 else (0.0, 0.0, 0.6)
    cuts = []
    for pen in (0.0, 5.0, 50.0):
        dm = _dp(srcs, dcfg=DPConfig(dt=0.02, cut_penalty=pen))
        dm._prepare(cmd, T)
        dm._reconstruct(dm._dp_backward(T), T)
        cuts.append(dm.n_cuts)
    assert cuts[0] >= cuts[1] >= cuts[2], f"more cut_penalty must not increase cuts: {cuts}"


def test_dp_forced_jump_past_short_clip():
    dm = _dp([_clip(40, seed=2, vx=1.0), _clip(150, seed=1, vx=1.0)], dcfg=DPConfig(dt=0.02))
    T = 220
    rp, _, _ = dm.generate(_const_cmd(1.0), T, debug=True)
    assert rp.shape[0] == T and torch.isfinite(rp).all()
    assert dm.n_cuts >= 1                                  # had to leave the 40-frame clip


def test_dp_objective_le_unconstrained_beam():
    # DP is the exact optimum of the local objective; an (unconstrained) beam is one feasible
    # path in the same space -> DP objective <= beam objective.
    srcs = [_clip(150, seed=1, vx=0.0), _clip(150, seed=2, vx=1.0), _clip(140, seed=3, vx=0.5, wz=0.4)]
    gcfg = GraphConfig(k_min=2, k_max=6, k_target=5, dt=0.02)
    W, D, P = 10, 0.85, 2.0
    dm = DPMatcher(srcs, VanillaConfig(dt=0.02),
                   DPConfig(dt=0.02, cut_penalty=P, w_track=1.0, track_window=W, track_decay=D), device="cpu")
    graph = _graph_for(dm, gcfg)
    dm.attach_graph(graph)
    bm = BeamMatcher(srcs, VanillaConfig(dt=0.02),
                     BeamConfig(dt=0.02, cut_penalty=P, w_track=1.0, w_pos=0.0, w_traj=0.0, w_tier=0.0,
                                min_cut_gap_s=0.0, track_window=W, track_decay=D, beam_width=4096), device="cpu")
    bm.attach_graph(graph)
    T = 250
    cmd = lambda t: (1.0, 0.0, 0.0) if t < 2.5 else (0.0, 0.5, 0.0)
    dm._prepare(cmd, T); dp_path, dp_j = dm._reconstruct(dm._dp_backward(T), T)
    bm._prepare(cmd, T); be_path, be_j = bm._beam_search(T)
    dp_obj = _path_objective(dm, dp_path, dp_j, T)
    be_obj = _path_objective(dm, be_path, be_j, T)          # score beam's path under the same objective
    assert dp_obj <= be_obj + 1e-4, f"DP {dp_obj} should be <= beam {be_obj}"


# ---- the per-worker memory footprint of one solve -------------------------------------
# These pin the arithmetic AGAINST the shape of the arrays it runs on.  The action table and the
# all-frames cost are sized by the DB (millions of frames) and are written by every solve, so they
# are what a fork worker pays privately -- and shrinking them once OOM-killed the whole desktop.
# Any of these three can be "fixed" by making the solve allocate the straightforward way again,
# which is why the pins are here rather than in a comment.

def test_traj_cost_chunking_is_bit_identical():
    # The chunked trajectory cost must be BIT-identical to the whole-array expression, not merely
    # close: the DP compares costs with `<=` to break ties toward `continue`, so a 1-ulp difference
    # can move a cut and change the clip that gets generated.
    srcs = [_clip(150, seed=1, vx=0.0), _clip(150, seed=2, vx=1.0), _clip(140, seed=3, vx=0.5, wz=0.4)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, w_traj=1.0, w_track=0.0))
    T = 60
    dm._prepare(lambda t: (1.0, 0.0, 0.0) if t < 0.6 else (0.0, 0.6, 0.4), T)
    for t in (0, 1, 17, T - 2, T - 1):
        d = dm._traj_feat - dm._dtraj[t]                        # the un-chunked form
        assert np.array_equal(dm._traj_cost_all(t), (d * d * dm._traj_w).sum(1)), f"t={t}"


def test_action_table_is_int8_and_time_major():
    # int8 halves the table and time-major makes each step a contiguous write; both are load-bearing
    # at DB scale (5.1 vs 10.2 GiB per concurrent solve at N=5.5M, T=1000).  The reconstruction reads
    # `action[t, f]`, so a layout change here silently transposes it -- the path check is the guard.
    srcs = [_clip(120, seed=1, vx=0.0), _clip(120, seed=2, vx=1.0)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, cut_penalty=3.0))
    T = 90
    dm._prepare(_const_cmd(1.0), T)
    action = dm._dp_backward(T)
    N = dm._bvel.shape[0]
    assert action.dtype == np.int8 and action.shape == (T, N), (action.dtype, action.shape)
    path, _ = dm._reconstruct(action, T)
    assert len(path) == T and dm.owner.numpy()[path].max() < len(srcs)     # a real, in-range path


def test_a_wide_candidate_graph_is_refused_not_truncated():
    # 0=continue and 1+k for a jump, so int8 holds k_max<=126.  A wider graph must RAISE: silently
    # wrapping would store slot 130 as a negative action and reconstruct a different motion.
    srcs = [_clip(120, seed=1, vx=0.0), _clip(120, seed=2, vx=1.0)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02))
    dm._cand = np.zeros((dm._bvel.shape[0], 130), dtype=np.int32)         # pretend k_max=130
    dm._prepare(_const_cmd(1.0), 10)
    with pytest.raises(ValueError, match="k_max=130"):
        dm._dp_backward(10)


def test_solve_bytes_tracks_the_horizon_and_dwarfs_the_greedy_matcher():
    # A fork pool is sized by this, so it has to MOVE with T (the action table is O(T*N)); a constant
    # would size a 1000-step solve like a 100-step one, which is how the pool came to ask for ~200 GiB.
    srcs = [_clip(150, seed=1, vx=0.0), _clip(150, seed=2, vx=1.0)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, w_traj=1.0, w_track=0.0))
    N = dm._bvel.shape[0]
    assert dm.solve_bytes(1000) - dm.solve_bytes(100) == 900 * N          # exactly the table's growth
    vm = VanillaMatcher(srcs, VanillaConfig(dt=0.02), device="cpu", build_tree=False)
    assert vm.solve_bytes(1000) < dm.solve_bytes(1000)


# ---- clean STEADY-tier cost (one-sided, regime-ramped) ---------------------------------
def _steady(s, vel):
    s.steady = True; s.wrap = "periodic"; s.steady_vel = torch.tensor(vel, dtype=torch.float32)
    return s


def test_dp_match_cost_all_equals_match_cost_with_tier():
    # exactness linchpin still holds with the (velocity-matched) tier term on -- it is pure (frame,t)
    srcs = [_steady(_clip(90, seed=1, vx=0.8), (0.8, 0.0, 0.0)), _clip(80, seed=2, vx=0.0)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, w_track=0.0, w_traj=1.0, w_tier=1.5, soft_tier_penalty=2.0))
    T = 100
    cmd = lambda t: (0.8, 0.0, 0.0) if t < 1.0 else (0.0, 0.0, 0.0)
    dm._prepare(cmd, T)
    fr = np.array([0, 40, 100, dm._bvel.shape[0] - 3])
    for t in (5, 30, 49, 50, 51, 90):
        assert np.allclose(dm._match_cost_all(t)[fr], dm._match_cost(fr, t), rtol=1e-9, atol=1e-9)


def test_dp_tier_nearest_neighbour():
    # 1-NN tier: under a steady command (regime==1), ONLY the velocity-nearest clean clip gets the
    # bonus (no extra cost); a clean clip at the WRONG velocity AND bones both pay the FULL penalty.
    # (The key fix: not a blanket "any clean" preference -- the wrong-velocity clean clip is not
    # rewarded, so a stop won't grab a moving clean clip.)
    match = _steady(_clip(90, seed=1, vx=0.8), (0.8, 0.0, 0.0))   # clean, matches command
    off = _steady(_clip(90, seed=2, vx=0.0), (0.0, 0.0, 0.0))     # clean, WRONG velocity
    bones = _clip(90, seed=3, vx=0.4)                             # VARIED
    W, soft = 1.5, 2.0
    base = _dp([match, off, bones], dcfg=DPConfig(dt=0.02, w_track=0.0, w_traj=1.0, w_tier=0.0))
    tier = _dp([match, off, bones], dcfg=DPConfig(dt=0.02, w_track=0.0, w_traj=1.0, w_tier=W,
                                                  soft_tier_penalty=soft, tier_nn_k=1))
    T = 100
    base._prepare(lambda t: (0.8, 0.0, 0.0), T); tier._prepare(lambda t: (0.8, 0.0, 0.0), T)
    delta = tier._match_cost_all(50) - base._match_cost_all(50)
    owner = tier._owner_np
    assert np.allclose(delta[owner == 0], 0.0, atol=1e-6)         # NN clean clip -> no penalty
    assert np.allclose(delta[owner == 1], W * soft)              # clean but wrong velocity -> full
    assert np.allclose(delta[owner == 2], W * soft)              # bones -> full penalty


def test_dp_tier_tiebreak_prefers_stillest():
    # two clean clips at the SAME steady velocity (0,0,0) -- a still "stand" and a moving
    # "shuffle" -- must be broken by stillness: the NN reward goes to the stiller clip.
    still = _steady(_clip(60, seed=1, vx=0.0), (0.0, 0.0, 0.0)); still.joint_vel = torch.zeros(60, _NJ)
    shuffle = _steady(_clip(60, seed=2, vx=0.0), (0.0, 0.0, 0.0)); shuffle.joint_vel = 2.0 * torch.randn(60, _NJ)
    dm = _dp([still, shuffle], dcfg=DPConfig(dt=0.02, w_track=0.0, w_traj=1.0, w_tier=1.0, tier_nn_k=1))
    dm._prepare(lambda t: (0.0, 0.0, 0.0), 20)
    assert dm._src_activity[0] < dm._src_activity[1]            # still clip has lower foot motion
    assert int(np.flatnonzero(dm._tier_rewarded[0])[0]) == 0    # ... and it is the rewarded NN clip


def test_dp_regime_neutral_in_transition():
    # regime ~0 around a command change (tier neutral there) and ~1 in steady stretches
    srcs = [_steady(_clip(90, seed=1, vx=0.8), (0.8, 0.0, 0.0)), _clip(90, seed=2, vx=0.0)]
    dm = _dp(srcs, dcfg=DPConfig(dt=0.02, w_track=0.0, w_traj=1.0, w_tier=2.0, regime_ramp_s=0.5))
    T = 100
    dm._prepare(lambda t: (0.8, 0.0, 0.0) if t < 1.0 else (0.0, 0.0, 0.0), T)   # jump at t=50
    assert dm._regime[50] < 0.1                             # neutral at the change
    assert dm._regime[5] > 0.9 and dm._regime[95] > 0.9     # steady far from it
