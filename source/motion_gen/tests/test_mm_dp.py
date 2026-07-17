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
from motion_gen.motion_matching.graph import GraphConfig, build_candidate_graph
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig


def _clip(n, *, seed, vx=0.6, wz=0.0, z=0.78):
    g = torch.Generator().manual_seed(seed)
    dt = 0.02
    yaw = torch.arange(n) * wz * dt
    step = vx * dt
    rp = torch.zeros(n, 3); rp[:, 2] = z
    dx = step * torch.cos(yaw); dy = step * torch.sin(yaw)
    rp[:, 0] = torch.cumsum(dx, 0) - dx[0]; rp[:, 1] = torch.cumsum(dy, 0) - dy[0]
    rq = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], -1)
    jp = 0.15 * torch.randn(n, 21, generator=g); jv = 0.15 * torch.randn(n, 21, generator=g)
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
    assert rp.shape == (T, 3) and rq.shape == (T, 4) and jj.shape == (T, 21)
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
