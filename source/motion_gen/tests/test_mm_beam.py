# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the offline beam-search motion matcher: candidate graph + beam DP.

All synthetic (no bones cache / IsaacLab) so they run anywhere.  The graph + beam operate
on a small hand-built DB via :class:`VanillaMatcher` (``build_tree=False``, so we reuse its
feature/DB build without the greedy full-feature KD-tree).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_gen.motion_matching.beam import BeamConfig, BeamMatcher
from motion_gen.motion_matching.features import G1_JOINT_NAMES as _G1_JN
from motion_gen.motion_matching.graph import (
    VARIED,
    CandidateGraph,
    GraphConfig,
    build_candidate_graph,
    graph_cache_key,
)

_NJ = len(_G1_JN)                                    # joint-DOF count (29); synthetic clips match the DB width
from motion_gen.motion_matching.vanilla import POSE_POSTURAL_IDX, VanillaConfig, VanillaMatcher


def _clip(n, *, seed, vx=0.6, wz=0.0, z=0.78):
    """A synthetic clip: root moves forward at ~vx (+ optional yaw rate), gentle joints."""
    g = torch.Generator().manual_seed(seed)
    dt = 0.02
    yaw = torch.arange(n) * wz * dt
    step = vx * dt
    rp = torch.zeros(n, 3)
    rp[:, 2] = z
    # integrate forward velocity along the (turning) heading
    dx = step * torch.cos(yaw); dy = step * torch.sin(yaw)
    rp[:, 0] = torch.cumsum(dx, 0) - dx[0]
    rp[:, 1] = torch.cumsum(dy, 0) - dy[0]
    rq = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], -1)
    jp = 0.15 * torch.randn(n, _NJ, generator=g)
    jv = 0.15 * torch.randn(n, _NJ, generator=g)
    return SimpleNamespace(name=f"clip{seed}", n=n, root_pos=rp, root_quat=rq,
                           joint_pos=jp, joint_vel=jv)


def _matcher(sources, dt=0.02):
    return VanillaMatcher(sources, VanillaConfig(dt=dt), device="cpu", build_tree=False)


def _pose_norm(m):
    return m._norm[:, m._traj_dim:]                     # full 27-D pose block (jump_cost metric)


def _select_feat(m):
    return _pose_norm(m)[:, POSE_POSTURAL_IDX]          # postural (feet+hand pos) selection metric


def _build_graph(m, cfg):
    return build_candidate_graph(_select_feat(m), _pose_norm(m), m.feat_raw, m.owner,
                                 m.src_start, m.src_len, m.yaw_all, cfg)


@pytest.fixture
def small_db():
    srcs = [_clip(70, seed=1, vx=0.4), _clip(80, seed=2, vx=1.2),
            _clip(75, seed=3, vx=0.8, wz=0.5), _clip(65, seed=4, vx=0.0)]
    return _matcher(srcs)


def test_graph_shapes_and_next_frame(small_db):
    cfg = GraphConfig(k_min=3, k_max=8, k_target=6, dt=0.02, min_same_clip_gap=3)
    g = _build_graph(small_db, cfg)
    N = small_db.feat_raw.shape[0]
    assert g.candidates.shape == (N, 8) and g.candidates.dtype == np.int32
    assert g.jump_cost.shape == (N, 8) and g.jump_cost.dtype == np.float16
    assert g.next_frame.shape == (N,) and g.next_frame.dtype == np.int32
    assert g.body_vel.shape == (N, 3) and g.tier.shape == (N,)
    # kept candidates per frame within [k_min, k_max]
    nkeep = np.isfinite(g.jump_cost).sum(1)
    assert nkeep.min() >= 1 and nkeep.max() <= 8
    # next_frame == -1 exactly at each clip's last frame
    ss = small_db.src_start.numpy(); sl = small_db.src_len.numpy()
    for a, n in zip(ss, sl):
        assert g.next_frame[a + n - 1] == -1
        if n > 1:
            assert g.next_frame[a] == a + 1


def test_graph_excludes_self_and_caps_same_clip(small_db):
    gap, cap = 3, 2
    cfg = GraphConfig(k_min=3, k_max=8, k_target=6, min_same_clip_gap=gap, same_clip_cap=cap)
    g = _build_graph(small_db, cfg)
    owner = small_db.owner.numpy(); ss = small_db.src_start.numpy()
    for f in range(g.n):
        same = 0
        for j, c in enumerate(g.candidates[f]):
            c = int(c)
            if not np.isfinite(g.jump_cost[f, j]):     # unused (padded) slot
                continue
            assert c != f, "self appeared as its own candidate"
            if owner[c] == owner[f]:
                lf = f - ss[owner[f]]; lc = c - ss[owner[c]]
                assert abs(lc - lf) > gap, "same-clip within-gap neighbour not excluded"
                same += 1
        assert same <= cap, f"frame {f}: {same} same-clip candidates > cap {cap}"


def test_graph_jump_cost_matches_full_pose_l2(small_db):
    # jump_cost is the FULL pose distance (incl. velocity), even though selection is postural
    cfg = GraphConfig(k_min=3, k_max=6, k_target=5)
    g = _build_graph(small_db, cfg)
    pn = _pose_norm(small_db).numpy().astype(np.float64)
    for f in (0, 5, 40, g.n - 10):
        for j in range(g.k):
            if not np.isfinite(g.jump_cost[f, j]):
                continue
            c = int(g.candidates[f, j])
            recomputed = float(np.linalg.norm(pn[f] - pn[c]))
            assert math.isclose(recomputed, float(g.jump_cost[f, j]), rel_tol=2e-2, abs_tol=2e-2)


def test_graph_body_vel_bounded_and_finite(small_db):
    g = _build_graph(small_db, GraphConfig(k_min=3, k_max=8))
    assert np.isfinite(g.body_vel).all()
    assert np.abs(g.body_vel[:, 2]).max() < 30.0          # wz sane (per-clip unwrap)
    assert (g.tier == VARIED).all()                        # bones-only -> all VARIED


def test_graph_save_load_roundtrip(small_db, tmp_path):
    cfg = GraphConfig(k_min=3, k_max=8)
    g = build_candidate_graph(_select_feat(small_db), _pose_norm(small_db), small_db.feat_raw,
                              small_db.owner, small_db.src_start, small_db.src_len,
                              small_db.yaw_all, cfg, key="abc")
    p = tmp_path / "graph.npz"
    g.save(p)
    r = CandidateGraph.load(p, expected_key="abc")
    assert r is not None
    assert np.array_equal(r.candidates, g.candidates)
    assert np.array_equal(r.jump_cost.astype(np.float32), g.jump_cost.astype(np.float32))
    assert np.array_equal(r.next_frame, g.next_frame)
    assert np.allclose(r.body_vel, g.body_vel)
    assert CandidateGraph.load(p, expected_key="different") is None      # key mismatch -> miss


def test_graph_cache_key_sensitivity(tmp_path):
    fake = tmp_path / "bones.h5"
    fake.write_bytes(b"x" * 100)
    base = dict(filters={"package": "Locomotion"}, styles=None, exclude_mirror=True,
                exclude_name_substrings=["injured"], limit=None,
                pose_weights=(0.75, 1.0, 0.3, 0.3, 1.0))
    k0 = graph_cache_key(str(fake), cfg=GraphConfig(k_max=50), **base)
    assert k0 == graph_cache_key(str(fake), cfg=GraphConfig(k_max=50), **base)      # stable
    assert k0 != graph_cache_key(str(fake), cfg=GraphConfig(k_max=40), **base)      # k_max matters
    assert k0 != graph_cache_key(str(fake), cfg=GraphConfig(k_max=50, same_clip_cap=1), **base)  # cap
    assert k0 != graph_cache_key(str(fake), cfg=GraphConfig(k_max=50),
                                 **{**base, "pose_weights": (0.75, 1.0, 0.5, 0.3, 1.0)})  # weights
    assert k0 != graph_cache_key(str(fake), cfg=GraphConfig(k_max=50),
                                 **{**base, "filters": {"package": "All"}})        # filter


# ----------------------------------------------------------------------------------------
# Beam DP
# ----------------------------------------------------------------------------------------
def _beam(sources, bcfg=None, gcfg=None, dt=0.02):
    """Build a BeamMatcher + candidate graph (from its own DB arrays) for testing."""
    vcfg = VanillaConfig(dt=dt)
    bm = BeamMatcher(sources, vcfg, bcfg or BeamConfig(dt=dt), device="cpu")
    bm.attach_graph(_build_graph(bm, gcfg or GraphConfig(k_min=3, k_max=8, k_target=6, dt=dt)))
    return bm


def _owner_fractions(bm, path):
    owner = bm.owner.numpy()
    o = owner[path]
    return {int(k): float((o == k).mean()) for k in np.unique(o)}


def _const_cmd(vx, vy=0.0, wz=0.0):
    return lambda t: (vx, vy, wz)


def test_beam_runs_and_plays():
    srcs = [_clip(120, seed=1, vx=0.0), _clip(120, seed=2, vx=1.0),
            _clip(120, seed=3, vx=0.6, wz=0.4)]
    bm = _beam(srcs)
    T = 200
    rp, rq, jj = bm.generate(_const_cmd(1.0), T, debug=True)
    assert rp.shape == (T, 3) and rq.shape == (T, 4) and jj.shape == (T, _NJ)
    assert torch.isfinite(rp).all() and torch.isfinite(rq).all() and torch.isfinite(jj).all()
    assert bm.dbg is not None and len(bm.dbg) == T
    assert bm.dbg[0][1] is False                     # step 0 is never a cut


def test_beam_prefers_matching_clip():
    # clip 1 stands, clip 2 walks forward at 1 m/s; a forward command should live on clip 2.
    srcs = [_clip(150, seed=1, vx=0.0), _clip(150, seed=2, vx=1.0)]
    bm = _beam(srcs, bcfg=BeamConfig(dt=0.02, cut_penalty=2.0))
    T = 250
    bm._prepare(_const_cmd(1.0), T)
    path, jumped = bm._beam_search(T)
    frac = _owner_fractions(bm, path)
    assert frac.get(1, 0.0) > 0.7, f"forward command should live on the walking clip: {frac}"


def test_beam_steering_switches_clips():
    # forward for the first half, stand for the second -> beam must switch clips.
    srcs = [_clip(150, seed=1, vx=0.0), _clip(150, seed=2, vx=1.0)]
    bm = _beam(srcs, bcfg=BeamConfig(dt=0.02, cut_penalty=1.0))
    T = 300
    cmd = lambda t: (1.0, 0.0, 0.0) if t < 3.0 else (0.0, 0.0, 0.0)
    bm._prepare(cmd, T)
    path, jumped = bm._beam_search(T)
    owner = bm.owner.numpy()
    first = owner[path[:120]]; second = owner[path[180:]]
    # first half dominated by the walking clip, second half by the standing clip
    assert (first == 1).mean() > 0.6
    assert (second == 0).mean() > 0.6
    assert bm.n_cuts >= 1


def test_beam_cut_penalty_monotonic():
    srcs = [_clip(120, seed=1, vx=0.0), _clip(120, seed=2, vx=1.0),
            _clip(120, seed=3, vx=0.5, wz=0.6)]
    T = 250
    cmd = lambda t: (1.0, 0.0, 0.0) if (t % 1.0) < 0.5 else (0.0, 0.0, 0.6)   # jittery target
    cuts = []
    for pen in (0.0, 5.0, 50.0):
        bm = _beam(srcs, bcfg=BeamConfig(dt=0.02, cut_penalty=pen))
        bm._prepare(cmd, T)
        bm._beam_search(T)
        cuts.append(bm.n_cuts)
    assert cuts[0] >= cuts[1] >= cuts[2], f"more cut_penalty must not increase cuts: {cuts}"


def test_desired_traj_integration():
    # 1 m/s forward command -> at horizon h seconds the desired displacement is ~h m fwd,
    # 0 lateral, facing unchanged.  (needs w_traj != 0 so _dtraj is built)
    bm = _beam([_clip(120, seed=1, vx=1.0)], bcfg=BeamConfig(dt=0.02, w_traj=1.0))
    T = 100
    bm._prepare(_const_cmd(1.0, 0.0, 0.0), T)
    row = bm._dtraj[0]
    for i, h in enumerate(bm.cfg.traj_times_s):
        px, py, c, s = row[4 * i:4 * i + 4]
        assert abs(px - h) < 0.05 and abs(py) < 1e-6
        assert abs(c - 1.0) < 1e-6 and abs(s) < 1e-6
    # a yaw-rate command turns the desired facing
    bm._prepare(_const_cmd(0.0, 0.0, 1.0), T)     # 1 rad/s
    row = bm._dtraj[0]
    for i, h in enumerate(bm.cfg.traj_times_s):
        c, s = row[4 * i + 2], row[4 * i + 3]
        assert abs(np.arctan2(s, c) - h) < 0.02   # facing ~ h radians


def test_beam_traj_cost_prefers_matching_trajectory():
    # Isolate the trajectory term (w_track=0): under a forward command, a frame from the
    # forward-moving clip must score cheaper than one from the standing clip.
    srcs = [_clip(150, seed=2, vx=1.0), _clip(150, seed=1, vx=0.0)]     # fwd, stand
    bm = _beam(srcs, bcfg=BeamConfig(dt=0.02, w_track=0.0, w_traj=1.0, w_pos=0.0))
    bm._prepare(_const_cmd(1.0), 100)
    owner = bm.owner.numpy()
    ff = int(np.where(owner == 0)[0][20])       # mid forward clip (full future window)
    sf = int(np.where(owner == 1)[0][20])       # mid standing clip
    cost = bm._match_cost(np.array([ff, sf]), t=20)
    assert cost[0] < cost[1], f"trajectory cost should prefer the forward clip: {cost}"


def test_reference_pos_integration():
    # the integrated reference: forward 1 m/s -> straight +x line; a yaw command curves it.
    bm = _beam([_clip(120, seed=1, vx=1.0)])
    T = 100
    ref = bm._precompute_reference_pos(bm._precompute_target(_const_cmd(1.0), T), T)
    assert abs(ref[T - 1, 0] - (T - 1) * 0.02) < 0.05 and abs(ref[T - 1, 1]) < 1e-6
    refc = bm._precompute_reference_pos(bm._precompute_target(_const_cmd(1.0, 0.0, 1.0), T), T)
    assert refc[T - 1, 1] > 0.05      # turning left -> +y accrues


def test_beam_pos_tracks_matchable_command():
    # With the global-position term on (default), a command the DB CAN execute should be
    # tracked: the output world path travels ~ the integrated-reference distance.
    srcs = [_clip(220, seed=1, vx=1.0), _clip(150, seed=2, vx=0.0)]   # forward + standing
    bm = _beam(srcs)                                                 # default w_pos=2
    T = 250
    bm._prepare(_const_cmd(1.0), T)
    assert bm._ref_xy is not None and bm._ref_xy.shape == (T, 2)     # position term active
    rp, _, _ = bm.generate(_const_cmd(1.0), T)
    travel = float(np.sqrt((np.diff(rp[:, :2].numpy(), axis=0) ** 2).sum(1)).sum())
    ref = (T - 1) * 0.02 * 1.0
    assert 0.7 * ref < travel < 1.3 * ref, f"travel {travel:.2f} vs reference {ref:.2f}"


def test_beam_forced_jump_past_short_clip():
    # a short walking clip that ends before T -> beam is forced to jump to keep going.
    srcs = [_clip(40, seed=2, vx=1.0), _clip(150, seed=1, vx=1.0)]
    bm = _beam(srcs, bcfg=BeamConfig(dt=0.02, min_cut_gap_s=2.0))   # gap > short clip length
    T = 220
    rp, rq, jj = bm.generate(_const_cmd(1.0), T, debug=True)
    assert rp.shape[0] == T and torch.isfinite(rp).all()
    assert bm.n_cuts >= 1                              # had to leave the 40-frame clip
