# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for periodic-clip handling in the feature build (``compute_vanilla_features`` wrap).

A clean steady-state clip is one closed cycle; its tail frames must see the NEXT period in
their future-trajectory window (not a clamped stall), or the matcher would read the clip as
decelerating.  These check the periodic extension against the episodic (clamped) baseline.
"""

from __future__ import annotations

import torch

from motion_gen.math_utils import quat_mul
from motion_gen.motion_matching.graph import (
    STEADY, VARIED, GraphConfig, build_candidate_graph, graph_cache_key)
from motion_gen.playback import play_frame_path
from motion_gen.motion_matching.vanilla import (
    VanillaConfig, compute_vanilla_features, make_ee_fk, _periodic_extension)
from motion_gen.features import G1_JOINT_NAMES as _G1_JN

_DT = 0.02
_NJ = len(_G1_JN)                                    # joint-DOF count (29); synthetic clips match the DB width


class _LoopDB:
    """Minimal duck-typed ``db`` for playback: one periodic clip, root advancing at (vx, wz)."""

    def __init__(self, P, vx, wz, z=0.78):
        dt = _DT
        yaw = torch.arange(P, dtype=torch.float32) * wz * dt
        xy = torch.zeros(P, 2)
        for i in range(1, P):
            c, s = torch.cos(yaw[i - 1]), torch.sin(yaw[i - 1])
            xy[i, 0] = xy[i - 1, 0] + c * vx * dt
            xy[i, 1] = xy[i - 1, 1] + s * vx * dt
        self.device = torch.device("cpu")
        self.root_pos = torch.cat([xy, torch.full((P, 1), z)], dim=1)
        self.root_quat = torch.stack(
            [torch.cos(yaw / 2), torch.zeros(P), torch.zeros(P), torch.sin(yaw / 2)], dim=1)
        self.joints = torch.zeros(P, 2)
        self.yaw_all = yaw.clone()
        self.owner = torch.zeros(P, dtype=torch.long)
        self.src_start = torch.tensor([0]); self.src_len = torch.tensor([P])
        self.is_periodic_frame = torch.ones(P, dtype=torch.bool)

    def _wquat(self, g_yaw, gi):
        yq = torch.tensor([torch.cos(torch.tensor(g_yaw / 2)), 0.0, 0.0,
                           torch.sin(torch.tensor(g_yaw / 2))])
        return quat_mul(yq.unsqueeze(0), self.root_quat[gi].unsqueeze(0))[0]

    def _local_delta(self, gi):
        a = int(self.src_start[0])
        if gi - a > 0:
            return self.root_pos[gi] - self.root_pos[gi - 1]
        return self.root_pos[gi + 1] - self.root_pos[gi]


def _ride(P, T):
    """Frame path following the periodic loop (last frame wraps to frame 1), no cuts."""
    path = [0]
    for _ in range(T - 1):
        f = path[-1]
        path.append(f + 1 if f < P - 1 else 1)
    return torch.tensor(path), [False] * T


def _yaw_of(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _mini_graph(periodic, steady, lens, seed=0):
    torch.manual_seed(seed)
    owner, ss, acc = [], [], 0
    for i, L in enumerate(lens):
        ss.append(acc); owner += [i] * L; acc += L
    N = acc
    sel = torch.randn(N, 6); cost = torch.randn(N, 12); feat = torch.randn(N, 16)
    g = build_candidate_graph(
        sel, cost, feat, torch.tensor(owner), torch.tensor(ss), torch.tensor(lens),
        torch.zeros(N), GraphConfig(k_min=1, k_max=3, k_target=2, same_clip_cap=1,
                                    min_same_clip_gap=1),
        steady_owner_mask=torch.tensor(steady), periodic_owner_mask=torch.tensor(periodic))
    return g, ss, lens


def _straight_clip(n, vx, z=0.78):
    """A closed straight clip: constant pose, root advancing at body vx (identity yaw)."""
    rp = torch.zeros(n, 3); rp[:, 0] = torch.arange(n) * vx * _DT; rp[:, 2] = z
    rq = torch.zeros(n, 4); rq[:, 0] = 1.0
    jp = torch.zeros(n, _NJ); jv = torch.zeros(n, _NJ)
    return rp, rq, jp, jv


def test_periodic_tail_future_continues():
    cfg = VanillaConfig(dt=_DT)
    fk = make_ee_fk()
    vx, n = 1.0, 120                              # n > max horizon so early frames don't wrap
    rp, rq, jp, jv = _straight_clip(n, vx)
    fe = compute_vanilla_features(rp, rq, jp, jv, fk, cfg, wrap="episodic")
    fp = compute_vanilla_features(rp, rq, jp, jv, fk, cfg, wrap="periodic")
    tdim = 4 * len(cfg.traj_times_s)
    max_df = max(cfg.traj_frames())
    # last frame: episodic future is clamped (~0 displacement); periodic continues forward.
    px_ep = fe[n - 1, 0].item()                 # first traj horizon Δx (local == world here)
    px_pe = fp[n - 1, 0].item()
    assert abs(px_ep) < 1e-4                     # clamped stall
    assert px_pe > 0.5 * vx * _DT * cfg.traj_frames()[0]   # real forward motion into next period
    # interior frame (full window inside the clip) is identical under both wraps
    assert torch.allclose(fe[0, :tdim], fp[0, :tdim], atol=1e-5)
    # pose block (non-trajectory dims) is identical regardless of wrap
    assert torch.allclose(fe[:, tdim:], fp[:, tdim:], atol=1e-5)
    assert n - 1 + max_df >= n                   # sanity: the tail really does wrap


def test_periodic_extension_matches_constant_velocity():
    # straight clip -> extension advances at exactly the per-step delta, headings unchanged
    vx, n, K = 0.8, 30, 50
    rp, rq, jp, jv = _straight_clip(n, vx)
    yaw = torch.zeros(n)
    ext_xy, ext_yaw = _periodic_extension(rp[:, :2], yaw, K)
    assert ext_xy.shape == (n + K, 2) and ext_yaw.shape == (n + K,)
    step = vx * _DT
    # continuation keeps advancing by `step` in x, y stays 0, yaw stays 0
    dx = ext_xy[1:, 0] - ext_xy[:-1, 0]
    assert torch.allclose(dx, torch.full_like(dx, step), atol=1e-5)
    assert torch.allclose(ext_xy[:, 1], torch.zeros(n + K), atol=1e-5)
    assert torch.allclose(ext_yaw, torch.zeros(n + K), atol=1e-5)


def test_periodic_extension_turning_spirals():
    # turning clip: constant body vx + yaw rate -> extension continues the arc (yaw keeps growing)
    vx, wz, n, K = 0.5, 0.4, 40, 60
    dt = _DT
    yaw = torch.arange(n) * wz * dt
    xy = torch.zeros(n, 2)
    for i in range(1, n):
        c, s = torch.cos(yaw[i - 1]), torch.sin(yaw[i - 1])
        xy[i, 0] = xy[i - 1, 0] + c * vx * dt
        xy[i, 1] = xy[i - 1, 1] + s * vx * dt
    ext_xy, ext_yaw = _periodic_extension(xy, yaw, K)
    # yaw continues to accumulate past the seam at the same rate
    dyaw = ext_yaw[1:] - ext_yaw[:-1]
    assert torch.allclose(dyaw, torch.full_like(dyaw, wz * dt), atol=1e-4)
    # forward speed preserved across the seam
    speed = torch.linalg.norm(ext_xy[1:] - ext_xy[:-1], dim=1)
    assert torch.allclose(speed, torch.full_like(speed, vx * dt), atol=1e-4)


def test_graph_wrap_edge_and_tier():
    # clip 0: periodic + steady (len 10);  clip 1: episodic + non-steady (len 8)
    g, ss, lens = _mini_graph(periodic=[True, False], steady=[True, False], lens=[10, 8])
    L0, L1 = lens
    # periodic clip's last frame loops to its SECOND frame (ss+1), not -1; interior is f+1
    assert g.next_frame[L0 - 1] == ss[0] + 1
    assert g.next_frame[0] == 1 and g.next_frame[L0 - 2] == L0 - 1
    # episodic clip's last frame terminates (-1); interior is f+1
    assert g.next_frame[ss[1] + L1 - 1] == -1
    assert g.next_frame[ss[1]] == ss[1] + 1
    # tier: clip 0 frames STEADY, clip 1 frames VARIED
    assert (g.tier[:L0] == STEADY).all() and (g.tier[L0:] == VARIED).all()


def test_graph_no_periodic_mask_is_episodic():
    # without a periodic mask, every clip terminates at -1 (original behavior)
    g, ss, lens = _mini_graph(periodic=[False, False], steady=[False, False], lens=[10, 8])
    assert g.next_frame[lens[0] - 1] == -1
    assert (g.tier == VARIED).all()


def test_cache_key_clean_coexistence(tmp_path):
    bones = tmp_path / "bones_cache.h5"; bones.write_bytes(b"x" * 16)
    cleanf = tmp_path / "manifest.csv"; cleanf.write_bytes(b"y" * 16)
    kw = dict(filters=None, styles=None, exclude_mirror=True, exclude_name_substrings=None,
              limit=None, pose_weights=(1.0, 2.0), cfg=GraphConfig())
    k_bones = graph_cache_key(str(bones), **kw)
    # bones-only key is stable and unaffected by the new params (existing caches stay valid)
    assert graph_cache_key(str(bones), **kw) == k_bones
    assert graph_cache_key(str(bones), include_clean=False, clean_sources=None, **kw) == k_bones
    # including clean -> a DISTINCT key
    k_clean = graph_cache_key(str(bones), include_clean=True, clean_sources=[str(cleanf)], **kw)
    assert k_clean != k_bones
    # changing the clean library (mtime) -> another distinct key
    import os, time
    os.utime(cleanf, (time.time() + 10, time.time() + 10))
    k_clean2 = graph_cache_key(str(bones), include_clean=True, clean_sources=[str(cleanf)], **kw)
    assert k_clean2 != k_clean


def test_playback_straight_loop_monotone():
    P, vx = 30, 1.0
    db = _LoopDB(P, vx=vx, wz=0.0)
    path, jumped = _ride(P, 90)                          # ride ~3 periods
    cfg = VanillaConfig(dt=_DT)
    rp, rq, jj, _ = play_frame_path(path, jumped, db, cfg)
    dx = rp[1:, 0] - rp[:-1, 0]
    assert torch.all(dx > 0)                             # world x strictly advances (no wrap reset)
    assert torch.allclose(dx, torch.full_like(dx, vx * _DT), atol=1e-5)   # constant speed at the seam
    assert torch.allclose(rp[:, 1], torch.zeros(len(path)), atol=1e-5)    # no lateral drift
    assert torch.allclose(_yaw_of(rq), torch.zeros(len(path)), atol=1e-5)  # heading stays 0


def test_playback_turning_loop_smooth_arc():
    P, vx, wz = 40, 0.6, 0.4
    db = _LoopDB(P, vx=vx, wz=wz)
    path, jumped = _ride(P, 120)                         # ride ~3 periods of the turn
    cfg = VanillaConfig(dt=_DT)
    rp, rq, jj, _ = play_frame_path(path, jumped, db, cfg)
    # constant forward speed across every step, INCLUDING the loop seams (no position pop)
    speed = torch.linalg.norm(rp[1:, :2] - rp[:-1, :2], dim=1)
    assert torch.allclose(speed, torch.full_like(speed, vx * _DT), atol=1e-4)
    # heading advances by exactly one step of yaw each frame, monotonically (no seam snap)
    yaw = torch.unwrap(_yaw_of(rq)) if hasattr(torch, "unwrap") else _yaw_of(rq)
    dyaw = yaw[1:] - yaw[:-1]
    assert torch.allclose(dyaw, torch.full_like(dyaw, wz * _DT), atol=1e-4)


def test_playback_wrap_matches_feature_extension():
    # the world path produced by looping playback == the periodic feature extension (risk #2:
    # playback and features must agree at the seam).
    P, vx, wz = 36, 0.5, 0.3
    db = _LoopDB(P, vx=vx, wz=wz)
    K = 40
    path, jumped = _ride(P, P + K)
    cfg = VanillaConfig(dt=_DT)
    rp, rq, _, _ = play_frame_path(path, jumped, db, cfg)
    ext_xy, ext_yaw = _periodic_extension(db.root_pos[:, :2], db.yaw_all, K)
    # playback rides frames 0..P-1 then 1..; ext indexes 0..P-1 then continues. Compare the
    # continued segment (playback steps P.. vs ext[P..]) up to a fixed world offset/rotation:
    # both start from frame P-1's pose and advance identically -> per-step deltas match.
    ps = torch.linalg.norm(rp[P:, :2] - rp[P - 1:-1, :2], dim=1)
    es = torch.linalg.norm(ext_xy[P:P + len(ps)] - ext_xy[P - 1:P - 1 + len(ps)], dim=1)
    assert torch.allclose(ps, es, atol=1e-4)


def _clip_ns(n, vx, seed, steady, wrap, wz=0.0, z=0.78):
    from types import SimpleNamespace
    g = torch.Generator().manual_seed(seed)
    yaw = torch.arange(n) * wz * _DT; step = vx * _DT
    rp = torch.zeros(n, 3); rp[:, 2] = z
    dx = step * torch.cos(yaw); dy = step * torch.sin(yaw)
    rp[:, 0] = torch.cumsum(dx, 0) - dx[0]; rp[:, 1] = torch.cumsum(dy, 0) - dy[0]
    rq = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], -1)
    jp = 0.15 * torch.randn(n, _NJ, generator=g); jv = 0.15 * torch.randn(n, _NJ, generator=g)
    sv = torch.tensor([vx, 0.0, wz], dtype=torch.float32) if steady else None
    return SimpleNamespace(name=f"c{seed}", n=n, root_pos=rp, root_quat=rq, joint_pos=jp,
                           joint_vel=jv, steady=steady, wrap=wrap, steady_vel=sv)


def test_vanilla_rides_periodic_clean_clip():
    from motion_gen.motion_matching.vanilla import VanillaMatcher
    clean = _clip_ns(40, 0.8, seed=1, steady=True, wrap="periodic")
    bones = _clip_ns(70, 0.8, seed=2, steady=False, wrap="episodic")
    m = VanillaMatcher([clean, bones], VanillaConfig(dt=_DT, w_tier=4.0, tier_k=8), device="cpu")
    path, jumped = m._select_path(lambda t: (0.8, 0.0, 0.0), 200)
    p = path.cpu().numpy(); owner = m.owner.cpu().numpy()[p]
    # tier bonus makes the greedy query prefer the clean (STEADY, owner 0) clip most of the time
    assert (owner == 0).mean() > 0.6
    # and it is ridden past its own 40-frame length -> at least one wrap (backward, not a cut)
    wraps = [t for t in range(1, len(p))
             if (not jumped[t]) and owner[t] == owner[t - 1] and p[t] < p[t - 1]]
    assert len(wraps) >= 1


def test_vanilla_periodic_loops_without_tier():
    # the periodic loop in _select_path is independent of the tier cost: even with w_tier=0 a
    # clean clip is ridden past its own length (wraps), and the k=1 query path is unchanged.
    from motion_gen.motion_matching.vanilla import VanillaMatcher
    clean = _clip_ns(50, 0.8, seed=3, steady=True, wrap="periodic")
    m = VanillaMatcher([clean], VanillaConfig(dt=_DT, w_tier=0.0), device="cpu")
    path, jumped = m._select_path(lambda t: (0.8, 0.0, 0.0), 150)   # 150 > 50 -> must wrap
    p = path.cpu().numpy()
    assert (m.owner.cpu().numpy()[p] == 0).all()                    # single clip, stays on it
    wraps = [t for t in range(1, len(p)) if (not jumped[t]) and p[t] < p[t - 1]]
    assert len(wraps) >= 1
