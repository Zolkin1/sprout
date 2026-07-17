# Copyright 2026 Zachary Olkin. All rights reserved.

"""Precomputed **candidate graph** for offline (beam-search) motion matching.

Vanilla motion matching searches greedily over the whole DB at run time.  Offline we know
the whole target command sequence, so we can search *globally* with a DP -- but a DP needs a
bounded branching factor.  This module precomputes each frame's jump options once (cached
like the feature cache).

Design (why it is the way it is):

* **Postural selection metric.**  Candidates are the nearest frames by *body posture* only
  (feet + hand positions relative to the root), NOT velocity.  The full pose feature is half
  velocity, so pose-nearest-neighbour would fence a frame into its own behaviour (a forward
  frame's neighbours would all be other forward frames).  Selecting on posture alone lets a
  frame reach frames heading *different* ways (a mid-stance posture connects "about to step
  forward / back / sideways") -- that reachable diversity is what lets the beam steer.
* **epsilon-ball, clamped to [k_min, k_max].**  A jump is valid iff the two poses are close
  enough to bridge -- a property of *distance*, not *rank*.  So candidates = frames within a
  postural radius ``eps`` (auto-calibrated so the median frame gets ~``k_target``), but never
  fewer than ``k_min`` (fall back to nearest, so no frame is disconnected) nor more than
  ``k_max`` (bound memory / branching).  Dense transition hubs get many options; sparse
  frames still get a floor.
* **Same-clip cap.**  At most ``same_clip_cap`` candidates may come from f's own clip (beyond
  the +/- ``min_same_clip_gap`` trivial-neighbour exclusion), so the fan-out is not wasted on
  "same clip, one stride over" (which cannot change behaviour).
* **jump_cost = FULL pose distance** (feet+hands+root-vel, incl. velocity).  Selection is
  velocity-agnostic (for reach), but the *cost* of taking a jump includes the velocity
  discontinuity, so the beam prefers smooth transitions and only pays for a big velocity
  change (e.g. forward->backward) when the command justifies it.

Also stores ``next_frame`` (the continue edge), ``body_vel`` (per-frame vx/vy/wz for the
beam's tracking cost) and ``tier`` (STEADY/VARIED; all VARIED for bones-only).  Unused
candidate slots carry ``inf`` jump_cost and are ignored by the beam, so a variable per-frame
fan-out is stored in a fixed ``[N, k_max]`` array.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_GRAPH_CACHE_VERSION = 2   # v2: postural eps-ball selection + same-clip cap (was v1 top-K full-pose)

STEADY = 0
VARIED = 1


@dataclass
class GraphConfig:
    k_min: int = 25                  # floor: always keep at least this many nearest (connectivity)
    k_max: int = 50                  # cap: never store more than this (memory / branching bound)
    k_target: int = 40               # eps auto-set so the median frame gets ~this many in-ball
    same_clip_cap: int = 2           # at most this many candidates from f's own clip
    min_same_clip_gap: int = 3       # drop f's own-clip neighbours within +/- this many frames
    dt: float = 0.02
    tier_mode: str = "bones_only"    # inert hook; "bones_only" -> every frame VARIED


@dataclass
class CandidateGraph:
    """Per-frame jump graph + tracking metadata (all CPU numpy)."""

    candidates: np.ndarray   # [N, k_max] int32  global frame indices (jump targets)
    jump_cost: np.ndarray    # [N, k_max] float16 FULL pose L2 (inf in unused / beyond-n_keep slots)
    next_frame: np.ndarray   # [N]    int32  f+1 within clip, else -1
    body_vel: np.ndarray     # [N, 3] float32 (vx_fwd, vy_left, wz_yaw)
    tier: np.ndarray         # [N]    uint8  STEADY / VARIED
    n: int
    k: int                   # = k_max (candidate array width)
    key: str | None = None
    eps: float | None = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            candidates=self.candidates,
            jump_cost=self.jump_cost,
            next_frame=self.next_frame,
            body_vel=self.body_vel,
            tier=self.tier,
            meta=np.array(json.dumps({"version": _GRAPH_CACHE_VERSION, "key": self.key,
                                      "n": self.n, "k": self.k, "eps": self.eps})),
        )

    @staticmethod
    def load(path: str | Path, expected_key: str | None = None) -> "CandidateGraph | None":
        """Load a cached graph, or ``None`` on any miss / version / key mismatch."""
        path = Path(path)
        if not path.exists():
            return None
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
        except Exception as e:  # noqa: BLE001 - corrupt / unreadable cache
            print(f"[WARN] CandidateGraph: failed to read {path} ({e}); rebuilding.")
            return None
        if meta.get("version") != _GRAPH_CACHE_VERSION or (
            expected_key is not None and meta.get("key") != expected_key
        ):
            return None
        return CandidateGraph(
            candidates=z["candidates"], jump_cost=z["jump_cost"], next_frame=z["next_frame"],
            body_vel=z["body_vel"], tier=z["tier"], n=int(meta["n"]), k=int(meta["k"]),
            key=meta.get("key"), eps=meta.get("eps"),
        )


def graph_cache_key(
    bones_cache_path: str,
    *,
    filters: dict | None,
    styles: list[str] | None,
    exclude_mirror: bool,
    exclude_name_substrings: list[str] | None,
    limit: int | None,
    pose_weights: tuple[float, ...],
    cfg: GraphConfig,
) -> str:
    """Content hash of everything the graph depends on (mirrors the library-cache key).

    Folds the bones cache file identity (path/size/mtime), the resolved filter set (which
    frames are in the DB), the pose weights (the metric), and the graph params + version.
    Any change -> a different key -> a fresh build (never a stale hit).
    """
    h = hashlib.sha256()
    h.update(f"version={_GRAPH_CACHE_VERSION};".encode())
    p = Path(bones_cache_path)
    st = p.stat()
    h.update(f"{p}:{st.st_size}:{st.st_mtime_ns};".encode())

    def _norm_list(x):
        return sorted(x) if x else None

    payload = {
        "filters": {k: _norm_list(v) if isinstance(v, (list, tuple, set)) else v
                    for k, v in (filters or {}).items()},
        "styles": _norm_list(styles),
        "exclude_mirror": bool(exclude_mirror),
        "exclude_name": _norm_list(exclude_name_substrings),
        "limit": limit,
        "pose_weights": [round(float(w), 6) for w in pose_weights],
        "k_min": cfg.k_min, "k_max": cfg.k_max, "k_target": cfg.k_target,
        "same_clip_cap": cfg.same_clip_cap, "min_same_clip_gap": cfg.min_same_clip_gap,
        "dt": round(float(cfg.dt), 6), "tier_mode": cfg.tier_mode,
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()


def graph_cache_path(key: str) -> Path:
    """``~/.cache/robot_rl/mm_graph/<key>.npz`` (override dir with ``ROBOT_RL_MM_GRAPH_DIR``)."""
    cache_dir = os.environ.get("ROBOT_RL_MM_GRAPH_DIR")
    base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "robot_rl" / "mm_graph"
    return base / f"{key}.npz"


def _central_diff_np(x: np.ndarray, dt: float) -> np.ndarray:
    """Central-interior / one-sided-end time derivative of a 1-D series (matches
    :func:`ingest_common.central_diff`)."""
    n = x.shape[0]
    d = np.zeros_like(x)
    if n == 1:
        return d
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


def _body_vel(feat_raw: torch.Tensor, yaw_all: torch.Tensor,
              src_start: torch.Tensor, src_len: torch.Tensor, dt: float) -> np.ndarray:
    """Per-frame body-frame ``(vx_fwd, vy_left, wz_yaw)`` [N,3] float32.

    vx/vy are the yaw-aligned root linear velocity already stored as the LAST 3 feature
    dims (m/s); wz is the per-clip **unwrapped** yaw central-difference (rad/s) -- unwrap +
    per-clip so clip seams and +/-pi wraps do not inject spurious spikes.
    """
    N = feat_raw.shape[0]
    bv = np.zeros((N, 3), dtype=np.float32)
    bv[:, 0] = feat_raw[:, -3].cpu().numpy()   # vx forward
    bv[:, 1] = feat_raw[:, -2].cpu().numpy()   # vy left
    yaw = yaw_all.cpu().numpy().astype(np.float64)
    ss = src_start.cpu().numpy(); sl = src_len.cpu().numpy()
    for a, n in zip(ss, sl):
        a = int(a); n = int(n)
        seg = np.unwrap(yaw[a:a + n])
        bv[a:a + n, 2] = _central_diff_np(seg, dt).astype(np.float32)
    return bv


def build_candidate_graph(
    select_feat: torch.Tensor,   # [N, Ds] POSTURAL metric for candidate selection (feet+hand pos)
    cost_feat: torch.Tensor,     # [N, Dp] FULL pose block (incl. velocity) for jump_cost
    feat_raw: torch.Tensor,      # [N, D]  raw vanilla feature (for body vx/vy = last 3 dims)
    owner: torch.Tensor,         # [N] long
    src_start: torch.Tensor,     # [C] long
    src_len: torch.Tensor,       # [C] long
    yaw_all: torch.Tensor,       # [N] float (world yaw per frame)
    cfg: GraphConfig,
    *,
    steady_owner_mask: torch.Tensor | None = None,  # [C] bool -> those clips' frames tagged STEADY
    key: str | None = None,
) -> CandidateGraph:
    from scipy.spatial import cKDTree

    N = int(select_feat.shape[0])
    kmin, kmax = int(cfg.k_min), int(cfg.k_max)
    ktarget = min(int(cfg.k_target), kmax)
    gap = int(cfg.min_same_clip_gap)
    sccap = int(cfg.same_clip_cap)
    owner_np = owner.cpu().numpy().astype(np.int64)
    ss = src_start.cpu().numpy().astype(np.int64)
    sl = src_len.cpu().numpy().astype(np.int64)
    local = np.arange(N, dtype=np.int64) - ss[owner_np]

    # continue edge: f+1 unless f is the last frame of its clip
    is_last = (local + 1) >= sl[owner_np]
    next_frame = np.where(is_last, np.int32(-1), (np.arange(N) + 1).astype(np.int32))

    # ---- postural nearest neighbours (KD-tree over the position-only metric) ----
    sel = select_feat.detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(sel)
    kq = min(N, kmax + 2 * gap + sccap + 24)     # query extra so kmax survive the exclusions
    dist, idx = tree.query(sel, k=kq, workers=-1)  # [N, kq] sorted ascending (col 0 = self)
    del sel, tree

    nb_owner = owner_np[idx]                                    # [N, kq]
    nb_local = idx - ss[nb_owner]
    same_clip = nb_owner == owner_np[:, None]
    # drop self + trivial within-gap same-clip neighbours
    too_close = same_clip & (np.abs(nb_local - local[:, None]) <= gap)
    dist = np.where(too_close, np.inf, dist)
    # cap same-clip candidates: keep only the `sccap` nearest own-clip frames (beyond the gap)
    sc_elig = same_clip & np.isfinite(dist)
    sc_rank = np.cumsum(sc_elig, axis=1)
    dist = np.where(sc_elig & (sc_rank > sccap), np.inf, dist)

    # nearest kmax survivors per frame (by the postural metric)
    order = np.argsort(dist, axis=1, kind="stable")[:, :kmax]   # [N, kmax]
    candidates = np.take_along_axis(idx, order, axis=1).astype(np.int32)
    sel_dist = np.take_along_axis(dist, order, axis=1)          # [N, kmax] (inf = invalid slot)
    n_valid = np.isfinite(sel_dist).sum(1)

    # ---- eps auto-calibration: median frame should get ~k_target in-ball neighbours ----
    kth = sel_dist[:, ktarget - 1]
    finite_kth = kth[np.isfinite(kth)]
    eps = float(np.median(finite_kth)) if finite_kth.size else float("inf")

    # per-frame kept count: within eps, clamped to [k_min, k_max], never above n_valid
    n_in_ball = (sel_dist <= eps).sum(1)
    n_keep = np.clip(n_in_ball, kmin, kmax)
    n_keep = np.minimum(n_keep, n_valid).astype(np.int64)
    short = int((n_valid < kmin).sum())
    if short:
        print(f"[WARN] CandidateGraph: {short}/{N} frames have < k_min={kmin} valid jump targets.")
    print(f"[graph] eps={eps:.3f}  kept/frame: min {int(n_keep.min())} median "
          f"{int(np.median(n_keep))} max {int(n_keep.max())}  (k_min={kmin} k_max={kmax})")

    # ---- jump_cost = FULL pose distance to the kept candidates (chunked; inf elsewhere) ----
    cost_np = cost_feat.detach().cpu().numpy().astype(np.float64)
    jump_cost = np.full((N, kmax), np.inf, dtype=np.float32)
    cols = np.arange(kmax)[None, :]
    CH = 100_000
    for s in range(0, N, CH):
        e = min(N, s + CH)
        cf = cost_np[candidates[s:e]]                          # [c, kmax, Dp]
        d = np.sqrt(((cf - cost_np[s:e][:, None, :]) ** 2).sum(-1))
        keep = (cols < n_keep[s:e][:, None]) & np.isfinite(sel_dist[s:e])
        jump_cost[s:e] = np.where(keep, d, np.inf)
    jump_cost = jump_cost.astype(np.float16)
    del cost_np

    body_vel = _body_vel(feat_raw, yaw_all, src_start, src_len, cfg.dt)
    wz_max = float(np.abs(body_vel[:, 2]).max()) if N else 0.0
    if wz_max > 30.0:
        print(f"[WARN] CandidateGraph: |wz| up to {wz_max:.1f} rad/s -- check per-clip yaw unwrap.")

    tier = np.full(N, VARIED, dtype=np.uint8)
    if steady_owner_mask is not None:
        m = steady_owner_mask.cpu().numpy().astype(bool)
        tier[m[owner_np]] = STEADY

    return CandidateGraph(candidates=candidates, jump_cost=jump_cost, next_frame=next_frame,
                          body_vel=body_vel, tier=tier, n=N, k=kmax, key=key, eps=eps)
