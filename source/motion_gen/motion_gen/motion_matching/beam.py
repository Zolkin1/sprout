# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline **beam-search** motion matching over a precomputed candidate graph.

Vanilla matching is greedy: it commits, one re-query at a time, to whatever frame looks
best *now*.  Offline we know the whole target command sequence in advance, so we can search
*globally* for the minimum-cost **path** of DB frames.  The path is built from two moves at
each step: **continue** the current clip (``next_frame`` edge, no jump cost) or **jump** to
one of the frame's precomputed top-K pose neighbours (``jump_cost`` + a fixed ``cut_penalty``).
A frame's cost at time ``t`` is a **bidirectional** velocity-tracking term -- it compares a
local window of the frame's body velocity against the *known* target both backward and
forward -- which is strictly more information than a real-time forward-only matcher has.

A beam (width ``B``) keeps the ``B`` cheapest partial paths; ``min_cut_gap`` + ``cut_penalty``
stop it from thrashing (jumping every frame for tiny local gains).  Playback (world
construction, inertialisation) is shared with vanilla via
:func:`~motion_gen.motion_matching.playback.play_frame_path`, so only the *selection* differs.

Bones-only for now: every frame is VARIED, so the STEADY-tier term is inert (``w_tier=0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .graph import STEADY, VARIED  # noqa: F401  (STEADY kept for the tier hook)
from .playback import play_frame_path
from .vanilla import VanillaConfig, VanillaMatcher


@dataclass
class BeamConfig:
    beam_width: int = 128
    min_cut_gap_s: float = 0.25        # cooldown between cuts (anti-thrash)
    cut_penalty: float = 3.0           # additive cost per jump
    w_track: float = 1.0               # instantaneous velocity-window tracking (local smoothness/gait)
    w_pos: float = 2.0                 # GLOBAL position tracking: accumulated world pos vs integrated ref
    w_traj: float = 0.0                # LOCAL 1s-trajectory match (superseded by w_pos; off by default)
    w_tier: float = 0.0                # clean STEADY-tier preference (0 = off / bones-only)
    track_window: int = 10             # W: velocity-match window half-width (frames)
    track_decay: float = 0.85          # geometric decay per |offset| in the window
    soft_tier_penalty: float = 2.0
    regime_ramp_s: float = 0.5
    tier_one_sided: bool = True        # one-sided (prefer clean in steady only) vs two-sided
    tier_nn_k: int = 1                 # reward the k velocity-nearest clean clips (k=1 = single NN)
    dt: float = 0.02


class BeamMatcher(VanillaMatcher):
    """Reuses the vanilla feature/DB build (no greedy KD-tree) + a candidate graph, then
    runs the beam-search DP.  Exposes the same ``generate`` / ``dbg`` / ``src_names`` as
    :class:`VanillaMatcher` so the CLI + diagnostics are method-agnostic."""

    def __init__(self, sources, vcfg: VanillaConfig, bcfg: BeamConfig, graph=None,
                 mjcf_path=None, device="cpu"):
        super().__init__(sources, vcfg, mjcf_path=mjcf_path, device=device, build_tree=False)
        self.bcfg = bcfg
        self._owner_np = self.owner.cpu().numpy().astype(np.int64)
        self._ss = self.src_start.cpu().numpy().astype(np.int64)
        self._sl = self.src_len.cpu().numpy().astype(np.int64)
        # Per-frame FUTURE-trajectory feature (integrated root displacement + facing at the
        # cfg.traj_times_s horizons, in the frame's own yaw frame) -- reuse vanilla's feature:
        # it is exactly feat_raw[:, :traj_dim].  Matched against the command's integrated
        # trajectory in the DP so beam tracks POSITION, not just instantaneous velocity.
        self._traj_feat = self.feat_raw[:, :self._traj_dim].detach().cpu().numpy().astype(np.float64)
        w_pos, w_dir = 1.0, 0.5            # facing is partly covered by wz velocity tracking
        self._traj_w = np.array([w for _ in vcfg.traj_times_s
                                 for w in (w_pos, w_pos, w_dir, w_dir)], dtype=np.float64)
        self._tier_rewarded = None                          # [T,C] set in _prepare when w_tier!=0
        self.warnings: list[str] = []
        self.graph = None
        if graph is not None:
            self.attach_graph(graph)

    def pose_weights(self) -> tuple[float, ...]:
        """The pose-block weights (feet/hands/root-vel) that define the jump metric --
        folded into the graph cache key so a weight change invalidates the cached graph."""
        c = self.cfg
        return (c.w_foot_pos, c.w_foot_vel, c.w_hand_pos, c.w_hand_vel, c.w_root_vel)

    def attach_graph(self, graph) -> None:
        """Bind a (freshly built or cache-loaded) candidate graph for the DP."""
        self.graph = graph
        self._cand = graph.candidates                       # [N,K] int32
        self._jcost = graph.jump_cost.astype(np.float64)    # [N,K] (promote from fp16)
        self._next = graph.next_frame                       # [N]  int32 (-1 = clip end)
        self._bvel = graph.body_vel.astype(np.float64)      # [N,3]
        self._tier = graph.tier                             # [N]  uint8

    # ---- per-target precompute -------------------------------------------------------
    def _precompute_target(self, command, T: int) -> np.ndarray:
        return np.array([command(i * self.cfg.dt) for i in range(T)], dtype=np.float64)  # [T,3]

    def _precompute_regime(self, target: np.ndarray) -> np.ndarray:
        """Smoothed steady/transition signal in [0,1] (1 = steady), ramped over
        ``regime_ramp_s`` around command changes.  Inert while all tiers are VARIED."""
        T = target.shape[0]
        changed = np.zeros(T, dtype=bool)
        if T > 1:
            changed[1:] = np.abs(np.diff(target, axis=0)).sum(1) > 1e-6
        # frames-to-nearest-change (both directions)
        big = T + 1
        dpast = np.full(T, big); d = big
        for t in range(T):
            d = 0 if changed[t] else d + 1
            dpast[t] = d
        dfut = np.full(T, big); d = big
        for t in range(T - 1, -1, -1):
            d = 0 if changed[t] else d + 1
            dfut[t] = d
        dist = np.minimum(dpast, dfut).astype(np.float64) * self.cfg.dt
        x = np.clip(dist / max(1e-6, self.bcfg.regime_ramp_s), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)                       # smoothstep -> steady weight

    def _precompute_desired_traj(self, target: np.ndarray, T: int) -> np.ndarray:
        """Command's integrated FUTURE trajectory at each time ``t``: root displacement
        (px, py) + facing (cos, sin) at each ``cfg.traj_times_s`` horizon, in the local frame
        (heading 0 at ``t``) -- same layout as the DB trajectory feature.  Integrates the
        KNOWN future target (so command changes within the horizon are handled)."""
        dt = self.cfg.dt
        frames = self.cfg.traj_frames()
        maxf = max(frames)
        frame_of = {f: i for i, f in enumerate(frames)}
        out = np.zeros((T, 4 * len(frames)), dtype=np.float64)
        for t in range(T):
            px = py = th = 0.0
            for s in range(1, maxf + 1):
                vx, vy, wz = target[min(t + s, T - 1)]
                c, sn = np.cos(th), np.sin(th)
                px += (c * vx - sn * vy) * dt
                py += (sn * vx + c * vy) * dt
                th += wz * dt
                if s in frame_of:
                    i = frame_of[s]
                    out[t, 4 * i:4 * i + 4] = (px, py, np.cos(th), np.sin(th))
        return out

    def _precompute_reference_pos(self, target: np.ndarray, T: int) -> np.ndarray:
        """The command's integrated **world** position at each step ``[T,2]`` (from origin,
        heading 0 -- same start as playback).  The DP compares each path's accumulated world
        position against this so falling behind (not going backward/strafing/full-speed)
        accrues a growing, unavoidable cost -> the beam is forced to commit to the motion."""
        dt = self.cfg.dt
        ref = np.zeros((T, 2), dtype=np.float64)
        x = y = th = 0.0
        for t in range(1, T):
            vx, vy, wz = target[t]
            c, s = np.cos(th), np.sin(th)
            x += (c * vx - s * vy) * dt
            y += (s * vx + c * vy) * dt
            th += wz * dt
            ref[t] = (x, y)
        return ref

    def _prepare(self, command, T: int) -> None:
        """Precompute the per-target arrays the DP reads (target velocity, reference position,
        desired trajectory, regime).  Called by ``generate``; tests that drive ``_beam_search``
        directly call it too."""
        self._target = self._precompute_target(command, T)
        self._ref_xy = (self._precompute_reference_pos(self._target, T)
                        if self.bcfg.w_pos != 0.0 else None)
        self._dtraj = (self._precompute_desired_traj(self._target, T)
                       if self.bcfg.w_traj != 0.0 else None)
        self._regime = self._precompute_regime(self._target) if self.bcfg.w_tier != 0.0 else None
        self._tier_rewarded = (self._precompute_tier_nn(self._target, self.bcfg.tier_nn_k)
                               if self.bcfg.w_tier != 0.0 else None)

    def _match_cost(self, frames: np.ndarray, t: int) -> np.ndarray:
        """Velocity-window + future-trajectory tracking cost for ``frames`` at time ``t``."""
        bc = self.bcfg
        W = bc.track_window
        w = np.arange(-W, W + 1)
        decay = bc.track_decay ** np.abs(w)                  # [2W+1]
        owner_f = self._owner_np[frames]
        lo = self._ss[owner_f]; hi = lo + self._sl[owner_f] - 1
        idx = np.clip(frames[:, None] + w[None, :], lo[:, None], hi[:, None])   # [M,2W+1]
        bv = self._bvel[idx]                                                    # [M,2W+1,3]
        tt = np.clip(t + w, 0, self._target.shape[0] - 1)
        tgt = self._target[tt]                                                  # [2W+1,3]
        diff = bv - tgt[None]
        track = ((diff * diff).sum(-1) * decay[None]).sum(-1)                   # [M]
        cost = bc.w_track * track
        if bc.w_traj != 0.0:
            d = self._traj_feat[frames] - self._dtraj[t][None]                  # [M, traj_dim]
            cost = cost + bc.w_traj * (d * d * self._traj_w[None]).sum(-1)
        if bc.w_tier != 0.0 and self._tier_rewarded is not None:
            r = float(self._regime[t])
            reward = self._tier_rewarded[t][self._owner_np[frames]]     # 1 = velocity-nearest clean clip
            # one-sided: reward the command-matching clean clip during steady regime (penalize
            # everything else); two-sided additionally prefers non-matching frames in transitions.
            mism = r * (1.0 - reward)
            if not bc.tier_one_sided:
                mism = mism + (1.0 - r) * reward
            cost = cost + bc.w_tier * bc.soft_tier_penalty * mism
        return cost

    # ---- the DP ----------------------------------------------------------------------
    def _beam_search(self, T: int):
        bc = self.bcfg
        B = bc.beam_width
        min_gap = max(1, int(round(bc.min_cut_gap_s / self.cfg.dt)))
        cand, jcost, nxt = self._cand, self._jcost, self._next

        dt = self.cfg.dt
        use_pos = bc.w_pos != 0.0
        bvel = self._bvel

        # frontier at t=0: the single stationary start, cut allowed immediately
        fr = np.array([self._stationary], dtype=np.int64)
        cost = np.zeros(1, dtype=np.float64)
        gap = np.array([10 ** 6], dtype=np.int64)
        # accumulated world pose per state (x, y, yaw), integrated from body velocity along
        # the path; starts at the origin/heading 0 (matching playback) -> compared to _ref_xy
        wx = np.zeros(1); wy = np.zeros(1); wyaw = np.zeros(1)
        step_frames = [fr.copy()]
        step_bp = [np.array([-1], dtype=np.int64)]
        step_jump = [np.array([False])]
        forced = 0

        for t in range(1, T):
            # --- continue edges ---
            nf = nxt[fr]                                     # [B'] (-1 = clip end)
            cmask = nf != -1
            cont_frame = nf[cmask].astype(np.int64)
            cont_base = cost[cmask]
            cont_gap = gap[cmask] + 1
            cont_src = np.nonzero(cmask)[0]

            # --- jump edges (cooldown, or forced when the clip has ended) ---
            allow = (gap >= min_gap) | (nf == -1)
            asrc = np.nonzero(allow)[0]
            if asrc.size:
                jfr = fr[asrc]                               # [A]
                jc = cand[jfr].astype(np.int64)              # [A,K]
                jcv = jcost[jfr]                             # [A,K]
                jump_frame = jc.reshape(-1)
                jump_base = (cost[asrc][:, None] + jcv + bc.cut_penalty).reshape(-1)
                jump_src = np.repeat(asrc, jc.shape[1])
                finite = np.isfinite(jump_base)              # drop unused (inf) graph slots
                jump_frame = jump_frame[finite]
                jump_base = jump_base[finite]
                jump_src = jump_src[finite]
            else:
                jump_frame = np.empty(0, np.int64)
                jump_base = np.empty(0, np.float64)
                jump_src = np.empty(0, np.int64)
            jump_gap = np.zeros(jump_frame.shape[0], dtype=np.int64)

            cand_frame = np.concatenate([cont_frame, jump_frame])
            cand_base = np.concatenate([cont_base, jump_base])
            cand_gap = np.concatenate([cont_gap, jump_gap])
            cand_src = np.concatenate([cont_src, jump_src])
            cand_isjump = np.concatenate([
                np.zeros(cont_frame.shape[0], bool), np.ones(jump_frame.shape[0], bool)])
            if cand_frame.size == 0:
                self.warnings.append(f"beam dead-ended at t={t}; truncating.")
                T = t
                break

            total = cand_base + self._match_cost(cand_frame, t)

            # global position tracking: extend each parent's accumulated world pose by one
            # step of the candidate frame's body velocity, then charge the squared distance
            # to the integrated reference position.  Falling behind accrues a growing cost.
            if use_pos:
                pyaw = wyaw[cand_src]
                bvc = bvel[cand_frame]                        # [M,3] vx,vy,wz (body frame)
                c = np.cos(pyaw); s = np.sin(pyaw)
                new_wx = wx[cand_src] + (c * bvc[:, 0] - s * bvc[:, 1]) * dt
                new_wy = wy[cand_src] + (s * bvc[:, 0] + c * bvc[:, 1]) * dt
                new_wyaw = pyaw + bvc[:, 2] * dt
                rx, ry = self._ref_xy[t]
                total = total + bc.w_pos * ((new_wx - rx) ** 2 + (new_wy - ry) ** 2)

            # dedup by frame (keep min-cost), then top-B
            order = np.argsort(total, kind="stable")
            _, first = np.unique(cand_frame[order], return_index=True)
            keep = order[first]
            if keep.size > B:
                sel = np.argpartition(total[keep], B)[:B]
                keep = keep[sel]
            keep = keep[np.argsort(total[keep], kind="stable")]

            fr = cand_frame[keep]
            cost = total[keep]
            gap = cand_gap[keep]
            if use_pos:
                wx = new_wx[keep]; wy = new_wy[keep]; wyaw = new_wyaw[keep]
            step_frames.append(fr.copy())
            step_bp.append(cand_src[keep].copy())
            step_jump.append(cand_isjump[keep].copy())

        # backtrack the cheapest full path
        path = np.zeros(T, dtype=np.int64)
        jumped = [False] * T
        b = int(np.argmin(cost))
        for t in range(T - 1, -1, -1):
            path[t] = step_frames[t][b]
            jumped[t] = bool(step_jump[t][b])
            b = int(step_bp[t][b])
        jumped[0] = False

        n_cuts = int(sum(jumped))
        for t in range(1, T):
            if jumped[t] and nxt[path[t - 1]] == -1:
                forced += 1
        if forced:
            self.warnings.append(
                f"{forced}/{n_cuts} cuts were forced at clip ends (min_cut_gap may exceed "
                f"typical clip length).")
        self.n_cuts = n_cuts
        return path, jumped

    def generate(self, command, T: int, debug: bool = False):
        """Beam-search the best DB-frame path against ``command`` then play it.
        Returns world ``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,21])``."""
        if self.graph is None:
            raise RuntimeError("BeamMatcher.generate called before a candidate graph was attached.")
        self.warnings = []
        self._prepare(command, T)
        path, jumped = self._beam_search(T)
        frame_path = torch.as_tensor(path, dtype=torch.long, device=self.device)
        root_pos, root_quat, joints, dbg = play_frame_path(
            frame_path, jumped, self, self.cfg, debug=debug)
        self.dbg = dbg
        return root_pos, root_quat, joints
