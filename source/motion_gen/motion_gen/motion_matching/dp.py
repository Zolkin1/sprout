# Copyright 2026 Zachary Olkin. All rights reserved.

"""Exact **backward dynamic program** (Viterbi) motion matching over the candidate graph.

Vanilla is greedy; beam is a forward, beam-width-pruned approximate DP.  This is the *exact*
version: because the per-frame ``match_cost`` (velocity-window tracking) depends only on
``(frame, t)`` -- the only path-dependent term in beam was the global-position cost ``w_pos``,
which we DROP here -- the objective becomes a shortest path on a time-layered DAG, solvable
exactly by a backward pass.

    V[f, T-1] = mc(f, T-1)
    V[f, t]   = mc(f, t) + min( V[next_frame[f], t+1],                       # continue (edge 0)
                                min_k jump_cost[f,k] + cut_penalty + V[cand[f,k], t+1] )  # jump

Store the argmin action per state; forward-reconstruct from the fixed ``_stationary`` start.
This is the global optimum of ``sum_t mc(f_t,t) + sum_cuts (jump_cost + cut_penalty)`` -- the
exact target beam approximates.  There is NO cooldown (``cut_penalty`` alone controls
switching) so the state stays a clean ``(frame, t)``.

Because it tracks LOCAL velocities only (no global position term), the world path will drift
from the commanded integrated position -- that is by design; it is what makes the DP exact,
and comparing its drift against beam's position-tracked path is the point of the method.

Playback (world construction) is shared with vanilla/beam via
:func:`~motion_gen.motion_matching.playback.play_frame_path`, so only the *selection* differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .playback import play_frame_path
from .vanilla import VanillaConfig, VanillaMatcher


@dataclass
class DPConfig:
    cut_penalty: float = 3.0           # additive cost per cut (only switching control -- no cooldown)
    w_track: float = 1.0               # velocity-window tracking weight
    w_traj: float = 0.0                # future-TRAJECTORY (local pos+facing lookahead) weight -- the
                                       # standard MM responsiveness signal.  "trajectory DP" = w_track=0,
                                       # w_traj=1.  Pose continuity is handled by the graph/jump_cost.
    track_window: int = 10             # W: velocity-match window half-width (frames)
    track_decay: float = 0.85          # geometric decay per |offset| in the window
    dt: float = 0.02


class DPMatcher(VanillaMatcher):
    """Reuses the vanilla DB build (no greedy KD-tree) + a candidate graph, then runs an exact
    backward DP.  Same output contract as :class:`BeamMatcher` (``generate`` / ``dbg`` /
    ``src_names`` / ``n_cuts``) so the CLI + experiment + diagnostics are method-agnostic."""

    def __init__(self, sources, vcfg: VanillaConfig, dcfg: DPConfig, graph=None,
                 mjcf_path=None, device="cpu"):
        super().__init__(sources, vcfg, mjcf_path=mjcf_path, device=device, build_tree=False)
        self.dcfg = dcfg
        self.bcfg = dcfg                 # alias: mm_experiment._analyze reads `.bcfg.cut_penalty`
        self._owner_np = self.owner.cpu().numpy().astype(np.int64)
        self._ss = self.src_start.cpu().numpy().astype(np.int64)
        self._sl = self.src_len.cpu().numpy().astype(np.int64)
        # per-frame future-trajectory feature (integrated Δpos+facing at cfg.traj_times_s, in the
        # frame's own yaw frame) = vanilla's feat_raw[:, :traj_dim]; the standard MM lookahead.
        self._traj_feat = self.feat_raw[:, :self._traj_dim].detach().cpu().numpy().astype(np.float64)
        w_pos, w_dir = 1.0, 0.5
        self._traj_w = np.array([w for _ in vcfg.traj_times_s
                                 for w in (w_pos, w_pos, w_dir, w_dir)], dtype=np.float64)
        self.warnings: list[str] = []
        self.graph = None
        if graph is not None:
            self.attach_graph(graph)

    def pose_weights(self) -> tuple[float, ...]:
        c = self.cfg
        return (c.w_foot_pos, c.w_foot_vel, c.w_hand_pos, c.w_hand_vel, c.w_root_vel)

    def attach_graph(self, graph) -> None:
        self.graph = graph
        self._cand = graph.candidates                       # [N,K] int32
        self._jcost = graph.jump_cost.astype(np.float64)    # [N,K] (promote from fp16; inf in unused)
        self._next = graph.next_frame                       # [N]  int32 (-1 = clip end)
        self._bvel = graph.body_vel.astype(np.float64)      # [N,3]

    # ---- match cost (velocity window + / or trajectory lookahead) -------------------
    def _match_cost(self, frames: np.ndarray, t: int) -> np.ndarray:
        """Per-frame cost for a SUBSET of frames at time ``t`` = ``w_track`` * velocity-window +
        ``w_traj`` * future-trajectory match (same terms as beam; used for the ``_match_cost_all``
        oracle test + the experiment's cost decomposition).  No pose term -- the graph handles it."""
        dc = self.dcfg
        cost = np.zeros(len(frames), dtype=np.float64)
        if dc.w_track != 0.0:
            W = dc.track_window
            w = np.arange(-W, W + 1)
            decay = dc.track_decay ** np.abs(w)
            owner_f = self._owner_np[frames]
            lo = self._ss[owner_f]; hi = lo + self._sl[owner_f] - 1
            idx = np.clip(frames[:, None] + w[None, :], lo[:, None], hi[:, None])
            diff = self._bvel[idx] - self._target[np.clip(t + w, 0, self._target.shape[0] - 1)][None]
            cost = cost + dc.w_track * ((diff * diff).sum(-1) * decay[None]).sum(-1)
        if dc.w_traj != 0.0:
            d = self._traj_feat[frames] - self._dtraj[t]
            cost = cost + dc.w_traj * (d * d * self._traj_w).sum(-1)
        return cost

    def _precompute_ABC(self) -> None:
        """A[N], B[N,3], C so that, when the target is constant ``g`` over the whole window,
        ``mc(f,t)/w_track = A[f] - 2 g.B[f] + C ||g||^2`` (exact; per-clip frame clamp = beam's)."""
        dc = self.dcfg
        W = dc.track_window
        N = self._bvel.shape[0]
        lo = self._ss[self._owner_np]; hi = lo + self._sl[self._owner_np] - 1
        base = np.arange(N)
        A = np.zeros(N); B = np.zeros((N, 3)); C = 0.0
        for w in range(-W, W + 1):
            dcy = dc.track_decay ** abs(w)
            bvw = self._bvel[np.clip(base + w, lo, hi)]     # [N,3] (same clamp as _match_cost)
            A += dcy * (bvw * bvw).sum(1)
            B += dcy * bvw
            C += dcy
        self._A = A; self._B = B; self._C = float(C); self._mc_cache = {}

    def _precompute_constant_mask(self, T: int) -> None:
        """``constant_window[t]`` = the velocity window at t sees a single target value (so the
        fast A/B/C form is exact).  Conservative: require no command change within +/-W AND t
        far enough from the time-clamp ends; the ~O(#segments*W) other t use the exact path."""
        W = self.dcfg.track_window
        tgt = self._target
        changed = np.zeros(T, dtype=bool)
        if T > 1:
            changed[1:] = np.abs(np.diff(tgt, axis=0)).sum(1) > 1e-9
        csum = np.concatenate([[0], np.cumsum(changed)])
        const = np.zeros(T, dtype=bool)
        for t in range(T):
            if t < W or t > T - 1 - W:
                continue
            lo_c = max(0, t - W + 1); hi_c = min(T - 1, t + W)   # change-indices inside the window
            const[t] = (csum[hi_c + 1] - csum[lo_c]) == 0
        self._constant_window = const

    def _mc_from_g(self, g: np.ndarray) -> np.ndarray:
        key = (round(float(g[0]), 6), round(float(g[1]), 6), round(float(g[2]), 6))
        v = self._mc_cache.get(key)
        if v is None:
            v = self.dcfg.w_track * (self._A - 2.0 * (self._B @ g) + self._C * float(g @ g))
            self._mc_cache[key] = v
        return v

    def _match_cost_all_exact(self, t: int) -> np.ndarray:
        """Exact windowed cost for ALL frames at time ``t`` (chunked over frames)."""
        dc = self.dcfg
        W = dc.track_window
        w = np.arange(-W, W + 1); decay = dc.track_decay ** np.abs(w)
        tt = np.clip(t + w, 0, self._target.shape[0] - 1); tgt = self._target[tt]   # [2W+1,3]
        N = self._bvel.shape[0]
        lo = self._ss[self._owner_np]; hi = lo + self._sl[self._owner_np] - 1
        out = np.empty(N)
        CH = 500_000
        for s in range(0, N, CH):
            e = min(N, s + CH)
            base = np.arange(s, e)
            idx = np.clip(base[:, None] + w[None, :], lo[s:e, None], hi[s:e, None])
            diff = self._bvel[idx] - tgt[None]
            out[s:e] = dc.w_track * ((diff * diff).sum(-1) * decay[None]).sum(-1)
        return out

    def _precompute_desired_traj(self, target: np.ndarray, T: int) -> np.ndarray:
        """Command's integrated future trajectory (Δpos + facing at cfg.traj_times_s horizons, in
        the local frame at ``t``) -- same layout as the DB trajectory feature ``feat_raw[:, :traj_dim]``."""
        dt = self.cfg.dt
        frames = self.cfg.traj_frames(); maxf = max(frames)
        frame_of = {f: i for i, f in enumerate(frames)}
        out = np.zeros((T, 4 * len(frames)), dtype=np.float64)
        for t in range(T):
            px = py = th = 0.0
            for s in range(1, maxf + 1):
                vx, vy, wz = target[min(t + s, T - 1)]
                c, sn = np.cos(th), np.sin(th)
                px += (c * vx - sn * vy) * dt; py += (sn * vx + c * vy) * dt; th += wz * dt
                if s in frame_of:
                    i = frame_of[s]; out[t, 4 * i:4 * i + 4] = (px, py, np.cos(th), np.sin(th))
        return out

    def _match_cost_all(self, t: int) -> np.ndarray:
        """``mc(f,t)`` for every frame at time ``t`` = w_track*velocity + w_traj*trajectory.
        Velocity uses the fast A/B/C form (exact gather near command boundaries); trajectory is a
        direct ``[N,traj_dim]`` compare (no window, no pose term -- the graph handles pose)."""
        dc = self.dcfg
        cost = 0.0
        if dc.w_track != 0.0:
            cost = cost + (self._mc_from_g(self._target[t]) if self._constant_window[t]
                           else self._match_cost_all_exact(t))
        if dc.w_traj != 0.0:
            d = self._traj_feat - self._dtraj[t]                    # [N, traj_dim]
            cost = cost + dc.w_traj * (d * d * self._traj_w).sum(1)
        return cost

    def _prepare(self, command, T: int) -> None:
        self._target = np.array([command(i * self.cfg.dt) for i in range(T)], dtype=np.float64)
        if self.dcfg.w_track != 0.0:
            self._precompute_ABC()
            self._precompute_constant_mask(T)
        if self.dcfg.w_traj != 0.0:
            self._dtraj = self._precompute_desired_traj(self._target, T)

    # ---- the exact backward DP -------------------------------------------------------
    def _dp_backward(self, T: int):
        """Backward pass -> action table ``A_act[N,T]`` (0=continue, else 1+jump-slot); also
        sets ``self._V0`` = cost-to-go from the start layer."""
        N = self._bvel.shape[0]
        cut = self.dcfg.cut_penalty
        cand, jc, nxt = self._cand, self._jcost, self._next
        has_cont = nxt != -1
        safe_next = np.where(has_cont, nxt, 0)               # safe index; masked to inf below
        action = np.zeros((N, T), dtype=np.int16)            # 1+k <= 1+k_max; int16 headroom
        idxN = np.arange(N)

        V_next = self._match_cost_all(T - 1)                 # V[:, T-1]
        for t in range(T - 2, -1, -1):
            Vc = np.where(has_cont, V_next[safe_next], np.inf)          # continue value
            Vj_all = V_next[cand] + jc + cut                           # [N,K] (inf slots stay inf)
            kbest = np.argmin(Vj_all, axis=1)
            Vj = Vj_all[idxN, kbest]
            take_cont = Vc <= Vj                                       # ties -> continue (fewer cuts)
            action[:, t] = np.where(take_cont, 0, 1 + kbest).astype(np.int16)
            V_next = self._match_cost_all(t) + np.where(take_cont, Vc, Vj)
        self._V0 = V_next
        return action

    def _reconstruct(self, action: np.ndarray, T: int):
        cand, nxt = self._cand, self._next
        path = np.zeros(T, dtype=np.int64)
        jumped = [False] * T
        forced = 0
        f = int(self._stationary)
        for t in range(T):
            path[t] = f
            if t == T - 1:
                break
            a = int(action[f, t])
            if a == 0:                                       # continue
                nf = int(nxt[f])
                if nf == -1:                                 # guard: should not happen for finite V
                    self.warnings.append(f"DP continue off clip end at t={t}; stopping.")
                    return path[:t + 1], jumped[:t + 1]
                f = nf
            else:                                            # jump (cut into step t+1)
                if nxt[path[t]] == -1:
                    forced += 1
                f = int(cand[f, a - 1]); jumped[t + 1] = True
        jumped[0] = False
        self.n_cuts = int(sum(jumped))
        if forced:
            self.warnings.append(f"{forced}/{self.n_cuts} cuts were forced at clip ends.")
        return path, jumped

    def generate(self, command, T: int, debug: bool = False):
        """Exact backward-DP match against ``command`` then play the optimal path.
        Returns world ``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,21])``."""
        if self.graph is None:
            raise RuntimeError("DPMatcher.generate called before a candidate graph was attached.")
        self.warnings = []
        self._prepare(command, T)
        action = self._dp_backward(T)
        if not np.isfinite(self._V0[int(self._stationary)]):
            raise RuntimeError("DP found no finite-cost path from the stationary start.")
        path, jumped = self._reconstruct(action, T)
        frame_path = torch.as_tensor(path, dtype=torch.long, device=self.device)
        root_pos, root_quat, joints, dbg = play_frame_path(frame_path, jumped, self, self.cfg, debug=debug)
        self.dbg = dbg
        return root_pos, root_quat, joints
