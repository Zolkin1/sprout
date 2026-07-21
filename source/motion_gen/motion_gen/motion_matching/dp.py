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

from .graph import STEADY, VARIED
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
    # Clean steady-state preference (STEADY tier).  ONE-SIDED: during steady command regions add a
    # penalty to VARIED (bones) frames so clean/STEADY clips win; ramped to 0 (neutral) around
    # command changes so transitions still use bones.  w_tier=0 -> exactly the no-clean DP.
    w_tier: float = 0.0
    soft_tier_penalty: float = 2.0     # magnitude of the per-frame penalty (steady regime)
    regime_ramp_s: float = 0.5         # steady->neutral ramp half-width (~1 s total around a change)
    tier_nn_k: int = 1                 # reward the k clean clips nearest the command by velocity
                                       # (k=1 = the single best-matching "waypoint" trajectory);
                                       # velocity ties broken toward the stillest clip.
    # Ground/elevation penalty.  The descriptors are HEIGHT-BLIND (trajectory = relative Δxy, pose =
    # feet relative to the pelvis), so a "walking/standing ON A BOX" frame is descriptor-identical to
    # walking on the ground -- a ``come_up_50cm_box`` lead-in can leak an on-the-box (airborne) segment
    # into a flat run-up for free.  This penalizes candidates whose pelvis is above ``max_pelvis_z``
    # over the DP's ground.  ``ground_z`` is that ground height: 0 for the flat approach/exit; a future
    # terrain-aware DP can pass the terrain surface so the SAME term is correct on stairs/boxes (a
    # legitimately-elevated pose over its own terrain is then NOT penalized).  w_ground=0 -> off.
    w_ground: float = 0.0
    ground_z: float = 0.0
    max_pelvis_z: float = 0.95         # pelvis-above-ground ceiling (standing ~0.78, running bounce ~0.9)
    dt: float = 0.02


class DPMatcher(VanillaMatcher):
    """Reuses the vanilla DB build (no greedy KD-tree) + a candidate graph, then runs an exact
    backward DP.  Same output contract as :class:`BeamMatcher` (``generate`` / ``dbg`` /
    ``src_names`` / ``n_cuts``) so the CLI + experiment + diagnostics are method-agnostic."""

    def __init__(self, sources, vcfg: VanillaConfig, dcfg: DPConfig, graph=None,
                 mjcf_path=None, device="cpu", categories: list[str] | None = None):
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
        self._tier_np = self._tier.cpu().numpy().astype(np.int64)   # [N] STEADY/VARIED (from sources)
        self._pelvis_z = self.root_pos[:, 2].detach().cpu().numpy().astype(np.float64)  # [N] abs pelvis height
        # Per-frame source CATEGORY (e.g. "flat"/"clean"/"leadin"), so a generate() call can restrict
        # the DP to a subset of the DB -- the candidate graph is built ONCE over the union of all
        # categories and each tile type filters it per-generate (see :meth:`_set_category_filter`).
        if categories is not None and len(categories) != len(self.src_names):
            raise ValueError(f"categories ({len(categories)}) must match #sources ({len(self.src_names)}).")
        self._frame_cat = (np.array(categories, dtype=object)[self._owner_np]
                           if categories is not None else None)      # [N] category per frame, or None
        self._allowed_frame = None                                   # [N] bool set per-generate (None = all)
        self._cat_penalty = None                                     # [N] additive +inf on disallowed frames
        self._mc_scale = None                                       # [T] per-step tracking-cost scale (transition window)
        self._boundary_pin = None                                   # [N] per-frame boundary distance (approach funnel)
        self._pin_scale = None                                      # [T] weight ramping the pin up over the window
        self._regime = None                                         # [T] set in _prepare when w_tier!=0
        self._tier_rewarded = None                                  # [T,C] set in _prepare when w_tier!=0
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

    def _set_category_filter(self, allowed) -> None:
        """Restrict the DP to frames whose source category is in ``allowed`` (an iterable of category
        names, e.g. ``{"flat", "clean"}``); ``None`` clears the filter.  Disallowed frames get a +inf
        match cost, so the backward DP never routes onto them.  Safe on the SHARED candidate graph
        because a frame's CONTINUE edge stays within its own clip (same category) -- an allowed frame
        is never stranded mid-clip; only a dead-end allowed frame (clip end + every jump filtered) gets
        cost-to-go +inf and is simply avoided.  An unreachable start raises in :meth:`generate`."""
        if allowed is None or self._frame_cat is None:
            self._allowed_frame = None
            self._cat_penalty = None
            return
        self._allowed_frame = np.isin(self._frame_cat, list(allowed))
        self._cat_penalty = np.where(self._allowed_frame, 0.0, np.inf)

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
        if dc.w_tier != 0.0 and self._tier_rewarded is not None:
            reward = self._tier_rewarded[t][self._owner_np[frames]]           # 1 = velocity-nearest clean
            cost = cost + (dc.w_tier * dc.soft_tier_penalty * float(self._regime[t]) * (1.0 - reward))
        if dc.w_ground != 0.0:
            cost = cost + dc.w_ground * np.maximum(0.0, self._pelvis_z[frames] - dc.ground_z - dc.max_pelvis_z)
        if self._cat_penalty is not None:
            cost = cost + self._cat_penalty[frames]
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
        track = 0.0
        if dc.w_track != 0.0:
            track = track + (self._mc_from_g(self._target[t]) if self._constant_window[t]
                             else self._match_cost_all_exact(t))
        if dc.w_traj != 0.0:
            d = self._traj_feat - self._dtraj[t]                    # [N, traj_dim]
            track = track + dc.w_traj * (d * d * self._traj_w).sum(1)
        # Transition window: ramp the command-tracking weight down (``_mc_scale``) and the terrain-entry
        # pin up (``_pin_scale``) over the last frames, so the DP smoothly transitions from tracking the
        # steady approach to funnelling into the pinned entry -- a distributed pin (not just at the
        # terminal) gives the deceleration ROOM (a terminal-only pin makes it run then hard-stop).
        cost = track if self._mc_scale is None else (float(self._mc_scale[t]) * track)
        if self._boundary_pin is not None:
            cost = cost + float(self._pin_scale[t]) * self._boundary_pin
        if dc.w_tier != 0.0 and self._tier_rewarded is not None:
            reward = self._tier_rewarded[t][self._owner_np]        # [N] 1 = velocity-nearest clean clip
            cost = cost + (dc.w_tier * dc.soft_tier_penalty * float(self._regime[t]) * (1.0 - reward))
        if dc.w_ground != 0.0:
            cost = cost + dc.w_ground * np.maximum(0.0, self._pelvis_z - dc.ground_z - dc.max_pelvis_z)
        if self._cat_penalty is not None:
            cost = cost + self._cat_penalty
        return cost

    def _prepare(self, command, T: int) -> None:
        self._target = np.array([command(i * self.cfg.dt) for i in range(T)], dtype=np.float64)
        if self.dcfg.w_track != 0.0:
            self._precompute_ABC()
            self._precompute_constant_mask(T)
        if self.dcfg.w_traj != 0.0:
            self._dtraj = self._precompute_desired_traj(self._target, T)
        if self.dcfg.w_tier != 0.0:
            self._regime = self._precompute_regime(self._target, self.dcfg.regime_ramp_s)
            self._tier_rewarded = self._precompute_tier_nn(self._target, self.dcfg.tier_nn_k)

    # ---- the exact backward DP -------------------------------------------------------
    def _dp_backward(self, T: int):
        """Backward pass -> action table ``A_act[N,T]`` (0=continue, else 1+jump-slot); also
        sets ``self._V0`` = cost-to-go from the start layer.

        The terrain-entry pin (``self._boundary_pin`` weighted by ``self._pin_scale``) is folded
        into ``mc(f, t)`` in :meth:`_match_cost_all`, so the terminal layer already carries it."""
        N = self._bvel.shape[0]
        cut = self.dcfg.cut_penalty
        cand, jc, nxt = self._cand, self._jcost, self._next
        has_cont = nxt != -1
        safe_next = np.where(has_cont, nxt, 0)               # safe index; masked to inf below
        action = np.zeros((N, T), dtype=np.int16)            # 1+k <= 1+k_max; int16 headroom
        idxN = np.arange(N)

        V_next = self._match_cost_all(T - 1)                 # V[:, T-1] (incl. the terminal pin)
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

    def _reconstruct(self, action: np.ndarray, T: int, start_frame: int | None = None):
        """Forward-reconstruct the optimal frame path from the action table.

        Starts at ``self._stationary`` (the standing-est frame) unless ``start_frame`` is
        given -- the exit DP seeds this with the frame closest to the terrain EXIT state so
        the recovery path continues from where the played interior left off."""
        cand, nxt = self._cand, self._next
        path = np.zeros(T, dtype=np.int64)
        jumped = [False] * T
        forced = 0
        f = int(self._stationary if start_frame is None else start_frame)
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

    def generate(self, command, T: int, debug: bool = False,
                 boundary_pin: np.ndarray | None = None, start_frame: int | None = None,
                 relax_tail: int = 0, allowed_categories=None):
        """Exact backward-DP match against ``command`` then play the optimal path.

        ``boundary_pin[N]`` (per-frame terrain-entry distance) pins the approach onto the entry.
        With ``relax_tail>0`` the command-tracking ramps DOWN and the pin ramps UP over the last
        ``relax_tail`` frames (a distributed pin = smooth transition); with ``relax_tail=0`` the pin
        applies only at the terminal frame.  ``start_frame`` overrides the reconstruction start
        (exit: begin at the terrain exit).  ``allowed_categories`` (an iterable of source-category
        names, e.g. ``{"flat", "clean"}``) restricts the DB to those categories for this call
        (``None`` = the whole DB); requires the matcher to have been built with ``categories``.
        The chosen frame path is stashed on ``self.frame_path``.
        Returns world ``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,21])``."""
        if self.graph is None:
            raise RuntimeError("DPMatcher.generate called before a candidate graph was attached.")
        self.warnings = []
        self._set_category_filter(allowed_categories)        # restrict the DB to the allowed categories
        self._prepare(command, T)
        self._boundary_pin = None if boundary_pin is None else np.asarray(boundary_pin, dtype=np.float64)
        self._mc_scale = np.ones(T)
        self._pin_scale = np.zeros(T)
        if relax_tail > 0:                                   # transition window: track down, pin up
            tt = np.arange(T)
            self._mc_scale = np.clip((T - 1 - tt) / float(max(1, relax_tail)), 0.0, 1.0)
            self._pin_scale = 1.0 - self._mc_scale
        elif self._boundary_pin is not None:                 # terminal-only pin (no window)
            self._pin_scale[T - 1] = 1.0
        action = self._dp_backward(T)
        start = int(self._stationary if start_frame is None else start_frame)
        if not np.isfinite(self._V0[start]):
            extra = (f" under allowed_categories={sorted(allowed_categories)}"
                     if allowed_categories is not None else "")
            raise RuntimeError(f"DP found no finite-cost path from the start frame{extra}.")
        path, jumped = self._reconstruct(action, T, start_frame=start_frame)
        frame_path = torch.as_tensor(path, dtype=torch.long, device=self.device)
        self.frame_path = frame_path
        root_pos, root_quat, joints, dbg = play_frame_path(frame_path, jumped, self, self.cfg, debug=debug)
        self.dbg = dbg
        return root_pos, root_quat, joints

    # ---- boundary-value entry points (approach into / exit out of a played terrain segment) ----
    def generate_approach(self, command, T: int, entry_descs: np.ndarray,
                          boundary_weight: float = 50.0, debug: bool = False, relax_tail: int = 0,
                          allowed_categories=None):
        """Approach DP: track ``command`` and END at the NEAREST of the allowed terrain ENTRY frames.

        ``entry_descs`` is ``[K, dim]`` (or a single ``[dim]``) -- the allowed entry-frame
        descriptors (see :meth:`VanillaMatcher.entry_descriptors`).  The terminal is pinned to the
        cheapest-to-reach one, so a faster run-up merges into a later (moving) entry frame instead of
        being forced to a full stop at frame 0.  The matched entry-frame INDEX is stored on
        ``self.matched_entry`` -- the caller plays the terrain interaction FROM that frame.
        ``allowed_categories`` (forwarded to :meth:`generate`) restricts the DB to those source
        categories -- e.g. ``{"flat", "clean"}`` so a stair approach can't grab box-climb lead-ins."""
        descs = np.atleast_2d(entry_descs)
        pin = boundary_weight * self.boundary_cost_multi(descs)
        out = self.generate(command, T, debug=debug, boundary_pin=pin, relax_tail=relax_tail,
                            allowed_categories=allowed_categories)
        last = int(self.frame_path[-1])
        d = torch.as_tensor(descs, dtype=self._norm.dtype, device=self.device)
        self.matched_entry = int(torch.cdist(self._norm[last:last + 1], d).argmin())
        return out

    def generate_exit(self, command, T: int, exit_descs: np.ndarray, debug: bool = False,
                      allowed_categories=None):
        """Exit DP: START from the terrain's EXIT WINDOW and track the resumed ``command`` forward.

        Mirror of :meth:`generate_approach`: ``exit_descs`` is ``[K, dim]`` (or ``[dim]``) -- the
        allowed exit-frame descriptors (see :meth:`VanillaMatcher.exit_descriptors`).  We start the
        forward DP from the DB frame nearest ANY of them (the cheapest seam), and store on
        ``self.matched_exit`` WHICH exit frame that is (index into ``exit_descs``) -- the caller
        truncates the played interaction there, so it peels off at whichever phase seams cleanest
        into the resumed locomotion instead of being nailed to the final frame.  ``allowed_categories``
        restricts BOTH the seed frame and the forward DP to those source categories."""
        descs = np.atleast_2d(exit_descs)
        bc = self.boundary_cost_multi(descs)                         # per-DB-frame distance to the exit window
        if allowed_categories is not None and self._frame_cat is not None:
            bc = np.where(np.isin(self._frame_cat, list(allowed_categories)), bc, np.inf)
        start = int(bc.argmin())                                     # nearest ALLOWED DB frame to the exit
        d = torch.as_tensor(descs, dtype=self._norm.dtype, device=self.device)
        self.matched_exit = int(torch.cdist(self._norm[start:start + 1], d).argmin())
        return self.generate(command, T, debug=debug, start_frame=start,
                            allowed_categories=allowed_categories)
