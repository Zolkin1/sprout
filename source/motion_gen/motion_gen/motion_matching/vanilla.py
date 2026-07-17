# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vanilla motion matching (the canonical game-industry algorithm), bones-seed only.

No skills, no handoff, no traj-opt -- one continuous loop:

1. **Database**: every frame gets a feature = future **trajectory** (root Δ-position +
   facing at a few future times, in the current root's ground/yaw frame) + current
   **pose** (feet + hand position + velocity rel root, root linear velocity).
2. **Search**: every ``search_interval`` seconds build a query = the *current* pose
   (from the frame being played) + the *desired* future trajectory (integrated from a
   ``(vx, vy, omega)`` command) and jump to the nearest DB frame (weighted-L2 KD-tree).
3. **Playback**: play the matched clip forward, driving the world root from its motion,
   re-anchored (SE(2)+z) at each jump so the root stays continuous.

Drive it with a scripted ``(vx, vy, omega)`` command to synthesize a toy clip.  Output is
built by :class:`~motion_gen.motion_matching.clip_output.ClipOutputBuilder`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..fk import FKChain
from ..math_utils import quat_mul
from . import features as F
from . import ingest_common as IC
from .graph import STEADY, VARIED
from .playback import play_frame_path

_FOOT = ["left_ankle_roll_link", "right_ankle_roll_link"]
_HAND = ["left_palm_link", "right_palm_link"]
_EE = _FOOT + _HAND   # FK order: feet (0,1), hands (2,3)

# Indices WITHIN the pose block (feat[:, traj_dim:]) that are POSITIONS (feet + hands), used
# as the beam graph's velocity-agnostic *postural* candidate-selection metric.  Pose block
# layout: feet_pos(0:6), feet_vel(6:12), hand_pos(12:18), hand_vel(18:24), root_vel(24:27).
POSE_POSTURAL_IDX = list(range(0, 6)) + list(range(12, 18))   # feet_pos(6) + hand_pos(6)


@dataclass
class VanillaConfig:
    dt: float = 0.02
    traj_times_s: tuple[float, ...] = (0.33, 0.66, 1.0)
    search_interval_s: float = 0.1
    # feature-group weights (applied after z-score)
    w_traj_pos: float = 1.0
    w_traj_dir: float = 1.5
    w_foot_pos: float = 0.75
    w_foot_vel: float = 1.0
    w_hand_pos: float = 0.3   # hands down-weighted (locomotion pose is feet-dominated)
    w_hand_vel: float = 0.3
    w_root_vel: float = 1.0
    # Inertia: only jump to the best match if it beats *continuing the current clip* by
    # this ratio (jump iff best_dist < inertia_ratio * continue_dist).  <1 => sticky.
    inertia_ratio: float = 0.8
    # Minimum dwell: after a jump, commit to playing the clip for at least this long before
    # another (non-forced) jump.  Stops the matcher from re-snapping to one globally-best
    # frame under a sustained command (which replays a few frames while the root keeps
    # moving -> foot slide).  A clip is still cut short if it runs out.
    min_dwell_s: float = 0.4
    # Inertialization: on a jump, decay the (old - new) offset in joints / root
    # orientation / root velocity to zero over this window.  Shorter = less planted-foot slide at
    # cut seams (a long decay drags the foot), at the cost of a snappier transition; 0.12 chosen
    # over the old 0.25 after the foot-slide investigation.
    blend_time_s: float = 0.12
    # Clean steady-state preference (STEADY tier).  One-sided: during steady command regions,
    # penalize VARIED (bones) candidates by this much (in KD-tree distance units) so clean/STEADY
    # clips win the greedy query; ramped to 0 around command changes.  w_tier=0 -> vanilla as-is
    # (k=1 query).  tier_k = how many nearest to re-rank when the penalty is on.
    w_tier: float = 0.0
    soft_tier_penalty: float = 2.0
    regime_ramp_s: float = 0.5
    tier_k: int = 16
    # The clean-tier bonus is a NEAREST-NEIGHBOUR by velocity: it rewards the ``tier_nn_k`` clean
    # clips whose steady velocity is closest to the command (k=1 -> the single best-matching clip;
    # the DP then paths between these per-velocity "waypoints").  Velocity ties (clips at the same
    # steady velocity, e.g. `standing` vs a step-in-place `side_step`) are broken toward the clip
    # with the LEAST extraneous motion, so a stop picks `standing`, not a shuffle.
    tier_nn_k: int = 1

    def traj_frames(self) -> list[int]:
        return [int(round(t / self.dt)) for t in self.traj_times_s]

    def dim(self) -> int:
        # traj(pos2+dir2)*len + feet_pos6 + feet_vel6 + hand_pos6 + hand_vel6 + root_vel3
        return 4 * len(self.traj_times_s) + 6 + 6 + 6 + 6 + 3


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) from a wxyz quaternion series ``[F,4]`` -> ``[F]``."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def make_ee_fk(mjcf_path=None, device="cpu", dtype=torch.float32) -> FKChain:
    """FK chain over the pose end-effectors: feet (0,1) then hands (2,3)."""
    return FKChain(mjcf_path or F.DEFAULT_MJCF, F.G1_JOINT_NAMES, _EE,
                   base_link="pelvis_link", dtype=dtype).to(device)


def vanilla_weight_vector(cfg: VanillaConfig, device="cpu") -> torch.Tensor:
    parts = []
    for _ in cfg.traj_times_s:
        parts += [torch.full((2,), cfg.w_traj_pos), torch.full((2,), cfg.w_traj_dir)]
    parts += [torch.full((6,), cfg.w_foot_pos), torch.full((6,), cfg.w_foot_vel),
              torch.full((6,), cfg.w_hand_pos), torch.full((6,), cfg.w_hand_vel),
              torch.full((3,), cfg.w_root_vel)]
    return torch.cat(parts).to(device)


def _periodic_extension(xy: torch.Tensor, yaw: torch.Tensor, K: int):
    """Continue a periodic clip's world planar pose ``(xy[Fr,2], yaw[Fr])`` for ``K`` more
    frames by looping its own per-step local deltas, so the future-trajectory window of the
    clip's tail frames reflects the NEXT period (not a clamped stall).

    The clip is a single closed cycle whose last frame duplicates the first in pose (traj-opt
    closure), so the distinct period is ``P = Fr - 1`` and continuation after frame ``Fr-1``
    replays the deltas of frames ``1..Fr-1`` (== the next period's frames), rotated by the
    accumulated heading -- exactly how playback loops the clip.  Returns
    ``(ext_xy[Fr+K,2], ext_yaw[Fr+K])`` (``ext_yaw`` unwrapped past the seam)."""
    dev = xy.device
    Fr = xy.shape[0]
    P = Fr - 1
    dpos = xy[1:Fr] - xy[0:Fr - 1]                          # [P,2] world per-step delta
    yprev = yaw[0:Fr - 1]
    c, s = torch.cos(yprev), torch.sin(yprev)
    dl_x = c * dpos[:, 0] + s * dpos[:, 1]                  # [P] local (heading-aligned) delta
    dl_y = -s * dpos[:, 0] + c * dpos[:, 1]
    dth = yaw[1:Fr] - yaw[0:Fr - 1]                         # [P] per-step yaw delta
    idx = torch.arange(K, device=dev) % P                  # loop steps 1..P
    dth_seq = dth[idx]                                      # [K]
    cum = torch.cumsum(dth_seq, 0)
    heading = yaw[Fr - 1] + torch.cat([torch.zeros(1, device=dev), cum[:-1]])   # heading BEFORE each step
    ch, sh = torch.cos(heading), torch.sin(heading)
    wl_x = ch * dl_x[idx] - sh * dl_y[idx]                  # rotate local delta to world
    wl_y = sh * dl_x[idx] + ch * dl_y[idx]
    ext_xy_cont = xy[Fr - 1] + torch.stack(
        [torch.cumsum(wl_x, 0), torch.cumsum(wl_y, 0)], dim=-1)                 # [K,2]
    ext_yaw_cont = yaw[Fr - 1] + cum                       # [K]
    return torch.cat([xy, ext_xy_cont], 0), torch.cat([yaw, ext_yaw_cont], 0)


def compute_vanilla_features(root_pos, root_quat, joint_pos, joint_vel, fk_ee, cfg,
                             wrap: str = "episodic"):
    """Per-frame vanilla feature ``[F, dim]`` (raw). Layout: for each traj time
    [Δpos(2), facing(2)], then feet pos(6), feet vel(6), hand pos(6), hand vel(6),
    root lin vel(3).  Feet/hand pos+vel are in the pelvis (FK base) frame; root lin vel
    and the trajectory block are in the current frame's ground/yaw root frame.  ``root_vel``
    stays the LAST 3 dims (the beam graph reads body vx/vy from ``feat[..., -3:-1]``).

    ``wrap="periodic"`` continues the future-trajectory window across the clip's loop seam
    (via :func:`_periodic_extension`) so a single-cycle clean clip's tail frames get their
    real next-period future; ``"episodic"`` clamps to the clip end (edge-repeat), unchanged."""
    Fr = root_pos.shape[0]
    yaw = _yaw(root_quat)                                   # [F]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    # pose via FK (feet 0,1 then hands 2,3)
    pos_p, _, jv, _ = fk_ee.pose_and_jacobian(joint_pos)    # [F,4,3],[F,4,3,21]
    ee_vel = torch.einsum("fkij,fj->fki", jv, joint_vel)    # [F,4,3]
    feet_pos = pos_p[:, 0:2].reshape(Fr, 6)
    hand_pos = pos_p[:, 2:4].reshape(Fr, 6)
    feet_vel = ee_vel[:, 0:2].reshape(Fr, 6)
    hand_vel = ee_vel[:, 2:4].reshape(Fr, 6)
    # root linear velocity (world FD) -> yaw-aligned
    vw = IC.central_diff(root_pos, cfg.dt)                  # [F,3] world
    root_vel = torch.stack([cy * vw[:, 0] + sy * vw[:, 1],
                            -sy * vw[:, 0] + cy * vw[:, 1], vw[:, 2]], dim=-1)  # [F,3]
    # trajectory (future), in the current-frame yaw/ground frame
    periodic = (wrap == "periodic" and Fr >= 3)
    if periodic:
        ext_xy, ext_yaw = _periodic_extension(root_pos[:, :2], yaw, max(cfg.traj_frames()))
    traj = []
    for df in cfg.traj_frames():
        if periodic:
            j = torch.arange(Fr, device=root_pos.device) + df       # into the extended arrays
            dxy = ext_xy[j] - root_pos[:, :2]
            dyaw = ext_yaw[j] - yaw
        else:
            j = torch.clamp(torch.arange(Fr) + df, 0, Fr - 1)
            dxy = root_pos[j, :2] - root_pos[:, :2]                  # [F,2] world
            dyaw = yaw[j] - yaw
        px = cy * dxy[:, 0] + sy * dxy[:, 1]
        py = -sy * dxy[:, 0] + cy * dxy[:, 1]
        traj += [px, py, torch.cos(dyaw), torch.sin(dyaw)]
    traj = torch.stack(traj, dim=-1)                        # [F, 4*len]
    return torch.cat([traj, feet_pos, feet_vel, hand_pos, hand_vel, root_vel], dim=-1)


def desired_trajectory(vx: float, vy: float, omega: float, cfg: VanillaConfig,
                       device="cpu") -> torch.Tensor:
    """Integrate a constant ``(vx, vy, omega)`` command into a desired trajectory ``[4*len]``
    in the current root frame (same layout as the trajectory block of the DB feature)."""
    times = cfg.traj_times_s
    ds = cfg.dt
    n = int(round(max(times) / ds))
    frame_of = {int(round(t / ds)): i for i, t in enumerate(times)}
    px = py = th = 0.0
    out: list = [None] * len(times)
    for s in range(1, n + 1):
        c, sn = math.cos(th), math.sin(th)
        px += (c * vx - sn * vy) * ds
        py += (sn * vx + c * vy) * ds
        th += omega * ds
        if s in frame_of:
            i = frame_of[s]
            out[i] = (px, py, math.cos(omega * times[i]), math.sin(omega * times[i]))
    flat = []
    for i in range(len(times)):
        flat += list(out[i]) if out[i] is not None else [0.0, 0.0, 1.0, 0.0]
    return torch.tensor(flat, device=device)


class VanillaMatcher:
    """Builds the vanilla feature DB (+KD-tree) and runs the matching loop."""

    def __init__(self, sources, cfg: VanillaConfig, mjcf_path=None, device="cpu",
                 build_tree: bool = True):
        self.cfg = cfg
        self.device = torch.device(device)
        self.src_names = [s.name for s in sources]
        fk = make_ee_fk(mjcf_path, device=device)
        feats, self.src_start, self.src_len = [], [], []
        rp, rq, jp = [], [], []
        acc = 0
        for s in sources:
            f = compute_vanilla_features(s.root_pos.to(device), s.root_quat.to(device),
                                         s.joint_pos.to(device), s.joint_vel.to(device), fk, cfg,
                                         wrap=getattr(s, "wrap", "episodic"))
            feats.append(f)
            self.src_start.append(acc); self.src_len.append(s.n); acc += s.n
            rp.append(s.root_pos.to(device)); rq.append(s.root_quat.to(device)); jp.append(s.joint_pos.to(device))
        self.feat_raw = torch.cat(feats)                    # [N, dim]
        self.root_pos = torch.cat(rp); self.root_quat = torch.cat(rq); self.joints = torch.cat(jp)
        self.src_start = torch.tensor(self.src_start, dtype=torch.long, device=device)
        self.src_len = torch.tensor(self.src_len, dtype=torch.long, device=device)
        owner = torch.cat([torch.full((int(self.src_len[i]),), i, dtype=torch.long)
                           for i in range(len(sources))]).to(device)
        self.owner = owner
        self.yaw_all = _yaw(self.root_quat)                 # [N]
        # Per-source tier / periodicity (graph-independent -- computed from the sources so the
        # greedy matcher, which has no candidate graph, can read them too).  Clean steady-state
        # clips are STEADY + periodic; bones are VARIED + episodic.
        src_periodic = torch.tensor([getattr(s, "wrap", "episodic") == "periodic"
                                     for s in sources], dtype=torch.bool, device=device)
        src_steady = torch.tensor([bool(getattr(s, "steady", False))
                                   for s in sources], dtype=torch.bool, device=device)
        self._src_periodic = src_periodic                   # [C]
        self._src_steady = src_steady                        # [C]
        self.is_periodic_frame = src_periodic[owner]        # [N] bool
        self._tier = torch.full((owner.shape[0],), VARIED, dtype=torch.uint8, device=device)
        self._tier[src_steady[owner]] = STEADY              # [N] STEADY where clean
        # per-clip nominal steady velocity ((vx,vy,wz); 0 for non-steady) + per-clip "activity" =
        # mean foot speed (used to break velocity ties toward the stillest clip, e.g. stand vs
        # step-in-place).  The velocity-NN tier bonus (below) reads these.
        src_sv = torch.zeros(len(sources), 3, device=device)
        for i, s in enumerate(sources):
            sv = getattr(s, "steady_vel", None)
            if sv is not None:
                src_sv[i] = torch.as_tensor(sv, dtype=torch.float32, device=device)
        self._src_steady_vel = src_sv                       # [C, 3]
        _tdim = 4 * len(cfg.traj_times_s)
        feet_vel = self.feat_raw[:, _tdim + 6:_tdim + 12]   # pose block feet_vel(6)
        act_frame = feet_vel.norm(dim=1)                    # [N] per-frame foot speed
        self._src_activity = torch.zeros(len(sources), device=device)
        for i in range(len(sources)):
            self._src_activity[i] = act_frame[owner == i].mean()             # [C]
        # normalize + weight, build KD-tree (dim ~39 -> KD-tree works directly, no PCA)
        self.mean = self.feat_raw.mean(0)
        self.std = self.feat_raw.std(0).clamp_min(1e-6)
        self.weight = vanilla_weight_vector(cfg, device=device)
        self._norm = ((self.feat_raw - self.mean) / self.std) * self.weight   # [N, dim]
        self._traj_dim = 4 * len(cfg.traj_times_s)
        # A frame is a search CANDIDATE only if it has a full future-trajectory window
        # ahead of it in its own clip.  Frames in the last ``max_traj`` of a clip have a
        # clamped (fake ~0) future trajectory -- they'd spuriously match "stop" queries and
        # cause an end-of-clip latch -- so they are excluded as candidates (still playable).
        self._max_traj = max(cfg.traj_frames())
        local = torch.arange(self.feat_raw.shape[0], device=device) - self.src_start[self.owner]
        # Periodic (clean) clips have a valid wrapped future at every frame, so all their frames
        # are candidates; episodic clips still exclude their last max_traj frames (clamped future).
        cand = (local + self._max_traj < self.src_len[self.owner]) | self.is_periodic_frame
        self._cand_idx = cand.nonzero(as_tuple=False).flatten()               # [Nc] global indices
        # Full-feature KD-tree (greedy search only).  BeamMatcher passes build_tree=False --
        # it searches the precomputed candidate graph instead, so this build is skipped.
        self._tree = None
        if build_tree:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self._norm[self._cand_idx].cpu().numpy().astype("float64"))
        # standing-est start = candidate with the smallest future displacement
        ci = self._cand_idx
        self._stationary = int(ci[self.feat_raw[ci, :2].pow(2).sum(1).argmin()])

    def _precompute_regime(self, target: np.ndarray, ramp_s: float) -> np.ndarray:
        """Smoothed steady/transition signal in ``[0,1]`` (1 = steady), ramped over ``ramp_s``
        seconds around command changes (a ramp keeps it ~0 throughout; a jump dips it ~ramp_s
        each side).  Shared by the tier cost in the greedy/beam/DP matchers."""
        T = target.shape[0]
        changed = np.zeros(T, dtype=bool)
        if T > 1:
            changed[1:] = np.abs(np.diff(target, axis=0)).sum(1) > 1e-6
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
        x = np.clip(dist / max(1e-6, ramp_s), 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)                       # smoothstep -> steady weight

    def _precompute_tier_nn(self, target: np.ndarray, k: int, vel_eps: float = 0.15):
        """Per-timestep boolean ``[T, C]``: which clips are the velocity-nearest clean clips to the
        command (the ``k`` clean clips whose steady velocity is closest to ``target[t]``).  Velocity
        ties (within ``vel_eps`` of the nearest) are broken toward the clip with the LEAST foot
        motion, so a stop command selects ``standing`` over a step-in-place ``side_step``.  Returns
        ``None`` if there are no clean clips."""
        steady = self._src_steady.cpu().numpy()
        clean = np.nonzero(steady)[0]                        # clean clip indices
        if clean.size == 0:
            return None
        sv = self._src_steady_vel.cpu().numpy()[clean]       # [m,3]
        act = self._src_activity.cpu().numpy()[clean]        # [m]
        C = steady.shape[0]; T = target.shape[0]
        kk = min(max(1, k), clean.size)
        rewarded = np.zeros((T, C), dtype=bool)
        for t in range(T):
            d = np.sqrt(((sv - target[t]) ** 2).sum(1))      # velocity distance
            order = np.lexsort((act, np.round(d / max(1e-6, vel_eps))))   # bucket by vel, tie->stillness
            rewarded[t, clean[order[:kk]]] = True
        return rewarded

    def _local_delta(self, gi: int) -> torch.Tensor:
        """Per-step root position delta in the clip's own frame at frame ``gi``."""
        o = int(self.owner[gi]); a = int(self.src_start[o]); n = int(self.src_len[o])
        if gi - a > 0:
            return self.root_pos[gi] - self.root_pos[gi - 1]
        if n > 1:
            return self.root_pos[gi + 1] - self.root_pos[gi]
        return torch.zeros(3, device=self.device)

    def _wquat(self, g_yaw: float, gi: int) -> torch.Tensor:
        yq = torch.tensor([math.cos(g_yaw / 2), 0.0, 0.0, math.sin(g_yaw / 2)], device=self.device)
        return quat_mul(yq.unsqueeze(0), self.root_quat[gi].unsqueeze(0))[0]

    def _select_path(self, command, T: int):
        """Run the greedy search loop -> ``(frame_path[T] long, jumped[T] list[bool])``.

        Selection reads only DB-frame state (``feat_raw``/``_norm``/the KD-tree) + the
        dwell/inertia bookkeeping -- never the inertialized playback output -- so it is a
        clean input to :func:`~motion_gen.motion_matching.playback.play_frame_path`.
        """
        cfg = self.cfg
        stride = max(1, int(round(cfg.search_interval_s / cfg.dt)))
        min_dwell = max(1, int(round(cfg.min_dwell_s / cfg.dt)))
        dwell = min_dwell                                   # steps since last jump (allow jump at start)
        gi = self._stationary
        o = int(self.owner[gi]); l = int(gi - self.src_start[o])
        frame_path: list[int] = []
        jumped_path: list[bool] = []
        # Clean-tier preference: penalize VARIED candidates during steady command regions (ramped
        # to 0 around changes) so the greedy query prefers clean/STEADY clips.  Off (w_tier=0) ->
        # k=1 query, byte-identical to the no-clean vanilla.
        w_tier = float(cfg.w_tier)
        owner_np_sel = self.owner.cpu().numpy()
        cand_idx_np = self._cand_idx.cpu().numpy()
        src_periodic = self._src_periodic.cpu().numpy()
        regime = None
        tier_rewarded = None
        Kq = 1
        if w_tier != 0.0:
            target = np.array([command(i * cfg.dt) for i in range(T)], dtype=np.float64)
            regime = self._precompute_regime(target, cfg.regime_ramp_s)
            tier_rewarded = self._precompute_tier_nn(target, cfg.tier_nn_k)     # [T,C] or None
            Kq = int(min(max(1, cfg.tier_k), cand_idx_np.shape[0]))
        use_tier = w_tier != 0.0 and tier_rewarded is not None
        for step in range(T):
            jumped = False
            # Play each clip through to its natural end; jump only at a re-query tick or when the
            # clip runs out.  PERIODIC (clean) clips instead LOOP at their end -- continue to frame
            # 1 (frame 0 is the closure-duplicate of the last frame), a wrap-continue (not a cut);
            # the regular stride re-query still fires and can leave the clip when the command shifts.
            clip_ended = l >= int(self.src_len[o])
            if clip_ended and bool(src_periodic[o]) and int(self.src_len[o]) >= 2:
                l = 1                                        # wrap to the clip's second frame
                clip_ended = False                           # a loop, not an end -> no forced jump
            elif clip_ended:
                l = int(self.src_len[o]) - 1
            g_abs = int(self.src_start[o] + l)
            if step > 0 and (step % stride == 0 or clip_ended):
                vx, vy, om = command(step * cfg.dt)
                dtraj = desired_trajectory(vx, vy, om, cfg, device=self.device)
                pose = self.feat_raw[g_abs, self._traj_dim:]
                qn = ((torch.cat([dtraj, pose]) - self.mean) / self.std) * self.weight
                qn_np = qn.cpu().numpy().astype("float64")
                if use_tier:
                    r = float(regime[step])
                    dists, tis = self._tree.query(qn_np, k=Kq)
                    dists = np.atleast_1d(dists); tis = np.atleast_1d(tis)
                    cg = cand_idx_np[tis]                    # global candidate indices
                    # penalty on every candidate EXCEPT the velocity-nearest clean clip(s)
                    pen = w_tier * cfg.soft_tier_penalty * r
                    mism_cg = 1.0 - tier_rewarded[step][owner_np_sel[cg]]
                    scores = dists + pen * mism_cg
                    bi = int(np.argmin(scores))
                    best = int(cg[bi]); best_cost = float(scores[bi])
                    cont = (float((qn - self._norm[g_abs]).norm())
                            + pen * (1.0 - float(tier_rewarded[step][owner_np_sel[g_abs]])))
                else:
                    dist, ti = self._tree.query(qn_np, k=1)
                    best = int(self._cand_idx[int(ti)])      # tree index -> global candidate
                    best_cost = float(dist)
                    cont = float((qn - self._norm[g_abs]).norm())
                if clip_ended or (dwell >= min_dwell and best_cost < cfg.inertia_ratio * cont):
                    jumped = True
                    dwell = 0
                    o = int(self.owner[best]); l = int(best - self.src_start[o])
                    g_abs = int(self.src_start[o] + l)
            frame_path.append(g_abs)
            jumped_path.append(jumped)
            l += 1
            dwell += 1
        return torch.tensor(frame_path, dtype=torch.long, device=self.device), jumped_path

    def generate(self, command, T: int, debug: bool = False):
        """Greedy match against ``command(t_seconds) -> (vx, vy, omega)``, then play the
        selected path.  Returns world ``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,21])``."""
        frame_path, jumped = self._select_path(command, T)
        root_pos, root_quat, joints, dbg = play_frame_path(
            frame_path, jumped, self, self.cfg, debug=debug)
        self.dbg = dbg
        return root_pos, root_quat, joints
