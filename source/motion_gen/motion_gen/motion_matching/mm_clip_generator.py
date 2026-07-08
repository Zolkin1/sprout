# Copyright 2026 Zachary Olkin. All rights reserved.

"""Motion-matching clip generator (flat, v1).

Builds a reference clip as a single **root-driven world trajectory** of
``(root_pos, root_quat, joint_pos)`` -- no foot anchoring anywhere -- then produces the
standard clip outputs (``y_des`` via FK, ``dy_des`` by finite difference, a best-effort
kinematic ``contact``).  Two segment kinds, joined by a single rigid SE(2)+z transform:

* **STEADY** (skill/speed held): the root advances at the stride's natural velocity in
  the current heading; joints come from the nearest traj-opt stride (periodic wrap).  No
  anchoring -- if the commanded speed and the stride disagree the feet slide, and that is
  left visible on purpose.
* **TRANSITION** (skill/speed change): match the current state against the whole motion
  DB (bones-seed + traj-opt) with a velocity bias toward the target command, **play the
  matched motion forward**, and **hand off** into the target skill at the frame where the
  played motion's descriptor is closest to the target-skill frames.  There is **no
  cross-fade fallback**: a poor match is warned (which transition + its cost) and emitted
  anyway, so failures are visible and debuggable.

Output is the same batch dict :class:`~motion_gen.clips.clip_io.ClipLibraryWriter` writes,
so the clip is a drop-in for the existing RL training / play path.
"""

from __future__ import annotations

import torch

from ..math_utils import (
    matrix_to_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    yaw_quat,
)
from . import features as F
from . import ingest_common as IC
from .motion_db import TRAJOPT, MotionDB, MotionSource

_FOOT = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _heading_quat(h: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion for a yaw ``h`` (scalar tensor) about +z."""
    return torch.stack([torch.cos(h / 2), torch.zeros_like(h), torch.zeros_like(h), torch.sin(h / 2)])


class MotionMatchClipGenerator:
    def __init__(
        self,
        manager,
        db: MotionDB,
        sources: list[MotionSource],
        cfg: F.FeatureConfig,
        *,
        dt: float = 0.02,
        mjcf_path: str | None = None,
        device: str | torch.device = "cpu",
        w_vel: float = 3.0,
        handoff_horizon_s: float = 0.6,
        warn_cost_threshold: float = 8.0,
        ground_height: float = 0.0,
        contact_z_thresh: float = 0.06,
    ):
        self.m = manager
        self.db = db
        self.sources = sources
        self.cfg = cfg
        self.dt = float(dt)
        self.device = torch.device(device)
        self.w_vel = float(w_vel)
        self.horizon = max(1, int(round(handoff_horizon_s / dt)))
        self.warn_cost_threshold = float(warn_cost_threshold)
        self.ground_height = float(ground_height)
        self.contact_z_thresh = float(contact_z_thresh)
        self.warnings: list[str] = []

        # ---- output channel layout -----------------------------------------
        self.pos_names = list(manager.pos_output_names)
        self.vel_names = list(manager.vel_output_names)
        self.P = len(self.pos_names)
        self.V = len(self.vel_names)
        self.frames = []
        for n in self.pos_names:
            if ":" in n and not n.startswith("joint"):
                f = n.split(":")[0]
                if f not in self.frames:
                    self.frames.append(f)
        self.ydes_frames = [f for f in self.frames if f != "pelvis_link"]
        # FK chain for the non-pelvis constraint frames (y_des production)
        self.ydes_fk = F.FKChain(mjcf_path or F.DEFAULT_MJCF, F.G1_JOINT_NAMES,
                                 self.ydes_frames, base_link="pelvis_link").to(device)
        # feature FK (feet + palms) for query descriptors
        self.feat_fk = F.make_fk(cfg, mjcf_path, device=device)

        def pidx(name):
            return self.pos_names.index(name)

        def vidx(name):
            return self.vel_names.index(name)

        self._p_pos = {f: [pidx(f"{f}:pos_{a}") for a in "xyz"] for f in self.frames}
        self._p_ori = {f: [pidx(f"{f}:ori_{a}") for a in ("w", "x", "y", "z")] for f in self.frames}
        self._p_joint = [pidx(f"joint:{j}") for j in F.G1_JOINT_NAMES]
        self._v_lin = {f: [vidx(f"{f}:pos_{a}") for a in "xyz"] for f in self.frames}
        self._v_ang = {f: [vidx(f"{f}:ori_{a}") for a in "xyz"] for f in self.frames}
        self._v_joint = [vidx(f"joint:{j}") for j in F.G1_JOINT_NAMES]

        # ---- per-source / per-skill lookups --------------------------------
        self.src_start = []
        acc = 0
        for s in sources:
            self.src_start.append(acc)
            acc += s.n
        self.stride_to_src = {}
        self.trajopt_by_skill: dict[int, list[tuple[int, float]]] = {}
        cond = manager._global_conditioning
        for si, s in enumerate(sources):
            if s.source_kind == TRAJOPT and s.stride_idx is not None:
                gi = int(s.stride_idx[0])
                self.stride_to_src[gi] = si
                vx = float(cond[gi, 0])
                self.trajopt_by_skill.setdefault(s.skill_id, []).append((si, vx))
        self.skill_of_name = manager.skill_name_to_idx

    # ---- helpers -----------------------------------------------------------
    def _nearest_stride(self, skill_name: str, vx: float) -> tuple[int, float]:
        """(source_idx, natural_vx) of the traj-opt stride for ``skill`` nearest ``vx``."""
        sk = self.skill_of_name[skill_name]
        cands = self.trajopt_by_skill.get(sk, [])
        if not cands:
            raise ValueError(f"no traj-opt strides for skill {skill_name!r}")
        si, nvx = min(cands, key=lambda c: abs(c[1] - vx))
        return si, nvx

    def _skill_mask(self, skill_name: str) -> torch.Tensor:
        return self.db.skill_id == self.skill_of_name[skill_name]

    def _align(self, root_s, quat_s, tgt_pos, tgt_quat):
        """Rigid SE(2)+z that maps source root pose (root_s, quat_s) to (tgt_pos, tgt_quat)."""
        gq = quat_mul(yaw_quat(tgt_quat.unsqueeze(0)), quat_inv(yaw_quat(quat_s.unsqueeze(0))))[0]  # pure yaw
        t = tgt_pos - quat_apply(gq.unsqueeze(0), root_s.unsqueeze(0))[0]
        return gq, t

    def _query_descriptor(self, Wp, Wq, Wj) -> torch.Tensor:
        """Raw descriptor of the CURRENT (last) world frame from the recent buffer.

        Future offsets clamp to 'now' (episodic) -- the future is unknown at a transition;
        the desired direction is supplied separately as the velocity bias.
        """
        k = max(16, max(-min(self.cfg.offset_frames()), 1) + 2)
        jp = torch.stack(Wj[-k:]); rp = torch.stack(Wp[-k:]); rq = torch.stack(Wq[-k:])
        jv = IC.central_diff(jp, self.dt)
        rl = IC.central_diff(rp, self.dt)
        ra = IC.angular_velocity_from_quat(rq, self.dt)
        feat = F.compute_frame_features(jp, jv, rq, rl, ra, self.feat_fk, self.cfg)
        desc = F.assemble_descriptors(feat, self.cfg, wrap="episodic")  # [k, D]
        return desc[-1:]                                               # current frame

    # ---- generation --------------------------------------------------------
    def generate(self, schedule: list[dict], clip_length_s: float) -> dict:
        """Build one clip from ``schedule`` (list of ``{skill, vx, wz, duration_s}``)."""
        dt = self.dt
        Wp: list[torch.Tensor] = []
        Wq: list[torch.Tensor] = []
        Wj: list[torch.Tensor] = []
        skill_tr: list[int] = []
        stride_tr: list[int] = []
        tvel_tr: list[list[float]] = []

        def append(wp, wq, joints, skill_id, stride, tvel):
            Wp.append(wp); Wq.append(wq); Wj.append(joints)
            skill_tr.append(skill_id); stride_tr.append(stride); tvel_tr.append(tvel)

        # State: current heading + last world pose + steady stride cursor.
        heading = torch.tensor(0.0, device=self.device)

        def play_steady(src_idx, start_frame, n_steps, skill_id, stride_gi, vx, wz):
            nonlocal heading
            s = self.sources[src_idx]
            Fi = s.n
            frame = start_frame % Fi
            for _ in range(n_steps):
                heading = heading + wz * dt
                if Wp:
                    prev = Wp[-1]
                else:
                    prev = torch.tensor([0.0, 0.0, float(s.root_pos[frame, 2])], device=self.device)
                fwd = torch.stack([torch.cos(heading), torch.sin(heading), torch.zeros_like(heading)])
                wp = prev + vx * dt * fwd
                wp = torch.stack([wp[0], wp[1], s.root_pos[frame, 2]])
                tilt = quat_mul(quat_inv(yaw_quat(s.root_quat[frame].unsqueeze(0))),
                                s.root_quat[frame].unsqueeze(0))[0]
                wq = quat_mul(_heading_quat(heading).unsqueeze(0), tilt.unsqueeze(0))[0]
                append(wp, wq, s.joint_pos[frame], skill_id, stride_gi, [float(vx), 0.0, float(wz)])
                frame = (frame + 1) % Fi

        def play_transition(seg):
            """Match -> play matched source -> hand off. Returns (target_src, target_frame)."""
            nonlocal heading
            q = self._query_descriptor(Wp, Wq, Wj)
            vel_bias = torch.tensor([[seg["vx"], 0.0, seg["wz"]]], device=self.device)
            gi, cost, _ = self.db.nn_search(q, vel_bias=vel_bias, w_vel=self.w_vel)
            gi = int(gi[0])
            msi, mf = int(self.db.owner[gi]), int(self.db.local[gi])
            ms = self.sources[msi]
            tgt_mask = self._skill_mask(seg["skill"])

            # Pass 1: over the horizon, score each played frame against the target skill.
            best_k, best_cost, best_tgi = 0, float("inf"), -1
            for k in range(self.horizon):
                fr = (mf + k) % ms.n if ms.wrap == "periodic" else min(mf + k, ms.n - 1)
                g_idx = self.src_start[msi] + fr
                tgi, tcost, _ = self.db.nn_search(self.db.desc_raw[g_idx:g_idx + 1],
                                                  candidate_mask=tgt_mask)
                if float(tcost[0]) < best_cost:
                    best_cost, best_k, best_tgi = float(tcost[0]), k, int(tgi[0])
            if best_cost > self.warn_cost_threshold:
                self.warnings.append(
                    f"transition -> {seg['skill']} (vx={seg['vx']}): weak handoff match "
                    f"cost={best_cost:.2f} > {self.warn_cost_threshold} (emitted anyway)")

            # Pass 2: play the matched source up to the handoff frame, yaw-aligned to the
            # cursor.  Position is accumulated incrementally so a periodic wrap (frame
            # F-1 -> 0, where the anchor-relative root resets ~1 step back) stays
            # continuous -- the wrap step is replaced by the source's mean forward step.
            gq, _ = self._align(ms.root_pos[mf], ms.root_quat[mf], Wp[-1], Wq[-1])
            mean_step = (ms.root_pos[-1] - ms.root_pos[0]) / max(1, ms.n - 1)
            world_pos = Wp[-1]
            prev_fr = mf
            for k in range(best_k + 1):
                fr = (mf + k) % ms.n if ms.wrap == "periodic" else min(mf + k, ms.n - 1)
                if k > 0:
                    src_step = mean_step if (ms.wrap == "periodic" and fr < prev_fr) \
                        else ms.root_pos[fr] - ms.root_pos[prev_fr]
                    world_pos = world_pos + quat_apply(gq.unsqueeze(0), src_step.unsqueeze(0))[0]
                wq = quat_mul(gq.unsqueeze(0), ms.root_quat[fr].unsqueeze(0))[0]
                append(world_pos, wq, ms.joint_pos[fr], -1, -1, [float(seg["vx"]), 0.0, float(seg["wz"])])
                prev_fr = fr
            heading = torch.atan2(2 * (Wq[-1][0] * Wq[-1][3] + Wq[-1][1] * Wq[-1][2]),
                                  1 - 2 * (Wq[-1][2] ** 2 + Wq[-1][3] ** 2))
            # Hand off into the target stride at the matched frame's phase.
            tstride = int(self.db.stride_idx[best_tgi])
            tphase = float(self.db.phase[best_tgi])
            tsrc = self.stride_to_src.get(tstride)
            if tsrc is None:                       # matched a bones frame; fall to nearest stride
                tsrc, _ = self._nearest_stride(seg["skill"], seg["vx"])
                tphase = 0.0
            tfi = self.sources[tsrc].n
            return tsrc, int(round(tphase * tfi)) % tfi

        # --- segment 0: steady from phase 0 ---
        seg0 = schedule[0]
        si0, vx0 = self._nearest_stride(seg0["skill"], seg0["vx"])
        stride0 = int(self.sources[si0].stride_idx[0])
        n0 = max(1, int(round(seg0["duration_s"] / dt)))
        play_steady(si0, 0, n0, self.skill_of_name[seg0["skill"]], stride0, vx0, seg0["wz"])

        # --- remaining segments: transition then steady ---
        for seg in schedule[1:]:
            tsrc, tframe = play_transition(seg)
            _, vxn = self._nearest_stride(seg["skill"], seg["vx"])
            stride_gi = int(self.sources[tsrc].stride_idx[0])
            n = max(1, int(round(seg["duration_s"] / dt)))
            play_steady(tsrc, tframe, n, self.skill_of_name[seg["skill"]], stride_gi, vxn, seg["wz"])

        # --- assemble to exactly T steps ---
        T = max(1, int(round(clip_length_s / dt)))
        while len(Wp) < T:
            Wp.append(Wp[-1]); Wq.append(Wq[-1]); Wj.append(Wj[-1])
            skill_tr.append(skill_tr[-1]); stride_tr.append(stride_tr[-1]); tvel_tr.append(tvel_tr[-1])
        Wp, Wq, Wj = Wp[:T], Wq[:T], Wj[:T]
        skill_tr, stride_tr, tvel_tr = skill_tr[:T], stride_tr[:T], tvel_tr[:T]

        root_pos = torch.stack(Wp); root_quat = torch.stack(Wq); joints = torch.stack(Wj)
        y_des, dy_des, contact = self._world_to_outputs(root_pos, root_quat, joints)
        return {
            "y_des": y_des.unsqueeze(0),
            "dy_des": dy_des.unsqueeze(0),
            "contact": contact.unsqueeze(0),
            "skill_id": torch.tensor(skill_tr, dtype=torch.uint8, device=self.device).unsqueeze(0),
            "stride_idx": torch.tensor(stride_tr, dtype=torch.int32, device=self.device).unsqueeze(0),
            "target_vel": torch.tensor(tvel_tr, dtype=torch.float32, device=self.device).unsqueeze(0),
            "terrain_id": torch.tensor([-1], dtype=torch.int32, device=self.device),
            "terrain_type": "flat",
        }

    def _world_to_outputs(self, root_pos, root_quat, joints):
        T = root_pos.shape[0]
        pos_p, rot_p, _, _ = self.ydes_fk.pose_and_jacobian(joints)   # [T,Kf,3],[T,Kf,3,3]
        y = torch.zeros(T, self.P, device=self.device)
        # pelvis = root
        y[:, self._p_pos["pelvis_link"]] = root_pos
        y[:, self._p_ori["pelvis_link"]] = root_quat
        world_frame_pos = {"pelvis_link": root_pos}
        world_frame_quat = {"pelvis_link": root_quat}
        for i, f in enumerate(self.ydes_frames):
            fp = root_pos + quat_apply(root_quat, pos_p[:, i])
            fq = quat_mul(root_quat, matrix_to_quat(rot_p[:, i]))
            y[:, self._p_pos[f]] = fp
            y[:, self._p_ori[f]] = fq
            world_frame_pos[f] = fp
            world_frame_quat[f] = fq
        y[:, self._p_joint] = joints

        dy = torch.zeros(T, self.V, device=self.device)
        for f in self.frames:
            dy[:, self._v_lin[f]] = IC.central_diff(world_frame_pos[f], self.dt)
            dy[:, self._v_ang[f]] = IC.angular_velocity_from_quat(world_frame_quat[f], self.dt)
        dy[:, self._v_joint] = IC.central_diff(joints, self.dt)

        # Best-effort kinematic contact from foot-ground proximity (NOT used for matching).
        contact = torch.zeros(T, len(_FOOT), dtype=torch.uint8, device=self.device)
        for j, f in enumerate(_FOOT):
            fz = world_frame_pos[f][:, 2]
            contact[:, j] = (fz < self.ground_height + self.contact_z_thresh).to(torch.uint8)
        return y, dy, contact
