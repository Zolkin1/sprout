# Copyright 2026 Zachary Olkin. All rights reserved.

"""Vanilla motion matching (the canonical game-industry algorithm), bones-seed only.

No skills, no handoff, no traj-opt -- one continuous loop:

1. **Database**: every frame gets a feature = future **trajectory** (root Δ-position +
   facing at a few future times, in the current root's ground/yaw frame) + current
   **pose** (feet position + velocity rel root, root linear velocity).
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

import torch

from ..fk import FKChain
from ..math_utils import quat_inv, quat_mul
from . import features as F
from . import ingest_common as IC

_FOOT = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _nlerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Normalized lerp between wxyz quats (hemisphere-fixed); ~slerp for small angles."""
    if float((q0 * q1).sum()) < 0:
        q1 = -q1
    q = (1.0 - t) * q0 + t * q1
    return q / q.norm().clamp_min(1e-9)


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
    w_root_vel: float = 1.0
    # Inertia: only jump to the best match if it beats *continuing the current clip* by
    # this ratio (jump iff best_dist < inertia_ratio * continue_dist).  <1 => sticky.
    inertia_ratio: float = 0.9
    # Inertialization: on a jump, decay the (old - new) offset in joints / root
    # orientation / root velocity to zero over this window, so nothing pops.
    blend_time_s: float = 0.25

    def traj_frames(self) -> list[int]:
        return [int(round(t / self.dt)) for t in self.traj_times_s]

    def dim(self) -> int:
        return 4 * len(self.traj_times_s) + 6 + 6 + 3   # traj(pos2+dir2) + feetpos + feetvel + rootvel


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) from a wxyz quaternion series ``[F,4]`` -> ``[F]``."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def make_foot_fk(mjcf_path=None, device="cpu", dtype=torch.float32) -> FKChain:
    return FKChain(mjcf_path or F.DEFAULT_MJCF, F.G1_JOINT_NAMES, _FOOT,
                   base_link="pelvis_link", dtype=dtype).to(device)


def vanilla_weight_vector(cfg: VanillaConfig, device="cpu") -> torch.Tensor:
    parts = []
    for _ in cfg.traj_times_s:
        parts += [torch.full((2,), cfg.w_traj_pos), torch.full((2,), cfg.w_traj_dir)]
    parts += [torch.full((6,), cfg.w_foot_pos), torch.full((6,), cfg.w_foot_vel),
              torch.full((3,), cfg.w_root_vel)]
    return torch.cat(parts).to(device)


def compute_vanilla_features(root_pos, root_quat, joint_pos, joint_vel, fk_feet, cfg):
    """Per-frame vanilla feature ``[F, dim]`` (raw). Layout: for each traj time
    [Δpos(2), facing(2)], then feet pos(6), feet vel(6), root lin vel(3) -- all in the
    current frame's ground/yaw root frame."""
    Fr = root_pos.shape[0]
    yaw = _yaw(root_quat)                                   # [F]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    # pose via FK (feet)
    pos_p, _, jv, _ = fk_feet.pose_and_jacobian(joint_pos)  # [F,2,3],[F,2,3,21]
    feet_pos = pos_p.reshape(Fr, 6)
    feet_vel = torch.einsum("fkij,fj->fki", jv, joint_vel).reshape(Fr, 6)
    # root linear velocity (world FD) -> yaw-aligned
    vw = IC.central_diff(root_pos, cfg.dt)                  # [F,3] world
    root_vel = torch.stack([cy * vw[:, 0] + sy * vw[:, 1],
                            -sy * vw[:, 0] + cy * vw[:, 1], vw[:, 2]], dim=-1)  # [F,3]
    # trajectory (future), in the current-frame yaw/ground frame
    traj = []
    for df in cfg.traj_frames():
        j = torch.clamp(torch.arange(Fr) + df, 0, Fr - 1)
        dxy = root_pos[j, :2] - root_pos[:, :2]             # [F,2] world
        px = cy * dxy[:, 0] + sy * dxy[:, 1]
        py = -sy * dxy[:, 0] + cy * dxy[:, 1]
        dyaw = yaw[j] - yaw
        traj += [px, py, torch.cos(dyaw), torch.sin(dyaw)]
    traj = torch.stack(traj, dim=-1)                        # [F, 4*len]
    return torch.cat([traj, feet_pos, feet_vel, root_vel], dim=-1)


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

    def __init__(self, sources, cfg: VanillaConfig, mjcf_path=None, device="cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        fk = make_foot_fk(mjcf_path, device=device)
        feats, self.src_start, self.src_len = [], [], []
        rp, rq, jp = [], [], []
        acc = 0
        for s in sources:
            f = compute_vanilla_features(s.root_pos.to(device), s.root_quat.to(device),
                                         s.joint_pos.to(device), s.joint_vel.to(device), fk, cfg)
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
        # normalize + weight, build KD-tree (dim ~27 -> KD-tree works directly, no PCA)
        self.mean = self.feat_raw.mean(0)
        self.std = self.feat_raw.std(0).clamp_min(1e-6)
        self.weight = vanilla_weight_vector(cfg, device=device)
        self._norm = ((self.feat_raw - self.mean) / self.std) * self.weight   # [N, dim]
        from scipy.spatial import cKDTree
        self._tree = cKDTree(self._norm.cpu().numpy().astype("float64"))
        self._traj_dim = 4 * len(cfg.traj_times_s)
        # standing-est start = smallest future displacement
        self._stationary = int(self.feat_raw[:, :2].pow(2).sum(1).argmin())

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

    def generate(self, command, T: int):
        """Run the loop with inertialization. ``command(t_seconds) -> (vx, vy, omega)``.
        Returns world ``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,21])``."""
        cfg = self.cfg
        dev = self.device
        stride = max(1, int(round(cfg.search_interval_s / cfg.dt)))
        blend_n = max(1, int(round(cfg.blend_time_s / cfg.dt)))

        def rz(g, v):   # rotate xy by yaw g, keep z
            c, s = math.cos(g), math.sin(g)
            return torch.stack([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])

        gi = self._stationary
        o = int(self.owner[gi]); l = int(gi - self.src_start[o])
        g_yaw = -float(self.yaw_all[gi])                    # start facing +x
        world_pos = torch.tensor([0.0, 0.0, float(self.root_pos[gi, 2])], device=dev)
        world_yaw = 0.0
        # last emitted output (for capturing the inertialization offset at a jump)
        prev_j = self.joints[gi].clone()
        prev_q = self._wquat(g_yaw, gi)
        prev_v = rz(g_yaw, self._local_delta(gi))
        off_j = torch.zeros(21, device=dev); off_v = torch.zeros(3, device=dev)
        off_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)
        blend_k = blend_n                                   # not blending initially
        Wp, Wq, Wj = [], [], []
        for step in range(T):
            g_abs = int(self.src_start[o] + l)
            clip_ended = l >= int(self.src_len[o])
            if clip_ended:
                l = int(self.src_len[o]) - 1; g_abs = int(self.src_start[o] + l)
            if step > 0 and (step % stride == 0 or clip_ended):
                vx, vy, om = command(step * cfg.dt)
                dtraj = desired_trajectory(vx, vy, om, cfg, device=dev)
                pose = self.feat_raw[g_abs, self._traj_dim:]
                qn = ((torch.cat([dtraj, pose]) - self.mean) / self.std) * self.weight
                dist, best = self._tree.query(qn.cpu().numpy().astype("float64"), k=1)
                best = int(best)
                cont = float((qn - self._norm[g_abs]).norm())
                if clip_ended or float(dist) < cfg.inertia_ratio * cont:
                    o = int(self.owner[best]); l = int(best - self.src_start[o])
                    g_yaw = world_yaw - float(self.yaw_all[best])
                    g_abs = int(self.src_start[o] + l)
                    # capture offset (old output - new clip) and start the decay
                    new_q = self._wquat(g_yaw, g_abs)
                    off_j = prev_j - self.joints[g_abs]
                    off_v = prev_v - rz(g_yaw, self._local_delta(g_abs))
                    off_q = quat_mul(prev_q.unsqueeze(0), quat_inv(new_q.unsqueeze(0)))[0]
                    blend_k = 0
            base_q = self._wquat(g_yaw, g_abs)
            base_v = rz(g_yaw, self._local_delta(g_abs))
            base_j = self.joints[g_abs]
            if blend_k < blend_n:                            # inertialize: decay the offset 1 -> 0
                d = _smoothstep(1.0 - blend_k / blend_n)
                out_j = base_j + off_j * d
                out_v = base_v + off_v * d
                out_q = _nlerp(base_q, quat_mul(off_q.unsqueeze(0), base_q.unsqueeze(0))[0], d)
                blend_k += 1
            else:
                out_j, out_v, out_q = base_j, base_v, base_q
            world_pos = world_pos + out_v
            world_yaw = g_yaw + float(self.yaw_all[g_abs])
            Wp.append(world_pos.clone()); Wq.append(out_q); Wj.append(out_j)
            prev_j, prev_q, prev_v = out_j, out_q, out_v
            l += 1
        return torch.stack(Wp), torch.stack(Wq), torch.stack(Wj)
