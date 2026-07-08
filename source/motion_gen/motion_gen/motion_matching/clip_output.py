# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a world-frame ``(root_pos, root_quat, joint_pos)`` trajectory into clip outputs.

Shared by any generator (vanilla motion matching now; the skill/traj-opt path later): the
constraint-frame poses come from FK, ``dy_des`` from finite differences, and ``contact`` is
a best-effort foot-ground estimate.  Produces exactly the dense arrays
:class:`~motion_gen.clips.clip_io.ClipLibraryWriter` expects, in the manager's channel order.
"""

from __future__ import annotations

import torch

from ..fk import FKChain
from ..math_utils import matrix_to_quat, quat_apply, quat_mul
from . import features as F
from . import ingest_common as IC

_FOOT = ["left_ankle_roll_link", "right_ankle_roll_link"]


class ClipOutputBuilder:
    """Builds ``y_des``/``dy_des``/``contact`` from a world trajectory in the clip layout."""

    def __init__(self, pos_output_names, vel_output_names, *, mjcf_path=None,
                 contact_frames=_FOOT, ground_height=0.0, contact_z_thresh=0.06,
                 device="cpu"):
        self.pos_names = list(pos_output_names)
        self.vel_names = list(vel_output_names)
        self.P = len(self.pos_names)
        self.V = len(self.vel_names)
        self.device = torch.device(device)
        self.contact_frames = list(contact_frames)
        self.ground_height = float(ground_height)
        self.contact_z_thresh = float(contact_z_thresh)

        self.frames = []
        for n in self.pos_names:
            if ":" in n and not n.startswith("joint"):
                f = n.split(":")[0]
                if f not in self.frames:
                    self.frames.append(f)
        self.ee_frames = [f for f in self.frames if f != "pelvis_link"]
        self.fk = FKChain(mjcf_path or F.DEFAULT_MJCF, F.G1_JOINT_NAMES,
                          self.ee_frames, base_link="pelvis_link").to(device)

        def pidx(n):
            return self.pos_names.index(n)

        def vidx(n):
            return self.vel_names.index(n)

        self._p_pos = {f: [pidx(f"{f}:pos_{a}") for a in "xyz"] for f in self.frames}
        self._p_ori = {f: [pidx(f"{f}:ori_{a}") for a in ("w", "x", "y", "z")] for f in self.frames}
        self._p_joint = [pidx(f"joint:{j}") for j in F.G1_JOINT_NAMES]
        self._v_lin = {f: [vidx(f"{f}:pos_{a}") for a in "xyz"] for f in self.frames}
        self._v_ang = {f: [vidx(f"{f}:ori_{a}") for a in "xyz"] for f in self.frames}
        self._v_joint = [vidx(f"joint:{j}") for j in F.G1_JOINT_NAMES]

    def build(self, root_pos, root_quat, joints, dt):
        """``(root_pos[T,3], root_quat[T,4] wxyz, joints[T,21])`` -> (y_des[T,P], dy_des[T,V], contact[T,C])."""
        T = root_pos.shape[0]
        pos_p, rot_p, _, _ = self.fk.pose_and_jacobian(joints)          # [T,K,3],[T,K,3,3]
        y = torch.zeros(T, self.P, device=self.device)
        wpos = {"pelvis_link": root_pos}
        wquat = {"pelvis_link": root_quat}
        y[:, self._p_pos["pelvis_link"]] = root_pos
        y[:, self._p_ori["pelvis_link"]] = root_quat
        for i, f in enumerate(self.ee_frames):
            fp = root_pos + quat_apply(root_quat, pos_p[:, i])
            fq = quat_mul(root_quat, matrix_to_quat(rot_p[:, i]))
            y[:, self._p_pos[f]] = fp
            y[:, self._p_ori[f]] = fq
            wpos[f], wquat[f] = fp, fq
        y[:, self._p_joint] = joints

        dy = torch.zeros(T, self.V, device=self.device)
        for f in self.frames:
            dy[:, self._v_lin[f]] = IC.central_diff(wpos[f], dt)
            dy[:, self._v_ang[f]] = IC.angular_velocity_from_quat(wquat[f], dt)
        dy[:, self._v_joint] = IC.central_diff(joints, dt)

        contact = torch.zeros(T, len(self.contact_frames), dtype=torch.uint8, device=self.device)
        for j, f in enumerate(self.contact_frames):
            contact[:, j] = (wpos[f][:, 2] < self.ground_height + self.contact_z_thresh).to(torch.uint8)
        return y, dy, contact
