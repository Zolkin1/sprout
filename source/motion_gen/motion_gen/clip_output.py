# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a world-frame ``(root_pos, root_quat, joint_pos)`` trajectory into clip outputs.

Shared by any generator (vanilla motion matching now; the skill/traj-opt path later): the
constraint-frame poses come from FK, ``dy_des`` from finite differences, and ``contact`` is a
foot-ground estimate from the foot collision BOX -- a foot is "in contact" when a bottom corner of
its box reaches the ground (the box + its corners come from ``g1_21j.xml``; this replaces the old
coarse ``ankle_roll_link``-origin height threshold, and is the same geometry the foot-lock uses).
Produces exactly the dense arrays :class:`~motion_gen.clips.clip_io.ClipLibraryWriter` expects.

Two families of Cartesian channels come out of the same FK pass:

* **World frames** (``pelvis_link`` + the clip's end-effectors) -- the FK pose lifted through the
  root pose, with velocities from finite differences.  This is the original output.
* **Root frames** (``<body>@root:*``, see :data:`~motion_gen.joint_names.G1_BODY_TRACK_FRAMES`) --
  the *unlifted* FK pose, i.e. the body in the pelvis frame, with velocities straight from the FK
  geometric Jacobian times the joint velocities.  Using the Jacobian rather than differencing keeps
  ``dy = d/dt y`` exact, which the CLF's double-integrator error model assumes.
"""

from __future__ import annotations

import torch

from .fk import FKChain
from .math_utils import matrix_to_quat, quat_apply, quat_mul
from . import features as F
from .ingest import ingest_common as IC

_FOOT = ["left_ankle_roll_link", "right_ankle_roll_link"]

# G1 foot collision box, per ``ankle_roll_link`` (transfer/sim/robots/g1/g1_21j.xml; MuJoCo box
# ``size`` = half-extents, no rotation).  The 4 BOTTOM corners are the ground-contact geometry
# used for both the ``contact`` signal and the foot-lock (flat -> all 4 down; heel/toe -> 2 down).
_FOOT_BOX_CENTER = (0.03592, 0.0, -0.02517)
_FOOT_BOX_HALF = (0.101555, 0.032735, 0.009255)


def foot_box_corners(device="cpu", dtype=torch.float32) -> torch.Tensor:
    """The 4 bottom corners of the foot collision box in the ``ankle_roll_link`` frame ``[4,3]``."""
    cx, cy, cz = _FOOT_BOX_CENTER
    hx, hy, hz = _FOOT_BOX_HALF
    return torch.tensor([[cx + sx * hx, cy + sy * hy, cz - hz]
                         for sx in (1.0, -1.0) for sy in (1.0, -1.0)], dtype=dtype, device=device)


def root_frame_channels(root_frames) -> tuple[list[str], list[str]]:
    """The ``(pos, vel)`` channel names for a list of root-frame bodies.

    Args:
        root_frames: Body names (no ``@root`` marker).

    Returns:
        ``(pos_names, vel_names)`` -- 7 position channels (xyz + wxyz quaternion) and 6 velocity
        channels (linear xyz + angular xyz) per body, in ``root_frames`` order.
    """
    pos, vel = [], []
    for body in root_frames:
        rf = F.root_frame_name(body)
        pos += [f"{rf}:pos_{a}" for a in "xyz"] + [f"{rf}:ori_{a}" for a in ("w", "x", "y", "z")]
        vel += [f"{rf}:pos_{a}" for a in "xyz"] + [f"{rf}:ori_{a}" for a in "xyz"]
    return pos, vel


def widen_output_names(pos_names, vel_names, root_frames) -> tuple[list[str], list[str]]:
    """Append root-frame channels to an existing channel layout.

    The single definition of where the root-frame block goes, shared by
    :class:`ClipOutputBuilder` (generation) and ``scripts/rsl_rl/backfill_root_frames.py``
    (widening an existing library) so the two layouts cannot drift.  Appending at the END
    means every pre-existing column keeps its index, so a backfilled library's leading
    columns are byte-identical to the original.

    Args:
        pos_names: Existing position channel names.
        vel_names: Existing velocity channel names.
        root_frames: Body names to add root-frame channels for.

    Returns:
        The widened ``(pos_names, vel_names)``.

    Raises:
        ValueError: If the input already declares root-frame channels.
    """
    if any(F.is_root_frame(n) for n in pos_names) or any(F.is_root_frame(n) for n in vel_names):
        raise ValueError(
            "channel layout already declares root-frame channels; refusing to add them twice."
        )
    rp, rv = root_frame_channels(root_frames)
    return list(pos_names) + rp, list(vel_names) + rv


class ClipOutputBuilder:
    """Builds ``y_des``/``dy_des``/``contact`` from a world trajectory in the clip layout."""

    def __init__(self, pos_output_names, vel_output_names, *, mjcf_path=None,
                 contact_frames=_FOOT, ground_height=0.0, contact_z_thresh=0.02,
                 root_frames=None, device="cpu"):
        """Resolve the channel layout and pre-build the FK chain.

        Args:
            pos_output_names: Source library position channel names; the ``joint:*`` entries are
                discarded and rebuilt, the frame channels keep their order.
            vel_output_names: Source library velocity channel names, same treatment.
            mjcf_path: FK model; defaults to :data:`~motion_gen.features.DEFAULT_MJCF`.
            contact_frames: Bodies the foot-box contact signal is computed for.
            ground_height: Ground plane height the contact test is against.
            contact_z_thresh: A foot box bottom-corner within this of the ground counts as loaded.
            root_frames: Bodies to additionally emit in the ROOT (pelvis) frame as
                ``<body>@root:*``.  Defaults to
                :data:`~motion_gen.joint_names.G1_BODY_TRACK_FRAMES`; pass ``()`` to emit none
                (reproducing the pre-root-frame layout).
            device: Torch device for the FK constants and the built arrays.
        """
        # The clip OUTPUT carries the FULL ``F.G1_JOINT_NAMES`` joint set: the joint VALUES come from the
        # matched 29-DoF motion sources (bones / clean / terrain), NOT from the ``--source`` traj-opt
        # library, which may declare a different (e.g. 21-DoF) joint set.  So keep the source's FRAME
        # channels + order and (re)build the ``joint:*`` channels from ``G1_JOINT_NAMES`` -- decoupling the
        # emitted joint set from however many joints the source library happened to solve.
        self.root_frames = list(F.G1_BODY_TRACK_FRAMES if root_frames is None else root_frames)
        _pj = [f"joint:{j}" for j in F.G1_JOINT_NAMES]
        _world_pos = [n for n in pos_output_names if not n.startswith("joint:")] + _pj
        _world_vel = [n for n in vel_output_names if not n.startswith("joint:")] + _pj
        self.pos_names, self.vel_names = widen_output_names(_world_pos, _world_vel, self.root_frames)
        self.P = len(self.pos_names)
        self.V = len(self.vel_names)
        self.device = torch.device(device)
        self.contact_frames = list(contact_frames)
        self.ground_height = float(ground_height)
        self.contact_z_thresh = float(contact_z_thresh)   # a box bottom-corner within this of ground
        self._foot_corners = foot_box_corners(device=device)              # [4,3] ankle-frame

        self.frames = F.output_frame_names(self.pos_names)
        self.world_frames = [f for f in self.frames if not F.is_root_frame(f)]
        self.ee_frames = [f for f in self.world_frames if f != "pelvis_link"]
        # One FK chain per distinct BODY: the end-effectors and the root-frame bodies overlap (the
        # feet and wrists are tracked both ways), and computing a chain twice would be waste.
        self._fk_bodies = list(dict.fromkeys(self.ee_frames + self.root_frames))
        self.fk = FKChain(mjcf_path or F.DEFAULT_MJCF, F.G1_JOINT_NAMES,
                          self._fk_bodies, base_link="pelvis_link").to(device)
        self._fk_idx = {b: i for i, b in enumerate(self._fk_bodies)}

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
        pos_p, rot_p, jv, jw = self.fk.pose_and_jacobian(joints)   # [T,K,3],[T,K,3,3],[T,K,3,nj]x2
        y = torch.zeros(T, self.P, device=self.device)
        wpos = {"pelvis_link": root_pos}
        wquat = {"pelvis_link": root_quat}
        y[:, self._p_pos["pelvis_link"]] = root_pos
        y[:, self._p_ori["pelvis_link"]] = root_quat
        for f in self.ee_frames:
            i = self._fk_idx[f]
            fp = root_pos + quat_apply(root_quat, pos_p[:, i])
            fq = quat_mul(root_quat, matrix_to_quat(rot_p[:, i]))
            y[:, self._p_pos[f]] = fp
            y[:, self._p_ori[f]] = fq
            wpos[f], wquat[f] = fp, fq
        y[:, self._p_joint] = joints

        dy = torch.zeros(T, self.V, device=self.device)
        for f in self.world_frames:
            dy[:, self._v_lin[f]] = IC.central_diff(wpos[f], dt)
            dy[:, self._v_ang[f]] = IC.angular_velocity_from_quat(wquat[f], dt)
        joint_vel = IC.central_diff(joints, dt)
        dy[:, self._v_joint] = joint_vel

        # Root-frame block: the FK pose UNLIFTED (already pelvis-relative) and its exact
        # derivative from the geometric Jacobian, so ``dy = d/dt y`` holds channel-for-channel.
        for body in self.root_frames:
            i, rf = self._fk_idx[body], F.root_frame_name(body)
            y[:, self._p_pos[rf]] = pos_p[:, i]
            y[:, self._p_ori[rf]] = matrix_to_quat(rot_p[:, i])
            dy[:, self._v_lin[rf]] = torch.einsum("taj,tj->ta", jv[:, i], joint_vel)
            dy[:, self._v_ang[rf]] = torch.einsum("taj,tj->ta", jw[:, i], joint_vel)

        # contact: a foot is loaded when a bottom corner of its collision box reaches the ground.
        contact = torch.zeros(T, len(self.contact_frames), dtype=torch.uint8, device=self.device)
        ce = self._foot_corners[None].expand(T, 4, 3)                     # [T,4,3] ankle-frame corners
        for j, f in enumerate(self.contact_frames):
            qe = wquat[f][:, None, :].expand(T, 4, 4)
            corners_w = wpos[f][:, None, :] + quat_apply(qe, ce)          # [T,4,3] world corners
            min_cz = corners_w[..., 2].amin(dim=1)                       # lowest corner per frame
            contact[:, j] = (min_cz < self.ground_height + self.contact_z_thresh).to(torch.uint8)
        return y, dy, contact
