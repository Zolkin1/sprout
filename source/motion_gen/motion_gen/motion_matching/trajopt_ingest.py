# Copyright 2026 Zachary Olkin. All rights reserved.

"""Ingest traj-opt library strides into motion-matching sources.

Each stride is densely sampled at ``dt`` over its period via the
:class:`~motion_gen.multiskill_manager.MultiSkillManager` Bezier evaluation.  Every
frame carries its ``(stride_idx, phase)`` so the clip generator can hand off into the
steady traj-opt path at exactly that stride/phase, and its ``skill_id`` so handoff /
speed-change candidate masks can restrict to the target skill.

The pelvis pose comes back anchor-relative (yaw-aligned stance-foot frame); that is a
self-consistent world-ish frame for a single stride, and the descriptor is
translation+heading invariant anyway, so absolute placement does not matter here.
"""

from __future__ import annotations

import torch

from ..multiskill_manager import _TRAJ_TYPE_TO_INT
from ..trajectory_manager import TrajectoryType
from . import features as F
from . import ingest_common as IC
from .motion_db import TRAJOPT, MotionSource

_EPISODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.EPISODIC]


def ingest_trajopt(
    manager,
    dt: float = 0.02,
    skills: list[str] | None = None,
    device: str | torch.device = "cpu",
) -> list[MotionSource]:
    """Build one :class:`MotionSource` per library stride (optionally skill-filtered)."""
    pn = manager.pos_output_names
    # Tolerate a source library that solved FEWER joints than the full G1 set: joints absent from the
    # library (e.g. waist roll/pitch + wrists in a 21-DoF traj-opt library) are held at 0 here, so a
    # legacy library still ingests as a 29-DoF motion source (0 for the unsolved DOFs).
    _pn_idx = {n: i for i, n in enumerate(pn)}
    _jcols = [_pn_idx.get(f"joint:{jn}", -1) for jn in F.G1_JOINT_NAMES]
    jidx = torch.tensor([c if c >= 0 else 0 for c in _jcols], device=device)
    _jmiss = torch.tensor([c < 0 for c in _jcols], device=device, dtype=torch.bool)
    ppos, pori = IC.frame_pose_index(pn, "pelvis_link")
    ppos = torch.tensor(ppos, device=device)
    pori = torch.tensor(pori, device=device)

    total_time = manager.data["total_time"].to(device)
    traj_type = manager.data["traj_type"].to(device)    # [n_traj]
    n_traj = int(total_time.shape[0])
    traj_skill = manager._traj_skill_idx.to(device)     # [n_traj]
    idx_to_skill = {v: k for k, v in manager.skill_name_to_idx.items()}
    allow = None if skills is None else {manager.skill_name_to_idx[s] for s in skills}

    sources: list[MotionSource] = []
    for gi in range(n_traj):
        skill_i = int(traj_skill[gi])
        if allow is not None and skill_i not in allow:
            continue
        T = float(total_time[gi])
        Fi = max(4, int(round(T / dt)))
        phase = torch.arange(Fi, device=device, dtype=torch.float32) / Fi   # [0,1)
        manager.set_trajectory_indices(torch.full((Fi,), gi, dtype=torch.long, device=device))
        ypos, _ = manager.get_output(phase)                                 # [Fi, P]
        joint_pos = ypos[:, jidx]
        if bool(_jmiss.any()):
            joint_pos = joint_pos.clone()
            joint_pos[:, _jmiss] = 0.0                  # unsolved joints (not in the source library) -> 0

        root_pos = ypos[:, ppos]
        root_quat = ypos[:, pori]                                           # wxyz
        root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        joint_vel = IC.central_diff(joint_pos, dt)
        root_lin_vel = IC.central_diff(root_pos, dt)
        root_ang_vel = IC.angular_velocity_from_quat(root_quat, dt)
        sources.append(MotionSource(
            joint_pos=joint_pos, joint_vel=joint_vel, root_pos=root_pos, root_quat=root_quat,
            root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
            source_kind=TRAJOPT, skill_id=skill_i,
            stride_idx=torch.full((Fi,), gi, dtype=torch.long, device=device),
            phase=phase,
            wrap=("episodic" if int(traj_type[gi]) == _EPISODIC_INT else "periodic"),
            name=f"trajopt:{idx_to_skill.get(skill_i, skill_i)}:{gi}",
        ))
    return sources
