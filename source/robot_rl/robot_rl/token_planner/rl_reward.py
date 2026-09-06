# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator-free reward for fine-tuning the AR planner on its own samples.

The signal is the distance between a DECODED sampled plan and the demonstration future, so
it needs no physics and adds no information beyond the clips -- what it changes is how the
planner SAMPLES. Every term is in physical units and is measured against
``decode(encode(future))`` rather than the raw future: the tokenizer reconstruction is the
best plan the vocabulary can express, so scoring against it removes the quantization floor
(0.0348 m base_pos on the shipped tokenizer) from the learning signal instead of asking the
policy to chase an error no choice of tokens can remove.
"""

from __future__ import annotations

import torch

from robot_rl.flow_matching.transforms import geodesic_angle_rot6d
from robot_rl.flow_matching.window import window_block_slices


def _diff2(x: torch.Tensor) -> torch.Tensor:
    """Second finite difference along the node axis, ``(..., H, C) -> (..., H-2, C)``."""
    return x[..., 2:, :] - 2.0 * x[..., 1:-1, :] + x[..., :-2, :]


class PlanReward:
    """Per-plan scalar reward from a decoded window against a decoded target.

    Args:
        num_joints: Joints in the window layout.
        num_extra_frames: End-effector frames in the layout.
        blocks: The checkpoint's reference-block selection.
        w_pos / w_rot / w_joint / w_vel / w_jit / w_jq: Term weights; see the module doc.
    """

    def __init__(self, num_joints: int, num_extra_frames: int, blocks,
                 w_pos: float = 1.0, w_rot: float = 0.5, w_joint: float = 1.0,
                 w_vel: float = 0.2, w_jit: float = 1.0, w_jq: float = 0.5) -> None:
        sl = window_block_slices(num_joints, num_extra_frames, blocks)
        if "base_pose" not in sl:
            raise ValueError("the reward needs a base_pose block; this window has none.")
        bp = sl["base_pose"]
        self.pos = slice(bp.start, bp.start + 3)
        self.rot = slice(bp.start + 3, bp.start + 9)
        self.vel = sl.get("base_vel")
        self.jnt = sl.get("joint_pos")
        self.w = dict(pos=w_pos, rot=w_rot, joint=w_joint, vel=w_vel, jit=w_jit, jq=w_jq)

    def _jitter(self, w: torch.Tensor) -> torch.Tensor:
        """Roughness: ``||diff2(base_pos)|| + w_jq * mean |diff2(joints)|``, per plan."""
        j = _diff2(w[..., self.pos]).norm(dim=-1).mean(dim=-1)
        if self.jnt is not None:
            j = j + self.w["jq"] * _diff2(w[..., self.jnt]).abs().mean(dim=(-1, -2))
        return j

    def __call__(self, plan: torch.Tensor, target: torch.Tensor
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reward and its per-term parts.

        Args:
            plan: ``(..., H, D)`` decoded sampled plans, PHYSICAL units.
            target: ``(..., H, D)`` decoded tokenizer reconstruction of the true future.

        Returns:
            ``(reward (...,), terms)`` with each term the same leading shape.
        """
        t = {}
        t["pos"] = (plan[..., self.pos] - target[..., self.pos]).norm(dim=-1).mean(dim=-1)
        t["rot"] = geodesic_angle_rot6d(plan[..., self.rot], target[..., self.rot]).mean(dim=-1)
        z = t["pos"] * 0.0
        if self.jnt is not None:
            t["joint"] = (plan[..., self.jnt] - target[..., self.jnt]).abs().mean(dim=(-1, -2))
        else:
            t["joint"] = z
        if self.vel is not None:
            # The WHOLE base_vel block (linear + angular). Every other term scores a whole
            # block too, and dropping the angular half would silently leave three channels
            # unconstrained; the cost is that this norm mixes m/s with rad/s, which the
            # startup per-term print makes visible so `w_vel` can absorb it.
            t["vel"] = (plan[..., self.vel] - target[..., self.vel]).norm(dim=-1).mean(dim=-1)
        else:
            t["vel"] = z
        # Only roughness BEYOND the demonstration's own is penalised: real motion is not
        # smooth, and a plain jitter penalty would push the policy off the data toward
        # something flatter than anything it was trained on.
        # Both halves are reported, not just the excess: an excess of 0 is otherwise
        # ambiguous between "the plan is as smooth as the demonstration" and "the
        # demonstration is rough enough to hide anything the policy does".
        t["jitter"] = self._jitter(plan)
        t["jitter_target"] = self._jitter(target)
        t["jitter_excess"] = (t["jitter"] - t["jitter_target"]).clamp_min(0.0)
        r = -(self.w["pos"] * t["pos"] + self.w["rot"] * t["rot"]
              + self.w["joint"] * t["joint"] + self.w["vel"] * t["vel"]
              + self.w["jit"] * t["jitter_excess"])
        return r, t


def group_advantages(r: torch.Tensor, eps: float = 1e-4, leave_one_out: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-relative advantages over the ``G`` axis of ``(B, G)`` rewards.

    No critic: the group's own spread is the baseline, which is what makes this stage cheap
    and is why the reward's overall SCALE is irrelevant -- only relative weighting matters.

    Args:
        r: ``(B, G)`` rewards.
        eps: Floor on the group standard deviation.
        leave_one_out: Use the mean of the OTHER ``G-1`` plans as each plan's baseline.

    Returns:
        ``(A (B, G), degenerate (B,))`` -- ``A`` is zero wherever the group is degenerate.
    """
    g = r.shape[-1]
    if leave_one_out and g > 1:
        base = (r.sum(dim=-1, keepdim=True) - r) / (g - 1)
    else:
        base = r.mean(dim=-1, keepdim=True)
    sd = r.std(dim=-1, keepdim=True)
    degenerate = (sd.squeeze(-1) < eps)
    a = (r - base) / (sd + eps)
    return a.masked_fill(degenerate.unsqueeze(-1), 0.0), degenerate
