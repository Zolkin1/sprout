# Copyright 2026 Zachary Olkin. All rights reserved.

"""Contact-free motion-matching features for the G1.

A *frame feature* is the pose + velocity of the end-effectors and the root, expressed
so it is invariant to where/which-way the robot is in the world (translation + heading):

* feet & hand **poses relative to the root** (pelvis frame): position + a 6-D rotation
  (first two columns of the rotation matrix -- continuity-safe, no quaternion double
  cover), from batched FK (:class:`motion_gen.fk.FKChain`);
* **root linear + angular velocity**, rotated into the root's yaw-aligned frame (so
  "run forward at 2 m/s" is the same feature regardless of world heading, and matches
  the ``(vx, vy, wz)`` command convention);
* feet & hand **linear + angular velocities relative to the root**, straight from the
  FK geometric Jacobian times the joint velocities (already in the root frame).

No contacts, no foot phase, no hand-labeled anything -- matching works from kinematics
alone (the project's move away from hand-tuned transition machinery).

A *descriptor* stacks the frame feature at several time offsets -- a couple in the past
and a short horizon into the future ``{-0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6}s`` by default --
so a match captures where the motion is going, not just the instantaneous pose.

Root quaternions here are **wxyz** (the IsaacLab / :mod:`motion_gen.math_utils` convention);
ingest code converts to wxyz before calling in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from ..fk import FKChain
from ..math_utils import quat_apply, quat_inv, yaw_quat

# Canonical G1 21-joint order (matches the MJCF joint names + the FM/clip joint set).
G1_JOINT_NAMES: list[str] = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]
)

FOOT_FRAMES: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
HAND_FRAMES: list[str] = ["left_palm_link", "right_palm_link"]

# Default location of the FK MJCF within the repo (transfer/ ships the isaaclab-free model).
DEFAULT_MJCF = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..",
    "transfer", "sim", "robots", "g1", "g1_21j.xml",
)


@dataclass
class FeatureConfig:
    """Feature/descriptor layout + per-group weights.

    Weights scale each feature group *after* per-dimension z-scoring, so they express
    relative importance (hands matter less than feet for locomotion), not unit fixes.
    """

    dt: float = 0.02
    future_offsets_s: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6)
    past_offsets_s: tuple[float, ...] = (-0.2, -0.1)
    include_hands: bool = True

    # Per-group weights (applied after z-score).
    w_foot_pos: float = 1.0
    w_foot_rot: float = 1.0
    w_hand_pos: float = 0.3
    w_hand_rot: float = 0.3
    w_root_lin_vel: float = 1.5
    w_root_ang_vel: float = 1.0
    w_foot_vel: float = 1.0
    w_hand_vel: float = 0.3

    # Filled lazily; the ordered (offset_s) list = past + [0] + future.
    _offsets_s: tuple[float, ...] = field(default=(), repr=False)

    def offsets_s(self) -> list[float]:
        return list(self.past_offsets_s) + [0.0] + list(self.future_offsets_s)

    def offset_frames(self) -> list[int]:
        return [int(round(o / self.dt)) for o in self.offsets_s()]

    def frames(self) -> list[str]:
        """FK end-effector frames in feature order: feet, then hands (if enabled)."""
        return list(FOOT_FRAMES) + (list(HAND_FRAMES) if self.include_hands else [])


def make_fk(cfg: FeatureConfig, mjcf_path: str | None = None,
            device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> FKChain:
    """Build the FK chain for the feature frames (feet [+ hands])."""
    path = mjcf_path or DEFAULT_MJCF
    return FKChain(path, G1_JOINT_NAMES, cfg.frames(), base_link="pelvis_link", dtype=dtype).to(device)


def _rot6d(rot: torch.Tensor) -> torch.Tensor:
    """``[..., 3, 3]`` rotation matrices -> ``[..., 6]`` (first two columns, flattened)."""
    return rot[..., :, :2].reshape(*rot.shape[:-2], 6)


def per_frame_feature_dim(cfg: FeatureConfig) -> int:
    n_ee = len(cfg.frames())              # feet (+ hands)
    n_hand = 2 if cfg.include_hands else 0
    n_foot = 2
    # poses: pos(3) + rot6d(6) per EE ; root vel: 6 ; EE vel: lin(3)+ang(3) per EE
    return n_ee * (3 + 6) + 6 + n_ee * 6


def descriptor_dim(cfg: FeatureConfig) -> int:
    return per_frame_feature_dim(cfg) * len(cfg.offset_frames())


def group_weight_vector(cfg: FeatureConfig, device="cpu") -> torch.Tensor:
    """Per-dimension weights for ONE frame block (same layout as
    :func:`compute_frame_features`), to tile across offsets when weighting a descriptor."""
    parts = [
        torch.full((2 * 3,), cfg.w_foot_pos),   # foot pos
        torch.full((2 * 6,), cfg.w_foot_rot),   # foot rot6d
    ]
    if cfg.include_hands:
        parts += [
            torch.full((2 * 3,), cfg.w_hand_pos),
            torch.full((2 * 6,), cfg.w_hand_rot),
        ]
    parts += [
        torch.full((3,), cfg.w_root_lin_vel),
        torch.full((3,), cfg.w_root_ang_vel),
        torch.full((2 * 3,), cfg.w_foot_vel),   # foot lin vel
        torch.full((2 * 3,), cfg.w_foot_vel),   # foot ang vel
    ]
    if cfg.include_hands:
        parts += [
            torch.full((2 * 3,), cfg.w_hand_vel),
            torch.full((2 * 3,), cfg.w_hand_vel),
        ]
    return torch.cat(parts).to(device)


def compute_frame_features(
    joint_pos: torch.Tensor,      # [M, 21]
    joint_vel: torch.Tensor,      # [M, 21]
    root_quat: torch.Tensor,      # [M, 4] wxyz (world)
    root_lin_vel_w: torch.Tensor,  # [M, 3] world
    root_ang_vel_w: torch.Tensor,  # [M, 3] world
    fk: FKChain,
    cfg: FeatureConfig,
) -> torch.Tensor:
    """Per-frame feature block ``[M, per_frame_feature_dim(cfg)]`` (raw, unweighted).

    Layout (in order): foot pos (2x3), foot rot6d (2x6), [hand pos (2x3), hand rot6d
    (2x6)], root lin vel (3), root ang vel (3), foot lin vel (2x3), foot ang vel (2x3),
    [hand lin vel (2x3), hand ang vel (2x3)].
    """
    m = joint_pos.shape[0]
    pos, rot, jv, jw = fk.pose_and_jacobian(joint_pos)   # [M,K,3],[M,K,3,3],[M,K,3,nj],[M,K,3,nj]
    # EE velocity relative to the (moving) root, expressed in the root frame.
    ee_lin = torch.einsum("mkij,mj->mki", jv, joint_vel)   # [M,K,3]
    ee_ang = torch.einsum("mkij,mj->mki", jw, joint_vel)   # [M,K,3]
    rot6 = _rot6d(rot)                                     # [M,K,6]

    # Root velocity into the yaw-aligned frame (heading invariant; command convention).
    yq_inv = quat_inv(yaw_quat(root_quat))                # [M,4]
    root_lin = quat_apply(yq_inv, root_lin_vel_w)         # [M,3]
    root_ang = quat_apply(yq_inv, root_ang_vel_w)         # [M,3]

    n_ee = pos.shape[1]
    feet = slice(0, 2)
    parts = [
        pos[:, feet].reshape(m, -1),     # foot pos
        rot6[:, feet].reshape(m, -1),    # foot rot6d
    ]
    if cfg.include_hands:
        hands = slice(2, 4)
        parts += [
            pos[:, hands].reshape(m, -1),
            rot6[:, hands].reshape(m, -1),
        ]
    parts += [
        root_lin,
        root_ang,
        ee_lin[:, feet].reshape(m, -1),
        ee_ang[:, feet].reshape(m, -1),
    ]
    if cfg.include_hands:
        parts += [
            ee_lin[:, 2:4].reshape(m, -1),
            ee_ang[:, 2:4].reshape(m, -1),
        ]
    return torch.cat(parts, dim=-1)


def offset_index(n: int, offsets: list[int], wrap: str, device="cpu") -> torch.Tensor:
    """``[n, O]`` frame indices for each base frame + each offset, handling the edges.

    * ``wrap="periodic"`` -- indices wrap modulo ``n`` (correct for a full-periodic
      motion: its feature is periodic, so off-the-edge past/future comes from the other
      end of the cycle).
    * ``wrap="episodic"`` -- indices clamp to ``[0, n-1]`` (repeat the first/last frame
      once the window runs off the ends of a finite recording).
    """
    base = torch.arange(n, device=device).unsqueeze(1)
    idx = base + torch.tensor(offsets, device=device)          # [n, O]
    if wrap == "periodic":
        return idx % n
    if wrap == "episodic":
        return idx.clamp(0, n - 1)
    raise ValueError(f"unknown wrap mode {wrap!r} (expected 'periodic' or 'episodic')")


def assemble_descriptors(
    frame_feats: torch.Tensor,   # [F, per_frame_dim] for ONE contiguous motion
    cfg: FeatureConfig,
    wrap: str = "episodic",
) -> torch.Tensor:
    """Stack per-frame features at the configured time offsets into descriptors.

    Args:
        frame_feats: ``[F, pfd]`` per-frame features for a single contiguous motion,
            frame-ordered in time (dt = ``cfg.dt``).
        wrap: ``"periodic"`` (wrap the offset window modulo F -- for full-periodic
            strides) or ``"episodic"`` (clamp to the ends -- for finite recordings).

    Returns:
        ``descriptors[F, D]`` -- every frame gets a full-window descriptor (the edges
        are handled by ``wrap``), so all frames are usable.
    """
    F, pfd = frame_feats.shape
    offs = cfg.offset_frames()
    idx = offset_index(F, offs, wrap, device=frame_feats.device)          # [F, O]
    gathered = frame_feats[idx.reshape(-1)].reshape(F, len(offs), pfd)     # [F, O, pfd]
    return gathered.reshape(F, len(offs) * pfd)
