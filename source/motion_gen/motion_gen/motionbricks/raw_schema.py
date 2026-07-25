# Copyright 2026 Zachary Olkin. All rights reserved.

"""Intermediate raw-motion file -- the only contract between Stage 1 and Stage 2.

Stage 1 (Motion Bricks env) writes one :class:`RawMotion` per generated clip; Stage 2
(:mod:`motion_gen`, isaaclab-free) reads it and converts to the clip-library ``.h5``. The format is
deliberately numpy-only (no torch, no Motion Bricks imports) so it loads in either environment.

A raw motion carries exactly what Stage 2 needs to rebuild a world trajectory plus the RL-visible
command:

* ``mujoco_qpos`` ``[T, 36]`` -- Motion Bricks' MuJoCo qpos: ``[root_pos(3), root_quat_wxyz(4),
  joints(29)]`` at ``fps`` Hz (Z-up; the quaternion is already reordered to wxyz by the agent).
* ``joint_names`` -- the 29 joint names in ``qpos[7:36]`` order, so Stage 2 reindexes *by name* to
  ``motion_gen.motion_matching.features.G1_JOINT_NAMES`` (rather than assuming positional identity).
* ``target_vel`` ``[T, 3]`` -- the RL-visible ``(vx, vy, wz)`` command; stored verbatim into the clip
  library (the FM tracker requires it).
* ``target_vel_snapped`` ``[T, 3]`` (optional) -- the command the motion was actually GENERATED at.
  Motion Bricks is driven from a reference stride picked out of a discrete bank, so the achievable
  velocity is snapped to that bank while the RL-visible command stays continuous. Keeping both lets
  Stage 2 choose which the RL sees (``generate_mb_clips.py --rl-vels``) and lets either side report
  the snap error. ``None`` means the two coincide (no snapping happened).
* ``provenance`` -- free-form metadata (backend, model checkpoint hash, seed, pinned-asset ids,
  terrain/segment info for the terrain path). Round-tripped as JSON; never consumed positionally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

_QPOS_DIM = 36
_N_JOINTS = 29


@dataclass
class RawMotion:
    """One Motion-Bricks-generated clip in the intermediate (pre-conversion) representation."""

    mujoco_qpos: np.ndarray            # [T, 36] float
    fps: float                         # source framerate (Motion Bricks is 30 Hz)
    joint_names: list[str]             # 29 names, order of qpos[7:36]
    target_vel: np.ndarray             # [T, 3] float -- RL-visible (vx, vy, wz) command
    kind: str = "flat"                 # "flat" | "terrain"
    provenance: dict = field(default_factory=dict)
    target_vel_snapped: np.ndarray | None = None   # [T, 3] the command the motion was generated at

    def __post_init__(self) -> None:
        """Validate array shapes and joint count."""
        self.mujoco_qpos = np.asarray(self.mujoco_qpos, dtype=np.float32)
        self.target_vel = np.asarray(self.target_vel, dtype=np.float32)
        t = self.mujoco_qpos.shape[0]
        if self.mujoco_qpos.shape != (t, _QPOS_DIM):
            raise ValueError(f"mujoco_qpos must be [T, {_QPOS_DIM}], got {self.mujoco_qpos.shape}")
        if self.target_vel.shape != (t, 3):
            raise ValueError(f"target_vel must be [T, 3] matching qpos T={t}, got {self.target_vel.shape}")
        if self.target_vel_snapped is not None:
            self.target_vel_snapped = np.asarray(self.target_vel_snapped, dtype=np.float32)
            if self.target_vel_snapped.shape != (t, 3):
                raise ValueError(f"target_vel_snapped must be [T, 3] matching qpos T={t}, "
                                 f"got {self.target_vel_snapped.shape}")
        if len(self.joint_names) != _N_JOINTS:
            raise ValueError(f"expected {_N_JOINTS} joint_names, got {len(self.joint_names)}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    @property
    def num_frames(self) -> int:
        """Number of source frames ``T``."""
        return self.mujoco_qpos.shape[0]


def save_raw_motion(path: str, motion: RawMotion) -> None:
    """Write a :class:`RawMotion` to ``path`` as a compressed ``.npz``."""
    meta = {
        "fps": motion.fps,
        "joint_names": list(motion.joint_names),
        "kind": motion.kind,
        "provenance": motion.provenance,
    }
    arrays = {"mujoco_qpos": motion.mujoco_qpos, "target_vel": motion.target_vel}
    if motion.target_vel_snapped is not None:
        arrays["target_vel_snapped"] = motion.target_vel_snapped
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_raw_motion(path: str) -> RawMotion:
    """Load a :class:`RawMotion` previously written by :func:`save_raw_motion`."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        return RawMotion(
            mujoco_qpos=data["mujoco_qpos"],
            fps=float(meta["fps"]),
            joint_names=list(meta["joint_names"]),
            target_vel=data["target_vel"],
            kind=meta.get("kind", "flat"),
            provenance=meta.get("provenance", {}),
            target_vel_snapped=(data["target_vel_snapped"] if "target_vel_snapped" in data else None),
        )
