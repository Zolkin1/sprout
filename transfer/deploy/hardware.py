# Copyright 2026 Zachary Olkin. All rights reserved.

"""Hardware-facing helpers for the torch-free generator+tracker deploy runtime.

Thin glue for the obelisk controller to use the shared
:class:`deploy.gen_tracker.GenTrackerPolicy` (numpy + onnxruntime) on the robot:

* :func:`load_gen_tracker_from_hf` -- download ``tracker.onnx`` + ``fm.onnx`` from
  HuggingFace (cached) and return a loaded policy.
* :func:`build_hardware_state` -- assemble the ``create_obs`` state dict from the
  estimator's readings (joint encoders + IMU/base estimate + velocity command).

ROS2 control-loop wiring (``controller.py`` ``compute_control``) -- TODO, validate
on-robot:
  * call ``policy.warm_start(build_hardware_state(...))`` on entering low-level
    control, then each 50 Hz cycle ``act = policy.get_action(policy.create_obs(
    build_hardware_state(...)), joint_names_mujoco)``;
  * the FM history needs the base **world-frame** linear velocity -- the current
    controller reads ``v_base[3:6]`` (angular) but not ``v_base[:3]``; confirm the
    estimator's ``v_base[:3]`` linear-velocity frame (world vs body) and pass it as
    ``base_lin_vel_w`` (rotate by the base quat if it is body-frame). The
    asymmetric FM needs **no** end-effector frames from the robot, so no FK.
"""

from __future__ import annotations

import os

import numpy as np

from .gen_tracker import GenTrackerPolicy
from .quat_math_np import quat_apply
from .tracker_session import convert_joint_order


def load_gen_tracker_from_hf(
    hf_repo_id: str,
    hf_policy_folder: str,
    cache_root: str,
    fm_update_every: int = 2,
    num_threads: int = 0,
) -> GenTrackerPolicy:
    """Download ``tracker.onnx`` + ``fm.onnx`` from HF (cached) and load the policy.

    Args:
        hf_repo_id: HuggingFace repo id (``user/repo``).
        hf_policy_folder: Folder within the repo holding the two ONNX files.
        cache_root: Local directory to cache downloads under.
        fm_update_every: Generator re-inference cadence (control steps). Fallback only:
            overridden by 'fm_update_every' in the tracker metadata if present.
        num_threads: onnxruntime intra-op threads (0 = auto).

    Returns:
        A loaded :class:`GenTrackerPolicy`.
    """
    cache_dir = os.path.join(cache_root, hf_repo_id.replace("/", "_"), hf_policy_folder)
    tracker = os.path.join(cache_dir, "tracker.onnx")
    fm = os.path.join(cache_dir, "fm.onnx")
    if not (os.path.exists(tracker) and os.path.exists(fm)):
        from huggingface_hub import hf_hub_download

        base = os.path.join(cache_root, hf_repo_id.replace("/", "_"))
        for fname in (f"{hf_policy_folder}/tracker.onnx", f"{hf_policy_folder}/fm.onnx"):
            hf_hub_download(repo_id=hf_repo_id, filename=fname, local_dir=base)
    policy = GenTrackerPolicy(tracker, fm, fm_update_every=fm_update_every, num_threads=num_threads)
    policy.load()
    return policy


def build_hardware_state(
    policy: GenTrackerPolicy,
    *,
    root_pos: np.ndarray,
    base_quat_xyzw: np.ndarray,
    base_lin_vel_w: np.ndarray,
    base_ang_vel_b: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_names: list[str],
    command: np.ndarray,
    base_ang_vel_w: np.ndarray | None = None,
    frames: np.ndarray | None = None,
) -> dict:
    """Assemble the ``GenTrackerPolicy.create_obs`` state dict from robot readings.

    Args:
        policy: The loaded policy (for the isaac joint order).
        root_pos: Base position in world ``(3,)``.
        base_quat_xyzw: Base orientation quaternion ``(4,)`` xyzw.
        base_lin_vel_w: Base **world-frame** linear velocity ``(3,)`` (link frame).
        base_ang_vel_b: Base **body-frame** angular velocity ``(3,)`` (the
            ``base_ang_vel`` obs term; from the IMU).
        joint_pos/joint_vel: Joint encoder readings ``(nj,)`` in ``joint_names`` order.
        joint_names: The encoder joint order (mapped to the isaac order here).
        command: Velocity command ``(vx, vy, wz)``.
        base_ang_vel_w: Base world-frame angular velocity ``(3,)``; if ``None`` it is
            derived as ``quat_apply(base_quat, base_ang_vel_b)``.
        frames: End-effector frame world states ``(K, 13)`` =
            ``[pos(3), quat(4 xyzw), lin_vel(3), ang_vel(3)]`` in
            ``policy.ctx.frame_names`` order, required when the FM conditions on extra
            frames (compute them from forward kinematics on the measured joints).

    Returns:
        State dict consumed by :meth:`GenTrackerPolicy.create_obs`.
    """
    isaac = policy.get_joint_names()
    q = np.asarray(base_quat_xyzw, dtype=np.float64)
    ang_b = np.asarray(base_ang_vel_b, dtype=np.float64)
    ang_w = np.asarray(base_ang_vel_w, dtype=np.float64) if base_ang_vel_w is not None else quat_apply(q, ang_b)
    state = {
        "root_pos": np.asarray(root_pos, dtype=np.float64),
        "root_quat": q,
        "root_lin_vel_w": np.asarray(base_lin_vel_w, dtype=np.float64),
        "root_ang_vel_w": ang_w,
        "base_ang_vel": ang_b,
        "joint_pos": convert_joint_order(np.asarray(joint_pos, dtype=np.float64), list(joint_names), isaac),
        "joint_vel": convert_joint_order(np.asarray(joint_vel, dtype=np.float64), list(joint_names), isaac),
        "command": np.asarray(command, dtype=np.float64),
    }
    if policy.ctx.num_extra_frames > 0:
        if frames is None:
            raise ValueError(
                f"This FM conditions on {policy.ctx.num_extra_frames} end-effector frames "
                f"({policy.ctx.frame_names}); pass `frames` (K, 13) in that order "
                "(compute them from forward kinematics on the measured joints)."
            )
        fr = np.asarray(frames, dtype=np.float64)
        state.update(frame_pos=fr[:, 0:3], frame_quat=fr[:, 3:7], frame_lin=fr[:, 7:10], frame_ang=fr[:, 10:13])
    return state
