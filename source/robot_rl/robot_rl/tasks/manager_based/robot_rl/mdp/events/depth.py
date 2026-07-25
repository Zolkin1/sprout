# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth-camera randomization events.

Ported from legged_locomotion_rl, adapted to IsaacLab v3.0.0-beta's
:class:`isaaclab.sensors.ray_caster.RayCasterCamera` API
(``set_intrinsic_matrices`` + per-env ``_data.intrinsic_matrices``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import (
    DepthNoiseCfg,
    ensure_depth_noiser,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def resample_depth_noise(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    noise_cfg: DepthNoiseCfg,
):
    """Re-sample a depth camera's per-episode noise state on reset.

    Registers the noiser for ``sensor_cfg.name`` on first call, so the matching
    :func:`~robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth.depth_image`
    term picks it up; omit this event term and that sensor's observation stays
    noise-free.

    Args:
        env: The environment instance.
        env_ids: Environments being reset. ``None`` re-samples all.
        sensor_cfg: Scene entity pointing at the depth camera.
        noise_cfg: Noise parameters; ``height``/``width``/``offset``/``scale`` must
            match the camera and its observation term.
    """
    noiser = ensure_depth_noiser(env, sensor_cfg.name, noise_cfg)
    noiser.reset(env_ids)


def randomize_camera_intrinsics(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    focal_length_distribution_params: tuple[float, float] | None = None,
    principal_point_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "gaussian"] = "uniform",
):
    """Perturb pinhole-camera intrinsics on reset.

    Operates on a :class:`MultiMeshRayCasterCamera` (or any
    :class:`RayCasterCamera`) configured with a
    :class:`PinholeCameraPatternCfg`. Snapshots the nominal K matrix on first
    use so successive resets compose against the original intrinsics, not the
    previously-jittered ones.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If ``None``, all envs.
        sensor_cfg: Scene entity pointing at the depth camera.
        focal_length_distribution_params: ``(low, high)`` for fx/fy. With
            ``operation="scale"``, ``(0.98, 1.02)`` ≈ ±2 %. ``None`` skips fx/fy.
        principal_point_distribution_params: ``(low, high)`` for cx/cy.
        operation: ``"scale"`` multiplies, ``"add"`` adds, ``"abs"`` replaces.
        distribution: ``"uniform"`` or ``"gaussian"``.
    """
    sensor = env.scene[sensor_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(sensor._view.count, device=sensor.device)
    n = len(env_ids)

    # Snapshot the nominal K matrix on first call so subsequent randomisations
    # compose against the original (not the most recently sampled) intrinsics.
    if not hasattr(sensor, "_nominal_intrinsic_matrices"):
        sensor._nominal_intrinsic_matrices = sensor._data.intrinsic_matrices.clone()

    K = sensor._nominal_intrinsic_matrices[env_ids].clone()

    if focal_length_distribution_params is not None:
        low, high = focal_length_distribution_params
        noise = _sample_noise(n, 2, low, high, distribution, sensor.device)
        K[:, 0, 0] = _apply_op(operation, K[:, 0, 0], noise[:, 0])  # fx
        K[:, 1, 1] = _apply_op(operation, K[:, 1, 1], noise[:, 1])  # fy

    if principal_point_distribution_params is not None:
        low, high = principal_point_distribution_params
        noise = _sample_noise(n, 2, low, high, distribution, sensor.device)
        K[:, 0, 2] = _apply_op(operation, K[:, 0, 2], noise[:, 0])  # cx
        K[:, 1, 2] = _apply_op(operation, K[:, 1, 2], noise[:, 1])  # cy

    # NOTE: ``env_ids`` is keyword-only in
    # ``RayCasterCamera.set_intrinsic_matrices(matrices, focal_length=1.0, env_ids=None)``.
    # Passing it positionally binds it to ``focal_length`` instead.
    sensor.set_intrinsic_matrices(K, env_ids=env_ids)


def randomize_camera_extrinsics(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    pos_range: tuple[float, float, float] = (0.01, 0.01, 0.01),
    rot_range: tuple[float, float, float] = (0.03, 0.03, 0.03),
    distribution: Literal["uniform", "gaussian"] = "uniform",
):
    """Perturb a camera's MOUNT pose on reset -- bracket tolerance and calibration error.

    Usually the dominant sim-to-real gap for a body-mounted camera: the lens is never
    exactly where the CAD says, and the frame it is calibrated in is never exactly the
    link frame.  Restores the equivalent of ``legged_locomotion_rl``'s
    ``SelfIntersectionRayCasterCfg.PoseErrorCfg``, which has no counterpart in
    IsaacLab's :class:`MultiMeshRayCasterCamera`.

    Writes the sensor's per-environment offset buffers directly, since the offset in the
    config is a single nominal pose shared by every env.  The nominal is snapshotted on
    first use so successive resets perturb around it rather than compounding.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If ``None``, all envs.
        sensor_cfg: Scene entity pointing at the depth camera.
        pos_range: Per-axis translation bound (m) in the MOUNT link's frame. Uniform
            treats it as a symmetric bound; gaussian treats it as one standard deviation.
        rot_range: Per-axis rotation bound (rad) about the CAMERA's own axes, as XYZ
            Euler angles, interpreted as for ``pos_range``.
        distribution: ``"uniform"`` or ``"gaussian"``.
    """
    sensor = env.scene[sensor_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(sensor._view.count, device=sensor.device)
    n = len(env_ids)

    if not hasattr(sensor, "_nominal_offset_pos"):
        sensor._nominal_offset_pos = sensor._offset_pos.clone()
        sensor._nominal_offset_quat = sensor._offset_quat.clone()

    # Sample in [-1, 1] (or N(0, 1)) then scale per axis, so each axis gets its own bound.
    unit_pos = _sample_noise(n, 3, -1.0, 1.0, distribution, sensor.device)
    delta_pos = unit_pos * torch.tensor(pos_range, device=sensor.device)
    sensor._offset_pos[env_ids] = sensor._nominal_offset_pos[env_ids] + delta_pos

    unit_rot = _sample_noise(n, 3, -1.0, 1.0, distribution, sensor.device)
    euler = unit_rot * torch.tensor(rot_range, device=sensor.device)
    # (x, y, z, w) in IsaacLab 3.0, matching OffsetCfg.rot and quat_mul.
    delta_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    sensor._offset_quat[env_ids] = math_utils.quat_mul(
        sensor._nominal_offset_quat[env_ids], delta_quat
    )


def _sample_noise(n: int, dim: int, low: float, high: float, distribution: str, device) -> torch.Tensor:
    if distribution == "uniform":
        return torch.empty(n, dim, device=device).uniform_(low, high)
    if distribution == "gaussian":
        mean = (low + high) / 2.0
        std = (high - low) / 2.0
        return torch.normal(mean, std, size=(n, dim), device=device)
    raise ValueError(f"Unsupported distribution: {distribution!r}")


def _apply_op(operation: str, base: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if operation == "scale":
        return base * noise
    if operation == "add":
        return base + noise
    if operation == "abs":
        return noise
    raise ValueError(f"Unsupported operation: {operation!r}")
