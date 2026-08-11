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


def _sensor_list(
    sensor_cfg: SceneEntityCfg | None, sensor_cfgs: list[SceneEntityCfg] | None
) -> list[SceneEntityCfg]:
    """Normalise the two spellings of "which camera(s)" into one list.

    Every depth DR event takes EITHER a single ``sensor_cfg`` or a ``sensor_cfgs`` list of
    sampled views of ONE physical camera. The list form exists so a single draw can cover
    all of them: when the generator samples at its own resolution there are two sensors
    behind one lens, and drawing independently would make them two different physical
    cameras -- ±1 cm of mount and ±2% of focal length apart -- which is exactly the class of
    silent inconsistency the whole camera contract exists to prevent.

    Args:
        sensor_cfg: A single camera, or ``None``.
        sensor_cfgs: Several sampled views of one camera, or ``None``.

    Returns:
        The cameras to apply this event's draw to, first entry leading.

    Raises:
        ValueError: If both or neither is given.
    """
    if (sensor_cfg is None) == (sensor_cfgs is None):
        raise ValueError("pass exactly one of sensor_cfg / sensor_cfgs.")
    return [sensor_cfg] if sensor_cfgs is None else list(sensor_cfgs)


def resample_depth_noise(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg | None = None,
    noise_cfg: DepthNoiseCfg = None,
    sensor_cfgs: list[SceneEntityCfg] | None = None,
):
    """Re-sample a depth camera's per-episode noise state on reset.

    Registers the noiser for each camera on first call, so the matching
    :func:`~robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth.depth_image`
    term picks it up; omit this event term and those sensors' observations stay
    noise-free.

    With ``sensor_cfgs``, the FIRST camera draws and the rest ADOPT its per-episode state,
    so two sampled views of one lens carry the same mis-calibration plane. Per-frame noise
    and dropout stay independent -- they cannot be shared across differing pixel counts, and
    independent realizations are the right model for them anyway.

    Args:
        env: The environment instance.
        env_ids: Environments being reset. ``None`` re-samples all.
        sensor_cfg: A single depth camera (mutually exclusive with ``sensor_cfgs``).
        noise_cfg: Noise parameters; ``offset``/``scale`` must match the observation term.
            ``height``/``width`` are resolved from each camera, so one cfg serves both.
        sensor_cfgs: Sampled views of ONE physical camera, sharing the draw.
    """
    if noise_cfg is None:
        raise ValueError("resample_depth_noise requires noise_cfg.")
    lead, *followers = _sensor_list(sensor_cfg, sensor_cfgs)
    noiser = ensure_depth_noiser(env, lead.name, noise_cfg)
    noiser.reset(env_ids)
    if followers:
        coeffs, sigma = noiser.episode_state(env_ids)
        for cfg in followers:
            ensure_depth_noiser(env, cfg.name, noise_cfg).reset(
                env_ids, plane_coeffs=coeffs, sigma=sigma
            )


def randomize_camera_intrinsics(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg | None = None,
    focal_length_distribution_params: tuple[float, float] | None = None,
    principal_point_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "gaussian"] = "uniform",
    sensor_cfgs: list[SceneEntityCfg] | None = None,
):
    """Perturb pinhole-camera intrinsics on reset.

    Operates on a :class:`MultiMeshRayCasterCamera` (or any
    :class:`RayCasterCamera`) configured with a
    :class:`PinholeCameraPatternCfg`. Snapshots the nominal K matrix on first
    use so successive resets compose against the original intrinsics, not the
    previously-jittered ones.

    With ``sensor_cfgs``, ONE draw is applied to every camera in the list, each against its
    OWN nominal K. That is what makes two sampled views of one lens share a calibration
    error: ``fx``, ``fy``, ``cx``, ``cy`` all scale with resolution, so the absolute matrix
    cannot be copied between them -- but with ``operation="scale"`` the draw is a ratio, and
    a ratio transfers exactly. (With ``"add"`` or ``"abs"`` it does not, so those raise for
    a multi-camera list rather than quietly producing two different lenses.)

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If ``None``, all envs.
        sensor_cfg: A single depth camera (mutually exclusive with ``sensor_cfgs``).
        focal_length_distribution_params: ``(low, high)`` for fx/fy. With
            ``operation="scale"``, ``(0.98, 1.02)`` ≈ ±2 %. ``None`` skips fx/fy.
        principal_point_distribution_params: ``(low, high)`` for cx/cy.
        operation: ``"scale"`` multiplies, ``"add"`` adds, ``"abs"`` replaces.
        distribution: ``"uniform"`` or ``"gaussian"``.
        sensor_cfgs: Sampled views of ONE physical camera, sharing the draw.

    Raises:
        ValueError: On a multi-camera list with an operation that is not ``"scale"``.
    """
    cameras = _sensor_list(sensor_cfg, sensor_cfgs)
    if len(cameras) > 1 and operation != "scale":
        raise ValueError(
            f"operation={operation!r} cannot be shared across {len(cameras)} sampled views "
            "of one camera: their intrinsics differ by resolution, so only a RELATIVE draw "
            "(operation='scale') transfers. Use 'scale', or give each camera its own term."
        )
    lead = env.scene[cameras[0].name]
    if env_ids is None:
        env_ids = torch.arange(lead._view.count, device=lead.device)
    n = len(env_ids)

    # Draw ONCE, then apply to each camera against its own nominal.
    focal_noise = point_noise = None
    if focal_length_distribution_params is not None:
        low, high = focal_length_distribution_params
        focal_noise = _sample_noise(n, 2, low, high, distribution, lead.device)
    if principal_point_distribution_params is not None:
        low, high = principal_point_distribution_params
        point_noise = _sample_noise(n, 2, low, high, distribution, lead.device)

    for cam_cfg in cameras:
        sensor = env.scene[cam_cfg.name]
        # Snapshot the nominal K matrix on first call so subsequent randomisations
        # compose against the original (not the most recently sampled) intrinsics.
        if not hasattr(sensor, "_nominal_intrinsic_matrices"):
            sensor._nominal_intrinsic_matrices = sensor._data.intrinsic_matrices.clone()

        K = sensor._nominal_intrinsic_matrices[env_ids].clone()
        if focal_noise is not None:
            K[:, 0, 0] = _apply_op(operation, K[:, 0, 0], focal_noise[:, 0])  # fx
            K[:, 1, 1] = _apply_op(operation, K[:, 1, 1], focal_noise[:, 1])  # fy
        if point_noise is not None:
            K[:, 0, 2] = _apply_op(operation, K[:, 0, 2], point_noise[:, 0])  # cx
            K[:, 1, 2] = _apply_op(operation, K[:, 1, 2], point_noise[:, 1])  # cy

        # NOTE: ``env_ids`` is keyword-only in
        # ``RayCasterCamera.set_intrinsic_matrices(matrices, focal_length=1.0, env_ids=None)``.
        # Passing it positionally binds it to ``focal_length`` instead.
        sensor.set_intrinsic_matrices(K, env_ids=env_ids)


def randomize_camera_extrinsics(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg | None = None,
    pos_range: tuple[float, float, float] = (0.01, 0.01, 0.01),
    rot_range: tuple[float, float, float] = (0.03, 0.03, 0.03),
    distribution: Literal["uniform", "gaussian"] = "uniform",
    sensor_cfgs: list[SceneEntityCfg] | None = None,
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

    With ``sensor_cfgs``, ONE mount perturbation is applied to every camera in the list. A
    mount pose is resolution-INDEPENDENT, so unlike the intrinsics this transfers directly:
    two sampled views of one lens end up at exactly the same place, which is what "one
    physical camera" has to mean. Their nominal mounts are required to agree first --
    otherwise the list is describing two different cameras and sharing a draw would hide it.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If ``None``, all envs.
        sensor_cfg: A single depth camera (mutually exclusive with ``sensor_cfgs``).
        pos_range: Per-axis translation bound (m) in the MOUNT link's frame. Uniform
            treats it as a symmetric bound; gaussian treats it as one standard deviation.
        rot_range: Per-axis rotation bound (rad) about the CAMERA's own axes, as XYZ
            Euler angles, interpreted as for ``pos_range``.
        distribution: ``"uniform"`` or ``"gaussian"``.
        sensor_cfgs: Sampled views of ONE physical camera, sharing the draw.

    Raises:
        ValueError: If the cameras in ``sensor_cfgs`` have different nominal mounts.
    """
    cameras = _sensor_list(sensor_cfg, sensor_cfgs)
    sensors = [env.scene[c.name] for c in cameras]
    lead = sensors[0]

    if env_ids is None:
        env_ids = torch.arange(lead._view.count, device=lead.device)
    n = len(env_ids)

    for sensor in sensors:
        if not hasattr(sensor, "_nominal_offset_pos"):
            sensor._nominal_offset_pos = sensor._offset_pos.clone()
            sensor._nominal_offset_quat = sensor._offset_quat.clone()
    for cam_cfg, sensor in zip(cameras[1:], sensors[1:]):
        if not (torch.allclose(sensor._nominal_offset_pos, lead._nominal_offset_pos, atol=1e-9)
                and torch.allclose(sensor._nominal_offset_quat, lead._nominal_offset_quat,
                                   atol=1e-9)):
            raise ValueError(
                f"'{cam_cfg.name}' and '{cameras[0].name}' do not describe one physical "
                "camera: their nominal mount poses differ, so they must not share a mount "
                "perturbation. Give each its own event term."
            )

    # Sample in [-1, 1] (or N(0, 1)) then scale per axis, so each axis gets its own bound.
    unit_pos = _sample_noise(n, 3, -1.0, 1.0, distribution, lead.device)
    delta_pos = unit_pos * torch.tensor(pos_range, device=lead.device)
    unit_rot = _sample_noise(n, 3, -1.0, 1.0, distribution, lead.device)
    euler = unit_rot * torch.tensor(rot_range, device=lead.device)
    # (x, y, z, w) in IsaacLab 3.0, matching OffsetCfg.rot and quat_mul.
    delta_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])

    for sensor in sensors:
        sensor._offset_pos[env_ids] = sensor._nominal_offset_pos[env_ids] + delta_pos
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
