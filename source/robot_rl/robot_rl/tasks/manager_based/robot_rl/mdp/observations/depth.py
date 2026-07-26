# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth observation wrappers around ``MultiMeshRayCasterCamera``.

These consume the camera's native ``distance_to_image_plane`` data type -- already
projected onto the camera z-axis -- and apply the ``(d - offset) * scale``
normalisation used by the legged_locomotion_rl training pipeline.

Noise is applied HERE rather than through the observation term's ``noise`` field;
see :mod:`robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise` for why the
:class:`~isaaclab.utils.noise.noise_model.NoiseModel` route is unusable.  The
per-episode noise state is re-sampled by the ``resample_depth_noise`` event.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import get_depth_noiser

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def normalize_depth(
    d: torch.Tensor,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
) -> torch.Tensor:
    """Clip invalid depth then apply the ``(d - offset) * scale`` transform.

    Pixels nearer than ``min_dist`` become ``min_clip_val``; pixels beyond
    ``max_dist``, plus NaN and inf (a ray that hit nothing), become
    ``max_clip_val``.  Both substitutions happen BEFORE the affine transform, so
    the clip values are expressed in metres like the rest of the arguments.

    Args:
        d: Raw depth in metres, any shape.
        offset: Subtracted before scaling.
        scale: Multiplies the offset depth.
        min_dist: Below this, a reading is treated as invalid.
        max_dist: Above this, a reading is treated as invalid.
        min_clip_val: Pre-normalisation replacement for too-near pixels.
        max_clip_val: Pre-normalisation replacement for too-far / missed pixels.

    Returns:
        The normalised depth, same shape as ``d``.
    """
    d = d.clone()
    d[d < min_dist] = min_clip_val
    bad = (d > max_dist) | torch.isnan(d) | torch.isinf(d)
    d[bad] = max_clip_val
    return (d - offset) * scale


def depth_image(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
) -> torch.Tensor:
    """Flattened, normalised, optionally-corrupted depth scan of shape ``(B, H*W)``.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity pointing at the depth camera.
        offset: See :func:`normalize_depth`.
        scale: See :func:`normalize_depth`.
        min_dist: See :func:`normalize_depth`.
        max_dist: See :func:`normalize_depth`.
        min_clip_val: See :func:`normalize_depth`.
        max_clip_val: See :func:`normalize_depth`.

    Returns:
        The flattened depth observation, shape ``(num_envs, H*W)``.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    d = sensor.data.output["distance_to_image_plane"].squeeze(-1)  # (B, H, W)
    d = normalize_depth(
        d,
        offset=offset,
        scale=scale,
        min_dist=min_dist,
        max_dist=max_dist,
        min_clip_val=min_clip_val,
        max_clip_val=max_clip_val,
    ).reshape(env.num_envs, -1)

    noiser = get_depth_noiser(env, sensor_cfg.name)
    if noiser is not None:
        d = noiser(d)
    return d


def find_depth_obs_terms(env: "ManagerBasedRLEnv") -> list[tuple[str, dict]]:
    """Discover every depth observation term, as ``(sensor_name, params)`` in obs order.

    The FM's depth conditioning must be normalised EXACTLY as the recorded training data
    was, and the observation terms are the single source of truth for that (each camera
    has its own ``min_dist`` / ``max_dist``).  Discovering them here -- rather than
    duplicating the constants -- means the recorder, the FM dataset and the deployed
    tracker agree by construction.

    Args:
        env: The environment instance.

    Returns:
        ``[(sensor_name, params), ...]``, deduplicated, in observation-group order.
    """
    manager = env.observation_manager
    found: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for group in manager.active_terms:
        names = manager.active_terms.get(group, [])
        cfgs = manager._group_obs_term_cfgs.get(group, [])
        for name, cfg in zip(names, cfgs):
            if getattr(cfg.func, "__name__", "") not in ("depth_image", "depth_image_4d"):
                continue
            params = dict(cfg.params or {})
            sensor_cfg = params.pop("sensor_cfg", None)
            if sensor_cfg is None or sensor_cfg.name in seen:
                continue
            # height/width are reshape-only args for the 4d variant; drop them so the
            # stacked reading is always flat per camera.
            params.pop("height", None)
            params.pop("width", None)
            seen.add(sensor_cfg.name)
            found.append((sensor_cfg.name, params))
    return found


def depth_obs_stack(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """All depth cameras' observations stacked on a token axis, ``(B, T, H*W)``.

    One row per camera, each normalised with that camera's own observation-term
    parameters.  This is the array recorded for the flow-matching generator and the one
    the deployed tracker rebuilds, so the two cannot drift apart.

    Args:
        env: The environment instance.

    Returns:
        Stacked depth, shape ``(num_envs, num_cameras, H*W)``.

    Raises:
        RuntimeError: If the environment has no depth observation terms.
    """
    terms = find_depth_obs_terms(env)
    if not terms:
        raise RuntimeError(
            "no depth observation terms found; run a depth task (e.g. "
            "G1-teacher-clf-terrain-depth) so mdp.depth_image terms exist to read."
        )
    return torch.stack(
        [depth_image(env, SceneEntityCfg(name), **params) for name, params in terms], dim=1
    )


def depth_image_4d(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
    height: int = 26,
    width: int = 30,
) -> torch.Tensor:
    """Depth image of shape ``(B, 1, H, W)`` for CNN backbones.

    Args:
        env: The environment instance.
        sensor_cfg: Scene entity pointing at the depth camera.
        offset: See :func:`normalize_depth`.
        scale: See :func:`normalize_depth`.
        min_dist: See :func:`normalize_depth`.
        max_dist: See :func:`normalize_depth`.
        min_clip_val: See :func:`normalize_depth`.
        max_clip_val: See :func:`normalize_depth`.
        height: Image height, used to reshape the flattened scan.
        width: Image width, used to reshape the flattened scan.

    Returns:
        The image-shaped depth observation, shape ``(num_envs, 1, height, width)``.
    """
    flat = depth_image(
        env,
        sensor_cfg,
        offset=offset,
        scale=scale,
        min_dist=min_dist,
        max_dist=max_dist,
        min_clip_val=min_clip_val,
        max_clip_val=max_clip_val,
    )
    return flat.reshape(env.num_envs, 1, height, width)
