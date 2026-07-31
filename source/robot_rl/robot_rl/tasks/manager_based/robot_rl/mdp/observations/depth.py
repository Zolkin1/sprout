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

import inspect
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import get_depth_noiser

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEPTH_OBS_FUNCS: tuple[str, ...] = ("depth_image", "depth_image_4d")
"""Observation-term functions that read a depth camera.

The marker for "this term consumes a camera", used to discover the cameras a task
actually observes rather than the ones its scene happens to hold.
"""


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


def _normalization_params(params: dict) -> dict[str, float]:
    """Resolve the :func:`normalize_depth` arguments an observation term will use.

    Reads the defaults off :func:`normalize_depth`'s own signature, so an exported
    contract states the values actually applied even for arguments the term leaves
    unset -- and cannot drift from the function.

    Args:
        params: An observation term's ``params`` dict (extra keys are ignored).

    Returns:
        Every :func:`normalize_depth` keyword argument, as floats.
    """
    signature = inspect.signature(normalize_depth)
    return {
        name: float(params.get(name, parameter.default))
        for name, parameter in signature.parameters.items()
        if name != "d"
    }


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


def _walk_depth_obs_terms(
    env: "ManagerBasedRLEnv", groups: list[str] | None = None
) -> list[tuple[str, str, dict, tuple | None]]:
    """Walk the observation manager for depth terms, deduplicated by camera.

    The single traversal behind :func:`find_depth_obs_terms` and
    :func:`depth_camera_specs`, so the recorder, the FM dataset, the exporter and the
    deployed tracker all see the same cameras in the same order.

    Args:
        env: The environment instance.
        groups: Observation groups to search, or ``None`` for all of them.

    Returns:
        ``[(term_name, sensor_name, params, clip), ...]`` in observation-group order,
        keeping the FIRST term found per camera.  ``params`` excludes ``sensor_cfg``;
        ``clip`` is the term's clip tuple or ``None``.
    """
    manager = env.observation_manager
    found: list[tuple[str, str, dict, tuple | None]] = []
    seen: set[str] = set()
    for group in (manager.active_terms if groups is None else groups):
        names = manager.active_terms.get(group, [])
        cfgs = manager._group_obs_term_cfgs.get(group, [])
        for name, cfg in zip(names, cfgs):
            if getattr(cfg.func, "__name__", "") not in DEPTH_OBS_FUNCS:
                continue
            params = dict(cfg.params or {})
            sensor_cfg = params.pop("sensor_cfg", None)
            if sensor_cfg is None or sensor_cfg.name in seen:
                continue
            seen.add(sensor_cfg.name)
            clip = getattr(cfg, "clip", None)
            found.append((name, sensor_cfg.name, params, tuple(clip) if clip is not None else None))
    return found


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
        ``params`` is directly usable as :func:`depth_image` keyword arguments.
    """
    out = []
    for _, sensor_name, params, _ in _walk_depth_obs_terms(env):
        # height/width are reshape-only args for the 4d variant; drop them so the
        # stacked reading is always flat per camera.
        params.pop("height", None)
        params.pop("width", None)
        out.append((sensor_name, params))
    return out


def depth_camera_specs(
    env: "ManagerBasedRLEnv", groups: list[str] | None = None
) -> list[dict]:
    """The deployment contract for every depth camera, in observation order.

    Everything an off-sim consumer (MuJoCo sim2sim, hardware) needs to REPRODUCE the
    depth observation this policy trained on: where the camera is mounted, its pinhole
    intrinsics, its range, and the exact normalisation applied on top.  Embedded in
    ``tracker.onnx`` and ``fm.onnx`` (see ``scripts/rsl_rl/export_parameters.py`` and
    ``scripts/flow_matching/export_fm_onnx.py``) so a deployed artifact carries its own
    camera geometry and two artifacts driving one physical camera pair can be checked
    against each other.

    Derived entirely from the sensor cfg and the observation term, never hand-copied.
    Intrinsics come from the CFG rather than ``sensor.data.intrinsic_matrices`` so
    per-environment intrinsics randomisation cannot leak into the export.

    Args:
        env: The environment instance.
        groups: Observation groups to describe, or ``None`` for all of them.  The
            exporter passes the group it is exporting, so a contract never names a camera
            whose observation term the deployed policy does not actually read.

    Returns:
        One dict per camera, ordered as :func:`depth_obs_stack`'s token axis.  Empty
        when the task has no depth cameras.

    Raises:
        ValueError: If a camera is configured in a way the off-sim port does not
            reproduce (a non-``world`` offset convention, a data type other than
            ``distance_to_image_plane``, or a clipping behaviour other than ``max``).
    """
    specs: list[dict] = []
    for term_name, sensor_name, params, clip in _walk_depth_obs_terms(env, groups):
        cfg = env.scene.sensors[sensor_name].cfg
        if cfg.offset.convention != "world":
            raise ValueError(
                f"depth camera '{sensor_name}' uses offset convention "
                f"'{cfg.offset.convention}'; the off-sim port assumes 'world' "
                "(camera x = optical axis, y = left, z = up)."
            )
        if list(cfg.data_types) != ["distance_to_image_plane"]:
            raise ValueError(
                f"depth camera '{sensor_name}' has data_types={list(cfg.data_types)}; "
                "the off-sim port reproduces 'distance_to_image_plane' only."
            )
        if cfg.depth_clipping_behavior != "max":
            raise ValueError(
                f"depth camera '{sensor_name}' has depth_clipping_behavior="
                f"'{cfg.depth_clipping_behavior}'; the off-sim port assumes 'max' (a "
                "missed ray reports exactly max_distance)."
            )
        pattern = cfg.pattern_cfg
        width, height = int(pattern.width), int(pattern.height)
        focal = float(pattern.focal_length)
        spec = {
            "obs_term": term_name,
            "sensor": sensor_name,
            # The mount prim's leaf name, which is also the rigid-body name off-sim.
            "mount_body": cfg.prim_path.rstrip("/").split("/")[-1],
            "offset_pos": [float(v) for v in cfg.offset.pos],
            "offset_quat_xyzw": [float(v) for v in cfg.offset.rot],
            "width": width,
            "height": height,
            # Mirrors RayCasterCamera's intrinsic matrix: the focal-length unit cancels,
            # only the focal/aperture ratio matters.
            "fx": width * focal / float(pattern.horizontal_aperture),
            "fy": height * focal / float(pattern.vertical_aperture),
            "cx": width / 2.0,
            "cy": height / 2.0,
            "max_distance": float(cfg.max_distance),
            "update_period": float(cfg.update_period),
            "data_type": "distance_to_image_plane",
            "depth_clipping_behavior": str(cfg.depth_clipping_behavior),
            "clip": [float(clip[0]), float(clip[1])] if clip is not None else None,
        }
        spec.update(_normalization_params(params))
        specs.append(spec)
    return specs


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
            "no depth observation terms found; run a depth task so mdp.depth_image terms "
            "exist to read -- G1-teacher-clf-terrain-depth to train, or "
            "G1-tracker-clf-fm-terrain-depth-play to deploy a depth-conditioned generator "
            "(the height-scan tracker task has no cameras)."
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
