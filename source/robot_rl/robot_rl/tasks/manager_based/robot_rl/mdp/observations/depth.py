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
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, yaw_quat

from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import get_depth_noiser

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


DEPTH_OBS_FUNCS: tuple[str, ...] = ("depth_image", "depth_image_4d")
"""Observation-term functions that read a depth camera.

The marker for "this term consumes a camera", used to discover the cameras a task
actually observes rather than the ones its scene happens to hold.
"""

FM_DEPTH_OBS_GROUP: str = "fm_depth"
"""Observation group holding the GENERATOR's cameras when it samples at its own resolution.

A separate group rather than extra terms in ``policy`` for two reasons: the policy's
observation vector must not change (that is the whole point of decoupling the two), and
:func:`depth_obs_stack` stacks every depth term it is given into one array, which cannot
hold images of two different widths. See :func:`fm_depth_groups`.
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


_RAY_DIRS_CACHE: dict[tuple, torch.Tensor] = {}
"""Per-camera ray directions, keyed by intrinsics + device + dtype.

The pattern depends only on the camera CONTRACT, which is fixed for the life of a run (the
intrinsics in the contract come from the cfg, so the per-env intrinsics randomisation does
not reach it -- see :func:`depth_camera_specs`). Recomputing a 780x3 meshgrid every step for
every camera is pure waste.
"""


def pinhole_ray_directions(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unit ray directions in the camera frame, matching IsaacLab's pinhole pattern.

    Torch counterpart of ``transfer/deploy/depth_obs.pinhole_ray_directions``, which is the
    reference implementation; the two are parity-tested in
    ``transfer/deploy/tests/test_depth_unproject.py``. They are separate implementations
    rather than one shared function because ``robot_rl`` must not import ``transfer``.

    Pixel CENTRES through the inverse intrinsic matrix, mapped into the robotics camera frame
    (x = optical axis, y = left, z = up). Row-major flat order ``k = v * width + u`` with
    ``v = 0`` the TOP row and ``u = 0`` the LEFT column -- the same ordering
    ``flow_matching.extero_encoder`` folds an image with, and the reason that ordering is a
    contract rather than a convention.

    Results are cached (see :data:`_RAY_DIRS_CACHE`), so the returned tensor is SHARED: treat
    it as read-only.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Horizontal principal point in pixels.
        cy: Vertical principal point in pixels.
        device: Device to build the directions on.
        dtype: Floating dtype of the result.

    Returns:
        Unit directions ``(height * width, 3)``.
    """
    key = (int(width), int(height), float(fx), float(fy), float(cx), float(cy),
           str(device), dtype)
    cached = _RAY_DIRS_CACHE.get(key)
    if cached is not None:
        return cached
    u = torch.arange(int(width), device=device, dtype=dtype)
    v = torch.arange(int(height), device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")          # row-major: v outer, u inner
    dirs = torch.stack(
        [torch.ones_like(uu), -(uu + 0.5 - cx) / fx, -(vv + 0.5 - cy) / fy], dim=-1
    ).reshape(-1, 3)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    _RAY_DIRS_CACHE[key] = dirs
    return dirs


def invalid_depth_mask(cam: dict, image: torch.Tensor) -> torch.Tensor:
    """Which pixels of a NORMALISED image carry no usable range reading.

    Torch counterpart of ``transfer/deploy/depth_obs.invalid_depth_mask``, whose docstring
    carries the reasoning in full. The short version: the test is ``denormalised depth <
    min_dist``, not equality with the sentinel, because :class:`~robot_rl.tasks.
    manager_based.robot_rl.mdp.sensors.noise.DepthNoiser` runs AFTER normalisation and blurs
    a missed pixel off its sentinel -- and because all four consumers of these channels see
    only a normalised image, so none of them may use a rule the others cannot.

    Args:
        cam: One camera contract entry (:func:`depth_camera_specs`).
        image: Normalised depth, any shape.

    Returns:
        Boolean mask, the shape of ``image``.
    """
    depth_m = image / float(cam["scale"]) + float(cam["offset"])
    return ~torch.isfinite(depth_m) | (depth_m < float(cam["min_dist"]))


def unproject_to_yaw_frame(
    cam: dict,
    image: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_rot: torch.Tensor,
    dirs_cam: torch.Tensor | None = None,
) -> torch.Tensor:
    """A normalised depth image -> its 3-D points in the root's yaw-aligned frame.

    Torch counterpart of ``transfer/deploy/depth_obs.unproject_to_yaw_frame``, which holds
    the full reasoning: the frame, why the heading cancels out of it (so nothing here needs
    odometry), the z-depth-to-ray-length conversion, and what happens at the clip band and at
    an invalid pixel. Parity-tested against it.

    Args:
        cam: One camera contract entry (:func:`depth_camera_specs`).
        image: Normalised depth ``(B, H*W)`` -- the SAME array the depth channel carries.
        cam_pos: Camera origin in the yaw frame, ``(B, 3)``.
        cam_rot: Camera-to-yaw-frame rotation, ``(B, 3, 3)``.
        dirs_cam: Optional precomputed :func:`pinhole_ray_directions` for this camera.

    Returns:
        Points ``(B, 3, H*W)``, row-major in the same pixel order as ``image``.

    Raises:
        ValueError: If the image does not hold ``height * width`` pixels.
    """
    width, height = int(cam["width"]), int(cam["height"])
    if image.shape[-1] != width * height:
        raise ValueError(
            f"camera '{cam.get('sensor', cam.get('obs_term'))}' is {width}x{height} "
            f"({width * height} px) but the image rows are {image.shape[-1]} wide."
        )
    if dirs_cam is None:
        dirs_cam = pinhole_ray_directions(
            width, height, float(cam["fx"]), float(cam["fy"]), float(cam["cx"]),
            float(cam["cy"]), image.device, image.dtype,
        )
    depth_m = image / float(cam["scale"]) + float(cam["offset"])
    # z-depth -> ray length for a real reading; an invalid pixel takes the ray length
    # DIRECTLY, so misses form a spherical shell at the sensor's reach instead of a plane
    # whose corners would sit far beyond it.
    ray_len = torch.where(
        invalid_depth_mask(cam, image),
        torch.as_tensor(float(cam["max_dist"]), device=image.device, dtype=image.dtype),
        depth_m / dirs_cam[:, 0],
    )
    points_cam = ray_len.unsqueeze(-1) * dirs_cam                    # (B, N, 3)
    return torch.bmm(cam_rot, points_cam.transpose(1, 2)) + cam_pos.unsqueeze(-1)


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


def fm_depth_groups(env: "ManagerBasedRLEnv") -> list[str] | None:
    """The observation group holding the GENERATOR's cameras, or ``None`` if there is none.

    ``None`` means "every group", which is the pre-existing behaviour -- so every consumer
    that scopes itself with this is a no-op on a task whose FM and policy share one camera
    pair, and no flag has to be plumbed anywhere to keep it that way. The pair only exists
    when :func:`~robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg.add_fm_depth_pair`
    put it there, i.e. when the generator's depth resolution differs from the policy's.

    Args:
        env: The environment instance.

    Returns:
        ``[FM_DEPTH_OBS_GROUP]`` when that group is active, else ``None``.
    """
    return ([FM_DEPTH_OBS_GROUP]
            if FM_DEPTH_OBS_GROUP in env.observation_manager.active_terms else None)


def rl_depth_groups(env: "ManagerBasedRLEnv") -> list[str]:
    """Every observation group EXCEPT the generator's -- i.e. the policy's own cameras.

    Not simply ``["policy"]``: the ``-image`` task variant puts the policy's depth in its
    own per-camera groups, so the policy's cameras are "whatever is not the FM's".

    Args:
        env: The environment instance.

    Returns:
        Active group names, excluding :data:`FM_DEPTH_OBS_GROUP`.
    """
    return [g for g in env.observation_manager.active_terms if g != FM_DEPTH_OBS_GROUP]


def find_depth_obs_terms(
    env: "ManagerBasedRLEnv", groups: list[str] | None = None
) -> list[tuple[str, dict]]:
    """Discover depth observation terms, as ``(sensor_name, params)`` in obs order.

    The FM's depth conditioning must be normalised EXACTLY as the recorded training data
    was, and the observation terms are the single source of truth for that (each camera
    has its own ``min_dist`` / ``max_dist``).  Discovering them here -- rather than
    duplicating the constants -- means the recorder, the FM dataset and the deployed
    tracker agree by construction.

    Args:
        env: The environment instance.
        groups: Observation groups to search, or ``None`` for all of them. Pass
            :func:`fm_depth_groups` / :func:`rl_depth_groups` when the task carries both a
            policy and a generator camera pair at different resolutions -- unscoped, the
            two would be returned as one list.

    Returns:
        ``[(sensor_name, params), ...]``, deduplicated, in observation-group order.
        ``params`` is directly usable as :func:`depth_image` keyword arguments.
    """
    out = []
    for _, sensor_name, params, _ in _walk_depth_obs_terms(env, groups):
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


def depth_obs_stack(
    env: "ManagerBasedRLEnv", groups: list[str] | None = None
) -> torch.Tensor:
    """Depth cameras' observations stacked on a sensor axis, ``(B, T, H*W)``.

    One row per camera, each normalised with that camera's own observation-term
    parameters.  This is the array recorded for the flow-matching generator and the one
    the deployed tracker rebuilds, so the two cannot drift apart.

    Args:
        env: The environment instance.
        groups: Observation groups to read, or ``None`` for all of them. A task carrying
            BOTH a policy pair and a generator pair at a different resolution must scope
            this (see :func:`fm_depth_groups`): the rows share one array, so cameras of
            two widths cannot be stacked together.

    Returns:
        Stacked depth, shape ``(num_envs, num_cameras, H*W)``.

    Raises:
        RuntimeError: If the environment has no depth observation terms in ``groups``.
        ValueError: If the selected cameras do not share one resolution.
    """
    terms = find_depth_obs_terms(env, groups)
    if not terms:
        raise RuntimeError(
            "no depth observation terms found"
            + (f" in groups {groups}" if groups else "")
            + "; run a depth task so mdp.depth_image terms exist to read -- "
            "G1-teacher-clf-terrain-depth to train, or "
            "G1-tracker-clf-fm-terrain-depth-play to deploy a depth-conditioned generator "
            "(the height-scan tracker task has no cameras)."
        )
    rows = [depth_image(env, SceneEntityCfg(name), **params) for name, params in terms]
    widths = {int(r.shape[-1]) for r in rows}
    if len(widths) > 1:
        # torch.stack would raise on its own, but with a shape message that says nothing
        # about WHY two cameras disagree. This is the mixed-group mistake, named.
        raise ValueError(
            f"cameras {[n for n, _ in terms]} have differing pixel counts {sorted(widths)}, "
            "so they cannot share one stacked array. This is what scoping is for: pass "
            "mdp.fm_depth_groups(env) for the generator's pair or mdp.rl_depth_groups(env) "
            "for the policy's, rather than reading every group at once."
        )
    return torch.stack(rows, dim=1)


#: Channel order of a multi-channel depth reading, outermost axis first.
#:
#: ``d`` is the normalised depth the single-channel path already carried, so channel 0 of a
#: 4-channel row is bit-identical to the 1-channel row -- which is what lets a 1-channel
#: checkpoint and a 4-channel one be compared on the same recording.
DEPTH_XYZ_CHANNELS: tuple[str, ...] = ("d", "x", "y", "z")


def depth_camera_pose_yaw_frame(
    env: "ManagerBasedRLEnv", cam: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """One camera's OPTICAL pose in the robot root's yaw-aligned frame.

    Two choices here, both load-bearing:

    **The pose is built from the MOUNT BODY plus the contract's offset**, never from
    ``RayCasterData.quat_w``. The sensor bakes ``cfg.offset.rot`` into its ray directions
    once at init, so reading ``quat_w`` back gives an optical axis of +x with no pitch,
    unchanged across postures. Mount-body-plus-offset is what the contract means, and it is
    the same construction ``MujocoDepthCameras._Camera.pose`` and
    ``compare_depth_sim2sim._isaac_camera_pose`` use -- so all three stacks describe one
    camera.

    **The offset is the NOMINAL one from the contract**, not the per-env randomised mount.
    A deployed robot knows its calibrated mount and nothing more, so this is the pose it can
    reproduce; the extrinsics DR then perturbs the IMAGE relative to the stated geometry,
    which is exactly the calibration error hardware has. Using the true randomised pose would
    teach the generator to trust a registration it will never be given.

    The frame is built with ``subtract_frame_transforms`` against ``yaw_quat`` of the root --
    the same construction ``flow_matching.transforms._frame_relative_block`` uses for the
    reference window -- so the points and the window the generator predicts cannot end up in
    two different frames.

    Args:
        env: The environment instance.
        cam: One camera contract entry (:func:`depth_camera_specs`).

    Returns:
        ``(pos (B, 3), rot (B, 3, 3))`` -- the camera origin and its camera-to-yaw-frame
        rotation, per environment.
    """
    robot = env.scene["robot"]
    body_i = list(robot.data.body_names).index(cam["mount_body"])
    # EVERY quaternion in this function is scalar-LAST (x, y, z, w) and stays that way:
    # Isaac Lab 3.0's articulation buffers are scalar-last AND so is ``isaaclab.utils.math``
    # (see ``matrix_from_quat`` / ``yaw_quat``), so nothing needs reordering. ``to_wxyz`` from
    # world_lift_isaac is for crossing into MOTION_GEN's helpers, which are scalar-first --
    # applying it here would be a silent ~180 deg roll about x, which is the bug that file
    # exists to document. The contract's own ``offset_quat_xyzw`` is already in this order.
    body_pos = _torch(robot.data.body_pos_w)[:, body_i]
    body_rot = matrix_from_quat(_torch(robot.data.body_quat_w)[:, body_i])

    dev, dtype = body_pos.device, body_pos.dtype
    off_pos = torch.as_tensor(cam["offset_pos"], device=dev, dtype=dtype)
    off_rot = matrix_from_quat(
        torch.as_tensor(cam["offset_quat_xyzw"], device=dev, dtype=dtype)
    )
    cam_pos_w = body_pos + (body_rot @ off_pos)
    cam_rot_w = body_rot @ off_rot

    root_pos = _torch(robot.data.root_pos_w)
    q_yaw = yaw_quat(_torch(robot.data.root_quat_w))
    cam_pos_rel, q_cam_rel = subtract_frame_transforms(
        root_pos, q_yaw, cam_pos_w, _quat_from_matrix(cam_rot_w)
    )
    return cam_pos_rel, matrix_from_quat(q_cam_rel)


def _quat_from_matrix(rot: torch.Tensor) -> torch.Tensor:
    """Quaternion ``(x, y, z, w)`` from a batch of rotation matrices ``(B, 3, 3)``.

    ``isaaclab.utils.math`` has ``matrix_from_quat`` but no inverse, and the composed camera
    rotation is naturally a matrix (mount rotation times mount offset) while
    ``subtract_frame_transforms`` wants a quaternion. Scalar-LAST, matching everything else
    on the Isaac side of this module.

    Shepperd's method: pick the branch whose denominator is largest, which is what keeps it
    stable near each of the four degenerate cases (a naive ``w``-branch loses all precision at
    a 180 deg rotation, and the camera mounts here are pitched 35-65 deg).

    Args:
        rot: Rotation matrices ``(B, 3, 3)``.

    Returns:
        Quaternions ``(B, 4)`` in ``(x, y, z, w)``.
    """
    m = rot
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Four candidate parametrisations; each is exact where its own denominator is largest.
    cands = torch.stack([1.0 + trace, 1.0 + 2.0 * m[:, 0, 0] - trace,
                         1.0 + 2.0 * m[:, 1, 1] - trace, 1.0 + 2.0 * m[:, 2, 2] - trace], -1)
    best = cands.argmax(dim=-1)
    s = torch.sqrt(cands.gather(-1, best[:, None]).squeeze(-1).clamp_min(1e-12)) * 0.5
    quat = torch.zeros(m.shape[0], 4, device=m.device, dtype=m.dtype)
    for branch in range(4):
        sel = best == branch
        if not bool(sel.any()):
            continue
        ms, ss = m[sel], s[sel]
        if branch == 0:                                        # w largest
            quat[sel] = torch.stack([
                (ms[:, 2, 1] - ms[:, 1, 2]) / (4 * ss),
                (ms[:, 0, 2] - ms[:, 2, 0]) / (4 * ss),
                (ms[:, 1, 0] - ms[:, 0, 1]) / (4 * ss), ss], -1)
        elif branch == 1:                                      # x largest
            quat[sel] = torch.stack([
                ss, (ms[:, 0, 1] + ms[:, 1, 0]) / (4 * ss),
                (ms[:, 0, 2] + ms[:, 2, 0]) / (4 * ss),
                (ms[:, 2, 1] - ms[:, 1, 2]) / (4 * ss)], -1)
        elif branch == 2:                                      # y largest
            quat[sel] = torch.stack([
                (ms[:, 0, 1] + ms[:, 1, 0]) / (4 * ss), ss,
                (ms[:, 1, 2] + ms[:, 2, 1]) / (4 * ss),
                (ms[:, 0, 2] - ms[:, 2, 0]) / (4 * ss)], -1)
        else:                                                  # z largest
            quat[sel] = torch.stack([
                (ms[:, 0, 2] + ms[:, 2, 0]) / (4 * ss),
                (ms[:, 1, 2] + ms[:, 2, 1]) / (4 * ss), ss,
                (ms[:, 1, 0] - ms[:, 0, 1]) / (4 * ss)], -1)
    return quat


def _torch(array) -> torch.Tensor:
    """A warp array or torch tensor as a torch tensor.

    Isaac Lab 3.0 hands back ``wp.array`` for the articulation buffers; the rest of this
    module is torch. Mirrors the ``wp.to_torch`` calls the command terms make.

    Args:
        array: A warp array or an existing torch tensor.

    Returns:
        The same data as a torch tensor (no copy).
    """
    if isinstance(array, torch.Tensor):
        return array
    import warp as wp  # noqa: PLC0415  (only present in an Isaac runtime)
    return wp.to_torch(array)


def depth_xyz_stack(
    env: "ManagerBasedRLEnv", groups: list[str] | None = None
) -> torch.Tensor:
    """Depth cameras as 4-channel ``[d, x, y, z]`` rows, ``(B, T, 4*H*W)``.

    The exteroception the CONVOLUTIONAL generator conditions on when
    ``FlowMatchingConfig.extero_channels == 4``. Channel 0 is exactly what
    :func:`depth_obs_stack` returns; channels 1-3 are that same image unprojected into the
    root's yaw-aligned frame (:func:`unproject_to_yaw_frame`), in metres.

    Why the extra channels exist: a convolution over depth alone can see THAT there is a
    step edge and where in the FRAME it sits, but the frame is a pixel grid. Turning "row 14,
    column 8, 2.1 m" into "0.8 m ahead and 0.15 m up" needs the camera pose and the pinhole
    model, neither of which the network ever sees, and both of which change every step as the
    waist joints move and the pelvis pitches. These channels state it directly, in the frame
    the generator already predicts its window in.

    Channel-major per camera (all of ``d``, then all of ``x``, ...) rather than interleaved,
    because ``extero_encoder.images_from_flat`` folds a row to ``(C, H, W)`` and that is the
    layout a conv2d wants.

    TODO(perf): this re-walks the observation manager and re-derives the camera contract
    every call, like :func:`depth_obs_stack`. ``obs`` is already 39% of the env step, so if
    the 4-channel path lands, cache the contract per env rather than paying the walk twice.

    Args:
        env: The environment instance.
        groups: Observation groups to read, or ``None`` for all of them. Scope this with
            :func:`fm_depth_groups` when the generator samples at its own resolution.

    Returns:
        ``(num_envs, num_cameras, 4 * H * W)``.

    Raises:
        RuntimeError: If the environment has no depth observation terms in ``groups``.
        ValueError: If the selected cameras do not share one resolution.
    """
    specs = depth_camera_specs(env, groups)
    if not specs:
        raise RuntimeError(
            "no depth observation terms found"
            + (f" in groups {groups}" if groups else "")
            + "; depth_xyz_stack needs a depth task (see depth_obs_stack)."
        )
    rows = []
    for cam in specs:
        params = {k: cam[k] for k in ("offset", "scale", "min_dist", "max_dist",
                                     "min_clip_val", "max_clip_val")}
        image = depth_image(env, SceneEntityCfg(cam["sensor"]), **params)   # (B, H*W)
        cam_pos, cam_rot = depth_camera_pose_yaw_frame(env, cam)
        points = unproject_to_yaw_frame(cam, image, cam_pos, cam_rot)       # (B, 3, H*W)
        rows.append(torch.cat([image, points.reshape(image.shape[0], -1)], dim=-1))
    widths = {int(r.shape[-1]) for r in rows}
    if len(widths) > 1:
        raise ValueError(
            f"cameras {[c['sensor'] for c in specs]} produce rows of differing widths "
            f"{sorted(widths)}, so they cannot share one stacked array -- scope the groups "
            "with mdp.fm_depth_groups(env) / mdp.rl_depth_groups(env)."
        )
    return torch.stack(rows, dim=1)


def depth_image_4d(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Depth image of shape ``(B, 1, H, W)`` for CNN backbones.

    ``height``/``width`` are REQUIRED (and keyword-only). They used to default to the G1's
    30x26, which is a silently-wrong reshape waiting for the first task that samples at
    another resolution -- and the only caller passes them explicitly anyway.

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
