# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth-camera deployment contract: parse, agree, normalise (numpy only).

The camera geometry, optics, range and normalisation the policy trained on are
embedded in ``tracker.onnx`` (``policy_parameters.depth_camera_spec``) and in
``fm.onnx`` (``fm_meta.extero_spec``) by ``mdp.depth_camera_specs``.  This module is
the consumer side, shared by the MuJoCo sim2sim caster
(``transfer/sim/depth_camera.py``) and the hardware controller:

* :func:`parse_depth_spec` validates an embedded contract;
* :func:`assert_depth_specs_agree` refuses a tracker/generator pair that disagrees
  about the one physical camera pair they both read;
* :func:`normalize_depth_np` reproduces ``mdp.observations.depth.normalize_depth``
  exactly, which is what turns metres of range into the numbers the networks saw --
  with :func:`denormalize_depth_np` and :func:`invalid_depth_value` for the diagnostics
  that need to read an image back.

No torch, no isaaclab.
"""

from __future__ import annotations

import math

import numpy as np


EXTERO_CLIP_NP: dict[str, float] = {"height_scan": 1.0, "depth": 2.0}
"""Symmetric clip band per exteroception source.

Mirrors ``robot_rl.flow_matching.transforms.EXTERO_CLIP``: the height scan is metres of
ground clearance (+-1 m), depth is already NORMALISED by ``(d - offset) * scale``, whose
observation terms clip at +-2.  Clamping depth at the scan's 1.0 would silently discard
real readings.
"""

#: Fields every camera entry must carry for an off-sim consumer to reproduce it.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "obs_term", "mount_body", "offset_pos", "offset_quat_xyzw",
    "width", "height", "fx", "fy", "cx", "cy", "max_distance",
    "offset", "scale", "min_dist", "max_dist", "min_clip_val", "max_clip_val",
)

#: Contract fields two artifacts never have to agree on. ``obs_term`` because a tracker
#: names its observation term and a generator has none; ``update_period`` because the deploy
#: recomputes every control step regardless. Everything not listed here or in
#: :data:`_SAMPLING_FIELDS` is PHYSICAL and always compared, so a field added to the
#: contract is checked without touching this module.
_NAMING_FIELDS: frozenset[str] = frozenset({"obs_term", "update_period"})

#: Fields that describe how a lens is SAMPLED rather than what it is. Comparable only when
#: the two artifacts are required to sample identically; otherwise replaced by the
#: resolution-invariant cone check (:func:`camera_half_angles_rad`).
#:
#: ``fx/fy/cx/cy`` are here because they are linear in the resolution -- once width and
#: height may differ, so must they. ``sensor`` is here rather than in
#: :data:`_NAMING_FIELDS` on purpose: the generator's cameras carry distinct scene names, so
#: it cannot be compared when the samplings are split, but a re-tightened check should still
#: catch a swapped or renamed camera.
_SAMPLING_FIELDS: frozenset[str] = frozenset(
    {"sensor", "width", "height", "fx", "fy", "cx", "cy"}
)

#: Whether a tracker and a generator must SAMPLE the depth cameras identically.
#:
#: ``False`` so the generator's resolution can be changed and retrained without retraining
#: the policy. The physical camera is still pinned (mount, cone, range, normalization); only
#: the pixel grid is free.
#:
#: TODO(depth-resolution): flip back to True once the FM and the RL policy train at one
#: resolution again. ``test_resolution_split_is_opt_in`` fails when this changes, so the
#: flip is a deliberate edit rather than something that drifts.
REQUIRE_SAME_DEPTH_RESOLUTION: bool = False

#: Tolerance on the pinhole half-angle comparison (radians). Tight: the two contracts are
#: derived from the same aperture, so any real difference is a configuration error, not
#: float noise.
_FOV_ATOL_RAD: float = 1e-7


def parse_depth_spec(raw, source: str = "spec") -> list[dict]:
    """Validate an embedded depth-camera contract.

    Args:
        raw: The list of per-camera dicts read from ONNX metadata (or ``None``).
        source: Label for error messages (e.g. ``"tracker.onnx"``).

    Returns:
        One validated dict per camera, in observation order.  Empty for ``None``.

    Raises:
        ValueError: If an entry is missing a required field, has a non-positive
            resolution, or carries an asymmetric observation clip (the deploy applies a
            single magnitude, matching every clip the env cfg sets).
    """
    if not raw:
        return []
    out: list[dict] = []
    for i, cam in enumerate(raw):
        missing = [f for f in _REQUIRED_FIELDS if f not in cam]
        if missing:
            raise ValueError(f"{source} camera {i} is missing field(s) {missing}.")
        if int(cam["width"]) <= 0 or int(cam["height"]) <= 0:
            raise ValueError(
                f"{source} camera {i} has non-positive resolution "
                f"{cam['width']}x{cam['height']}."
            )
        clip = cam.get("clip")
        if clip is not None and abs(float(clip[0])) != abs(float(clip[1])):
            raise ValueError(
                f"{source} camera {i} has an asymmetric obs clip {list(clip)}; the "
                "deploy applies one magnitude per term."
            )
        out.append(dict(cam))
    return out


#: The :func:`normalize_depth_np` arguments a contract entry carries.
_NORM_FIELDS: tuple[str, ...] = (
    "offset", "scale", "min_dist", "max_dist", "min_clip_val", "max_clip_val",
)


def depth_norm_kwargs(cam: dict) -> dict[str, float]:
    """The :func:`normalize_depth_np` keyword arguments for one camera.

    Args:
        cam: One parsed camera contract entry.

    Returns:
        The normalisation arguments, so a caller never spells the field list itself.
    """
    return {key: float(cam[key]) for key in _NORM_FIELDS}


def normalize_camera_images(spec: list[dict], images) -> np.ndarray:
    """Turn one raw metre reading per camera into the stacked ``state["depth"]``.

    The hardware sibling of ``transfer/sim/depth_camera.MujocoDepthCameras.compute``: the
    sim casts rays and this reads a camera, but both end at the same stacked, normalised
    array in contract order, so :class:`~deploy.gen_tracker.GenTrackerPolicy` cannot tell
    them apart.  Lives here rather than in the ROS node so it stays torch-free and
    testable without ROS.

    The reading must be **distance along the camera's optical axis** in metres --
    ``distance_to_image_plane``, which is what the training sensor produces and what the
    ZED SDK's depth measure already is.  A ray-length reading would need the per-pixel
    ``cos`` correction the MuJoCo caster applies, and passing one here is a silent
    sim2real error, worst at the frame edges where the cameras are widest.

    Invalid pixels need no special handling: NaN, ``±inf`` and out-of-range readings all
    land on :func:`invalid_depth_value` inside :func:`normalize_depth_np`, exactly as a
    missed ray does in sim.

    Args:
        spec: The resolved camera contract, from
            :meth:`~deploy.gen_tracker.GenTrackerPolicy.get_depth_camera_spec`.
        images: One reading per camera, in contract order.  Each is row-major with
            ``v = 0`` the top row, either flat ``(H*W,)`` or ``(H, W)``.

    Returns:
        ``(T, H*W)`` normalised images, ready for ``state["depth"]``.

    Raises:
        ValueError: On a camera-count mismatch, a missing reading, a reading whose size
            does not match that camera's resolution, or cameras of differing resolution
            (the images share one array, as in
            ``MujocoDepthCameras``).
    """
    images = list(images)
    if len(images) != len(spec):
        raise ValueError(
            f"Got {len(images)} depth reading(s) for {len(spec)} camera(s) in the "
            "contract; the policy needs one per camera, in contract order."
        )
    sizes = [int(cam["width"]) * int(cam["height"]) for cam in spec]
    if len(set(sizes)) > 1:
        raise ValueError(
            f"depth cameras have differing pixel counts ({sizes}); they are returned as "
            "one array (and the FM stacks them on one token axis), so they must match."
        )
    rows: list[np.ndarray] = []
    for i, (cam, image) in enumerate(zip(spec, images)):
        label = cam.get("sensor") or cam.get("obs_term") or f"camera {i}"
        if image is None:
            raise ValueError(
                f"No depth reading for '{label}'. The controller must hold a frame from "
                "every camera before stepping the policy."
            )
        flat = np.asarray(image, dtype=np.float64).reshape(-1)
        expected = int(cam["width"]) * int(cam["height"])
        if flat.size != expected:
            raise ValueError(
                f"Depth reading for '{label}' has {flat.size} pixel(s) but the contract "
                f"says {cam['width']}x{cam['height']} = {expected}."
            )
        rows.append(normalize_depth_np(flat, **depth_norm_kwargs(cam)))
    return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float64)


def denormalize_depth_np(image: np.ndarray, offset: float, scale: float) -> np.ndarray:
    """Recover metres from a normalised image -- the inverse of :func:`normalize_depth_np`.

    Invalid pixels come back as :func:`invalid_depth_value` rather than a
    real range, since the normalisation is not injective there.

    Args:
        image: Normalised depth, any shape.
        offset: The ``offset`` used to normalise.
        scale: The ``scale`` used to normalise.

    Returns:
        Depth in metres, same shape.
    """
    return np.asarray(image, dtype=np.float64) / scale + offset


def invalid_depth_value(cam: dict) -> float:
    """The normalised value a camera reports for "no reading".

    Every invalid class -- too near, beyond range, non-finite, dropped -- lands on this
    single value, so a consumer identifies them by comparing against it.

    Args:
        cam: One parsed camera contract entry.

    Returns:
        ``(max_clip_val - offset) * scale``.
    """
    return (float(cam["max_clip_val"]) - float(cam["offset"])) * float(cam["scale"])


def camera_half_angles_rad(cam: dict) -> tuple[float, float, float, float]:
    """The four pinhole half-angles ``(left, right, up, down)`` in radians.

    The quantity that says two cameras look through the same LENS whatever grid they sample
    it on, and therefore what makes a differing resolution safe to allow. It is exactly
    resolution-free: with ``fx = W * f / aperture`` and ``cx = W / 2``,
    ``atan(cx / fx) = atan(aperture / 2f)`` -- the width cancels.

    All four, not their sum: a total FOV would be preserved by a principal point pushed
    off-centre in one direction and compensated in the other, which is a different camera.

    Args:
        cam: One parsed camera contract entry.

    Returns:
        ``(left, right, up, down)`` half-angles.
    """
    return (
        math.atan(float(cam["cx"]) / float(cam["fx"])),
        math.atan((float(cam["width"]) - float(cam["cx"])) / float(cam["fx"])),
        math.atan(float(cam["cy"]) / float(cam["fy"])),
        math.atan((float(cam["height"]) - float(cam["cy"])) / float(cam["fy"])),
    )


def assert_depth_specs_agree(
    a: list[dict], b: list[dict], name_a: str = "tracker", name_b: str = "generator",
    require_same_resolution: bool | None = None,
) -> None:
    """Raise unless two artifacts describe the same physical depth cameras.

    A deployment has ONE camera pair, so a tracker and a generator running together must
    have trained on the same mount, optics, range and normalisation.  Mismatched geometry
    is otherwise silent: both networks accept the wrong image and simply behave worse.

    The generator may SAMPLE that pair on its own grid (so its vision input can change
    without retraining the policy), which is what ``require_same_resolution=False`` allows.
    Relaxing it is not merely "drop width and height": that would also permit a different
    LENS, since ``fx/fy/cx/cy`` are free to move once the resolution is. So the sampling
    fields are exchanged for a positive check on the resolution-invariant cone
    (:func:`camera_half_angles_rad`) -- the constraint is weakened, not removed.

    Args:
        a: Parsed contract from the first artifact.
        b: Parsed contract from the second artifact.
        name_a: Label for the first artifact, used in the error.
        name_b: Label for the second artifact, used in the error.
        require_same_resolution: Whether the two must sample identically. ``None`` takes
            :data:`REQUIRE_SAME_DEPTH_RESOLUTION`, the deployment-wide default.

    Raises:
        ValueError: On a differing camera count, any differing physical field, or (when
            resolutions may differ) a differing cone -- listing every disagreement rather
            than only the first.
    """
    strict = REQUIRE_SAME_DEPTH_RESOLUTION if require_same_resolution is None \
        else bool(require_same_resolution)
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} has {len(a)} depth camera(s) but {name_b} has {len(b)}; they "
            "cannot be reading the same physical pair."
        )
    ignored = _NAMING_FIELDS if strict else (_NAMING_FIELDS | _SAMPLING_FIELDS)
    diffs: list[str] = []
    for i, (cam_a, cam_b) in enumerate(zip(a, b)):
        label = cam_a.get("sensor") or cam_a.get("obs_term") or f"camera {i}"
        for field in sorted((set(cam_a) | set(cam_b)) - ignored):
            va, vb = cam_a.get(field), cam_b.get(field)
            if _values_differ(va, vb):
                diffs.append(f"  {label}.{field}: {name_a}={va!r} {name_b}={vb!r}")
        if strict:
            continue
        # The sampling is allowed to differ, so the LENS has to be checked directly.
        for axis, va, vb in zip(("hfov_left", "hfov_right", "vfov_up", "vfov_down"),
                                camera_half_angles_rad(cam_a), camera_half_angles_rad(cam_b)):
            if abs(va - vb) > _FOV_ATOL_RAD:
                diffs.append(
                    f"  {label}.{axis}: {name_a}={math.degrees(va):.4f}deg "
                    f"{name_b}={math.degrees(vb):.4f}deg -- same pixel grid or not, these "
                    "are different lenses"
                )
    if diffs:
        raise ValueError(
            f"{name_a} and {name_b} disagree about the depth cameras:\n" + "\n".join(diffs)
            + "\nRe-export both from the same camera configuration."
        )


def _values_differ(va, vb) -> bool:
    """Whether two contract values differ, comparing numbers within float tolerance.

    A parsed contract holds only ``None``, strings, numbers and lists of numbers, so a
    JSON round-trip's last-bit wobble must not read as a geometry mismatch.

    Args:
        va: Value from the first contract.
        vb: Value from the second contract.

    Returns:
        True if the values disagree.
    """
    if va is None or vb is None or isinstance(va, str) or isinstance(vb, str):
        return va != vb
    return not np.allclose(np.asarray(va, dtype=np.float64), np.asarray(vb, dtype=np.float64),
                           rtol=1e-9, atol=1e-9)


def normalize_depth_np(
    d: np.ndarray,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
) -> np.ndarray:
    """Clip invalid depth then apply ``(d - offset) * scale``, in metres.

    Numpy port of ``robot_rl...mdp.observations.depth.normalize_depth``; the two are
    parity-tested (``transfer/deploy/tests/test_depth_obs.py``).  The ORDER matters and
    matches the torch original: the too-near test runs first, so ``-inf`` is caught there
    and becomes ``min_clip_val`` rather than ``max_clip_val``.  Comparisons are strict, so
    a reading exactly at ``min_dist`` / ``max_dist`` stays valid, and both substitutions
    happen in metres, before the affine transform.

    Args:
        d: Raw depth in metres (the camera's ``distance_to_image_plane``), any shape.
        offset: Subtracted before scaling.
        scale: Multiplies the offset depth.
        min_dist: Below this, a reading is treated as invalid.
        max_dist: Above this, a reading is treated as invalid.
        min_clip_val: Pre-normalisation replacement for too-near pixels.
        max_clip_val: Pre-normalisation replacement for too-far / missed pixels.

    Returns:
        The normalised depth, same shape as ``d``, dtype float64.
    """
    out = np.array(d, dtype=np.float64, copy=True)
    out[out < min_dist] = min_clip_val
    out[(out > max_dist) | ~np.isfinite(out)] = max_clip_val
    return (out - offset) * scale


def pinhole_ray_directions(width: int, height: int, fx: float, fy: float,
                           cx: float, cy: float) -> np.ndarray:
    """Unit ray directions in the camera frame, matching IsaacLab's pinhole pattern.

    Reproduces ``isaaclab.sensors.ray_caster.patterns.pinhole_camera_pattern``: pixel
    CENTRES through the inverse intrinsic matrix, then mapped into the robotics camera
    frame (x = optical axis, y = left, z = up).  Row-major flat order ``k = v * width + u``
    with ``v = 0`` the TOP image row (camera +z) and ``u = 0`` the LEFT column (camera +y).
    The rays are tangent-linear (a true pinhole), not equiangular.

    Lives here rather than in ``transfer/sim`` because three consumers need it and only one
    of them is a simulator: the MuJoCo caster, :func:`unproject_to_yaw_frame` (which the
    hardware runtime also calls), and the diagnostics that rebuild a cloud from an image.
    ``transfer/sim`` imports ``transfer/deploy``, never the reverse.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Horizontal principal point in pixels.
        cy: Vertical principal point in pixels.

    Returns:
        Unit directions ``(width * height, 3)``, float64.
    """
    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64), indexing="xy")
    dirs = np.stack(
        [np.ones_like(u), -(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy], axis=-1
    ).reshape(-1, 3)
    return dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)


def invalid_depth_mask(cam: dict, image: np.ndarray) -> np.ndarray:
    """Which pixels of a NORMALISED image carry no usable range reading.

    The test is ``denormalised depth < min_dist``, NOT equality with the sentinel
    :func:`normalize_depth_np` writes, and the difference is load-bearing.

    The sentinels (``min_clip_val`` for too-near and ``-inf``, ``max_clip_val`` for
    beyond-range, non-finite and dropped) are exact only in a CLEAN image. In training they
    are not: ``mdp.sensors.noise.DepthNoiser`` runs after normalisation and adds a plane
    bias, per-pixel noise and a 3x3 BLUR, which mixes a missed pixel with its neighbours and
    leaves it near the sentinel rather than on it. An equality test would call those pixels
    valid and unproject them to ~0.1 m -- a fake surface on the robot itself.

    Thresholding the range instead catches all of it, and catches it the SAME way in every
    consumer: the live Isaac term, the offline dataset, the MuJoCo caster and the hardware
    runtime all see only a normalised image, so all four must agree without one of them
    having privileged access to a clean signal. Both sentinels denormalise to at most
    ``max(min_clip_val, max_clip_val)`` metres -- 0 m in the shipped contract -- so they are
    always below the threshold.

    The honest limit: the noise half-width (~0.13 m of range) is comparable to ``min_dist``
    (0.10-0.15 m), so the classes genuinely overlap and a few missed pixels survive as
    readings just inside the near limit. That is a handful of sky pixels landing close to the
    camera, and it is identical in training and at deploy, which is what matters.

    Args:
        cam: One parsed camera contract entry.
        image: Normalised depth, any shape.

    Returns:
        Boolean mask, the shape of ``image``.
    """
    arr = np.asarray(image, dtype=np.float64)
    depth_m = denormalize_depth_np(arr, float(cam["offset"]), float(cam["scale"]))
    return ~np.isfinite(depth_m) | (depth_m < float(cam["min_dist"]))


def unproject_to_yaw_frame(
    cam: dict,
    image: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    dirs_cam: np.ndarray | None = None,
) -> np.ndarray:
    """A normalised depth image -> its 3-D points in the root's yaw-aligned frame.

    The frame is the FM's own anchor frame (``flow_matching.transforms``): origin at the
    root link, axes yaw-aligned, so the frame's z IS gravity and the frame itself carries no
    roll and no pitch.

    **Static terrain therefore keeps its coordinates when the robot tilts**, and that
    invariance is the whole reason these channels exist.  A pelvis pitch rewrites every
    pixel of the image, but it moves the CAMERA by exactly the same amount, and the pose the
    caller supplies moves with it -- so composing the two puts the step edge back at the
    coordinate it already had.  The network sees neither joint angles nor root orientation,
    so this is a registration it cannot perform for itself from depth alone.

    What a tilt does move is the robot's own geometry within the level frame: the camera,
    and the limbs the cameras can see.  The tilt therefore stays observable; it just stops
    being confusable with the terrain moving.  Expressing the cloud here is also what lets
    the generator compare "the step edge is at x = 0.8" against its own predicted base
    position in the same numbers.

    **Nothing here needs a world pose.** The caller supplies the camera pose already in that
    frame, and the heading cancels out of it: writing ``R_world_root = Rz(psi) @ R_rp``, a
    point's yaw-frame coordinates are ``R_rp @ p_root``, in which ``psi`` does not appear.
    So a deployment needs only forward kinematics from the root to the camera and the root's
    roll/pitch (i.e. gravity, from the IMU) -- never odometry, and a drifting yaw estimate
    cannot corrupt the result.

    That is a statement about the COMPUTATION, not an invariance: the frame turns with the
    robot, so yawing in place does rotate a static point's coordinates by ``-psi``.  It is
    the window the generator predicts turning with them that makes this consistent, not the
    cloud standing still.

    Depth is converted to ray LENGTH before it is projected out
    (``distance_to_image_plane`` is the perpendicular z-depth, and
    ``dirs_cam[:, 0]`` is the cosine to the optical axis), matching the reconstruction
    ``view_depth_compare_viser.py`` documents.

    Two range subtleties, both consequences of feeding the model's own input back through
    the pinhole model rather than reading the sensor:

    * ``image`` must be the SAME array the depth channel carries -- post-clip and
      post-sanitise. A valid pixel therefore saturates at the clip band's equivalent range
      (4.75 m for the shipped ``+-2`` / ``0.5`` / ``0.75``), and the cloud has a wall there.
      Deriving xyz from an unclipped image instead would describe a different distance than
      the depth channel it sits beside.
    * invalid pixels are placed at ``max_dist`` of RAY LENGTH, which is the sensor's honest
      per-ray "nothing within range this way" -- a spherical shell at its reach. Setting
      their z-depth to ``max_dist`` instead would put the image corners 13 m out on a 7.9 m
      camera, a claim it never makes. Left alone they would denormalise to 0 m and pile a
      fake surface onto the robot's own origin, the worst place for a convolution to find
      one. The depth channel keeps its sentinel, so validity stays recoverable (see
      :func:`invalid_depth_mask`).

    Args:
        cam: One parsed camera contract entry (intrinsics, range, normalisation).
        image: Normalised depth for this camera, ``(H*W,)`` or ``(H, W)``.
        cam_pos: Camera origin in the yaw frame, ``(3,)``.
        cam_rot: Camera-to-yaw-frame rotation, ``(3, 3)``.
        dirs_cam: Optional precomputed :func:`pinhole_ray_directions` for this camera;
            recomputed when omitted. Pass it on a hot path -- it depends only on the
            contract, so it is constant for the life of a run.

    Returns:
        Points ``(3, H*W)``, float64, row-major in the same pixel order as ``image``.

    Raises:
        ValueError: If the image does not hold ``height * width`` pixels.
    """
    width, height = int(cam["width"]), int(cam["height"])
    flat = np.asarray(image, dtype=np.float64).reshape(-1)
    if flat.size != width * height:
        raise ValueError(
            f"camera '{cam.get('sensor', cam.get('obs_term'))}' is {width}x{height} "
            f"({width * height} px) but the image holds {flat.size}."
        )
    if dirs_cam is None:
        dirs_cam = pinhole_ray_directions(width, height, float(cam["fx"]), float(cam["fy"]),
                                          float(cam["cx"]), float(cam["cy"]))
    depth_m = denormalize_depth_np(flat, float(cam["offset"]), float(cam["scale"]))
    # z-depth -> ray length for a real reading; an invalid pixel is given the ray length
    # DIRECTLY, so the misses form a spherical shell at the sensor's reach rather than a
    # plane whose corners would sit 13 m out (see the docstring).
    ray_len = np.where(invalid_depth_mask(cam, flat), float(cam["max_dist"]),
                       depth_m / dirs_cam[:, 0])
    points_cam = dirs_cam * ray_len[:, None]                             # (N, 3)
    return (np.asarray(cam_rot, dtype=np.float64) @ points_cam.T
            + np.asarray(cam_pos, dtype=np.float64).reshape(3, 1))


def sanitize_extero_np(
    values: np.ndarray, source: str, clip: float | None = None
) -> np.ndarray:
    """Make an exteroception reading finite and clamp it to its band.

    Numpy counterpart of ``robot_rl.flow_matching.transforms.sanitize_extero``.  The
    ``fm.onnx`` graph folds in the exteroception normalization, so one non-finite entry
    would poison the conditioning.

    Both non-finite ends map to ``+clip``, matching the torch side: a missed downward ray
    means "no ground within range", which is ground very far BELOW, so the drop-off end is
    where it belongs.  Mapping it to ``-clip`` would encode a hole as a wall, a value real
    terrain also produces.

    Args:
        values: Raw reading, any shape.
        source: An :data:`EXTERO_CLIP_NP` key.
        clip: Band override (the checkpoint's ``fm_meta.extero_clip``).  ``None`` uses the
            per-source default, which is what a graph exported before the field used.

    Returns:
        A finite array clamped to the band.

    Raises:
        KeyError: On an unknown ``source``.
    """
    band = EXTERO_CLIP_NP[source] if clip is None else float(clip)
    finite = np.nan_to_num(
        np.asarray(values, dtype=np.float64), nan=0.0, posinf=band, neginf=band
    )
    return np.clip(finite, -band, band)
