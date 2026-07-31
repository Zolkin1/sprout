# Copyright 2026 Zachary Olkin. All rights reserved.

"""Terrain teacher variants with two self-scanning depth cameras.

Both cameras hang off ``torso_link`` and look forward-and-down:

- ``depth_camera_near`` -- mid-torso, pitched steeply down, to see the terrain the
  robot is about to step on.
- ``depth_camera_far`` -- at head height, pitched shallowly, to pick up where the
  near camera's field ends.

Both are :class:`MultiMeshRayCasterCameraCfg` sensors that ray-cast against the
terrain **and** against the robot's own limb meshes, so the robot occludes itself
the way a real depth camera would see it.  See
:mod:`~robot_rl.tasks.manager_based.robot_rl.g1.g1_self_scan` for why every limb
needs its own ray-cast target pointed at an inner visual Xform.

Depth REPLACES the height scan in the policy group: the hardware has cameras, not a
heightmap, so a policy that leans on the scan is not deployable.  Two flavours differing
only in the observation SHAPE:

- :class:`G1TeacherTerrainDepthEnvCfg` -- flattened ``depth_near`` / ``depth_far`` terms
  in the policy group.
- :class:`G1TeacherTerrainDepthImageEnvCfg` -- each camera as an image-shaped
  observation in its own group, for CNN backbones.

The CRITIC keeps its height scan, and the pelvis ``height_scanner`` therefore stays in
the scene.  The critic is privileged and never deployed, so a complete terrain
description there is free and stabilises the value function while the policy works the
harder problem from depth alone.  The scan also still feeds the flow-matching
conditioning by default and the MuJoCo sim2sim path, which are separate entry points.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import NamedTuple

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.sensors import DepthNoiseCfg

from .g1_self_scan import link_prim_path, self_scan_prim_paths
from .g1_teacher_env_cfg import G1TeacherEventsCfg, G1TeacherObservationsCfg
from .g1_teacher_terrain_env_cfg import (
    G1TeacherTerrainBodyEnvCfg,
    G1TeacherTerrainBodyEnvCfgPlay,
    G1TeacherTerrainSceneCfg,
)

# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

_DEPTH_W, _DEPTH_H = 30, 26

# Sensor refresh rate.  The env control loop runs at 50 Hz (dt 0.02), so this updates
# the depth every control step -- no decimation between the policy and its camera.
_DEPTH_FPS = 50.0

# Both cameras mount on torso_link -- a real physics body (child of waist_pitch_joint,
# parent of the shoulder joints), so it can host a sensor view.  head_link CANNOT: it,
# logo_link and waist_support_link are instanceable visual-only leaves welded under
# torso_link with no joints of their own, and all three carry the same
# (0.0039635, 0, -0.054) translate -- a URDF-conversion artifact -- so head_link's
# Xform is not even a reliable "head height" reference.
_CAMERA_MOUNT = link_prim_path("torso_link")

# Depth normalisation, shared by the observation terms and the noise model (which
# needs them to express its dropout value in normalised units).
_DEPTH_OFFSET = 0.75
_DEPTH_SCALE = 0.5

# A ray that hits nothing reports exactly ``max_distance`` (depth_clipping_behavior
# "max").  The observation must treat that sentinel as "no reading" rather than as a
# real surface at max range, so its cut-off sits this far below the sensor's.
_RANGE_SENTINEL_MARGIN_M = 0.1


class _CameraSpec(NamedTuple):
    """Mounting, optics and range for one depth camera.

    ``focal_length_mm`` and the field of view come from the hardware datasheet.  The
    ray pattern is built from an intrinsic matrix derived from the focal length and the
    aperture, so only their RATIO matters -- the absolute unit cancels, and quoting the
    datasheet's mm keeps the number traceable even though IsaacLab documents these
    fields in cm.  The diagonal FOV is not set directly; it follows from H and V.

    The datasheet FOV is the PHYSICAL cone.  Rays are cast over the smaller
    :attr:`effective_fov_h_deg` / :attr:`effective_fov_v_deg`, since depth quality
    falls off towards the frame edge -- see :attr:`fov_crop_deg`.
    """

    pos: tuple[float, float, float]
    """Offset from ``torso_link``, in its frame."""
    pitch_deg: float
    """Downward pitch of the camera's view."""
    focal_length_mm: float
    """Lens focal length, from the datasheet."""
    fov_h_deg: float
    """Horizontal field of view of the physical lens."""
    fov_v_deg: float
    """Vertical field of view of the physical lens."""
    fov_crop_deg: float
    """Degrees discarded from EACH edge, so an axis loses twice this.

    Real depth modules degrade towards the frame border (falling illumination, worse
    calibration, more stereo/ToF dropout), so the outer band is not worth handing to
    the policy.  Cropping keeps the pixel count and redistributes the same rays over
    the trustworthy cone, which also sharpens the angular resolution.
    """
    min_distance_m: float
    """Datasheet minimum range; nearer readings are treated as invalid."""
    max_distance_m: float
    """Range beyond which readings are clipped to the sentinel."""

    @property
    def effective_fov_h_deg(self) -> float:
        """Horizontal FOV actually sampled, after cropping both edges."""
        return self.fov_h_deg - 2.0 * self.fov_crop_deg

    @property
    def effective_fov_v_deg(self) -> float:
        """Vertical FOV actually sampled, after cropping both edges."""
        return self.fov_v_deg - 2.0 * self.fov_crop_deg


# Extrinsics (pos / pitch), in the torso_link frame.  The intent is a near field covering
# the ground about to be stepped on and a far field starting where the near one ends.  The
# OPTICS are datasheet values throughout.
#
# _NEAR is the CALIBRATED lower-camera mount, taken from legged_locomotion_rl branch
# nav-simple-lidar (make_depth_camera in
# .../config/common/depth_student_cfg.py, cross-checked against depth_config.yml, the
# g1_27dof.xml depth_camera_site and build_models.py DEPTH_CAMERA_POS, which are all kept
# in lock-step there).  Its authored values are pos (0.106799, 0, 0.153426), 65 deg pitch
# down.
#
# THE FRAME IS NOT OURS -- a 54 mm correction is applied.  That pose is authored in the
# waist-yaw frame AT THE PELVIS ORIGIN: the body they call ``torso_link`` is the waist-yaw
# link (it carries waist_yaw_joint, is a child of pelvis with no offset, and their real
# upper torso is a separate ``torso_upper_link``).  Our naming differs -- our
# ``torso_link`` IS the upper torso:
#
#     pelvis_link
#      \_ waist_yaw_link    (0, 0, 0)                <- their "torso_link"
#          \_ waist_roll_link (-0.0039635, 0, 0.035)
#              \_ torso_link  (0, 0, 0.019)          <- their "torso_upper_link", our mount
#
# so our mount sits at (-0.0039635, 0, 0.054) from theirs (verified identical in the
# training USD payloads/base.usda and transfer/sim/robots/g1/g1_29j.xml; the logo_link geom
# at (0.0039635, 0, -0.054) in our torso frame is the same offset inverted).  Subtracting
# it gives the values below.  Taking the published numbers verbatim would sit the camera
# 54 mm too high and 4 mm too far forward.
#
# The OPTICS are deliberately NOT taken from that reference: its calibrated intrinsics
# (30x26, fx 25.9928, fy 27.8703, cx 14.9936, cy 12.9966) describe a ~60 x 50 deg frustum,
# i.e. a much narrower lens than the wide module below.  Only the MOUNT transfers.
#
# _FAR is still a PLACEHOLDER -- no calibrated pose exists for the top camera yet, so its
# offset and pitch need visual confirmation against a rendered frame
# (scripts/rsl_rl/verify_depth_self_scan.py --viz kit).
_NEAR = _CameraSpec(
    pos=(0.1107625, 0.0, 0.0994260),
    pitch_deg=65.0,
    focal_length_mm=2.2,
    fov_h_deg=110.0,     # physical 110 x 80 -> 117.8 deg diagonal
    fov_v_deg=80.0,
    fov_crop_deg=5.0,    # sampled 100 x 70 -> 108.2 deg diagonal
    min_distance_m=0.10,
    max_distance_m=8.0,
)
_FAR = _CameraSpec(
    pos=(0.06, 0.0, 0.38),
    pitch_deg=35.0,
    focal_length_mm=4.0,
    fov_h_deg=80.0,      # physical 80 x 52 -> 88.3 deg diagonal
    fov_v_deg=52.0,
    fov_crop_deg=5.0,    # sampled 70 x 42 -> 77.2 deg diagonal
    min_distance_m=0.15,
    max_distance_m=12.0,
)


def _aperture(focal_length: float, fov_deg: float) -> float:
    """Sensor aperture that yields ``fov_deg`` at ``focal_length``.

    Args:
        focal_length: Lens focal length, in any unit.
        fov_deg: Desired full field of view for that axis.

    Returns:
        The aperture, in the same unit as ``focal_length``.
    """
    return 2.0 * focal_length * math.tan(math.radians(fov_deg) / 2.0)


def _pitch_down_quat_xyzw(degrees: float) -> tuple[float, float, float, float]:
    """Rotation about the mount's y-axis, as ``(x, y, z, w)``.

    ``OffsetCfg.rot`` takes ``(x, y, z, w)``, unlike the ``(w, x, y, z)`` used
    elsewhere in IsaacLab.  Positive ``degrees`` pitches the camera's view downwards,
    matching the legged_locomotion_rl reference (whose 65 deg pitch was
    ``(0, 0.537300, 0, 0.843391)``).

    Args:
        degrees: Downward pitch angle.

    Returns:
        The quaternion as ``(x, y, z, w)``.
    """
    half = math.radians(degrees) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def _self_scan_targets() -> list[MultiMeshRayCasterCameraCfg.RaycastTargetCfg]:
    """Ray-cast targets for the robot's own body -- one per moving rigid body.

    ``track_mesh_transforms`` makes the sensor re-read each link's pose every update,
    so the limbs move in the depth image.  ``is_shared`` lets the parser read each
    mesh once and reference it across environments, which matters at 4096 envs.

    Returns:
        One target per self-scan prim path.
    """
    return [
        MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
            prim_expr=path,
            track_mesh_transforms=True,
            is_shared=True,
        )
        for path in self_scan_prim_paths()
    ]


def _make_depth_camera(spec: _CameraSpec) -> MultiMeshRayCasterCameraCfg:
    """Build one torso-mounted, self-scanning depth camera.

    Args:
        spec: Mounting, optics and range for this camera.

    Returns:
        The configured sensor.
    """
    return MultiMeshRayCasterCameraCfg(
        prim_path=_CAMERA_MOUNT,
        update_period=1.0 / _DEPTH_FPS,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=spec.pos,
            rot=_pitch_down_quat_xyzw(spec.pitch_deg),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        max_distance=spec.max_distance_m,
        depth_clipping_behavior="max",
        # Apertures come from the EFFECTIVE (cropped) FOV: the same 30x26 rays are
        # spread over the trustworthy cone instead of out to the degraded border.
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=spec.focal_length_mm,
            horizontal_aperture=_aperture(spec.focal_length_mm, spec.effective_fov_h_deg),
            vertical_aperture=_aperture(spec.focal_length_mm, spec.effective_fov_v_deg),
            width=_DEPTH_W,
            height=_DEPTH_H,
        ),
        # A plain string entry becomes a target with track_mesh_transforms disabled,
        # which is what we want for the static terrain mesh.
        mesh_prim_paths=["/World/ground"] + _self_scan_targets(),
        debug_vis=False,
    )


def _depth_obs_params(spec: _CameraSpec) -> dict:
    """Observation-term parameters matching one camera's range.

    Args:
        spec: The camera this observation reads.

    Returns:
        Keyword parameters for :func:`mdp.depth_image` / :func:`mdp.depth_image_4d`.
    """
    return {
        "offset": _DEPTH_OFFSET,
        "scale": _DEPTH_SCALE,
        "min_dist": spec.min_distance_m,
        "max_dist": spec.max_distance_m - _RANGE_SENTINEL_MARGIN_M,
    }


def _make_depth_noise() -> DepthNoiseCfg:
    """Noise parameters shared by both cameras.

    ``offset`` / ``scale`` must match the observation terms so the dropout value
    lands in the same normalised units as the rest of the image.

    Returns:
        The noise configuration.
    """
    return DepthNoiseCfg(
        max_plane_deviation=0.05,
        dropout_prob=0.02,
        dropout_val=0.0,
        height=_DEPTH_H,
        width=_DEPTH_W,
        offset=_DEPTH_OFFSET,
        scale=_DEPTH_SCALE,
    )




# ---------------------------------------------------------------------------
# Scene / observations / events
# ---------------------------------------------------------------------------

@configclass
class G1TeacherTerrainDepthSceneCfg(G1TeacherTerrainSceneCfg):
    """Clip-library terrain scene + two self-scanning depth cameras.

    The inherited ``height_scanner`` is untouched.
    """

    depth_camera_near: MultiMeshRayCasterCameraCfg = _make_depth_camera(_NEAR)
    depth_camera_far: MultiMeshRayCasterCameraCfg = _make_depth_camera(_FAR)


@configclass
class G1TeacherTerrainDepthObservationsCfg(G1TeacherObservationsCfg):
    """Teacher observations where DEPTH REPLACES the height scan in the policy group.

    The policy sees terrain only through the cameras -- that is the point, since the
    hardware has cameras and not a heightmap.  The critic keeps its height scan: it is
    privileged, never deployed, so a complete terrain description there is free and gives
    the value function a stable target while the policy works the harder problem.  The
    ``height_scanner`` sensor therefore stays in the scene to feed it.

    To strip the scan entirely, also set ``observations.critic.height_scan`` and
    ``scene.height_scanner`` to None -- and drop the inherited
    ``scene.height_scanner.debug_vis = True`` in the play variant, which would otherwise
    raise on None.

    Only the ``policy`` group is overridden here.  The proprioceptive history on the
    inherited terms, the ``reference`` encoder group and the privileged ``critic`` all
    come straight from :class:`G1TeacherObservationsCfg`.
    """

    @configclass
    class PolicyCfg(G1TeacherObservationsCfg.PolicyCfg):
        # Depth is exteroception, so it is NOT history-stacked -- same call as the
        # height scan it replaces (the inherited proprioception terms keep their
        # PROPRIO_HISTORY_LENGTH stacking).
        # Noise is applied inside mdp.depth_image (driven by the resample_depth_noise
        # event), NOT via the term's `noise` field -- see mdp/sensors/noise.py.
        depth_near = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_near"), **_depth_obs_params(_NEAR)},
            clip=(-2.0, 2.0),
        )
        depth_far = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_far"), **_depth_obs_params(_FAR)},
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            super().__post_init__()
            self.height_scan = None

    policy: PolicyCfg = PolicyCfg()


# Mount tolerance and calibration error, from legged_locomotion_rl's PoseErrorCfg:
# +-1 cm per axis on position, +-0.03 rad (~1.7 deg) per axis on orientation.
_EXTRINSICS_POS_RANGE = (0.01, 0.01, 0.01)
_EXTRINSICS_ROT_RANGE = (0.03, 0.03, 0.03)


@configclass
class G1TeacherTerrainDepthEventsCfg(G1TeacherEventsCfg):
    """Teacher events + per-camera extrinsics, intrinsics and noise DR on reset."""

    randomize_camera_extrinsics_near = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera_near"),
            "pos_range": _EXTRINSICS_POS_RANGE,
            "rot_range": _EXTRINSICS_ROT_RANGE,
            "distribution": "uniform",
        },
    )
    randomize_camera_extrinsics_far = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera_far"),
            "pos_range": _EXTRINSICS_POS_RANGE,
            "rot_range": _EXTRINSICS_ROT_RANGE,
            "distribution": "uniform",
        },
    )

    resample_depth_noise_near = EventTerm(
        func=mdp.resample_depth_noise,
        mode="reset",
        params={"sensor_cfg": SceneEntityCfg("depth_camera_near"), "noise_cfg": _make_depth_noise()},
    )
    resample_depth_noise_far = EventTerm(
        func=mdp.resample_depth_noise,
        mode="reset",
        params={"sensor_cfg": SceneEntityCfg("depth_camera_far"), "noise_cfg": _make_depth_noise()},
    )
    randomize_camera_intrinsics_near = EventTerm(
        func=mdp.randomize_camera_intrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera_near"),
            "focal_length_distribution_params": (0.98, 1.02),
            "principal_point_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_camera_intrinsics_far = EventTerm(
        func=mdp.randomize_camera_intrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera_far"),
            "focal_length_distribution_params": (0.98, 1.02),
            "principal_point_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# ---------------------------------------------------------------------------
# Env classes
# ---------------------------------------------------------------------------

@configclass
class G1TeacherTerrainDepthEnvCfg(G1TeacherTerrainBodyEnvCfg):
    """Terrain teacher + two self-scanning depth cameras, flattened into the policy.

    Extends the BODY-tracking terrain teacher, so depth defaults to body-link tracking and
    the depth-vs-height-scan comparison runs against ``G1-teacher-clf-terrain-body`` with
    the exteroception as the only variable.  Joint mode stays reachable per-run with
    ``env.commands.traj_ref.tracking_mode=joint``; both modes read the SAME widened clip
    library (joint mode simply ignores the ``@root`` channels), which is what keeps that
    A/B controlled too.
    """

    scene: G1TeacherTerrainDepthSceneCfg = G1TeacherTerrainDepthSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: G1TeacherTerrainDepthObservationsCfg = G1TeacherTerrainDepthObservationsCfg()
    events: G1TeacherTerrainDepthEventsCfg = G1TeacherTerrainDepthEventsCfg()


@configclass
class G1TeacherTerrainDepthImageObservationsCfg(G1TeacherTerrainDepthObservationsCfg):
    """Moves the depth out of the policy group into per-camera image groups."""

    @configclass
    class PolicyCfg(G1TeacherTerrainDepthObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.depth_near = None
            self.depth_far = None

    @configclass
    class DepthNearImageCfg(ObsGroup):
        depth_near_image = ObsTerm(
            func=mdp.depth_image_4d,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera_near"),
                "height": _DEPTH_H,
                "width": _DEPTH_W,
                **_depth_obs_params(_NEAR),
            },
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            # Single-term group: concatenate_terms=True is a no-op (a cat of one
            # tensor) and preserves the (B, 1, H, W) shape.  concatenate_terms=False
            # would hand back a TensorDict, which rsl_rl's NaN check and our
            # train.py debug print do not dispatch on.

    @configclass
    class DepthFarImageCfg(ObsGroup):
        depth_far_image = ObsTerm(
            func=mdp.depth_image_4d,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera_far"),
                "height": _DEPTH_H,
                "width": _DEPTH_W,
                **_depth_obs_params(_FAR),
            },
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()
    depth_near_image: DepthNearImageCfg = DepthNearImageCfg()
    depth_far_image: DepthFarImageCfg = DepthFarImageCfg()


@configclass
class G1TeacherTerrainDepthImageEnvCfg(G1TeacherTerrainDepthEnvCfg):
    """Terrain teacher + depth as image-shaped observations in their own groups."""

    observations: G1TeacherTerrainDepthImageObservationsCfg = (
        G1TeacherTerrainDepthImageObservationsCfg()
    )


# ---------------------------------------------------------------------------
# PLAY variants -- inherit the terrain teacher's play trims, then add depth ones.
# ---------------------------------------------------------------------------

def _apply_depth_play_trims(cfg) -> None:
    """Disable depth randomisation and draw both cameras' rays.

    Args:
        cfg: The play env cfg to trim, modified in place.
    """
    cfg.events.resample_depth_noise_near = None
    cfg.events.resample_depth_noise_far = None
    cfg.events.randomize_camera_intrinsics_near = None
    cfg.events.randomize_camera_intrinsics_far = None
    cfg.events.randomize_camera_extrinsics_near = None
    cfg.events.randomize_camera_extrinsics_far = None
    cfg.scene.depth_camera_near.debug_vis = True
    cfg.scene.depth_camera_far.debug_vis = True


@configclass
class G1TeacherTerrainDepthEnvCfgPlay(G1TeacherTerrainBodyEnvCfgPlay):
    """Play variant of :class:`G1TeacherTerrainDepthEnvCfg` (body tracking, as above)."""

    scene: G1TeacherTerrainDepthSceneCfg = G1TeacherTerrainDepthSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: G1TeacherTerrainDepthObservationsCfg = G1TeacherTerrainDepthObservationsCfg()
    events: G1TeacherTerrainDepthEventsCfg = G1TeacherTerrainDepthEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_depth_play_trims(self)


@configclass
class G1TeacherTerrainDepthImageEnvCfgPlay(G1TeacherTerrainDepthEnvCfgPlay):
    """Play variant of :class:`G1TeacherTerrainDepthImageEnvCfg`."""

    observations: G1TeacherTerrainDepthImageObservationsCfg = (
        G1TeacherTerrainDepthImageObservationsCfg()
    )


# ---------------------------------------------------------------------------
# Contract, without a simulator
# ---------------------------------------------------------------------------

def depth_camera_specs_from_cfg(
    observations: G1TeacherTerrainDepthObservationsCfg | None = None,
    scene: G1TeacherTerrainDepthSceneCfg | None = None,
    group: str = "policy",
) -> list[dict]:
    """The depth contract for THIS task's real cfgs, with no simulator running.

    Instantiates the actual observation and scene configurations and runs the production
    producer (:func:`mdp.depth_camera_specs`) over them, so an off-sim tool sees the same
    camera numbers, term names and clips the training task declares -- rather than a
    restatement of them that can silently go stale.  Used by the sim2sim verification
    tools and the contract tests; the deployment path reads the contract out of the ONNX
    metadata instead.

    Args:
        observations: Observation cfg to read the depth terms from (defaults to this
            task's).
        scene: Scene cfg to read the camera sensors from (defaults to this task's).
        group: Observation group whose depth terms to describe.

    Returns:
        One dict per camera, in observation order.
    """
    if observations is None:
        observations = G1TeacherTerrainDepthObservationsCfg()
    if scene is None:
        scene = G1TeacherTerrainDepthSceneCfg(num_envs=1, env_spacing=2.5)
    # Same iteration order the ObservationManager uses (dataclass field order, skipping
    # terms set to None), so the resulting contract is ordered like the real obs vector.
    terms = {n: t for n, t in vars(getattr(observations, group)).items() if isinstance(t, ObsTerm)}
    sensors = {
        n: SimpleNamespace(cfg=s)
        for n, s in vars(scene).items()
        if isinstance(s, MultiMeshRayCasterCameraCfg)
    }
    env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            active_terms={group: list(terms)}, _group_obs_term_cfgs={group: list(terms.values())}
        ),
        scene=SimpleNamespace(sensors=sensors),
    )
    return mdp.depth_camera_specs(env, [group])
