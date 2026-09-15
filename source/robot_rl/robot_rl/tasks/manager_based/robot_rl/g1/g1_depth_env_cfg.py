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
heightmap, so a policy that leans on the scan is not deployable.

**The POLICY reads the NEAR camera only; the GENERATOR reads both.**  The far camera is the
generator's instrument -- it reads the terrain AHEAD, which is what a reference has to be
planned onto -- while the tracker follows a reference that has already been placed and gains
nothing from seeing 12 m out.  Dropping it from the policy group takes roughly half the depth
cost off a 12k-iteration teacher run (``obs`` is 39% of the env step and ~19 ms of that is the
two cameras).  Both cameras stay on the SCENE regardless: ``lazy_sensor_update`` (on by
default) never casts a sensor that no observation term reads, so the far camera costs its
per-env buffers and no rays, and leaving it there keeps its DR terms, the mount test
(``tests/test_camera_mount_matches_urdf.py``) and MuJoCo's extrinsics mirroring describing the
same physical pair the robot carries.  :func:`install_fm_depth_group` is what puts it back in
front of an observation term, and is called wherever the generator runs or is recorded.

Two flavours differing only in the observation SHAPE:

- :class:`G1TeacherTerrainDepthEnvCfg` -- a flattened ``depth_near`` term in the policy group.
- :class:`G1TeacherTerrainDepthImageEnvCfg` -- that camera as an image-shaped observation in
  its own group, for CNN backbones.

The CRITIC keeps its height scan, and the pelvis ``height_scanner`` therefore stays in
the scene.  The critic is privileged and never deployed, so a complete terrain
description there is free and stabilises the value function while the policy works the
harder problem from depth alone.  The scan also still feeds the flow-matching
conditioning by default and the MuJoCo sim2sim path, which are separate entry points.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
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
from robot_rl.tasks.manager_based.robot_rl.terrains.decor import DECOR_PRIM_PATH

from .g1_flat_blind_env_cfg import ACTION_RATE_WEIGHT
from .g1_self_scan import link_prim_path, self_scan_prim_paths
from .g1_teacher_env_cfg import G1TeacherEventsCfg, G1TeacherObservationsCfg
from .g1_teacher_terrain_env_cfg import (
    G1TeacherTerrainBodyEnvCfg,
    G1TeacherTerrainBodyEnvCfgPlay,
    G1TeacherTerrainSceneCfg,
)


def ensure_decor_prim() -> None:
    """Give a decoration-free scene the placeholder decoration prim.

    Every camera built by :func:`_make_depth_camera` lists :data:`DECOR_PRIM_PATH` as a
    ray-cast target, and ``MultiMeshRayCaster`` needs that target to resolve to at least one
    real mesh -- it raises on a path matching nothing, and dies later on ``n_meshes = 0`` for a
    path matching a mesh-less prim. A scene that uses these cameras over a plain
    ``TerrainImporter`` -- the flat-mesh probe scenes in ``verify_depth_self_scan.py`` and
    ``compare_depth_sim2sim.py`` -- therefore has to build one itself.

    Call before ``sim.reset()``, which is when the sensors initialise. Idempotent: a scene
    that already has real decoration is left alone.
    """
    # Imported HERE, not at module scope: the importer transitively pulls in ``pxr``, and
    # this cfg module is imported by hydra BEFORE ``AppLauncher`` runs.  See
    # ``terrains/decor.py``.  By the time this function is called the app is up.
    import isaaclab.sim as sim_utils

    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_importer import (
        create_decor_prim,
    )

    if not sim_utils.is_prim_path_valid(DECOR_PRIM_PATH):
        create_decor_prim(None, DECOR_PRIM_PATH)

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
# SUPERSEDED, 2026-08-19: both poses below are now taken from the AS-BUILT CAD of the real
# mount, so nothing above is doing arithmetic on a published number any more.  The frame
# chain is kept because it says what ``torso_link`` is, and mode_11 moved it: waist_roll
# z 0.035 -> 0.044 and waist_pitch z 0.019 -> 0, so ``torso_link`` now sits at
# (-0.0039635, 0, 0.044) from the pelvis rather than (-0.0039635, 0, 0.054) -- 10 mm lower.
#
# That 10 mm does NOT get added back here, and the plan said it should.  The correction was
# specified when these numbers were assumed to be in the OLD torso frame; they are not.
# mode_11 does not move the torso MESH within its own frame, only where that frame sits on
# the pelvis, so a torso-relative CAD measurement of the real robot is already right in the
# new frame.  Adding 10 mm would put the cameras 10 mm too high.  What legitimately changes
# is the physical height above ground, which drops 10 mm because the old model had the
# torso 10 mm too high.
#
# The pose is the LEFT IMAGER, not the mount centre: a stereo depth camera reports depth in
# its left camera's frame.  Hence the non-zero y (+25 mm near, +60 mm far, i.e. half of each
# 50/120 mm baseline).  ``robot_assets/g1_perceptive/`` carries the same two numbers as the
# mounts' ``*_optical_link``, pinned there against the STL datum, and
# tests/test_camera_mount_matches_urdf.py fails if the two ever disagree.
_NEAR = _CameraSpec(
    pos=(0.107, 0.025, 0.045),
    pitch_deg=65.0,
    focal_length_mm=2.2,
    fov_h_deg=110.0,     # physical 110 x 80 -> 117.8 deg diagonal
    fov_v_deg=80.0,
    fov_crop_deg=5.0,    # sampled 100 x 70 -> 108.2 deg diagonal
    min_distance_m=0.10,
    max_distance_m=8.0,
)
_FAR = _CameraSpec(
    pos=(0.085, 0.060, 0.285),
    pitch_deg=15.0,
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


def _make_depth_camera(
    spec: _CameraSpec, width: int = _DEPTH_W, height: int = _DEPTH_H
) -> MultiMeshRayCasterCameraCfg:
    """Build one torso-mounted, self-scanning depth camera.

    Args:
        spec: Mounting, optics and range for this camera.
        width: Rays across. Only the SAMPLING changes with this -- the apertures below come
            from the effective FOV, so a different resolution spreads the same cone over
            more or fewer rays rather than looking through a different lens.
        height: Rays down.

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
            width=width,
            height=height,
        ),
        # A plain string entry becomes a target with track_mesh_transforms disabled,
        # which is what we want for the static terrain mesh.
        # /World/decor is the VISUAL-ONLY decoration: walls and clutter the cameras must see
        # and the height scan must not. It is a sibling of /World/ground precisely so the
        # scanner -- a RayCaster pointed at that one path -- cannot reach it.
        #
        # LISTING IT MAKES THE PRIM MANDATORY. MultiMeshRayCaster tolerates a target that
        # exists but holds no meshes (it logs "No mesh prims found" and skips), and does NOT
        # tolerate a target that matches nothing -- ``_initialize_warp_meshes`` raises
        # "Failed to find a prim at path expression". So ``ClipTerrainImporter`` creates the
        # Xform even for an undecorated library, and any scene that builds these cameras
        # WITHOUT that importer must create it too: see :data:`DECOR_PRIM_PATH` and the
        # flat-mesh probe scenes in ``verify_depth_self_scan.py`` /
        # ``compare_depth_sim2sim.py``, both of which died on exactly this.
        mesh_prim_paths=["/World/ground", DECOR_PRIM_PATH] + _self_scan_targets(),
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
    """Noise parameters shared by every camera.

    ``offset`` / ``scale`` must match the observation terms so the dropout value
    lands in the same normalised units as the rest of the image. The RESOLUTION is left
    unset on purpose: ``ensure_depth_noiser`` reads it off each camera's own
    ``pattern_cfg``, so one cfg serves cameras that sample at different resolutions and
    nothing here restates a number the sensor already carries.

    Returns:
        The noise configuration.
    """
    return DepthNoiseCfg(
        max_plane_deviation=0.05,
        dropout_prob=0.02,
        dropout_val=0.0,
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

    The policy sees terrain only through the NEAR camera -- that is the point, since the
    hardware has cameras and not a heightmap, and the far one is the generator's (see the
    module docstring).  The critic keeps its height scan: it is
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
        #
        # NEAR ONLY, and the far camera is absent by decision rather than by omission: it
        # reads terrain ahead, which is what the GENERATOR needs to place a reference and
        # what a tracker following an already-placed one does not. The sensor stays on the
        # scene and is simply never cast (lazy_sensor_update), so this costs nothing but the
        # rays. install_fm_depth_group re-observes it for the generator.
        depth_near = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_near"), **_depth_obs_params(_NEAR)},
            clip=(-2.0, 2.0),
        )

        #: The far camera's slot in the POLICY vector, declared and left OFF.
        #:
        #: Declared rather than deleted because a term's position in this class IS its position
        #: in the observation vector, and every tracker checkpoint that exists was trained with
        #: the far camera HERE -- between ``depth_near`` and ``reference_window`` (read off
        #: ``2026-08-20_16-36-10_mode11_decor_plan_v1``'s exported ``observation_terms``).
        #: Re-adding it in a ``__post_init__`` would APPEND it, past ``command_window``, and the
        #: vector would be the right WIDTH with two blocks transposed -- a checkpoint that loads
        #: clean and reads scrambled input, which is the failure mode this repo keeps paying for.
        #:
        #: ``None`` is the default because near-only is the decision for the next teacher
        #: retrain; :func:`enable_policy_far_camera` turns it back on for running an EXISTING
        #: two-camera checkpoint, which is every one on disk today.
        #:
        #: **THE ASYMMETRY IS DELIBERATE: the RL policy reads the NEAR camera alone, and the
        #: FM generator still reads BOTH.** They are different consumers of the same pair --
        #: the policy has to run on a robot and the generator places a reference from what it
        #: can see ahead, which is what the far camera is for. So this default is NOT a
        #: statement that the far camera is unhelpful, and it must not be propagated to the
        #: generator's own exteroception (``extero_source`` / ``add_fm_depth_pair``, which
        #: samples the same physical pair on its own grid).
        #:
        #: Practical consequence for a retrain: a tracker trained under this default has a
        #: DIFFERENT policy observation from every checkpoint on disk, so it is not a drop-in
        #: and its numbers are not comparable to the two-camera ones without re-running the
        #: eval arms.
        #: NOTE the missing type annotation, and it is load-bearing: ``depth_near`` above is an
        #: un-annotated class attribute, and ``configclass`` orders ANNOTATED fields ahead of
        #: un-annotated ones. Written as ``depth_far: ObsTerm | None = None`` this slot lands
        #: BEFORE ``depth_near`` -- measured, and the tracker then reads the far camera as the
        #: near one: right width, transposed blocks, robot on the floor in a second.
        depth_far = None

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

    # 8192, matching the tracker that every result on disk was measured with
    # (`2026-08-20_16-36-10_mode11_decor_plan_v1`, which passed `--num_envs 8192`). The repo
    # said 4096, so the pairing lived only in a shell history. The PLAY cfgs below stay small
    # on purpose -- nothing wants 8192 robots in a viewer.
    scene: G1TeacherTerrainDepthSceneCfg = G1TeacherTerrainDepthSceneCfg(
        num_envs=8192, env_spacing=2.5
    )
    observations: G1TeacherTerrainDepthObservationsCfg = G1TeacherTerrainDepthObservationsCfg()
    events: G1TeacherTerrainDepthEventsCfg = G1TeacherTerrainDepthEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 10x the -0.01 every teacher inherits. Shared with the blind flat teacher through
        # one constant, because a different smoothness cost on each would confound the
        # blind-vs-camera comparison. Raised for the runs queued 2026-08-21; the number is
        # a starting point, not a tuned value.
        self.rewards.action_rate_l2.weight = ACTION_RATE_WEIGHT


@configclass
class G1TeacherTerrainDepthImageObservationsCfg(G1TeacherTerrainDepthObservationsCfg):
    """Moves the depth out of the policy group into its own image-shaped group.

    Follows the flat variant's camera SET exactly -- the near camera alone -- so the two
    depth teachers cannot disagree about how many cameras the policy reads. They differ in
    the observation's SHAPE and in nothing else, which is what makes them an A/B.
    """

    @configclass
    class PolicyCfg(G1TeacherTerrainDepthObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.depth_near = None

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

    policy: PolicyCfg = PolicyCfg()
    depth_near_image: DepthNearImageCfg = DepthNearImageCfg()


@configclass
class G1TeacherTerrainDepthImageEnvCfg(G1TeacherTerrainDepthEnvCfg):
    """Terrain teacher + depth as image-shaped observations in their own groups."""

    observations: G1TeacherTerrainDepthImageObservationsCfg = (
        G1TeacherTerrainDepthImageObservationsCfg()
    )


# ---------------------------------------------------------------------------
# PLAY variants -- inherit the terrain teacher's play trims, then add depth ones.
# ---------------------------------------------------------------------------

def apply_depth_play_trims(cfg) -> None:
    """Disable depth randomisation and draw both cameras' rays.

    Public because the FM tracker's depth play variant
    (``g1_tracker_env_cfg.G1FmTrackerTerrainDepthPlayEnvCfg``) needs the same trims: it
    derives from the depth teacher rather than from the play class below, so it cannot
    inherit them.

    Args:
        cfg: The play env cfg to trim, modified in place.
    """
    # Derived rather than a list of six field names, which went stale the moment a seventh
    # camera event could exist: null every reset term whose function comes from the depth
    # events module, and turn on debug_vis for every camera the SCENE declares.
    for name, term in list(vars(cfg.events).items()):
        func = getattr(term, "func", None)
        if getattr(func, "__module__", "").endswith("mdp.events.depth"):
            setattr(cfg.events, name, None)
    for name, sensor in vars(cfg.scene).items():
        # The generator's own cameras stay invisible: they sit on the same mounts as the
        # policy's, so drawing both just doubles every frustum in the viewport.
        if isinstance(sensor, MultiMeshRayCasterCameraCfg) and not name.startswith(_FM_PREFIX):
            sensor.debug_vis = True


#: Name prefix for the generator's own cameras, so they are distinguishable from the
#: policy's everywhere a scene is walked by name.
_FM_PREFIX = "fm_"

#: The cameras the POLICY observes, in observation order, with the spec each was built from.
#:
#: NEAR ONLY -- the far camera reads terrain ahead, which is the generator's job (module
#: docstring). Kept apart from :data:`_DEPTH_CAMERAS` rather than folded into it because the
#: two are no longer the same list, and the invariant that matters is that this one is a
#: strict SUBSET: there is ONE physical camera pair, and the policy reads part of it.
#: Anything reasoning about the pair itself wants the full set; only the policy's own
#: observation vector wants this one.
_POLICY_CAMERAS: tuple[tuple[str, _CameraSpec], ...] = (
    ("depth_camera_near", _NEAR),
)

def enable_policy_far_camera(cfg) -> None:
    """Put the FAR camera back into the POLICY observation, in its historical slot.

    The policy reads the near camera alone by decision (see :data:`_POLICY_CAMERAS`), taken
    ahead of a teacher retrain that has not happened.  So every tracker checkpoint on disk was
    trained on BOTH -- 3002 inputs against the 2222 a near-only task builds -- and cannot run
    on a near-only task at all.  This restores the term for those runs.

    **It fills the declared slot rather than adding one**, so the far block lands between
    ``depth_near`` and ``reference_window`` exactly as the trained vector has it.

    Call it BEFORE ``install_fm_depth_group``: the generator's group is built from the scene's
    cameras and is unaffected either way, but the ordering of the two calls is what a reader
    checks, so keep it the same everywhere.

    Args:
        cfg: A composed env cfg whose ``observations.policy`` carries the declared slot.

    Raises:
        AttributeError: If the policy group has no ``depth_far`` slot -- which would mean the
            declaration was deleted and the ordering guarantee with it.
    """
    policy = cfg.observations.policy
    if not hasattr(policy, "depth_far"):
        raise AttributeError(
            "observations.policy has no `depth_far` slot to fill. It is declared (as None) so "
            "that enabling it preserves the trained term ORDER; adding it here instead would "
            "append it past command_window and silently transpose the vector.")
    policy.depth_far = ObsTerm(
        func=mdp.depth_image,
        params={"sensor_cfg": SceneEntityCfg("depth_camera_far"), **_depth_obs_params(_FAR)},
        clip=(-2.0, 2.0),
    )


#: Every depth camera the SCENE carries, in observation order -- the set the GENERATOR
#: conditions on. What :func:`install_fm_depth_group` observes, mirrors and extends the DR
#: over, and what :func:`assert_fm_depth_group_covers_every_camera` checks against.
_DEPTH_CAMERAS: tuple[tuple[str, _CameraSpec], ...] = (
    ("depth_camera_near", _NEAR),
    ("depth_camera_far", _FAR),
)

#: The generator group's term names, aligned with :data:`_DEPTH_CAMERAS`.
_FM_DEPTH_TERMS: tuple[str, ...] = ("fm_depth_near", "fm_depth_far")

#: The per-camera DR terms one draw must cover, as ``format`` templates on the camera suffix.
_DEPTH_DR_TERMS: tuple[str, ...] = (
    "randomize_camera_extrinsics_{}",
    "randomize_camera_intrinsics_{}",
    "resample_depth_noise_{}",
)


@configclass
class G1FmDepthObservationsCfg(ObsGroup):
    """The GENERATOR's depth, in its own group so the policy's vector is untouched.

    Covers BOTH cameras, always. The ``sensor_cfg`` entries below name the POLICY's sensors,
    which is the normal case -- the generator samples the same 30x26 grid and simply reads
    them. :func:`install_fm_depth_group` rewrites them to the ``fm_``-prefixed duplicates
    when the generator samples on its own grid.

    Flat per camera (``mdp.depth_image``), because ``depth_obs_stack`` -- which is what the
    recorder writes and what ``FmTrackingCommand`` reads -- stacks flat rows; the FM's
    convolutional encoder folds them back using the geometry from its own camera contract.

    The normalization comes from the SAME ``_depth_obs_params`` as the policy's term, so the
    two readings of one lens cannot end up on different affine transforms. When the group
    reads the policy's own sensor, one physical camera is read by two observation terms in
    one step -- ``DepthNoiser`` shares the frame's draw so they see the same image, which is
    what the hardware does and what ``--dump_replay`` needs.
    """

    fm_depth_near = ObsTerm(
        func=mdp.depth_image,
        params={"sensor_cfg": SceneEntityCfg("depth_camera_near"), **_depth_obs_params(_NEAR)},
        clip=(-2.0, 2.0),
    )
    fm_depth_far = ObsTerm(
        func=mdp.depth_image,
        params={"sensor_cfg": SceneEntityCfg("depth_camera_far"), **_depth_obs_params(_FAR)},
        clip=(-2.0, 2.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # Corruption ON, like the policy's group: the generator conditions on a real noisy
        # camera at deploy, so it must train and record against one.
        self.enable_corruption = True


def assert_fm_depth_group_covers_every_camera(
    cfg, expect: Collection[str] | None = None
) -> None:
    """Refuse a generator group that does not name every depth camera on the scene.

    THE FAILURE THIS EXISTS FOR IS SILENT, and nothing downstream can catch it. The policy
    group holds the near camera alone; with no ``fm_depth`` group, ``mdp.fm_depth_groups``
    returns ``None`` = "every group", so the recorder would capture ONE camera, stamp a
    one-camera ``depth_camera_spec`` into the recording, train a generator on it, export a
    matching ``fm.onnx``, and pass the deploy-time agreement check -- every artifact
    self-consistent about a half-blind generator, and indistinguishable from a deliberate
    one-camera design.

    ``expect`` is the ONE way to ask for less, and it exists because a near-only generator is
    a real artifact: the camera ablation drops the far camera at dataset build, so its
    checkpoint conditions on one sensor row and an env serving two is refused downstream by
    ``FmTrackingCommand._check_extero_shape``. What keeps this from being the hole the
    paragraph above describes is WHERE the subset comes from -- the CHECKPOINT'S OWN
    ``extero_spec``, read by ``read_fm_extero_meta``, never a flag and never a default. A
    generator that names both cameras still gets both, and a group that quietly lost one is
    still a fault.

    Args:
        cfg: A depth env cfg carrying ``scene`` and ``observations``.
        expect: The camera set the group is REQUIRED to name, when a checkpoint declares a
            subset. ``None`` (the default) means every camera on the scene.

    Raises:
        ValueError: If the group is absent, names a different set of physical cameras than
            it is required to, or if ``expect`` names a camera the scene does not carry.
    """
    group = getattr(cfg.observations, mdp.FM_DEPTH_OBS_GROUP, None)
    if group is None:
        raise ValueError(
            f"no '{mdp.FM_DEPTH_OBS_GROUP}' observation group; the generator would fall back "
            "to 'every group', which on this task is the policy's NEAR camera alone. Call "
            "install_fm_depth_group(env_cfg) before gym.make."
        )
    named = {
        term.params["sensor_cfg"].name
        for term in vars(group).values()
        if isinstance(term, ObsTerm) and "sensor_cfg" in (term.params or {})
    }
    observed = {name.removeprefix(_FM_PREFIX) for name in named}
    on_scene = {
        name for name, sensor in vars(cfg.scene).items()
        if isinstance(sensor, MultiMeshRayCasterCameraCfg) and not name.startswith(_FM_PREFIX)
    }
    if expect is None:
        if observed != on_scene:
            raise ValueError(
                f"the generator's '{mdp.FM_DEPTH_OBS_GROUP}' group observes "
                f"{sorted(observed)} but the scene carries {sorted(on_scene)}. The generator "
                "must see every camera the robot has -- a missing one is a half-blind "
                "generator whose recording, checkpoint and fm.onnx all agree with each other."
            )
        return
    want = {str(name) for name in expect}
    off_scene = sorted(want - on_scene)
    if off_scene:
        raise ValueError(
            f"a checkpoint asks the generator for cameras {off_scene}, which this scene does "
            f"not carry (it has {sorted(on_scene)}). The artifact and the task describe "
            "different robots; nothing here can reconcile them."
        )
    if observed != want:
        raise ValueError(
            f"the generator's '{mdp.FM_DEPTH_OBS_GROUP}' group observes {sorted(observed)} "
            f"but its checkpoint declares {sorted(want)}. The subset is read off the "
            "checkpoint, so this is a wiring fault rather than a choice."
        )


def _clear_fm_depth_sampling(cfg) -> None:
    """Remove any generator-owned camera duplicates, restoring the single-sensor DR terms.

    Makes :func:`install_fm_depth_group` re-entrant, which it has to be: a task installs the
    group at the policy's resolution in ``__post_init__`` and ``play.py`` then re-installs it
    at whatever resolution the checkpoint states. Without this a split-then-shared call would
    leave the first call's ``fm_`` cameras on the scene -- cast every step, read by nothing.

    Args:
        cfg: A depth env cfg (``scene`` and ``events``), modified in place.
    """
    for name, _ in _DEPTH_CAMERAS:
        fm_name = f"{_FM_PREFIX}{name}"
        if hasattr(cfg.scene, fm_name):
            delattr(cfg.scene, fm_name)
        suffix = name.rsplit("_", 1)[-1]                      # "near" / "far"
        for template in _DEPTH_DR_TERMS:
            term = getattr(cfg.events, template.format(suffix), None)
            if term is None or "sensor_cfgs" not in (term.params or {}):
                continue
            term.params.pop("sensor_cfgs")
            term.params["sensor_cfg"] = SceneEntityCfg(name)


def install_fm_depth_group(cfg, width: int | None = None, height: int | None = None,
                           cameras: Sequence[str] | None = None) -> str:
    """Give the GENERATOR its observation group, covering EVERY depth camera on the scene.

    ALWAYS installs the group, and by default the group names both cameras. That is the whole
    point: the policy reads the near camera alone, so "no group" no longer means "the
    generator and the policy see the same thing" -- it means the generator sees half the
    robot's cameras, silently (see :func:`assert_fm_depth_group_covers_every_camera`). Call
    this wherever the generator runs or is recorded: the FM tracker and fine-tune tasks do it
    in ``__post_init__``, ``record_data.py`` / ``play.py`` / ``augment_registration.py`` do it
    before ``gym.make``.

    ``cameras`` is the near-only arm and the ONLY way to install less than everything. Pass
    the sensor names a CHECKPOINT declares (``read_fm_extero_meta(...)["cameras"]``), never a
    list someone typed: the camera ablation drops the far camera at DATASET build, so its
    generator conditions on one sensor row and an env serving two rows is refused by
    ``FmTrackingCommand._check_extero_shape`` -- correctly, and three stages downstream of
    anything that says why. A dropped camera also stops being ray-cast (``lazy_sensor_update``
    casts only what an observation term reads), so the near-only arm really is near-only
    rather than paying for a view it discards.

    Two arms, and only the SAMPLING differs between them:

    * ``width x height`` equal to the policy's (the normal case, and the default when they
      are left unset) -- the group's terms point at the EXISTING scene sensors. No second
      camera, no second ray cast, DR untouched. One physical camera is then read by two
      observation terms in one step, and ``DepthNoiser`` shares the frame's draw between them
      so they get the same image rather than two independent dropout patterns.
    * anything else -- a duplicate ``fm_``-prefixed camera per lens, built from the SAME
      ``_CameraSpec`` (same mount, apertures, range and normalization; only the ray grid
      differs), and the three per-camera DR terms EXTENDED to ``sensor_cfgs`` lists so one
      draw moves both samplings. Everything physical is shared by CONSTRUCTION rather than
      checked afterwards.

    Re-entrant: calling it again replaces the group and tears down any duplicates the
    previous call built.

    Args:
        cfg: A depth env cfg (needs ``scene``, ``observations`` and ``events``), modified in
            place.
        width: Rays across for the generator's cameras. ``None`` = the policy's.
        height: Rays down. ``None`` = the policy's.
        cameras: Scene sensor names the generator conditions on, as a checkpoint declares
            them. ``None`` (the default) = every camera on the scene.

    Returns:
        The observation group name that was installed.

    Raises:
        ValueError: If ``cfg`` carries no depth cameras to observe, or if ``cameras`` is
            empty or names one the scene does not have.

    TODO(one-sampling): the SPLIT arm is a DEVELOPMENT feature -- it lets the generator's grid
    change without retraining the teacher. The long-term intent is that the two match, and
    then the ``fm_`` cameras, MuJoCo's second cast and
    ``GenTrackerPolicy.depth_sampling_is_split`` all go, leaving this function as the shared
    arm alone. Hardware never supported the split: ``export_sensor_configs.py`` raises rather
    than emit configs covering only the tracker.
    """
    missing = [name for name, _ in _DEPTH_CAMERAS if getattr(cfg.scene, name, None) is None]
    if missing:
        raise ValueError(
            f"install_fm_depth_group needs the scene's depth cameras {missing}; this looks "
            "like a task without cameras (the height-scan teacher/tracker)."
        )
    w = _DEPTH_W if width is None else int(width)
    h = _DEPTH_H if height is None else int(height)
    split = (w, h) != (_DEPTH_W, _DEPTH_H)

    on_scene = [name for name, _ in _DEPTH_CAMERAS]
    if cameras is None:
        keep = set(on_scene)
    else:
        keep = {str(name) for name in cameras}
        if not keep:
            raise ValueError(
                "install_fm_depth_group(cameras=[]) would leave the generator with no "
                "exteroception at all. Pass None for every camera on the scene.")
        unknown = sorted(keep - set(on_scene))
        if unknown:
            raise ValueError(
                f"install_fm_depth_group was asked for cameras {unknown}, which this scene "
                f"does not carry (it has {on_scene}). Names come from a checkpoint's "
                "extero_spec 'sensor' field and must be scene sensor names.")

    _clear_fm_depth_sampling(cfg)

    group = G1FmDepthObservationsCfg()
    for term_name, (camera, spec) in zip(_FM_DEPTH_TERMS, _DEPTH_CAMERAS, strict=True):
        if camera not in keep:
            # None, not delattr: the ObservationManager skips a None term, and leaving the
            # ATTRIBUTE in place keeps the declared field order intact for the cameras that
            # remain -- the same reason `depth_far` is declared as None on the policy group
            # rather than added later, which is how two 780-wide blocks once got transposed.
            setattr(group, term_name, None)
            continue
        if split:
            setattr(cfg.scene, f"{_FM_PREFIX}{camera}",
                    _make_depth_camera(spec, width=w, height=h))
            getattr(group, term_name).params["sensor_cfg"] = SceneEntityCfg(
                f"{_FM_PREFIX}{camera}"
            )
    setattr(cfg.observations, mdp.FM_DEPTH_OBS_GROUP, group)

    if split:
        # One draw per physical camera, now covering both of its samplings. Only the kept
        # ones have a second sampling to cover; a dropped camera keeps its single-sensor DR
        # term, which `_clear_fm_depth_sampling` has just restored.
        for camera, _ in _DEPTH_CAMERAS:
            if camera not in keep:
                continue
            suffix = camera.rsplit("_", 1)[-1]                # "near" / "far"
            shared = [SceneEntityCfg(camera), SceneEntityCfg(f"{_FM_PREFIX}{camera}")]
            for template in _DEPTH_DR_TERMS:
                term = getattr(cfg.events, template.format(suffix), None)
                if term is None:
                    continue                                  # trimmed away (a play cfg)
                term.params.pop("sensor_cfg", None)
                term.params["sensor_cfgs"] = shared

    assert_fm_depth_group_covers_every_camera(cfg, expect=keep)

    rays = w * h
    kept = [name for name in on_scene if name in keep]
    if split:
        print(f"[INFO] fm depth cameras added at {w}x{h} ({rays} px/camera, "
              f"{len(kept) * rays} total) in obs group '{mdp.FM_DEPTH_OBS_GROUP}' over "
              f"{kept}; the policy keeps {_DEPTH_W}x{_DEPTH_H}. Ray cost scales with the "
              f"pixel count, so each is {rays / (_DEPTH_W * _DEPTH_H):.2g}x a policy "
              "camera's.")
    else:
        print(f"[INFO] obs group '{mdp.FM_DEPTH_OBS_GROUP}' installed over the scene's own "
              f"{w}x{h} cameras {kept}; no second sampling, no second ray cast. The policy "
              f"reads {[n for n, _ in _POLICY_CAMERAS]}.")
    if len(kept) != len(on_scene):
        dropped = [name for name in on_scene if name not in keep]
        print(f"[INFO] the generator conditions on {len(kept)} of this scene's "
              f"{len(on_scene)} cameras; {dropped} is NOT observed and therefore not "
              "ray-cast. This is the camera ablation, and it is legitimate only because the "
              "checkpoint itself declares the subset.")
    return mdp.FM_DEPTH_OBS_GROUP


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
        apply_depth_play_trims(self)


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
