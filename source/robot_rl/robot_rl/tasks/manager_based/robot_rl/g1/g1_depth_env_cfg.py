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

Two flavours, mirroring the height-scan teacher's observation layout:

- :class:`G1TeacherTerrainDepthEnvCfg` adds flattened ``depth_near`` / ``depth_far``
  terms to the policy group.
- :class:`G1TeacherTerrainDepthImageEnvCfg` instead exposes each camera as an
  image-shaped observation in its own group, for CNN backbones.

The inherited pelvis ``height_scanner`` is deliberately left in place: it still
feeds the policy, the critic, the flow-matching conditioning
(:mod:`robot_rl.flow_matching.dataset` requires it) and the MuJoCo sim2sim path, so
depth is additive until a retrain replaces it.
"""

from __future__ import annotations

import math

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
    G1TeacherTerrainEnvCfg,
    G1TeacherTerrainEnvCfgPlay,
    G1TeacherTerrainSceneCfg,
)

# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

# Pinhole intrinsics (3x3 K, flattened) and resolution, from the
# legged_locomotion_rl reference.  Shared by both cameras.
_DEPTH_K = [25.9928, 0.0, 14.9936, 0.0, 27.8703, 12.9966, 0.0, 0.0, 1.0]
_DEPTH_W, _DEPTH_H = 30, 26

# Both cameras mount on torso_link -- a real physics body (child of waist_pitch_joint,
# parent of the shoulder joints), so it can host a sensor view.  head_link CANNOT: it,
# logo_link and waist_support_link are instanceable visual-only leaves welded under
# torso_link with no joints of their own, and all three carry the same
# (0.0039635, 0, -0.054) translate -- a URDF-conversion artifact -- so head_link's
# Xform is not even a reliable "head height" reference.
_CAMERA_MOUNT = link_prim_path("torso_link")

# PLACEHOLDER extrinsics, in the torso_link frame.  The intent is a near field
# covering the ground about to be stepped on and a far field starting where the near
# one ends; the exact offsets and pitches still need visual confirmation against a
# rendered frame (see scripts note in the module docstring of g1_self_scan).
_NEAR_POS = (0.10, 0.0, 0.05)
_NEAR_PITCH_DEG = 75.0
_FAR_POS = (0.06, 0.0, 0.38)
_FAR_PITCH_DEG = 35.0

# Depth normalisation, shared by the observation terms and the noise model (which
# needs them to express its dropout value in normalised units).
_DEPTH_OFFSET = 0.75
_DEPTH_SCALE = 0.5


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
    pos: tuple[float, float, float], pitch_deg: float
) -> MultiMeshRayCasterCameraCfg:
    """Build one torso-mounted, self-scanning depth camera.

    Args:
        pos: Offset from ``torso_link``, in its own frame.
        pitch_deg: Downward pitch of the camera's view.

    Returns:
        The configured sensor.
    """
    return MultiMeshRayCasterCameraCfg(
        prim_path=_CAMERA_MOUNT,
        update_period=1.0 / 30.0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=pos,
            rot=_pitch_down_quat_xyzw(pitch_deg),
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        max_distance=10.0,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=_DEPTH_K,
            width=_DEPTH_W,
            height=_DEPTH_H,
        ),
        # A plain string entry becomes a target with track_mesh_transforms disabled,
        # which is what we want for the static terrain mesh.
        mesh_prim_paths=["/World/ground"] + _self_scan_targets(),
        debug_vis=False,
    )


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


_DEPTH_OBS_PARAMS = {"offset": _DEPTH_OFFSET, "scale": _DEPTH_SCALE}


# ---------------------------------------------------------------------------
# Scene / observations / events
# ---------------------------------------------------------------------------

@configclass
class G1TeacherTerrainDepthSceneCfg(G1TeacherTerrainSceneCfg):
    """Clip-library terrain scene + two self-scanning depth cameras.

    The inherited ``height_scanner`` is untouched.
    """

    depth_camera_near: MultiMeshRayCasterCameraCfg = _make_depth_camera(_NEAR_POS, _NEAR_PITCH_DEG)
    depth_camera_far: MultiMeshRayCasterCameraCfg = _make_depth_camera(_FAR_POS, _FAR_PITCH_DEG)


@configclass
class G1TeacherTerrainDepthObservationsCfg(G1TeacherObservationsCfg):
    """Teacher observations + flattened depth scans in the policy group."""

    @configclass
    class PolicyCfg(G1TeacherObservationsCfg.PolicyCfg):
        # Noise is applied inside mdp.depth_image (driven by the resample_depth_noise
        # event), NOT via the term's `noise` field -- see mdp/sensors/noise.py.
        depth_near = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_near"), **_DEPTH_OBS_PARAMS},
            clip=(-2.0, 2.0),
        )
        depth_far = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_far"), **_DEPTH_OBS_PARAMS},
            clip=(-2.0, 2.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1TeacherTerrainDepthEventsCfg(G1TeacherEventsCfg):
    """Teacher events + per-camera intrinsics and noise randomisation on reset."""

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
class G1TeacherTerrainDepthEnvCfg(G1TeacherTerrainEnvCfg):
    """Terrain teacher + two self-scanning depth cameras, flattened into the policy."""

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
                **_DEPTH_OBS_PARAMS,
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
                **_DEPTH_OBS_PARAMS,
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
    cfg.scene.depth_camera_near.debug_vis = True
    cfg.scene.depth_camera_far.debug_vis = True


@configclass
class G1TeacherTerrainDepthEnvCfgPlay(G1TeacherTerrainEnvCfgPlay):
    """Play variant of :class:`G1TeacherTerrainDepthEnvCfg`."""

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
