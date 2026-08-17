# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rebuilding the xyz channels OFFLINE, from what a recording already carries.

The Isaac term computes these channels from the live articulation. The dataset cannot: it has
a recording. So the same cloud is rebuilt from the recorded root quaternion, the recorded
joint angles and the NOMINAL mount out of the camera contract -- which is exactly the set a
robot has (IMU + encoders + calibration), and is why no existing recording needs redoing.

What these pin is the frame, because a wrong one is invisible: every shape stays right and the
numbers stay plausible.

* The frame is LEVEL. Its z is gravity, so the frame itself has zero roll and pitch, and
  **static terrain keeps its coordinates when the robot tilts** -- the mechanism the channels
  exist for. A pelvis pitch or a waist turn rewrites the image, moves the CAMERA by exactly
  that much, and the unprojection composes the two back out.
* What the tilt DOES move is the robot's own geometry within the level frame (the camera, and
  the limbs the cameras see), so the tilt stays observable rather than being confusable with
  the terrain moving.
* The heading CANCELS OUT OF THE COMPUTATION -- ``psi`` never appears in it -- which is what
  lets a deployment run this without odometry and survive a drifting yaw estimate. That is
  not a yaw invariance of the cloud: the frame turns with the robot, so a static point's
  coordinates rotate by ``-psi``, consistently with the window the generator predicts.

The terrain invariances are tested by RENDERING a static plane, since the offline path takes
no world pose: the robot is the origin here, so "the robot translated toward the stairs" and
"the robot yawed" are not expressible in this file at all and belong to a sim-level test.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")
pytest.importorskip("motion_gen")

from motion_gen.features import _REPO_ROOT  # noqa: E402

from robot_rl.flow_matching.dataset import FutureStateDataset as DS  # noqa: E402

#: The same MJCF ``FlowMatchingConfig.fk_model_path`` names, resolved off the ONE repo-root
#: helper rather than by counting ``..`` hops from this file -- moving a module has silently
#: broken such a path before.
_MJCF = str(_REPO_ROOT / "transfer" / "sim" / "robots" / "g1" / "g1_29j.xml")
_H, _W = 4, 5
_K = _H * _W

#: The G1's joint order as the recording states it, abbreviated to what FK needs: the names
#: must match the MJCF, and the waist triple is what moves the camera.
_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]


def _default_channel_dataset(hw: tuple[int, int] = (6, 5), **cfg_kw):
    """Build a train split on a 29-joint recording under the SHIPPED defaults.

    The tiny synthetic robots elsewhere in the suite are four joints called ``j0..j3``, which
    cannot resolve an FK chain, so every one of them pins ``extero_channels=1`` and the default
    path would otherwise be covered by nothing. This fixture is the counterpart: real MJCF joint
    names, so the dataset actually runs the camera-pose reconstruction the default asks for.

    Args:
        hw: ``(height, width)`` per camera. Small on purpose -- the geometry is exercised by
            the parity tests above, and this one is about the row layout and the normalizer.

    Returns:
        ``(train_dataset, cfg, n_cameras, pixels_per_camera)``.
    """
    from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: PLC0415
    from robot_rl.flow_matching.dataset import build_train_val  # noqa: PLC0415
    from robot_rl.flow_matching.tests.recording_fixtures import (  # noqa: PLC0415
        camera_spec,
        windowed_recording,
    )

    n_cam, (h, w) = 2, hw
    data, stn, std, frames = windowed_recording(
        n_ep=3, windows_per_ep=5, history_len=4, future_len=6, nj=len(_JOINTS), k=2,
        num_scan_rays=7, depth=(n_cam, h, w), seed=0,
    )
    meta = SimpleNamespace(
        state_term_names=stn, state_term_dims=std, state_frame_names=frames,
        state_joint_names=list(_JOINTS), ref_future_len=6, history_anchor="plan",
        history_len=4, depth_camera_spec=camera_spec(n_cam, h, w),
    )
    cfg = FlowMatchingConfig(
        num_history_states=2, future_len=6, cmd_array="target_vel", cmd_stride=4,
        target_source="ego_anchored_clip_future", val_fraction=0.34, seed=0,
        d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
        # Absolute, because the field's default is repo-relative and a test's cwd is not a
        # contract (see the module note on _REPO_ROOT).
        fk_model_path=_MJCF,
        **cfg_kw,
    )
    train, _ = build_train_val(data, meta, cfg)
    return train, cfg, n_cam, h * w


def _cam(obs_term: str = "depth_near") -> dict:
    """One camera contract entry, mounted on the torso like the real pair."""
    return {
        "obs_term": obs_term, "sensor": obs_term, "mount_body": "torso_link",
        "offset_pos": [0.1107625, 0.0, 0.0994260],
        "offset_quat_xyzw": [0.0, 0.53730, 0.0, 0.84339],
        "width": _W, "height": _H, "fx": 2.5, "fy": 2.2, "cx": _W / 2, "cy": _H / 2,
        "max_distance": 8.0, "min_dist": 0.10, "max_dist": 7.9,
        "offset": 0.75, "scale": 0.5, "min_clip_val": 0.0, "max_clip_val": 0.0,
    }


def _quat_z(yaw: torch.Tensor) -> torch.Tensor:
    """Scalar-LAST quaternion for a rotation about world z (the axis the frame removes)."""
    half = yaw / 2.0
    return torch.stack([torch.zeros_like(half), torch.zeros_like(half),
                        torch.sin(half), torch.cos(half)], dim=-1)


def _quat_y(pitch: torch.Tensor) -> torch.Tensor:
    """Scalar-LAST quaternion for a pitch about world y (a tilt the frame RETAINS)."""
    half = pitch / 2.0
    return torch.stack([torch.zeros_like(half), torch.sin(half),
                        torch.zeros_like(half), torch.cos(half)], dim=-1)


def _poses(root_quat: torch.Tensor, joints: torch.Tensor, spec=None):
    """Camera poses in the yaw frame for the given robot state."""
    return DS._camera_pose_yaw_frame(spec or [_cam()], root_quat, joints, _JOINTS, _MJCF)


#: Two static surfaces, as ``(normal, offset)`` planes in the yaw frame: the ground the robot
#: stands on, and a stair riser ahead of it. The pelvis is the origin, so the ground sits at
#: negative z. A plane is enough to catch a wrong frame -- a frame that kept the root's pitch
#: would hand back a ground that TILTS with the robot.
_GROUND = (torch.tensor([0.0, 0.0, 1.0]), -0.75)
_RISER = (torch.tensor([1.0, 0.0, 0.0]), 1.20)


def _render_plane(
    cam: dict, cam_pos: torch.Tensor, cam_rot: torch.Tensor,
    normal: torch.Tensor, dist: float,
) -> torch.Tensor:
    """The normalised image a camera at this pose would measure of one infinite plane.

    Built independently of the code under test, and as the true inverse of it: ray-plane
    intersection in the yaw frame, ray LENGTH converted back to the z-depth a camera actually
    reports (``distance_to_image_plane``), then pushed through the real
    :func:`normalize_depth` so the clip band and the miss sentinel behave as they do in
    training. Rays that miss -- parallel to the plane, or meeting it behind the lens -- come
    back ``inf``, which the normalisation maps to the sentinel like any other missed ray.

    Args:
        cam: One camera contract entry.
        cam_pos: Camera origin in the yaw frame, ``(B, 3)``.
        cam_rot: Camera-to-yaw-frame rotation, ``(B, 3, 3)``.
        normal: Unit plane normal in the yaw frame, ``(3,)``.
        dist: Plane offset along ``normal``, in metres.

    Returns:
        Normalised depth ``(B, H*W)``, the same array the depth channel would carry.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
        normalize_depth,
        pinhole_ray_directions,
    )

    dirs = pinhole_ray_directions(
        _W, _H, float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"]),
        cam_pos.device, cam_pos.dtype,
    )                                                                   # (K, 3)
    normal = normal.to(cam_pos)
    rays = torch.einsum("bij,kj->bki", cam_rot, dirs)                   # (K rays per env)
    denom = torch.einsum("bki,i->bk", rays, normal)
    ray_len = (dist - torch.einsum("bi,i->b", cam_pos, normal)).unsqueeze(-1) / denom
    ray_len = torch.where(
        (denom.abs() > 1e-9) & (ray_len > 0.0), ray_len, torch.full_like(ray_len, torch.inf)
    )
    params = {k: float(cam[k]) for k in
              ("offset", "scale", "min_dist", "max_dist", "min_clip_val", "max_clip_val")}
    return normalize_depth(ray_len * dirs[:, 0], **params)


def _plane_residual(
    cam: dict, image: torch.Tensor, cam_pos: torch.Tensor, cam_rot: torch.Tensor,
    normal: torch.Tensor, dist: float,
) -> torch.Tensor:
    """How far the unprojected cloud sits off the plane it was rendered from, in metres.

    Over the pixels carrying a real reading only: a missed ray is placed at the sensor's
    reach by design, so it is nowhere near the surface and is not evidence about the frame.

    Args:
        cam: One camera contract entry.
        image: The normalised image, ``(B, H*W)``.
        cam_pos: Camera origin in the yaw frame, ``(B, 3)``.
        cam_rot: Camera-to-yaw-frame rotation, ``(B, 3, 3)``.
        normal: Unit plane normal in the yaw frame, ``(3,)``.
        dist: Plane offset along ``normal``, in metres.

    Returns:
        ``|p . n - dist|`` for every valid pixel, flattened.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
        invalid_depth_mask,
        unproject_to_yaw_frame,
    )

    points = unproject_to_yaw_frame(cam, image, cam_pos, cam_rot)       # (B, 3, K)
    resid = (torch.einsum("bik,i->bk", points, normal.to(points)) - dist).abs()
    return resid[~invalid_depth_mask(cam, image)]


@pytest.fixture(scope="module")
def rest():
    """A batch of upright robots with every joint at zero."""
    n = 6
    return torch.zeros(n, len(_JOINTS)), torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(n, 4)


def test_the_camera_sits_where_the_mount_puts_it(rest):
    """Upright and un-posed, the camera is the nominal offset above the pelvis."""
    joints, quat = rest
    (pos, rot), = _poses(quat, joints)
    assert pos.shape == (joints.shape[0], 3) and rot.shape == (joints.shape[0], 3, 3)
    # Torso-mounted on the G1: forward of and well above the pelvis origin.
    assert 0.05 < float(pos[0, 0]) < 0.30
    assert 0.05 < float(pos[0, 2]) < 0.60
    assert abs(float(pos[0, 1])) < 1e-3                    # centred laterally
    eye = torch.eye(3).expand_as(rot)
    assert torch.allclose(rot @ rot.transpose(1, 2), eye, atol=1e-5)


def test_yaw_does_not_move_the_camera(rest):
    """THE deployability property: no odometry, and a drifting yaw cannot corrupt this.

    ``R_world_root = Rz(psi) @ R_rp`` and the yaw-frame coordinates are ``R_rp @ p_root``,
    in which ``psi`` does not appear -- so the camera's pose in the frame is the same at
    every heading, and nothing downstream of here has to know one.

    This is about the POSE, not the cloud: a yaw turns the frame, so a static point's
    coordinates do rotate with it. Untestable in this file, which never sees a world.
    """
    joints, _ = rest
    yawed = _quat_z(torch.linspace(-3.0, 3.0, joints.shape[0]))
    (pos_y, rot_y), = _poses(yawed, joints)
    (pos_0, rot_0), = _poses(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand_as(yawed), joints)
    assert torch.allclose(pos_y, pos_0, atol=1e-6)
    assert torch.allclose(rot_y, rot_0, atol=1e-6)


def test_pitch_moves_the_camera_in_the_level_frame(rest):
    """A pelvis pitch swings the CAMERA, because the frame it is expressed in stays level.

    The camera sits well above the pelvis, so tilting about it sweeps the lens through an
    arc. This is the motion that has to be composed back out for the terrain to hold still
    (:func:`test_a_pelvis_pitch_leaves_static_terrain_where_it_was`) -- measuring it here is
    what makes that test's null result mean something rather than being a robot that never
    moved.
    """
    joints, upright = rest
    pitched = _quat_y(torch.full((joints.shape[0],), 0.25))
    (pos_p, _), = _poses(pitched, joints)
    (pos_0, _), = _poses(upright, joints)
    assert float((pos_p - pos_0).norm(dim=-1).min()) > 0.01


@pytest.mark.parametrize("plane", [_GROUND, _RISER], ids=["ground", "riser"])
def test_a_pelvis_pitch_leaves_static_terrain_where_it_was(rest, plane):
    """THE property these channels exist for: tilt the robot, the terrain does not move.

    The frame is anchored at the root and level, so with the root position pinned a pure
    roll/pitch moves neither its origin nor its axes -- a static plane has to come back at
    the coordinate it already had. The image is completely different at each pitch, which is
    the point: the camera swings, :meth:`_camera_pose_yaw_frame` swings with it, and the
    unprojection composes the two back out.

    Asserted on the plane RESIDUAL rather than pixel-by-pixel, because a pitch re-aims every
    ray: pixel ``k`` sees a different patch of ground before and after, so a pixel-wise diff
    would measure the re-sampling and not the frame.

    Note what this cannot cover, because the offline path takes no world pose: the robot is
    the origin here, so translating toward the stairs (which MUST move their coordinates) is
    not expressible and needs a sim.

    Measured over a +-0.30 rad sweep: residual 2.4e-7 m (the float32 floor) while the image
    itself moves over 1.2 of its normalised band, against 1.19 m for a frame that keeps the
    pitch (:func:`test_a_frame_that_kept_the_root_pitch_would_fail_that`).
    """
    joints, _ = rest
    normal, dist = plane
    cam = _cam()
    (pos, rot), = _poses(_quat_y(torch.linspace(-0.30, 0.30, joints.shape[0])), joints)
    image = _render_plane(cam, pos, rot, normal, dist)
    resid = _plane_residual(cam, image, pos, rot, normal, dist)

    # Half the pixels, not a handful: the surface has to be genuinely in view for the null
    # result to mean anything. Ground 120/120, riser 85/120 as shipped.
    assert resid.numel() >= 0.5 * joints.shape[0] * _K
    assert float(resid.max()) < 1e-6                 # metres


def test_a_waist_turn_leaves_static_terrain_where_it_was(rest):
    """The same invariance for the joints BETWEEN the root and the camera.

    A pitch exercises the ``yaw_quat`` half of the frame; the waist exercises the FK half,
    which is the part a deployment rebuilds from encoders. Both have to hold, and they fail
    independently.
    """
    joints, upright = rest
    normal, dist = _GROUND
    cam = _cam()
    n = joints.shape[0]
    turned = joints.clone()
    turned[:, _JOINTS.index("waist_yaw_joint")] = torch.linspace(-0.5, 0.5, n)
    turned[:, _JOINTS.index("waist_pitch_joint")] = torch.linspace(-0.3, 0.3, n)
    (pos, rot), = _poses(upright, turned)
    image = _render_plane(cam, pos, rot, normal, dist)
    resid = _plane_residual(cam, image, pos, rot, normal, dist)

    assert resid.numel() >= 0.5 * n * _K
    assert float(resid.max()) < 1e-6                 # measured 1.2e-7


def test_a_frame_that_kept_the_root_pitch_would_fail_that(rest):
    """The negative control, without which the two above pass on any constant cloud.

    Contaminate the camera pose with the root pitch the real frame drops -- i.e. build the
    cloud in a frame that rotates with the body instead of staying level -- and the same
    ground plane comes back tilted: 1.19 m worst case, 0.16 m mean, against a 2.4e-7 m null.
    That six-order-of-magnitude gap is what makes the null result evidence.
    """
    from isaaclab.utils.math import matrix_from_quat

    joints, _ = rest
    normal, dist = _GROUND
    cam = _cam()
    pitch = torch.linspace(-0.30, 0.30, joints.shape[0])
    (pos, rot), = _poses(_quat_y(pitch), joints)
    image = _render_plane(cam, pos, rot, normal, dist)                # measured correctly

    tilted = matrix_from_quat(_quat_y(pitch).to(pos))                 # ... read back wrongly
    resid = _plane_residual(
        cam, image, torch.einsum("bij,bj->bi", tilted, pos), tilted @ rot, normal, dist
    )
    assert float(resid.max()) > 0.10                                  # metres


def test_the_waist_moves_the_camera(rest):
    """The reason the channels exist: a waist turn changes the view without the terrain moving."""
    joints, upright = rest
    turned = joints.clone()
    turned[:, _JOINTS.index("waist_yaw_joint")] = 0.6
    turned[:, _JOINTS.index("waist_pitch_joint")] = 0.3
    (pos_t, rot_t), = _poses(upright, turned)
    (pos_0, rot_0), = _poses(upright, joints)
    assert float((pos_t - pos_0).norm(dim=-1).min()) > 1e-3
    assert float((rot_t - rot_0).abs().max()) > 1e-2


def test_two_cameras_on_one_mount_are_both_placed(rest):
    """The pair shares a mount, so one FK chain serves both -- at different offsets."""
    joints, upright = rest
    far = dict(_cam("depth_far"), offset_pos=[0.06, 0.0, 0.38])
    poses = _poses(upright, joints, spec=[_cam(), far])
    assert len(poses) == 2
    assert float(poses[1][0][0, 2] - poses[0][0][0, 2]) > 0.15   # the far camera sits higher


def test_a_second_mount_body_is_refused(rest):
    """One resolved chain cannot serve two mounts; using it anyway misplaces a camera."""
    joints, upright = rest
    other = dict(_cam("depth_far"), mount_body="pelvis_link")
    with pytest.raises(ValueError, match="share a mount"):
        _poses(upright, joints, spec=[_cam(), other])


def test_the_row_is_channel_major_and_keeps_the_depth_plane(rest):
    """Channel 0 must be the depth image itself, untouched, with xyz after it."""
    joints, upright = rest
    n = joints.shape[0]
    spec = [_cam(), dict(_cam("depth_far"), offset_pos=[0.06, 0.0, 0.38])]
    poses = _poses(upright, joints, spec=spec)
    depth = torch.rand(n, 2, _K) - 0.4
    rows = DS._depth_xyz_rows(depth.reshape(n, -1), spec, poses, 2, _K)

    assert rows.shape == (n, 2 * 4 * _K)
    img = rows.reshape(n, 2, 4, _K)
    assert torch.equal(img[:, :, 0], depth)                     # depth plane preserved
    assert torch.isfinite(rows).all()
    # The xyz planes are metres in the root frame, not a copy of the depth plane.
    assert not torch.allclose(img[:, 0, 1], img[:, 0, 0])


def test_the_shipped_default_widens_a_depth_row_to_five_channels():
    """A run that names no channel count gets ``[d, x, y, z, v]``, resolved from the source.

    ``extero_channels=None`` is the default and means "ask the source": 5 for depth, 1 for a
    height scan, whose rays already are ground points. The dataset resolves it and writes the
    int back, so a checkpoint states a number rather than inheriting whatever the default meant
    on the day it was trained.
    """
    train, cfg, n_cam, n_px = _default_channel_dataset()
    assert cfg.extero_channels == 5
    assert list(cfg.extero_channel_names) == ["d", "x", "y", "z", "v"]
    assert cfg.extero_sensor_dim == 5 * n_px          # the row COUNTS channels
    assert cfg.extero_sensor_count == n_cam
    assert train.extero.shape[-1] == n_cam * 5 * n_px
    assert torch.isfinite(train.extero).all()


def test_a_height_scan_resolves_the_same_default_to_one_channel():
    """The same ``None`` must not widen a scan -- and must not need a flag to avoid it."""
    from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: PLC0415

    assert FlowMatchingConfig(extero_source="depth").resolve_extero_channels() == 5
    assert FlowMatchingConfig(extero_source="height_scan").resolve_extero_channels() == 1
    with pytest.raises(ValueError, match="already are ground points"):
        FlowMatchingConfig(extero_source="height_scan", extero_channels=4
                           ).resolve_extero_channels()


def test_per_channel_normalization_groups_xyz_and_leaves_validity_alone():
    """The default normalizer: ``d`` on its own, ``xyz`` sharing one scale, ``v`` untouched.

    Three things a 5-channel row needs and ``per_sensor`` cannot give, because it pools every
    channel into one statistic whose std is dominated by metres of xyz:

    * ``d`` gets its own scale, so normalised depth is not compressed by the position channels;
    * ``x``, ``y``, ``z`` share ONE scale, so a metre stays a metre in every direction. A
      per-axis scale would rescale the axes against each other and destroy the metric
      comparability the channels exist to provide. Their OFFSETS stay per axis, which is the
      subtle half: subtracting one is a rigid translation, so it preserves every relative
      distance while actually centring each channel -- a shared offset would leave x and z
      parked a couple of sigma off zero, since the cameras look forward and down;
    * ``v`` is an identity -- mean 0, std 1, standardize_mask False. Fitting a 0/1 indicator
      would map "valid" onto a value that moves with the invalid RATE of whichever recording
      was used, so a cleaner recording would state the same fact with a different number.

    Uniformity over PIXELS is checked too, since that is the property the shared convolution
    kernel needs and the reason the old per-pixel mode is gone.
    """
    train, cfg, n_cam, n_px = _default_channel_dataset(extero_norm_mode="per_channel")
    assert cfg.extero_norm_mode == "per_channel"
    norm = train.extero_normalizer
    mean = norm.mean.reshape(n_cam, 5, n_px)
    std = norm.std.reshape(n_cam, 5, n_px)
    mask = norm.standardize_mask.reshape(n_cam, 5, n_px)

    # Uniform over pixels, every channel, every camera.
    assert torch.allclose(std, std[:, :, :1].expand_as(std))
    assert torch.allclose(mean, mean[:, :, :1].expand_as(mean))

    # xyz share one SCALE but keep per-axis OFFSETS; d has its own; v is the identity.
    for c in range(n_cam):
        xyz = std[c, 1:4, 0]
        assert torch.allclose(xyz, xyz[:1].expand_as(xyz)), "x, y, z must share one scale"
        assert not torch.allclose(std[c, 0, 0], xyz[0]), "d must not inherit the xyz scale"
        assert float(std[c, 4, 0]) == 1.0 and float(mean[c, 4, 0]) == 0.0
        # Each axis centred on its own mean. The cameras look forward and down, so x and z sit
        # far from zero and from each other; one shared offset would leave them uncentred.
        centred = train.extero_normalizer.normalize(train.extero).reshape(-1, n_cam, 5, n_px)
        for axis in (1, 2, 3):
            assert abs(float(centred[:, c, axis].mean())) < 1e-4, "xyz axes must be centred"
    assert not mask[:, 4].any(), "validity must not be standardized"
    assert mask[:, :4].all(), "depth and xyz must be standardized"


def test_contract_normalization_fits_nothing_and_shares_one_xyz_scale():
    """The DEFAULT: every constant comes from the camera contract, none from the data.

    ``per_channel`` fixed the conditioning but left the scales hostage to whichever recording
    was on disk when the normalizer was fitted -- the same terrain maps to different network
    inputs in two datasets, and a checkpoint's notion of "one metre" moves with it. Under
    ``contract`` nothing is fitted at all:

    * ``d`` is an IDENTITY, because it already arrives declared-normalised out of the contract
      (``(depth - 0.75) * 0.5``) and clipped to the band;
    * ``xyz`` share ONE declared scale across both cameras and all three axes, so a metre reads
      the same number everywhere -- across cameras too, which matters because both feed one
      transformer;
    * the OFFSET is per camera and geometric -- the boresight on the nominal ground plane -- so
      each camera is centred despite looking down at a different angle from a different height;
    * ``v`` is an identity, as in every mode.
    """
    train, cfg, n_cam, n_px = _default_channel_dataset()
    assert cfg.extero_norm_mode == "contract"
    assert len(cfg.extero_xyz_offset) == n_cam and cfg.extero_nominal_ground_z is not None

    norm = train.extero_normalizer
    mean = norm.mean.reshape(n_cam, 5, n_px)
    std = norm.std.reshape(n_cam, 5, n_px)
    mask = norm.standardize_mask.reshape(n_cam, 5, n_px)

    # d and v untouched; xyz on the one declared scale, offset per camera per axis.
    for ch in (0, 4):
        assert (std[:, ch] == 1.0).all() and (mean[:, ch] == 0.0).all()
        assert not mask[:, ch].any()
    assert (std[:, 1:4] == float(cfg.extero_xyz_scale)).all(), "one scale, every camera and axis"
    assert mask[:, 1:4].all()
    for c in range(n_cam):
        for axis, value in enumerate(cfg.extero_xyz_offset[c]):
            row = mean[c, 1 + axis]
            assert torch.allclose(row, torch.full_like(row, float(value)), atol=1e-6)


def test_the_declared_constants_do_not_depend_on_the_recording():
    """Two different recordings of the same robot must produce identical constants.

    This is the property the whole mode exists for, and the one a fitted normalizer cannot
    have. Different seed, different terrain, different pixel values -- same numbers.
    """
    a_train, a_cfg, _, _ = _default_channel_dataset()
    b_train, b_cfg, _, _ = _default_channel_dataset(hw=(8, 7))

    assert a_cfg.extero_xyz_offset == b_cfg.extero_xyz_offset
    assert a_cfg.extero_xyz_scale == b_cfg.extero_xyz_scale
    assert a_cfg.extero_nominal_ground_z == b_cfg.extero_nominal_ground_z
    # ... and the data genuinely differs, so the equality above is not vacuous.
    assert a_train.extero.shape != b_train.extero.shape


def test_the_xyz_clamp_cannot_bite_real_geometry():
    """The clamp is a safety net sized to the max-range cone, not a band that shapes data.

    Sizing it to bind would repeat the mistake the +-2.0 depth band made (13.8% of windows
    clipped). The envelope is derived from the clip band divided by the corner cosine -- NOT
    from ``max_dist``, which under-bounds a frame corner by exactly that cosine -- and the
    dataset raises rather than silently squashing if the data ever exceeds it.
    """
    train, cfg, n_cam, n_px = _default_channel_dataset()
    xyz = train.extero.reshape(-1, n_cam, 5, n_px)[:, :, 1:4]
    bound = float(cfg.extero_xyz_envelope) * float(cfg.extero_xyz_scale)
    assert float(xyz.abs().max()) < bound, "the clamp must not bind on real geometry"


def test_the_trained_clip_band_is_stamped_rather_than_left_to_a_fallback():
    """The artifact must STATE its band, because the two fallback tables no longer agree.

    ``config.EXTERO_CLIP`` says 5.575 for depth; ``deploy.depth_obs.EXTERO_CLIP_NP`` says 2.0,
    and each is correct for the artifacts of its own era. So a checkpoint that stored ``None``
    resolved to a different band on each side of the fence: trained at 5.575, deployed at 2.0,
    saturating range at 4.75 m and standing a false wall in the xyz cloud at exactly the
    distance the far camera is for. Nothing would have raised.

    Stamping the resolved value at dataset build removes the ambiguity for everything trained
    from here, which is the same contract ``extero_encoder`` and ``extero_channels`` follow.
    """
    _train, cfg, _n_cam, _n_px = _default_channel_dataset()
    assert cfg.extero_clip is not None, "the band must be recorded, not left to a fallback"
    assert cfg.extero_clip == pytest.approx(cfg.resolve_extero_clip())

    # And the two fallbacks really do differ, so the assertion above is load-bearing rather
    # than belt-and-braces. If these are ever reconciled, say so here.
    #
    # ``transfer`` is put on the path HERE rather than relied on from another test module: a
    # sibling happens to insert it, which would make this pass by collection order and fail
    # when run alone -- the exact fragility test_train_cli_defaults exists to document.
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(_REPO_ROOT / "transfer"))
    from deploy.depth_obs import EXTERO_CLIP_NP  # noqa: PLC0415

    from robot_rl.flow_matching.config import EXTERO_CLIP  # noqa: PLC0415

    assert EXTERO_CLIP["depth"] != EXTERO_CLIP_NP["depth"]
