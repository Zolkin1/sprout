# Copyright 2026 Zachary Olkin. All rights reserved.

"""Rebuilding the xyz channels OFFLINE, from what a recording already carries.

The Isaac term computes these channels from the live articulation. The dataset cannot: it has
a recording. So the same cloud is rebuilt from the recorded root quaternion, the recorded
joint angles and the NOMINAL mount out of the camera contract -- which is exactly the set a
robot has (IMU + encoders + calibration), and is why no existing recording needs redoing.

What these pin is the frame, because a wrong one is invisible: every shape stays right and the
numbers stay plausible.

* The frame is LEVEL. Its z is gravity, so the frame itself has zero roll and pitch; the
  robot's tilt survives as MOTION of the points within it, which is the whole mechanism (the
  terrain did not move, so the cloud must).
* The heading CANCELS. Yawing the robot leaves the cloud unchanged, which is what lets a
  deployment run this without odometry and survive a drifting yaw estimate.
"""

from __future__ import annotations

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


def test_yaw_does_not_move_the_cloud(rest):
    """THE deployability property: no odometry, and a drifting yaw cannot corrupt this.

    ``R_world_root = Rz(psi) @ R_rp`` and the yaw-frame coordinates are ``R_rp @ p_root``,
    in which ``psi`` does not appear.
    """
    joints, _ = rest
    yawed = _quat_z(torch.linspace(-3.0, 3.0, joints.shape[0]))
    (pos_y, rot_y), = _poses(yawed, joints)
    (pos_0, rot_0), = _poses(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand_as(yawed), joints)
    assert torch.allclose(pos_y, pos_0, atol=1e-6)
    assert torch.allclose(rot_y, rot_0, atol=1e-6)


def test_pitch_does_move_the_cloud(rest):
    """The frame is level, so a pelvis pitch shows up as motion -- the terrain did not move."""
    joints, upright = rest
    pitched = _quat_y(torch.full((joints.shape[0],), 0.25))
    (pos_p, _), = _poses(pitched, joints)
    (pos_0, _), = _poses(upright, joints)
    assert float((pos_p - pos_0).norm(dim=-1).min()) > 0.01


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
