# Copyright 2026 Zachary Olkin. All rights reserved.

"""Unprojecting a depth image back into 3-D, in the root's yaw-aligned frame.

These channels exist because the depth image alone cannot say WHERE a step edge is in
metres, only where it is in pixels -- so the arithmetic that turns one into the other is
load-bearing, and every way of getting it subtly wrong keeps the shapes right:

* z-depth used as ray length (or the correction applied twice) -- caught by the round trip,
  which projects a known point and demands it back;
* a transposed image fold -- caught by :func:`test_a_transposed_camera_is_a_different_cloud`,
  since 30x26 and 26x30 hold the same 780 pixels and no size check can separate them;
* invalid pixels denormalising to 0 m, piling a fake surface onto the robot's own origin;
* a rotation composed the wrong way round, which looks plausible until the camera is
  pitched.

The frame's defining property is tested here too: the camera pose is supplied already in
the yaw frame, so this function never sees a world coordinate and cannot depend on heading.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "transfer"))

from deploy.depth_obs import (  # noqa: E402
    invalid_depth_mask,
    invalid_depth_value,
    normalize_depth_np,
    pinhole_ray_directions,
    unproject_to_yaw_frame,
)

WIDTH, HEIGHT = 30, 26


def _cam(width: int = WIDTH, height: int = HEIGHT) -> dict:
    """A camera contract entry with the trained near-camera optics."""
    fov_h, fov_v = 100.0, 70.0
    return {
        "obs_term": "depth_near", "sensor": "depth_camera_near",
        "width": width, "height": height,
        "fx": width / (2.0 * math.tan(math.radians(fov_h) / 2.0)),
        "fy": height / (2.0 * math.tan(math.radians(fov_v) / 2.0)),
        "cx": width / 2.0, "cy": height / 2.0,
        "max_distance": 8.0, "min_dist": 0.10, "max_dist": 7.9,
        "offset": 0.75, "scale": 0.5, "min_clip_val": 0.0, "max_clip_val": 0.0,
    }


def _rot_y(deg: float) -> np.ndarray:
    """Rotation about +y (a camera pitch), degrees."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    """Rotation about +z (a yaw), degrees."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_round_trip_recovers_a_known_point():
    """Project a point to (u, v, z-depth), normalise, unproject -- get the point back.

    The one test that pins z-depth-vs-ray-length: the projection below uses the pinhole
    relation on the OPTICAL AXIS depth, so an unprojection that treats that number as a ray
    length lands short by 1/cos(theta) everywhere off centre.
    """
    cam = _cam()
    # A point 2.5 m down the optical axis, off-centre in both image axes.
    p_cam = np.array([2.5, 0.4, -0.3])
    u = cam["cx"] - p_cam[1] / p_cam[0] * cam["fx"] - 0.5
    v = cam["cy"] - p_cam[2] / p_cam[0] * cam["fy"] - 0.5
    ui, vi = int(round(u)), int(round(v))
    # Re-derive the exact point that pixel CENTRE sees at this z-depth, so the comparison is
    # not limited by the rounding to a pixel.
    dirs = pinhole_ray_directions(WIDTH, HEIGHT, cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    d = dirs[vi * WIDTH + ui]
    want = d * (p_cam[0] / d[0])

    image = np.full(WIDTH * HEIGHT, invalid_depth_value(cam))
    image[vi * WIDTH + ui] = normalize_depth_np(p_cam[0], cam["offset"], cam["scale"],
                                                cam["min_dist"], cam["max_dist"])
    got = unproject_to_yaw_frame(cam, image, np.zeros(3), np.eye(3))[:, vi * WIDTH + ui]
    assert np.allclose(got, want, atol=1e-12)


def test_invalid_pixels_sit_at_max_dist_along_their_own_ray():
    """No valid return -> a point at the sensor's reach, not at the robot's origin.

    Denormalising the sentinel gives 0 m, which would put a fake surface exactly where the
    convolution is most sensitive. The distance is measured from the CAMERA, so this also
    catches an invalid pixel that forgot the ray-length correction.
    """
    cam = _cam()
    image = np.full(WIDTH * HEIGHT, invalid_depth_value(cam))
    assert invalid_depth_mask(cam, image).all()
    cam_pos = np.array([0.11, 0.0, 0.45])
    points = unproject_to_yaw_frame(cam, image, cam_pos, _rot_y(65.0))
    radius = np.linalg.norm(points - cam_pos[:, None], axis=0)
    assert np.allclose(radius, cam["max_dist"], atol=1e-9)


def test_a_noised_sentinel_is_still_invalid():
    """The mask must survive the training-time noiser, which destroys the exact sentinel.

    DepthNoiser adds a plane bias, per-pixel noise and a 3x3 blur AFTER normalisation, so a
    missed pixel arrives NEAR the sentinel rather than on it. Equality would call every one
    of them valid and put a surface ~0.1 m from the camera; the range threshold catches
    almost all.

    "Almost" is the honest number and is asserted as such. At the peak plane bias the noise
    half-width (0.13 m of range) exceeds the near camera's 0.10 m ``min_dist``, so ~11% of
    its missed pixels survive as readings just inside the near limit; the far camera's
    0.15 m floor is beyond the band and loses none. Closing that gap would mean exporting
    the DR amplitudes into the deployment contract -- a training-only number in an artifact
    hardware reads -- which is a worse trade than a few stray sky pixels.
    """
    rng = np.random.default_rng(3)
    # The measured DR half-widths at their peak: plane 0.05 + uniform 0.015, normalised.
    noise = rng.uniform(-0.065, 0.065, 20_000)
    near, far = _cam(), {**_cam(), "min_dist": 0.15}
    assert not np.any(noise == 0.0)
    caught_near = invalid_depth_mask(near, invalid_depth_value(near) + noise).mean()
    caught_far = invalid_depth_mask(far, invalid_depth_value(far) + noise).mean()
    assert caught_far == 1.0
    assert 0.85 < caught_near < 0.90, f"near camera caught {caught_near:.3f}"


def test_the_pose_is_applied_as_rotation_then_translation():
    """``p_yaw = R @ p_cam + t`` -- not the inverse, and not translate-then-rotate."""
    cam = _cam()
    image = np.full(WIDTH * HEIGHT, normalize_depth_np(3.0, cam["offset"], cam["scale"]))
    rot, pos = _rot_y(65.0), np.array([0.1, -0.2, 0.4])
    at_origin = unproject_to_yaw_frame(cam, image, np.zeros(3), np.eye(3))
    posed = unproject_to_yaw_frame(cam, image, pos, rot)
    assert np.allclose(posed, rot @ at_origin + pos[:, None], atol=1e-12)


def test_a_pitched_camera_puts_the_ground_below_the_root():
    """A camera pitched 65 deg down sees points below its own height, not in front of it.

    The sanity check that the mount rotation is composed in the right direction: with the
    optical axis 65 deg below horizontal, a 2 m reading must land well under the camera.
    """
    cam = _cam()
    image = np.full(WIDTH * HEIGHT, normalize_depth_np(2.0, cam["offset"], cam["scale"]))
    cam_pos = np.array([0.11, 0.0, 0.45])
    points = unproject_to_yaw_frame(cam, image, cam_pos, _rot_y(65.0))
    centre = points[:, (HEIGHT // 2) * WIDTH + WIDTH // 2]
    assert centre[2] < cam_pos[2] - 1.5          # 2 m * sin(65 deg) = 1.81 m down
    assert 0.0 < centre[0] - cam_pos[0] < 1.0    # and only 0.85 m forward


def test_a_transposed_camera_is_a_different_cloud():
    """30x26 and 26x30 hold the same 780 pixels, so only the geometry can separate them.

    A transposed fold keeps every shape right and scrambles every neighbourhood, which is
    the failure mode a convolutional encoder makes load-bearing.
    """
    cam = _cam()
    rng = np.random.default_rng(0)
    image = normalize_depth_np(rng.uniform(1.0, 4.0, WIDTH * HEIGHT),
                              cam["offset"], cam["scale"])
    upright = unproject_to_yaw_frame(cam, image, np.zeros(3), np.eye(3))
    swapped = unproject_to_yaw_frame(_cam(HEIGHT, WIDTH), image, np.zeros(3), np.eye(3))
    assert not np.allclose(upright, swapped, atol=1e-3)


def test_a_wrong_pixel_count_is_refused():
    """A row of the wrong width is a mismatched camera, not something to reshape."""
    cam = _cam()
    with pytest.raises(ValueError, match="but the image holds"):
        unproject_to_yaw_frame(cam, np.zeros(WIDTH * HEIGHT + 1), np.zeros(3), np.eye(3))


def test_an_image_and_a_flat_row_agree():
    """``(H, W)`` and ``(H*W,)`` are the same input -- the fold is row-major."""
    cam = _cam()
    rng = np.random.default_rng(1)
    flat = normalize_depth_np(rng.uniform(1.0, 4.0, WIDTH * HEIGHT),
                             cam["offset"], cam["scale"])
    a = unproject_to_yaw_frame(cam, flat, np.zeros(3), np.eye(3))
    b = unproject_to_yaw_frame(cam, flat.reshape(HEIGHT, WIDTH), np.zeros(3), np.eye(3))
    assert np.array_equal(a, b)


##
# Parity with the torch original in robot_rl
##

def test_parity_with_the_torch_unprojection():
    """The two implementations must agree to double precision.

    ``robot_rl`` cannot import ``transfer``, so the geometry exists twice -- once in torch for
    the Isaac observation term, once in numpy for the dataset, MuJoCo and hardware. That is
    the same split ``normalize_depth`` / ``normalize_depth_np`` and
    ``quat_math`` / ``quat_math_np`` live with, and it is only safe with this test: the four
    consumers of these channels must produce the same cloud from the same image, or the
    generator trains on one geometry and deploys against another.
    """
    torch = pytest.importorskip("torch")
    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import (
        pinhole_ray_directions as torch_dirs,
        unproject_to_yaw_frame as torch_unproject,
    )

    cam = _cam()
    rng = np.random.default_rng(4)
    # Every class in one image: real readings, the sentinel, and clip-band saturation.
    image = np.concatenate([
        normalize_depth_np(rng.uniform(0.3, 6.0, WIDTH * HEIGHT - 40),
                           cam["offset"], cam["scale"], cam["min_dist"], cam["max_dist"]),
        np.full(20, invalid_depth_value(cam)),
        np.full(20, 2.0),
    ])
    pos, rot = np.array([0.11, -0.03, 0.45]), _rot_y(65.0) @ _rot_z(7.0)

    want = unproject_to_yaw_frame(cam, image, pos, rot)
    got = torch_unproject(
        cam,
        torch.from_numpy(image).double()[None],
        torch.from_numpy(pos).double()[None],
        torch.from_numpy(rot).double()[None],
    )[0].numpy()
    assert np.abs(got - want).max() < 1e-12

    dirs = torch_dirs(WIDTH, HEIGHT, cam["fx"], cam["fy"], cam["cx"], cam["cy"],
                      "cpu", torch.float64).numpy()
    assert np.abs(dirs - pinhole_ray_directions(
        WIDTH, HEIGHT, cam["fx"], cam["fy"], cam["cx"], cam["cy"])).max() < 1e-15


def test_the_cloud_carries_no_heading():
    """Yawing the robot leaves the channels UNCHANGED -- the deployability claim.

    The yaw-aligned frame turns with the robot, so a pure heading change moves the frame
    and the world by the same rotation and the coordinates do not move at all. Stated as a
    test because it is what lets a deployment compute these from FK and gravity, with no
    odometry and no yaw estimate: here, yawing the whole camera pose about the root leaves
    every point where it was.
    """
    cam = _cam()
    rng = np.random.default_rng(2)
    image = normalize_depth_np(rng.uniform(1.0, 4.0, WIDTH * HEIGHT),
                              cam["offset"], cam["scale"])
    pos, rot = np.array([0.11, 0.0, 0.45]), _rot_y(65.0)
    base = unproject_to_yaw_frame(cam, image, pos, rot)
    for yaw in (17.0, -90.0, 180.0):
        # The camera pose expressed in the yaw frame is yaw-independent by construction;
        # rotating the WORLD and the frame together is a no-op on it.
        rz = _rot_z(yaw)
        turned = unproject_to_yaw_frame(cam, image, rz @ pos, rz @ rot)
        assert np.allclose(turned, rz @ base, atol=1e-12)
