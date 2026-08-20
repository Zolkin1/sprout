# Copyright 2026 Zachary Olkin. All rights reserved.

"""The MuJoCo depth caster: ray-pattern parity with IsaacLab, then geometry in a scene.

Three levels, cheapest first:

1. ``deploy.depth_obs.pinhole_ray_directions`` vs ``isaaclab...patterns.pinhole_camera_pattern`` --
   exact, and the check that the exported intrinsics mean the same thing on both sides.
2. The cast against a flat MuJoCo scene vs the CLOSED-FORM ray/ground intersection --
   catches a wrong mount pose, a transposed rotation, or ray-length-vs-z-depth confusion,
   none of which the pattern test can see.
3. Self-occlusion and articulation: the robot must appear in its own near image, and
   bending the knees must change it.

Levels 2-3 need the MuJoCo scene assets, so run from ``transfer/sim``:
``python tests/test_depth_camera.py``
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_SIM, os.path.dirname(_SIM)):     # transfer/sim, transfer/
    if _path not in sys.path:
        sys.path.insert(0, _path)

import mujoco  # noqa: E402

from deploy.depth_obs import (  # noqa: E402
    denormalize_depth_np,
    invalid_depth_value,
    normalize_camera_images,
    pinhole_ray_directions,
)

from depth_camera import MujocoDepthCameras, quat_xyzw_to_matrix  # noqa: E402

WIDTH, HEIGHT = 30, 26
SCENE = os.path.join(_SIM, "robots", "g1", "g1_21j_basic_scene.xml")

# The two trained cameras (g1_depth_env_cfg._NEAR / _FAR), as the exported contract.
NEAR = {
    "obs_term": "depth_near", "sensor": "depth_camera_near", "mount_body": "torso_link",
    "offset_pos": [0.1107625, 0.0, 0.0994260], "width": WIDTH, "height": HEIGHT,
    "max_distance": 8.0, "min_dist": 0.10, "max_dist": 7.9,
    "offset": 0.75, "scale": 0.5, "min_clip_val": 0.0, "max_clip_val": 0.0,
    "clip": [-2.0, 2.0], "update_period": 0.02,
}
FAR = {
    **NEAR, "obs_term": "depth_far", "sensor": "depth_camera_far",
    "offset_pos": [0.06, 0.0, 0.38], "max_distance": 12.0, "max_dist": 11.9,
}


def _pitch_quat_xyzw(degrees: float) -> list[float]:
    """The mount quaternion for a downward pitch, as the env cfg builds it."""
    half = math.radians(degrees) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _intrinsics(fov_h_deg: float, fov_v_deg: float) -> dict:
    """Pinhole intrinsics for an effective field of view, as the exporter derives them."""
    return {
        "fx": WIDTH / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0)),
        "fy": HEIGHT / (2.0 * math.tan(math.radians(fov_v_deg) / 2.0)),
        "cx": WIDTH / 2.0, "cy": HEIGHT / 2.0,
    }


def _spec() -> list[dict]:
    """The full two-camera contract with the trained optics."""
    return [
        {**NEAR, **_intrinsics(100.0, 70.0), "offset_quat_xyzw": _pitch_quat_xyzw(65.0)},
        {**FAR, **_intrinsics(70.0, 42.0), "offset_quat_xyzw": _pitch_quat_xyzw(35.0)},
    ]


##
# 1. Ray-pattern parity with IsaacLab
##

@pytest.mark.parametrize("fov_h, fov_v", [(100.0, 70.0), (70.0, 42.0)])
def test_ray_directions_match_isaaclab_pinhole_pattern(fov_h, fov_v):
    """Identical ray directions, so both sims sample the same pixels.

    IsaacLab builds the pattern from an intrinsic matrix; we build it from the fx/fy/cx/cy
    the exporter derives from that same matrix. This is the check that the round trip
    through the ONNX contract is lossless.
    """
    torch = pytest.importorskip("torch")
    from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
    from isaaclab.sensors.ray_caster.patterns.patterns import pinhole_camera_pattern

    focal = 2.2
    aperture = lambda fov: 2.0 * focal * math.tan(math.radians(fov) / 2.0)
    cfg = PinholeCameraPatternCfg(
        focal_length=focal, horizontal_aperture=aperture(fov_h),
        vertical_aperture=aperture(fov_v), width=WIDTH, height=HEIGHT,
    )
    intrinsic = torch.tensor([[
        [WIDTH * focal / aperture(fov_h), 0.0, WIDTH / 2.0],
        [0.0, HEIGHT * focal / aperture(fov_v), HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ]], dtype=torch.float32)
    _, want = pinhole_camera_pattern(cfg, intrinsic, torch.device("cpu"))

    got = pinhole_ray_directions(WIDTH, HEIGHT, **_intrinsics(fov_h, fov_v))
    # IsaacLab builds the pattern in float32, so compare at single precision.
    err = float(np.abs(got - want[0].numpy()).max())
    assert err < 1e-6, f"max abs direction error {err:.3e}"


def test_ray_layout_is_row_major_top_left_first():
    """Flat index ``v*W + u``, with row 0 up and column 0 left (camera frame y/z)."""
    dirs = pinhole_ray_directions(WIDTH, HEIGHT, **_intrinsics(100.0, 70.0))
    top_left, top_right = dirs[0], dirs[WIDTH - 1]
    bottom_left = dirs[(HEIGHT - 1) * WIDTH]
    assert top_left[1] > 0 > top_right[1]      # +y is camera-left
    assert top_left[2] > 0 > bottom_left[2]    # +z is camera-up
    assert np.allclose(np.linalg.norm(dirs, axis=-1), 1.0)


def test_mount_quaternion_pitches_the_optical_axis_down():
    """A positive pitch tilts the optical axis toward -z, per the env cfg's convention."""
    axis = quat_xyzw_to_matrix(_pitch_quat_xyzw(75.0)) @ np.array([1.0, 0.0, 0.0])
    assert axis[2] == pytest.approx(-math.sin(math.radians(75.0)))
    assert axis[0] == pytest.approx(math.cos(math.radians(75.0)))


##
# 2. Geometry: the cast vs the closed-form ground intersection
##

@pytest.fixture
def flat_scene():
    """The G1 on the flat MuJoCo scene at its standing keyframe."""
    if not os.path.exists(SCENE):
        pytest.skip(f"MuJoCo scene not found: {SCENE} (run from transfer/sim)")
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def cams(flat_scene):
    """Both trained cameras, resolved against the flat scene."""
    model, _ = flat_scene
    return MujocoDepthCameras(model, _spec())


def _denormalise(image: np.ndarray, cam: dict) -> np.ndarray:
    """Undo ``(d - offset) * scale`` to recover metres for comparison."""
    return denormalize_depth_np(image, cam["offset"], cam["scale"])


def test_ground_pixels_match_the_analytic_plane(flat_scene, cams):
    """Every floor-hitting pixel equals the closed-form ray/``z=0`` z-depth.

    The strongest single check on the port: it pins the mount pose, the rotation
    composition, and the ray-length -> image-plane-depth projection at once.
    """
    _, data = flat_scene
    spec = _spec()
    images, hits = cams.compute(data, want_hits=True)

    for i, (cam, camera) in enumerate(zip(spec, cams.cameras)):
        pos, rot = camera.pose(data)
        dirs_w = camera.dirs_cam @ rot.T
        # Analytic intersection with the ground plane z = 0, then project onto the optical
        # axis to get image-plane depth.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -pos[2] / dirs_w[:, 2]
        ground = (dirs_w[:, 2] < 0) & (t > 0)
        # Compare only pixels the cast resolved on the floor (not on the robot).
        on_floor = ground & np.isfinite(hits[i, :, 2]) & (np.abs(hits[i, :, 2]) < 1e-6)
        assert on_floor.sum() > 100, f"{cam['sensor']}: only {on_floor.sum()} floor pixels"
        want = np.minimum(t[on_floor] * camera.cos_theta[on_floor], cam["max_distance"])
        got = _denormalise(images[i][on_floor], cam)
        err = float(np.abs(got - want).max())
        assert err < 1e-9, f"{cam['sensor']}: max abs depth error {err:.3e} m"


def test_depth_is_image_plane_not_ray_length(flat_scene, cams):
    """Off-axis pixels are SHORTER than their ray length, by cos(theta).

    Confuses easily with ``distance_to_camera``: on a flat floor the difference reaches
    tens of centimetres at the near camera's corners.
    """
    _, data = flat_scene
    spec = _spec()
    images, hits = cams.compute(data, want_hits=True)
    camera = cams.cameras[0]
    pos, _ = camera.pose(data)
    resolved = np.isfinite(hits[0, :, 0])
    ray_len = np.linalg.norm(hits[0][resolved] - pos, axis=-1)
    depth_z = _denormalise(images[0][resolved], spec[0])
    np.testing.assert_allclose(depth_z, ray_len * camera.cos_theta[resolved], atol=1e-9)
    corner = np.argmin(camera.cos_theta[resolved])
    assert ray_len[corner] - depth_z[corner] > 0.05


def test_out_of_range_reads_as_no_reading(flat_scene):
    """A camera pointed at nothing returns the invalid value everywhere, not garbage.

    A ray that hits nothing reports exactly ``max_distance`` (the training sensor's
    ``depth_clipping_behavior="max"``), which the normalisation maps to "no reading" --
    NOT to a surface at max range.
    """
    model, data = flat_scene
    # Pitch the near camera UP and cast against the terrain only, so nothing is in view.
    spec = [{**NEAR, **_intrinsics(100.0, 70.0), "offset_quat_xyzw": _pitch_quat_xyzw(-80.0)}]
    up_cams = MujocoDepthCameras(model, spec, self_scan_group=2, terrain_group=2)
    images, hits = up_cams.compute(data, want_hits=True)
    assert np.allclose(images[0], invalid_depth_value(spec[0]))
    assert not np.isfinite(hits).any()


##
# 3. Self-occlusion and articulation
##

def _self_pixels(data, cams, index=0) -> int:
    """Count pixels in one camera's image that landed on the robot rather than terrain."""
    _, hits = cams.compute(data, want_hits=True)
    return int((np.isfinite(hits[index, :, 2]) & (hits[index, :, 2] > 1e-6)).sum())


def test_the_robot_occludes_itself(flat_scene, cams):
    """The near camera sees the robot's own body -- the point of the self-scan targets."""
    _, data = flat_scene
    assert _self_pixels(data, cams) > 20


def test_limbs_articulate_in_the_image(flat_scene, cams):
    """Bending the knees changes the near image.

    The failure this guards against on the IsaacLab side is a whole limb welded into one
    rigid mesh frozen at its spawn pose; here it guards against reading stale kinematics.
    """
    model, data = flat_scene
    before = cams.compute(data).copy()
    for joint in ("left_knee_joint", "right_knee_joint"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        data.qpos[model.jnt_qposadr[jid]] = 1.2
    mujoco.mj_forward(model, data)
    assert not np.allclose(before[0], cams.compute(data)[0])


def test_collision_geoms_are_not_visible(flat_scene, cams):
    """Only the VISUAL meshes are cast, so the box foot collider cannot appear.

    Casting group 0 as well would put a box where IsaacLab has the foot's STL.
    """
    model, data = flat_scene
    assert _self_pixels(data, cams) > 0
    with_collision = MujocoDepthCameras(model, _spec(), self_scan_group=0)
    # Group 0 alone shows a different (collision-shaped) robot, so the images differ.
    assert not np.allclose(cams.compute(data)[0], with_collision.compute(data)[0])


def test_mount_body_must_exist():
    """A contract naming a body this model lacks fails at construction."""
    if not os.path.exists(SCENE):
        pytest.skip("MuJoCo scene not found (run from transfer/sim)")
    model = mujoco.MjModel.from_xml_path(SCENE)
    spec = [{**NEAR, **_intrinsics(100.0, 70.0), "offset_quat_xyzw": _pitch_quat_xyzw(65.0),
             "mount_body": "not_a_body"}]
    with pytest.raises(ValueError, match="not_a_body"):
        MujocoDepthCameras(model, spec)


##
# Optional noise model
##

def test_noise_perturbs_without_breaking_the_image(flat_scene, cams):
    """The noiser changes the image, stays finite, and holds its per-episode state."""
    model, data = flat_scene
    base = cams.compute(data).copy()
    noisy_cams = MujocoDepthCameras(model, _spec())
    noisy_cams.enable_noise(np.random.default_rng(0))
    noisy = noisy_cams.compute(data)
    assert noisy.shape == base.shape
    assert np.isfinite(noisy).all()
    assert not np.allclose(noisy, base)
    # Dropout is per frame, so two calls differ even without a reset.
    assert not np.allclose(noisy, noisy_cams.compute(data))


@pytest.mark.parametrize("x", [0.0, 8.5, 12.0, 40.0, 200.0])
def test_ground_is_still_seen_far_from_the_world_origin(flat_scene, cams, x):
    """The ground must not disappear as the robot walks away from spawn.

    Regression for a real and badly-behaved bug: ``mj_multiRay``'s ``cutoff`` culls a geom
    by ``dist(ray_origin, geom_ORIGIN) - rbound``, and a plane has ``rbound = 0`` with the
    ground plane's origin at the world origin. Passing the camera's ``max_distance`` as the
    cutoff therefore discarded the whole floor once the robot got that far from spawn --
    every ground pixel read "no reading" while self-occlusion pixels stayed perfect, so the
    policy was handed a depth image with no terrain in it. The near camera (cutoff 8 m) lost
    it first, the far camera (12 m) later, which is what made it look intermittent.
    """
    model, data = flat_scene
    data.qpos[0] = x
    mujoco.mj_forward(model, data)
    cams.compute(data)
    for cam, camera in zip(_spec(), cams.cameras):
        on_terrain = int(((model.geom_group[np.clip(camera._geomid, 0, None)] == 2)
                          & camera.hit).sum())
        assert on_terrain > 100, (
            f"{cam['sensor']}: only {on_terrain} ground pixels at x={x} m "
            "(the floor plane is being culled by the mj_multiRay cutoff)"
        )


def test_ground_pixel_count_is_independent_of_position(flat_scene, cams):
    """Same flat floor, same pose, different world offset: the image must not change.

    Pins the fix as a pure restoration rather than a different image -- the count and the
    depths both have to be invariant, since translating along a plane changes nothing the
    camera can see.
    """
    model, data = flat_scene
    ref = None
    for x in (0.0, 6.0, 30.0, 150.0):
        data.qpos[0] = x
        mujoco.mj_forward(model, data)
        img = cams.compute(data).copy()
        if ref is None:
            ref = img
        else:
            assert np.abs(img - ref).max() < 1e-12, f"image changed at x={x} m"


def test_extrinsics_jitter_does_not_compound(cams):
    """Repeated jitter perturbs the NOMINAL mount, so it cannot random-walk away."""
    nominal = cams.cameras[0].nominal_offset_pos.copy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        cams.jitter_extrinsics(rng)
    assert (np.abs(cams.cameras[0].offset_pos - nominal) <= 0.01 + 1e-12).all()
    np.testing.assert_array_equal(cams.cameras[0].nominal_offset_pos, nominal)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_raw_depth_renormalises_to_the_same_image(cams, flat_scene):
    """``want_raw`` must be a faithful pre-image of the normalised reading.

    The replay stream feeds the C++ runtime RAW METRES, because that is what
    HardwareReadings carries and the C++ normalises internally. If the raw return and the
    normalised return were not two views of the same cast, every replay comparison would be
    measuring the recording rather than the runtime -- and it would look like a C++ bug.

    This also checks the two normalisers against each other for free: the MuJoCo-side
    ``normalize_depth_np`` in depth_camera.py, and ``deploy.depth_obs`` -- which is the port
    hardware runs. They are separate implementations of one contract.
    """
    _model, data = flat_scene
    images, raw = cams.compute(data, want_raw=True)
    assert raw.shape == images.shape
    # Metres along the optical axis, so bounded by the camera's own range.
    assert np.all(raw > 0.0)
    assert np.all(raw <= max(c["max_dist"] for c in _spec()) + 1e-9)
    again = normalize_camera_images(_spec(), [raw[i] for i in range(raw.shape[0])])
    assert np.allclose(np.asarray(again), images, atol=0.0, rtol=0.0), (
        "deploy.depth_obs and depth_camera.py disagree on the SAME cast -- the replay stream "
        "would be feeding the C++ a different image than python scored")


def test_want_raw_refuses_to_lie_when_a_noiser_is_attached(flat_scene):
    """Noise is applied AFTER normalisation, so the raw reading does not carry it.

    Returning it anyway would hand a replay a noise-free image while the python run saw a
    noisy one: a guaranteed, unexplainable parity failure. Refused instead.
    """
    model, data = flat_scene
    noisy = MujocoDepthCameras(model, _spec())
    noisy.enable_noise(np.random.default_rng(0))
    assert noisy.noiser is not None
    with pytest.raises(ValueError, match="raw"):
        noisy.compute(data, want_raw=True)
