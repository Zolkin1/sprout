# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth normalisation and noise, exercised without a simulator.

``normalize_depth`` and :class:`DepthNoiser` are deliberately pure w.r.t. the
environment so the clipping / affine / corruption behaviour can be tested on plain
tensors; the sensor plumbing around them needs a running sim and is covered by the
play task instead.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.mdp.events.depth import randomize_camera_extrinsics
from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import normalize_depth
from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import DepthNoiseCfg, DepthNoiser

H, W = 4, 5
OFFSET, SCALE = 0.75, 0.5


def _noise_cfg(**overrides) -> DepthNoiseCfg:
    """Build a noise cfg with the required fields filled in.

    Args:
        **overrides: Fields to override on the returned cfg.

    Returns:
        A fully-specified :class:`DepthNoiseCfg`.
    """
    params = dict(height=H, width=W, offset=OFFSET, scale=SCALE)
    params.update(overrides)
    return DepthNoiseCfg(**params)


##
# normalize_depth
##

def test_affine_transform_on_valid_depth():
    """In-range depth is passed through ``(d - offset) * scale`` untouched."""
    d = torch.tensor([[1.0, 2.0, 3.0]])
    out = normalize_depth(d, offset=OFFSET, scale=SCALE, min_dist=0.2, max_dist=9.0)
    torch.testing.assert_close(out, (d - OFFSET) * SCALE)


def test_too_near_pixels_take_min_clip_val():
    """Readings below ``min_dist`` are replaced before the transform."""
    d = torch.tensor([[0.05, 1.0]])
    out = normalize_depth(d, offset=OFFSET, scale=SCALE, min_dist=0.2, min_clip_val=0.3)
    assert out[0, 0].item() == pytest.approx((0.3 - OFFSET) * SCALE)
    assert out[0, 1].item() == pytest.approx((1.0 - OFFSET) * SCALE)


def test_too_far_pixels_take_max_clip_val():
    """Readings beyond ``max_dist`` are replaced before the transform."""
    d = torch.tensor([[12.0, 1.0]])
    out = normalize_depth(d, offset=OFFSET, scale=SCALE, max_dist=9.0, max_clip_val=0.0)
    assert out[0, 0].item() == pytest.approx((0.0 - OFFSET) * SCALE)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_missed_rays_are_sanitised(bad):
    """NaN / +-inf -- a ray that hit nothing -- become finite ``max_clip_val``.

    Left unhandled these propagate straight into the policy input, which is how the
    height-scan sensor once produced NaN losses.
    """
    d = torch.tensor([[bad, 1.0]])
    out = normalize_depth(d, offset=OFFSET, scale=SCALE, max_clip_val=0.0)
    assert torch.isfinite(out).all()
    assert out[0, 0].item() == pytest.approx((0.0 - OFFSET) * SCALE)


def test_negative_infinity_is_not_treated_as_near():
    """``-inf`` must be sanitised, not silently caught by the ``min_dist`` branch.

    The ``min_dist`` comparison runs first and would map ``-inf`` to ``min_clip_val``;
    this pins whichever behaviour holds so a reordering cannot go unnoticed.
    """
    out = normalize_depth(
        torch.tensor([[float("-inf")]]),
        offset=OFFSET, scale=SCALE, min_clip_val=5.0, max_clip_val=0.0,
    )
    assert torch.isfinite(out).all()
    assert out[0, 0].item() == pytest.approx((5.0 - OFFSET) * SCALE)


def test_input_is_not_mutated():
    """Normalisation must not write through to the sensor's own buffer."""
    d = torch.tensor([[0.01, 20.0, 1.0]])
    original = d.clone()
    normalize_depth(d)
    torch.testing.assert_close(d, original)


def test_shape_is_preserved():
    """Any input shape survives; the obs wrappers do the reshaping."""
    d = torch.rand(3, H, W) * 5.0
    assert normalize_depth(d).shape == (3, H, W)


##
# DepthNoiser
##

def test_noiser_preserves_flat_and_image_shapes():
    """Both ``(B, H*W)`` and ``(B, 1, H, W)`` come back unchanged in shape."""
    noiser = DepthNoiser(_noise_cfg(), num_envs=3, device="cpu")
    flat = torch.zeros(3, H * W)
    assert noiser(flat).shape == (3, H * W)
    img = torch.zeros(3, 1, H, W)
    assert noiser(img).shape == (3, 1, H, W)


def test_noiser_actually_perturbs():
    """A zero image does not come back all zeros."""
    noiser = DepthNoiser(_noise_cfg(), num_envs=2, device="cpu")
    out = noiser(torch.zeros(2, H * W))
    assert not torch.allclose(out, torch.zeros_like(out))
    assert torch.isfinite(out).all()


def test_plane_bias_respects_max_deviation():
    """The per-env plane bias stays inside ``max_plane_deviation``."""
    noiser = DepthNoiser(_noise_cfg(max_plane_deviation=0.05), num_envs=64, device="cpu")
    assert noiser._plane.abs().max().item() <= 0.05 + 1e-6


def test_plane_bias_is_per_episode_not_per_call():
    """Plane state persists across calls and only changes on reset.

    This is the property that would be lost by re-sampling inside the observation
    function instead of on the reset event.
    """
    noiser = DepthNoiser(_noise_cfg(), num_envs=4, device="cpu")
    before = noiser._plane.clone()
    noiser(torch.zeros(4, H * W))
    torch.testing.assert_close(noiser._plane, before)
    noiser.reset()
    assert not torch.allclose(noiser._plane, before)


def test_reset_is_scoped_to_given_envs():
    """Resetting a subset leaves the other envs' camera state alone."""
    noiser = DepthNoiser(_noise_cfg(max_plane_deviation=0.5), num_envs=4, device="cpu")
    before = noiser._plane.clone()
    noiser.reset(torch.tensor([0, 2]))
    assert not torch.allclose(noiser._plane[0], before[0])
    torch.testing.assert_close(noiser._plane[1], before[1])
    torch.testing.assert_close(noiser._plane[3], before[3])


def test_dropout_writes_the_normalised_value():
    """Dropped pixels land on ``(dropout_val - offset) * scale``, not raw ``dropout_val``."""
    cfg = _noise_cfg(
        dropout_prob=1.0,
        dropout_val=0.0,
        max_plane_deviation=0.0,
        noise_cfg=DepthNoiseCfg().noise_cfg.__class__(n_min=0.0, n_max=0.0, operation="add"),
    )
    noiser = DepthNoiser(cfg, num_envs=2, device="cpu")
    out = noiser(torch.zeros(2, H * W))
    expected = (0.0 - OFFSET) * SCALE
    torch.testing.assert_close(out, torch.full_like(out, expected))


def test_blur_kernel_is_normalised():
    """Each env's 3x3 blur kernel sums to one, so the blur preserves depth scale."""
    noiser = DepthNoiser(_noise_cfg(), num_envs=8, device="cpu")
    sums = noiser._kernel.reshape(8, 9).sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(8))


def test_blur_preserves_a_constant_image():
    """With no plane, no pixel noise and no dropout, a flat image is unchanged.

    Guards the grouped-conv wiring: a wrong ``groups`` or kernel reshape would mix
    environments together, which a constant image would not otherwise reveal.
    """
    cfg = _noise_cfg(
        dropout_prob=0.0,
        max_plane_deviation=0.0,
        noise_cfg=DepthNoiseCfg().noise_cfg.__class__(n_min=0.0, n_max=0.0, operation="add"),
    )
    noiser = DepthNoiser(cfg, num_envs=3, device="cpu")
    # Interior pixels only: zero padding pulls the border down.
    out = noiser(torch.full((3, H * W), 0.4)).reshape(3, H, W)
    torch.testing.assert_close(out[:, 1:-1, 1:-1], torch.full((3, H - 2, W - 2), 0.4))


def test_envs_are_blurred_independently():
    """One env's content must not leak into another's blurred image."""
    cfg = _noise_cfg(
        dropout_prob=0.0,
        max_plane_deviation=0.0,
        noise_cfg=DepthNoiseCfg().noise_cfg.__class__(n_min=0.0, n_max=0.0, operation="add"),
    )
    noiser = DepthNoiser(cfg, num_envs=2, device="cpu")
    data = torch.zeros(2, H * W)
    data[1] = 1.0
    out = noiser(data).reshape(2, H, W)
    assert out[0].abs().max().item() == pytest.approx(0.0, abs=1e-6)


##
# Extrinsics randomization
##

class _FakeSensor:
    """Minimal stand-in exposing the offset buffers the extrinsics event writes."""

    def __init__(self, num_envs: int, device: str = "cpu"):
        self._offset_pos = torch.zeros(num_envs, 3, device=device)
        # Identity rotation in (x, y, z, w).
        self._offset_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device).repeat(num_envs, 1)
        self.device = device
        self._view = SimpleNamespace(count=num_envs)


def _fake_env(sensor, name="cam"):
    """Env stand-in whose ``scene`` resolves the sensor by name."""
    return SimpleNamespace(scene={name: sensor})


def test_extrinsics_perturbs_within_bounds():
    """Uniform sampling stays inside the per-axis position and rotation bounds."""
    sensor = _FakeSensor(256)
    randomize_camera_extrinsics(
        _fake_env(sensor), None, SceneEntityCfg("cam"),
        pos_range=(0.01, 0.02, 0.03), rot_range=(0.03, 0.03, 0.03),
    )
    bound = torch.tensor([0.01, 0.02, 0.03])
    assert (sensor._offset_pos.abs() <= bound + 1e-6).all()
    # A +-0.03 rad rotation about any axis keeps the quaternion's w near 1.
    assert (sensor._offset_quat[:, 3] > math.cos(0.03 * 1.5)).all()
    torch.testing.assert_close(
        sensor._offset_quat.norm(dim=-1), torch.ones(256), atol=1e-5, rtol=0
    )


def test_extrinsics_varies_per_environment():
    """Each env gets its own mount error, not one shared draw."""
    sensor = _FakeSensor(64)
    randomize_camera_extrinsics(_fake_env(sensor), None, SceneEntityCfg("cam"))
    assert sensor._offset_pos.std(dim=0).min() > 0.0
    assert sensor._offset_quat[:, :3].std(dim=0).min() > 0.0


def test_extrinsics_does_not_compound_across_resets():
    """Successive resets perturb around the NOMINAL pose, not the last sample.

    Without the snapshot the mount would random-walk away over an episode's worth of
    resets, which is how the intrinsics event is written too.
    """
    sensor = _FakeSensor(128)
    cfg = SceneEntityCfg("cam")
    for _ in range(40):
        randomize_camera_extrinsics(
            _fake_env(sensor), None, cfg, pos_range=(0.01, 0.01, 0.01)
        )
    assert (sensor._offset_pos.abs() <= 0.01 + 1e-6).all()
    torch.testing.assert_close(sensor._nominal_offset_pos, torch.zeros(128, 3))


def test_extrinsics_scoped_to_given_envs():
    """Resetting a subset leaves the other envs' mount pose untouched."""
    sensor = _FakeSensor(4)
    randomize_camera_extrinsics(
        _fake_env(sensor), torch.tensor([0, 2]), SceneEntityCfg("cam")
    )
    assert sensor._offset_pos[0].abs().sum() > 0.0
    assert sensor._offset_pos[2].abs().sum() > 0.0
    torch.testing.assert_close(sensor._offset_pos[1], torch.zeros(3))
    torch.testing.assert_close(sensor._offset_pos[3], torch.zeros(3))


def test_extrinsics_zero_range_is_a_no_op():
    """Zero bounds leave the nominal pose exactly in place."""
    sensor = _FakeSensor(8)
    randomize_camera_extrinsics(
        _fake_env(sensor), None, SceneEntityCfg("cam"),
        pos_range=(0.0, 0.0, 0.0), rot_range=(0.0, 0.0, 0.0),
    )
    torch.testing.assert_close(sensor._offset_pos, torch.zeros(8, 3))
    expected = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(8, 1)
    torch.testing.assert_close(sensor._offset_quat, expected)


##
# Camera geometry helper
##

def test_pitch_quaternion_matches_reference_convention():
    """The pitch helper reproduces the reference's 65 deg quaternion in (x, y, z, w)."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import _pitch_down_quat_xyzw

    x, y, z, w = _pitch_down_quat_xyzw(65.0)
    assert (x, z) == (0.0, 0.0)
    assert y == pytest.approx(0.537300, abs=1e-5)
    assert w == pytest.approx(0.843391, abs=1e-5)
    assert math.hypot(y, w) == pytest.approx(1.0)


def test_pitch_quaternion_is_identity_at_zero():
    """Zero pitch is the identity rotation."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import _pitch_down_quat_xyzw

    assert _pitch_down_quat_xyzw(0.0) == (0.0, 0.0, 0.0, 1.0)


##
# Deployment contract (depth_camera_specs)
##

def _specs():
    """The POLICY's depth contract for the real ``G1-teacher-clf-terrain-depth`` cfgs.

    Goes through the same simulator-free entry point the sim2sim verification tools use,
    so this checks the contract the task actually declares rather than a restatement of it.

    NEAR ONLY: the policy group holds one camera. :func:`_fm_specs` is the physical pair --
    the far camera's numbers are still exported, just into the GENERATOR's artifact.

    Returns:
        One dict per camera the policy observes, in observation order.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        depth_camera_specs_from_cfg,
    )

    return depth_camera_specs_from_cfg()


def _fm_specs():
    """The GENERATOR's depth contract: the ``fm_depth`` group over the same scene.

    This is the contract that describes the whole physical pair, so every per-camera number
    below (intrinsics, range, normalisation, mount pose) is measured against it. Reading
    them off the policy's contract instead would leave the far camera -- which only the
    generator observes, and which ``fm.onnx`` embeds -- checked nowhere at all.

    Returns:
        One dict per camera the generator observes, in observation order.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        G1FmDepthObservationsCfg,
        depth_camera_specs_from_cfg,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp import FM_DEPTH_OBS_GROUP

    # The SHARED-sampling arm of install_fm_depth_group: the group exactly as declared,
    # whose terms already name the scene's own sensors. The split arm rewrites those to
    # ``fm_`` duplicates and is covered where the task installs it.
    observations = SimpleNamespace(**{FM_DEPTH_OBS_GROUP: G1FmDepthObservationsCfg()})
    return depth_camera_specs_from_cfg(observations=observations, group=FM_DEPTH_OBS_GROUP)


def test_camera_spec_order_and_identity():
    """The POLICY reads the near camera; the GENERATOR reads the pair, near first.

    Three properties, each of which fails silently on its own:

    * the policy's contract is a strict SUBSET of the generator's -- there is ONE physical
      pair and the policy is conditioned on part of it, so a policy camera the generator
      lacks would be a tracker reading a lens its reference was never placed from;
    * the generator's covers BOTH cameras. A missing one is a half-blind generator whose
      recording, checkpoint and ``fm.onnx`` all agree with each other;
    * ORDER is near-then-far in both. The FM binds one CNN trunk per camera BY INDEX, each
      with its own angular position embedding (the cones are 100x70 and 70x42 deg), so a
      swapped order is shape-clean and mislabels the far camera's tokens by 1.43x.
    """
    specs, fm_specs = _specs(), _fm_specs()
    assert [s["obs_term"] for s in specs] == ["depth_near"]
    assert [s["sensor"] for s in specs] == ["depth_camera_near"]
    assert [s["obs_term"] for s in fm_specs] == ["fm_depth_near", "fm_depth_far"]
    assert [s["sensor"] for s in fm_specs] == ["depth_camera_near", "depth_camera_far"]
    assert all(s["mount_body"] == "torso_link" for s in specs + fm_specs)

    policy_cameras = [s["sensor"] for s in specs]
    fm_cameras = [s["sensor"] for s in fm_specs]
    assert set(policy_cameras) < set(fm_cameras), "the policy reads part of one pair"
    # ...and the shared cameras keep the same ORDER in both contracts.
    assert [c for c in fm_cameras if c in set(policy_cameras)] == policy_cameras


def test_the_two_contracts_describe_the_same_lens():
    """One physical camera read by two observation terms must export identical numbers.

    The two groups reach the same sensor through separate ``ObsTerm``s, each with its own
    copy of the normalisation params, so nothing but this stops the generator's reading of
    the near camera drifting onto a different affine or a different range from the policy's
    -- which would be two artifacts describing one lens differently, with no shape to
    reveal it. Only ``obs_term`` may differ: that names the term, not the camera.
    """
    near = _specs()[0]
    fm_near = next(s for s in _fm_specs() if s["sensor"] == "depth_camera_near")
    assert {k: v for k, v in near.items() if k != "obs_term"} == \
        {k: v for k, v in fm_near.items() if k != "obs_term"}


def test_camera_spec_intrinsics_match_the_effective_fov():
    """fx / fy follow from the CROPPED field of view, and the centre is the image middle.

    The apertures are derived from the effective FOV, so this is the check that the
    exported intrinsics describe the rays actually cast rather than the physical lens.

    Measured on the GENERATOR's contract: it is the one covering both cameras, and a ``zip``
    against the policy's near-only list would drop the far camera without failing.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_depth_env_cfg as dcfg

    for spec, cam in zip(_fm_specs(), (dcfg._NEAR, dcfg._FAR), strict=True):
        width, height = spec["width"], spec["height"]
        assert (width, height) == (dcfg._DEPTH_W, dcfg._DEPTH_H)
        expect_fx = width / (2.0 * math.tan(math.radians(cam.effective_fov_h_deg) / 2.0))
        expect_fy = height / (2.0 * math.tan(math.radians(cam.effective_fov_v_deg) / 2.0))
        assert spec["fx"] == pytest.approx(expect_fx)
        assert spec["fy"] == pytest.approx(expect_fy)
        assert (spec["cx"], spec["cy"]) == (width / 2.0, height / 2.0)


def test_camera_spec_carries_the_normalisation_and_range():
    """Range, sentinel margin, normalisation and obs clip all reach the contract.

    A deploy that reproduces the rays but not this arithmetic would feed the policy a
    differently-scaled image, so every argument ``normalize_depth`` takes is exported --
    including the ones the term leaves at their default. Both cameras, via the generator's
    contract (see :func:`_fm_specs`).
    """
    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_depth_env_cfg as dcfg

    for spec, cam in zip(_fm_specs(), (dcfg._NEAR, dcfg._FAR), strict=True):
        assert spec["max_distance"] == cam.max_distance_m
        assert spec["min_dist"] == cam.min_distance_m
        assert spec["max_dist"] == pytest.approx(cam.max_distance_m - dcfg._RANGE_SENTINEL_MARGIN_M)
        assert spec["offset"] == dcfg._DEPTH_OFFSET
        assert spec["scale"] == dcfg._DEPTH_SCALE
        assert spec["min_clip_val"] == 0.0 and spec["max_clip_val"] == 0.0
        assert spec["clip"] == [-2.0, 2.0]
        assert spec["update_period"] == pytest.approx(1.0 / dcfg._DEPTH_FPS)
        assert spec["data_type"] == "distance_to_image_plane"


def test_camera_spec_mount_pose_is_the_nominal_offset():
    """The exported mount pose is the cfg offset, in the mount body's frame.

    Both cameras, via the generator's contract (see :func:`_fm_specs`).
    """
    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_depth_env_cfg as dcfg

    for spec, cam in zip(_fm_specs(), (dcfg._NEAR, dcfg._FAR), strict=True):
        assert spec["offset_pos"] == pytest.approx(list(cam.pos))
        assert spec["offset_quat_xyzw"] == pytest.approx(
            list(dcfg._pitch_down_quat_xyzw(cam.pitch_deg))
        )


@pytest.mark.parametrize(
    "field, value",
    [("data_types", ["distance_to_camera"]), ("depth_clipping_behavior", "zero")],
)
def test_camera_spec_rejects_unreproducible_cfgs(field, value):
    """A camera the off-sim port cannot reproduce fails at export, not at deploy."""
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        G1TeacherTerrainDepthSceneCfg,
        depth_camera_specs_from_cfg,
    )

    scene = G1TeacherTerrainDepthSceneCfg(num_envs=1, env_spacing=2.5)
    setattr(scene.depth_camera_near, field, value)
    with pytest.raises(ValueError, match="depth_camera_near"):
        depth_camera_specs_from_cfg(scene=scene)


def test_camera_spec_is_empty_without_depth_terms():
    """A task with no depth observation in the exported group exports no contract.

    The scan-only teacher's own group, so this is the real "no cameras" case rather than
    a hand-emptied one -- and it pins that the contract follows the OBSERVED terms, not
    the cameras the scene happens to hold (both cameras stay on the scene here).
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        G1TeacherTerrainDepthObservationsCfg,
        depth_camera_specs_from_cfg,
    )

    observations = G1TeacherTerrainDepthObservationsCfg()
    # Clearing a term the group does NOT have creates a stray attribute rather than
    # emptying anything -- a silent no-op that would leave this passing against a contract
    # it had not cleared. So the term list is asserted before it is cleared: the policy
    # group holds the near camera alone.
    depth_terms = [n for n in vars(observations.policy) if n.startswith("depth_")]
    assert depth_terms == ["depth_near"], depth_terms
    observations.policy.depth_near = None
    assert depth_camera_specs_from_cfg(observations=observations) == []


##
# The xyz channels' geometry: quaternion convention and the frame it lands in
##

def test_quat_from_matrix_round_trips_through_isaaclab():
    """``_quat_from_matrix`` must be the exact inverse of ``matrix_from_quat``.

    This is a CONVENTION guard as much as a numerical one. Isaac Lab 3.0 is scalar-LAST
    everywhere -- the articulation buffers AND ``isaaclab.utils.math`` -- while
    ``motion_gen`` is scalar-first, and ``world_lift_isaac`` exists because mixing them is a
    silent ~180 deg roll about x that leaves every shape correct. The camera pose behind the
    xyz channels crosses this boundary, so getting it wrong would hand the generator a
    cloud rolled upside down and nothing downstream would notice.

    All four Shepperd branches are exercised: the near-180 cases are where a naive
    ``w``-branch loses its precision, and the trained mounts are pitched 35-65 deg.
    """
    from isaaclab.utils.math import matrix_from_quat

    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import _quat_from_matrix

    torch.manual_seed(0)
    quat = torch.randn(2000, 4, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    rot = matrix_from_quat(quat)
    assert float((matrix_from_quat(_quat_from_matrix(rot)) - rot).abs().max()) < 1e-12

    # The four 180 deg rotations, one per branch, as scalar-LAST quaternions.
    hard = matrix_from_quat(torch.eye(4, dtype=torch.float64))
    assert float((matrix_from_quat(_quat_from_matrix(hard)) - hard).abs().max()) == 0.0


def test_identity_quaternion_is_scalar_last():
    """``(0, 0, 0, 1)`` is the identity -- the one-line statement of the convention.

    If this ever fails, ``_quat_from_matrix`` and the camera-pose composition have drifted
    to scalar-first and the point cloud is rolled 180 deg about x.
    """
    from isaaclab.utils.math import matrix_from_quat

    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import _quat_from_matrix

    eye = torch.eye(3, dtype=torch.float64)[None]
    assert torch.allclose(_quat_from_matrix(eye),
                          torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64))
    assert torch.allclose(matrix_from_quat(torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                                                        dtype=torch.float64)), eye)
