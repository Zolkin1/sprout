# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth deployment contract: normalisation parity, spec parsing, spec agreement.

The normalisation half is a PARITY test against the torch original
(``robot_rl...mdp.observations.depth.normalize_depth``), so it needs the isaaclab env;
the spec half is numpy-only.  A drift here silently rescales the depth image the
policy reads, which is why every invalid-reading class is pinned individually.

Run: ``python transfer/deploy/tests/test_depth_obs.py``  (or via pytest)
"""

from __future__ import annotations

import os
import math
import sys

import numpy as np
import pytest

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy.depth_obs import (
    REQUIRE_SAME_DEPTH_RESOLUTION,
    camera_half_angles_rad,  # noqa: E402
    EXTERO_CLIP_NP,
    assert_depth_specs_agree,
    denormalize_depth_np,
    invalid_depth_value,
    normalize_camera_images,
    normalize_depth_np,
    parse_depth_spec,
    sanitize_extero_np,
)

# The near camera's real parameters (g1_depth_env_cfg._NEAR + _depth_obs_params).
NEAR = dict(offset=0.75, scale=0.5, min_dist=0.10, max_dist=7.9)
SENTINEL = 8.0  # a missed ray reports exactly max_distance


def _cam(**overrides) -> dict:
    """A complete camera contract entry, with overrides applied."""
    cam = {
        "obs_term": "depth_near", "sensor": "depth_camera_near", "mount_body": "torso_link",
        "offset_pos": [0.1107625, 0.0, 0.0994260], "offset_quat_xyzw": [0.0, 0.53730, 0.0, 0.84339],
        "width": 30, "height": 26, "fx": 12.586494, "fy": 18.565924, "cx": 15.0, "cy": 13.0,
        "max_distance": 8.0, "update_period": 0.02, "clip": [-2.0, 2.0],
        "min_clip_val": 0.0, "max_clip_val": 0.0, **NEAR,
    }
    cam.update(overrides)
    return cam


##
# normalize_depth_np -- behaviour
##

def test_valid_depth_is_affine():
    """In-range readings pass straight through ``(d - offset) * scale``."""
    d = np.array([0.10, 1.0, 2.5, 7.9])
    out = normalize_depth_np(d, **NEAR)
    np.testing.assert_allclose(out, (d - 0.75) * 0.5)


def test_range_sentinel_reads_as_no_reading():
    """A missed ray (exactly ``max_distance``) lands on the invalid value, not at range.

    This is the whole point of the 0.1 m margin between ``max_dist`` and the sensor's
    ``max_distance``: without it a miss would look like a real surface 8 m away.
    """
    out = normalize_depth_np(np.array([SENTINEL]), **NEAR)
    assert out[0] == pytest.approx((0.0 - 0.75) * 0.5)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_becomes_the_invalid_value(bad):
    """NaN / +inf are finite-ised to ``max_clip_val`` before the transform."""
    out = normalize_depth_np(np.array([bad, 1.0]), **NEAR)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx((0.0 - 0.75) * 0.5)


def test_negative_infinity_takes_the_near_branch():
    """``-inf`` is caught by the too-NEAR test, which runs first.

    Pins the ordering that makes this port faithful: swapping the two tests would send
    ``-inf`` to ``max_clip_val`` instead.
    """
    out = normalize_depth_np(np.array([-np.inf]), min_clip_val=0.3, max_clip_val=9.0, **{
        k: v for k, v in NEAR.items() if k not in ("min_clip_val", "max_clip_val")
    })
    assert out[0] == pytest.approx((0.3 - 0.75) * 0.5)


def test_boundaries_are_inclusive():
    """Readings exactly at ``min_dist`` / ``max_dist`` are valid (strict comparisons)."""
    out = normalize_depth_np(np.array([NEAR["min_dist"], NEAR["max_dist"]]), **NEAR)
    np.testing.assert_allclose(out, (np.array([0.10, 7.9]) - 0.75) * 0.5)


def test_input_is_not_mutated():
    """The caller's buffer survives -- the caster reuses its ray-cast output."""
    d = np.array([0.01, 1.0, 99.0])
    before = d.copy()
    normalize_depth_np(d, **NEAR)
    np.testing.assert_array_equal(d, before)


def test_saturation_horizon_matches_the_obs_clip():
    """Beyond 4.75 m the +-2 obs clip saturates, so the policy cannot resolve further.

    A documented consequence of the contract worth keeping visible: the cameras are
    configured for 8 / 12 m but the observation cannot express more than 4.75 m.
    """
    horizon = 0.75 + 2.0 / 0.5
    out = normalize_depth_np(np.array([horizon - 0.1, horizon, horizon + 3.0]), **NEAR)
    assert out[0] < 2.0 and out[1] == pytest.approx(2.0) and out[2] > 2.0


##
# normalize_depth_np -- parity with the torch original
##

def test_parity_with_torch_normalize_depth():
    """Bit-for-bit agreement with the IsaacLab observation term over every class."""
    torch = pytest.importorskip("torch")
    from robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth import normalize_depth

    rng = np.random.default_rng(0)
    d = np.concatenate([
        rng.uniform(0.0, 13.0, 4096),
        np.array([0.0, 0.05, 0.10, 1.0, 7.9, 8.0, 12.0, np.nan, np.inf, -np.inf]),
    ]).reshape(2, -1)
    for params in (NEAR, dict(offset=0.75, scale=0.5, min_dist=0.15, max_dist=11.9)):
        got = normalize_depth_np(d, **params)
        want = normalize_depth(torch.from_numpy(d).double(), **params).numpy()
        err = float(np.abs(got - want).max())
        assert err == 0.0, f"max abs err {err:.3e} for {params}"


##
# sanitize_extero_np
##

def test_sanitize_clamps_per_source():
    """Depth keeps its +-2 band; the height scan is clamped to +-1."""
    assert EXTERO_CLIP_NP == {"height_scan": 1.0, "depth": 2.0}
    values = np.array([-3.0, -1.5, 0.0, 1.5, 3.0, np.nan, np.inf, -np.inf])
    depth = sanitize_extero_np(values, "depth")
    scan = sanitize_extero_np(values, "height_scan")
    assert np.isfinite(depth).all() and np.isfinite(scan).all()
    # A real 1.5 depth reading survives, but would have been truncated by the scan band.
    assert depth[3] == pytest.approx(1.5)
    assert scan[3] == pytest.approx(1.0)
    assert depth.max() == pytest.approx(2.0) and depth.min() == pytest.approx(-2.0)


##
# parse_depth_spec / denormalize / invalid value
##

def test_parse_returns_empty_for_no_cameras():
    """A height-scan policy has no camera contract, which is not an error."""
    assert parse_depth_spec(None) == []
    assert parse_depth_spec([]) == []


def test_parse_rejects_an_incomplete_entry():
    """A contract missing a field the caster needs fails at load, not mid-rollout."""
    cam = _cam()
    del cam["fx"]
    with pytest.raises(ValueError, match="fx"):
        parse_depth_spec([cam], source="tracker.onnx")


def test_parse_rejects_an_asymmetric_clip():
    """The deploy applies one clip magnitude, so an asymmetric band is refused."""
    with pytest.raises(ValueError, match="asymmetric"):
        parse_depth_spec([_cam(clip=[-1.0, 2.0])])


def test_denormalize_inverts_normalize():
    """Valid readings round-trip, so a diagnostic can read an image back in metres."""
    metres = np.array([0.10, 1.0, 4.75, 7.9])
    norm = normalize_depth_np(metres, **NEAR)
    np.testing.assert_allclose(denormalize_depth_np(norm, NEAR["offset"], NEAR["scale"]), metres)


def test_invalid_value_matches_what_normalisation_produces():
    """The "no reading" constant equals what an out-of-range pixel actually becomes."""
    cam = parse_depth_spec([_cam()])[0]
    assert invalid_depth_value(cam) == pytest.approx(
        normalize_depth_np(np.array([SENTINEL]), **NEAR)[0]
    )


##
# assert_depth_specs_agree
##

def test_identical_specs_agree():
    """The normal case: one camera configuration exported into both artifacts."""
    a = parse_depth_spec([_cam(), _cam(obs_term="depth_far", sensor="depth_camera_far")])
    assert_depth_specs_agree(a, [dict(c) for c in a])


def _near_far() -> list[dict]:
    """The shipped pair, near then far -- two lenses, matched by NAME not by position."""
    return parse_depth_spec([
        _cam(),
        _cam(obs_term="depth_far", sensor="depth_camera_far", max_distance=12.0),
    ])


def test_a_tracker_reading_fewer_cameras_agrees():
    """The shipped configuration: the policy reads the near camera, the generator both.

    The policy tracks a reference and does not need to read terrain ahead, so the far camera
    is the generator's instrument alone and the two contracts legitimately differ in length.
    """
    both = _near_far()
    assert_depth_specs_agree(both[:1], both)


def test_the_shared_camera_is_still_compared_across_a_subset():
    """Dropping the far camera must not stop the near one being checked."""
    both = _near_far()
    with pytest.raises(ValueError, match="depth_camera_near.mount_body"):
        assert_depth_specs_agree(parse_depth_spec([_cam(mount_body="pelvis_link")]), both)


def test_cameras_are_matched_by_name_not_by_position():
    """THE test that the subset relaxation did not become an index pairing.

    Tracker = [far], generator = [near, far]. Matched by name these agree; matched by index
    the far camera would be compared against the near one, whose range and optics differ --
    so a pass here is only possible if the match is by name.
    """
    both = _near_far()
    assert_depth_specs_agree(both[1:], both)


def test_a_tracker_camera_the_generator_lacks_is_refused():
    """The other direction is still an error, and the message says so."""
    both = _near_far()
    with pytest.raises(ValueError, match=r"does not have.*depth_camera_far|"
                                         r"depth_camera_far.*does not have"):
        assert_depth_specs_agree(both, both[:1])


def test_the_generators_own_sampling_prefix_is_not_a_different_camera():
    """``install_fm_depth_group`` names the generator's second grid ``fm_depth_camera_near``.

    That is one lens sampled twice, not two cameras, so the prefix is stripped before the
    match -- otherwise a split sampling would read as two artifacts with disjoint cameras.
    """
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_cam(sensor="fm_depth_camera_near", obs_term="fm_depth_near")])
    assert_depth_specs_agree(a, b)


def test_a_duplicated_camera_identity_is_refused():
    """Two entries resolving to one lens make every by-name match ambiguous."""
    with pytest.raises(ValueError, match="both resolve to"):
        assert_depth_specs_agree(parse_depth_spec([_cam()]),
                                 parse_depth_spec([_cam(), _cam()]))


@pytest.mark.parametrize(
    "field, value",
    [("offset_quat_xyzw", [0.0, 0.5, 0.0, 0.866]), ("mount_body", "pelvis_link"),
     ("max_distance", 12.0), ("scale", 0.25), ("clip", [-1.0, 1.0])],
)
def test_any_physical_disagreement_names_the_field(field, value):
    """Every PHYSICAL contract field is checked, and the error says which one differs.

    A mismatched mount or range is otherwise silent -- both networks accept the wrong
    image and just behave worse. These are refused whether or not the two artifacts are
    allowed to sample at different resolutions; the sampling fields have their own tests
    below.
    """
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_cam(**{field: value})])
    for strict in (True, False):
        with pytest.raises(ValueError, match=f"depth_camera_near.{field}"):
            assert_depth_specs_agree(a, b, require_same_resolution=strict)


def test_a_changed_focal_length_is_reported_as_a_different_lens():
    """``fx`` alone is not comparable once resolutions may differ -- the CONE is.

    With the sampling relaxed, fx is expected to move (it is linear in the width), so it
    is not compared field-wise. Moving it WITHOUT the width is a different lens, and the
    half-angle check is what still catches that -- reported as the fov it changes rather
    than as the field.
    """
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_cam(fx=21.42)])
    with pytest.raises(ValueError, match="depth_camera_near.fx"):
        assert_depth_specs_agree(a, b, require_same_resolution=True)
    with pytest.raises(ValueError, match="different lenses"):
        assert_depth_specs_agree(a, b, require_same_resolution=False)


def test_obs_term_naming_is_not_a_disagreement():
    """A generator has no observation terms, so term names are not compared."""
    a = parse_depth_spec([_cam(obs_term="depth_near")])
    b = parse_depth_spec([_cam(obs_term="whatever")])
    assert_depth_specs_agree(a, b)


def test_float_noise_within_tolerance_agrees():
    """A JSON round-trip must not read as a geometry mismatch."""
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_cam(fx=12.586494 + 1e-12)])
    assert_depth_specs_agree(a, b)


##
# normalize_camera_images -- the hardware ingestion path
##

FAR = dict(offset=0.75, scale=0.5, min_dist=0.15, max_dist=11.9)


def _far_cam(**overrides) -> dict:
    """The second camera: same resolution as the near one, different optics and range."""
    cam = _cam(obs_term="depth_far", sensor="depth_camera_far", max_distance=12.0, **FAR)
    cam.update(overrides)
    return cam


def test_ingestion_matches_per_camera_normalisation():
    """Each reading is normalised with its OWN parameters, stacked in contract order."""
    spec = parse_depth_spec([_cam(), _far_cam()])
    near = np.full(30 * 26, 1.5)
    far = np.full(30 * 26, 11.95)   # beyond the NEAR range, inside nothing for FAR either
    out = normalize_camera_images(spec, [near, far])
    assert out.shape == (2, 780)
    np.testing.assert_allclose(out[0], normalize_depth_np(near, **NEAR))
    np.testing.assert_allclose(out[1], normalize_depth_np(far, **FAR))
    # The far reading is out of range for BOTH, but only the far camera's own max_dist
    # decides that -- applying the near camera's parameters would be indistinguishable
    # here, so pin a range where the two genuinely differ.
    mid = np.full(30 * 26, 9.0)     # valid for FAR (max 11.9), invalid for NEAR (max 7.9)
    both = normalize_camera_images(spec, [mid, mid])
    assert both[0][0] == invalid_depth_value(spec[0])
    np.testing.assert_allclose(both[1][0], (9.0 - 0.75) * 0.5)


def test_ingestion_accepts_a_2d_image():
    """A ``(H, W)`` frame is equivalent to its row-major flattening."""
    spec = parse_depth_spec([_cam()])
    img = np.linspace(0.2, 7.0, 26 * 30).reshape(26, 30)
    np.testing.assert_allclose(
        normalize_camera_images(spec, [img])[0],
        normalize_camera_images(spec, [img.reshape(-1)])[0],
    )


def test_ingestion_handles_camera_dropouts():
    """NaN / inf / out-of-range all land on the camera's invalid value, as in sim."""
    spec = parse_depth_spec([_cam()])
    raw = np.full(30 * 26, 1.0)
    raw[:3] = [np.nan, np.inf, 99.0]
    out = normalize_camera_images(spec, [raw])[0]
    np.testing.assert_allclose(out[:3], invalid_depth_value(spec[0]))


def test_ingestion_rejects_a_wrong_sized_reading():
    """A resolution mismatch is caught here, not as a garbled observation."""
    spec = parse_depth_spec([_cam()])
    with pytest.raises(ValueError, match="pixel"):
        normalize_camera_images(spec, [np.zeros(100)])


def test_ingestion_rejects_a_missing_frame():
    """A camera that has never delivered is an error, not a silent blank image."""
    spec = parse_depth_spec([_cam()])
    with pytest.raises(ValueError, match="No depth reading"):
        normalize_camera_images(spec, [None])


def test_ingestion_rejects_a_camera_count_mismatch():
    """One reading per camera, in contract order -- a short list is a wiring bug."""
    spec = parse_depth_spec([_cam(), _far_cam()])
    with pytest.raises(ValueError, match="1 depth reading"):
        normalize_camera_images(spec, [np.zeros(30 * 26)])


def test_ingestion_rejects_differing_resolutions():
    """The images share one array, so mismatched cameras fail like they do in sim."""
    spec = parse_depth_spec([_cam(), _far_cam(width=10, height=6)])
    with pytest.raises(ValueError, match="differing pixel counts"):
        normalize_camera_images(spec, [np.zeros(30 * 26), np.zeros(60)])


def test_ingestion_of_no_cameras_is_empty():
    """A height-scan policy resolves an empty contract and ingests nothing."""
    assert normalize_camera_images([], []).size == 0


##
# Splitting the sampling: the generator may resample the policy's lenses
##

def _resampled_cam(scale: int, **overrides) -> dict:
    """``_cam()`` sampled on a ``scale``x finer grid -- the SAME lens, more rays.

    Derived the way the producer does it (``fx = width * focal / aperture``,
    ``cx = width / 2``), so this is what a real second camera pair at another resolution
    exports, not an approximation of one.
    """
    base = _cam()
    w, h = int(base["width"]) * scale, int(base["height"]) * scale
    cam = _cam(sensor="fm_depth_camera_near", width=w, height=h,
               fx=float(base["fx"]) * scale, fy=float(base["fy"]) * scale,
               cx=w / 2.0, cy=h / 2.0)
    cam.update(overrides)
    return cam


def test_resolution_split_is_opt_in():
    """A canary on the deployment-wide default, so re-tightening is a deliberate edit.

    The relaxation exists only because the FM's vision input is being changed without
    retraining the policy. When those converge again this flips back, and this test is what
    makes that a decision rather than a drift.
    """
    assert REQUIRE_SAME_DEPTH_RESOLUTION is False


def test_a_finer_sampling_of_the_same_lens_agrees():
    """The point of the whole exercise: 2x the rays through the same cone is allowed."""
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_resampled_cam(2)])
    assert_depth_specs_agree(a, b, require_same_resolution=False)


def test_a_finer_sampling_is_refused_when_strict():
    """...and refused when the two are required to sample identically."""
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_resampled_cam(2)])
    with pytest.raises(ValueError, match="depth_camera_near.(width|height)"):
        assert_depth_specs_agree(a, b, require_same_resolution=True)


def test_a_different_lens_at_a_new_resolution_is_still_refused():
    """THE test that the relaxation is not a hole.

    Resizing without rescaling fx/fy is a narrower cone on a bigger grid -- a different
    lens wearing a resampling's clothes. Dropping width/height from the comparison and
    stopping there would have accepted it.
    """
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_resampled_cam(2, fx=float(_cam()["fx"]))])   # width doubled, fx not
    with pytest.raises(ValueError, match="different lenses"):
        assert_depth_specs_agree(a, b, require_same_resolution=False)


def test_an_off_centre_principal_point_is_refused_even_when_the_total_fov_matches():
    """Half-angles, not their SUM: a camera can match the total FOV and still be wrong.

    Comparing the summed horizontal FOV would be the obvious cheaper check, and this is the
    camera it lets through: the principal point is pushed off-centre and the focal length
    re-solved so the two half-angles trade off to the same total. Every pixel maps to a
    different ray than the policy's camera, and a total-FOV check cannot see it.

    ``fx`` has to be solved for numerically because ``atan`` is not linear -- at fixed
    ``fx`` the sum is maximal at ``cx = W/2``, so no other principal point can match it.
    """
    base = _cam()
    target = sum(camera_half_angles_rad(base)[:2])
    shifted = _resampled_cam(2)
    width = float(shifted["width"])
    shifted["cx"] = width * 0.35                       # deliberately off-centre

    def h_sum(fx: float) -> float:
        return (math.atan(shifted["cx"] / fx)
                + math.atan((width - shifted["cx"]) / fx))

    lo, hi = 1e-3, 1e4                                  # h_sum decreases in fx
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if h_sum(mid) > target else (lo, mid)
    shifted["fx"] = 0.5 * (lo + hi)

    a, b = parse_depth_spec([base]), parse_depth_spec([shifted])
    left, right, _, _ = camera_half_angles_rad(shifted)
    assert (left + right) == pytest.approx(target, abs=1e-9), "fixture is not a wash"
    assert abs(left - right) > 0.05, "fixture is not actually off-centre"
    with pytest.raises(ValueError, match="hfov_(left|right)"):
        assert_depth_specs_agree(a, b, require_same_resolution=False)


def test_physical_fields_are_refused_across_a_split_sampling():
    """Relaxing the grid must not relax the camera: a moved mount still fails."""
    a = parse_depth_spec([_cam()])
    b = parse_depth_spec([_resampled_cam(2, mount_body="pelvis_link")])
    with pytest.raises(ValueError, match="depth_camera_near.mount_body"):
        assert_depth_specs_agree(a, b, require_same_resolution=False)


##
# The convolutional fold: (H, W) is load-bearing, not just H*W
##

def _fm_ctx(**overrides):
    """A minimal :class:`FmContext` stand-in for ``_resolve_depth_cameras``."""
    from types import SimpleNamespace
    base = dict(extero_source="depth", extero_sensor_count=2, extero_sensor_dim=30 * 26,
                extero_spec=[],
                extero_encoder="cnn", extero_image_hw=[(26, 30), (26, 30)],
                extero_tokens_per_sensor=16)
    base.update(overrides)
    return SimpleNamespace(**base)


def _resolve(ctx, spec):
    """Run ``GenTrackerPolicy._resolve_depth_cameras`` against a hand-built context."""
    from types import SimpleNamespace
    from deploy.gen_tracker import GenTrackerPolicy
    policy = object.__new__(GenTrackerPolicy)
    policy.ctx = ctx
    policy._trk = SimpleNamespace(get_depth_camera_spec=lambda: list(spec))
    GenTrackerPolicy._resolve_depth_cameras(policy)
    return policy._depth_spec


def test_matching_fold_is_accepted():
    """The shipped pairing: contract HxW equals what the graph folds."""
    spec = [_cam(), _far_cam()]
    assert len(_resolve(_fm_ctx(), spec)) == 2


def test_transposed_camera_is_refused_by_the_fold_check():
    """30x26 and 26x30 are the SAME pixel count and different images.

    The pixel-count check passes either way, so before the fold shape travelled this was
    undetectable -- a convolution would read every kernel's neighbourhood off a transposed
    frame with every tensor the right size. Only reachable for a convolutional graph, which
    is the only one it can hurt.
    """
    spec = [_cam(width=26, height=30), _far_cam(width=26, height=30)]
    with pytest.raises(ValueError, match="transposed image is scrambled"):
        _resolve(_fm_ctx(), spec)


def test_a_flat_graph_does_not_check_the_fold():
    """An MLP graph carries no image shape, and genuinely does not care about pixel order."""
    spec = [_cam(width=26, height=30), _far_cam(width=26, height=30)]
    assert len(_resolve(_fm_ctx(extero_encoder="mlp", extero_image_hw=[],
                                extero_tokens_per_sensor=1), spec)) == 2


def test_pixel_count_mismatch_is_still_refused():
    """The pre-existing width check must survive alongside the new fold check."""
    spec = [_cam(width=10, height=6), _far_cam(width=10, height=6)]
    with pytest.raises(ValueError, match="wide"):
        _resolve(_fm_ctx(extero_image_hw=[(6, 10), (6, 10)]), spec)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
