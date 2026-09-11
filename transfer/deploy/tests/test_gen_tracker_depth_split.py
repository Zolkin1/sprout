# Copyright 2026 Zachary Olkin. All rights reserved.

"""``GenTrackerPolicy`` when the generator samples the depth cameras on its own grid.

Two contracts are resolved rather than one: the tracker's (keyed by observation term, which
is how ``_build_obs`` finds its cameras) and the generator's (which is what every sensor-row
width and image-fold check has to be measured against). Getting that backwards is silent --
the checks pass against the wrong list and the generator is fed the wrong-width array.

No ONNX and no simulator: the contracts and the context are built by hand and the real
resolution method is run over them.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.depth_obs import camera_key, parse_depth_spec  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402

_FOCAL = 2.2
_APERTURE = 5.24   # the near camera's effective aperture; only the ratio matters


def _cam(width: int, height: int, sensor: str, obs_term: str = "depth_near", **over) -> dict:
    """A camera contract entry SAMPLING one fixed lens on a ``width x height`` grid."""
    cam = {
        "obs_term": obs_term, "sensor": sensor, "mount_body": "torso_link",
        "offset_pos": [0.1107625, 0.0, 0.0994260],
        "offset_quat_xyzw": [0.0, 0.53730, 0.0, 0.84339],
        "width": width, "height": height,
        "fx": width * _FOCAL / _APERTURE, "fy": height * _FOCAL / _APERTURE,
        "cx": width / 2.0, "cy": height / 2.0,
        "max_distance": 8.0, "update_period": 0.02, "clip": [-2.0, 2.0],
        "offset": 0.75, "scale": 0.5, "min_dist": 0.10, "max_dist": 7.9,
        "min_clip_val": 0.0, "max_clip_val": 0.0,
    }
    cam.update(over)
    return cam


def _policy(tracker_spec: list[dict], fm_spec: list[dict], **ctx_over) -> GenTrackerPolicy:
    """A ``GenTrackerPolicy`` with its two contracts resolved, without loading any ONNX."""
    hw = [(int(c["height"]), int(c["width"])) for c in fm_spec]
    ctx = dict(extero_source="depth", extero_sensor_count=len(fm_spec) or 1,
               extero_sensor_dim=(int(fm_spec[0]["width"]) * int(fm_spec[0]["height"])
                                  if fm_spec else 1),
               extero_spec=fm_spec, extero_encoder="cnn", extero_image_hw=hw,
               extero_tokens_per_sensor=16)
    ctx.update(ctx_over)
    policy = object.__new__(GenTrackerPolicy)
    policy.ctx = SimpleNamespace(**ctx)
    policy._trk = SimpleNamespace(get_depth_camera_spec=lambda: [dict(c) for c in tracker_spec])
    GenTrackerPolicy._resolve_depth_cameras(policy)
    return policy


def _pair(width: int, height: int, prefix: str = "") -> list[dict]:
    """A near/far pair sampled at one resolution."""
    return parse_depth_spec([
        _cam(width, height, f"{prefix}depth_camera_near", "depth_near"),
        _cam(width, height, f"{prefix}depth_camera_far", "depth_far"),
    ])


##
# The unsplit case must be untouched
##

def test_identical_sampling_is_not_split():
    """The usual case: one grid, one contract, nothing new required of the runner."""
    p = _policy(_pair(30, 26), _pair(30, 26))
    assert p.depth_sampling_is_split() is False
    assert p.get_fm_depth_camera_spec() == p.get_depth_camera_spec()
    # ...and the generator reads the tracker's array, as it always did.
    state = {"depth": np.zeros((2, 30 * 26))}
    assert p._fm_depth_reading(state).shape == (2, 780)


def test_a_tracker_without_cameras_falls_back_to_the_generators_contract():
    """A height-scan tracker paired with a depth generator: the FM's spec is all there is."""
    fm = _pair(30, 26)
    p = _policy([], fm)
    assert p.get_depth_camera_spec() == fm and p.get_fm_depth_camera_spec() == fm
    assert p.depth_sampling_is_split() is False


##
# The split case
##

def test_both_contracts_are_resolved_and_the_split_is_reported():
    """Two grids on one lens: each artifact keeps its OWN contract and the split is reported.

    ``depth_sampling_is_split()`` is what tells the runner it owes a second reading, and the
    resolved ``get_depth_camera_spec()`` stays the TRACKER's because ``_build_obs`` looks its
    cameras up by observation term. This is the SAMPLING relaxation, independent of the camera
    SET one below.
    """
    p = _policy(_pair(30, 26), _pair(60, 52, prefix="fm_"))
    assert p.depth_sampling_is_split() is True
    assert [(c["width"], c["height"]) for c in p.get_depth_camera_spec()] == [(30, 26)] * 2
    assert [(c["width"], c["height"]) for c in p.get_fm_depth_camera_spec()] == [(60, 52)] * 2
    # The tracker still wins the resolved contract: _build_obs looks its cameras up BY TERM.
    assert [c["obs_term"] for c in p.get_depth_camera_spec()] == ["depth_near", "depth_far"]


def test_the_width_check_is_measured_against_the_generators_own_contract():
    """The bug this guards: checking ``extero_sensor_dim`` against the TRACKER's cameras.

    With the samplings split, the tracker's pixel count is not the generator's, so a check
    pointed at the tracker's list would either reject a correct pair or accept a wrong one.
    Here the generator's graph is 60x52 and its contract agrees, while the tracker's is
    30x26 -- which must NOT be what the width is compared to.
    """
    p = _policy(_pair(30, 26), _pair(60, 52, prefix="fm_"))
    assert p.ctx.extero_sensor_dim == 60 * 52

    with pytest.raises(ValueError, match="sensor row holds"):
        _policy(_pair(30, 26), _pair(60, 52, prefix="fm_"), extero_sensor_dim=30 * 26)


def test_the_image_fold_is_checked_against_the_generators_own_contract():
    """Same for the (H, W) fold: transposed against the FM's cameras must still be caught."""
    fm = _pair(60, 52, prefix="fm_")
    with pytest.raises(ValueError, match="transposed image is scrambled"):
        _policy(_pair(30, 26), fm, extero_image_hw=[(60, 52), (60, 52)])   # swapped H/W


def test_a_split_sampling_needs_its_own_reading():
    """``state["depth"]`` is the TRACKER's and is the wrong width -- say so, do not resize."""
    p = _policy(_pair(30, 26), _pair(60, 52, prefix="fm_"))
    with pytest.raises(KeyError, match="state\\['fm_depth'\\]"):
        p._fm_depth_reading({"depth": np.zeros((2, 30 * 26))})
    got = p._fm_depth_reading({"depth": np.zeros((2, 780)), "fm_depth": np.zeros((2, 3120))})
    assert got.shape == (2, 3120)


def test_a_different_lens_is_still_refused_across_a_split():
    """Relaxing the grid must not relax the camera."""
    fm = _pair(60, 52, prefix="fm_")
    for cam in fm:
        cam["mount_body"] = "pelvis_link"
    with pytest.raises(ValueError, match="mount_body"):
        _policy(_pair(30, 26), fm)


def test_a_tracker_camera_the_generator_lacks_is_refused():
    """A subset in the WRONG direction: nothing in the pair can supply that camera."""
    fm = _pair(60, 52, prefix="fm_")[:1]
    with pytest.raises(ValueError, match="does not have"):
        _policy(_pair(30, 26), fm)


##
# The tracker's cameras are a SUBSET of the generator's (the shipped configuration)
#
# The policy is conditioned on the NEAR camera alone; the generator reads both. So the two
# contracts have different lengths, the runtime casts their UNION, and each consumer selects
# its own rows BY NAME. An index pairing here is shape-clean and would simply feed one
# consumer the other's image, which is why these assert on VALUES.
##

def _rows(policy, spec, rows, images):
    """Run the selector for one consumer over a supplied reading."""
    return policy._select_depth_rows(images, spec, rows, "test", "depth")


def test_a_near_only_tracker_casts_the_union_and_needs_no_second_cast():
    """The shipped pair: the tracker reads the near camera, the generator reads both.

    Differing camera SETS is not a split SAMPLING. Both consumers want the same 30x26 grid,
    so the runtime casts their UNION once -- the generator's contract, which is the superset
    -- and each selects its own rows by name. Reporting a split here would make the runner
    demand a second ``state["fm_depth"]`` cast that nothing produces and nothing needs.
    """
    p = _policy(_pair(30, 26)[:1], _pair(30, 26))
    assert p.depth_camera_sets_differ() is True
    assert p.depth_sampling_is_split() is False, "same grid: one cast serves both"
    assert [camera_key(c) for c in p.get_depth_camera_spec()] == \
        ["depth_camera_near", "depth_camera_far"]
    assert [camera_key(c) for c in p.get_tracker_depth_camera_spec()] == ["depth_camera_near"]
    assert [camera_key(c) for c in p.get_fm_depth_camera_spec()] == \
        ["depth_camera_near", "depth_camera_far"]


def test_the_union_feed_routes_the_right_image_to_each_consumer():
    """VALUES, not shapes: an index pairing gives the tracker the far camera's image."""
    p = _policy(_pair(30, 26)[:1], _pair(30, 26))
    union = np.stack([np.full(780, 1.0), np.full(780, 2.0)])     # near = 1, far = 2
    trk = _rows(p, p.get_tracker_depth_camera_spec(), p._trk_depth_rows, union)
    assert trk.shape == (1, 780) and np.all(trk[0] == 1.0)
    fm = p._fm_depth_reading({"depth": union})
    assert fm.shape == (2, 780) and np.all(fm[0] == 1.0) and np.all(fm[1] == 2.0)


def test_the_tracker_selects_by_name_and_not_by_position():
    """A FAR-only tracker: by index it would read the union's row 0, the near camera."""
    p = _policy(_pair(30, 26)[1:], _pair(30, 26))
    union = np.stack([np.full(780, 1.0), np.full(780, 2.0)])
    trk = _rows(p, p.get_tracker_depth_camera_spec(), p._trk_depth_rows, union)
    assert np.all(trk[0] == 2.0), "row 1 (far) is this tracker's only camera"


def test_a_reading_that_is_neither_the_union_nor_the_consumers_own_is_refused():
    """Three rows are not two cameras and not one; say what both counts are."""
    p = _policy(_pair(30, 26)[:1], _pair(30, 26))
    with pytest.raises(ValueError, match="cast union"):
        _rows(p, p.get_tracker_depth_camera_spec(), p._trk_depth_rows, np.zeros((3, 780)))


def test_a_consumers_own_reading_is_still_accepted_unchanged():
    """A caller that supplies exactly the tracker's cameras (hardware) is untouched."""
    p = _policy(_pair(30, 26)[:1], _pair(30, 26))
    own = np.full((1, 780), 7.0)
    assert np.all(_rows(p, p.get_tracker_depth_camera_spec(), p._trk_depth_rows, own) == 7.0)


def test_a_moved_mount_on_the_shared_camera_is_still_refused_across_a_subset():
    """The relaxation is about which cameras, never about what they are."""
    fm = _pair(30, 26)
    fm[0]["mount_body"] = "pelvis_link"
    with pytest.raises(ValueError, match="depth_camera_near.mount_body"):
        _policy(_pair(30, 26)[:1], fm)


def test_a_split_sampling_on_top_of_a_subset_still_needs_its_own_reading():
    """Both relaxations at once: differing grids AND differing camera sets."""
    p = _policy(_pair(30, 26)[:1], _pair(60, 52, prefix="fm_"))
    assert p.depth_sampling_is_split() is True and p.depth_camera_sets_differ() is True
    # No castable union: one array cannot hold 780 and 3120 pixel rows, so the cast is the
    # tracker's own cameras and the generator's row map is withheld.
    assert [(c["width"], c["height"]) for c in p.get_depth_camera_spec()] == [(30, 26)]
    assert p._fm_depth_rows is None
    with pytest.raises(KeyError, match="state\\['fm_depth'\\]"):
        p._fm_depth_reading({"depth": np.zeros((1, 780))})
    got = p._fm_depth_reading({"depth": np.zeros((1, 780)),
                               "fm_depth": np.zeros((2, 60 * 52))})
    assert got.shape == (2, 3120)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
