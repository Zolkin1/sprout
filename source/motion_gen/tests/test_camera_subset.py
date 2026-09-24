# Copyright 2026 Zachary Olkin. All rights reserved.
"""The near-only ABLATION seam: dropping a camera at dataset build.

The generator reads both cameras. Asking what it would do on the near one alone is a question
about the MODEL, so the honest comparison holds the data fixed and drops a camera at training
time -- re-recording near-only would vary the rollouts as well as the input, and would trip
``assert_fm_depth_group_covers_every_camera`` on the recording side, where a missing camera
group is a FAULT rather than an opt-out.

**The property this file exists for is INERTNESS.** A seam in the shared training path is only
acceptable if the default is provably untouched, so the first test asserts the unsliced call
returns its inputs as the SAME OBJECTS -- not merely equal ones, which a copy would also
satisfy while changing memory behaviour on a 14 GB recording.

The rest is about selecting the RIGHT camera. Selection is by NAME because index pairing is
what once compared the policy's near camera against the generator's far one, and the recordings
spell the name three different ways.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

fm = pytest.importorskip("robot_rl.flow_matching")
from robot_rl.flow_matching.dataset import camera_name  # noqa: E402

apply_camera_subset = fm.apply_camera_subset


@dataclasses.dataclass
class _Meta:
    """The one metadata field the subset reads."""

    depth_camera_spec: list

    
def _spec(names=("near", "far")):
    """A depth contract spelled the way a real recording spells it."""
    return [{"sensor": f"depth_camera_{n}", "obs_term": f"fm_depth_{n}",
             "width": 30, "height": 26, "mount_body": "torso_link"} for n in names]


def _data(n_win=7, n_cam=2, k=780, n_lag=4):
    """Arrays with the shapes a real depth recording has."""
    rng = np.random.default_rng(0)
    return {
        "depth": rng.normal(size=(n_win, n_cam, k)).astype(np.float32),
        "depth_lag": rng.normal(size=(n_win, n_lag, n_cam, k)).astype(np.float32),
        # No sensor axis: per-LAG robot state. Must survive untouched.
        "depth_lag_state": rng.normal(size=(n_win, n_lag, 123)).astype(np.float32),
        "ref_future": rng.normal(size=(n_win, 62, 123)).astype(np.float32),
    }


# ---------------------------------------------------------------------------------------------
# INERTNESS -- the licence for putting this in the shared path at all
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("keep", [None, [], ""])
def test_no_selection_returns_the_INPUTS_not_copies(keep) -> None:
    """Identity, not equality. A copy would satisfy equality while doubling peak memory on a
    14 GB recording, and the default path must not pay anything for a flag it did not use."""
    data, meta = _data(), _Meta(_spec())
    out_d, out_m = apply_camera_subset(data, meta, keep)
    assert out_d is data and out_m is meta


def test_selecting_EVERY_camera_is_also_inert() -> None:
    """Naming both cameras is the same request as naming none, and must cost the same."""
    data, meta = _data(), _Meta(_spec())
    out_d, out_m = apply_camera_subset(data, meta, ["near", "far"])
    assert out_d is data and out_m is meta


# ---------------------------------------------------------------------------------------------
# Selecting the right camera
# ---------------------------------------------------------------------------------------------
def test_the_kept_camera_keeps_its_OWN_rows() -> None:
    """The whole point: near-only must be the NEAR camera's pixels, not the far one's.

    Checked against the source array rather than by shape -- a slice on the wrong index has
    exactly the right shape, which is the shape of this entire class of bug.
    """
    data, meta = _data(), _Meta(_spec())
    out, out_meta = apply_camera_subset(data, meta, ["near"])
    assert out["depth"].shape == (7, 1, 780)
    assert np.array_equal(out["depth"][:, 0], data["depth"][:, 0])
    assert not np.array_equal(out["depth"][:, 0], data["depth"][:, 1])
    assert camera_name(out_meta.depth_camera_spec[0]) == "near"


def test_selecting_the_FAR_camera_takes_the_far_rows() -> None:
    """The other direction, so 'it kept index 0' cannot pass for 'it kept the one asked for'."""
    data, meta = _data(), _Meta(_spec())
    out, out_meta = apply_camera_subset(data, meta, ["far"])
    assert np.array_equal(out["depth"][:, 0], data["depth"][:, 1])
    assert camera_name(out_meta.depth_camera_spec[0]) == "far"


def test_the_LAG_buffer_is_sliced_on_its_own_axis() -> None:
    """`depth_lag` is (W, L, T, K) -- the sensor axis is 2, not 1. Slicing axis 1 would take a
    LAG instead of a camera, at a shape that looks right."""
    data, meta = _data(), _Meta(_spec())
    out, _ = apply_camera_subset(data, meta, ["far"])
    assert out["depth_lag"].shape == (7, 4, 1, 780)
    assert np.array_equal(out["depth_lag"][:, :, 0], data["depth_lag"][:, :, 1])


def test_arrays_without_a_sensor_axis_are_untouched() -> None:
    """`depth_lag_state` is per-LAG robot state; slicing it would corrupt the capture poses."""
    data, meta = _data(), _Meta(_spec())
    out, _ = apply_camera_subset(data, meta, ["near"])
    assert out["depth_lag_state"] is data["depth_lag_state"]
    assert out["ref_future"] is data["ref_future"]


def test_the_contract_is_filtered_to_match_the_arrays() -> None:
    """A spec still listing two cameras against a one-camera array is exactly the disagreement
    `_resolve_extero_spec` raises on -- which is the good case; the bad one is it not noticing."""
    data, meta = _data(), _Meta(_spec())
    out, out_meta = apply_camera_subset(data, meta, ["near"])
    assert len(out_meta.depth_camera_spec) == 1 == out["depth"].shape[1]
    assert meta.depth_camera_spec == _spec(), "the caller's metadata was mutated"


# ---------------------------------------------------------------------------------------------
# Refusals -- each alternative is a silently different experiment
# ---------------------------------------------------------------------------------------------
def test_a_name_the_recording_does_not_have_is_refused() -> None:
    """A typo would otherwise leave every camera in and quietly run the two-camera arm under
    the ablation's name."""
    with pytest.raises(ValueError, match="does not have"):
        apply_camera_subset(_data(), _Meta(_spec()), ["nearr"])


def test_a_recording_with_no_contract_is_refused() -> None:
    """Without it there is no way to say which camera is which."""
    with pytest.raises(ValueError, match="carries none"):
        apply_camera_subset(_data(), _Meta([]), ["near"])


def test_a_camera_that_names_itself_nowhere_is_refused() -> None:
    """Falling back to an INDEX is how the policy's near camera got paired with the
    generator's far one."""
    spec = [{"width": 30, "height": 26}, {"width": 30, "height": 26}]
    with pytest.raises(ValueError, match="names itself in none"):
        apply_camera_subset(_data(), _Meta(spec), ["near"])


def test_a_non_depth_source_is_refused() -> None:
    """A height scan has no cameras to choose between."""
    with pytest.raises(ValueError, match="no cameras to choose"):
        apply_camera_subset(_data(), _Meta(_spec()), ["near"], source="height_scan")


def test_an_array_whose_axis_is_not_the_camera_count_is_refused() -> None:
    """Rather than slicing something else to the right shape."""
    data, meta = _data(), _Meta(_spec())
    data["depth"] = data["depth"][:, :1]          # one camera against a two-camera contract
    with pytest.raises(ValueError, match="refusing to slice"):
        apply_camera_subset(data, meta, ["near"])


@pytest.mark.parametrize("entry,want", [
    ({"sensor": "depth_camera_near"}, "near"),
    ({"obs_term": "fm_depth_far"}, "far"),
    ({"camera_key": "near"}, "near"),
    ({"camera_key": "depth_far"}, "far"),
    ({"width": 30}, None),
])
def test_a_camera_names_itself_in_any_of_three_fields(entry, want) -> None:
    """Recordings have spelled it all three ways; the ablation must not care which."""
    assert camera_name(entry) == want
