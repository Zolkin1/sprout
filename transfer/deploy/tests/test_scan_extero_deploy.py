# Copyright 2026 Zachary Olkin. All rights reserved.

"""A generator conditioned on the HEIGHT SCAN, off-sim.

Three things this pins, each of which silently refused a perfectly good artifact before:

* ``fm_meta.extero_spec`` is parsed by the contract's own ``extero_source``.  It used to go
  through the depth parser unconditionally, so a scan graph died at load naming missing
  camera fields (``fx``, ``cx``, ...) that a ray grid has no reason to carry.
* a DEPTH tracker paired with a SCAN generator is a legitimate pair -- the policy has
  cameras, the generator reads the ground -- and must not be reconciled as though both
  described the one camera pair.
* the PRIVILEGED dense scan is refused by default and runnable only on an explicit opt-in,
  which is what keeps an Isaac-only diagnostic out of a deployment while letting MuJoCo,
  which can cast the same grid, run it.

No ONNX and no simulator: the contracts and the context are built by hand and the real
resolution method is run over them.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from deploy.depth_obs import parse_depth_spec  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.scan_obs import (  # noqa: E402
    assert_scan_specs_agree,
    is_privileged,
    parse_extero_spec,
    parse_scan_spec,
)

_FOCAL = 2.2
_APERTURE = 5.24


def _cam(width: int, height: int, sensor: str, obs_term: str) -> dict:
    """One depth-camera contract entry, as ``mdp.depth_camera_specs`` emits it."""
    return {
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


def _pair() -> list[dict]:
    """The tracker's near/far depth cameras."""
    return parse_depth_spec([
        _cam(30, 26, "depth_camera_near", "depth_near"),
        _cam(30, 26, "depth_camera_far", "depth_far"),
    ])


def _scan(sensor: str = "fm_height_scanner", resolution: float = 0.025,
          size=(3.0, 3.0), offset_pos=(0.0, 0.0, 20.0), privileged: bool = True,
          **over) -> dict:
    """One height-scanner contract entry, as ``mdp.height_scan_specs`` emits it."""
    height = int(round(size[1] / resolution)) + 1
    width = int(round(size[0] / resolution)) + 1
    scan = {
        "sensor": sensor, "mount_body": "pelvis_link",
        "offset_pos": list(offset_pos), "ray_alignment": "yaw",
        "direction": [0.0, 0.0, -1.0], "max_distance": 1e6,
        "resolution": resolution, "size": list(size), "ordering": "xy",
        "num_x": width, "num_y": height, "num_rays": height * width,
        "height": height, "width": width, "sensor_offset": 0.5,
        "drift_range": [0.0, 0.0], "ray_cast_drift_range": {},
        "update_period": 0.0, "privileged": privileged,
    }
    scan.update(over)
    return scan


def _tracker_scan(resolution: float = 0.025, size=(3.0, 3.0),
                  offset_pos=(0.0, 0.0, 20.0)) -> dict:
    """The SMALLER contract ``export_parameters`` embeds in ``tracker.onnx``."""
    return {
        "resolution": resolution, "size": list(size), "offset_pos": list(offset_pos),
        "ray_alignment": "yaw", "sensor_offset": 0.5, "clip": [-1.0, 1.0],
    }


def _policy(tracker_spec: list[dict], scan_spec: list[dict], tracker_scan: dict | None = None,
            allow_privileged: bool = False) -> GenTrackerPolicy:
    """A ``GenTrackerPolicy`` with its contracts resolved, without loading any ONNX."""
    policy = object.__new__(GenTrackerPolicy)
    policy.ctx = SimpleNamespace(
        extero_source="height_scan", extero_sensor_count=1,
        extero_sensor_dim=(int(scan_spec[0]["num_rays"]) if scan_spec else 1),
        extero_spec=scan_spec, extero_encoder="cnn",
        extero_image_hw=([(int(scan_spec[0]["height"]), int(scan_spec[0]["width"]))]
                         if scan_spec else []),
        extero_tokens_per_sensor=256,
    )
    policy.allow_privileged_extero = allow_privileged
    policy._trk = SimpleNamespace(
        get_depth_camera_spec=lambda: [dict(c) for c in tracker_spec],
        get_height_scan_spec=lambda: (dict(tracker_scan) if tracker_scan else None),
    )
    GenTrackerPolicy._resolve_depth_cameras(policy)
    return policy


##
# The parse is chosen by the contract's own source
##

def test_scan_spec_is_not_parsed_as_a_camera():
    """The regression: the depth parser refuses a scan contract for missing ``fx``."""
    with pytest.raises(ValueError, match=r"missing field"):
        parse_depth_spec([_scan()], "fm.onnx")
    assert parse_extero_spec([_scan()], "height_scan", "fm.onnx")[0]["num_rays"] == 121 * 121


def test_depth_spec_still_takes_the_depth_parser():
    """The dispatch must not have quietly loosened the camera path."""
    parsed = parse_extero_spec([_cam(30, 26, "depth_camera_near", "depth_near")],
                               "depth", "fm.onnx")
    assert parsed[0]["fx"] == pytest.approx(30 * _FOCAL / _APERTURE)
    with pytest.raises(ValueError, match=r"missing field"):
        parse_extero_spec([{"sensor": "depth_camera_near"}], "depth", "fm.onnx")


def test_unknown_source_has_no_parser():
    """Better than defaulting to one of them and validating the wrong contract."""
    with pytest.raises(ValueError, match=r"neither 'depth' nor 'height_scan'"):
        parse_extero_spec([_scan()], "lidar", "fm.onnx")


##
# What the scan parser itself refuses -- each silent-when-wrong, not merely absent
##

def test_a_sideways_cast_is_refused():
    """'pos_z - hit_z - offset' is only ground clearance for a straight-down ray."""
    with pytest.raises(ValueError, match=r"not straight down"):
        parse_scan_spec([_scan(direction=[1.0, 0.0, 0.0])], "fm.onnx")


def test_a_base_aligned_grid_is_refused():
    """A roll/pitch-coupled grid moves its rays under the robot, unlike the window frame."""
    with pytest.raises(ValueError, match=r"ray_alignment"):
        parse_scan_spec([_scan(ray_alignment="base")], "fm.onnx")


def test_a_fold_that_disagrees_with_the_ray_count_is_refused():
    """The convolutional encoder reads an image; a wrong fold scrambles it."""
    with pytest.raises(ValueError, match=r"folds as"):
        parse_scan_spec([_scan(num_rays=999)], "fm.onnx")


##
# A depth tracker + a scan generator is a legitimate pair
##

def test_depth_tracker_with_scan_generator_resolves():
    """The policy has cameras and the generator reads the ground; nothing to reconcile.

    This raised "tracker.onnx has 2 depth camera(s) but fm.onnx has 1" before -- the FM's
    scan contract was being counted as a camera list.
    """
    policy = _policy(_pair(), [_scan()], allow_privileged=True)
    assert [c["obs_term"] for c in policy._depth_spec] == ["depth_near", "depth_far"]
    # The generator reads no camera at all, so it must not be handed its own SPLIT sampling.
    assert policy.depth_sampling_is_split() is False


##
# The privileged opt-in
##

def test_privileged_scan_is_refused_by_default():
    """Hardware constructs the policy the plain way, so hardware still cannot load this."""
    with pytest.raises(ValueError, match=r"PRIVILEGED"):
        _policy(_pair(), [_scan(privileged=True)])


def test_privileged_scan_runs_on_the_explicit_opt_in():
    """MuJoCo casts the same grid against the same terrain, so it may honour the contract."""
    policy = _policy(_pair(), [_scan(privileged=True)], allow_privileged=True)
    assert is_privileged(policy.ctx.extero_spec)


def test_a_deployable_scan_needs_no_opt_in():
    """The gate is on ``privileged``, not on "the generator reads a scan"."""
    policy = _policy([], [_scan(sensor="height_scanner", resolution=0.1, size=(1.5, 1.0),
                                offset_pos=(0.75, 0.0, 20.0), privileged=False)])
    assert policy.ctx.extero_spec[0]["num_rays"] == 16 * 11


##
# Tracker/generator grid agreement, across the two contract SHAPES
##

def test_matching_grids_agree_across_contract_shapes():
    """The FM embeds the full sensor description, the tracker six fields; both are valid.

    The old inline check compared the FM's ``ordering`` against a tracker key that does not
    exist, so it refused every scan tracker + scan generator pair.
    """
    assert_scan_specs_agree(parse_scan_spec([_scan(privileged=False)], "fm.onnx"),
                            [_tracker_scan()], "fm.onnx", "tracker.onnx")


def test_a_differing_grid_is_still_refused():
    """The runner casts ONE grid per step, so the two cannot each want their own."""
    with pytest.raises(ValueError, match=r"DIFFERENT height-scan grids"):
        assert_scan_specs_agree(parse_scan_spec([_scan(privileged=False)], "fm.onnx"),
                                [_tracker_scan(resolution=0.1, size=(1.5, 1.0))],
                                "fm.onnx", "tracker.onnx")


def test_a_contract_too_sparse_to_compare_is_refused():
    """Intersecting keys without a floor would let an empty contract agree with anything."""
    with pytest.raises(ValueError, match=r"too little of the scan contract"):
        assert_scan_specs_agree(parse_scan_spec([_scan(privileged=False)], "fm.onnx"),
                                [{"resolution": 0.025}], "fm.onnx", "tracker.onnx")
