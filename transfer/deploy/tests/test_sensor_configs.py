# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generated Obelisk sensor configs reproduce the trained camera contract.

``scripts/rsl_rl/export_sensor_configs.py`` is the third consumer of the camera contract,
after the MuJoCo caster and the hardware controller.  The conversions it performs are
small and silent when wrong -- a quaternion written xyzw where Obelisk reads wxyz aims the
camera somewhere else, and an intrinsic matrix laid out column-major changes the field of
view -- so they are pinned here rather than discovered on the robot.

Lives with the deploy tests because the contract does; it needs no artifact.

Run: ``python transfer/deploy/tests/test_sensor_configs.py``  (or via pytest)
"""

from __future__ import annotations

import math
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_ROOT, "transfer"))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "rsl_rl"))

from deploy.depth_obs import parse_depth_spec  # noqa: E402
from export_sensor_configs import (  # noqa: E402
    DEPTH_GEOM_GROUPS,
    depth_scan_config,
    height_scan_config,
    zed_targets,
)

CAM = {
    "obs_term": "depth_near", "sensor": "depth_camera_near", "mount_body": "torso_link",
    "offset_pos": [0.10, 0.0, 0.05],
    # 75 deg pitch down about y: xyzw.
    "offset_quat_xyzw": [0.0, math.sin(math.radians(37.5)), 0.0, math.cos(math.radians(37.5))],
    "width": 30, "height": 26, "fx": 12.586494, "fy": 18.565924, "cx": 15.0, "cy": 13.0,
    "max_distance": 8.0, "update_period": 0.02, "clip": [-2.0, 2.0],
    "offset": 0.75, "scale": 0.5, "min_dist": 0.10, "max_dist": 7.9,
    "min_clip_val": 0.0, "max_clip_val": 0.0,
}

SCAN = {"resolution": 0.1, "size": [1.5, 1.0], "offset_pos": [0.75, 0.0, 20.0],
        "ray_alignment": "yaw", "sensor_offset": 0.5, "clip": [-1.0, 1.0]}


def _cam() -> dict:
    """The near camera, through the contract parser the runtime uses."""
    return parse_depth_spec([CAM])[0]


def test_intrinsic_matrix_is_row_major_k():
    """Obelisk reads a row-major K; a transposed one silently changes the FOV."""
    k = depth_scan_config(_cam())["pattern"]["intrinsic_matrix"]
    assert k == [CAM["fx"], 0.0, CAM["cx"], 0.0, CAM["fy"], CAM["cy"], 0.0, 0.0, 1.0]


def test_offset_quaternion_is_converted_to_wxyz():
    """The contract stores xyzw; Obelisk wants wxyz. Getting this wrong aims the camera."""
    x, y, z, w = CAM["offset_quat_xyzw"]
    assert depth_scan_config(_cam())["pattern"]["offset"]["rot"] == [w, x, y, z]


def test_offset_position_is_passed_through():
    """Position needs no reordering -- but it must not be dropped either."""
    assert depth_scan_config(_cam())["pattern"]["offset"]["pos"] == CAM["offset_pos"]


def test_resolution_is_carried_over():
    """The image the sim casts must be the shape the policy reads."""
    pattern = depth_scan_config(_cam())["pattern"]
    assert (pattern["width"], pattern["height"]) == (CAM["width"], CAM["height"])
    assert pattern["type"] == "depth_camera"


def test_geom_groups_match_the_mujoco_caster():
    """Self-occlusion is part of the observation: visual meshes + terrain, not collision."""
    assert depth_scan_config(_cam())["geom_groups"] == DEPTH_GEOM_GROUPS == [1, 2]


def test_zed_fov_round_trips_through_the_intrinsics():
    """The FOV handed to the camera must be the one the intrinsics describe."""
    t = zed_targets(_cam())
    # Rebuild fx from the emitted FOV and check it returns the contract's value.
    fx = (CAM["width"] / 2.0) / math.tan(math.radians(t["hfov_deg"]) / 2.0)
    assert fx == pytest.approx(CAM["fx"], rel=1e-9)
    fy = (CAM["height"] / 2.0) / math.tan(math.radians(t["vfov_deg"]) / 2.0)
    assert fy == pytest.approx(CAM["fy"], rel=1e-9)


def test_zed_range_and_rate_come_from_the_contract():
    """The camera must be configured to the range the policy was trained against."""
    t = zed_targets(_cam())
    assert (t["min_range_m"], t["max_range_m"]) == (CAM["min_dist"], CAM["max_dist"])
    assert t["rate_hz"] == pytest.approx(50.0)


def test_height_scan_geometry_is_carried_over():
    """The scan grid must match, or the 176 rays land somewhere else."""
    pattern = height_scan_config(SCAN)["pattern"]
    assert pattern["type"] == "height_scan"
    assert pattern["resolution"] == SCAN["resolution"]
    assert pattern["size"] == SCAN["size"]
    assert pattern["offset"]["pos"] == SCAN["offset_pos"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
