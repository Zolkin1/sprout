# Copyright 2026 Zachary Olkin. All rights reserved.

"""``depth_cloud_viz.unproject`` against the ray builder that produced the images.

The cloud in RViz is a DIAGNOSTIC, which makes a silent error in it worse than no cloud at
all: a mis-unprojected image looks like a plausible scene, and the first thing it would be
used for is deciding whether a camera can see an obstacle. So this is held to the sim's own
``pinhole_ray_directions`` -- not to a second derivation of the pinhole model, but to the
exact function that CAST the rays whose returns these images are.

Two conventions are easy to get wrong and are both pinned here:

* the camera frame is ``x = optical axis, y = left, z = up``, NOT the ROS optical frame
  (z forward, x right, y down). Getting this wrong yields a cloud that is a rotation of
  the truth -- still plausible, still wrong.
* the half-pixel. Rays leave pixel CENTRES, so the offset is ``u + 0.5 - cx``. Dropping it
  shifts everything by half a pixel of angle, which reads as a calibration error rather
  than as a bug.

Imported by path, because the node module pulls in rclpy at import time and the ROS
packages are not installed in the off-sim test env. Only the pure function is wanted.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest

OBELISK = pathlib.Path(__file__).resolve().parent.parent
TRANSFER = OBELISK.parent
NODE_PY = OBELISK / "g1_control" / "g1_control" / "depth_cloud_viz_node.py"

#: The near camera's real contract, from tracker.onnx. Its fx/fy differ (12.6 vs 18.6), so a
#: test that swapped them would fail -- which a square-ish placeholder would not catch.
NEAR = {
    "width": 30, "height": 26,
    "fx": 12.586494467659199, "fy": 18.56592408764749, "cx": 15.0, "cy": 13.0,
    "min_dist": 0.1, "max_dist": 7.9,
}


def _unproject():
    """Load ``unproject`` from the node module without importing rclpy.

    Returns:
        The ``unproject`` function.
    """
    src = NODE_PY.read_text()
    body = "def unproject" + src.split("def unproject")[1].split("\nclass ")[0]
    mod = types.ModuleType("_depth_cloud_viz_fn")
    mod.np = np
    exec(compile("import numpy as np\n" + body, str(NODE_PY), "exec"), mod.__dict__)
    return mod.unproject


unproject = _unproject()


def test_constant_depth_image_is_a_plane_at_that_depth() -> None:
    """A uniform z-depth image must put every point at exactly that x."""
    pts = unproject(np.full(NEAR["width"] * NEAR["height"], 2.0), NEAR)
    assert len(pts) == NEAR["width"] * NEAR["height"]
    # Exactly, not approximately: the ray's x-component is 1 by construction, so the
    # unprojection is a multiplication and introduces no error of its own.
    np.testing.assert_array_equal(pts[:, 0], 2.0)


def test_matches_the_ray_builder_that_generated_the_image() -> None:
    """Parity with transfer/sim's pinhole_ray_directions, to machine precision."""
    pytest.importorskip("mujoco", reason="transfer/sim.depth_camera imports mujoco")
    sys.path.insert(0, str(TRANSFER / "sim"))
    spec = importlib.util.spec_from_file_location(
        "_depth_camera", TRANSFER / "sim" / "depth_camera.py")
    dc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dc)

    d = 2.0
    pts = unproject(np.full(NEAR["width"] * NEAR["height"], d), NEAR)
    dirs = dc.pinhole_ray_directions(NEAR["width"], NEAR["height"],
                                     NEAR["fx"], NEAR["fy"], NEAR["cx"], NEAR["cy"])
    # The builder returns UNIT directions; scale each so its x-component is the z-depth.
    oracle = dirs * (d / dirs[:, 0:1])
    np.testing.assert_allclose(pts, oracle, atol=1e-12)


def test_pixel_centres_carry_the_half_pixel() -> None:
    """The principal ray lands half a pixel off centre, because cx is on a pixel EDGE.

    With cx = 15.0 and 30 columns, pixel centres sit at 0.5 .. 29.5, so the nearest is
    0.5 px away and the point is offset by d * 0.5 / fx. Asserting the offset rather than
    zero is what pins the convention: drop the +0.5 and this becomes exactly zero.
    """
    d = 2.0
    pts = unproject(np.full(NEAR["width"] * NEAR["height"], d), NEAR)
    u = np.tile(np.arange(NEAR["width"]), NEAR["height"])
    v = np.repeat(np.arange(NEAR["height"]), NEAR["width"])
    k = int(np.argmin((u + 0.5 - NEAR["cx"]) ** 2 + (v + 0.5 - NEAR["cy"]) ** 2))
    assert pts[k][1] == pytest.approx(d * 0.5 / NEAR["fx"], rel=1e-9)
    assert pts[k][2] == pytest.approx(d * 0.5 / NEAR["fy"], rel=1e-9)


def test_unreadable_pixels_are_dropped_not_clamped() -> None:
    """NaN and out-of-range pixels leave holes.

    Clamping a saturated pixel to max range draws a wall that is not there, and on this
    display that is the difference between "the camera cannot see the box" and "there is a
    box". A hole is honest.
    """
    img = np.full(NEAR["width"] * NEAR["height"], 2.0)
    img[0] = np.nan
    img[1] = np.inf
    img[2] = 99.0                      # beyond max_dist
    img[3] = 0.01                      # inside min_dist
    assert len(unproject(img, NEAR)) == NEAR["width"] * NEAR["height"] - 4


def test_row_major_order_matches_the_image() -> None:
    """One lit pixel must unproject to the direction of THAT pixel, not its transpose.

    The images are 30 wide x 26 tall and both dimensions are free to be swapped silently;
    a square test image could not tell the difference. Lighting a single off-diagonal pixel
    can.
    """
    w, h = NEAR["width"], NEAR["height"]
    img = np.full(w * h, np.nan)
    u0, v0, d = 4, 20, 3.0
    img[v0 * w + u0] = d               # row-major: k = v * width + u
    pts = unproject(img, NEAR)
    assert len(pts) == 1
    np.testing.assert_allclose(
        pts[0],
        [d, -d * (u0 + 0.5 - NEAR["cx"]) / NEAR["fx"],
         -d * (v0 + 0.5 - NEAR["cy"]) / NEAR["fy"]],
        atol=1e-12)
