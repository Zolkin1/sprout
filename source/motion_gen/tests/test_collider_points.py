# Copyright 2026 Zachary Olkin. All rights reserved.

"""The robot's collider geometry is READ from the robot files, and the two agree.

``collider_points`` exists so the terrain reward's notion of how big a foot is comes from the
MJCF rather than from a constant that can fall behind it.  These tests close that loop from
both ends: the MJCF-derived geometry must equal the URDF's own collision elements (the two are
generated from one source and a divergence is a real defect), and it must equal the two places
that already restate the foot box by hand.

The last of those is the point of the module.  ``clip_output._FOOT_BOX_*`` and
``terrain.footprint.FOOT_BOX_*`` are two copies of the same six numbers, with a comment saying
so; this asserts the robot file agrees with both, so a foot change moves all three or fails
here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from motion_gen import features as F
from motion_gen.clip_output import _FOOT_BOX_CENTER, _FOOT_BOX_HALF, foot_box_corners
from motion_gen.collision import collider_points, flatten_colliders
from motion_gen.plan_geometry import FOOT_BODIES
from motion_gen.terrain.footprint import FOOT_BOX_CENTER, FOOT_BOX_HALF, foot_corners

# Derived from the package root, not by counting ``..`` hops from this file -- moving a
# module between packages has silently broken such a path here before.
_URDF = F._REPO_ROOT / "robot_assets" / "g1_perceptive" / "g1_perceptive.urdf"


def test_the_collider_set_is_what_the_robot_declares():
    """Every body with a collider, and nothing invented for one without.

    Named explicitly because the reward's coverage IS this list: ``pelvis_link`` and the
    shoulder links carry no collision geometry in either robot file, so "the pelvis" and "the
    shoulders" are covered by the one ``torso_link`` box and by nothing else.
    """
    got = collider_points(F.DEFAULT_MJCF)
    assert set(got) == {
        "left_hip_roll_link", "right_hip_roll_link",        # thighs
        "left_knee_link", "right_knee_link",                # shanks
        "left_ankle_roll_link", "right_ankle_roll_link",    # feet
        "torso_link", "backpack_shell_link",
        "left_elbow_link", "right_elbow_link",              # forearms
        "left_wrist_yaw_link", "right_wrist_yaw_link",      # hands
    }
    assert "pelvis_link" not in got
    assert not any("shoulder" in b for b in got)


def test_mjcf_collision_geometry_equals_the_urdf_s():
    """The MJCF is generated from the URDF; a collider divergence is a real defect."""
    root = ET.parse(_URDF).getroot()
    urdf: dict[str, tuple[str, tuple[float, ...], tuple[float, ...]]] = {}
    for link in root.findall("link"):
        for col in link.findall("collision"):
            geom = col.find("geometry")
            kid = list(geom)[0]
            origin = col.find("origin")
            xyz = tuple(float(v) for v in (origin.get("xyz", "0 0 0").split()))
            if kid.tag == "box":
                size = tuple(float(v) / 2.0 for v in kid.get("size").split())
            elif kid.tag == "cylinder":
                size = (float(kid.get("radius")), float(kid.get("length")) / 2.0)
            elif kid.tag == "sphere":
                size = (float(kid.get("radius")),)
            else:
                pytest.fail(f"unhandled URDF collision shape {kid.tag}")
            urdf[link.get("name")] = (kid.tag, size, xyz)

    mjcf = collider_points(F.DEFAULT_MJCF)
    assert set(mjcf) == set(urdf), (set(mjcf) ^ set(urdf))
    for body, (tag, size, xyz) in urdf.items():
        got = mjcf[body]
        if tag == "box":
            assert got.kinds == ("box",) * 8
            centre = got.points.mean(axis=0)
            half = (got.points.max(axis=0) - got.points.min(axis=0)) / 2.0
            assert np.allclose(centre, xyz, atol=1e-9), body
            assert np.allclose(np.sort(half), np.sort(size), atol=1e-9), body
            assert np.all(got.radii == 0.0)
        elif tag == "sphere":
            assert got.points.shape == (1, 3) and np.allclose(got.points[0], xyz, atol=1e-9)
            assert got.radii[0] == pytest.approx(size[0])
        else:                                    # cylinder -> two end-cap centres
            assert got.points.shape == (2, 3)
            assert np.allclose(got.points.mean(axis=0), xyz, atol=1e-9), body
            span = np.linalg.norm(got.points[1] - got.points[0]) / 2.0
            assert span == pytest.approx(size[1], abs=1e-9), body
            assert np.allclose(got.radii, size[0]), body


@pytest.mark.parametrize("foot", FOOT_BODIES)
def test_the_derived_foot_box_equals_both_hand_kept_copies(foot):
    """The robot file, ``clip_output`` and ``terrain.footprint`` must state one foot box."""
    assert _FOOT_BOX_CENTER == FOOT_BOX_CENTER and _FOOT_BOX_HALF == FOOT_BOX_HALF

    pts = collider_points(F.DEFAULT_MJCF, [foot])[foot].points
    eight = foot_corners(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(np.sort(pts, axis=0), np.sort(eight, axis=0), atol=1e-12)

    bottom = pts[pts[:, 2] < _FOOT_BOX_CENTER[2]]
    assert bottom.shape == (4, 3)
    assert np.allclose(np.sort(bottom, axis=0),
                       np.sort(np.asarray(foot_box_corners()), axis=0), atol=1e-7)


def test_flatten_maps_every_point_back_to_its_body():
    """``body_index`` is what makes a per-body reduction a scatter instead of a loop."""
    colliders = collider_points(F.DEFAULT_MJCF)
    order, pts, radii, areas, idx = flatten_colliders(colliders)
    assert pts.shape[0] == radii.shape[0] == areas.shape[0] == idx.shape[0]
    assert pts.shape[0] == sum(v.points.shape[0] for v in colliders.values())
    for i, body in enumerate(order):
        own = pts[idx == i]
        assert np.allclose(own, colliders[body].points)


def test_an_unknown_or_collider_free_body_is_refused():
    """A body with no points could never be penalised, so asking for one is an error."""
    with pytest.raises(ValueError, match="no collision geometry"):
        collider_points(F.DEFAULT_MJCF, ["pelvis_link"])
    with pytest.raises(ValueError, match="no collision geometry"):
        collider_points(F.DEFAULT_MJCF, ["not_a_link"])


def test_surface_sampling_recovers_the_analytic_area_of_every_primitive():
    """The area weights are what make a SUM an integral, so they have to be right.

    Checked against the closed form per shape rather than against each other. The cylinder's
    end caps are sampled as concentric rings at mid-radius, which is a quadrature and lands
    within a fraction of a percent; the box and the sphere are exact.
    """
    got = collider_points(F.DEFAULT_MJCF, spacing=0.02)

    foot = got["left_ankle_roll_link"]
    h = np.array(_FOOT_BOX_HALF)
    exact = 2.0 * (4 * h[0] * h[1] + 4 * h[0] * h[2] + 4 * h[1] * h[2])
    assert foot.areas.sum() == pytest.approx(exact, rel=1e-9)

    wrist = got["left_wrist_yaw_link"]
    assert wrist.areas.sum() == pytest.approx(4.0 * np.pi * 0.055 ** 2, rel=1e-9)

    knee = got["left_knee_link"]
    r, half = 0.05, 0.15
    assert knee.areas.sum() == pytest.approx(
        2 * np.pi * r * 2 * half + 2 * np.pi * r * r, rel=0.01)


def test_surface_mode_carries_no_radius_and_extreme_mode_carries_no_area():
    """The two modes are exclusive: a point is either ON the surface or a swept sphere."""
    surf = collider_points(F.DEFAULT_MJCF, spacing=0.03)
    assert all((v.radii == 0.0).all() for v in surf.values())
    assert all((v.areas > 0.0).all() for v in surf.values())

    ext = collider_points(F.DEFAULT_MJCF)
    assert all((v.areas == 0.0).all() for v in ext.values())
    assert any((v.radii > 0.0).any() for v in ext.values())


def test_the_sampling_is_deterministic_and_denser_spacing_converges():
    """No RNG anywhere: a lattice, so the reward is a fixed function of the plan."""
    a = collider_points(F.DEFAULT_MJCF, spacing=0.03)
    b = collider_points(F.DEFAULT_MJCF, spacing=0.03)
    for k in a:
        assert np.array_equal(a[k].points, b[k].points)
    coarse = sum(v.areas.sum() for v in a.values())
    fine = sum(v.areas.sum() for v in collider_points(F.DEFAULT_MJCF, spacing=0.01).values())
    assert abs(coarse - fine) / fine < 0.01, "area should be converged well before 3 cm"
