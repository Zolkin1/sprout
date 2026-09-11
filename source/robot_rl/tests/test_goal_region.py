# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the SPATIAL success test, because every other outcome signal here is temporal.

``mdp.clip_end_reached`` and every ``eval.schema.terrain_*`` column compare step INDICES.
None of them can tell a robot that walked up a staircase from one that stood still while the
clip cursor ran out.  ``mdp.goal_region`` is the only thing in the tracker task that answers
"did it physically get there", so the properties below are the ones the reward rests on.

These tests import no IsaacLab: the geometry half of the module is deliberately torch-only,
and the one function that needs a live command term (``goal_anchor_world``) imports
``isaaclab.utils.math`` lazily inside its body so this file runs on any box.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from robot_rl.tasks.manager_based.robot_rl.mdp.goal_region import (  # noqa: E402
    GoalRegionSpec,
    GoalTolerances,
    goal_distance_m,
    goal_frame_offsets,
    in_goal,
)

TOL = GoalTolerances(approach_margin_m=1.0, half_width_m=2.5, height_tol_m=0.5)


def _offsets(robot_xyz, goal_xyz, yaw):
    """Resolve one (robot, goal, heading) triple through the module under test."""
    return goal_frame_offsets(
        torch.tensor([robot_xyz], dtype=torch.float64),
        torch.tensor([goal_xyz], dtype=torch.float64),
        torch.tensor([yaw], dtype=torch.float64),
    )


def _inside(robot_xyz, goal_xyz=(0.0, 0.0, 0.0), yaw=0.0, tol=TOL):
    """Whether one robot position is inside the region."""
    return bool(in_goal(*_offsets(robot_xyz, goal_xyz, yaw), tol)[0])


# ----------------------------------------------------------------------------------------
# The half-plane: short is a miss, past is a hit.
# ----------------------------------------------------------------------------------------

def test_arriving_exactly_at_the_clip_endpoint_is_inside():
    assert _inside((0.0, 0.0, 0.0))


def test_stopping_short_beyond_the_margin_is_outside():
    """The stall case, and the whole reason this module exists."""
    assert _inside((-0.99, 0.0, 0.0))       # just inside the 1.0 m margin
    assert not _inside((-1.01, 0.0, 0.0))   # just outside it


def test_overshoot_is_never_a_miss():
    """A half-plane, not a box: falling off the far end is base_fallen's job, not this test's.

    Bounding downrange above would add a second way to fail that the reward does not intend,
    and on a finite tile the robot cannot overshoot far without already having fallen.
    """
    for extra in (0.1, 5.0, 500.0):
        assert _inside((extra, 0.0, 0.0))


# ----------------------------------------------------------------------------------------
# The lateral band: nothing in this stack steers.
# ----------------------------------------------------------------------------------------

def test_lateral_band_is_two_sided_and_symmetric():
    assert _inside((0.0, 2.49, 0.0))
    assert _inside((0.0, -2.49, 0.0))
    assert not _inside((0.0, 2.51, 0.0))
    assert not _inside((0.0, -2.51, 0.0))


# ----------------------------------------------------------------------------------------
# The height band -- the failure `freeform_goal_reached` leaves open.
# ----------------------------------------------------------------------------------------

def test_a_robot_on_the_floor_beside_an_elevated_tile_is_NOT_inside():
    """The specific case the height band is for, with the real numbers.

    A G1 standing on the ground plane has its base near 0.79 m.  ``base_fallen`` at a 0.40 m
    ABSOLUTE world height therefore does not fire there.  So a robot that walks off the side
    of a 0.85 m staircase and ends up standing on the floor, downrange of the flight, passes
    both the half-plane and the lateral band -- and without this third test would be scored a
    success at having climbed a staircase it fell off.
    """
    goal_on_top_of_the_flight = (6.0, 0.0, 0.85 + 0.79)
    robot_on_the_floor_below = (6.0, 0.0, 0.79)
    assert not _inside(robot_on_the_floor_below, goal_on_top_of_the_flight)
    # ... and the same robot, having actually climbed, is inside.
    assert _inside((6.0, 0.0, 0.85 + 0.79), goal_on_top_of_the_flight)


def test_height_band_is_signed_symmetric():
    assert _inside((0.0, 0.0, 0.49))
    assert _inside((0.0, 0.0, -0.49))
    assert not _inside((0.0, 0.0, 0.51))
    assert not _inside((0.0, 0.0, -0.51))


# ----------------------------------------------------------------------------------------
# The frame. A tile whose heading is not world-x is where a wrong rotation hides.
# ----------------------------------------------------------------------------------------

def test_the_region_turns_with_the_terrain_frame():
    """At yaw=90 deg the downrange axis is world +y, so world +x becomes LATERAL.

    Getting the sign of this rotation wrong is invisible on the world-axis-aligned tiles that
    make up most of the library and wrong on every other one, which is exactly the shape of
    defect that produces a plausible number.
    """
    yaw = math.pi / 2
    # 5 m along world +y is 5 m downrange -> past the goal, inside.
    assert _inside((0.0, 5.0, 0.0), yaw=yaw)
    # 5 m along world +x is 5 m LATERAL -> outside the 2.5 m band.
    assert not _inside((5.0, 0.0, 0.0), yaw=yaw)


def test_downrange_and_lateral_agree_with_the_termination_convention():
    """Same rotation `base_pose_error_termination` applies, but on `robot - goal`.

    That function forms ``ref - root``; this module forms ``root - goal`` so that a positive
    downrange reads as "past the goal".  The rotation itself must match, or the two disagree
    about which way a tile faces.
    """
    yaw = 0.7
    dx, dy = 3.0, -1.5
    down, lat, _ = _offsets((dx, dy, 0.0), (0.0, 0.0, 0.0), yaw)
    assert down.item() == pytest.approx(math.cos(yaw) * dx + math.sin(yaw) * dy)
    assert lat.item() == pytest.approx(-math.sin(yaw) * dx + math.cos(yaw) * dy)


# ----------------------------------------------------------------------------------------
# The reported distance.
# ----------------------------------------------------------------------------------------

def test_distance_is_zero_inside_and_measures_the_boundary_outside():
    assert goal_distance_m(*_offsets((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0), TOL)[0] == 0.0
    # 3 m short of the boundary (4 m short of the anchor, 1 m of which is free margin).
    d = goal_distance_m(*_offsets((-4.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0), TOL)[0]
    assert d.item() == pytest.approx(3.0)
    # Excess on two axes combines Euclidean-ly: 3 m short and 3 m wide of a 2.5 m band.
    d = goal_distance_m(*_offsets((-4.0, 3.5, 0.0), (0.0, 0.0, 0.0), 0.0), TOL)[0]
    assert d.item() == pytest.approx(math.hypot(3.0, 1.0))


def test_distance_separates_a_near_miss_from_never_leaving_the_spawn():
    """The reason the column exists at all -- both are timeouts, and they are not the same."""
    near = goal_distance_m(*_offsets((-1.5, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0), TOL)[0]
    spawn = goal_distance_m(*_offsets((-9.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0), TOL)[0]
    assert near.item() < spawn.item()


# ----------------------------------------------------------------------------------------
# The artifact.
# ----------------------------------------------------------------------------------------

def test_spec_round_trips_through_json(tmp_path):
    spec = GoalRegionSpec(
        default=GoalTolerances(1.25, 2.75, 0.45),
        by_family={"stairs_down": GoalTolerances(2.0, 3.0, 0.6)},
        provenance={"clip_library_sha256": "deadbeef", "episodes": 3300},
    )
    path = str(tmp_path / "goal_regions.json")
    spec.save(path)
    back = GoalRegionSpec.load(path)
    assert back.to_dict() == spec.to_dict()
    assert back.for_family("stairs_down").approach_margin_m == 2.0
    assert back.for_family("flat").approach_margin_m == 1.25       # falls back to default


def test_a_spec_without_measured_defaults_is_refused():
    """A missing `default` must raise, not silently hand back the placeholder dataclass.

    The dataclass defaults are placeholders; a spec that inherited them would look calibrated
    and be a guess, which is the failure mode the whole measure-it-from-the-clip-arm procedure
    exists to avoid.
    """
    with pytest.raises(ValueError, match="no 'default' block"):
        GoalRegionSpec.from_dict({"by_family": {}})


def test_batched_evaluation_is_elementwise():
    """Everything above is N=1; the real call site is N=1024."""
    robot = torch.tensor([[0.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 9.0, 0.0]], dtype=torch.float64)
    goal = torch.zeros(3, 3, dtype=torch.float64)
    yaw = torch.zeros(3, dtype=torch.float64)
    got = in_goal(*goal_frame_offsets(robot, goal, yaw), TOL)
    assert got.tolist() == [True, False, False]


# ----------------------------------------------------------------------------------------
# Per-env tolerances: the real call site spans several families in one batch.
# ----------------------------------------------------------------------------------------

def test_per_env_tolerances_apply_elementwise():
    """A batch of 1024 envs spans several families, each with its own calibrated region.

    A scalar spec broadcast over the batch would silently apply one family's tolerances to
    all of them -- which is exactly the shape of error that produces a plausible number, so
    the tensor path is pinned against the scalar one it must agree with.
    """
    down = torch.tensor([-1.5, -1.5], dtype=torch.float64)
    lat = torch.zeros(2, dtype=torch.float64)
    hz = torch.zeros(2, dtype=torch.float64)
    # env 0 gets a 1.0 m margin (misses at -1.5), env 1 gets 2.0 m (arrives).
    per_env = torch.tensor([[1.0, 2.5, 0.5], [2.0, 2.5, 0.5]], dtype=torch.float64)
    assert in_goal(down, lat, hz, per_env).tolist() == [False, True]


def test_scalar_and_tensor_tolerances_agree():
    down = torch.tensor([-1.5, 0.3, 4.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 3.0, -1.0], dtype=torch.float64)
    hz = torch.tensor([0.0, 0.0, 0.9], dtype=torch.float64)
    as_tensor = torch.tensor([[TOL.approach_margin_m, TOL.half_width_m, TOL.height_tol_m]] * 3,
                             dtype=torch.float64)
    assert in_goal(down, lat, hz, TOL).tolist() == in_goal(down, lat, hz, as_tensor).tolist()
    assert torch.allclose(goal_distance_m(down, lat, hz, TOL),
                          goal_distance_m(down, lat, hz, as_tensor))


def test_a_wrong_width_tolerance_tensor_is_refused():
    """Reading three columns out of a four-column tensor would misbound two of three axes."""
    down = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        in_goal(down, down, down, torch.zeros(2, 4, dtype=torch.float64))


# ----------------------------------------------------------------------------------------
# The bootstrap spec, which breaks the calibration's circularity.
# ----------------------------------------------------------------------------------------

def test_the_bootstrap_spec_is_FINITE_so_the_gate_can_be_seen():
    """An infinite region renders as nothing, and the first pass is a visual check.

    The bootstrap run's whole purpose is (a) to record the goal geometry so the tolerances
    can be fitted, and (b) to put a gate on screen so the anchor, the terrain frame and the
    height can be eyeballed before anything is measured. (b) needs finite numbers.
    """
    t = GoalRegionSpec.bootstrap().default
    for v in (t.approach_margin_m, t.half_width_m, t.height_tol_m):
        assert math.isfinite(v) and v > 0.0


def test_the_bootstrap_spec_declares_itself_uncalibrated():
    """Its reached_goal is a placeholder, and a run under it must not read as a result.

    `runner.py` copies this flag into the table's meta and `report.py` refuses to derive
    spatial success from a table that carries it -- so the flag is the only thing standing
    between a placeholder and a plausible success rate.
    """
    assert GoalRegionSpec.bootstrap().provenance["bootstrap"] is True
    assert GoalRegionSpec().provenance.get("bootstrap") is None


def test_the_bootstrap_spec_survives_the_json_round_trip(tmp_path):
    path = str(tmp_path / "boot.json")
    GoalRegionSpec.bootstrap().save(path)
    assert GoalRegionSpec.load(path).provenance["bootstrap"] is True
