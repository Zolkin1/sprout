# Copyright 2026 Zachary Olkin. All rights reserved.

"""The velocity-tracking lanes must be the terrain the figure DRAWS, and end where they claim.

``vel_tracking_sweep`` emits two descriptions of one lane: the MuJoCo geoms the robot walks on
and the ``profile_x``/``profile_z`` polyline ``plot_vel_tracking`` fills under the trajectory.
They come out of one function from one set of numbers, and this is the file that keeps them
honest -- a drawn staircase half a tread from the simulated one would attribute every velocity
excursion in the figure to the wrong step, and nothing in the figure would look wrong.

The geometry side is checked by RAYCASTING the loaded model, not by re-deriving the slab
extents: re-deriving them here would restate the builder a third time, and the third statement
would be the one that agrees with the bug.

Also pinned: the goal line really is ``--runout`` metres past the feature (the episode's stop
condition is a band about a CENTRE, so the near edge is what the runout has to land on), and
the trace's column order, which the ``.npz``'s per-step arrays are named from.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.dirname(_HERE)
for _d in (os.path.join(_SIM, "mjeval"), _SIM, os.path.dirname(_SIM)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

mujoco = pytest.importorskip("mujoco")
terrains = pytest.importorskip("terrains")
vt = pytest.importorskip("vel_tracking_sweep")

APPROACH, RUNOUT = 7.0, 6.0

#: How far (m) a raycast sample must stay from a VERTICAL FACE to be asserted on.
#:
#: On a face the two descriptions are both right and disagree: the profile's segment lookup
#: takes the higher surface at a shared x while ``mj_ray`` at the geom's exact boundary may
#: return whatever is behind it. It is a knife edge, and float accumulation in ``arange`` lands
#: on one -- ``140 * 0.05`` is ``7.000000000000001``, a femtometre inside a box that starts at
#: 7.0, which the profile calls "box top" and the ray called "floor". That is the same coin flip
#: the height-scan grid test documents on stair edges. Skipping a millimetre either side costs
#: nothing (every surface here is at least 0.46 m along travel) and keeps the assertion exact
#: everywhere it means something.
FACE_SKIP_M: float = 1e-3


@pytest.fixture(scope="module")
def built() -> dict:
    """One lane per kind, at a non-default approach/runout so no constant can pass by luck."""
    geom = vt.Geometry()
    return {k: vt.build_terrain(k, approach=APPROACH, runout=RUNOUT, geom=geom,
                                speeds=(0.8,), tag="_pytest")
            for k in vt.TERRAIN_KINDS}


def surface_z(profile_x, profile_z, x: float) -> float | None:
    """The profile's surface height at downrange ``x``, or ``None`` where it says nothing.

    The polyline carries a vertical face as two samples sharing an x, so only the segments
    with a positive x extent are surfaces; the highest of those covering ``x`` is the one the
    robot would stand on.
    """
    best = None
    for i in range(len(profile_x) - 1):
        x0, x1 = profile_x[i], profile_x[i + 1]
        if x1 > x0 and x0 <= x <= x1:
            best = profile_z[i] if best is None else max(best, profile_z[i])
    return best


@pytest.mark.parametrize("kind", vt.TERRAIN_KINDS)
def test_profile_matches_the_simulated_geometry(built, kind):
    """A ray dropped anywhere along the lane hits the surface the plotter would draw."""
    t = built[kind]
    path = terrains.write_scene(t.course)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Group 2 is the terrain group -- the same one the depth cameras cast and the fall datum
    # reads, which is why a course's geoms are all in it.
    group = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    faces = [t.profile_x[i] for i in range(len(t.profile_z) - 1)
             if t.profile_z[i] != t.profile_z[i + 1]]
    worst = 0.0
    for x in np.arange(0.0, t.goal_x + 2.0, 0.05):
        if any(abs(x - fx) < FACE_SKIP_M for fx in faces):
            continue
        want = surface_z(t.profile_x, t.profile_z, float(x))
        if want is None:
            continue
        gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, np.array([x, 0.0, 5.0]),
                             np.array([0.0, 0.0, -1.0]), group, 1, -1, gid)
        assert dist >= 0.0, f"{kind}: no terrain under x={x:.2f} m"
        worst = max(worst, abs((5.0 - dist) - want))
    assert worst < 1e-6, f"{kind}: drawn profile is {worst:.3e} m from the simulated surface"


@pytest.mark.parametrize("kind", vt.TERRAIN_KINDS)
def test_goal_line_is_exactly_runout_past_the_feature(built, kind):
    """The episode stops ``--runout`` metres past the feature, not at the goal CENTRE.

    ``Course.in_goal`` is a ``+-GOAL_DEPTH / 2`` band about ``target_xy``, so the target has to
    sit half a band FURTHER on for the near edge -- the boundary that actually ends the episode
    -- to land on the requested runout.
    """
    t = built[kind]
    near_edge = t.course.target_xy[0] - terrains.GOAL_DEPTH / 2.0
    assert near_edge == pytest.approx(t.x_exit + RUNOUT, abs=1e-9)
    assert t.goal_x == pytest.approx(near_edge, abs=1e-9)
    assert t.course.in_goal(t.goal_x + 1e-6, 0.0)
    assert not t.course.in_goal(t.goal_x - 0.01, 0.0)


@pytest.mark.parametrize("kind", vt.TERRAIN_KINDS)
def test_the_approach_is_flat_at_the_spawn_height(built, kind):
    """Every lane starts flat at z = 0, which is what the spawn keyframe is written against.

    The descending lane earns this by being dug DOWN -- a raised approach slab whose TOP is
    z = 0 over a lowered floor -- rather than by spawning the robot up in the air, which would
    need the keyframe and ``--spawn_z_offset`` restated.
    """
    t = built[kind]
    for x in (0.0, 0.5 * APPROACH, APPROACH - 0.1):
        assert surface_z(t.profile_x, t.profile_z, x) == pytest.approx(0.0, abs=1e-9), \
            f"{kind}: approach is not at z=0 at x={x}"


def test_flat_is_a_lane_with_no_terrain_in_it(built):
    """The control arm has no geoms and no height change -- otherwise it is not a control."""
    t = built["flat"]
    assert t.course.geoms == ()
    assert t.course.flight_height_m == 0.0
    assert t.x_enter == t.x_exit == APPROACH
    assert set(t.profile_z) == {0.0}


def test_stairs_rise_and_fall_by_their_declared_flight_height(built):
    """``flight_height_m`` is signed and is what the traversal gate reads, so it has to match."""
    up, dn = built["stairs_up"], built["stairs_down"]
    assert max(up.profile_z) == pytest.approx(up.course.flight_height_m, abs=1e-9)
    assert min(dn.profile_z) == pytest.approx(dn.course.flight_height_m, abs=1e-9)
    assert dn.course.floor_z == pytest.approx(dn.course.flight_height_m, abs=1e-9)
    # Same rise and run in both directions, so the two columns of the figure differ in sign
    # and in nothing else.
    assert up.course.params["rise_m"] == dn.course.params["rise_m"]
    assert up.course.params["run_m"] == dn.course.params["run_m"]


def test_the_stairs_up_platform_carries_the_whole_runout(built):
    """The lane cannot end in mid-air past the platform edge."""
    up = built["stairs_up"]
    far_edge = max(up.profile_x)
    assert far_edge >= up.goal_x + terrains.GOAL_DEPTH / 2.0


@pytest.mark.parametrize("kind", vt.TERRAIN_KINDS)
def test_the_time_budget_covers_the_lane(built, kind):
    """A trial that walks the lane at half the commanded speed still finishes inside budget."""
    t = built[kind]
    for speed in (0.5, 0.8, 1.5, 2.5):
        budget = vt.episode_budget_s(t.course, speed, hold_s=0.5, speed_frac=0.5)
        assert budget > 0.5 + t.course.course_length_m / speed


def test_trace_columns_match_the_recorder():
    """``TRACE_COLUMNS`` names the recorder's tuple, which the ``.npz``'s arrays are split by.

    A column added to one and not the other is silent: the arrays keep their shapes and every
    series after the insertion is read under its neighbour's name.
    """
    rec = vt.TraceRecorder("pelvis_link")
    assert rec.array().shape[1] == len(vt.TRACE_COLUMNS)
    assert vt.TRACE_COLUMNS[:4] == ("t", "x", "y", "z")
    # Achieved before commanded, each in (vx, vy, wz) order, so a pairing is positional.
    assert vt.TRACE_COLUMNS[4:] == ("vx", "vy", "wz", "cmd_vx", "cmd_vy", "cmd_wz")


def test_the_velocity_frame_comes_from_accumulate():
    """The trace must resolve velocity through ``accumulate``, not restate the frame.

    The commanded ``(vx, vy)`` is issued in the root's YAW frame; rotating the world velocity
    through the FULL root quaternion instead bleeds vertical velocity into the downrange
    channel exactly when the torso pitches, which is what a staircase does -- so a restated
    frame would show a tracking error that is really a frame error, on the one panel the figure
    is about.  ``accumulate.fold`` is the producer the eval's own velocity columns come from.
    """
    src = open(os.path.join(_SIM, "vel_tracking_sweep.py")).read()
    assert "mjacc.rotate_inverse(mjacc.yaw_quat(" in src
    assert "get_base_world_velocity" in src


def test_write_traces_round_trips_ragged_episodes(tmp_path):
    """The flat-plus-index layout has to reproduce each episode exactly, at differing lengths.

    Ragged by nature -- an episode ends when it arrives or falls -- and stored flat because a
    ragged ``.npz`` needs ``allow_pickle`` to read back.
    """
    rng = np.random.default_rng(0)
    arrs = [rng.normal(size=(n, len(vt.TRACE_COLUMNS))).astype(np.float32)
            for n in (7, 1, 23, 4)]
    eps = [{"kind": "flat", "terrain": 0, "speed": 0.8, "seed": i, "verdict": 0,
            "failure_code": 0, "steps": a.shape[0], "end_t": 1.0 * i, "end_x": 2.0 * i,
            "end_y": 0.0, "end_z": 0.79} for i, a in enumerate(arrs)]
    path = str(tmp_path / "traces.npz")
    vt.write_traces(path, arrs, eps)
    with np.load(path) as z:
        got = {k: z[k] for k in z.files}
    assert list(got["ep_len"]) == [a.shape[0] for a in arrs]
    for i, a in enumerate(arrs):
        s = slice(int(got["ep_start"][i]), int(got["ep_start"][i]) + int(got["ep_len"][i]))
        for c, name in enumerate(vt.TRACE_COLUMNS):
            np.testing.assert_array_equal(got[name][s], a[:, c])
    assert list(got["ep_seed"]) == [0, 1, 2, 3]


def test_an_unknown_terrain_is_refused():
    """Silently building nothing would put an empty column in the figure."""
    with pytest.raises(ValueError, match="unknown terrain"):
        vt.build_terrain("cliff", approach=1.0, runout=1.0, geom=vt.Geometry(),
                         speeds=(0.8,))
