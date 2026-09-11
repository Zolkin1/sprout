# Copyright 2026 Zachary Olkin. All rights reserved.

"""The scene ``stair_probe`` writes must BE the staircase it says it is.

Every scene here is written with ``tag="pytest"``.  Without it these would land on exactly the
filenames a live labelling session uses -- the name is a pure function of the geometry -- and
the cleanup would delete the scene a queued trial was about to load.

The probe's whole output is a human label attached to a (width, speed) pair, so a scene whose
geometry does not match the pair it is filed under corrupts the measurement silently and
permanently -- there is no residual to notice afterwards, only a grid of labels against
numbers that were never realised.  These tests read the geometry back out of the COMPILED
MuJoCo model rather than out of the emitted string, so an XML that parses but compiles to
something else cannot pass.
"""

from __future__ import annotations

import os
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stair_probe as sp  # noqa: E402


def _geoms(model) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Named geoms -> ``(half_size, pos, group)`` from a compiled model."""
    out = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name:
            out[name] = (model.geom_size[i].copy(), model.geom_pos[i].copy(),
                         int(model.geom_group[i]))
    return out


@pytest.mark.parametrize("width,rise,run,steps", [(0.86, 0.15, 0.25, 6),
                                                  (5.11, 0.13, 0.28, 4),
                                                  (1.96, 0.17, 0.209, 8)])
def test_written_scene_has_the_requested_geometry(tmp_path, width, rise, run, steps):
    """Every tread's width, rise, run and position is the one that was asked for."""
    name, path = sp.write_scene(width, rise, run, steps, x_start=6.0, platform_depth=2.0,
                                descend=True, robot="g1_perceptive", tag="pytest")
    try:
        geoms = _geoms(mujoco.MjModel.from_xml_path(path))
        for n in range(1, steps + 1):
            half, pos, group = geoms[f"stairs_step{n}"]
            assert group == 2, "terrain must be in the raycast/collision group"
            # Half-extents, because that is what MuJoCo stores and what the ray hits.
            assert half == pytest.approx([run / 2, width / 2, n * rise / 2], abs=1e-6)
            assert pos == pytest.approx([6.0 + (n - 0.5) * run, 0.0, n * rise / 2], abs=1e-6)
        # The first riser's FRONT FACE is x_start: the approach the operator was told about.
        half, pos, _ = geoms["stairs_step1"]
        assert pos[0] - half[0] == pytest.approx(6.0, abs=1e-6)
        # And the flight is centred on the spawn line, so the approach is head-on.
        assert pos[1] == pytest.approx(0.0, abs=1e-9)
    finally:
        os.remove(path)
    assert name.startswith("stair_probe_pytest_")
    # The approach length is part of the identity: two sweeps that differ only in --x_start
    # are different experiments and must not share a scene file.
    assert name != sp.scene_name(width, rise, run, steps, 3.0, tag="pytest")


def test_the_probe_scene_holds_nothing_but_one_staircase():
    """No jump box, no second lane, no wall -- a stairs-vs-jump question cannot afford one.

    ``generate_stairs_scene``'s default playground carries four jump boxes and three extra
    lanes.  Any of them in frame is the obvious alternative explanation for a plan that
    looks like a jump, so the probe scene must contain none.
    """
    _, path = sp.write_scene(0.86, 0.15, 0.25, 6, x_start=6.0, platform_depth=2.0,
                             descend=True, robot="g1_perceptive", tag="pytest")
    try:
        names = set(_geoms(mujoco.MjModel.from_xml_path(path)))
    finally:
        os.remove(path)
    terrain = {n for n in names if n.startswith(("stairs_", "box", "play_wall", "wide_",
                                                 "center_", "narrow_", "slow_tread_"))}
    assert not {n for n in terrain if not n.startswith("stairs_")}, (
        f"the probe scene carries geometry beyond its one staircase: {sorted(terrain)}")


def test_every_label_has_a_colour():
    """A label with no colour would KeyError when the session's figure is drawn -- at the end.

    :func:`stair_probe.plot_grid` builds its colormap by indexing a dict with every member of
    :data:`stair_probe.LABELS`, and it runs only after the operator has labelled everything.
    ``_finish`` catches the failure so the labels survive, but the figure would be lost.
    """
    import inspect

    src = inspect.getsource(sp.plot_grid)
    for value in sp.LABELS.values():
        assert f'"{value}"' in src, f"label {value!r} has no entry in plot_grid's colour map"


@pytest.mark.parametrize("speed", [0.6, 1.1, 1.6, 2.1, 2.6, 3.0])
def test_the_trial_budget_outlasts_the_climb(speed):
    """A budget that expires mid-flight is indistinguishable from a refusal to climb.

    The check is deliberately generous-side-only: the operator can always end a trial early by
    closing the viewer, so the cost of a long budget is patience and the cost of a short one is
    a wrong label.
    """
    x_start, run, steps, platform = 6.0, 0.25, 6, 2.0
    budget = sp.trial_seconds(speed, x_start, run, steps, platform, settle=4.0, cap=45.0)
    # Distance to the far edge of the top platform, walked at the commanded speed.
    needed = (x_start + steps * run + platform) / speed
    assert budget >= needed + 1.0, (
        f"at {speed} m/s the budget is {budget:.1f} s but reaching the top platform's far "
        f"edge takes {needed:.1f} s even at the commanded speed")


def test_the_default_widths_and_speeds_are_ones_the_library_can_be_asked_about():
    """Each default sits where its note says: inside a trained band, or knowingly outside."""
    for w in sp.DEFAULT_WIDTHS:
        assert "OUTSIDE" not in sp.width_note(w), f"default width {w} is outside every band"
    over = [v for v in sp.DEFAULT_SPEEDS if "ABOVE" in sp.speed_note(v)]
    assert over, ("the sweep must reach past the 2.600 m/s stair ceiling -- that gap is the "
                  "mechanism it exists to probe")
    assert not [v for v in sp.DEFAULT_SPEEDS if "BELOW" in sp.speed_note(v)]


# --- the joystick arm -----------------------------------------------------------------------
# These exist because two of this arm's pieces were written and silently did not land: a
# `str.replace` against a needle that was not in the file leaves the file unchanged and says
# nothing, so `--joystick` ran with its cell key and its approach readout simply absent. A
# test that the wiring is REACHABLE is the cheap guard against that whole class.


def test_a_joystick_cell_has_no_speed_and_still_keys_uniquely():
    """`speed=None` must round-trip: a joystick trial has no commanded speed to key on."""
    a = sp._cell_key(0.86, None, 0.0, 0.0, 0)
    b = sp._cell_key(0.86, None, 0.0, 0.0, 1)
    c = sp._cell_key(0.86, 1.6, 0.0, 0.0, 0)
    assert a != b, "repeats of one width must be distinct cells"
    assert a != c, "a joystick trial and a 1.6 m/s trial are not the same cell"
    assert a == sp._cell_key("0.8600", None, 0.0, 0.0, 0), "must survive the JSON round trip"


def test_the_command_axes_separate_cells_and_leave_old_sessions_resumable():
    """A lateral or yaw command is part of a cell's identity, and absent means zero.

    Every trial run before these axes existed had v_y = w_z = 0 -- that is what a
    ``--const_cmd vx 0 0`` run WAS -- so a row with no ``lateral`` field must key identically
    to one that records 0.0, or every earlier session would re-run on resume.
    """
    base = sp._cell_key(1.52, 1.0, 0.0, 0.0, 0)
    assert sp._cell_key(1.52, 1.0, None, None, 0) == base, "absent must mean zero"
    assert sp._cell_key(1.52, 1.0, 0.2, 0.0, 0) != base, "a lateral command is a new cell"
    assert sp._cell_key(1.52, 1.0, 0.0, 0.2, 0) != base, "a yaw command is a new cell"
    # And lateral must not be confusable with yaw: the whole point is telling them apart.
    assert sp._cell_key(1.52, 1.0, 0.2, 0.0, 0) != sp._cell_key(1.52, 1.0, 0.0, 0.2, 0)


def test_the_grid_x_axis_follows_whatever_was_actually_swept():
    """Sweeping --lateral at one speed must not plot a one-column speed axis.

    The failure this pins is silent: the figure renders, the labels are all there, and the
    swept variable simply does not appear on it.
    """
    lat = [{"width": 1.52, "speed": 1.0, "lateral": v, "yaw": 0.0, "label": "stairs"}
           for v in (0.0, 0.1, 0.2)]
    field, vals, label = sp._x_axis(lat, [1.0])
    assert field == "lateral" and vals == [0.0, 0.1, 0.2], (field, vals)
    assert "v_y" in label

    spd = [{"width": 1.52, "speed": v, "lateral": 0.0, "yaw": 0.0, "label": "stairs"}
           for v in (0.6, 1.6, 2.6)]
    field, vals, label = sp._x_axis(spd, [0.6, 1.6, 2.6])
    assert field == "speed" and vals == [0.6, 1.6, 2.6], (field, vals)


def test_command_stats_recovers_a_known_constant_command(tmp_path):
    """The approach summary must read back what was commanded, split at the first riser."""
    log = tmp_path / "sim_log.csv"
    rows = ["time,cmd_vx,cmd_vy,cmd_wz,act_pos_x"]
    for i in range(100):                       # x from 0 to ~5 m, first riser at 3.0
        rows.append(f"{i * 0.02},1.6,-0.25,0.4,{i * 0.05}")
    log.write_text("\n".join(rows) + "\n")

    st = sp.command_stats(str(log), x_start=3.0)
    assert st["all_n"] == 100
    assert st["app_n"] == 60, "the approach is every step before x = 3.0"
    assert st["app_reached"] is True
    assert st["app_vx_med"] == pytest.approx(1.6)
    # SIGNED, so a stick held one way is distinguishable from one flicked both ways.
    assert st["app_vy_med"] == pytest.approx(-0.25)
    assert st["app_vy_absmed"] == pytest.approx(0.25)
    assert st["app_wz_med"] == pytest.approx(0.4)


def test_command_stats_reports_a_trial_that_never_reached_the_flight(tmp_path):
    """`app_reached` False means the label describes an approach, not a staircase."""
    log = tmp_path / "sim_log.csv"
    rows = ["time,cmd_vx,cmd_vy,cmd_wz,act_pos_x"] + [
        f"{i * 0.02},0.1,0,0,{i * 0.001}" for i in range(50)]
    log.write_text("\n".join(rows) + "\n")
    st = sp.command_stats(str(log), x_start=3.0)
    assert st["app_reached"] is False
    assert st["app_n"] == 50


def test_every_label_has_a_colour_in_the_joystick_figure_too():
    """`plot_joystick` builds its own colour map, so it can rot independently of the grid."""
    import inspect

    src = inspect.getsource(sp.plot_joystick)
    for value in sp.LABELS.values():
        assert f'"{value}"' in src, f"label {value!r} has no entry in plot_joystick's colours"
