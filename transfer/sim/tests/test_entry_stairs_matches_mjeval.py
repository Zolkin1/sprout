# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac entry-stairs tile and the MuJoCo entry-stairs scene must be the SAME building.

``motion_gen.terrain.entry_stairs`` restates the dimensions in
``transfer/sim/generate_entry_stairs_scene.py`` rather than importing them, for the same reason
``box2`` and ``step_courses`` restate mjeval's: ``source/motion_gen`` is the offline generation
package and stays free of ``transfer`` exactly as it stays free of IsaacLab -- and ``transfer``
runs in ``env_transfer``, which does not carry ``motion_gen``, so an import either way breaks
one of the two environments.

This is the file that makes the restatement safe. It imports BOTH and requires them to agree,
so a tape-measure number edited on one side fails here instead of quietly producing an Isaac
fine-tune and a MuJoCo evaluation on two different staircases.

**These are MEASURED dimensions**, which makes a silent divergence worse than usual: there is no
generating rule to re-derive them from, so a wrong number here cannot be caught by inspection --
6.00 in and 6.25 in are equally plausible risers, and only the tape says which.

The comparison is in INCHES, the units they were taken in, because that is where a transcription
error is visible: 0.1524 and 0.1587 m look alike, 6.00 and 6.25 in do not.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.dirname(_HERE)
# Both levels, as in the sibling mjeval tests: mjeval's modules import each other by bare name
# while ``generate_*_scene`` imports ``sim.robot`` as a package.
for _d in (os.path.join(_SIM, "mjeval"), _SIM, os.path.dirname(_SIM)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

es = pytest.importorskip("motion_gen.terrain.entry_stairs")
gen = pytest.importorskip("generate_entry_stairs_scene")


#: Every measured dimension, as ``(attribute on each side)``. Stated once here so a dimension
#: added to one side and not the other fails the completeness test below rather than being
#: silently uncompared -- which is the failure mode a hand-written list of asserts has.
_MEASURED = (
    "WIDE_STEPS", "WIDE_RISE_IN", "WIDE_RUN_IN", "WIDE_WIDTH_IN",
    "LANDING_LEN_IN",
    "HALL_STEPS", "HALL_RISE_IN", "HALL_RUN_IN", "HALL_WIDTH_IN",
)

#: The dimensions that were NOT measured but still define one building: the run-up, the lobby
#: the hallway opens into, and the walls. Stated in metres on both sides, which is why they
#: need their own list -- the ``*_IN`` sweep below cannot see them, and that blind spot is not
#: hypothetical. The first Isaac port matched every tape measurement exactly and still built a
#: different building: it ended the hallway in a 2.54 m corridor where MuJoCo opens into an
#: 8 x 8 m lobby, so the WIDENING -- half the reason this course exists -- was absent from one
#: simulator, at exactly the moment the generator is looking at it.
_SHARED_M = ("APPROACH_M", "CLEARING_LEN_M", "CLEARING_WIDTH_M", "WALL_H_M", "WALL_T_M")

#: Dimensions that exist on the MuJoCo side ALONE, by design -- the side box.
#:
#: It is a joystick test fixture, not part of the building: one scene that offers the
#: generator both modes it has to choose between (drive straight = staircase, steer = box).
#: The Isaac tile deliberately does NOT have it, and that asymmetry has to be enforced rather
#: than merely allowed, which is what the second test below does.
#:
#: **Why it must not be added to the Isaac side.** ``entry_stairs`` tiles are in the ``multi``
#: fine-tune library, and a tile's geometry is hashed into the terrain fingerprint that a
#: training run records. Adding a box there changes the fingerprint, so every artifact built
#: against the old one silently describes different ground.
_MUJOCO_ONLY = ("SIDE_BOX_H_M", "SIDE_BOX_DEPTH_M", "SIDE_BOX_WIDTH_M",
                "SIDE_BOX_X_M", "SIDE_BOX_Y_M")


@pytest.mark.parametrize("name", _MEASURED)
def test_the_two_sims_agree_on_every_measured_dimension(name: str) -> None:
    """One assert per dimension, so a failure names the number that moved."""
    assert hasattr(es, name), f"{name} missing from the Isaac side"
    assert hasattr(gen, name), f"{name} missing from the MuJoCo side"
    assert getattr(es, name) == pytest.approx(getattr(gen, name)), (
        f"{name}: Isaac has {getattr(es, name)}, MuJoCo has {getattr(gen, name)}. These are "
        "TAPE-MEASURED dimensions of one real building -- one of the two is now wrong and "
        "nothing but this test can tell which.")


def test_the_inch_conversion_is_the_same_on_both_sides() -> None:
    """A different IN would make every dimension disagree while every constant matched."""
    assert es.IN == pytest.approx(gen.IN)
    assert es.IN == pytest.approx(0.0254)


@pytest.mark.parametrize("name", _SHARED_M)
def test_the_two_sims_agree_on_every_shared_metre_dimension(name: str) -> None:
    """One assert per dimension, so a failure names the number that moved."""
    assert hasattr(es, name), f"{name} missing from the Isaac side"
    assert hasattr(gen, name), f"{name} missing from the MuJoCo side"
    assert getattr(es, name) == pytest.approx(getattr(gen, name)), (
        f"{name}: Isaac has {getattr(es, name)}, MuJoCo has {getattr(gen, name)}. These are "
        "not tape measurements, but they are still ONE building -- a robot that climbs this "
        "staircase in two simulators must arrive in the same room.")


def test_no_dimension_escapes_the_comparison() -> None:
    """The lists above must cover every dimensional constant the MuJoCo side declares.

    Without this, adding a dimension to both sides and forgetting to list it here leaves it
    uncompared forever -- and it would be uncompared in exactly the way this file exists to
    prevent.
    """
    declared = {n for n in dir(gen)
                if (n.endswith(("_IN", "_STEPS", "_M"))) and not n.startswith("_")}
    missing = declared - set(_MEASURED) - set(_SHARED_M) - set(_MUJOCO_ONLY)
    assert not missing, (
        f"dimension(s) not compared: {sorted(missing)}. Add them to _MEASURED / _SHARED_M if "
        "both simulators must agree, or to _MUJOCO_ONLY -- with a reason -- if the Isaac tile "
        "deliberately does not have them.")


def test_the_side_box_is_mujoco_only() -> None:
    """The asymmetry is ENFORCED, not merely tolerated.

    Without this, someone adding the box to the Isaac tile to "make the two match" would be
    changing the terrain fingerprint of a library that fine-tune runs were recorded against --
    and every check in the repo would go on passing, because both sides would agree.
    """
    for name in _MUJOCO_ONLY:
        assert hasattr(gen, name), f"{name} vanished from the MuJoCo side"
        assert not hasattr(es, name), (
            f"{name} has appeared on the Isaac side. The side box is a MuJoCo joystick "
            "fixture: putting it in the tile changes the terrain fingerprint that the multi "
            "fine-tune library was built with, so artifacts would describe different ground "
            "while every comparison still passed.")


def test_the_derived_heights_follow_from_the_measurements() -> None:
    """The conversion, checked against the arithmetic rather than against itself."""
    g = es.geometry_m()
    assert g["wide_height_m"] == pytest.approx(5 * 6.00 * 0.0254)      # 0.762 m
    assert g["hall_height_m"] == pytest.approx(3 * 5.25 * 0.0254)      # 0.400 m
    assert g["total_height_m"] == pytest.approx(g["wide_height_m"] + g["hall_height_m"])


def test_the_two_flights_really_do_disagree() -> None:
    """The property the tile exists for, asserted so a 'tidy-up' cannot quietly remove it.

    If the two flights ever share a geometry this stops being the course that asks the
    generator to re-read mid-traverse, and becomes a long staircase with a landing -- which
    ``varied_courses`` already produces.
    """
    g = es.geometry_m()
    assert g["wide_rise_m"] != pytest.approx(g["hall_rise_m"])
    assert g["wide_run_m"] != pytest.approx(g["hall_run_m"])
    # And the RATIO differs too, so it is not one staircase drawn at two scales.
    assert (g["wide_rise_m"] / g["wide_run_m"]) != pytest.approx(
        g["hall_rise_m"] / g["hall_run_m"], rel=0.05)


def test_the_corridor_narrows() -> None:
    """The other property it exists for: a hard lateral constraint before a riser."""
    g = es.geometry_m()
    assert g["hall_width_m"] < g["wide_width_m"] / 1.5, (
        "the hallway must be materially narrower than the entrance, or the tile carries no "
        "lateral constraint the other courses lack")
