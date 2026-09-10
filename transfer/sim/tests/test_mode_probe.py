# Copyright 2026 Zachary Olkin. All rights reserved.
"""The mode-selection probe, and the two course options it needed.

The regression that matters most here is the LAST one: ``isaac_tile_course``'s defaults decide
the approach/on-course split every existing mjeval table was reduced with, so the two new
options have to be invisible unless asked for.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import generate_clip_tile_scene as gcts  # noqa: E402
import mode_probe  # noqa: E402
from mjeval import terrains  # noqa: E402

REPO = os.path.dirname(os.path.dirname(_HERE))
BOX2 = os.path.join(REPO, "clips", "box2_v2spawn_train.h5")
UPDOWN = os.path.join(REPO, "clips", "updown_p020_r030_train.h5")
CLIPLIB = os.path.join(REPO, "clips", "g1_mb_v5_body_mode11_decor.h5")


def _ov(cx: float, depth: float, base_z: float, height: float, *,
        walkable: bool = True) -> dict:
    """One overlay dict, centred laterally at 0."""
    return {"kind": "box", "pos_xy": [cx, 0.0], "size_xyz": [depth, 2.0, height],
            "base_z": base_z, "walkable_top": walkable, "yaw": 0.0, "decor": False,
            "variant": "solid", "slab_thickness": 0.1}


# ---------------------------------------------------------------------------------------------
# first_raised: the half of the TODO that first_step does not answer
# ---------------------------------------------------------------------------------------------
def test_first_raised_finds_a_box_too_deep_to_be_a_tread() -> None:
    """The bug this exists for: a 1.20 m box is invisible to ``first_step``.

    ``_STEP_MAX_TREAD_M`` is 0.60, and the filter is there so a railing post cannot pass as a
    riser -- so the two-box course's boxes are skipped and the tile reads as "not a feature
    course" at all.
    """
    overlays = [_ov(7.0, 14.0, -0.05, 0.05), _ov(3.1, 1.2, 0.0, 0.40)]
    assert gcts.first_step(overlays) is None, "the tread filter must still reject a 1.2 m box"
    got = gcts.first_raised(overlays, 0.0)
    assert got is not None and got["size_xyz"][0] == pytest.approx(1.2)


def test_first_raised_ignores_the_ground_it_stands_on() -> None:
    """The ground overlay is walkable too, and picking it would make the approach zero."""
    overlays = [_ov(7.0, 14.0, -0.05, 0.05), _ov(3.1, 1.2, 0.0, 0.40)]
    got = gcts.first_raised(overlays, 0.0, min_rise_m=0.05)
    assert got["base_z"] + got["size_xyz"][2] == pytest.approx(0.40)


def test_first_raised_takes_the_NEAREST_raised_overlay() -> None:
    overlays = [_ov(7.0, 14.0, -0.05, 0.05), _ov(6.0, 1.2, 0.0, 0.40),
                _ov(3.1, 1.2, 0.0, 0.40)]
    assert gcts.first_raised(overlays, 0.0)["pos_xy"][0] == pytest.approx(3.1)


def test_first_raised_skips_a_surface_already_behind_the_spawn() -> None:
    """A box the robot is standing ON is not its own approach datum.

    That is what a box-top spawn is, and without ``from_x`` the datum would be the surface
    under the robot's feet -- an approach of zero, and a label window that closes immediately.
    """
    overlays = [_ov(7.0, 14.0, -0.05, 0.05), _ov(3.1, 1.2, 0.0, 0.40),
                _ov(7.3, 1.2, 0.0, 0.40)]
    got = gcts.first_raised(overlays, 0.40, from_x=3.5, min_rise_m=0.05)
    # Standing on box 1 (surface 0.40), the next datum must be box 2 and not box 1.
    assert got is None or got["pos_xy"][0] == pytest.approx(7.3)


def test_first_raised_returns_none_when_nothing_stands_above() -> None:
    assert gcts.first_raised([_ov(7.0, 14.0, -0.05, 0.05)], 0.0) is None


# ---------------------------------------------------------------------------------------------
# route_spawn
# ---------------------------------------------------------------------------------------------
def test_route_spawn_takes_the_tiles_own_declared_start() -> None:
    entry = {"terrain_id": 0, "size_x": 14.0, "size_y": 6.0,
             "route": {"start_x": 1.25, "start_y": 3.0, "start_z": 0.0}}
    overlays = [_ov(7.0, 14.0, -0.05, 0.05)]
    overlays[0]["pos_xy"] = [7.0, 3.0]
    x, y, z, why = gcts.route_spawn(entry, overlays)
    assert (x, y) == pytest.approx((1.25, 3.0))
    # z is the SURFACE there, not the route's declared start_z: the route states a base offset
    # and the robot has to be stood on whatever it is standing on.
    assert z == pytest.approx(0.0)
    assert "route spawn" in why


def test_route_spawn_refuses_a_tile_with_no_route() -> None:
    with pytest.raises(ValueError, match="declares no route start"):
        gcts.route_spawn({"terrain_id": 3}, [_ov(7.0, 14.0, -0.05, 0.05)])


# ---------------------------------------------------------------------------------------------
# isaac_tile_course: the options, and the defaults staying put
# ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.isfile(BOX2), reason="box2 library not built here")
def test_the_two_box_tile_is_buildable_only_with_the_raised_datum() -> None:
    """Both halves: the default still refuses, and it says what to pass."""
    with pytest.raises(ValueError, match="obstacle_from='raised'"):
        terrains.isaac_tile_course(BOX2, 2, speeds=(1.17,), klass="two_box_course",
                                   spawn_from="route")
    c = terrains.isaac_tile_course(BOX2, 2, speeds=(1.17,), klass="two_box_course",
                                   spawn_from="route", obstacle_from="raised")
    assert c.obstacle_x_m > 0.0


@pytest.mark.skipif(not os.path.isfile(BOX2), reason="box2 library not built here")
def test_the_route_spawn_reproduces_the_tiles_own_approach_distance() -> None:
    """Tile i of the two-box grid was authored at ``TRAIN_SPAWN_DISTANCES_M[i]``.

    So a course built from the ROUTE must measure exactly that distance to the first box face
    -- which is what makes the MuJoCo opening state the Isaac one rather than merely similar.
    """
    want = (1.5, 2.0, 2.5, 3.0, 4.0)
    for tid, d in enumerate(want):
        c = terrains.isaac_tile_course(BOX2, tid, speeds=(1.17,), klass="two_box_course",
                                       spawn_from="route", obstacle_from="raised")
        assert c.obstacle_x_m == pytest.approx(d, abs=1e-6), f"tile {tid}"


@pytest.mark.skipif(not os.path.isfile(UPDOWN), reason="updown library not built here")
def test_the_updown_tile_builds_both_ways() -> None:
    """Its treads are 0.25 m, so the default datum works -- and both must agree on the face."""
    a = terrains.isaac_tile_course(UPDOWN, 2, speeds=(1.17,), klass="up_into_down",
                                   spawn_from="route")
    b = terrains.isaac_tile_course(UPDOWN, 2, speeds=(1.17,), klass="up_into_down",
                                   spawn_from="route", obstacle_from="raised")
    assert a.obstacle_x_m == pytest.approx(b.obstacle_x_m)


@pytest.mark.skipif(not os.path.isfile(CLIPLIB), reason="clip library not present")
def test_the_defaults_are_unchanged_on_a_clip_library_tile() -> None:
    """The regression guard.

    ``obstacle_x_m`` is the approach/on-course split ``mjeval.accumulate`` reduces every table
    with, and ``spawn_point`` places the spawn from the same datum -- so if either new option
    leaked into the defaults, every number already measured would move without anything
    failing.  A clip-library tile declares no route, so the route arm must not be reachable by
    accident either.
    """
    for tid in (6, 20, 40):
        c = terrains.isaac_tile_course(CLIPLIB, tid, speeds=(0.8,))
        assert c.obstacle_x_m == pytest.approx(2.5, abs=1e-6), f"tile {tid}"
    with pytest.raises(ValueError, match="declares no route start"):
        terrains.isaac_tile_course(CLIPLIB, 6, speeds=(0.8,), spawn_from="route")


def test_the_option_names_are_validated() -> None:
    """A typo must not fall through to a silent default."""
    for kw, msg in (({"spawn_from": "routes"}, "spawn_from must be"),
                    ({"obstacle_from": "treads"}, "obstacle_from must be")):
        with pytest.raises(ValueError, match=msg):
            terrains.isaac_tile_course(UPDOWN, 2, speeds=(1.17,), klass="up_into_down", **kw)


# ---------------------------------------------------------------------------------------------
# The probe's own bookkeeping
# ---------------------------------------------------------------------------------------------
def test_every_class_the_family_table_maps_to_declares_an_expected_mode() -> None:
    """Otherwise the report scores the correct-mode rate against an empty string, silently."""
    for fam, klass in mode_probe.FAMILY_KLASS.items():
        assert klass in terrains.CLASS_CODES, fam
        assert mode_probe.EXPECTED_MODE.get(klass), f"{fam} -> {klass} has no expected mode"


def test_the_expected_modes_are_labels_a_human_can_actually_give() -> None:
    for klass, mode in mode_probe.EXPECTED_MODE.items():
        assert mode in mode_probe.MODES, klass


def test_the_key_map_covers_every_mode() -> None:
    """A mode with no key could never be recorded."""
    assert set(mode_probe.KEYS.values()) == set(mode_probe.MODES)


def test_the_blind_order_is_a_deterministic_function_of_the_seed() -> None:
    """A resumed labelling session must continue the same sequence.

    The labels are keyed by POSITION in this order, so a different order on resume would
    attribute every existing label to the wrong trial.
    """
    pm = {"arms": {"parent": {"rows": [{"seed": i} for i in range(4)]},
                   "finetuned": {"rows": [{"seed": i} for i in range(4)]}}}
    a = mode_probe.order_of(pm, 7)
    assert a == mode_probe.order_of(pm, 7)
    assert a != mode_probe.order_of(pm, 8)
    assert sorted(a) == sorted([("parent", i) for i in range(4)]
                               + [("finetuned", i) for i in range(4)])


def test_the_blind_order_interleaves_the_arms() -> None:
    """Both arms must be mixed, or the labeller sees one arm's block then the other's and the
    blinding is defeated by position alone."""
    pm = {"arms": {"parent": {"rows": [{"seed": i} for i in range(25)]},
                   "finetuned": {"rows": [{"seed": i} for i in range(25)]}}}
    order = mode_probe.order_of(pm, 20260909)
    first_half = [a for a, _ in order[:25]]
    assert 5 <= first_half.count("parent") <= 20, (
        f"the first half is {first_half.count('parent')}/25 parent, which is not interleaved")


def test_resolve_klass_prefers_an_explicit_override() -> None:
    entry = {"family": "box2_v2"}
    assert mode_probe.resolve_klass(entry, "up_into_down") == "up_into_down"
    assert mode_probe.resolve_klass(entry, "") == "two_box_course"


def test_resolve_klass_refuses_an_unknown_family_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="maps to no course class"):
        mode_probe.resolve_klass({"family": "family_something_new.h5"}, "")


def test_the_probe_manifest_round_trips(tmp_path) -> None:
    """``--stage label`` and ``--stage report`` read only this file, so it must carry
    everything they need to rebuild the same course."""
    pm = {"clip": "x.h5", "terrain_id": 2, "klass": "two_box_course", "speed": 1.17,
          "approach_m": 2.5, "obstacle_x_m": 2.5, "label_x_m": 4.5,
          "robot_name": "g1_perceptive", "expected_mode": "box",
          "arms": {"parent": {"fm_onnx": "p.onnx", "trials": 2,
                              "rows": [{"seed": 0, "verdict": "success"},
                                       {"seed": 1, "verdict": "fall"}]}}}
    (tmp_path / "probe.json").write_text(json.dumps(pm))
    got = mode_probe.load_probe(str(tmp_path))
    for k in ("clip", "terrain_id", "klass", "speed", "robot_name", "expected_mode"):
        assert got[k] == pm[k]


def test_a_missing_probe_manifest_says_what_to_run(tmp_path) -> None:
    with pytest.raises(SystemExit, match="--stage run first"):
        mode_probe.load_probe(str(tmp_path))


# ---------------------------------------------------------------------------------------------
# The duplicate ground plane
# ---------------------------------------------------------------------------------------------
def test_authors_ground_needs_the_footprint_AND_the_height() -> None:
    """Either test alone is wrong.

    A clip-library tile's approach platform also spans the whole footprint -- 7.50 x 3.84 on
    one of them -- but its top is a riser up at +0.170, so it is a PLATFORM and the tile still
    needs a floor beneath it.  A free-form ground slab is full-footprint AND tops out at 0.
    """
    entry = {"size_x": 7.5, "size_y": 3.84}
    platform = _ov(3.75, 7.5, 0.0, 0.17)
    platform["pos_xy"] = [3.75, 1.92]
    platform["size_xyz"] = [7.5, 3.84, 0.17]
    assert not gcts.authors_ground(entry, [platform]), "a raised platform is not the ground"
    slab = dict(platform)
    slab["base_z"], slab["size_xyz"] = -0.5, [7.5, 3.84, 0.5]
    assert gcts.authors_ground(entry, [slab]), "a full-footprint slab topping at 0 IS the ground"


def test_authors_ground_ignores_a_small_patch_at_zero() -> None:
    """A step whose top happens to be at z=0 is not the tile's floor."""
    entry = {"size_x": 7.5, "size_y": 3.84}
    patch = _ov(1.0, 0.25, -0.5, 0.5)
    assert not gcts.authors_ground(entry, [patch])


@pytest.mark.skipif(not os.path.isfile(BOX2), reason="box2 library not built here")
def test_a_free_form_course_emits_exactly_one_ground_surface() -> None:
    """Two coplanar COLLIDABLE surfaces is not a cosmetic problem.

    The manifest's slab tops out at z=0 and ``write_scene``'s substitute plane sits at the same
    z, both in geom group 2 -- so they z-fought in the viewer AND doubled the ground contacts.
    Removing the plane moved one trajectory from 409 to 419 steps, so this was changing the
    physics, not just the picture.  Isaac builds only the slab (``has_floor=False``).
    """
    c = terrains.isaac_tile_course(BOX2, 2, speeds=(1.17,), klass="two_box_course",
                                   spawn_from="route", obstacle_from="raised",
                                   name="test_oneground_t2")
    assert c.has_own_ground is True
    path = terrains.write_scene(c, "g1_perceptive")
    xml = open(path, encoding="utf-8").read()
    assert 'type="plane"' not in xml, "the substitute plane is coplanar with the authored slab"
    os.remove(path)


@pytest.mark.skipif(not os.path.isfile(CLIPLIB), reason="clip library not present")
def test_a_clip_library_course_still_gets_its_plane() -> None:
    """The regression guard for the other direction: those tiles author no ground, so the
    plane is their ONLY floor and withholding it would drop the robot into the void."""
    c = terrains.isaac_tile_course(CLIPLIB, 6, speeds=(0.8,))
    assert c.has_own_ground is False
    path = terrains.write_scene(c, "g1_perceptive")
    xml = open(path, encoding="utf-8").read()
    assert 'type="plane"' in xml
    os.remove(path)


def test_the_rollout_fingerprint_moves_when_a_trial_moves() -> None:
    """Labels describe trajectories, so a re-roll must invalidate them.

    The arm set and the shuffle seed are unchanged by a re-roll, so they cannot catch it.
    """
    base = {"arms": {"parent": {"rows": [{"seed": 0, "verdict": "fall", "failure": "tilt",
                                          "steps": 409, "cut": 409}]}}}
    same = {"arms": {"parent": {"rows": [{"seed": 0, "verdict": "fall", "failure": "tilt",
                                          "steps": 409, "cut": 409}]}}}
    moved = {"arms": {"parent": {"rows": [{"seed": 0, "verdict": "fall", "failure": "tilt",
                                           "steps": 419, "cut": 413}]}}}
    assert mode_probe._rollout_fingerprint(base) == mode_probe._rollout_fingerprint(same)
    assert mode_probe._rollout_fingerprint(base) != mode_probe._rollout_fingerprint(moved)
