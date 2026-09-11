# Copyright 2026 Zachary Olkin. All rights reserved.

"""The command may disagree with the motion on a traversal, but never point at a different one.

A terrain clip plays ONE fixed solved reference while the command varies across it. That is
deliberate and is most of what the generator has to learn: a command SELECTS a reference rather than
describing one, and a clip whose command and motion always agree cannot teach that.

The one invariant is that every commanded value must be inside the played reference's Voronoi cell --
the speeds for which ``select_terrain_reference`` returns that same reference. Outside it the clip
says "you asked for something nearer the other motion, and we gave you this one", which is the
opposite lesson. Pinned here by ROUND TRIP against the selector itself, not against a restatement of
its rule, so the two cannot drift apart.
"""

from __future__ import annotations

import pathlib
import sys
import types

import numpy as np
import pytest

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "source" / "motion_gen" / "motion_gen").is_dir():
        _REPO = _p
        break
else:                                                          # pragma: no cover - layout guard
    raise RuntimeError("could not locate the checkout above this test")
sys.path.insert(0, str(_REPO / "source" / "motion_gen"))
sys.path.insert(0, str(_REPO / "scripts" / "rsl_rl"))

torch = pytest.importorskip("torch")

from motion_gen.terrain.terrain_place import (  # noqa: E402
    select_terrain_reference, terrain_speed_cell)


def _src(speed: float, riser: float = 0.15, tread: float = 0.25):
    return types.SimpleNamespace(
        terrain_meta={"terrain": {"riser": riser, "tread": tread}},
        steady_vel=torch.tensor([float(speed), 0.0, 0.0]))


# Two solved speeds at one geometry, plus a decoy at a different one.
_STAIRS = [_src(0.31), _src(0.69), _src(0.30, riser=0.13, tread=0.22)]


def test_the_cell_splits_at_the_midpoint_of_the_solved_speeds():
    """The plan's worked example: {0.31, 0.69} -> cells (0, 0.50) and (0.50, inf)."""
    lo0, hi0 = terrain_speed_cell(_STAIRS, 0, riser=0.15, tread=0.25)
    lo1, hi1 = terrain_speed_cell(_STAIRS, 1, riser=0.15, tread=0.25)
    assert (lo0, hi0) == pytest.approx((0.0, 0.5))
    assert lo1 == pytest.approx(0.5) and hi1 == float("inf")


def test_every_speed_in_a_cell_ROUND_TRIPS_through_the_selector():
    """The invariant, checked against the real selector rather than a restatement of its rule."""
    for idx in (0, 1):
        lo, hi = terrain_speed_cell(_STAIRS, idx, riser=0.15, tread=0.25)
        hi_s = min(hi, 5.0)                                # the cell's outer edge is unbounded
        for v in np.linspace(lo + 1e-6, hi_s - 1e-6, 40):
            got = select_terrain_reference(_STAIRS, riser=0.15, tread=0.25, speed=float(v))
            assert got == idx, f"speed {v:.3f} is in cell {idx}'s range but selects {got}"


def test_a_speed_just_OUTSIDE_the_cell_selects_the_other_reference():
    """The boundary has to bite, or the cell is vacuous."""
    _lo, hi = terrain_speed_cell(_STAIRS, 0, riser=0.15, tread=0.25)
    assert select_terrain_reference(_STAIRS, riser=0.15, tread=0.25, speed=hi - 1e-3) == 0
    assert select_terrain_reference(_STAIRS, riser=0.15, tread=0.25, speed=hi + 1e-3) == 1


def test_one_solved_speed_gives_an_UNBOUNDED_cell():
    """Where a geometry has a single solved speed the command is free, and only the family band binds.

    That is not a loophole -- there is no other reference for it to point at. The cell narrows on its
    own as the solved library gains speeds at that geometry.
    """
    lo, hi = terrain_speed_cell([_src(0.33)], 0, riser=0.15, tread=0.25)
    assert lo == 0.0 and hi == float("inf")


def test_the_cell_is_a_property_of_the_GEOMETRY_candidates_only():
    """A reference at a different geometry must not move the cell -- the selector filters it out first."""
    near = terrain_speed_cell(_STAIRS, 1, riser=0.15, tread=0.25)
    with_decoys = terrain_speed_cell(
        _STAIRS + [_src(0.55, riser=0.13, tread=0.22), _src(2.0, riser=0.17, tread=0.28)],
        1, riser=0.15, tread=0.25)
    assert near == with_decoys


def test_asking_for_a_cell_of_a_reference_the_geometry_excludes_is_an_error():
    with pytest.raises(ValueError):
        terrain_speed_cell(_STAIRS, 2, riser=0.15, tread=0.25)   # index 2 is the other geometry


# ------------------------------------------------------------------ the sampler that uses the cell
def test_the_schedule_never_leaves_the_cell_or_the_band():
    from motion_gen import command_schedule as mtc

    rng = np.random.default_rng(0)
    for cell, band in (((0.5, float("inf")), (0.2, 2.6)), ((0.0, 0.5), (0.2, 2.6)),
                       ((0.0, float("inf")), (0.2, 3.5))):
        for _ in range(50):
            sched = mtc.terrain_interaction_schedule(rng, cell, band, 0.31, 3, 1.0)
            assert sched and abs(sum(seg[0] for seg in sched) - 1.0) < 1e-6
            for _f, vx, _vy, _wz in sched:
                assert cell[0] - 1e-9 <= abs(vx) <= cell[1] + 1e-9
                assert min(band) - 1e-9 <= abs(vx) <= max(band) + 1e-9


def test_a_family_sign_is_carried_and_an_empty_intersection_falls_back():
    from motion_gen import command_schedule as mtc

    rng = np.random.default_rng(0)
    back = mtc.terrain_interaction_schedule(rng, (0.0, float("inf")), (0.2, 1.0), 0.5, 2, -1.0)
    assert all(seg[1] < 0 for seg in back), back
    # A cell entirely above the band has no legal command; fall back to the reference's own speed.
    fb = mtc.terrain_interaction_schedule(rng, (9.0, 10.0), (0.2, 2.6), 0.44, 3, 1.0)
    assert len(fb) == 3 and all(seg[1] == pytest.approx(0.44) for seg in fb)


def test_one_segment_reproduces_the_old_constant():
    from motion_gen import command_schedule as mtc

    sched = mtc.terrain_interaction_schedule(np.random.default_rng(3), (0.0, float("inf")),
                                      (0.2, 2.6), 0.31, 1, 1.0)
    assert len(sched) == 1 and sched[0][0] == pytest.approx(1.0)


# --------------------------------------------- vy / wz on a traversal: free, and that is deliberate
def test_lateral_and_yaw_are_UNCONSTRAINED_by_the_cell():
    """The selector compares forward speed only, so vy/wz cannot move which reference is nearest.

    They are also unconstrained physically: the replay is world-pinned (``anchor_mask == 2``), so a
    lateral or yaw command on a traversal moves nothing whatever it says. Both channels used to be
    identically zero on every terrain clip, so the generator had never been asked to turn while a
    staircase was under it -- and could not have learned that the answer is to keep climbing.
    """
    from motion_gen import command_schedule as mtc

    rng = np.random.default_rng(1)
    seen_vy = seen_wz = False
    for _ in range(200):
        for _f, _vx, vy, wz in mtc.terrain_interaction_schedule(
                rng, (0.5, float("inf")), (0.2, 2.6), 0.69, 3, 1.0,
                vy_range=(-0.6, 0.6), wz_range=(-0.8, 0.8)):
            assert abs(vy) <= 0.6 + 1e-9 and abs(wz) <= 0.8 + 1e-9
            # At most one of the two -- the flat sampler's rule, and the solved library has no cell
            # carrying both.
            assert not (abs(vy) > 1e-9 and abs(wz) > 1e-9), (vy, wz)
            seen_vy |= abs(vy) > 1e-9
            seen_wz |= abs(wz) > 1e-9
    assert seen_vy and seen_wz, "both channels must actually be exercised"


def test_the_default_ranges_are_off_so_a_caller_must_opt_in():
    from motion_gen import command_schedule as mtc

    for _f, _vx, vy, wz in mtc.terrain_interaction_schedule(
            np.random.default_rng(2), (0.0, float("inf")), (0.2, 2.6), 0.5, 3, 1.0):
        assert vy == 0.0 and wz == 0.0


def test_the_schedule_lands_ONLY_on_the_replay_span():
    """vy/wz belong where the pose is world-pinned and nowhere else -- off the replay they would
    describe motion the clip does not contain."""
    import mb_terrain_clips as mtc          # the APPLIER stays torch-side; only the sampler moved

    track = torch.zeros(100)
    track[:] = 1.5
    sched = [(0.5, 0.7, 0.4, 0.0), (0.5, 0.9, 0.0, -0.6)]
    out = mtc._apply_interaction_schedule(track.clone(), (40, 80), sched)
    assert out.shape == (100, 3)
    assert torch.all(out[:40, 1:] == 0) and torch.all(out[80:, 1:] == 0)
    assert torch.all(out[:40, 0] == 1.5) and torch.all(out[80:, 0] == 1.5)
    assert out[40, 0] == pytest.approx(0.7) and out[40, 1] == pytest.approx(0.4)
    assert out[60, 0] == pytest.approx(0.9) and out[60, 2] == pytest.approx(-0.6)


def test_p_straight_is_drawn_per_clip_not_per_segment():
    """A per-segment coin still puts lateral on nearly every traversal, which is the whole problem.

    Measured on g1_mb_v6c, where `_lat()` drew per segment at n_seg=2: 88.5% of traversals carried
    lateral or yaw while the flat pieces carried none, making `P(on terrain | lateral commanded)`
    exactly 1.0000 inside a terrain clip. The draw has to be per clip so that a whole traversal is
    either straight or not.
    """
    import numpy as np
    from motion_gen import command_schedule as mtc

    n_straight = 0
    for seed in range(400):
        sched = mtc.terrain_interaction_schedule(
            np.random.default_rng(seed), (0.0, float("inf")), (0.2, 2.5), 0.5, n_seg=4, sign=1.0,
            vy_range=(-0.6, 0.6), wz_range=(-0.8, 0.8), p_straight=0.5)
        lat = [(abs(vy) > 1e-9 or abs(wz) > 1e-9) for _f, _vx, vy, wz in sched]
        # per CLIP: a traversal is either wholly straight, or free to carry lateral in any segment.
        if not any(lat):
            n_straight += 1
    # A per-SEGMENT coin at n_seg=4 would make an all-straight traversal (1/3)**4 = 1.2% likely.
    assert 0.40 < n_straight / 400 < 0.60, (
        f"{n_straight}/400 traversals straight; p_straight=0.5 must be a per-clip draw"
    )


def test_p_straight_extremes_are_reachable():
    """1.0 silences lateral/yaw entirely; 0.0 leaves the per-segment behaviour intact."""
    import numpy as np
    from motion_gen import command_schedule as mtc

    for seed in range(50):
        sched = mtc.terrain_interaction_schedule(
            np.random.default_rng(seed), (0.0, float("inf")), (0.2, 2.5), 0.5, n_seg=3, sign=1.0,
            vy_range=(-0.6, 0.6), wz_range=(-0.8, 0.8), p_straight=1.0)
        assert all(abs(vy) < 1e-9 and abs(wz) < 1e-9 for _f, _vx, vy, wz in sched)

    any_lat = False
    for seed in range(50):
        sched = mtc.terrain_interaction_schedule(
            np.random.default_rng(seed), (0.0, float("inf")), (0.2, 2.5), 0.5, n_seg=3, sign=1.0,
            vy_range=(-0.6, 0.6), wz_range=(-0.8, 0.8), p_straight=0.0)
        any_lat |= any(abs(vy) > 1e-9 or abs(wz) > 1e-9 for _f, _vx, vy, wz in sched)
    assert any_lat, "p_straight=0.0 must leave lateral/yaw reachable"


def test_a_zero_range_silences_the_channel():
    """How the TRANSITION families are made straight: their vy/wz ranges are zero in the recipe.

    No code branch does it -- `_lat()` already refuses a range whose magnitude is zero -- so the
    transitions and the pure stairs run the same function with different numbers.
    """
    import numpy as np
    from motion_gen import command_schedule as mtc

    for seed in range(80):
        sched = mtc.terrain_interaction_schedule(
            np.random.default_rng(seed), (0.0, float("inf")), (0.2, 2.5), 0.5, n_seg=3, sign=1.0,
            vy_range=(0.0, 0.0), wz_range=(0.0, 0.0), p_straight=0.0)
        assert all(abs(vy) < 1e-9 and abs(wz) < 1e-9 for _f, _vx, vy, wz in sched)
