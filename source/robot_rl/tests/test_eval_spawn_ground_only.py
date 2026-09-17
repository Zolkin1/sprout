# Copyright 2026 Zachary Olkin. All rights reserved.
"""The eval spawns from the INITIAL area only, and it is the same rule on every terrain.

Training declares raised spawn areas -- a box top, the platform between two flights -- because
a complete traverse is rare enough that the behaviour policy never produced one, and AWR can
only reweight what it is shown (of 352 successful two-box episodes, every one reached the goal
by dismounting).  **Scoring from those areas would report a fraction of the course as a
traverse**: a robot placed on the second box has been handed the jump the course exists to
measure.  So every eval draws ``spawn_from="ground"``, on every terrain, and the asymmetry is
declared rather than emergent.

The failures this guards are all silent.  A filter that dropped the wrong areas still spawns a
robot somewhere legal and still produces a plausible rate; a filter left out of the table's
cache key serves whichever arm ran first, at exactly the same shape; and a ground area authored
at the wrong extents changes the spawn distribution without changing anything a test would
otherwise look at (box2 paid that one at 4x the intended lateral spread).
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from motion_gen.terrain import box2, freeform, step_courses, varied_courses
from robot_rl.tasks.manager_based.robot_rl.mdp.events import resets

#: <repo>/source/robot_rl/tests/<this> -> parents[3] == <repo>. Derived rather than counting
#: ``..`` hops, per the repo's own rule about paths that break silently when a file moves.
_REPO = pathlib.Path(__file__).resolve().parents[3]

#: Every fixed and varied family, as ``name -> route block``.  Built from the REAL generators:
#: a hand-written route would pin this test to a copy of the terrain rather than to the terrain.
def _routes() -> dict[str, dict]:
    """One route block per shipped family, keyed by a readable name."""
    cfg = varied_courses.VariedCoursesCfg()
    return {
        "box2": box2.make_box2_tile(0, 2.5, family=box2.BOX2_V2_FAMILY)["route"],
        "updown": step_courses.make_updown_tile(0, 2.5, 1.5)["route"],
        "varied_boxes": varied_courses.make_boxes_tile(0, cfg)["route"],
        "varied_updown": varied_courses.make_updown_tile(0, cfg)["route"],
    }


class _StubImporter:
    """The two attributes ``_freeform_spawn_table`` reads, and nothing else."""

    def __init__(self, routes: dict[int, dict], n_envs: int = 8) -> None:
        """Put every env on tile 0 unless the route map says otherwise."""
        self.terrain_route = routes
        ids = torch.arange(n_envs, dtype=torch.long) % max(len(routes), 1)
        self.env_terrain_id = ids


_JITTER = freeform.FREEFORM_SPAWN_JITTER_M
_TOL = resets.SPAWN_Z_TIE_TOL_M


@pytest.mark.parametrize("name", sorted(_routes()))
def test_ground_only_keeps_exactly_the_ground_areas(name: str) -> None:
    """The filter is ``z <= tol``, and it must leave every ground area and no raised one."""
    route = _routes()[name]
    areas = route.get("spawn_areas") or []
    if not areas:
        pytest.skip(f"{name} declares no areas; the synthesised start is already ground level")
    want = [a for a in areas if float(a["z"]) <= _TOL]
    assert want, f"{name} declares no ground-level area at all"
    imp = _StubImporter({0: route})
    table, index, lo = resets._freeform_spawn_table(imp, _JITTER, "ground")
    assert int(index[0, 1]) == len(want), (
        f"{name}: ground-only kept {int(index[0, 1])} of {len(areas)} areas, expected "
        f"{len(want)}")
    assert torch.all(table[:, 2] <= _TOL), f"{name}: a raised area survived: {table[:, 2]}"


@pytest.mark.parametrize("name", sorted(_routes()))
def test_the_ground_area_is_authored_at_the_ordinary_jitter_extents(name: str) -> None:
    """THIS is what makes the ground-only eval comparable to every number already measured.

    Declared areas REPLACE ``pos_jitter_m``, so a ground area sized to anything else silently
    changes the spawn distribution -- and the frozen eval grids were recorded on libraries that
    declare no areas at all and therefore got the synthesised area at exactly these extents.
    """
    areas = _routes()[name].get("spawn_areas") or []
    ground = [a for a in areas if float(a["z"]) <= _TOL]
    for a in ground:
        # varied's ground CHAIN tiles one continuous approach band with n abutting intervals,
        # so its downrange half-extent is the interval's, not the jitter's; the LATERAL one is
        # still the shared constant, which is the axis the 4x bug was on.
        assert abs(float(a["half_y"]) - _JITTER[1]) < 1e-9, (
            f"{name}: ground area half_y {a['half_y']} is not "
            f"freeform.FREEFORM_SPAWN_JITTER_M[1] = {_JITTER[1]}")
        if name.startswith("varied"):
            continue
        assert abs(float(a["half_x"]) - _JITTER[0]) < 1e-9, (
            f"{name}: ground area half_x {a['half_x']} is not "
            f"freeform.FREEFORM_SPAWN_JITTER_M[0] = {_JITTER[0]}")


def test_the_varied_ground_chain_covers_the_declared_approach_band() -> None:
    """Ground-only leaves several areas on the varied courses, and it should.

    They are abutting intervals tiling ``approach_range_m``, each weighted ``weight/n``, so the
    chain IS one continuous approach-distance band -- the analogue of box2's per-tile distance
    grid, which varied has no other axis for. Ground-only there means "anywhere on the
    approach", not "part-way through the course".
    """
    cfg = varied_courses.VariedCoursesCfg()
    route = varied_courses.make_boxes_tile(0, cfg)["route"]
    ground = [a for a in (route.get("spawn_areas") or []) if float(a["z"]) <= _TOL]
    assert len(ground) > 1, "the varied ground chain collapsed to a single area"
    lo, hi = (float(v) for v in cfg.approach_range_m)
    span = sum(2.0 * float(a["half_x"]) for a in ground)
    assert abs(span - (hi - lo)) < 1e-6, (
        f"the chain spans {span:.4f} m against a declared band of {hi - lo:.4f} m")
    xs = sorted(float(a["x"]) for a in ground)
    for a, b in zip(xs, xs[1:]):
        assert b > a, "two chain intervals share a centre"


@pytest.mark.parametrize("dist", [1.5, 3.0])
def test_ground_only_reproduces_the_PRE_AREAS_spawn_EXACTLY(dist: float) -> None:
    """The comparability claim, checked rather than argued.

    Every box2 number measured to date was collected on a library written BEFORE spawn areas
    existed, which declares none and gets the reset's synthesised single start. If the
    ground-only draw on a library that DOES declare areas is the same rectangle, those numbers
    stay comparable and the unreproducible pre-areas library can be retired; if it is not,
    every one of them is against a different initial condition.

    Measured on the real thing: the two tables are bit-identical, ``[1.25, 3.0, 0.0, 0.25,
    0.25, 0.0, 1.0]`` on both tiles of the quick grid.
    """
    with_areas = box2.make_box2_tile(0, dist, family=box2.BOX2_V2_FAMILY)["route"]
    # The same route with its areas stripped -- i.e. what the pre-areas generator emitted.
    pre_areas = {k: v for k, v in with_areas.items() if k != "spawn_areas"}
    a = resets._freeform_spawn_table(_StubImporter({0: pre_areas}), _JITTER, "all")[0]
    b = resets._freeform_spawn_table(_StubImporter({0: with_areas}), _JITTER, "ground")[0]
    assert torch.equal(a, b), (
        f"ground-only gives {b.tolist()} where the pre-areas library spawned at {a.tolist()}; "
        "every box2 rate measured before spawn areas would be against a different IC")


def test_the_filter_is_part_of_the_tables_cache_key() -> None:
    """The table is built once and cached ON the importer.

    Without the filter in the key, a process that ran the eval arm after the training arm
    would be served the training table -- silently, and at exactly the same shape, which is
    the one failure mode a spawn distribution has no other symptom for.
    """
    route = _routes()["box2"]
    imp = _StubImporter({0: route})
    all_t, all_i, _ = resets._freeform_spawn_table(imp, _JITTER, "all")
    n_all = int(all_i[0, 1])
    gnd_t, gnd_i, _ = resets._freeform_spawn_table(imp, _JITTER, "ground")
    assert int(gnd_i[0, 1]) < n_all, (
        f"the second call was served the cached 'all' table: {int(gnd_i[0, 1])} == {n_all}")
    # And back again, so the key discriminates in both directions rather than merely rebuilding.
    again, again_i, _ = resets._freeform_spawn_table(imp, _JITTER, "all")
    assert int(again_i[0, 1]) == n_all


def test_a_DESCENDING_tile_keeps_its_raised_start_rather_than_being_refused() -> None:
    """A descent has no ground start, and the eval rule must not demand one.

    This test asserted the OPPOSITE until 2026-09-10: the filter tested ``z <= tol`` against
    absolute zero, so a tile whose only declared start is raised was refused outright, on the
    stated grounds that "every shipped family declares a ground area". That was false. The
    in-distribution routes are built from real clip-library tiles, and two of their four
    families -- ``stairs_down`` and ``jump_off_front`` -- spawn on the platform because you
    cannot begin a descent from the floor. The refusal killed all three in-distribution evals
    at env construction, before a single episode was collected.

    The rule is now the ROUTE'S OWN ``start_z``, which expresses what the eval actually wants
    -- start where the course starts, never part-way along it -- without asserting that every
    course starts on the floor, and without assuming that "where the course starts" is the
    tile's lowest point. It is not on a course that DESCENDS.
    """
    # `start_z` matches the declared area, which is what every real library does: measured
    # over 536 routed tiles on disk, 526 have `min(area z) == route.start_z` and the only
    # exceptions were the descending entry stairs this rule was fixed for.
    raised = freeform._route((1.0, 0.0), 0.4, 0.0, (6.0, 0.0), 1.0, 0.0,
                             spawn_areas=[freeform.spawn_area(3.0, 0.0, 0.4, 0.3, 0.3)])
    imp = _StubImporter({0: raised})
    table, index, lo = resets._freeform_spawn_table(imp, _JITTER, "ground")
    assert int(index[0, 1]) == 1, "the single declared start must survive the eval filter"
    assert table[0, 2] == pytest.approx(0.4), (
        "the platform start must be served AS AUTHORED; falling back to the synthesised "
        "ground start would place the robot at a pose the terrain never declared legal")


def test_ground_only_keeps_the_STARTING_tier_when_a_tile_has_several() -> None:
    """The discriminating case: a tile with a starting platform AND a higher landing.

    Keeping the start alone is the point. An absolute-ground test keeps nothing here (the
    2026-09-10 bug) and a "keep everything" fallback keeps both -- which would score a
    fraction of the course as a traverse, the very thing the eval rule exists to prevent.
    """
    tiered = freeform._route(
        (1.0, 0.0), 0.40, 0.0, (6.0, 0.0), 1.0, 0.0,
        spawn_areas=[freeform.spawn_area(3.0, 0.0, 0.40, 0.3, 0.3),
                     freeform.spawn_area(4.0, 0.0, 0.95, 0.3, 0.3)])
    imp = _StubImporter({0: tiered})
    table, index, lo = resets._freeform_spawn_table(imp, _JITTER, "ground")
    assert int(index[0, 1]) == 1
    assert table[0, 2] == pytest.approx(0.40)


def test_a_DESCENDING_tile_keeps_its_START_not_the_tiles_lowest_area() -> None:
    """THE case the minimum-z rule got wrong, and it cost half of every descending eval.

    ``entry_stairs_down`` starts in the lobby at 1.162 m and declares a mid-course area on the
    landing at 0.762. The minimum picked the LANDING, so every eval spawned PAST the first
    flight -- 5.89 m of a declared 11.82 m course -- while the cells filled and the rates
    looked entirely normal. There is no other symptom: a spawn distribution that is wrong in
    this way produces a complete, plausible table.
    """
    descending = freeform._route(
        (4.0, 0.0), 1.162, 0.0, (15.8, 0.0), 0.75, 0.0,
        spawn_areas=[freeform.spawn_area(4.0, 0.0, 1.162, 0.25, 0.25),   # the lobby: START
                     freeform.spawn_area(9.9, 0.0, 0.762, 0.9, 2.2)])    # the landing: mid
    imp = _StubImporter({0: descending})
    table, index, lo = resets._freeform_spawn_table(imp, _JITTER, "ground")
    assert int(index[0, 1]) == 1, "both tiers survived; the eval starts part-way along"
    assert table[0, 2] == pytest.approx(1.162), (
        f"the eval spawns at z={float(table[0, 2]):.3f}, the mid-course landing, rather than "
        "at the lobby the route declares as its start")


def test_areas_that_disagree_with_the_routes_own_start_are_refused() -> None:
    """A route and its areas must agree about where the course begins.

    Silently falling back to the lowest area is exactly what put every descending eval
    part-way along its course, so the disagreement is refused rather than resolved.
    """
    inconsistent = freeform._route(
        (1.0, 0.0), 0.0, 0.0, (6.0, 0.0), 1.0, 0.0,
        spawn_areas=[freeform.spawn_area(3.0, 0.0, 0.40, 0.3, 0.3)])
    with pytest.raises(ValueError, match="start_z"):
        resets._freeform_spawn_table(_StubImporter({0: inconsistent}), _JITTER, "ground")


def test_an_unknown_subset_is_refused_rather_than_treated_as_all() -> None:
    """A typo'd override must not quietly run the training distribution."""
    imp = _StubImporter({0: _routes()["box2"]})
    with pytest.raises(ValueError, match="spawn_from must be one of"):
        resets._freeform_spawn_table(imp, _JITTER, "gorund")


def test_every_draw_lands_on_the_ground_under_the_eval_rule() -> None:
    """The end-to-end property, over the weighted draw rather than over the table.

    box2 weights its three areas as thirds, so a draw that ignored the filter would put ~2/3
    of envs on a box top -- and every one of those episodes would score as a traverse.
    """
    torch.manual_seed(0)
    imp = _StubImporter({0: _routes()["box2"]}, n_envs=512)
    env_ids = torch.arange(512, dtype=torch.long)
    drawn = resets._draw_spawn_areas(imp, env_ids, _JITTER, "ground")
    # box2's lowest declared area IS the floor, so the bound here is still zero -- the
    # point of stating it as the minimum is that the assertion keeps holding on a course
    # whose lowest start is a platform.
    z_lo = min(float(a["z"]) for a in _routes()["box2"]["spawn_areas"])
    assert torch.all(drawn[:, 2] <= z_lo + _TOL), "an eval draw landed above the lowest start"
    assert z_lo == pytest.approx(0.0), "box2 should still start on the floor"
    # The control: with the filter off, the same draw must reach the box tops -- otherwise
    # this test would pass just as well against a broken weighted draw.
    imp2 = _StubImporter({0: _routes()["box2"]}, n_envs=512)
    assert torch.any(resets._draw_spawn_areas(imp2, env_ids, _JITTER, "all")[:, 2] > _TOL)


def test_training_is_UNCHANGED_by_default() -> None:
    """``all`` is the default at every layer, so no clip task or training run moves."""
    import inspect
    for fn in (resets._freeform_spawn_table, resets._draw_spawn_areas,
               resets.reset_on_freeform_section):
        assert inspect.signature(fn).parameters["spawn_from"].default == "all", (
            f"{fn.__name__} does not default to the training distribution")


def test_the_strict_tracking_bounds_are_a_TERMINATION_on_the_finetune_task() -> None:
    """The bounds must end the episode, not be applied to it afterwards.

    While they were an offline latch the pipeline was incoherent in a way that cost the whole
    success signal: the recording ran to the goal and banked the progress reward's terminal
    bonus, then every consumer cut the episode at the latch and deleted the row that bonus was
    paid on -- measured at 95.3% of arriving episodes. With the termination in place nothing
    downstream truncates anything, and ``reached_goal`` and
    ``rollout_outcomes.terminal_reward`` become the same question.
    """
    import re

    cfg = (_REPO / "source" / "robot_rl" / "robot_rl" / "tasks" / "manager_based"
           / "robot_rl" / "g1" / "g1_finetune_env_cfg.py").read_text()
    assert "mdp.fm_tracking_violation_termination" in cfg, (
        "the finetune task does not terminate on the strict tracking bounds")
    m = re.search(r"tracking_violation = DoneTerm\((.*?)\)", cfg, re.S)
    assert m, "the term is not where this test expects it"
    assert "time_out" not in m.group(1), (
        "a tracking violation is a FAILURE, not a truncation -- marking it time_out would "
        "make `terminal_reward` score it 0 rather than -1")


def test_the_height_bound_is_the_MEASURED_012_in_both_modules() -> None:
    """The numpy and torch specs are parity-tested, so a bound that moves in one only is a
    silent disagreement between the simulator and every offline consumer of its recordings."""
    import sys

    sys.path.insert(0, str(_REPO / "transfer"))
    from deploy.tracking_error import BOOTSTRAP_SPEC as NP_SPEC
    from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import (
        BOOTSTRAP_SPEC as T_SPEC,
    )
    assert NP_SPEC.max_height_m == T_SPEC.max_height_m == 0.12
    # Unchanged, so the height bound is the only thing this decision moved.
    assert NP_SPEC.max_lateral_m == T_SPEC.max_lateral_m == 0.2
    assert abs(NP_SPEC.max_tilt_rad - T_SPEC.max_tilt_rad) < 1e-12
    # Still un-fitted: `fit_tracking_thresholds.py` does not exist, so publishing consumers
    # must keep refusing this spec.
    assert NP_SPEC.is_bootstrap and T_SPEC.is_bootstrap


def test_the_spawn_HEIGHT_is_recorded_so_the_eval_rule_is_checkable_from_the_artifact() -> None:
    """The rule above is enforced in code; this is what lets a RECORDING confirm it.

    Without the column the only evidence that an eval drew from the course's lowest area was
    this test file plus the absence of a refusal -- and the verification plan asked for it in
    the artifact, because a setting that changes what a number MEANS has to travel with the
    number rather than in a launcher's banner.

    Three properties, each of which was a way to get this wrong:

    * the column EXISTS on the episode table, so `spawn_from` is auditable offline;
    * it is initialised to NaN, not 0.0. A course whose spawn is a platform records a
      positive height legitimately, so zero cannot double as "nobody recorded this" -- the
      same "absent is not zero" rule `penetration_volume_m3` is documented under;
    * it is latched BEFORE the goal-spec early return in ``RolloutEpisodeTracker._refresh_goal``, because the spawn
      height describes the RUN and not the scoring: a rollout with no goal spec attached must
      still record where the robot started.
    """
    import inspect

    import rollout_outcomes as ro
    from eval import accumulate as acc

    assert "spawn_base_z_w" in ro.ROLLOUT_COLUMN_NAMES, (
        "the episode table must carry the spawn height, or the ground-only rule is only ever "
        "confirmable from a banner")

    src = inspect.getsource(acc.EpisodeAccumulator.__init__)
    assert 'self.spawn_z = torch.full((self.n,), float("nan")' in src, (
        "spawn_z must start as NaN: 0.0 is a legitimate recorded value and cannot also mean "
        "'not recorded'")

    ref = inspect.getsource(ro.RolloutEpisodeTracker._refresh_goal)
    latch = ref.index("record_spawn_z")
    early = ref.index('_gr", None) is None')
    assert latch < early, (
        "record_spawn_z must be called BEFORE the goal-spec early return, or a rollout "
        "without a goal spec records no spawn height at all")
