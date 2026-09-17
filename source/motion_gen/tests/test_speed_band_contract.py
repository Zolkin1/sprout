# Copyright 2026 Zachary Olkin. All rights reserved.
"""Every fine-tune course declares its command band, and the READER can find it.

``run_finetune_loop.sh`` asks the library what band to sample and falls back to 0.5-2.2 when
it declares none.  That fallback is the hazard this file exists for, because it is invisible
from both ends: the library's ``generator_config`` says ``0.6-2.5`` in plain text, the reader
returns ``None`` because the block is not one it looks under, and the run trains on a band
nobody chose -- below the up-into-down course's own 0.6 floor, in the case that was live.

So the contract is two-sided and both sides are checked here:

1. every builder stamps ``commanded_speed_range_mps``, not merely an ``intended_speeds_mps``
   ladder (which nothing reads -- box2 and the curb both shipped one for weeks);
2. every block NAME it stamps under is in ``read_commanded_speed_range``'s search list.

A ladder is not a substitute for a band.  It records which rungs an EVAL pins, one recording
per rung; training samples the interval continuously per episode, and the two are different
statements about the same course.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from motion_gen.clips import clip_io
from motion_gen.terrain import box2, step_courses, varied_courses, wall_field

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts" / "rsl_rl"

#: Every builder, and the ``generator_config`` block it writes its course's constants under.
BUILDERS: dict[str, str] = {
    "make_box2_terrain.py": "box2",
    "make_step_courses_terrain.py": "step_course",
    "make_varied_terrain.py": "varied_courses",
    "make_indist_routes.py": "indist_routes",
    "make_freeform_terrain.py": "freeform",
    # "freeform", not "wall_field": this table names the block the BAND is in, and the reader
    # searches a fixed list that a new course's own block is not on. The wall field stamps
    # both -- its own block for provenance, "freeform" for the band -- and stamping only the
    # former made the band unreadable, so the fine-tune loop silently fell back to its own
    # default while the eval hard-refused.
    "make_walls_terrain.py": "freeform",
}

#: The band each course declares, as a constant, so a builder and a test cannot drift apart
#: by restating a number.
BANDS: dict[str, tuple[float, float]] = {
    "box2": box2.COMMANDED_SPEED_RANGE_MPS,
    "updown": step_courses.UPDOWN_SPEED_RANGE_MPS,
    "varied": varied_courses.COMMANDED_SPEED_RANGE_MPS,
    "walls": wall_field.COMMANDED_SPEED_RANGE_MPS,
}


@pytest.mark.parametrize("block", sorted(set(BUILDERS.values())))
def test_every_generator_block_is_one_the_reader_searches(block: str) -> None:
    """The reader's list IS the contract; a name missing from it is a silent no-op.

    Read out of the reader's source rather than duplicated here -- a copy of the list would
    pass while the real one was short.
    """
    src = (_REPO / "source" / "motion_gen" / "motion_gen" / "clips" / "clip_io.py").read_text()
    m = re.search(r'for block in \(([^)]*)\):', src)
    assert m, "read_commanded_speed_range's block list is not where this test expects it"
    searched = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert block in searched, (
        f"builders write generator_config['{block}'] but the reader only looks under "
        f"{sorted(searched)} -- the band would be declared and never read")


@pytest.mark.parametrize("name", sorted(BANDS))
def test_every_band_is_a_positive_ordered_pair(name: str) -> None:
    """The reader raises on a reversed or zero band; a course must not be able to author one."""
    lo, hi = BANDS[name]
    assert 0.0 < lo < hi, f"{name} declares {(lo, hi)}, which samples a constant or worse"


def test_the_freeform_band_is_the_command_terms_own_default() -> None:
    """``motion_gen`` cannot import the cfg to read them, so the pair is pinned here.

    The library has to DECLARE its band rather than inherit whatever the env cfg says on the
    day it is recorded -- that is the whole reason any course stamps one -- but the numbers
    themselves came from the command term and must not drift from it.
    """
    src = (_REPO / "source" / "robot_rl" / "robot_rl" / "tasks" / "manager_based"
           / "robot_rl" / "mdp" / "commands" / "traj_tracking"
           / "fm_tracking_cmd_cfg.py").read_text()
    lo = re.search(r"^    freeform_speed_min: float = ([0-9.]+)", src, re.M)
    hi = re.search(r"^    freeform_speed_max: float = ([0-9.]+)", src, re.M)
    assert lo and hi, "the cfg's freeform speed defaults are not where this test expects them"
    from motion_gen.terrain import freeform
    assert freeform.COMMANDED_SPEED_RANGE_MPS == (float(lo.group(1)), float(hi.group(1))), (
        f"the free-form library declares {freeform.COMMANDED_SPEED_RANGE_MPS} against the "
        f"command term's ({lo.group(1)}, {hi.group(1)})")


@pytest.mark.parametrize("script,block", sorted(BUILDERS.items()))
def test_every_builder_stamps_the_band_key_and_not_only_a_ladder(script: str,
                                                                 block: str) -> None:
    """Parsed as text, the way the loop's own algorithm switch is pinned.

    Building a library here would need a multi-gigabyte ``--like`` for its channel names, so
    what is checkable without one is that the KEY is written at all -- which is the whole of
    the defect being guarded (box2 stamped a ladder and no band; the curb the same; the
    up-into-down holdout stamped neither).
    """
    path = _SCRIPTS / script
    if not path.is_file():
        pytest.skip(f"{script} not present")
    # The key may sit in the builder OR in the terrain module the builder delegates its
    # config block to -- `make_varied_terrain.py` calls `varied_courses.generator_config`,
    # so the string is legitimately not in the script. Both are searched; what is not
    # allowed is neither.
    terr = _REPO / "source" / "motion_gen" / "motion_gen" / "terrain"
    src = path.read_text() + "".join(
        f.read_text() for f in sorted(terr.glob("*.py")) if block.split("_")[0] in f.stem
        or f.stem in ("freeform", "box2", "step_courses", "varied_courses", "wall_field"))
    assert "commanded_speed_range_mps" in src, (
        f"{script} writes generator_config['{block}'] without a band; a loop pointed at its "
        "library falls back to a default nobody chose")


def test_the_updown_HOLDOUT_stamps_the_band_too() -> None:
    """It used to stamp nothing, on the reasoning that a holdout's command is pinned.

    Withholding it did not pin anything -- a recording pins the command -- it only made a
    runner fall back to 0.5-2.2, whose floor is below this course's own 0.6. The band
    describes the COURSE, and the holdout is the same course.
    """
    src = (_SCRIPTS / "make_step_courses_terrain.py").read_text()
    assert 'if role == "train" else None' not in src, (
        "the holdout is being written with speed_range=None again")
    assert 'speed_range=spec.get("speed_range")' in src


#: The libraries the fine-tune actually runs on.  Skipped where absent, since `clips/` is not
#: in the repo -- so this is a local/talos check rather than a CI one.
_LIBS = ("box2_v2spawn_train.h5", "box2_v2spawn_holdout.h5",
         "updown_p020_r030_train.h5", "updown_p020_r030_holdout.h5",
         "varied60_train.h5", "varied60_holdout.h5", "indist_train.h5",
         "ft_walls_train.h5", "ft_walls_holdout.h5")


@pytest.mark.parametrize("lib", _LIBS)
def test_the_library_on_disk_reads_back_a_band(lib: str) -> None:
    """The end-to-end property: what is stamped is what the reader returns."""
    path = _REPO / "clips" / lib
    if not path.is_file():
        pytest.skip(f"{lib} not built here")
    rng = clip_io.read_commanded_speed_range(str(path))
    assert rng is not None, (
        f"{lib} declares no readable band, so a run against it silently uses the loop's "
        "own fallback. Rebuild it with its builder.")
    lo, hi = rng
    assert 0.0 < lo < hi


#: The goal-geometry constants each fixed course RESTATES rather than imports, and the module
#: each copy lives in.
GOAL_CONSTANTS: tuple[str, ...] = ("TARGET_LEAD_M", "GOAL_DEPTH_M")


@pytest.mark.parametrize("name", GOAL_CONSTANTS)
def test_every_course_restates_the_same_goal_geometry(name: str) -> None:
    """Three copies, deliberately -- so they need a test rather than a hoist.

    Restating rather than importing is this package's own convention for a number that is
    mirrored from mjeval: ``step_courses`` says it explicitly of its speed ladder ("restated
    rather than imported so that an edit there cannot move this library's terrain
    fingerprint"), and the same argument holds here -- a course's provenance block should not
    move because a sibling course was edited.

    The cost of that convention is drift, and it was real: ``box2`` and ``step_courses`` are
    pinned to their mjeval twins by the parity tests in ``transfer/sim/tests``, but
    ``varied_courses`` mirrors nothing and its copies were unpinned, so a change to either of
    the others would have left the varied library authoring a goal at a different lead with
    nothing to notice. This is the missing third pin.
    """
    vals = {m.__name__.rsplit(".", 1)[-1]: getattr(m, name)
            for m in (box2, step_courses, varied_courses)}
    assert len(set(vals.values())) == 1, (
        f"{name} differs across the courses: {vals}. Every one of them maps mjeval's goal "
        "RECTANGLE onto the line-plus-band the RL termination tests, so a course whose lead "
        "or depth differs scores arrival at a different place while reporting the same "
        "column.")


# ---------------------------------------------------------------------------------------------
# The GOAL spec, which is the other half of the contract and used to be hand-kept.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("script", ["make_box2_terrain.py", "make_step_courses_terrain.py",
                                    "make_varied_terrain.py", "make_indist_routes.py",
                                    "make_walls_terrain.py"])
def test_every_builder_emits_its_goal_spec(script: str) -> None:
    """A course whose geometry regenerates but whose goal spec does not is half reproducible.

    Four of the five specs were hand-written, and `logs/` is gitignored -- so they do not
    travel with a git sync. That bit us when talos held a stale ``height_tol_m: 0`` after a
    pull, which is the value that cost an arm five iterations of mistraining; and
    `logs/indist/goal_regions.json` named as its source a script that wrote no spec at all.
    """
    p = _SCRIPTS / script
    if not p.is_file():
        pytest.skip(f"{script} not present")
    src = p.read_text()
    assert '"--goals"' in src, f"{script} cannot write a goal spec"
    assert "goal_regions.json" in src


@pytest.mark.parametrize("name,fn,kw", [
    ("box2", "goal_region_spec", {"variant": "v2"}),
    ("step_courses", "goal_region_spec", {}),
])
def test_the_declared_spec_never_disables_the_height_test(name: str, fn: str, kw: dict) -> None:
    """``height_tol_m`` must be a real band in every emitted spec.

    0 disables the height test, and on a course whose goal sits ABOVE the spawn that lets an
    arrival be scored from the floor beside the terrain -- which is the whole thing the
    tolerance exists to catch.
    """
    import importlib

    mod = importlib.import_module(f"motion_gen.terrain.{name}")
    spec = getattr(mod, fn)(**kw)
    d = spec["default"]
    assert d["height_tol_m"] > 0.0, f"{name} declares height_tol_m={d['height_tol_m']}"
    assert d["half_width_m"] > 0.0
    assert spec["provenance"]["declared"] is True


def test_the_updown_band_does_not_reach_past_its_walls() -> None:
    """mjeval's 2.5 m band would score an arrival BESIDE a 1.96 m corridor's flight.

    The course's own half-width is half the corridor, floored at what the RL termination
    clamps up to -- and the free-form stairs family's wider margin is the documented apron
    trap, where 0.65 m of ground-level apron beside the flight is a clean route AROUND it that
    scores as a success.
    """
    d = step_courses.goal_region_spec()["default"]
    assert d["half_width_m"] <= step_courses.UPDOWN_WIDTH_M / 2.0 + 1e-9
    assert d["half_width_m"] < 2.5
