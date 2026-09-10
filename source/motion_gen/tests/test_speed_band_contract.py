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
from motion_gen.terrain import box2, step_courses, varied_courses

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SCRIPTS = _REPO / "scripts" / "rsl_rl"

#: Every builder, and the ``generator_config`` block it writes its course's constants under.
BUILDERS: dict[str, str] = {
    "make_box2_terrain.py": "box2",
    "make_step_courses_terrain.py": "step_course",
    "make_varied_terrain.py": "varied_courses",
    "make_indist_routes.py": "indist_routes",
    "make_freeform_terrain.py": "freeform",
}

#: The band each course declares, as a constant, so a builder and a test cannot drift apart
#: by restating a number.
BANDS: dict[str, tuple[float, float]] = {
    "box2": box2.COMMANDED_SPEED_RANGE_MPS,
    "curb": step_courses.CURB_SPEED_RANGE_MPS,
    "updown": step_courses.UPDOWN_SPEED_RANGE_MPS,
    "varied": varied_courses.COMMANDED_SPEED_RANGE_MPS,
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


def test_the_curb_and_box2_share_a_band_so_a_cell_is_the_same_COMMAND() -> None:
    """The curb asks "how low before it stops being a box", against box2.

    A different command band would confound that with a speed difference. The constants are
    deliberately restated rather than imported (so an edit to one cannot move the other
    library's provenance), which is exactly why they need pinning equal.
    """
    assert step_courses.CURB_SPEED_RANGE_MPS == box2.COMMANDED_SPEED_RANGE_MPS


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
        or f.stem in ("freeform", "box2", "step_courses", "varied_courses"))
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
_LIBS = ("box2_v2spawn_train.h5", "box2_v2spawn_holdout.h5", "curb_train.h5",
         "curb_holdout.h5", "updown_p020_r030_train.h5", "updown_p020_r030_holdout.h5",
         "varied60_train.h5", "varied60_holdout.h5", "indist_train.h5")


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
