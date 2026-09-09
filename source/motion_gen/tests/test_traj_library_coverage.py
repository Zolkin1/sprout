# Copyright 2026 Zachary Olkin. All rights reserved.

"""The solved-library coverage report, and the numbers it currently produces.

Pinned so a library refresh that changes coverage fails loudly rather than quietly shifting what the
clip recipe is asking for. The headline numbers below are measured against
``hf/trajectories/skills/library_v2``; when that library is replaced they SHOULD change, and this
test is the place that says so out loud.

The script is stdlib-only on purpose -- it runs under a bare ``python3`` against a solved library
that has not been ingested yet, which is exactly when its answer is worth having -- so this test
imports it without numpy or torch in the way.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "scripts" / "rsl_rl" / "traj_library_coverage.py").is_file():
        _REPO = _p
        break
else:                                                          # pragma: no cover - layout guard
    raise RuntimeError("could not locate the checkout above this test")
sys.path.insert(0, str(_REPO / "scripts" / "rsl_rl"))

import traj_library_coverage as tlc  # noqa: E402

_LIB = _REPO / "hf" / "trajectories" / "skills" / "library_v2"
_needs_lib = pytest.mark.skipif(not (_LIB / "manifest.csv").is_file(),
                                reason="the solved library is not in the local hf/ cache")


def test_the_converged_markers_match_the_ingest_side():
    """Duplicated for the bare-python3 property, so the two must be pinned equal.

    ``clean_ingest`` is the authority; this script cannot import it without dragging in numpy.
    """
    sys.path.insert(0, str(_REPO / "source" / "motion_gen"))
    from motion_gen.ingest.clean_ingest import _CONVERGED_MARKERS

    assert tuple(tlc.CONVERGED_MARKERS) == tuple(_CONVERGED_MARKERS)


def test_converged_reads_the_solver_status_and_not_its_absence():
    assert tlc.converged("Algorithm terminated successfully at a locally optimal point")
    assert tlc.converged("  OPTIMAL SOLUTION FOUND  ")
    assert not tlc.converged("Iteration limit reached")
    assert not tlc.converged("")
    assert not tlc.converged(None)


def test_a_blank_column_is_None_and_never_zero():
    """``vel_x`` is blank on essentially every box row. Reading that as 0 m/s is what made the box
    reference pick degenerate to 'take the first'."""
    assert tlc._f({"vel_x": ""}, "vel_x") is None
    assert tlc._f({"vel_x": "  "}, "vel_x") is None
    assert tlc._f({}, "vel_x") is None
    assert tlc._f({"vel_x": "0"}, "vel_x") == 0.0          # an explicit zero IS zero


@_needs_lib
def test_library_v2_coverage_headline_numbers():
    """The seven numbers a refresh must not change silently."""
    rows = tlc.read_manifest(str(_LIB))
    s = tlc.summarise(rows)
    assert len(rows) == 302
    assert sum(d["ok"] for d in s.values()) == 242            # 20% of rows did not converge
    assert (s["run_turn"]["ok"], s["run_turn"]["n"]) == (14, 42)      # the worst, 67% failed
    assert (s["stairs_down"]["ok"], s["stairs_down"]["n"]) == (2, 8)  # two converged descents
    assert (s["jump_off_front"]["ok"], s["jump_off_front"]["n"]) == (3, 7)
    assert (s["walk_forward"]["ok"], s["walk_forward"]["n"]) == (41, 42)
    assert (s["jog_forward"]["ok"], s["jog_forward"]["n"]) == (4, 5)


@_needs_lib
def test_only_CONVERGED_rows_contribute_cells():
    """A non-converged solve must not appear as coverage -- that is the whole point of reading the
    status column, which nothing in the ingest chain used to do."""
    rows = tlc.read_manifest(str(_LIB))
    full = tlc.summarise(rows)
    only_ok = tlc.summarise([r for r in rows if tlc.converged(r.get("status", ""))])
    for sk in full:
        assert sorted(set(full[sk]["vx"])) == sorted(set(only_ok[sk]["vx"])), sk


def test_the_recipe_parser_refuses_a_pre_v6_recipe(tmp_path):
    """Defaulting a missing envelope would make the report describe a recipe nobody ran."""
    p = tmp_path / "old.yaml"
    p.write_text("flat_walk_vx_range: [0.1, 1.0]\n")
    with pytest.raises(SystemExit):
        tlc.read_recipe_ranges(str(p))


def test_the_recipe_parser_reads_all_three_envelopes():
    ranges = tlc.read_recipe_ranges(str(_REPO / "configs" / "clip_gen" / "g1_mb_v6.yaml"))
    assert ranges["flat_vx_range"] == (-1.0, 2.5)
    assert ranges["flat_vy_range"] == (-0.6, 0.6)
    assert ranges["flat_wz_range"] == (-0.8, 0.8)


def test_nearest_cell_normalises_per_channel():
    """An unweighted metric trades 1 rad/s of yaw against 1 m/s of speed and picks the wrong cell.

    Same shape of metric as ``flat_plan.nearest_stride``, so the report and the generator agree about
    which cell a command selects.
    """
    cells = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.8)]
    # yaw spans 0.8 and vx spans 1.0, so a 0.5 rad/s yaw is a LARGER fraction of its axis than
    # 0.5 m/s of vx: the yaw cell must win.
    got = tlc.nearest_cell(cells, (0.5, 0.0, 0.5), scale=(1.0, 1.0, 0.8))
    assert got == (0.0, 0.0, 0.8)
