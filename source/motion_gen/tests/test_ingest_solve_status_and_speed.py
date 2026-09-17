# Copyright 2026 Zachary Olkin. All rights reserved.

"""Two things the ingest threw away: whether a solve converged, and how fast a box reference moves.

Both were silent, and both are the same shape of bug -- information that exists in the source and
is dropped on the way in, so every consumer downstream reasons from a value that was never checked.

1. NOTHING read the manifest's ``status`` column. Measured on ``library_v2``, 60 of 302 rows
   (19.9%) did not converge, concentrated rather than spread: ``stairs_down`` 6/8, ``run_turn``
   28/42, ``jump_off_front`` 4/7. All were ingested identically to converged ones, and the stair
   TILE grid -- the geometry the RL and the depth cameras actually see -- was built from all of
   them.

2. ``steady_vel`` came from manifest columns that are BLANK on essentially every box row, and the
   parse failure became ``None``. ``select_terrain_reference`` scores a source with no recorded
   command as 0 m/s, so its "then nearest speed" stage collapsed to "take the first".
"""

from __future__ import annotations

import copy

import pytest
import torch

from motion_gen.ingest.clean_ingest import solve_converged, summarize_convergence
from motion_gen.terrain.terrain_place import select_terrain_reference


class _Ref:
    """Minimal stand-in for a MotionSource, for the selector."""

    def __init__(self, box_height, speed):
        self.terrain_meta = {"terrain": {"box_height": box_height}}
        self.steady_vel = None if speed is None else torch.tensor([speed, 0.0, 0.0])


@pytest.mark.parametrize("status,want", [
    ("Algorithm terminated successfully at a locally optimal point, ...", True),
    ("Optimal Solution Found.", True),
    ("Maximum number of iterations exceeded (can be specified by an option).", False),
    ("Restoration phase failed.", False),
    ("", True),          # older manifests carry no column; unknown must not read as failure
    (None, True),
])
def test_solve_converged_reads_the_status_column(status, want):
    assert solve_converged({"status": status} if status is not None else {}) is want


def test_summarize_convergence_counts_per_skill():
    rows = [{"skill": "a", "status": "locally optimal"},
            {"skill": "a", "status": "Restoration phase failed."},
            {"skill": "b", "status": "locally optimal"}]
    assert summarize_convergence(rows) == {"a": (1, 2), "b": (1, 1)}


def test_an_unknown_status_is_not_treated_as_a_failure():
    """The asymmetry is deliberate and matches the fingerprint rule elsewhere in the repo: a
    MISSING signal is not evidence of a problem, because treating it as one turns every older
    artifact into a failure and the check into a migration gate."""
    assert solve_converged({}) is True
    assert solve_converged({"status": "   "}) is True


def test_box_speed_selection_is_dead_when_the_command_is_None():
    """The regression this guards: with no recorded command every candidate scores 0 m/s, so the
    selector returns the same reference whatever is asked for."""
    refs = [_Ref(0.45, None) for _ in range(3)]
    picked = {select_terrain_reference(refs, box_height=0.45, speed=w) for w in (0.2, 0.9, 1.5, 3.0)}
    assert picked == {0}, "expected the degenerate behaviour this test documents"


def test_box_speed_selection_discriminates_once_the_command_is_known():
    """...and works as soon as the speed is present, which is what the ingest fallback restores.

    This is the case a solve delivering several approach speeds per box height would hit. Without
    the fallback that solve would have bought nothing: the extra references would have been
    ingested, scored 0 m/s alongside the originals, and never selected.
    """
    refs = [_Ref(0.45, s) for s in (0.3, 1.0, 1.8)]
    got = [float(refs[select_terrain_reference(refs, box_height=0.45, speed=w)].steady_vel[0])
           for w in (0.2, 0.9, 1.5, 3.0)]
    # approx, not ==: steady_vel is float32, so 0.3 round-trips to 0.30000001192092896
    assert got == pytest.approx([0.3, 1.0, 1.8, 1.8], abs=1e-6), got


def test_geometry_still_dominates_speed():
    """Two-stage selection: nearest GEOMETRY first, and only then nearest speed within it.

    Feet only land on a box the reference was solved for, so a perfect speed match at the wrong
    height must lose to a poor speed match at the right one.
    """
    refs = [_Ref(0.25, 1.5), _Ref(0.65, 0.3)]
    i = select_terrain_reference(refs, box_height=0.65, speed=1.5)
    assert float(refs[i].terrain_meta["terrain"]["box_height"]) == 0.65
