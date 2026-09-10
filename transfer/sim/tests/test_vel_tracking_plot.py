# Copyright 2026 Zachary Olkin. All rights reserved.

"""The figure's episode selection must key on the terrain CODE, not on a run's list position.

``traces.npz``'s ``ep_terrain`` holds the GLOBAL :data:`vel_tracking_sweep.TERRAIN_CODES`
value. A run's own ``meta.terrains`` is whatever was asked for, so the two coincide only when
a run happens to request every terrain in code order -- which the four-terrain sweep does.
Keying on ``enumerate(meta.terrains)`` therefore works on exactly the run most likely to be
tried first and returns an EMPTY selection for a single-terrain run, which draws a blank
column and reads as a policy that never traversed rather than as a lookup bug. It cost a
wrong analysis table once; these tests are what stop it reaching the figure.

Everything here is sim-free: no mujoco, no onnxruntime, no matplotlib (``plot_run`` imports
pyplot lazily), so it runs anywhere.
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

vt = pytest.importorskip("vel_tracking_sweep")
pv = pytest.importorskip("plot_vel_tracking")


def fake_run(kinds: tuple[str, ...], speeds: tuple[float, ...], per_cell: int = 3,
             steps: int = 5) -> tuple[dict, dict]:
    """A minimal summary/traces pair, shaped exactly as the sweep writes them.

    Deliberately built from :data:`vel_tracking_sweep.TERRAIN_CODES` rather than from
    ``enumerate``, because that is the property under test: a stairs-only run's single terrain
    carries code 3 while sitting at index 0.
    """
    summary = {
        "meta": {"terrains": list(kinds), "speeds": [float(s) for s in speeds],
                 "terrain_codes": vt.TERRAIN_CODES,
                 "verdict_codes": {"success": 0, "fall": 1, "timeout": 2},
                 "trials": per_cell, "approach_m": 10.0, "runout_m": 10.0,
                 "cmd_hold_s": 0.5, "steer_kp": 1.5, "fm_onnx": "/fake/g_fm.onnx",
                 "tracker_onnx": "/fake/tracker.onnx"},
        "terrains": {k: {"profile_x": [-2.0, 10.0, 10.0, 25.0],
                         "profile_z": [0.0, 0.0, 0.4, 0.4],
                         "x_enter": 10.0, "x_exit": 10.5, "goal_x": 20.5,
                         "flight_height_m": 0.4, "floor_z": 0.0, "note": k, "params": {}}
                     for k in kinds},
        "cells": [],
    }
    rows, eps = [], []
    for k in kinds:
        for sp in speeds:
            for seed in range(per_cell):
                a = np.zeros((steps, len(vt.TRACE_COLUMNS)), dtype=np.float32)
                a[:, 0] = np.arange(steps) * 0.02                    # t
                a[:, 1] = np.arange(steps) * float(sp)               # x
                a[:, 3] = 0.79                                       # z
                a[:, 4] = float(sp)                                  # vx
                a[:, 7] = float(sp)                                  # cmd_vx
                rows.append(a)
                eps.append({"kind": k, "terrain": vt.TERRAIN_CODES[k], "speed": float(sp),
                            "seed": seed, "verdict": 0 if seed else 1, "failure_code": 0,
                            "steps": steps, "end_t": 0.08, "end_x": 1.0, "end_y": 0.0,
                            "end_z": 0.5})
    lens = np.array([r.shape[0] for r in rows], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = np.concatenate(rows, axis=0)
    traces = {n: flat[:, i] for i, n in enumerate(vt.TRACE_COLUMNS)}
    traces["ep_start"], traces["ep_len"] = starts, lens
    for key, dt in (("terrain", np.int64), ("speed", np.float64), ("seed", np.int64),
                    ("verdict", np.int64), ("failure_code", np.int64), ("end_t", np.float64),
                    ("end_x", np.float64), ("end_y", np.float64), ("end_z", np.float64)):
        traces[f"ep_{key}"] = np.array([e[key] for e in eps], dtype=dt)
    return summary, traces


def test_a_single_terrain_run_selects_its_episodes():
    """The regression: stairs_down sits at INDEX 0 of its run and carries CODE 3."""
    kinds = ("stairs_down",)
    assert vt.TERRAIN_CODES["stairs_down"] != 0, "the test needs a code that is not the index"
    s, tr = fake_run(kinds, (0.8, 1.5), per_cell=4)
    col = pv.Column("ft", "stairs_down", s, tr)
    for sp in (0.8, 1.5):
        idx = col.pick(sp)
        assert idx.size == 4, f"empty/short selection at {sp}: {idx}"
        assert set(tr["ep_seed"][idx].tolist()) == {0, 1, 2, 3}
        assert np.allclose(tr["ep_speed"][idx], sp)
    assert col.survived(col.pick(0.8)).size == 3      # seed 0 is the fall in the fixture


def test_selection_is_exact_when_several_terrains_share_a_run():
    """No column may pick up another terrain's episodes."""
    kinds = ("flat", "box", "stairs_up", "stairs_down")
    s, tr = fake_run(kinds, (0.8,), per_cell=2)
    seen = []
    for k in kinds:
        idx = pv.Column("", k, s, tr).pick(0.8)
        assert idx.size == 2, k
        assert set(tr["ep_terrain"][idx].tolist()) == {vt.TERRAIN_CODES[k]}
        seen.extend(idx.tolist())
    assert sorted(seen) == list(range(len(kinds) * 2)), "columns must partition the episodes"


def test_columns_are_terrain_major_so_compared_arms_are_adjacent():
    """Two runs' box columns must touch; grouping by run would separate them."""
    a = fake_run(("box", "stairs_down"), (0.8,))
    b = fake_run(("box", "stairs_down"), (0.8,))
    cols = pv.build_columns([("A", *a), ("B", *b)])
    assert [(c.kind, c.label) for c in cols] == [
        ("box", "A"), ("box", "B"), ("stairs_down", "A"), ("stairs_down", "B")]


def test_a_run_missing_the_requested_terrain_is_skipped_not_faked():
    """A run without the terrain contributes no column; one nobody has is refused."""
    a = fake_run(("box",), (0.8,))
    b = fake_run(("box", "stairs_down"), (0.8,))
    cols = pv.build_columns([("A", *a), ("B", *b)], ("stairs_down",))
    assert [(c.kind, c.label) for c in cols] == [("stairs_down", "B")]
    with pytest.raises(SystemExit, match="no run offers terrain"):
        pv.build_columns([("A", *a)], ("stairs_up",))


def test_a_run_spec_can_restrict_which_terrains_it_contributes():
    """"Parent on every terrain, with the IN-DISTRIBUTION box" needs two runs, split by hand.

    A run holds one geometry per terrain, so the box column has to come from the sweep that
    ran the trained depth while flat and the stairs come from the sweep that ran them. A
    single --terrain list cannot express that: it applies to every run and would pull in the
    other box as a fifth column.
    """
    wide = fake_run(("flat", "box", "stairs_up", "stairs_down"), (0.8,))
    narrow = fake_run(("box",), (0.8,))
    cols = pv.build_columns(
        [("parent", *wide, ("flat", "stairs_up", "stairs_down")), ("trained box", *narrow, ())],
        ("flat", "box", "stairs_up", "stairs_down"))
    assert [(c.kind, c.label) for c in cols] == [
        ("flat", "parent"), ("box", "trained box"),
        ("stairs_up", "parent"), ("stairs_down", "parent")]


@pytest.mark.parametrize("spec,want", [
    ("logs/v1", ("logs/v1", ())),
    ("logs/v1:box", ("logs/v1", ("box",))),
    ("logs/v1:flat,stairs_up", ("logs/v1", ("flat", "stairs_up"))),
    # Not a selection: the suffix is not a terrain, so the whole token stays a path.
    ("logs/v1:2026", ("logs/v1:2026", ())),
    ("/abs/path:with:colons", ("/abs/path:with:colons", ())),
])
def test_run_spec_parsing(spec, want):
    """A colon is only a selector when every part after it names a real terrain."""
    assert pv.parse_run_spec(spec) == want


def test_resample_is_nan_outside_an_episode_rather_than_held():
    """Holding the last value would enter a fallen robot's final speed into the median."""
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([10.0, 11.0, 12.0])
    grid = np.array([-0.5, 0.0, 1.0, 2.0, 2.5])
    out = pv.resample(xs, ys, grid)
    assert np.isnan(out[0]) and np.isnan(out[-1])
    np.testing.assert_allclose(out[1:4], [10.0, 11.0, 12.0])


def test_band_blanks_grid_points_no_episode_covers():
    """Past the shortest episode the band must be absent, not extrapolated."""
    grid = np.linspace(0.0, 3.0, 4)
    a = pv.resample(np.array([0.0, 1.0]), np.array([1.0, 1.0]), grid)
    b = pv.resample(np.array([0.0, 3.0]), np.array([2.0, 2.0]), grid)
    med, p10, p90, n = pv.band([a, b])
    assert n[0] == 2 and n[-1] == 1
    assert np.isfinite(med).all()
    np.testing.assert_allclose(med[-1], 2.0)
    med2, *_ = pv.band([np.full(4, np.nan)])
    assert np.isnan(med2).all()


def test_trial_alpha_keeps_total_ink_bounded():
    """A fixed alpha is a ghost at n=2 and a solid fill at n=50."""
    assert pv.trial_alpha(2) > pv.trial_alpha(50)
    assert 0.03 <= pv.trial_alpha(500) <= pv.trial_alpha(1) <= 0.25
    assert pv.trial_alpha(0) == pv.trial_alpha(1)     # no divide-by-zero on an empty cell
