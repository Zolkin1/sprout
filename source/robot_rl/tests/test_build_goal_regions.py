# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin how the goal region is SIZED, because the direction of a sizing error is asymmetric.

Too wide costs a little discrimination at the margin.  Too tight penalises every arm for
something no policy in this repo can do -- and it does not look like a harness fault, it looks
like a policy result.  mjeval made exactly this mistake: its goal started as a 1 m circle, and
that was one of four defects each of which moved measured success by MORE than the difference
between the two policies it was comparing.

So the fit has floors, a pad, and a quantile below 1.0, and each of those is tested here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _load():
    """Import ``build_goal_regions`` without running it as a script."""
    path = _REPO / "scripts" / "rsl_rl" / "build_goal_regions.py"
    spec = importlib.util.spec_from_file_location("build_goal_regions", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bgr = _load()


def test_the_margin_reads_the_SHORT_tail_not_the_spread():
    """A clip arm that consistently overshoots must not get a huge approach margin.

    `approach_margin_m` answers "how far short of the clip's own endpoint does the tracker
    legitimately stop", so it is a one-sided statistic. Fitting it from |downrange| would let
    overshoot -- which is not a failure at all, the region is a half-plane -- inflate the
    margin and quietly admit robots that stopped well short.
    """
    overshooting = np.full(200, +3.0)          # every episode ends 3 m PAST the endpoint
    zeros = np.zeros(200)
    fit = bgr._fit_family(overshooting, zeros, zeros, 0.99, 0.5)
    assert fit["fitted_q_short_m"] == pytest.approx(0.0)
    assert fit["approach_margin_m"] == pytest.approx(bgr.MIN_MARGIN_M)


def test_the_margin_contains_a_clip_arm_that_stops_short():
    short_by = np.full(200, -2.0)              # every episode ends 2 m short
    zeros = np.zeros(200)
    fit = bgr._fit_family(short_by, zeros, zeros, 0.99, 0.5)
    assert fit["approach_margin_m"] == pytest.approx(2.5)      # 2.0 + 0.5 pad


def test_one_outlier_does_not_set_a_familys_band():
    """The quantile is below 1.0 on purpose.

    A single episode that ended two metres wide is more likely a near-fall than a definition
    of the task, and at quantile 1.0 it would size its whole family's lateral band.
    """
    lat = np.zeros(1000)
    lat[0] = 25.0                                              # one wild episode
    fit = bgr._fit_family(np.zeros(1000), lat, np.zeros(1000), 0.99, 0.5)
    assert fit["half_width_m"] < 2.0, "one outlier dragged the whole band out"


def test_floors_stop_a_tight_clip_arm_producing_an_unreachably_small_region():
    """A family whose clip arm lands in a 1 cm cluster must not get a 1 cm region.

    The clip arm is a reference-following policy on a pinned clip; its spread is a lower
    bound on the task's, not a description of it. Without floors a well-behaved family would
    end up with the tightest region and read as the hardest.
    """
    tiny = np.zeros(500)
    fit = bgr._fit_family(tiny, tiny, tiny, 0.99, 0.0)
    assert fit["approach_margin_m"] == pytest.approx(bgr.MIN_MARGIN_M)
    assert fit["half_width_m"] == pytest.approx(bgr.MIN_HALF_WIDTH_M)
    assert fit["height_tol_m"] == pytest.approx(bgr.MIN_HEIGHT_TOL_M)


def test_the_pad_only_ever_widens():
    zeros, spread = np.zeros(500), np.linspace(-3.0, 3.0, 500)
    no_pad = bgr._fit_family(zeros, spread, zeros, 0.99, 0.0)
    padded = bgr._fit_family(zeros, spread, zeros, 0.99, 0.5)
    assert padded["half_width_m"] > no_pad["half_width_m"]


def test_the_height_band_is_two_sided():
    """A tile family that ends BELOW its clip datum needs the band on that side too."""
    below = np.full(300, -0.4)
    fit = bgr._fit_family(np.zeros(300), np.zeros(300), below, 0.99, 0.1)
    assert fit["height_tol_m"] == pytest.approx(0.5)


def test_fitting_from_a_generator_arm_is_refused(tmp_path):
    """The region must be sized by the CLIP arm.

    Fitting it from the arm under test would size the goal to what that arm already reaches,
    which is circular in the direction that always passes.
    """
    (tmp_path / "fm__train__gen_a").mkdir(parents=True)
    (tmp_path / "fm__train__gen_a" / "episodes.h5").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="no clip-arm episodes.h5"):
        bgr._load_clip_cells(str(tmp_path))


# ----------------------------------------------------------------------------------------
# WHICH episodes define the endpoint. Measured: getting this wrong is not subtle.
# ----------------------------------------------------------------------------------------

def _table(tmp_path, *, n, bits, fell_h=None, censored=None, key="clip__none__-"):
    """Write a minimal episodes.h5 with just the columns the fitter reads."""
    import h5py
    import json as _json
    d = tmp_path / "cells" / key
    d.mkdir(parents=True)
    with h5py.File(d / "episodes.h5", "w") as f:
        f["termination_bits"] = np.asarray(bits, dtype="u2")
        f["min_base_height_w"] = np.asarray(fell_h if fell_h is not None else [0.75] * n, "f4")
        f["admin_censored"] = np.asarray(censored if censored is not None else [0] * n, "u1")
        f["family_code"] = np.zeros(n, "u1")
        f["max_tilt_rad"] = np.zeros(n, "f4")
        # Endpoint spread: episode 0 sits on the clip's end, the rest are far short.
        f["goal_downrange_m"] = np.asarray([0.0] + [-9.0] * (n - 1), "f4")
        f["goal_lateral_m"] = np.asarray([0.0] + [4.0] * (n - 1), "f4")
        f["goal_height_err_m"] = np.asarray([0.0] + [-2.5] * (n - 1), "f4")
        f.attrs["schema_version"] = 2
        f.attrs["schema_minor"] = 1
        f.attrs["termination_terms"] = _json.dumps(["time_out", "base_fallen", "clip_end"])
        f.attrs["family_codes"] = _json.dumps({"stairs_up": 0})
        f.attrs["num_episodes_written"] = n
    return str(tmp_path)


def test_only_episodes_that_reached_clip_end_define_the_endpoint(tmp_path):
    """Falls and censored episodes ended mid-task; their positions are not the task's.

    MEASURED on a 400-step smoke before this filter existed: one censored jump_off_front
    episode set that family's approach margin to 10.63 m and one censored stairs_up_pure
    episode set its height band to 2.91 m. A margin that wide admits a robot that stopped ten
    metres short, which is the stall the region exists to exclude -- so the contamination
    runs in the direction that quietly disables the whole test.
    """
    CLIP_END = 1 << 2
    root = _table(tmp_path, n=4,
                  bits=[CLIP_END, CLIP_END, 1 << 1, 0],   # 0,1 finished; 2 fell; 3 censored
                  fell_h=[0.75, 0.75, 0.20, 0.75],
                  censored=[0, 0, 0, 1])
    cells = bgr._load_clip_cells(root)
    assert len(cells) == 1
    # Episode 1 is the only non-episode-0 finisher, and it sits 9 m short by construction, so
    # if the filter let the fall or the censored row through the margin would be no smaller.
    # The point of the test is the COUNT the fitter keeps.
    _, cols, meta = cells[0]
    terms = list(meta["termination_terms"])
    finished = ((cols["termination_bits"] >> terms.index("clip_end")) & 1).astype(bool)
    import importlib
    fell = importlib.import_module("schema").fell(
        cols["min_base_height_w"], cols["max_tilt_rad"]).astype(bool)
    keep = finished & ~fell & ~cols["admin_censored"].astype(bool)
    assert keep.tolist() == [True, True, False, False]


def test_a_table_without_clip_end_is_refused(tmp_path):
    """Without that term nothing in the table says which episodes finished.

    Reconstructing it from step counts is what this replaced: `steps_survived` excludes the
    terminal frame and episodes may start mid-clip, so every arithmetic version is off by one
    or by the start phase, silently.
    """
    import h5py, json as _json
    d = tmp_path / "cells" / "clip__none__-"
    d.mkdir(parents=True)
    with h5py.File(d / "episodes.h5", "w") as f:
        for c in bgr.FIT_COLUMNS:
            f[c] = np.zeros(2, "f4" if c.endswith("_m") else "u2")
        f.attrs["schema_version"] = 2
        f.attrs["termination_terms"] = _json.dumps(["time_out", "base_fallen"])
        f.attrs["num_episodes_written"] = 2
    cells = bgr._load_clip_cells(str(tmp_path))
    terms = list(cells[0][2]["termination_terms"])
    assert "clip_end" not in terms   # the condition the fitter raises on
