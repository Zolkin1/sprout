# Copyright 2026 Zachary Olkin. All rights reserved.

"""The flat command is the DRIVER'S envelope, sampled continuously -- not the library's capability.

The motion is whichever solved stride is nearest; the command recorded is the continuous value it was
asked for. A generator only ever commanded speeds it had motions for cannot learn to SELECT, and
driving it then requires exactly those velocities, which no real driver produces. So the support is
deliberately wider than the solved library, and the snap error that follows is a designed
disagreement rather than a defect.

What is pinned here is that the sampler has no STRUCTURE of its own -- the previous one drew each
mode's forward speed from a different sub-band, and that showed up in the output as cliffs.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

for _p in pathlib.Path(__file__).resolve().parents:
    if (_p / "source" / "motion_gen" / "motion_gen").is_dir():
        _REPO = _p
        break
else:                                                          # pragma: no cover - layout guard
    raise RuntimeError("could not locate the checkout above this test")
sys.path.insert(0, str(_REPO / "source" / "motion_gen"))

from motion_gen.command_schedule import (  # noqa: E402
    StratifiedSpeeds, sample_command_schedule, total_duration)

ENV = dict(vx_range=(-1.0, 3.6), vy_range=(-0.6, 0.6), wz_range=(-0.8, 0.8),
           mass_stand=0.08, mass_sidestep=0.10)


def _draw(n_clips, quota, seed=7):
    rng = np.random.default_rng(seed)
    q = StratifiedSpeeds(rng, ENV["vx_range"], 0.2) if quota == "stratified" else None
    segs = []
    for _ in range(n_clips):
        segs += sample_command_schedule(rng, duration=20.0, seg_dur_range=(1.5, 4.0),
                                        vx_draw=q, **ENV)
    return segs


def _marginal(segs, width=0.2):
    """Time-weighted share per bin over the MOVING commands only."""
    lo, hi = ENV["vx_range"]
    edges = np.arange(lo, hi + 1e-9, width)
    d = np.array([s[0] for s in segs]); v = np.array([s[1] for s in segs])
    m = np.abs(v) > 1e-9                      # the vx == 0 mass is DECLARED, not sampled
    h, _ = np.histogram(v[m], bins=edges, weights=d[m])
    return h / h.sum()


def test_the_marginal_is_flat_at_the_real_build_size():
    """The done-when: max/min bin share <= 1.25, no empty bin, none below 60% of the mean.

    The predecessor measured 10.3 : 1 with 3.0-3.6 entirely empty. Checked at 3000 clips because
    that is the flat family's real size -- a 16-clip smoke build has ~60 independent draws against 23
    strata and cannot resolve this either way.
    """
    sh = _marginal(_draw(3000, "stratified"))
    nz = sh[sh > 0]
    assert (sh == 0).sum() == 0, "a bin with zero commanded time"
    assert sh.max() / nz.min() <= 1.25, sh.max() / nz.min()
    assert nz.min() / sh.mean() >= 0.60


def test_stratification_changes_the_REALISATION_not_the_distribution():
    """Both arms sample the same uniform; only the finite-sample ripple differs.

    Pinned because it is the whole justification for the machinery: at the flat family's real size it
    is worth 1.18 -> 1.04, and at a smoke build's size 9.4 -> 1.9.
    """
    strat, iid = _marginal(_draw(3000, "stratified")), _marginal(_draw(3000, "iid"))
    nzs, nzi = strat[strat > 0], iid[iid > 0]
    assert strat.max() / nzs.min() < iid.max() / nzi.min()
    # Same target: the two means agree to well inside their own spread.
    assert abs(strat.mean() - iid.mean()) < 1e-9
    small_s = _marginal(_draw(16, "stratified"))
    small_i = _marginal(_draw(16, "iid"))
    assert (small_s == 0).sum() <= (small_i == 0).sum()


def test_nothing_quantizes_the_command():
    """A draw is uniform WITHIN its stratum, so every command is a distinct real."""
    v = [s[1] for s in _draw(200, "stratified") if abs(s[1]) > 1e-9]
    assert len(set(np.round(v, 9))) > 0.99 * len(v), "commands are landing on a lattice"


def test_at_most_one_of_vy_and_wz():
    """The user's constraint, and it costs nothing: no solved cell carries both."""
    for _d, _vx, vy, wz in _draw(500, "stratified"):
        assert not (abs(vy) > 1e-9 and abs(wz) > 1e-9), (vy, wz)


def test_the_zero_masses_are_DECLARED_not_emergent():
    """They used to be a side effect: the old sidestep mode set vx = 0, so 12.8% of flat time sat at
    zero because nobody had chosen it. Now both are recipe numbers and must be honoured."""
    segs = _draw(3000, "stratified")
    d = np.array([s[0] for s in segs])
    vx = np.array([s[1] for s in segs]); vy = np.array([s[2] for s in segs])
    wz = np.array([s[3] for s in segs])
    stand = d[(vx == 0) & (vy == 0) & (wz == 0)].sum() / d.sum()
    side = d[(vx == 0) & (np.abs(vy) > 1e-9)].sum() / d.sum()
    assert stand == pytest.approx(ENV["mass_stand"], abs=0.02), stand
    assert side == pytest.approx(ENV["mass_sidestep"], abs=0.02), side


def test_turning_and_lateral_are_NOT_confined_to_walk_speed():
    """The old sampler drew turn/diagonal vx from a truncated sub-band, so `jog_turn`, `run_turn` and
    `run_forward x vy` were solved in the library and never once commanded."""
    fast_turn = [s for s in _draw(500, "stratified") if abs(s[3]) > 0.1 and s[1] > 2.0]
    fast_lat = [s for s in _draw(500, "stratified") if abs(s[2]) > 0.1 and s[1] > 2.0]
    assert fast_turn and fast_lat


def test_the_schedule_still_totals_exactly_its_duration():
    for seed in range(20):
        segs = sample_command_schedule(np.random.default_rng(seed), duration=20.0,
                                       seg_dur_range=(1.5, 4.0), **ENV)
        assert total_duration(segs) == pytest.approx(20.0)


def test_a_channel_with_a_zero_range_is_disabled_without_a_branch():
    """How the terrain families stay forward-only -- no mode table, no special case."""
    env = dict(ENV, vy_range=(0.0, 0.0), wz_range=(0.0, 0.0), mass_sidestep=0.0)
    for _d, _vx, vy, wz in sample_command_schedule(
            np.random.default_rng(0), duration=60.0, seg_dur_range=(1.5, 4.0), **env):
        assert vy == 0.0 and wz == 0.0
