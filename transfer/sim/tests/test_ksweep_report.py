# Copyright 2026 Zachary Olkin. All rights reserved.

"""The K-sweep reducer, against fixtures whose answers are known by construction.

`ksweep_report.py` turns a directory of `live_select.py` runs into the numbers a best-of-K
claim rests on, and every one of them has a plausible wrong version:

* a delta pooled over EPISODES instead of paired within CELL, which reports an interval too
  tight by roughly the square root of the episodes per cell;
* a decomposition whose parts are each paired on their own cells and therefore do not add up
  -- measured once as `critic - K1` +0.098 over 27 cells beside `critic - random` +0.167 and
  `random - K1` +0.008 over 9, three correct numbers that read as an arithmetic error;
* a time-to-goal difference taken between each arm's own successes, which compares different
  episode sets and moves with the success rate rather than with the pace;
* a K=1 row inside a `--random_select` run labelled as the control arm, when K=1 makes no
  choice at all and belongs to neither rule.

Each of those is pinned here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest

_SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "_ksweep_under_test", os.path.join(_SIM, "ksweep_report.py"))
_ks = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _ks
_spec.loader.exec_module(_ks)


def _write_cell(root, tile, speed, arms, *, random_select=False, same_noise=False,
                episodes=10, critic="V_priv.fold0.pt", steer=0.0):
    """One ``<cell>/live_select.json``.

    Args:
        root: Sweep directory.
        tile: Tile id.
        speed: Command speed.
        arms: ``{k: [(verdict, n_replans), ...]}``, one entry per episode.
        random_select: Stamp the random-pick control flag.
        same_noise: Stamp the same-noise control flag.
        episodes: Recorded episode count.
        critic: Recorded critic basename.
        steer: Recorded ``steer_kp``.
    """
    d = os.path.join(root, f"t{tile}_s{speed:g}_nosteer")
    os.makedirs(d, exist_ok=True)
    rows = []
    for k, eps in arms.items():
        for i, (verdict, nr) in enumerate(eps):
            rows.append({"tile": tile, "unit": f"tile{tile}", "k": k, "episode": i,
                         "verdict": verdict, "failure": "", "n_replans": nr,
                         "spread": 0.0, "track_reward": 0.5, "selector": "V"})
    with open(os.path.join(d, "live_select.json"), "w") as fh:
        json.dump({"rows": rows,
                   "args": {"speed": speed, "k": ",".join(str(k) for k in arms),
                            "episodes": episodes, "random_select": random_select,
                            "same_noise": same_noise, "critic_ckpt": f"/x/{critic}",
                            "steer_kp": steer, "tiles": str(tile)}}, fh)
    return d


def _lib(tmp_path, families):
    """A minimal clip library carrying only a ``terrain_manifest``."""
    import h5py

    p = str(tmp_path / "lib.h5")
    with h5py.File(p, "w") as f:
        f.attrs["terrain_manifest"] = json.dumps(
            [{"terrain_id": t, "family": f"family_{fam}.h5"} for t, fam in families.items()])
    return p


def _eps(n_success, n_fail, *, replans_success=10, replans_fail=5):
    """``n_success`` successes then ``n_fail`` falls, with fixed re-plan counts."""
    return ([("success", replans_success)] * n_success
            + [("fall", replans_fail)] * n_fail)


def test_the_family_map_comes_from_the_manifest_and_is_not_guessed(tmp_path):
    """Tiles 40-59 are jump_on, and a range table here got them written up as stairs.

    So the map is read from the library, and an unreadable library is refused rather than
    silently defaulting every cell to one family -- which would put the jump_on cells, where
    selection pays most, into whatever bucket the default named.
    """
    lib = _lib(tmp_path, {6: "stairs_up", 20: "stairs_down", 40: "jump_on",
                          75: "jump_off_front"})
    got = _ks.tile_families(lib)
    assert got == {6: "stairs_up", 20: "stairs_down", 40: "jump_on", 75: "jump_off_front"}
    with pytest.raises(SystemExit, match="not readable"):
        _ks.tile_families(str(tmp_path / "absent.h5"))


def test_a_k1_row_is_never_labelled_with_a_choice_rule(tmp_path):
    """K=1 has nothing to choose, so it belongs to neither the critic nor the random arm.

    A `--random_select` invocation that also ran K=1 would otherwise contribute a phantom
    `K1_random` arm, and the decomposition's `random - K=1` term would then compare a control
    against itself.
    """
    root = str(tmp_path / "sw")
    _write_cell(root, 6, 0.8, {1: _eps(5, 5), 4: _eps(7, 3)}, random_select=True)
    rows, meta = _ks.load_sweeps([root], only_nosteer=True)
    assert meta["arms"] == ["K1", "K4_random"]
    assert {r["arm"] for r in rows if r["k"] == 1} == {"K1"}


def test_the_steer_arm_is_dropped_when_asked(tmp_path):
    """Steering is a different policy; pooling it averages two experiments."""
    root = str(tmp_path / "sw")
    _write_cell(root, 6, 0.8, {1: _eps(5, 5)})
    d = _write_cell(root, 7, 0.8, {1: _eps(9, 1)}, steer=1.5)
    os.rename(d, d.replace("_nosteer", "_steer"))
    rows, _m = _ks.load_sweeps([root], only_nosteer=True)
    assert {int(r["tile"]) for r in rows} == {6}
    rows_all, _m = _ks.load_sweeps([root], only_nosteer=False)
    assert {int(r["tile"]) for r in rows_all} == {6, 7}


def test_the_contrast_is_paired_within_cell_and_drops_unpaired_cells(tmp_path):
    """A cell only one arm reached cannot contribute a difference.

    Built so the unpaired cell would move the answer a lot if it were averaged in: arm B is
    absent exactly where it would have scored worst.
    """
    root = str(tmp_path / "sw")
    _write_cell(root, 6, 0.8, {1: _eps(5, 5), 4: _eps(9, 1)})     # both arms: +0.4
    _write_cell(root, 7, 0.8, {4: _eps(0, 10)})                   # K=4 only, terrible
    rows, _m = _ks.load_sweeps([root], only_nosteer=True)
    cells = _ks.per_cell(rows, replan_seconds=0.24)
    rng = np.random.default_rng(0)
    c = _ks.contrast(cells, "K4_critic", "K1", reps=200, rng=rng, replan_seconds=0.24)
    assert c["n_cells"] == 1
    assert c["delta"] == pytest.approx(0.4, abs=1e-12)


def test_the_decomposition_adds_up_exactly_on_common_cells(tmp_path):
    """`(critic - random) + (random - K1)` must equal `critic - K1`, to the last digit.

    Only true on the cells all three arms reached, which is why the report computes it on the
    intersection. Two cells where every arm ran, one where the control did not.
    """
    root = str(tmp_path / "sw")
    for tile, (k1, k4c, k4r) in {6: (5, 9, 6), 20: (3, 8, 4)}.items():
        _write_cell(root, tile, 0.8,
                    {1: _eps(k1, 10 - k1), 4: _eps(k4c, 10 - k4c)})
        _write_cell(root + "_r", tile, 0.8, {4: _eps(k4r, 10 - k4r)}, random_select=True)
    _write_cell(root, 40, 0.8, {1: _eps(2, 8), 4: _eps(9, 1)})     # no control arm here
    rows, _m = _ks.load_sweeps([root, root + "_r"], only_nosteer=True)
    cells = _ks.per_cell(rows, replan_seconds=0.24)
    common = {c for c in {cc for (cc, _a) in cells}
              if all((c, a) in cells for a in ("K4_critic", "K4_random", "K1"))}
    assert len(common) == 2
    rng = np.random.default_rng(0)
    kw = {"reps": 200, "rng": rng, "replan_seconds": 0.24, "subset": common}
    rank = _ks.contrast(cells, "K4_critic", "K4_random", **kw)["delta"]
    key = _ks.contrast(cells, "K4_random", "K1", **kw)["delta"]
    tot = _ks.contrast(cells, "K4_critic", "K1", **kw)["delta"]
    assert rank + key == pytest.approx(tot, abs=1e-12)
    # And the SAME three contrasts on their own paired sets do NOT add up -- which is the
    # misreading the common-cell section exists to prevent.
    kw_all = {"reps": 200, "rng": rng, "replan_seconds": 0.24}
    tot_all = _ks.contrast(cells, "K4_critic", "K1", **kw_all)["delta"]
    assert abs(rank + key - tot_all) > 0.01


def test_time_to_goal_is_paired_per_episode_over_jointly_successful_ones(tmp_path):
    """A difference of per-arm means over each arm's own successes is a different number.

    Built so the two disagree in SIGN: arm B succeeds on one extra, very fast episode that
    arm A never completes, so B's own mean looks fast while every episode both arms finished
    was faster in A.
    """
    root = str(tmp_path / "sw")
    a = [("success", 10), ("success", 10), ("fall", 3)]
    b = [("success", 12), ("success", 12), ("success", 1)]
    _write_cell(root, 6, 0.8, {4: a, 1: b})
    rows, _m = _ks.load_sweeps([root], only_nosteer=True)
    cells = _ks.per_cell(rows, replan_seconds=0.24)
    rng = np.random.default_rng(0)
    c = _ks.contrast(cells, "K4_critic", "K1", reps=200, rng=rng, replan_seconds=0.24)
    # Paired on episodes 0 and 1 only: A took 10 re-plans where B took 12, so A is FASTER.
    assert c["n_paired_success_episodes"] == 2
    assert c["delta_time_to_goal_s"] == pytest.approx(0.24 * (10 - 12), abs=1e-12)
    # The unpaired difference of means has the opposite sign, which is the trap.
    naive = (cells[("t6_s0.8", "K4_critic")]["mean_time_to_goal_s"]
             - cells[("t6_s0.8", "K1")]["mean_time_to_goal_s"])
    assert naive > 0 > c["delta_time_to_goal_s"]


def test_the_bootstrap_unit_is_the_cell_not_the_episode(tmp_path):
    """Ten times the episodes per cell must not shrink the interval like ten times the cells.

    The interval is over the per-cell deltas, so within-cell precision enters only through
    each delta's own noise. Pinned by giving one dataset 3 cells and another 3 cells with 10x
    the episodes: the interval width must stay the same order, where a pooled-episode
    bootstrap would report it ~3x tighter.
    """
    rng = np.random.default_rng(0)
    few = np.array([0.10, 0.20, 0.30])
    wide = _ks.boot_ci(few, reps=4000, rng=rng)
    # Same three cell values, but the bootstrap knows nothing of episode counts.
    again = _ks.boot_ci(np.repeat(few, 1), reps=4000, rng=rng)
    assert (wide[1] - wide[0]) == pytest.approx(again[1] - again[0], rel=0.25)
    # More CELLS is what tightens it.
    many = _ks.boot_ci(np.tile(few, 10), reps=4000, rng=rng)
    assert (many[1] - many[0]) < 0.5 * (wide[1] - wide[0])


def test_a_shared_arm_that_disagrees_across_sweeps_is_reported(tmp_path):
    """Taking K=4 from one sweep and K=8 from another needs their shared arm to reproduce.

    63 of 63 did on the real pair, which is what licensed that comparison; a tool that could
    not have noticed otherwise is not evidence.
    """
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _write_cell(a, 6, 0.8, {1: _eps(5, 5), 4: _eps(7, 3)})
    _write_cell(b, 6, 0.8, {1: _eps(5, 5), 8: _eps(8, 2)})
    rows, _m = _ks.load_sweeps([a, b], only_nosteer=True)
    notes = _ks.check_shared_arms(rows)
    assert any("1 reproduce EXACTLY, 0 do not" in n for n in notes)
    _write_cell(b, 6, 0.8, {1: _eps(4, 6), 8: _eps(8, 2)})        # K=1 now differs
    rows, _m = _ks.load_sweeps([a, b], only_nosteer=True)
    notes = _ks.check_shared_arms(rows)
    assert any("DISAGREE" in n for n in notes)
    assert any("not paired" in n for n in notes)


def test_the_sign_test_counts_only_cells_that_moved():
    """A tied cell is not evidence either way, and counting it as one would dilute the p."""
    assert _ks._sign_p(9, 0) == pytest.approx(2 / 512, abs=1e-12)
    # No cell moved: NaN, not a p-value and not a crash.
    assert np.isnan(_ks._sign_p(0, 0))
    assert _ks._sign_p(5, 5) == pytest.approx(1.0)


def test_one_cell_gets_no_interval_rather_than_a_zero_width_one():
    """Resampling a single value returns it every time, so the interval would print [x, x].

    That reads as certainty where it means "nothing is known about the between-cell spread",
    and it is the shape a per-cell breakdown hits constantly -- a one-cell stratum.
    """
    rng = np.random.default_rng(0)
    assert all(np.isnan(x) for x in _ks.boot_ci(np.array([0.075]), reps=100, rng=rng))
    lo, hi = _ks.boot_ci(np.array([0.05, 0.10]), reps=2000, rng=rng)
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo


def test_the_selection_RULE_is_part_of_the_arm_name(tmp_path):
    """A measured-quality arm must not be pooled with a critic arm under one label.

    `--select_on measured` reads no model and orders candidates by the realized tracking
    reward of the chunk just executed. Its `live_select.json` is schema-identical to a critic
    run's, so nothing but the recorded flag distinguishes them -- and a reducer that ignored
    it would average two different selectors into one number with no error.
    """
    root = str(tmp_path / "sw")
    _write_cell(root, 6, 0.8, {1: _eps(5, 5), 4: _eps(7, 3)})
    d = os.path.join(root, "t6_s0.8_nosteer")
    with open(os.path.join(d, "live_select.json")) as fh:
        blob = json.load(fh)
    blob["args"]["select_on"] = "measured"
    blob["args"]["critic_ckpt"] = ""
    with open(os.path.join(d, "live_select.json"), "w") as fh:
        json.dump(blob, fh)
    _rows, meta = _ks.load_sweeps([root], only_nosteer=True)
    assert meta["arms"] == ["K1", "K4_measured"]
    # A pre-flag file has no `select_on` and must still read as the critic arm.
    _write_cell(root + "_old", 6, 0.8, {1: _eps(5, 5), 4: _eps(7, 3)})
    _rows, meta = _ks.load_sweeps([root + "_old"], only_nosteer=True)
    assert meta["arms"] == ["K1", "K4_critic"]
