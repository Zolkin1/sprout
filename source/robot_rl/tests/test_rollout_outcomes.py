# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pin the rollout outcome table, and above all the JOIN KEY.

A closed-loop rollout writes two files: one row per generator re-plan, and one row per
episode. An advantage label is a function of both, so the key that joins them is the single
thing most able to produce a table that is well formed, reduces without error, and describes
something that never happened.

There were already two episode counters in the repo and they are not each other:

* ``collect_windows``' ``episode_id`` is per ENV -- it starts at the env index and steps by
  ``num_envs``, so it says which episode of that env this was;
* ``eval.accumulate``'s is assigned at HARVEST from a global counter, in whatever order envs
  happen to finish.

Joining a window table on one against an outcome table on the other lines up different
episodes, for every env that has reset a different number of times -- which after a few
hundred steps is all of them. ``episode_uid`` exists so there is one counter, and these tests
are what stop it drifting back into two.
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve()
_SCRIPTS = _HERE.parents[3] / "scripts" / "rsl_rl"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import rollout_outcomes as ro  # noqa: E402


def _row(**over):
    """One episode row with every column present, overridable."""
    base = {
        "episode_uid": 0, "env_id": 0, "clip_id": 0, "terrain_id": -1, "family_code": 0,
        "start_step": 0, "last_usable_step": 300, "n_replans": 25,
        "steps_survived": 299, "terminated": 0, "truncated": 1, "censored": 0,
        "first_fall_step": -1,
        # Schema v2's strict checks. Default "never fired", so every existing test keeps
        # measuring the loose rule it was written for.
        "first_violation_step": -1, "first_lateral_step": -1, "first_height_step": -1,
        "first_tilt_step": -1, "track_reward_mean": float("nan"),
        "min_base_height_w": 0.75, "max_tilt_rad": 0.1,
        "reached_goal": 1, "first_goal_step": 200, "min_dist_to_goal_m": 0.0,
        "goal_downrange_m": 0.0, "goal_lateral_m": 0.0, "goal_height_err_m": 0.0,
        "max_goal_downrange_m": 0.0,
        "goal_pos_w_x": 0.0, "goal_pos_w_y": 0.0, "goal_pos_w_z": 0.0,
        "goal_frame_yaw": 0.0, "spawned_in_goal": 0, "goal_travel_m": 5.0,
    }
    base.update(over)
    return {k: np.asarray([v]) for k, v in base.items()}


# ----------------------------------------------------------------------------------------
# The reward. Every clause here is a way the method fails if it is wrong.
# ----------------------------------------------------------------------------------------

def test_arriving_upright_is_the_only_zero_reward():
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(reached_goal=1, terminated=0))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [0.0]


def test_a_free_success_is_flagged_rather_than_hidden():
    """An episode already inside the goal on its first step had no task to do.

    With the clip's random start phase on, a large fraction of arrivals are this -- episodes
    opening at or past the clip's own endpoint. They are still scored a success (the rule is
    the rule) but the flag is what lets a fit drop them, and what makes the contamination
    visible instead of silent.
    """
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(reached_goal=1, first_goal_step=1, spawned_in_goal=1))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [0.0]
    assert cols["spawned_in_goal"].tolist() == [1]


def test_surviving_without_arriving_is_a_FAILURE():
    """The stall clause, and the single most important line in the reward.

    If merely staying upright scored 0, standing still would score identically to climbing
    the staircase with none of the risk -- and an operator that reshapes a generator toward
    high-scoring outcomes finds that immediately. The generator is conditioned on a velocity
    command but nothing forces it to obey one.
    """
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(reached_goal=0, first_goal_step=-1, terminated=0, truncated=1,
                        steps_survived=299))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [-1.0]


def test_falling_BEFORE_arriving_is_a_failure():
    """Fell at step 40, dragged over the line later -- that is not doing the task."""
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(reached_goal=1, terminated=1, first_fall_step=40, first_goal_step=90))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [-1.0]


def test_arriving_BEFORE_falling_is_a_SUCCESS():
    """Reached the goal at step 90, fell in the run-out at 140. The task was done.

    Not a corner case: a substantial share of episodes arrive and then fall, so scoring those
    as failures -- which `arrived AND not terminated` does -- shifts the measured success rate
    by more than the policy differences this is meant to resolve. `mjeval` has always used the
    first-event rule (it tests `robot.failed()` before the target each step and breaks on
    either).
    """
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(reached_goal=1, terminated=1, first_goal_step=90, first_fall_step=140))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [0.0]


def test_never_arriving_is_a_failure_however_it_ended():
    t = ro.RolloutOutcomeTable()
    for kw in ({"terminated": 1, "first_fall_step": 40}, {"terminated": 0, "truncated": 1}):
        t.append_batch(_row(reached_goal=0, first_goal_step=-1, **kw))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [-1.0, -1.0]


def test_reward_does_not_read_steps_survived():
    """Duration is not evidence of anything here; if this fails a temporal term crept in."""
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(reached_goal=1, terminated=0, steps_survived=5))
    t.append_batch(_row(reached_goal=1, terminated=0, steps_survived=999))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert ro.terminal_reward(cols).tolist() == [0.0, 0.0]


# ----------------------------------------------------------------------------------------
# The table contract.
# ----------------------------------------------------------------------------------------

def test_a_missing_column_is_refused_naming_both_sets():
    t = ro.RolloutOutcomeTable()
    bad = _row()
    del bad["reached_goal"]
    with pytest.raises(ValueError, match="missing.*reached_goal"):
        t.append_batch(bad)


def test_an_unexpected_column_is_refused():
    t = ro.RolloutOutcomeTable()
    bad = _row()
    bad["something_new"] = np.asarray([1])
    with pytest.raises(ValueError, match="unexpected.*something_new"):
        t.append_batch(bad)


def test_a_ragged_batch_is_refused():
    t = ro.RolloutOutcomeTable()
    bad = _row()
    bad["env_id"] = np.asarray([0, 1])
    with pytest.raises(ValueError, match="ragged"):
        t.append_batch(bad)


def test_round_trip_through_h5(tmp_path):
    t = ro.RolloutOutcomeTable(family_names=["unknown", "family_stairs_up"])
    t.append_batch(_row(episode_uid=7, family_code=1, reached_goal=0))
    t.append_batch(_row(episode_uid=8, family_code=1, reached_goal=1))
    path = str(tmp_path / "episodes.h5")
    t.write(path, attrs={"goal_regions": "/tmp/x.json"})
    cols, attrs = ro.read_rollout_episodes(path)
    assert cols["episode_uid"].tolist() == [7, 8]
    assert cols["reached_goal"].tolist() == [0, 1]
    assert attrs["family_names"] == ["unknown", "family_stairs_up"]
    assert attrs["goal_regions"] == "/tmp/x.json"
    assert ro.terminal_reward(cols).tolist() == [-1.0, 0.0]


def test_a_foreign_schema_version_is_refused(tmp_path):
    import h5py
    path = str(tmp_path / "old.h5")
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version 99"):
        ro.read_rollout_episodes(path)


# ----------------------------------------------------------------------------------------
# Family normalisation: goal_regions.json is keyed the way the eval's sampler spells it.
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [
    ("family_stairs_up", "stairs_up"),
    ("family_stairs_up.h5", "stairs_up"),
    ("stairs_up", "stairs_up"),
    ("unknown", "unknown"),
])
def test_family_names_normalise_the_way_the_eval_spells_them(raw, want):
    assert ro._normalized([raw]) == [want]


def test_normalisation_matches_the_evals_own_sampler():
    """One spelling, checked against the definition rather than restated.

    `goal_regions.json` is written keyed by `eval.sampler.normalize_family`'s output, so a
    second normaliser that disagrees would silently hand every family the DEFAULT tolerances
    -- which is a plausible region, on every tile, and invisible in the table.
    """
    sampler = pytest.importorskip("sampler")
    raws = ["family_stairs_up", "family_stairs_up.h5", "family_jump_off_front", "unknown"]
    assert ro._normalized(raws) == [sampler.normalize_family(r) for r in raws]


# ----------------------------------------------------------------------------------------
# The join invariant. This is the failure that produces a table which reduces without error
# and describes episodes that never happened.
# ----------------------------------------------------------------------------------------

class _FakeEnv:
    """The two attributes the tracker reads at construction."""

    def __init__(self, n):
        self.num_envs = n
        self.device = "cpu"


class _FakeAccum:
    device = "cpu"


def _tracker(n=4):
    return ro.RolloutEpisodeTracker(
        base_func=lambda *a, **k: None, env=_FakeEnv(n), accum=_FakeAccum(),
        table=ro.RolloutOutcomeTable())


def test_replan_index_starts_at_zero_and_counts_up():
    import torch
    t = _tracker(4)
    ids = torch.tensor([0, 1])
    for expect in (0, 1, 2):
        _, ridx = t.on_replan(ids)
        assert ridx.tolist() == [expect, expect]


def test_on_replan_returns_the_index_BEFORE_bumping():
    """The first window of an episode must be index 0, not 1.

    An off-by-one here shifts every n-step advantage by one re-plan, which is a quarter of a
    second of look-ahead silently attributed to the wrong chunk.
    """
    import torch
    t = _tracker(2)
    _, first = t.on_replan(torch.tensor([0]))
    assert first.tolist() == [0]


def test_reissue_moves_BOTH_counters():
    import torch
    t = _tracker(4)
    ids = torch.tensor([1, 2])
    t.on_replan(ids)
    t.on_replan(ids)
    assert t.replan_index[ids].tolist() == [2, 2]
    before = t.episode_uid[ids].tolist()
    t.reissue(ids)
    assert t.replan_index[ids].tolist() == [0, 0], "index not reset -> keys continue the old episode"
    assert t.episode_uid[ids].tolist() != before, "uid not bumped -> two episodes share a key"


def test_uids_are_never_reused():
    """Every episode across the whole rollout needs its own key, not just every env."""
    import torch
    t = _tracker(4)
    seen = set(t.episode_uid.tolist())
    for _ in range(50):
        ids = torch.tensor([0, 3])
        t.reissue(ids)
        new = t.episode_uid[ids].tolist()
        assert not (seen & set(new)), f"uid reused: {sorted(seen & set(new))}"
        seen |= set(new)
    assert len(seen) == 4 + 100


def test_untouched_envs_keep_their_identity_across_a_partial_reset():
    """Resets are per-env. Reissuing the whole batch would orphan every running episode."""
    import torch
    t = _tracker(4)
    t.on_replan(torch.tensor([0, 1, 2, 3]))
    before_uid = t.episode_uid.tolist()
    before_idx = t.replan_index.tolist()
    t.reissue(torch.tensor([2]))
    assert t.episode_uid[[0, 1, 3]].tolist() == [before_uid[i] for i in (0, 1, 3)]
    assert t.replan_index[[0, 1, 3]].tolist() == [before_idx[i] for i in (0, 1, 3)]
    assert t.replan_index[2].item() == 0


def test_reissue_of_nothing_is_a_no_op():
    import torch
    t = _tracker(4)
    before = (t.episode_uid.tolist(), t._next_uid)
    t.reissue(torch.tensor([], dtype=torch.long))
    assert (t.episode_uid.tolist(), t._next_uid) == before


def test_goal_travel_is_recorded_so_a_backwards_family_cannot_hide():
    """A negative travel means the goal sits BEHIND where the episode opened.

    A forward half-plane cannot express that, so the robot is inside it on step 1 and stays
    there. On the shipped library `jump_off` travels NEGATIVE -- it is the jump-off-a-box
    BACKWARDS family -- and most of its episodes scored a free success, while the forward
    families travel positive and none did. Recording the travel is what makes that visible on
    a library nobody has measured yet; it was found the first time by drawing six episodes
    and looking.
    """
    t = ro.RolloutOutcomeTable()
    t.append_batch(_row(goal_travel_m=-0.41, spawned_in_goal=1, first_goal_step=1))
    t.append_batch(_row(goal_travel_m=2.68, spawned_in_goal=0, first_goal_step=180))
    cols = {k: np.asarray([r[k] for r in t._rows]) for k in ro.ROLLOUT_COLUMN_NAMES}
    assert cols["goal_travel_m"].tolist() == pytest.approx([-0.41, 2.68])
    # Both still score successes -- the rule is the rule. The flag and the travel are what
    # say the first one meant nothing.
    assert ro.terminal_reward(cols).tolist() == [0.0, 0.0]
    assert cols["spawned_in_goal"].tolist() == [1, 0]


# ----------------------------------------------------------------------------------------
# Schema v2: the per-chunk tracking-error block and the strict failure limb.
# ----------------------------------------------------------------------------------------
def test_the_track_err_ring_drops_the_new_chunks_first_row():
    """K+1 deep, newest last, and the WRITTEN block is `buf[:, :-1]`.

    The re-plan is applied inside ``env.step``, so the step folded at the emit instant is
    already served by the NEW chunk. A ring of depth K would hand out K-1 steps of the
    previous chunk plus one of the new one -- a mix belonging to neither action, and one that
    would fit, train, and produce a plausible weak critic rather than an error.
    """
    import torch

    nj, k, n = 29, 12, 4
    dim = _TRACK_DIM(nj)
    ring = ro.TrackErrRing(n, "cpu", k + 1, dim)
    for step in range(3 * k):
        ring.push(torch.full((n, dim), float(step)))
    last = 3 * k - 1
    assert float(ring.buf[0, -1, 0]) == last, "newest row is not last"
    written = ring.buf[:, :-1]
    assert written.shape[1] == k, "the written block is not one chunk wide"
    assert [float(v) for v in written[0, :, 0]] == [float(last - k + i) for i in range(k)], (
        "the written block is not exactly the K steps before the emit instant")

    ring.invalidate(torch.tensor([1, 0, 0, 0]))
    assert float(ring.buf[0].abs().max()) == 0.0, "a reset env kept its rows"
    assert float(ring.buf[1].abs().max()) > 0.0, "invalidate zeroed an env that did not reset"

    # Before the generator's first plan there is no row; pushing None must be a NO-OP rather
    # than repeating the last row, which would look like a robot holding a constant error.
    ring.push(None)
    assert float(ring.buf[0].abs().max()) == 0.0


def _TRACK_DIM(num_joints: int) -> int:
    """Width of one error row, from the shared contract."""
    from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error

    return tracking_error.NUM_FIXED_CHANNELS + int(num_joints)


def test_terminal_reward_strict_adds_the_violation_and_old_tables_fall_back():
    """The strict limb only ever REMOVES successes, and a v1 table reads under the loose rule.

    The fall-only rule stays reachable so every number published before the tracking checks
    existed is still reproducible from the same data -- which is the whole reason the checks
    are latched rather than terminating.
    """
    cols = {
        "reached_goal": np.array([1, 1, 1, 0]),
        "first_goal_step": np.array([10, 40, 10, -1]),
        "first_fall_step": np.array([-1, 10, -1, -1]),
        #             clean | fell first | violated BEFORE arriving | never arrived
        "first_violation_step": np.array([-1, -1, 5, -1]),
    }
    assert ro.terminal_reward(cols, strict=False).tolist() == [0.0, -1.0, 0.0, -1.0]
    assert ro.terminal_reward(cols, strict=True).tolist() == [0.0, -1.0, -1.0, -1.0]

    old = {k: v for k, v in cols.items() if k != "first_violation_step"}
    assert ro.terminal_reward(old, strict=True).tolist() == [0.0, -1.0, 0.0, -1.0]


def test_a_violation_AFTER_arriving_is_still_a_success():
    """First-event ordering applies to the strict limb exactly as it does to the fall."""
    cols = {
        "reached_goal": np.array([1]),
        "first_goal_step": np.array([10]),
        "first_fall_step": np.array([-1]),
        "first_violation_step": np.array([40]),
    }
    assert ro.terminal_reward(cols, strict=True).tolist() == [0.0]


def test_the_window_writer_sizes_the_track_err_block():
    """`track_hw` is optional, and omitting it leaves the column out rather than zero-filling.

    A zero-filled block would be read by the critic as "tracked perfectly for the whole
    chunk", which is the most flattering possible wrong answer.
    """
    import tempfile

    import h5py

    dim = _TRACK_DIM(29)
    with tempfile.TemporaryDirectory() as d:
        on = os.path.join(d, "on.h5")
        with ro.RolloutWindowWriter(on, num_windows=2, num_history=4, state_dim=44,
                                    future_len=64, cmd_dim=3, extero_shape=(2, 16),
                                    proprio_hw=(10, 64), track_hw=(12, dim), attrs={}):
            pass
        with h5py.File(on, "r") as f:
            assert f["track_err_prev"].shape == (2, 12, dim)

        off = os.path.join(d, "off.h5")
        with ro.RolloutWindowWriter(off, num_windows=2, num_history=4, state_dim=44,
                                    future_len=64, cmd_dim=3, extero_shape=(2, 16),
                                    proprio_hw=(10, 64), attrs={}):
            pass
        with h5py.File(off, "r") as f:
            assert "track_err_prev" not in f


def test_every_window_column_is_actually_allocated(tmp_path):
    """A column in the schema tuple but not in the writer's `shapes` is dropped in SILENCE.

    `RolloutWindowWriter` allocates from an explicit `shapes` dict, not from
    :data:`ROLLOUT_WINDOW_COLUMNS`, so the two can disagree -- and the failure is a column
    that every reader believes exists, that nothing wrote, and that no error mentions.
    """
    import rollout_outcomes as ro

    w = ro.RolloutWindowWriter(
        str(tmp_path / "w.h5"), num_windows=4, num_history=2, state_dim=3,
        extero_shape=(2, 5), future_len=6, cmd_dim=3, proprio_hw=(2, 4), scan_dim=7,
        track_hw=(2, 3), attrs={})
    try:
        allocated = set(w._keys)
    finally:
        w.close()
    declared = set(ro.ROLLOUT_WINDOW_NAMES)
    assert declared - allocated == set(), (
        f"declared but never allocated: {sorted(declared - allocated)}")
    assert allocated - declared == set(), (
        f"allocated but not declared: {sorted(allocated - declared)}")


def test_the_perturbation_arms_are_recorded_per_window():
    """The only on-disk record that a row's ACTION came from perturbed conditioning.

    `history` and `extero` store the TRUE measurement by design, so without these flags the
    perturbed and pristine populations are indistinguishable and the advantage cannot be split
    by arm.  Distinct from `history_null`, which is the post-reset transient and is true of the
    recorded history.
    """
    import rollout_outcomes as ro

    for col in ("perturb_null_history", "perturb_noisy_upper_cam"):
        assert col in ro.ROLLOUT_WINDOW_NAMES, col
        assert dict(ro.ROLLOUT_WINDOW_COLUMNS)[col] == "u1", f"{col} should be one byte"
    assert "history_null" in ro.ROLLOUT_WINDOW_NAMES, (
        "history_null is a different signal and must not have been replaced")
    assert ro.SCHEMA_VERSION >= 3, "adding columns must bump the schema version"
