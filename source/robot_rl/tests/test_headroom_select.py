# Copyright 2026 Zachary Olkin. All rights reserved.

"""Selector bookkeeping: the parts that decide an outcome without touching a simulator."""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_ROOT, "scripts", "rsl_rl", "eval"),
           os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import headroom_select as hs  # noqa: E402


def test_a_terminated_candidate_never_wins_however_good_its_score():
    """The failure this guards is the one Isaac's auto-reset makes easy to write.

    A candidate that fell is respawned before it is scored, so it presents the critic with a
    fresh upright robot -- a state a failure critic likes. Ranking on the raw score would
    hand the group the one chunk that is known to have killed it.
    """
    # Group 0: candidate 1 died but scores best by a mile.
    scores = torch.tensor([0.0, 99.0, 1.0, 0.0])
    alive = torch.tensor([True, False, True, True])
    winner, group_alive = hs.choose(scores, alive, num_groups=1, group_size=4)
    assert int(winner[0]) == 2, "the dead candidate's score must not be able to win"
    assert bool(group_alive[0])


def test_a_group_with_no_survivor_is_reported_dead():
    """All K dead is the episode ending, not a selection with an unlucky winner."""
    scores = torch.tensor([5.0, 4.0, 3.0])
    alive = torch.tensor([False, False, False])
    _, group_alive = hs.choose(scores, alive, num_groups=1, group_size=3)
    assert not bool(group_alive[0])


def test_groups_are_chosen_independently():
    """A winner is a group-local argmax mapped back to a global env index."""
    scores = torch.tensor([0.0, 1.0, 7.0, 2.0])
    alive = torch.tensor([True, True, True, True])
    winner, alive_g = hs.choose(scores, alive, num_groups=2, group_size=2)
    assert [int(w) for w in winner] == [1, 2]
    assert bool(alive_g.all())


def test_masking_uses_neg_inf_so_no_score_scale_can_beat_it():
    """A large-negative sentinel would break silently if the critic's range moved."""
    scores = torch.tensor([1e30, 0.0])
    alive = torch.tensor([False, True])
    winner, _ = hs.choose(scores, alive, num_groups=1, group_size=2)
    assert int(winner[0]) == 1


def test_finish_emits_one_row_per_group_episode_not_one_per_env():
    """K+1 envs run one episode; K+1 outcome rows would silently multiply the sample."""
    ge = hs.GroupEpisodes(num_groups=2, group_size=4)
    assert list(ge.carrier) == [0, 4]
    assert list(ge.envs_of(1)) == [4, 5, 6, 7]
    ge.cycles[0] = 9
    ge.finish(0, verdict="success", clip_id=3, family=1, next_episode_id=7)
    assert len(ge.rows) == 1
    assert ge.rows[0]["verdict"] == "success" and ge.rows[0]["cycles"] == 9
    assert int(ge.episode_id[0]) == 7 and int(ge.cycles[0]) == 0


class _FakeTm:
    """A termination manager exposing only what WindowOutcome reads."""

    def __init__(self, terms):
        self.active_terms = list(terms)
        self._t = terms

    def get_term(self, name):
        return self._t[name]


class _FakeEnv:
    def __init__(self, terms):
        self.termination_manager = _FakeTm(terms)


def test_reaching_the_end_of_the_clip_is_not_a_fall():
    """The bug this pins: `dones` sets one bit for two opposite outcomes.

    A group whose candidate reaches clip_end upright has finished the episode successfully.
    Reading `dones` would retire it identically to a wipeout, so every completed episode
    would be scored as a failure -- and because Isaac auto-resets on the same step, the
    reason is gone by the time anything downstream could look.
    """
    n = 2
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([False, False]),
                       "clip_end": torch.tensor([True, True])}))
    assert hs.group_verdict(out, 0, n) == "survived"
    assert not bool(out.alive().any()), "a finished env is not still selectable"


def test_all_candidates_falling_ends_the_episode_as_a_failure():
    n = 3
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([True, True, True]),
                       "clip_end": torch.tensor([False, False, False])}))
    assert hs.group_verdict(out, 0, n) == "fell"


def test_one_survivor_keeps_the_episode_running():
    n = 3
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([True, False, True]),
                       "clip_end": torch.tensor([False, False, False])}))
    assert hs.group_verdict(out, 0, n) is None
    assert [bool(x) for x in out.alive()] == [False, True, False]


def test_a_finish_beside_falls_is_a_completed_episode_not_a_wipeout():
    """Order matters: one candidate finishing while the others fall is a success."""
    n = 3
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([True, False, True]),
                       "clip_end": torch.tensor([False, True, False])}))
    assert hs.group_verdict(out, 0, n) == "survived"


def test_latches_persist_across_steps_within_a_window():
    """A termination fires on ONE step; Isaac resets the env immediately after."""
    n = 2
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([True, False]),
                       "clip_end": torch.tensor([False, False])}))
    out.note(_FakeEnv({"base_fallen": torch.tensor([False, False]),
                       "clip_end": torch.tensor([False, False])}))
    assert [bool(x) for x in out.fell] == [True, False], "the latch must survive the reset"


def test_slot0_verdict_ignores_the_siblings_entirely():
    """The structural control must not let the group survive on a sibling.

    Reading group_verdict there would reintroduce selection through the outcome rule while
    the scorer looked disabled -- and the arm would then not be a control at all.
    """
    n = 3
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([True, False, False]),
                       "clip_end": torch.tensor([False, True, False])}))
    assert hs.group_verdict(out, 0, n) == "survived", "the group has a live/finished sibling"
    assert hs.slot0_verdict(out, 0, n) == "fell", "but candidate 0 itself fell"


def test_slot0_verdict_continues_while_candidate_zero_is_running():
    n = 3
    out = hs.WindowOutcome(n, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([False, True, True]),
                       "clip_end": torch.tensor([False, False, False])}))
    assert hs.slot0_verdict(out, 0, n) is None


def test_slot0_verdict_reads_the_right_group():
    n = 2
    out = hs.WindowOutcome(4, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([False, False, True, False]),
                       "clip_end": torch.tensor([False, False, False, False])}))
    assert hs.slot0_verdict(out, 0, n) is None
    assert hs.slot0_verdict(out, 1, n) == "fell"


def test_arrival_after_a_fall_is_not_a_success():
    """`terminal_reward`'s first-event rule, which its docstring prices at 17 points.

    An episode dragged across the goal line by a reset AFTER it fell did not do the task.
    A window-level 'arrived and also fell' flag cannot tell that from arriving and then
    falling in the run-out, which IS a success.
    """
    out = hs.WindowOutcome(2, "cpu")
    # env 0: falls at step 1, enters the goal at step 2 -> NOT a success.
    # env 1: enters the goal at step 1, falls at step 2 -> success.
    out.note(_FakeEnv({"base_fallen": torch.tensor([True, False]),
                       "clip_end": torch.tensor([False, False])}),
             in_goal=torch.tensor([False, True]))
    out.note(_FakeEnv({"base_fallen": torch.tensor([False, True]),
                       "clip_end": torch.tensor([False, False])}),
             in_goal=torch.tensor([True, False]))
    assert [bool(x) for x in out.arrived_before_falling()] == [False, True]


def test_never_arriving_is_a_failure_however_upright():
    """Standing still must not score. This is the metric bug the goal region exists to stop.

    Survival alone scored a robot that stood still identically to one that climbed the
    staircase, which is why the baseline read 0.864 against the eval's ~0.67.
    """
    out = hs.WindowOutcome(1, "cpu")
    for _ in range(5):
        out.note(_FakeEnv({"base_fallen": torch.tensor([False]),
                           "clip_end": torch.tensor([False])}),
                 in_goal=torch.tensor([False]))
    assert not bool(out.arrived_before_falling()[0])
    assert not bool(out.fell[0]), "it never fell -- it simply never arrived"


def test_first_goal_and_first_fall_latch_only_once():
    out = hs.WindowOutcome(1, "cpu")
    for _ in range(3):
        out.note(_FakeEnv({"base_fallen": torch.tensor([True]),
                           "clip_end": torch.tensor([False])}),
                 in_goal=torch.tensor([True]))
    assert int(out.first_goal[0]) == 1 and int(out.first_fall[0]) == 1


def test_goal_accounting_is_optional():
    """The termination half must still work when no goal is supplied."""
    out = hs.WindowOutcome(1, "cpu")
    out.note(_FakeEnv({"base_fallen": torch.tensor([True]),
                       "clip_end": torch.tensor([False])}))
    assert bool(out.fell[0]) and int(out.first_goal[0]) == -1
