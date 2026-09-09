# Copyright 2026 Zachary Olkin. All rights reserved.
"""The progress reward's three new pieces: the gate, the bootstrap, and the worst-so-far.

Each test here is aimed at a failure that trains cleanly and means the wrong thing -- a gate
that passes a mediocre chunk, a bootstrap convention that drifts, an inclusive worst-so-far
that cancels its own advantage, or a context builder that is the right WIDTH and the wrong
layout. None of them is catchable downstream: every one produces plausible floats.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_rl.flow_matching import advantage as adv
from robot_rl.flow_matching import value


# --------------------------------------------------------------------------- the gate

def _gate(reward: np.ndarray, sharpness: float) -> np.ndarray:
    """The terrain-consistency gate, as `inspect_terrain_reward._progress_terms` builds it."""
    return np.sin(0.5 * np.pi * np.power(np.clip(reward, 0.0, 1.0), sharpness))


def test_the_gate_is_the_cosine_form_the_request_was_written_in():
    """`sin((pi/2) R) == cos((pi/2)(1 - R))` -- the same function, and the reason no new
    geometry was needed: `R` is already on disk as the sidecar's bounded goodness."""
    r = np.linspace(0.0, 1.0, 501)
    assert np.allclose(np.sin(0.5 * np.pi * r), np.cos(0.5 * np.pi * (1.0 - r)), atol=1e-15)


def test_the_gate_endpoints_are_exact():
    """A perfect chunk passes everything and a hopeless one passes nothing. Both ends matter:
    a gate that floors above 0 pays velocity credit for a plan that is inside the terrain."""
    assert _gate(np.array([1.0]), 4.0)[0] == pytest.approx(1.0)
    assert _gate(np.array([0.0]), 4.0)[0] == pytest.approx(0.0)
    assert _gate(np.array([1e-9]), 4.0)[0] < 1e-30


def test_sharpness_is_exactly_a_penalty_multiplier():
    """The flag's stated reading: `s` gates as though every penalty were `s` times larger.
    `exp(-s p / tau) = R^s`, so the s-gate at p equals the 1-gate at s*p -- which is the one
    sentence that makes "sharpness 4" mean something to a reader."""
    tau, s = 1.0, 4.0
    p = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    assert np.allclose(_gate(np.exp(-p / tau), s), _gate(np.exp(-s * p / tau), 1.0), atol=1e-12)


def test_sharpness_4_shuts_the_mediocre_chunk_the_default_was_chosen_for():
    """At s=1 a penalty of 0.5 keeps 82% of its credit -- the failure the whole change exists
    to avoid, reintroduced one level up. s=4 is what the box-top spawns were set to."""
    r_half = float(np.exp(-0.5))
    assert _gate(np.array([r_half]), 1.0)[0] > 0.80
    assert _gate(np.array([r_half]), 4.0)[0] < 0.25


def test_the_episode_gate_takes_the_WORST_chunk():
    """One bad chunk collapses the episode's success credit. This is the user's decision and
    the reason the stairs-down dismounts stopped reading as successes: a min over chunks
    cannot be rescued by a long tail of clean ones."""
    good = np.full(40, 0.99)
    with_one_bad = good.copy()
    with_one_bad[17] = float(np.exp(-2.0))
    assert _gate(np.array([good.min()]), 4.0)[0] > 0.93
    assert _gate(np.array([with_one_bad.min()]), 4.0)[0] < 0.01


# ------------------------------------------------------------------- the bootstrap

def _one_episode(n: int, r: float, *, failed: bool = False):
    """One episode of `n` chunks carrying a constant ongoing reward and no terminal bonus."""
    return (np.full(n, r), np.zeros(n), np.ones(n, bool), np.zeros(n, np.int64),
            np.arange(n, dtype=np.int64), np.full(n, failed))


def test_a_constant_reward_is_the_bootstrap_fixed_point():
    """`G = v = r` for a constant reward, which is the identity that keeps value iteration
    from drifting. A convention failing it moves the target every round in a direction no
    metric on the fit reports -- so this is the test that has to hold before any round 1."""
    r, rmax = 1.3, 2.5
    ong, term, ok, uid, rep, failed = _one_episode(30, r)
    v = np.full(30, r)                                  # a perfect frozen critic
    g, g_ok = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed, v=v,
                                             window=3, gamma=0.95, return_max=rmax)
    assert g_ok.all()
    assert np.allclose(g[g_ok], r / rmax, atol=1e-12)


def test_the_MC_round_and_a_perfect_critic_agree():
    """Round 0 (v=None, full-episode MC) and a bootstrap off the exact value must give the
    same target on a constant reward. If they disagree, one of the two paths has the wrong
    (1-gamma) weighting and round 1 would shift every label."""
    ong, term, ok, uid, rep, failed = _one_episode(40, 0.8)
    mc, _ = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed, v=None,
                                           window=3, gamma=0.95, return_max=2.5)
    bs, _ = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed,
                                           v=np.full(40, 0.8), window=3, gamma=0.95,
                                           return_max=2.5)
    assert np.allclose(mc, bs, atol=1e-9)


def test_the_non_failure_tail_does_not_punish_arriving():
    """An ABSORBING tail made G *fall* toward the goal (0.282 -> 0.224 measured), teaching the
    critic that approaching the goal is worse than being far from it. The tail continues at
    the last observed quality instead, so G must be non-decreasing on a constant reward."""
    ong, term, ok, uid, rep, failed = _one_episode(24, 0.9)
    g, g_ok = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed, v=None,
                                             window=3, gamma=0.95, return_max=2.5)
    tail = g[g_ok][-8:]
    assert np.all(np.diff(tail) > -1e-9), f"G falls toward the goal: {tail}"


def test_a_failed_episode_gets_the_absorbing_zero_tail():
    """Failure really is absorbing, and it must separate from a survived episode carrying the
    same rewards -- otherwise the terminal half of the signal says nothing."""
    ong, term, ok, uid, rep, _ = _one_episode(12, 0.9)
    g_f, _ = adv.bootstrapped_window_return(ong, term, ok, uid, rep, np.ones(12, bool),
                                            v=None, window=3, gamma=0.95, return_max=2.5)
    g_s, _ = adv.bootstrapped_window_return(ong, term, ok, uid, rep, np.zeros(12, bool),
                                            v=None, window=3, gamma=0.95, return_max=2.5)
    assert g_f[-1] < g_s[-1]


def test_the_terminal_bonus_enters_once_and_reaches_back_past_the_window():
    """The bonus is a one-off event, so it must not be re-earned every chunk -- and it has to
    reach the EARLY chunks that chose the mode, which is why the target is bootstrapped at all
    (a 3-chunk window on a 40-chunk episode cannot see it)."""
    n = 40
    ong, term, ok, uid, rep, failed = _one_episode(n, 0.5)
    term = term.copy(); term[-1] = 10.0
    g, g_ok = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed, v=None,
                                             window=3, gamma=0.95, return_max=2.5)
    base, _ = adv.bootstrapped_window_return(ong, np.zeros(n), ok, uid, rep, failed, v=None,
                                             window=3, gamma=0.95, return_max=2.5)
    assert g[0] > base[0], "the terminal bonus never reached the mode-choosing chunk"
    assert g.max() <= 1.0 + 1e-9, "G escaped its declared bound"


# --------------------------------------------------------- worst penalty so far

def test_worst_so_far_is_STRICTLY_before_the_state():
    """Including chunk t's OWN penalty leaks r_t into V(s_t) and cancels exactly that much of
    A_t -- the critic would be told the answer. The first chunk therefore has no history and
    must read 0, and every later chunk must lag the running max by one."""
    pen = np.array([0.2, 1.5, 0.4, 3.0, 0.1])
    out, ok = adv.worst_penalty_before(pen, np.ones(5, bool), np.zeros(5, np.int64),
                                       np.arange(5, dtype=np.int64))
    assert ok.all()
    assert out[0] == 0.0
    np.testing.assert_allclose(out, [0.0, 0.2, 1.5, 1.5, 3.0])


def test_worst_so_far_is_monotone_and_per_episode():
    """It is a running max, so it never falls -- and it must RESET across episodes, or one
    episode's disaster is attributed to the next one's opening chunk."""
    pen = np.array([0.5, 2.0, 0.1, 0.3, 0.2])
    uid = np.array([7, 7, 7, 9, 9], dtype=np.int64)
    rep = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    out, _ = adv.worst_penalty_before(pen, np.ones(5, bool), uid, rep)
    np.testing.assert_allclose(out, [0.0, 0.5, 2.0, 0.0, 0.3])
    assert np.all(np.diff(out[:3]) >= 0.0)


def test_worst_so_far_follows_replan_order_not_row_order():
    """Rows arrive in whatever order the window table was sampled in. Reading them positionally
    would attribute a late chunk's penalty to an early state -- a leak of the FUTURE, which is
    the one thing the "strictly before" rule exists to prevent."""
    pen = np.array([3.0, 0.2, 1.0])                     # rows are chunks 2, 0, 1
    rep = np.array([2, 0, 1], dtype=np.int64)
    out, _ = adv.worst_penalty_before(pen, np.ones(3, bool), np.zeros(3, np.int64), rep)
    assert out[1] == 0.0                                # chunk 0 has no history
    assert out[2] == pytest.approx(0.2)                 # chunk 1 sees chunk 0
    assert out[0] == pytest.approx(1.0)                 # chunk 2 sees max(0.2, 1.0)


# --------------------------------------------------------------- the context builder

def test_the_context_widths_are_what_each_arm_declares():
    """Four arms, four widths, and the two 6/7-wide ones are the deployable critic. A width
    that merely AGREES is the trap: a checkpoint fitted with the worst block loads clean
    against a layer built without it whenever the numbers happen to line up."""
    assert value.context_width(privileged=True, goal_context=True) == 9
    assert value.context_width(privileged=True, goal_context=True,
                               worst_penalty_context=True) == 10
    assert value.context_width(privileged=False, goal_context=True) == 6
    assert value.context_width(privileged=False, goal_context=True,
                               worst_penalty_context=True) == 7


def test_the_builder_refuses_a_missing_or_extra_piece():
    """The whole point of one builder over five hand-assemblies: a piece that is absent, spare
    or the wrong width is an error rather than a silently mis-laid token."""
    import torch
    goal = torch.zeros(4, 3)
    twist = torch.zeros(4, 6)
    worst = torch.zeros(4, 1)
    value.build_context(privileged=True, goal=goal, twist=twist)          # baseline is fine
    with pytest.raises((ValueError, TypeError, KeyError)):
        value.build_context(privileged=True, twist=twist)                 # goal missing
    with pytest.raises((ValueError, TypeError, KeyError)):
        value.build_context(privileged=True, goal=goal, twist=twist, worst_penalty=worst)
    with pytest.raises((ValueError, TypeError, KeyError)):
        value.build_context(privileged=True, worst_penalty_context=True,
                            goal=goal, twist=twist)                       # piece missing


def test_the_worst_block_lands_LAST_so_older_checkpoints_keep_their_layout():
    """Appending, not inserting: every block an existing checkpoint reads must keep its
    offset, or a critic on disk reads its goal offsets out of the twist. Same class of bug as
    the transposed depth term, which read as a policy that could not do the task."""
    plain = value.context_layout(privileged=True, goal_context=True)
    wide = value.context_layout(privileged=True, goal_context=True, worst_penalty_context=True)
    assert wide[:len(plain)] == plain
    assert wide[-1][0] == "worst_penalty"


def test_a_missing_frozen_value_falls_back_to_the_tail_rather_than_poisoning_G():
    """An out-of-fold prediction covers only the folds' validation episodes, so a source
    fitted with too few folds has no estimate for most rows. Those arrive non-finite, and
    they must degrade to the Monte Carlo target -- letting the nan through makes G nan for
    every row whose window reaches it (measured: a 40%-covered source gave median G nan),
    and reading it as 0 asserts a FAILED state's value on a healthy one."""
    n = 20
    ong, term, ok, uid, rep, failed = _one_episode(n, 1.0)
    v = np.full(n, np.nan)
    v[:5] = 1.0                                     # only the first five are covered
    g, g_ok = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed, v=v,
                                             window=3, gamma=0.95, return_max=2.5)
    assert np.isfinite(g[g_ok]).all(), "a missing frozen value poisoned G"
    # And it degrades to the MC answer rather than to zero: r = tail_v = 1.0 everywhere, so
    # the fixed point still holds on the uncovered rows.
    assert np.allclose(g[g_ok], 1.0 / 2.5, atol=1e-12)


def test_a_frozen_value_of_zero_is_HONOURED_where_it_is_real():
    """The fallback must key on FINITENESS, not on the value: a genuine prediction of 0 (a
    state the critic really thinks is worthless) has to survive, or the guard would silently
    overwrite the critic's most negative opinions with the tail."""
    n = 12
    ong, term, ok, uid, rep, failed = _one_episode(n, 1.0)
    g_zero, ok0 = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed,
                                                 v=np.zeros(n), window=3, gamma=0.95,
                                                 return_max=2.5)
    g_nan, _ = adv.bootstrapped_window_return(ong, term, ok, uid, rep, failed,
                                              v=np.full(n, np.nan), window=3, gamma=0.95,
                                              return_max=2.5)
    assert g_zero[ok0][0] < g_nan[ok0][0], "a real zero was treated as missing"
