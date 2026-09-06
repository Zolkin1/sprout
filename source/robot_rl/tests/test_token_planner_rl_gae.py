# Copyright 2026 Zachary Olkin. All rights reserved.

"""The RL stage's return arithmetic, which is invisible when it is wrong.

Every test here corresponds to a way the truncation bootstrap has actually been got wrong on
this stack, not to a hypothetical:

* **Dropped entirely.** Arrival carries no reward term on this task -- its whole payoff is
  ``gamma * V(s)`` -- so without the bootstrap running the clock out is worth MORE than
  arriving, and the optimal policy is to track well and never finish. The dense reward then
  RISES while arrivals fall, which reads as a healthy run.
* **Counted twice**, by folding it into the reward AND leaving the next-state bootstrap alive.
  The Bellman operator's modulus goes 0.985 -> 1.97 and the value scale diverges; measured,
  mean reward reached 3.9e8 over 250 iterations while explained variance sat at 0.65, because
  EV is scale-invariant and cannot see it.

Simulator-free on purpose: this is pure arithmetic on tensors and it is exactly the part that
should never need a GPU to check.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robot_rl.token_planner.rl_gae import compute_gae  # noqa: E402

GAMMA, LAM = 0.985, 0.95


def _flat(t: int, n: int = 1, r: float = 0.0):
    """Zero-filled ``(rewards, values, terminated, truncated)`` of the right shapes."""
    return (torch.full((t, n), r), torch.zeros(t + 1, n),
            torch.zeros(t, n, dtype=torch.bool), torch.zeros(t, n, dtype=torch.bool))


def test_a_termination_does_not_bootstrap() -> None:
    """No future exists past a fall, so the next value must not enter the target."""
    r, v, trm, trc = _flat(1)
    v[1, 0] = 100.0                     # a large "next" value that must be ignored
    trm[0, 0] = True
    adv, ret = compute_gae(r, v, trm, trc, GAMMA, LAM)
    assert ret[0, 0] == pytest.approx(0.0), (
        "a terminated step bootstrapped its successor: the return should be the reward "
        "alone, since there is no future to discount"
    )


def test_a_truncation_bootstraps_exactly_once() -> None:
    """Arrival's entire payoff, and the thing that makes arriving beat dawdling."""
    r, v, trm, trc = _flat(1)
    v[0, 0] = 10.0                      # V(s_t): the state the agent arrived in
    v[1, 0] = 999.0                     # the post-reset state; must NOT be used
    trc[0, 0] = True
    adv, ret = compute_gae(r, v, trm, trc, GAMMA, LAM)
    # Convention B: r' = r + gamma*V(s_t); delta = r' - V(s_t); ret = adv + V(s_t) = r'
    assert ret[0, 0] == pytest.approx(GAMMA * 10.0, rel=1e-6), (
        f"expected exactly one bootstrap of gamma*V(s_t) = {GAMMA * 10.0}, got {ret[0, 0]}. "
        "Twice would be ~2*gamma*V and would diverge; zero would make arrival worthless"
    )


def test_double_counting_the_truncation_would_be_caught() -> None:
    """The regression guard: a second bootstrap changes the target by gamma*V, not epsilon."""
    r, v, trm, trc = _flat(1)
    v[0, 0], v[1, 0] = 10.0, 10.0
    trc[0, 0] = True
    _, ret = compute_gae(r, v, trm, trc, GAMMA, LAM)
    once = GAMMA * 10.0
    twice = 2.0 * GAMMA * 10.0
    assert ret[0, 0] == pytest.approx(once, rel=1e-6)
    assert abs(float(ret[0, 0]) - twice) > 1.0, (
        "the single- and double-bootstrap targets must be far apart, or this test cannot "
        "distinguish them"
    )


def test_arriving_beats_running_the_clock_out() -> None:
    """The reward design's central claim, on the real weights.

    Dense trackability ~0.8/interval, timeout penalty -2, and the goal carrying no term of its
    own. The critic is converged on a task that CONTINUES -- ``V(s) = dense / (1 - gamma)``
    everywhere -- which is what a truncation means: the world did not stop, we stopped
    watching. (Setting ``V(s_goal) = 0`` instead encodes the TERMINAL assumption and makes
    arrival worthless by construction; that mistake is what this docstring exists to prevent
    a reader from repeating.)

    Under it, arriving at macro-step 50 must beat surviving to the deadline at 100. This is
    the comparison that inverted when the bootstrap was dropped.
    """
    dense, v_inf = 0.8, 0.8 / (1.0 - GAMMA)

    def rollout(n: int, arrive: bool) -> float:
        """Value target at ``t = 0`` for one episode of ``n`` macro-steps."""
        r, v, trm, trc = _flat(n, 1, dense)
        v.fill_(v_inf)
        if arrive:
            trc[n - 1, 0] = True
        else:
            r[n - 1, 0] = dense - 2.0
            trm[n - 1, 0] = True
            v[n, 0] = 0.0                     # nothing follows a termination
        _, ret = compute_gae(r, v, trm, trc, GAMMA, LAM)
        return float(ret[0, 0])

    arriving, dawdling = rollout(50, arrive=True), rollout(100, arrive=False)
    assert arriving > dawdling, (
        f"running the clock out ({dawdling:.2f}) beat arriving ({arriving:.2f}). With no "
        "reward term on the goal the bootstrap is the ONLY thing that makes finishing "
        "worthwhile -- if this fails the agent is being trained to avoid the goal while its "
        "dense reward rises, which reads as a healthy run"
    )


def test_dropping_the_truncation_bootstrap_inverts_the_task() -> None:
    """The mutation guard: with arrival treated as a termination, dawdling must WIN.

    Without this, the test above could pass for reasons unrelated to the bootstrap.
    """
    dense, v_inf = 0.8, 0.8 / (1.0 - GAMMA)

    def rollout(n: int, arrive_as_termination: bool) -> float:
        """Value target at ``t = 0``, with arrival mis-flagged as a termination."""
        r, v, trm, trc = _flat(n, 1, dense)
        v.fill_(v_inf)
        trm[n - 1, 0] = True
        if not arrive_as_termination:
            r[n - 1, 0] = dense - 2.0
        v[n, 0] = 0.0
        _, ret = compute_gae(r, v, trm, trc, GAMMA, LAM)
        return float(ret[0, 0])

    assert rollout(50, True) < rollout(100, False), (
        "with the bootstrap dropped, arriving should be WORSE than running the clock out -- "
        "if it is not, this pair of tests is not measuring the bootstrap at all"
    )


def test_a_step_cannot_be_both_terminated_and_truncated() -> None:
    """Not a representable ending, and silently it would double-count the bootstrap."""
    r, v, trm, trc = _flat(1)
    trm[0, 0] = trc[0, 0] = True
    with pytest.raises(ValueError, match="BOTH terminated and truncated"):
        compute_gae(r, v, trm, trc, GAMMA, LAM)


def test_the_trace_is_cut_at_every_ending() -> None:
    """A later episode's advantage must not leak backwards across a reset."""
    r, v, trm, trc = _flat(3)
    r[2, 0] = 100.0                     # a big reward in the NEXT episode
    trm[0, 0] = True                    # episode ends at t=0
    adv, _ = compute_gae(r, v, trm, trc, GAMMA, LAM)
    assert adv[0, 0] == pytest.approx(0.0, abs=1e-6), (
        "reward from after a reset leaked backwards into the previous episode's advantage"
    )


def test_no_ending_reduces_to_plain_gae() -> None:
    """With nothing ending, the recursion must be textbook TD(lambda)."""
    torch.manual_seed(0)
    r = torch.randn(6, 4)
    v = torch.randn(7, 4)
    trm = torch.zeros(6, 4, dtype=torch.bool)
    adv, ret = compute_gae(r, v, trm, trm.clone(), GAMMA, LAM)
    exp = torch.zeros_like(r)
    last = torch.zeros(4)
    for i in reversed(range(6)):
        last = (r[i] + GAMMA * v[i + 1] - v[i]) + GAMMA * LAM * last
        exp[i] = last
    assert torch.allclose(adv, exp, atol=1e-6)
    assert torch.allclose(ret, exp + v[:6], atol=1e-6)


def test_shape_errors_are_refused() -> None:
    """A [T, N] values array instead of [T+1, N] silently loses the tail bootstrap."""
    r, v, trm, trc = _flat(4)
    with pytest.raises(ValueError, match=r"values must be"):
        compute_gae(r, v[:-1], trm, trc, GAMMA, LAM)
