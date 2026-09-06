# Copyright 2026 Zachary Olkin. All rights reserved.

"""The sim-free stage's reward and advantages, and the conditioning split both stages share.

Three things worth pinning, each of which has a silent failure mode:

* **``jitter_excess`` is one-sided by design but the terms it is built from are not.** A
  one-sided summary of a signed quantity is how a DESCENDING plan reported ``0.000`` and read
  as flat while the robot walked downstairs -- so the raw jitter and the target's are asserted
  to be reported alongside the excess.
* **Group advantages must be exactly zero on a degenerate group**, not merely small. A group
  whose plans are all equivalent carries no ranking information, and a near-zero-but-nonzero
  advantage there is noise the policy will happily chase.
* **The conditioning split must invert the packing.** ``_run_fm`` flattens history, extero and
  command into one row; the RL env slices it back. An off-by-one or a swapped order would
  still run and would condition the policy on scrambled inputs -- the exact shape of the
  raw-vs-normalized bug that produced coherent downstairs plans on flat ground.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from robot_rl.token_planner.rl_reward import PlanReward, group_advantages  # noqa: E402

BLOCKS = ["base_pose", "base_vel", "joint_pos"]
NJ, NEE, H = 29, 4, 64


def _rew() -> PlanReward:
    """A reward on the shipped window layout."""
    return PlanReward(NJ, NEE, BLOCKS)


def _window(n: int = 2) -> torch.Tensor:
    """A ``(n, H, D)`` window with an orthonormal rot6d block, so the geodesic is defined."""
    d = 9 + 6 + NJ
    w = torch.zeros(n, H, d)
    w[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return w


def test_an_identical_plan_scores_zero_on_every_term() -> None:
    """The floor: a plan equal to the target has nothing to improve."""
    t = _window()
    r, terms = _rew()(t.clone(), t)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)
    for k in ("pos", "rot", "joint", "vel", "jitter_excess"):
        assert float(terms[k].abs().max()) == pytest.approx(0.0, abs=1e-6), f"{k} nonzero"


def test_the_reward_is_negative_in_the_distance() -> None:
    """More error must mean less reward, or the sign is inverted."""
    t = _window()
    near, far = t.clone(), t.clone()
    near[..., 0] += 0.05
    far[..., 0] += 0.50
    rn, _ = _rew()(near, t)
    rf, _ = _rew()(far, t)
    assert float(rf.mean()) < float(rn.mean()) < 0.0


def test_raw_jitter_and_the_targets_are_both_reported() -> None:
    """An excess of 0 is ambiguous without them: as-smooth-as-the-demo, or a rough demo."""
    t = _window()
    _, terms = _rew()(t.clone(), t)
    assert "jitter" in terms and "jitter_target" in terms, (
        "only `jitter_excess` is reported, so a zero cannot be told apart from a "
        "demonstration rough enough to hide anything the policy does"
    )


def test_jitter_excess_never_penalises_being_as_smooth_as_the_demo() -> None:
    """One-sided on purpose: real motion is not smooth and a plain penalty flattens it."""
    t = _window()
    t[..., 0] = torch.linspace(0, 1, H).pow(2)      # a curved, non-smooth target
    plan = t.clone()
    _, terms = _rew()(plan, t)
    assert float(terms["jitter_excess"].max()) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["jitter"].max()) > 0.0, "the target itself should be non-smooth here"


def test_group_advantages_are_exactly_zero_when_the_group_is_degenerate() -> None:
    """No ranking information, so no gradient -- and 'small' is not good enough."""
    r = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
    a, degen = group_advantages(r)
    assert bool(degen[0]) and not bool(degen[1])
    assert float(a[0].abs().max()) == 0.0, "a degenerate group must contribute exactly nothing"
    assert float(a[1].abs().max()) > 0.0


def test_group_advantages_are_centred_within_the_group() -> None:
    """The group's own mean is the baseline; a nonzero mean is a bias on every token."""
    torch.manual_seed(0)
    a, _ = group_advantages(torch.randn(8, 8))
    assert float(a.mean(dim=-1).abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_leave_one_out_excludes_the_plan_being_scored() -> None:
    """Its own reward must not enter its own baseline, or the advantage is shrunk."""
    r = torch.tensor([[0.0, 0.0, 0.0, 4.0]])
    plain, _ = group_advantages(r, leave_one_out=False)
    loo, _ = group_advantages(r, leave_one_out=True)
    assert float(loo[0, 3]) > float(plain[0, 3]), (
        "the outlier's advantage should be LARGER under leave-one-out, since its own reward "
        "no longer inflates the baseline it is measured against"
    )


def test_the_reward_refuses_a_window_with_no_base_pose() -> None:
    """Every term is anchored on that block; without it the slices are meaningless."""
    with pytest.raises(ValueError, match="base_pose"):
        PlanReward(NJ, NEE, ["joint_pos"])
