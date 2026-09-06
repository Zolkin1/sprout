# Copyright 2026 Zachary Olkin. All rights reserved.

"""The token-ADJUSTER RL task's config: the parts that would ruin a run without a sound.

This env is the cross of the two beside it -- the flow-matching generator plans, and a
discrete policy corrects the plan -- so the ways it can be silently wrong are the union of
theirs:

* **The generator must be the FLOW one.** There has to be a plan to correct; a token planner
  in that slot leaves the agent adjusting another token model's output, which is a different
  experiment and a different residual dataset.
* **The history arm must match what the RESIDUAL was built under.** ``measured`` and
  ``reference`` rows are the same WIDTH, and a mismatch makes the plan being corrected a
  different object from the one the codebook was fitted against.
* **The residual reward terms must be present at weight ZERO** -- present because they are the
  only place the served correction's magnitude is computed, and zero because the spec puts
  that magnitude under the KL rather than in the reward.
* **The termination flags ARE the reward design.** Arrival carries no reward term, so
  ``gamma * V(s)`` on the one truncation is what makes finishing beat running the clock out.
* **The absolute fall detector must cost what the relative one costs**, or ending the episode
  by falling is cheaper than running out of time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

from robot_rl.tasks.manager_based.robot_rl.g1.g1_token_adjuster_env_cfg import (  # noqa: E402
    G1TokenAdjusterEnvCfg,
)


@pytest.fixture(scope="module")
def cfg() -> G1TokenAdjusterEnvCfg:
    """The composed task cfg."""
    return G1TokenAdjusterEnvCfg()


def test_only_the_goal_bootstraps_its_value(cfg) -> None:
    """Arrival is the sole truncation; everything else is a hard end.

    A second ``time_out=True`` would collect ``gamma * V(s)`` on a FAILURE, which is the
    agent being paid for the state it died in.
    """
    flags = {n: bool(getattr(cfg.terminations, n).time_out)
             for n in dir(cfg.terminations) if not n.startswith("_")
             and getattr(getattr(cfg.terminations, n), "func", None) is not None}
    assert flags.get("goal_reached") is True
    assert [n for n, v in flags.items() if v] == ["goal_reached"], flags


def test_the_generator_is_the_flow_one(cfg) -> None:
    """There must be a plan to correct."""
    assert cfg.commands.traj_ref.generator_kind == "flow_matching"


def test_the_history_arm_matches_the_residual_dataset(cfg) -> None:
    """``measured`` is what build_residual_cache ran the generator under."""
    assert cfg.commands.traj_ref.history_from == "measured"


def test_the_residual_terms_are_measured_but_not_priced(cfg) -> None:
    """Present AND NON-ZERO, so the term actually runs.

    The earlier version of this test asserted ``weight == 0.0`` and called it "measured but
    not priced" -- enshrining the defect it was meant to guard. ``RewardManager.compute``
    SKIPS a zero-weight term entirely, so the one quantity saying how hard the agent is
    pushing would read 0.0 forever while ``reset`` kept emitting an all-zero
    ``Episode_Reward`` for it. Indistinguishable from the "does nothing" failure this stage
    exists to escape.
    """
    for name in ("residual_l2", "residual_rate_l2"):
        term = getattr(cfg.rewards, name, None)
        assert term is not None, f"{name} is the only place |residual| is computed"
        assert term.weight != 0.0, (
            f"{name} at weight 0 is never CALLED by RewardManager.compute, so the "
            "diagnostic it exists for is a structural zero")
        # ...and small enough not to price anything: the dense reward is order 1 and a fall
        # is -10, so a diagnostic weight must sit far below both.
        assert abs(term.weight) < 1e-6, (
            f"{name} at {term.weight} is pricing the correction, not measuring it; the "
            "magnitude bound is the zero-code anchor in the loss")


def test_falling_costs_the_same_however_it_is_detected(cfg) -> None:
    """A termination with no reward term is an escape hatch: it ends the episode for free,
    which beats the -2 for running the clock out."""
    assert cfg.terminations.base_fallen is not None
    assert set(cfg.rewards.failed.params["term_keys"]) == {
        "tracking_violation", "base_fallen"}
    assert cfg.rewards.failed.weight < cfg.rewards.ran_out_of_time.weight < 0.0


def test_the_conditioning_stash_is_on(cfg) -> None:
    """The policy samples from the rows the runtime built for THIS re-plan, and the cfg is
    where it has to be turned on -- ``enable_cond_stash()`` runs after the observation
    manager has probed shapes."""
    assert cfg.commands.traj_ref.stash_conditioning is True


def test_alpha_defaults_to_one_but_zero_is_expressible(cfg) -> None:
    """``alpha = 0`` is the measured stage-C recommendation and has to be reachable from the
    config rather than by editing the env."""
    assert cfg.residual_alpha == 1.0
    cfg2 = G1TokenAdjusterEnvCfg()
    cfg2.residual_alpha = 0.0
    assert cfg2.residual_alpha == 0.0


def test_the_generator_is_sampled_the_way_the_adjuster_was_fitted(cfg) -> None:
    """The residual dataset was built at x0 = 0 and the deployed controller runs x0 = 0, so
    RL must correct the same deterministic plans. A stochastic generator here would train the
    policy against a plan distribution neither end of the pipeline ever sees, and a plan is a
    plan -- nothing about the shapes would say so."""
    assert cfg.commands.traj_ref.fm_zero_noise is True
