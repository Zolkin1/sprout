# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster's MDP terms: one reward per decision, and the deadline's slack.

Four things are pinned here, each because getting it wrong is silent rather than loud:

* **``slack_s`` defaults to 0.0**, so every existing ``clip_end_reached`` caller is
  bit-identical. The flag exists because without slack the episode budget is *identically*
  the clip's own remaining duration -- the cursor is ``start_step + episode_length_buf`` -- so
  a task where arriving is the objective makes hurrying the reliable strategy, and commanded
  speed is an established degradation vector on stairs.
* **The residual rewards are MEANS over ``F * D``, in normalizer sigmas.** A sum re-weights
  itself whenever the node count or the block selection moves, so one configured weight would
  quietly mean something different after a layout change. Sigmas because a node mixes metres,
  dimensionless rot6d components and radians.
* **The residual RATE forgets across a reset.** The state lives on the command term, which has
  a reset hook; a reward term holding its own would have no way to know a reset happened and
  would charge the policy once per episode for a change it did not make.
* **The two new terminations refuse a placeholder spec.** Their bounds and tolerances are
  measurements, and a bootstrap spec's numbers are declared placeholders -- so falling back to
  one would silently decide which episodes counted as failures and arrivals.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

from robot_rl.flow_matching.window import window_block_slices, window_dim  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.mdp.rewards import rewards as rew  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.mdp.terminations import terminations as term  # noqa: E402

N_ENVS, NJ, NEE, F = 4, 29, 4, 62
BLOCKS = ("base_pose", "base_vel", "joint_pos")
D = window_dim(NJ, NEE, BLOCKS)
STEP_DT = 0.02


# ----------------------------------------------------------------------------------
# clip_end_reached's slack
# ----------------------------------------------------------------------------------
def _clip_env(clip_len: int = 400, elapsed: int = 0, max_offset: int = 50):
    """An env exposing only what ``clip_end_reached`` reads."""
    cmd = SimpleNamespace(
        manager=SimpleNamespace(clip_len=torch.full((8,), clip_len, dtype=torch.long)),
        clip_id=torch.zeros(N_ENVS, dtype=torch.long),
        start_step=torch.zeros(N_ENVS, dtype=torch.long),
        obs_window_offsets=lambda: torch.tensor([0, max_offset], dtype=torch.long),
    )
    return SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _n: cmd),
        episode_length_buf=torch.full((N_ENVS,), elapsed, dtype=torch.long),
        step_dt=STEP_DT,
    )


def test_the_deadline_without_slack_is_the_clips_own_remaining_duration() -> None:
    """The historical behaviour, and the reason slack was needed: zero margin.

    ``last_usable = clip_len - 1 - max_offset``, and the cursor is
    ``start_step + episode_length_buf`` -- so the robot has exactly the clip's time.
    """
    last_usable = 400 - 1 - 50
    assert not bool(term.clip_end_reached(_clip_env(elapsed=last_usable - 1)).any())
    assert bool(term.clip_end_reached(_clip_env(elapsed=last_usable)).all())


@pytest.mark.parametrize("slack_s", [1.0, 3.0])
def test_slack_extends_the_deadline_by_exactly_its_own_seconds(slack_s: float) -> None:
    """Converted with ``env.step_dt``, so it is stated in seconds and not in steps."""
    last_usable = 400 - 1 - 50
    extra = int(round(slack_s / STEP_DT))

    at_old = term.clip_end_reached(_clip_env(elapsed=last_usable), slack_s=slack_s)
    assert not bool(at_old.any()), "slack did not push the deadline out at all"

    just_short = term.clip_end_reached(_clip_env(elapsed=last_usable + extra - 1),
                                       slack_s=slack_s)
    assert not bool(just_short.any())
    at_new = term.clip_end_reached(_clip_env(elapsed=last_usable + extra), slack_s=slack_s)
    assert bool(at_new.all())


def test_slack_defaults_to_zero_so_existing_callers_are_unchanged() -> None:
    """The flag is opt-in; a clip-graded task would be grading a frozen reference with it."""
    import inspect

    assert inspect.signature(term.clip_end_reached).parameters["slack_s"].default == 0.0
    last_usable = 400 - 1 - 50
    env = _clip_env(elapsed=last_usable)
    assert torch.equal(term.clip_end_reached(env), term.clip_end_reached(env, slack_s=0.0))


# ----------------------------------------------------------------------------------
# the residual rewards
# ----------------------------------------------------------------------------------
class _Norm:
    """Just the ``std`` the sigma conversion reads."""

    def __init__(self, std: torch.Tensor) -> None:
        self.std = std


def _residual_env(residual: torch.Tensor, delta: torch.Tensor | None = None):
    """An env whose command term reports a fixed applied residual (and delta)."""
    slices = window_block_slices(NJ, NEE, BLOCKS)
    std = torch.ones(D)
    std[slices["base_pose"].start:slices["base_pose"].start + 3] = 0.4   # position, metres
    std[slices["base_vel"]] = 0.5
    std[slices["joint_pos"]] = 0.2
    cmd = SimpleNamespace(
        ctx=SimpleNamespace(state_norm=_Norm(std)),
        plan_residual=lambda: residual,
        plan_residual_delta=lambda: (residual if delta is None else delta),
    )
    return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _n: cmd))


def test_the_residual_penalty_is_in_sigmas_not_raw_units() -> None:
    """Equal SIGMA residuals on two blocks cost the same, though their metres differ.

    Without the conversion a 0.2 m position residual and a 0.2 rad joint residual would enter
    the same squared mean as equal, though one is half a sigma of position and one is a whole
    sigma of joint angle.
    """
    slices = window_block_slices(NJ, NEE, BLOCKS)
    pos = slices["base_pose"].start

    only_pos = torch.zeros(N_ENVS, F, D)
    only_pos[..., pos:pos + 1] = 0.4                      # exactly 1 sigma of position
    only_joint = torch.zeros(N_ENVS, F, D)
    only_joint[..., slices["joint_pos"].start:slices["joint_pos"].start + 1] = 0.2  # 1 sigma

    a = rew.plan_residual_l2(_residual_env(only_pos))
    b = rew.plan_residual_l2(_residual_env(only_joint))
    torch.testing.assert_close(a, b)
    # One of D channels at 1 sigma on every one of the F nodes -> mean square 1/D.
    torch.testing.assert_close(a, torch.full((N_ENVS,), 1.0 / D))


def test_the_residual_penalty_is_a_mean_so_the_weight_survives_a_layout_change() -> None:
    """The same per-node residual costs the same at any node count.

    A sum over ``F * D`` would scale with the horizon, so one configured weight would mean
    something different the day ``future_len`` moved (it already differs between shipped
    generators: 62 on the measured pair, 64 on the deployed nojumpoff one).
    """
    slices = window_block_slices(NJ, NEE, BLOCKS)

    def cost(nodes: int) -> torch.Tensor:
        residual = torch.zeros(N_ENVS, nodes, D)
        residual[..., slices["base_pose"].start] = 0.4    # 1 sigma, one channel, every node
        return rew.plan_residual_l2(_residual_env(residual))

    torch.testing.assert_close(cost(62), cost(64))
    torch.testing.assert_close(cost(62), torch.full((N_ENVS,), 1.0 / D))


def test_a_zero_residual_costs_nothing() -> None:
    zero = torch.zeros(N_ENVS, F, D)
    assert float(rew.plan_residual_l2(_residual_env(zero)).abs().max()) == 0.0
    assert float(rew.plan_residual_rate_l2(_residual_env(zero)).abs().max()) == 0.0


def test_the_rate_term_reads_the_commands_delta_not_its_own_state() -> None:
    """So a reset can zero it -- a reward term has no reset hook of its own."""
    slices = window_block_slices(NJ, NEE, BLOCKS)
    residual = torch.zeros(N_ENVS, F, D)
    residual[..., slices["base_pose"].start] = 0.4
    delta = torch.zeros(N_ENVS, F, D)                     # held steady between intervals

    env = _residual_env(residual, delta=delta)
    assert float(rew.plan_residual_l2(env).max()) > 0.0, "the residual itself is non-zero"
    assert float(rew.plan_residual_rate_l2(env).abs().max()) == 0.0, (
        "a residual held steady between intervals must cost the rate term nothing"
    )


def test_the_interval_reward_refuses_an_env_that_does_not_fold_it() -> None:
    """Zeros would read as a robot that tracked nothing -- plausible, and silently wrong."""
    with pytest.raises(RuntimeError, match=r"track_mean"):
        rew.served_track_reward(SimpleNamespace())

    env = SimpleNamespace(track_mean=torch.full((N_ENVS,), 0.7))
    torch.testing.assert_close(rew.served_track_reward(env), env.track_mean)


# ----------------------------------------------------------------------------------
# the two terminations refuse placeholders
# ----------------------------------------------------------------------------------
def _cfg(**params):
    return SimpleNamespace(params=params)


def _env_stub():
    return SimpleNamespace(num_envs=N_ENVS, device=torch.device("cpu"))


@pytest.mark.parametrize("klass", [term.TrackingViolation, term.GoalReached])
def test_both_terminations_refuse_an_unnamed_spec(klass) -> None:
    """There is no safe default: the numbers decide which episodes counted."""
    with pytest.raises(ValueError, match=r"spec_path"):
        klass(_cfg(command_name="traj_ref", spec_path=""), _env_stub())


def test_the_violation_term_refuses_a_bootstrap_tracking_spec(tmp_path) -> None:
    """``BOOTSTRAP_SPEC``'s kernel widths are ~2x blunter than the fitted ones."""
    from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error

    path = tmp_path / "spec.json"
    path.write_text(json.dumps(tracking_error.BOOTSTRAP_SPEC.to_dict()))
    assert tracking_error.BOOTSTRAP_SPEC.is_bootstrap, "the fixture must be a bootstrap spec"

    with pytest.raises(ValueError, match=r"BOOTSTRAP"):
        term.TrackingViolation(_cfg(spec_path=str(path)), _env_stub())


def test_the_shipped_tracking_spec_is_fitted_and_loads() -> None:
    """The adopted file, named explicitly -- there is a superseded one beside it.

    ``logs/tracking/v1/tracking_spec.json`` holds survivor-p99.9 bounds roughly 3x tighter
    (lateral 0.067 / height 0.123 / tilt 16.7 deg), which would terminate on the stumbles this
    design delegates to the dense reward. Globbing for a spec would find either.
    """
    from motion_gen.features import _REPO_ROOT
    from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error

    path = _REPO_ROOT / "logs" / "tracking" / "tracking_spec_v1.json"
    if not path.exists():
        pytest.skip("the adopted tracking spec is not on this machine")
    spec = tracking_error.TrackingErrorSpec.from_dict(json.loads(path.read_text()))
    assert not spec.is_bootstrap
    assert spec.max_lateral_m == pytest.approx(0.2)
    assert spec.max_height_m == pytest.approx(0.2)
    assert spec.sigma_pos_m < tracking_error.BOOTSTRAP_SPEC.sigma_pos_m, (
        "the fitted kernel should be tighter than the placeholder it replaces"
    )
