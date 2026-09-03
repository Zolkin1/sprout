# Copyright 2026 Zachary Olkin. All rights reserved.

"""Loading a trained tracker frozen, and refusing one that does not fit the environment.

Runs against the REAL deployed checkpoint with a stubbed observation manager, so no simulator
is needed and the architecture under test is the shipped one -- read from the run's own
``params/agent.yaml`` rather than restated here, which is the point of loading it that way.

The contract checks are the reason this class exists. A tracker whose expected observation
width happens to match loads clean and steers wrong; one whose width does not match fails
inside ``load_state_dict`` with a message naming a tensor and no observation term. So each
oracle gets a negative test that asserts WHAT the failure says, not merely that it failed:

* a group of the wrong width names the group and both widths;
* a missing or resized observation term names the TERM -- the one message that points at the
  cfg line someone has to change;
* a different checkpoint of the same width is caught by hash, i.e. by identity.

The positive test also pins that the tracker stays out of any module graph, because that is
what keeps it out of the PPO checkpoint and out of ``load(strict=True)``'s demands.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("rsl_rl.utils", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

from motion_gen.features import _REPO_ROOT  # noqa: E402
from robot_rl.rl.frozen_tracker import FrozenTracker  # noqa: E402

#: The deployed tracker -- the half of the pair that drove the robot.
RUN_DIR = _REPO_ROOT / "logs" / "rsl_rl" / "g1" / "2026-08-20_16-36-10_mode11_decor_plan_v1"
CHECKPOINT = "model_19999.pt"

#: What that checkpoint's own weights imply, so the stub env is right by construction.
POLICY_DIM, REFERENCE_DIM, N_JOINTS = 2490, 611, 29

#: The policy group's terms, as the generator's recording carries them.
POLICY_TERMS = {
    "base_ang_vel": 30, "projected_gravity": 30, "joint_pos": 290, "joint_vel": 290,
    "actions": 290, "depth_near": 780, "depth_far": 780,
}

pytestmark = pytest.mark.skipif(
    not (RUN_DIR / CHECKPOINT).exists(),
    reason="the deployed tracker checkpoint is not on this machine",
)

N_ENVS = 4


def _env(policy_dim: int = POLICY_DIM, reference_dim: int = REFERENCE_DIM,
         terms: dict[str, int] | None = None, action_dim: int = N_JOINTS):
    """A stub exposing only the manager fields the contract check and the build read."""
    terms = POLICY_TERMS if terms is None else terms
    obs_mgr = SimpleNamespace(
        group_obs_dim={"policy": (policy_dim,), "reference": (reference_dim,)},
        active_terms={"policy": list(terms)},
        group_obs_term_dim={"policy": [(int(w),) for w in terms.values()]},
    )
    return SimpleNamespace(
        device=torch.device("cpu"),
        num_envs=N_ENVS,
        observation_manager=obs_mgr,
        action_manager=SimpleNamespace(total_action_dim=action_dim),
    )


def _fm_meta(terms: dict[str, int] | None = None, sha: str | None = None) -> dict:
    """The generator-recorded meta oracles 2 and 3 read."""
    terms = POLICY_TERMS if terms is None else terms
    meta: dict = {"obs_term_names": list(terms), "obs_term_dims": [int(w) for w in terms.values()]}
    if sha is not None:
        meta["checkpoint_sha256"] = sha
    return meta


def _load(**env_kwargs) -> FrozenTracker:
    return FrozenTracker(_env(**env_kwargs), str(RUN_DIR), CHECKPOINT)


def test_it_loads_the_real_checkpoint_and_acts() -> None:
    """The architecture comes from the run's own cfg, and the action is the right width."""
    tracker = _load()
    assert tracker.action_dim == N_JOINTS
    assert tracker.obs_groups == ["policy", "reference"]

    obs = {
        "policy": torch.zeros(N_ENVS, POLICY_DIM),
        "reference": torch.zeros(N_ENVS, REFERENCE_DIM),
        # An extra group the actor does not read: it must be ignored, not concatenated.
        "critic": torch.zeros(N_ENVS, 999),
    }
    action = tracker(obs)
    assert action.shape == (N_ENVS, N_JOINTS)
    assert torch.isfinite(action).all()


def test_it_is_frozen_and_deterministic() -> None:
    """No gradients, eval mode, and the same observation twice gives the same action.

    Deterministic matters beyond reproducibility: this policy is not being trained, so
    sampling would inject exploration nothing benefits from and make the adjuster's credit
    assignment noisier for free.
    """
    tracker = _load()
    assert not any(p.requires_grad for p in tracker.model.parameters())
    assert not tracker.model.training

    obs = {"policy": torch.randn(N_ENVS, POLICY_DIM),
           "reference": torch.randn(N_ENVS, REFERENCE_DIM)}
    torch.testing.assert_close(tracker(obs), tracker(obs), atol=0.0, rtol=0.0)


def test_it_is_not_reachable_from_a_module_graph() -> None:
    """Which is what keeps it out of the PPO checkpoint and out of load(strict=True).

    ``requires_grad_(False)`` is not sufficient on its own: Adam skips ``None``-grad
    parameters, but the weights would still be saved inside ``actor_state_dict`` and demanded
    back on a resume -- so a resume against a different tracker would restore the wrong one.
    """
    tracker = _load()
    assert not isinstance(tracker, torch.nn.Module), (
        "FrozenTracker must not be an nn.Module, or assigning it to one would register it"
    )


def test_a_group_of_the_wrong_width_names_the_group() -> None:
    """The message has to point at an observation group, not at ``mlp.0.weight``."""
    with pytest.raises(ValueError, match=r"direct groups.*'policy'.*expects 2490"):
        _load(policy_dim=POLICY_DIM - 780)          # as if depth_far were dropped
    with pytest.raises(ValueError, match=r"encoder groups.*expects 611"):
        _load(reference_dim=REFERENCE_DIM - 44)


def test_a_missing_observation_term_names_the_TERM() -> None:
    """The one message that points at the cfg line someone has to change.

    Dropping ``depth_far`` is the concrete case: it is the near-only policy group that exists
    on another branch, and it would present as a 780-wide shape complaint about a tensor.
    """
    dropped = {k: v for k, v in POLICY_TERMS.items() if k != "depth_far"}
    env = _env(policy_dim=POLICY_DIM, terms=dropped)
    with pytest.raises(ValueError, match=r"'depth_far' \(780\) is missing from group 'policy'"):
        FrozenTracker(env, str(RUN_DIR), CHECKPOINT, fm_meta=_fm_meta())


def test_an_added_observation_term_is_also_refused() -> None:
    """Symmetrically: the tracker has never seen it, whatever the widths add up to."""
    added = dict(POLICY_TERMS, brand_new_sensor=16)
    env = _env(policy_dim=POLICY_DIM + 16, terms=added)
    with pytest.raises(ValueError, match=r"'brand_new_sensor' \(16\).*never seen it"):
        FrozenTracker(env, str(RUN_DIR), CHECKPOINT, fm_meta=_fm_meta())


def test_a_different_checkpoint_of_the_same_width_is_caught_by_hash() -> None:
    """Identity, not width -- two trackers of one architecture are indistinguishable else."""
    with pytest.raises(ValueError, match=r"hashes.*but the generator was recorded against"):
        FrozenTracker(_env(), str(RUN_DIR), CHECKPOINT, fm_meta=_fm_meta(sha="0" * 64))


def test_allow_mismatch_downgrades_every_check(capsys) -> None:
    """The deliberate cross-pair escape, and it prints what it let through."""
    dropped = {k: v for k, v in POLICY_TERMS.items() if k != "depth_far"}
    env = _env(policy_dim=POLICY_DIM, terms=dropped)
    FrozenTracker(env, str(RUN_DIR), CHECKPOINT, fm_meta=_fm_meta(), allow_mismatch=True)
    printed = capsys.readouterr().out
    assert "depth_far" in printed and "allow_mismatch=True" in printed


def test_a_missing_run_or_checkpoint_says_what_it_wanted() -> None:
    """Naming params/agent.yaml, because that is the non-obvious half of the requirement."""
    with pytest.raises(FileNotFoundError, match=r"params/agent.yaml"):
        FrozenTracker(_env(), os.path.join(str(RUN_DIR), "not-a-run"), CHECKPOINT)
    with pytest.raises(FileNotFoundError):
        FrozenTracker(_env(), str(RUN_DIR), "model_does_not_exist.pt")
