# Copyright 2026 Zachary Olkin. All rights reserved.

"""The box2 task's POLICY observation must be laid out the way its tracker was trained.

This exists because the first attempt was wrong in the one way that does not announce itself.
``enable_policy_far_camera`` re-adds the far camera to a near-only policy group so an existing
two-camera checkpoint can run; written as an ANNOTATED field (``depth_far: ObsTerm | None =
None``) the slot lands BEFORE ``depth_near``, because ``configclass`` orders annotated fields
ahead of un-annotated ones and ``depth_near`` carries no annotation.  The vector is then the
right WIDTH with two 780-wide blocks transposed, so ``load_state_dict`` passes and the tracker
reads the far camera as the near one.

It is not a subtle effect, but it is a silent one -- measured over 64 episodes on the two-box
terrain:

===================  ==============  ==================  ==================
order                reached_goal    steps_survived med  track_reward mean
===================  ==============  ==================  ==================
transposed           0.000           143                 0.57
correct              0.188           517                 0.69
===================  ==============  ==================  ==================

Nothing in that first table looks like a config bug; it looks like a policy that cannot do the
task.  So the order is checked against the deployed artifact's OWN
``observation_terms``/``policy`` block rather than against a list restated here -- the artifact
is what the weights were fitted against, and a list in a test would just be a third place to
get it wrong.

``reference_window`` and ``command_window`` appear in that block but are NOT observation terms
on the cfg: they are strings naming the inputs of the actor's reference encoder
(``RefEncoderMLPModel``), which is why the MLP takes 2490 raw + 512 encoded = 3002 rather than
the block's full 3101.  So the comparison is over the artifact's terms that the cfg actually
declares, in order, and the two extras are asserted to be exactly those two.
"""

from __future__ import annotations

import os

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("isaaclab")

#: The tracker whose observation layout the box2 task must reproduce -- the pair the MuJoCo
#: box2 sweep measured, which is what makes the Isaac and MuJoCo numbers comparable.
_RUN = "2026-08-20_16-36-10_mode11_decor_plan_v1"

#: Terms the trained artifact lists that are encoder inputs rather than cfg observation terms.
_ENCODER_INPUTS: tuple[str, ...] = ("reference_window", "command_window")


def _repo_root() -> str:
    """The repository root, derived from this file rather than from a counted ``..`` hop."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


@pytest.fixture(scope="module")
def trained_terms() -> list[str]:
    """The deployed tracker's policy observation term order, from its exported parameters."""
    path = os.path.join(_repo_root(), "logs", "rsl_rl", "g1", _RUN, "exported",
                        "policy_parameters.yaml")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not present (logs are not in the repo)")
    with open(path) as f:
        return list(yaml.safe_load(f)["observation_terms"]["policy"])


@pytest.fixture(scope="module")
def policy_group():
    """The composed ``G1-box2-finetune`` policy observation group."""
    import robot_rl  # noqa: F401  registers the tasks
    from isaaclab_tasks.utils import parse_env_cfg

    return parse_env_cfg("G1-box2-finetune", device="cpu", num_envs=2).observations.policy


def _declared_terms(group) -> list[str]:
    """The group's observation terms, in the order the ObservationManager will read them.

    Args:
        group: A composed observation group cfg.

    Returns:
        Field names whose value is a live term, in declaration order.
    """
    return [f for f in group.__class__.__dataclass_fields__
            if getattr(group, f, None) is not None and hasattr(getattr(group, f), "func")]


def test_the_policy_terms_are_in_the_trained_order(policy_group, trained_terms):
    """Every term the cfg declares appears in the artifact's order, contiguously and first."""
    declared = _declared_terms(policy_group)
    expected = [t for t in trained_terms if t not in _ENCODER_INPUTS]
    assert declared == expected, (
        "the box2 task's policy observation order does not match the tracker it loads.\n"
        f"  task:    {declared}\n  trained: {expected}\n"
        "A mismatch here is the transposed-camera failure: right width, wrong blocks, and "
        "load_state_dict cannot see it. See this module's docstring.")


def test_the_far_camera_is_present_and_after_the_near_one(policy_group):
    """The whole point of the task: both cameras, far AFTER near."""
    declared = _declared_terms(policy_group)
    assert "depth_near" in declared and "depth_far" in declared, (
        f"expected both policy cameras on G1-box2-finetune, got {declared}. Without the far "
        "one no existing tracker checkpoint loads (3002 inputs against 2222).")
    assert declared.index("depth_far") == declared.index("depth_near") + 1


def test_the_artifacts_extra_terms_are_exactly_the_encoder_inputs(trained_terms, policy_group):
    """Whatever the artifact lists and the cfg does not must be the encoder's two inputs.

    Pins the reason the two lists legitimately differ.  If a THIRD name ever shows up on that
    side, the explanation in this module is no longer the whole story and the order comparison
    above is comparing the wrong sets.
    """
    extra = [t for t in trained_terms if t not in _declared_terms(policy_group)]
    assert extra == list(_ENCODER_INPUTS), (
        f"the trained artifact lists {extra} beyond the cfg's own terms; only "
        f"{list(_ENCODER_INPUTS)} are accounted for (they are RefEncoderMLPModel inputs, not "
        "observation terms). A new name here means the layout changed.")


def test_the_far_camera_slot_is_declared_after_the_near_one_on_the_class():
    """The DECLARED field order puts ``depth_far`` immediately after ``depth_near``.

    A regression guard on the mechanism rather than on the symptom, and it needs no artifact:
    ``logs/`` is not in the repo, so the order test above SKIPS on a fresh checkout while this
    one still runs.

    ``configclass`` orders ANNOTATED fields ahead of un-annotated ones, and ``depth_near``
    carries no annotation -- so writing the far slot as ``depth_far: ObsTerm | None = None``
    moves it in front and transposes the two cameras in the policy vector. Checking the
    resolved field order catches that however it is spelled, where grepping the source for an
    annotation would also match this file's own explanation of it.
    """
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_depth_env_cfg import (
        G1TeacherTerrainDepthObservationsCfg,
    )

    fields = list(G1TeacherTerrainDepthObservationsCfg.PolicyCfg.__dataclass_fields__)
    assert "depth_near" in fields and "depth_far" in fields, fields
    assert fields.index("depth_far") == fields.index("depth_near") + 1, (
        f"declared field order is {fields}. `depth_far` must sit immediately after "
        "`depth_near`: that is the trained layout, and configclass resolves the order from "
        "these declarations. An annotation on `depth_far` moves it ahead of the un-annotated "
        "`depth_near` and transposes the cameras -- right width, wrong blocks.")
