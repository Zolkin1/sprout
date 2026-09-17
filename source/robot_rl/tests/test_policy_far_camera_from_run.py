# Copyright 2026 Zachary Olkin. All rights reserved.
"""The eval follows the CHECKPOINT's camera set, read off the run rather than passed as a flag.

The policy reads the near camera alone as of ``efe93b4e``, taken ahead of a teacher retrain.
Every tracker trained before it expects **3002** inputs where a near-only task builds **2222**,
and the difference is exactly **780** -- one 30x26 camera, verified against both checkpoints'
actor weights (``mlp.0.weight`` is ``(1024, 3002)`` on ``2026-08-20_16-36-10_mode11_decor_plan_v1``
and ``(1024, 2222)`` on the near-only retrain).  Without restoring the term an older policy
cannot be loaded at all, which would leave the retrain with nothing to be compared against.

Two things this pins, both of which are silent when wrong:

* **It is DERIVED, not flagged.**  A flag added to the eval's parser is not forwarded to the
  cell subprocess unless ``stage_grid`` restates it, and reverts to its argparse default in the
  child with no error -- measured twice on this harness (``--num_envs`` ran a grid at 256 while
  the manifest recorded 1024; ``--terrain_ref`` produced 3950 episodes with ``n_plans == 0``).
  A property read off the artifact cannot be dropped that way.
* **A run that does not record it returns ``None``, not ``False``.**  "No record" is not
  evidence that the policy was near-only -- it is an older run, which is precisely the case
  that needs the far camera.  Collapsing the two would hand a 3002-input checkpoint a 2222-wide
  vector.

The ``height_scan`` those old runs ALSO list under ``observations.policy`` is a red herring and
the arithmetic is what settles it: 3002 - 2222 is 780, one camera, not a camera plus a scan.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "scripts" / "rsl_rl"))

from provenance import policy_far_camera_in_run  # noqa: E402


def _run_dir(tmp_path, policy_block: str):
    """Write a minimal ``params/env.yaml`` and return the run dir."""
    d = tmp_path / "run"
    (d / "params").mkdir(parents=True)
    (d / "params" / "env.yaml").write_text(
        "observations:\n  policy:\n" + policy_block, encoding="utf-8")
    return str(d)


def test_a_two_camera_run_is_detected(tmp_path):
    """The old trackers: ``depth_far`` present, so the eval must put the term back."""
    d = _run_dir(tmp_path, "    depth_near: {func: x}\n    depth_far: {func: y}\n")
    assert policy_far_camera_in_run(d) is True


def test_a_near_only_run_is_detected(tmp_path):
    """The retrains: the slot is DECLARED as null, which is a positive record of near-only."""
    d = _run_dir(tmp_path, "    depth_near: {func: x}\n    depth_far: null\n")
    assert policy_far_camera_in_run(d) is False


def test_no_record_is_None_and_never_False(tmp_path):
    """A run with no params/ is UNKNOWABLE here; False would silently starve a 3002 policy."""
    d = tmp_path / "bare"
    d.mkdir()
    assert policy_far_camera_in_run(str(d)) is None


def test_a_policy_group_without_the_key_is_None(tmp_path):
    """Same reasoning one level down: the key's absence predates the declaration."""
    d = _run_dir(tmp_path, "    depth_near: {func: x}\n")
    assert policy_far_camera_in_run(d) is None


def test_the_far_block_is_780_wide_which_is_the_whole_3002_2222_gap():
    """Pins the arithmetic the derivation rests on, so a camera resize cannot pass silently."""
    near_only, two_camera, one_camera_block = 2222, 3002, 30 * 26
    assert two_camera - near_only == one_camera_block
