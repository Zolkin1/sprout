# Copyright 2026 Zachary Olkin. All rights reserved.

"""A recorded plan must be FK'd in the POLICY's joint order, never the clip library's.

This pins a bug that was live and produced a believable false finding.  The terrain reward
FKs a recorded ``fm_plan`` to get foot geometry; the plan's ``joint_pos`` block is in the
policy's interleaved order (``left_hip_pitch, right_hip_pitch, waist_yaw, ...``) while
``motion_gen.joint_names.G1_JOINT_NAMES`` is the clip library's per-limb order
(``left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, ...``).  **27 of the G1's 29
positions differ.**

Under the wrong order the pipeline returns a plausible wrong skeleton -- a robot standing
roughly in the right place with the wrong posture -- so nothing errors.  Measured on 20k real
plans, it moved the median plan foot speed 0.19 -> 0.72 m/s and contact coverage 43% -> 3%,
and it survived a placement check that requires 90% of soles to sit over their own tile
(it scored 99.8%).  It looked exactly like a real discovery about the generator.

So the order comes from an ARTIFACT and there is no fallback.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
_SCRIPTS = _HERE.parents[3] / "scripts" / "rsl_rl"
sys.path.insert(0, str(_SCRIPTS))


def test_the_two_orders_really_do_differ():
    """The premise. If these ever coincide, the whole hazard is gone and so is this test."""
    import yaml

    from motion_gen.joint_names import G1_JOINT_NAMES                  # noqa: PLC0415

    # Scan for an artifact of THIS robot rather than taking the first one on disk: the log
    # tree still holds pre-29-DoF runs, whose 21-joint order is a different robot's.
    isaac = None
    for cand in sorted((_HERE.parents[3] / "logs" / "rsl_rl" / "g1").glob(
            "*/exported/policy_parameters.yaml")):
        names = yaml.safe_load(cand.read_text()).get("joint_names_isaac") or []
        if len(names) == len(G1_JOINT_NAMES):
            isaac = names
            break
    if isaac is None:
        pytest.skip("no exported policy_parameters.yaml for a 29-joint G1 on this machine")
    assert set(isaac) == set(G1_JOINT_NAMES), "same joints, different order -- that is the trap"
    differ = sum(a != b for a, b in zip(isaac, G1_JOINT_NAMES))
    assert differ >= 27, (
        f"only {differ} positions differ; the orders have converged and the guard below "
        "no longer protects anything worth protecting")


def test_the_resolver_refuses_rather_than_defaulting(tmp_path):
    """No artifact, no order. A fallback here is what makes the failure silent."""
    import h5py
    import numpy as np

    import terrain_reward_io as trio                                   # noqa: PLC0415

    path = tmp_path / "rollout.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_uid", data=np.zeros(1, dtype=np.int64))
        f.create_dataset("fm_plan", data=np.zeros((1, 4, 44), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match="no fallback"):
        trio.isaac_joint_names(str(path))


def test_a_stamped_order_is_preferred_over_any_derivation(tmp_path):
    """Once the recorder stamps it, the rollout is self-describing and nothing is derived."""
    import h5py
    import json
    import numpy as np

    import terrain_reward_io as trio                                   # noqa: PLC0415

    names = ["j%02d" % i for i in range(29)]
    path = tmp_path / "rollout.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("episode_uid", data=np.zeros(1, dtype=np.int64))
        f.attrs[trio.ROLLOUT_JOINT_ORDER_ATTR] = json.dumps(names)
        f.attrs["checkpoint_path"] = "/does/not/exist/model.pt"
    assert trio.isaac_joint_names(str(path)) == names


def test_the_inspector_refuses_a_model_built_in_the_clip_order():
    """The last line of defence: if the resolved order IS the clip order, stop."""
    src = (_SCRIPTS / "inspect_terrain_reward.py").read_text()
    assert "if list(names) == list(G1_JOINT_NAMES):" in src
    assert "refusing rather than FK'ing a wrong skeleton" in src
