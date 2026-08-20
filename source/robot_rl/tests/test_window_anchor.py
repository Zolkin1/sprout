# Copyright 2026 Zachary Olkin. All rights reserved.

"""``window_anchor`` picks what the reference window's ``base_pose`` position MEANS.

Two conventions, same width, different meaning -- the exact shape of failure the
``base_pose_convention`` string exists to catch:

* ``"robot_now"`` -- ``p_ref - p_actual(t)``, the tracking error. Every run before
  2026-08-20, including the one training on talos right now.
* ``"plan"`` -- ``p_ref - p_plan``, the reference's position in the frame of the last
  re-plan. A pure function of the reference, which is what lets the deploy runtime serve the
  generator's prediction without ever reading the robot's world position.

The load-bearing property this file pins is that **adding the flag changed nothing for
anyone who does not set it**. A run trained before the field existed has no ``window_anchor``
in its saved config, so it resolves to the default, and ``check_run_conventions`` refuses a
mismatch -- if the default ever stopped reproducing the original string byte-for-byte, every
such run would become unreplayable at the moment someone tried to play or export it.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("isaaclab", reason="needs the full Isaac install")

from robot_rl.flow_matching.window import (  # noqa: E402
    BASE_POSE_CONVENTION, BASE_POSE_CONVENTIONS, DEFAULT_WINDOW_ANCHOR, base_pose_convention)

#: The exact string every pre-flag run recorded into its ``params/``. Hard-coded on purpose:
#: importing the constant to compare against itself would pass no matter what it became.
_LEGACY = "pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d"


def test_the_default_reproduces_the_legacy_string_exactly():
    assert DEFAULT_WINDOW_ANCHOR == "robot_now"
    assert base_pose_convention() == _LEGACY
    assert base_pose_convention(DEFAULT_WINDOW_ANCHOR) == _LEGACY
    assert BASE_POSE_CONVENTION == _LEGACY, (
        "the bare module constant is still read by older call sites and by any run whose "
        "config predates window_anchor; it must stay the legacy string"
    )


def test_the_two_conventions_are_distinct():
    assert BASE_POSE_CONVENTIONS["plan"] != BASE_POSE_CONVENTIONS["robot_now"], (
        "same string for both arms means a plan-trained tracker loads clean under the "
        "re-anchoring runtime and steers wrong -- the failure the string exists to prevent"
    )
    assert set(BASE_POSE_CONVENTIONS) == {"robot_now", "plan"}


def test_an_unknown_anchor_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown window_anchor"):
        base_pose_convention("plan_frame")      # plausible typo, must not resolve
    with pytest.raises(ValueError, match="unknown window_anchor"):
        base_pose_convention("")


def test_the_cfg_defaults_to_robot_now():
    """A teacher that says nothing gets today's behaviour, bit-identical."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.\
        motion_sequence_cmd_cfg import MotionSequenceCommandCfg

    # Off __dataclass_fields__, not the class attribute: @configclass turns every field into
    # a dataclass field, so `Cfg.window_anchor` raises. And it rewrites the plain default
    # into a default_factory, so `field.default` is MISSING rather than the value -- both
    # ways of asking give a wrong answer instead of an error.
    field = MotionSequenceCommandCfg.__dataclass_fields__["window_anchor"]
    default = field.default_factory() if field.default is dataclasses.MISSING else field.default
    assert default == "robot_now"


def test_provenance_records_the_runs_own_anchor_not_a_global():
    """Two arms train at once, so a module-level constant would stamp one of them wrongly."""
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]
                           / "scripts" / "rsl_rl"))
    from provenance import run_conventions  # noqa: PLC0415

    def _cfg(anchor):
        traj = SimpleNamespace(fm_snap_every=12, ego_anchor_every=12, fm_update_every=12)
        if anchor is not None:
            traj.window_anchor = anchor
        return SimpleNamespace(commands=SimpleNamespace(traj_ref=traj))

    assert run_conventions(_cfg("robot_now"))["base_pose_convention"] == _LEGACY
    assert run_conventions(_cfg("plan"))["base_pose_convention"] == BASE_POSE_CONVENTIONS["plan"]
    # The case that protects the talos run: a config from before the field existed.
    assert run_conventions(_cfg(None))["base_pose_convention"] == _LEGACY


def test_a_pre_flag_run_still_passes_its_own_convention_check(tmp_path):
    """The regression that would silently strand every existing run."""
    import sys
    from types import SimpleNamespace

    import yaml

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]
                           / "scripts" / "rsl_rl"))
    from provenance import RUN_CONVENTIONS_FILE, check_run_conventions  # noqa: PLC0415

    params = tmp_path / "params"
    params.mkdir()
    # Exactly what a run trained before this branch wrote: no window_anchor key.
    (params / RUN_CONVENTIONS_FILE).write_text(yaml.safe_dump({
        "base_pose_convention": _LEGACY,
        "fm_snap_every": 12, "ego_anchor_every": 12, "fm_update_every": 12,
    }))
    traj = SimpleNamespace(fm_snap_every=12, ego_anchor_every=12, fm_update_every=12)
    check_run_conventions(str(tmp_path),
                          SimpleNamespace(commands=SimpleNamespace(traj_ref=traj)))


def test_a_plan_run_is_refused_by_the_robot_now_code_path(tmp_path):
    """And the converse: the two arms cannot be crossed."""
    import sys
    from types import SimpleNamespace

    import yaml

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]
                           / "scripts" / "rsl_rl"))
    from provenance import RUN_CONVENTIONS_FILE, check_run_conventions  # noqa: PLC0415

    params = tmp_path / "params"
    params.mkdir()
    (params / RUN_CONVENTIONS_FILE).write_text(yaml.safe_dump({
        "base_pose_convention": BASE_POSE_CONVENTIONS["plan"],
        "window_anchor": "plan",
        "fm_snap_every": 12, "ego_anchor_every": 12, "fm_update_every": 12,
    }))
    traj = SimpleNamespace(fm_snap_every=12, ego_anchor_every=12, fm_update_every=12,
                           window_anchor="robot_now")
    with pytest.raises(ValueError, match="base_pose_convention"):
        check_run_conventions(str(tmp_path),
                              SimpleNamespace(commands=SimpleNamespace(traj_ref=traj)))


def test_the_deploy_runtime_serves_both_arms():
    """One build plays a tracker from either training arm; the ARTIFACT picks the arm."""
    import importlib.util
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_gen_tracker_probe", root / "transfer" / "deploy" / "gen_tracker.py")
    # Import the module text rather than the package, to avoid dragging onnxruntime in.
    src = (root / "transfer" / "deploy" / "gen_tracker.py").read_text(encoding="utf-8")
    assert "_BASE_POSE_CONVENTION_ARMS" in src
    for conv, arm in BASE_POSE_CONVENTIONS.items():
        assert arm in src, f"the runtime has no table entry for {conv}"
    assert 'if self._window_anchor == "plan":' in src, (
        "the plan arm must short-circuit _serve_nodes; that line IS the odometry saving"
    )
    assert spec is not None


def _stub(anchor: str):
    """A ``_StubCommand`` from the sibling window test, switched to ``anchor``.

    Reuses that fixture rather than growing another: it is the maintained stand-in for the
    command's look-ahead surface, and a second copy here would drift from it.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_reference_window import _StubCommand  # noqa: PLC0415

    cmd = _StubCommand(seed=3)
    cmd.cfg.window_anchor = anchor
    return cmd


def _both_arms(shift):
    """Build the two arms over one reference, differing only in the anchor ORIGIN."""
    import torch
    import warp as wp

    now = _stub("robot_now")
    plan = _stub("plan")
    # Same yaw on both anchors, so the frames coincide and the origin is the only difference.
    plan._plan_anchor_pos = wp.to_torch(now.robot.data.root_pos_w) + shift
    plan._plan_anchor_quat = wp.to_torch(now.robot.data.root_quat_w).clone()
    return now, plan


def test_the_two_arms_produce_DIFFERENT_observations():
    """A flag that changes nothing is worse than none: it reads as an A/B and is inert."""
    import torch

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd \
        import MotionSequenceCommand

    offsets = torch.tensor([0, 4, 8])
    now, plan = _both_arms(torch.tensor([0.3, -0.2, 0.05]))
    a = MotionSequenceCommand.reference_window_obs(now, offsets, None)
    b = MotionSequenceCommand.reference_window_obs(plan, offsets, None)
    assert a.shape == b.shape, "the arms must be the same WIDTH -- that is why the string exists"
    assert not torch.allclose(a, b), "window_anchor='plan' changed nothing; the flag is inert"


def test_the_plan_arm_moves_the_position_block_by_exactly_the_anchor_offset():
    """And it is the RIGHT difference, on the right 3 numbers."""
    import torch
    import warp as wp
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat

    from robot_rl.flow_matching.window import window_dim
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd \
        import MotionSequenceCommand

    offsets = torch.tensor([0, 4, 8])
    shift = torch.tensor([0.3, -0.2, 0.05])
    now, plan = _both_arms(shift)

    a = MotionSequenceCommand.reference_window_obs(now, offsets, None)
    b = MotionSequenceCommand.reference_window_obs(plan, offsets, None)
    nj = len(now.robot.joint_names)
    d = window_dim(nj, len(now.resolve_ee_frames(None)), None)
    a = a.view(a.shape[0], len(offsets), d)
    b = b.view(b.shape[0], len(offsets), d)

    # Moving the anchor by +shift must move the reported reference by exactly -shift,
    # rotated into the (shared) yaw frame.
    q = yaw_quat(wp.to_torch(now.robot.data.root_quat_w))
    expect = -quat_apply_inverse(q, shift.expand(a.shape[0], 3))
    delta = b[..., 0:3] - a[..., 0:3]
    for k in range(len(offsets)):
        assert torch.allclose(delta[:, k, :], expect, atol=1e-5), (
            f"offset {k}: anchor moved by {shift.tolist()}, so the reported position should "
            f"move by minus that; got {delta[:, k, :][0].tolist()}"
        )
    # Everything after base_pose's 9 numbers is anchor-invariant.
    assert torch.allclose(a[..., 9:], b[..., 9:], atol=1e-6), (
        "only base_pose references the anchor: base_vel is a pure rotation (same yaw here) "
        "and the joint / ee blocks are frame-invariant"
    )
