# Copyright 2026 Zachary Olkin. All rights reserved.

"""Standing spawn + opening command hold: the settle window before the first plan."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

from robot_rl.tasks.manager_based.robot_rl.mdp.events.resets import (  # noqa: E402
    _yaw_only_quat_xyzw,
)


def test_the_standing_spawn_keeps_heading_and_drops_lean() -> None:
    """A standing spawn must be UPRIGHT: a clip's first frame may carry a lean.

    Writing that lean with a standing joint pose would put the robot down already tilted --
    a state the reference-relative tilt termination grades, and one the settle window cannot
    recover from because it is the spawn itself.
    """
    from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, wrap_to_pi

    roll = torch.tensor([0.30, -0.20, 0.0])
    pitch = torch.tensor([-0.15, 0.25, 0.0])
    yaw = torch.tensor([1.10, -2.40, 0.7])
    # XYZW, in BOTH directions. This build's `quat_from_euler_xyz` returns xyzw -- verified
    # by hand: for roll 0.3 / pitch -0.15 / yaw 1.1 the scalar part is 0.8347 and it is the
    # LAST element. That is also what `write_root_pose_to_sim_index` takes, which is why
    # `_yaw_only_quat_xyzw` is written for that order; swapping here would double-convert.
    xyzw = quat_from_euler_xyz(roll, pitch, yaw)
    # Pin the convention by ROUND TRIP rather than by asserting a component: this build's
    # `quat_from_euler_xyz` returns XYZW (the scalar part is last), which is also what
    # `write_root_pose_to_sim_index` takes and what `_yaw_only_quat_xyzw` is written for.
    # Swapping to WXYZ here would double-convert and the lean would appear to survive.
    rr, pp, yy = euler_xyz_from_quat(xyzw)
    assert torch.allclose(wrap_to_pi(rr - roll), torch.zeros(3), atol=1e-5)
    assert torch.allclose(wrap_to_pi(pp - pitch), torch.zeros(3), atol=1e-5)

    out = _yaw_only_quat_xyzw(xyzw)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(out[:, :2], torch.zeros(3, 2), atol=1e-7), "x/y survived"

    r2, p2, y2 = euler_xyz_from_quat(out)
    assert torch.allclose(wrap_to_pi(r2), torch.zeros(3), atol=1e-6), "roll survived"
    assert torch.allclose(wrap_to_pi(p2), torch.zeros(3), atol=1e-6), "pitch survived"
    # Heading is what places the robot along the tile's traversal; it must NOT change.
    assert torch.allclose(wrap_to_pi(y2 - yaw), torch.zeros(3), atol=1e-6)


class _Cmd:
    """The two fields `_hold` reads, with no simulator."""

    def __init__(self, hold_s: float, elapsed: list[int], dt: float = 0.02) -> None:
        from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import (
            motion_sequence_cmd,
        )

        self._fn = motion_sequence_cmd.MotionSequenceCommand._hold
        self.cfg = type("C", (), {"cmd_hold_s": hold_s})()
        self.env = type("E", (), {
            "step_dt": dt,
            "episode_length_buf": torch.tensor(elapsed),
        })()

    def __call__(self, vel):
        return self._fn(self, vel)


def test_a_zero_hold_is_a_bit_exact_no_op() -> None:
    """Every existing task leaves this at 0.0 and must be unchanged."""
    hold = _Cmd(0.0, [0, 5, 100])
    vel = torch.randn(3, 3)
    assert torch.equal(hold(vel), vel)
    horizon = torch.randn(3, 7, 3)
    assert torch.equal(hold(horizon), horizon)


def test_the_hold_zeroes_the_current_command_only_while_it_is_active() -> None:
    """0.3 s at a 0.02 s control step is 15 steps."""
    hold = _Cmd(0.3, [0, 14, 15, 40])
    vel = torch.ones(4, 3)
    out = hold(vel)
    assert bool((out[0] == 0).all()) and bool((out[1] == 0).all()), "hold ended early"
    assert bool((out[2] == 1).all()), "hold ran one step long"
    assert bool((out[3] == 1).all())


def test_the_whole_horizon_is_zeroed_while_the_hold_is_active() -> None:
    """Not per-step, and that is the deliberate choice -- MEASURED, not argued.

    Per-step masking is the literal description of the future (entry k is step s+1+k, and
    some of those are past the hold), but the plan spans 1.24 s at F=62, so a hold four
    times shorter barely moves it. Opening-plan travel over the 62 nodes on the shipped
    0.3 s hold: 0.361 m per-step, 0.071 m whole-horizon, 0.925 m under a normal command.
    Per-step gave "stand a moment, then accelerate", which is not a settle.
    """
    hold = _Cmd(0.3, [10])            # 15-step hold; step 10 is inside it
    horizon = torch.ones(1, 8, 3)     # would describe steps 11 .. 18
    assert hold(horizon)[0, :, 0].tolist() == [0.0] * 8

    # ...and nothing is held once it expires, on the same call shape.
    expired = _Cmd(0.3, [15])
    assert expired(torch.ones(1, 8, 3))[0, :, 0].tolist() == [1.0] * 8


def test_the_adjuster_holds_long_enough_to_cover_the_first_replan() -> None:
    """The point of the window: the opening plan is drawn while the command is still zero."""
    from robot_rl.tasks.manager_based.robot_rl.g1 import g1_adjuster_env_cfg as cfg

    env = cfg.G1AdjusterEnvCfg()
    hold_steps = cfg.STAND_SETTLE_S / cfg.CONTROL_DT
    assert env.commands.traj_ref.stand_start is True
    assert env.commands.traj_ref.cmd_hold_s == cfg.STAND_SETTLE_S
    assert hold_steps > env.commands.traj_ref.fm_update_every, (
        "the settle window is shorter than one re-plan interval, so the first plan the robot "
        "must follow is drawn before it has settled"
    )
    assert math.isclose(hold_steps, 15.0)
