# Copyright 2026 Zachary Olkin. All rights reserved.

"""The free-form velocity command: the yaw-rate P controller and the held (vx, 0, wz) triple.

The FM fine-tune's terrain sections carry no clips, so the joystick command that conditions
the generator and feeds the policy's ``command_window`` is manufactured by
:class:`FmTrackingCommand` instead of read from the library. Four things about that are easy
to get wrong and silent when wrong, and each has a test here:

* **The wrap.** A heading error wrapped the wrong way turns the long way round -- which on a
  section a few metres wide is a fall, not a slow correction. Pinned against hand-computed
  bearings, both sides of the +/-pi branch.
* **The gain.** A pure P on BEARING has effective cross-track gain ``kp / lookahead``, not
  ``kp``, so the two knobs are coupled. Stated in the docstring and measured here, because a
  reader tuning ``kp`` alone would otherwise be surprised by a factor of five.
* **The hold.** The whole triple is zero for the opening hold, yaw rate included -- a yaw rate
  applied to a robot standing at rest is as much a step input as a forward speed is.
* **The clip tasks are untouched.** Every accessor is a pass-through with the flag off, which
  is the only thing standing between this and a silent change to five trained tasks.

No simulator: the controller is a pure function, and the command term's methods are driven on
a bare instance (``object.__new__``) carrying only the buffers each one reads -- the same
idiom as ``test_fm_tracking_lookahead.py``, so the arithmetic under test is the shipped
arithmetic.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp", exc_type=ImportError)
pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
wp.init()

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd import (  # noqa: E402
    FREEFORM_VY,
    FmTrackingCommand,
    freeform_yaw_rate,
    wrap_to_pi,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (  # noqa: E402
    MotionSequenceCommand,
)

N_ENVS = 4
STEP_DT = 0.02
#: The shipped defaults, restated here so a test failure names the number that moved.
KP, LOOKAHEAD, WZ_MAX = 1.0, 5.0, 0.68


# ----------------------------------------------------------------------
# The controller, as a pure function
# ----------------------------------------------------------------------

def _yaw_rate(px: float, py: float, yaw: float, *, ox: float = 0.0, oy: float = 0.0,
              heading: float = 0.0, lookahead: float = LOOKAHEAD, kp: float = KP,
              wz_max: float = WZ_MAX) -> float:
    """One scalar call into the batched controller, for the hand-computed cases.

    Args:
        px: Root world x (m).
        py: Root world y (m).
        yaw: Root heading (rad).
        ox: Centreline origin x (m).
        oy: Centreline origin y (m).
        heading: Centreline direction (rad).
        lookahead: Metres down the centreline to the aim point.
        kp: Proportional gain (1/s).
        wz_max: Symmetric output clip (rad/s).

    Returns:
        The commanded yaw rate (rad/s).
    """
    return float(freeform_yaw_rate(
        torch.tensor([[px, py]]), torch.tensor([yaw]),
        torch.tensor([[ox, oy]]), torch.tensor([heading]),
        lookahead, kp, wz_max,
    )[0])


def test_on_the_centreline_pointing_down_it_commands_nothing():
    """The equilibrium. Anything else here is a controller that fights its own setpoint."""
    assert _yaw_rate(0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # And the same 12 m downrange: the aim point RECEDES with the robot, so a section is not
    # "reached" the way mjeval's goal point is and the command does not go quiet part-way.
    assert _yaw_rate(12.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_the_bearing_is_the_hand_computed_one():
    """A metre LEFT of the line, pointing down it, turns RIGHT by ``atan2(-1, L)``.

    The sign is the half of this that a shape check cannot see: a controller with it inverted
    is a divergent one, and on a 1.219 m flight that is a fall within a couple of metres.
    """
    got = _yaw_rate(0.0, 1.0, 0.0)
    assert got == pytest.approx(math.atan2(-1.0, LOOKAHEAD))
    assert got < 0.0
    # Mirror image, exactly.
    assert _yaw_rate(0.0, -1.0, 0.0) == pytest.approx(-got)


def test_the_law_is_covariant_with_the_section_heading():
    """A section along +y gives the same command as one along +x, with the frame rotated.

    The tile's ``frame_yaw`` is 0.0 on the shipped axis-aligned manifests, so a controller
    that quietly assumed world +x would pass every test on today's terrain and steer wrong on
    the first rotated tile.
    """
    along_x = _yaw_rate(0.0, 1.0, 0.0, heading=0.0)
    # Rotate everything by +90 deg: the robot's 1 m of LEFT offset becomes -1 m of world x.
    along_y = _yaw_rate(-1.0, 0.0, math.pi / 2.0, heading=math.pi / 2.0)
    assert along_y == pytest.approx(along_x)


def test_the_cross_track_gain_is_kp_over_the_lookahead():
    """Shortening ``L`` by 5x stiffens the loop by 5x -- the coupling the docstring claims.

    Measured at a small offset, where ``atan2(-e, L) ~= -e / L`` holds to well under a
    percent; the point is the RATIO, which is what a reader tuning ``kp`` alone gets wrong.
    """
    near = _yaw_rate(0.0, 0.1, 0.0, lookahead=1.0)
    far = _yaw_rate(0.0, 0.1, 0.0, lookahead=5.0)
    assert near / far == pytest.approx(5.0, rel=0.01)
    # Both are the small-angle prediction to within a percent.
    assert far == pytest.approx(-0.1 / 5.0, rel=0.01)


def test_the_output_is_clipped_at_wz_max():
    """100 m off the line asks for 1.52 rad/s; the command is 0.68, the library's own maximum."""
    assert _yaw_rate(0.0, 100.0, 0.0) == pytest.approx(-WZ_MAX)
    assert _yaw_rate(0.0, -100.0, 0.0) == pytest.approx(WZ_MAX)


def test_zero_gain_disables_steering_exactly():
    """``kp = 0`` is the off switch: the command is exactly 0.0, not a small residue."""
    assert _yaw_rate(0.0, 3.0, 1.1, kp=0.0) == 0.0


def test_the_error_wraps_the_short_way_across_pi():
    """Facing a hair either side of exactly-backwards turns opposite ways.

    This is the property a modulo wrap gets wrong at the branch. The SIGN at exactly
    antiparallel is deliberately not asserted -- it follows the sign of the floating-point
    ``sin`` and ``float32(pi)`` rounds above pi, so ``wrap_to_pi(pi)`` comes back negative.
    Only the magnitude means anything there, and the docstring says so.
    """
    just_under = _yaw_rate(0.0, 0.0, math.pi - 1e-6, wz_max=10.0)
    just_over = _yaw_rate(0.0, 0.0, -(math.pi - 1e-6), wz_max=10.0)
    assert just_under < -3.0 and just_over > 3.0
    assert just_under == pytest.approx(-just_over)
    for a in (math.pi, -math.pi, 3.0 * math.pi, -5.0 * math.pi):
        assert abs(float(wrap_to_pi(torch.tensor(a)))) == pytest.approx(math.pi, abs=1e-5)
    # Well inside one turn, the wrap is exact and folds a full turn away.
    assert float(wrap_to_pi(torch.tensor(2.0 * math.pi + 0.4))) == pytest.approx(0.4, abs=1e-6)


def test_a_non_positive_lookahead_is_refused():
    """At L = 0 the aim point IS the robot's projection and the bearing means nothing."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="look-ahead"):
            _yaw_rate(0.0, 1.0, 0.0, lookahead=bad)


def test_it_is_vectorised_over_envs():
    """Every env carries its own pose, centreline and heading -- this is a command term."""
    pos = torch.tensor([[0.0, 1.0], [0.0, -1.0], [3.0, 0.0], [-1.0, 0.5]])
    yaw = torch.tensor([0.0, 0.0, 0.0, 0.3])
    origin = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.5]])
    heading = torch.tensor([0.0, 0.0, 0.0, math.pi / 2.0])
    batched = freeform_yaw_rate(pos, yaw, origin, heading, LOOKAHEAD, KP, WZ_MAX)
    assert batched.shape == (4,)
    for i in range(4):
        one = freeform_yaw_rate(pos[i:i + 1], yaw[i:i + 1], origin[i:i + 1],
                                heading[i:i + 1], LOOKAHEAD, KP, WZ_MAX)
        assert float(batched[i]) == pytest.approx(float(one[0]))


# ----------------------------------------------------------------------
# The command term
# ----------------------------------------------------------------------

class _BareFfCommand(FmTrackingCommand):
    """``FmTrackingCommand`` with ``device`` / ``num_envs`` as plain class attributes.

    The base exposes both as read-only properties off the env; shadowing them lets a bare
    instance exist without one. Every method under test is inherited unchanged.
    """

    device = torch.device("cpu")
    num_envs = N_ENVS
    #: ``CommandTerm.__del__`` touches this; a bare instance never ran the ctor that sets it.
    _debug_vis_handle = None


class _FakeManager:
    """The two clip-library reads the inherited command-velocity accessors make."""

    num_steps = 40
    #: This double stands for a library WITH motion, which is what the pass-through arms below
    #: are about. ``_update_command`` now gates the whole clip half on it, so a fake that did
    #: not answer would make every test here take the clip-free path by accident. The clip-free
    #: arm is driven against a REAL manager in ``test_freeform_clipless_command.py``.
    has_clips = True

    def __init__(self, target_vel: torch.Tensor, has: bool = True) -> None:
        """Hold one ``[C, T, 3]`` command table.

        Args:
            target_vel: The library's ``target_vel``, ``[num_clips, num_steps, 3]``.
            has: What :meth:`has_target_vel` answers.
        """
        self.lib = SimpleNamespace(target_vel=target_vel)
        self._has = has

    def has_target_vel(self) -> bool:
        """Does the library carry a per-step commanded velocity?"""
        return self._has

    def gather_target_vel(self, clip_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """``[N, 3]`` command at ``(clip_id, step)``."""
        return self.lib.target_vel[clip_id, step]

    def gather(self, table: torch.Tensor, clip_id: torch.Tensor,
               steps: torch.Tensor) -> torch.Tensor:
        """``[N, K, 3]`` gather over a ``[N, K]`` index grid."""
        return table[clip_id, steps]


def _cmd(freeform: bool, **over) -> FmTrackingCommand:
    """A bare command carrying only what the command-velocity path reads.

    Args:
        freeform: The ``freeform_command`` flag under test.
        **over: Overrides folded into the cfg namespace.

    Returns:
        The bare instance.
    """
    cmd = object.__new__(_BareFfCommand)
    cfg = dict(freeform_command=freeform, freeform_speed_min=0.6, freeform_speed_max=2.6,
               freeform_cmd_hold_s=0.5, freeform_steer_kp=KP, freeform_lookahead_m=LOOKAHEAD,
               freeform_wz_max=WZ_MAX, reference_source="fm")
    cfg.update(over)
    cmd.cfg = SimpleNamespace(**cfg)

    cmd._ff_cmd = torch.zeros(N_ENVS, 3)
    cmd._ff_speed = torch.zeros(N_ENVS)
    cmd._ff_origin_w = torch.zeros(N_ENVS, 2)
    cmd._ff_heading_w = torch.zeros(N_ENVS)
    cmd._ff_hold_steps = int(round(0.5 / STEP_DT))

    # The clip half, for the pass-through tests: a distinct value per (clip, step, channel).
    table = (torch.arange(3)[None, None, :].float()
             + torch.arange(_FakeManager.num_steps)[None, :, None] * 10.0
             + torch.arange(N_ENVS)[:, None, None] * 1000.0)
    cmd.manager = _FakeManager(table)
    cmd.clip_id = torch.arange(N_ENVS)
    cmd._step = torch.tensor([0, 5, 11, 39])

    root_quat = torch.zeros(N_ENVS, 4)
    root_quat[:, 3] = 1.0                                    # identity, xyzw
    cmd.robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=wp.from_torch(torch.zeros(N_ENVS, 3).contiguous()),
        root_quat_w=wp.from_torch(root_quat.contiguous()),
    ))
    cmd.env = SimpleNamespace(
        step_dt=STEP_DT, scene=None,
        episode_length_buf=torch.zeros(N_ENVS, dtype=torch.long),
    )
    return cmd


def test_the_flag_is_off_by_default():
    """The five clip tasks must not acquire a manufactured command by accident."""
    pytest.importorskip("isaaclab")
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.fm_tracking_cmd_cfg import (
        FmTrackingCommandCfg,
    )
    cfg = FmTrackingCommandCfg()
    assert cfg.freeform_command is False
    # The asked-for defaults, pinned so a silent retune is visible in the diff.
    assert cfg.freeform_lookahead_m == 5.0
    assert cfg.freeform_cmd_hold_s == 0.5
    assert cfg.freeform_wz_max == 0.68


def test_the_clip_referenced_tasks_keep_the_clip_command():
    """The three tasks that build an ``FmTrackingCommandCfg`` over a CLIP library keep it, and
    the free-form one is the single task that manufactures its own.

    Named rather than registry-walked, deliberately: enumerating the registry is exactly what
    ``test_freeform_env_cfg.test_exactly_one_registered_task_manufactures_its_own_command``
    already does, so a second copy here would add no coverage and a second thing to keep in
    step.  What this one is for is the other direction -- naming the trained tasks whose
    command must not move, in the file that owns the flag's behaviour.  It cannot see a task
    added later; that is the registry test's job, and the positive arm below is what stops
    this list from quietly asserting nothing if the flag's default ever flips.
    """
    pytest.importorskip("isaaclab")
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_finetune_env_cfg import (
        G1FinetuneFreeformEnvCfg,
        G1FinetuneTerrainDepthEnvCfg,
    )
    from robot_rl.tasks.manager_based.robot_rl.g1.g1_tracker_env_cfg import (
        G1FmTrackerTerrainDepthEnvCfg,
        G1FmTrackerTerrainEnvCfg,
    )
    for cfg_cls in (G1FinetuneTerrainDepthEnvCfg, G1FmTrackerTerrainEnvCfg,
                    G1FmTrackerTerrainDepthEnvCfg):
        assert cfg_cls().commands.traj_ref.freeform_command is False, cfg_cls.__name__
    assert G1FinetuneFreeformEnvCfg().commands.traj_ref.freeform_command is True


def test_with_the_flag_off_every_accessor_is_the_clip_one():
    """All four overrides are pass-throughs, value for value, against the inherited methods.

    Not a smoke test: these four are read by the reward, the ``command_window`` observation,
    the recorder's ``target_vel`` column and the generator's own conditioning, so a
    difference here is a different trained policy.
    """
    cmd = _cmd(freeform=False)
    offsets = torch.tensor([0, 5, 10])
    assert FmTrackingCommand.command_velocity_available.fget(cmd) is True
    torch.testing.assert_close(FmTrackingCommand.get_command_velocity(cmd),
                               MotionSequenceCommand.get_command_velocity(cmd))
    torch.testing.assert_close(FmTrackingCommand.get_command_velocity_horizon(cmd, 7),
                               MotionSequenceCommand.get_command_velocity_horizon(cmd, 7))
    torch.testing.assert_close(FmTrackingCommand.get_future_command_velocity(cmd, offsets),
                               MotionSequenceCommand.get_future_command_velocity(cmd, offsets))
    # And the manufactured buffer is never written on that path.
    assert torch.all(cmd._ff_cmd == 0.0)


def test_with_the_flag_off_a_library_without_target_vel_still_reports_missing():
    """The availability override must not paper over a genuinely command-less clip library."""
    cmd = _cmd(freeform=False)
    cmd.manager._has = False
    assert FmTrackingCommand.command_velocity_available.fget(cmd) is False


def test_the_manufactured_command_needs_no_library_target_vel():
    """A manifest-only ``.h5`` (terrain, zero clips) is the free-form library, by design."""
    cmd = _cmd(freeform=True)
    cmd.manager._has = False
    assert FmTrackingCommand.command_velocity_available.fget(cmd) is True


def test_the_command_is_held_across_every_offset_and_the_whole_horizon():
    """13 copies of one row, not a trajectory -- what both deploy runtimes serve.

    The clip path serves ``target_vel`` sampled per offset, a future no robot has. Holding it
    removes that train/deploy mismatch, and this is the test that says so.
    """
    cmd = _cmd(freeform=True)
    cmd._ff_cmd = torch.tensor([[1.0, 0.0, 0.2], [2.0, 0.0, -0.3],
                                [0.5, 0.0, 0.0], [2.6, 0.0, 0.68]])
    horizon = FmTrackingCommand.get_command_velocity_horizon(cmd, 62)
    window = FmTrackingCommand.get_future_command_velocity(cmd, torch.tensor([0, 1, 5, 50]))
    assert horizon.shape == (N_ENVS, 62, 3)
    assert window.shape == (N_ENVS, 4, 3)
    for f in range(62):
        torch.testing.assert_close(horizon[:, f], cmd._ff_cmd)
    for k in range(4):
        torch.testing.assert_close(window[:, k], cmd._ff_cmd)
    torch.testing.assert_close(FmTrackingCommand.get_command_velocity(cmd), cmd._ff_cmd)
    # Both are broadcast VIEWS (stride 0 down the offset axis), which is what makes the
    # generator's [N, F, 3] conditioning free. ``mdp.command_window`` flattens the result, so
    # the flatten must survive a non-contiguous tensor -- ``reshape`` copies where ``view``
    # would raise, and this is the line that says so.
    flat = window.reshape(N_ENVS, -1)
    assert flat.shape == (N_ENVS, 4 * 3)
    torch.testing.assert_close(flat[:, :3], cmd._ff_cmd)
    torch.testing.assert_close(flat[:, 9:], cmd._ff_cmd)


def test_the_opening_hold_is_zero_and_then_steps_to_the_draw():
    """Zero for the whole hold -- yaw rate included -- then the episode's constant speed.

    A yaw rate applied to a robot standing at rest is as much a step input as a forward speed
    is, which is why the hold covers the whole triple rather than ``vx`` alone.
    """
    cmd = _cmd(freeform=True)
    cmd._ff_speed = torch.tensor([1.0, 1.5, 2.0, 2.5])
    # Every env a metre left of its centreline, so a non-zero yaw rate is on offer.
    root = torch.zeros(N_ENVS, 3)
    root[:, 1] = 1.0
    cmd.robot.data.root_pos_w = wp.from_torch(root.contiguous())
    assert cmd._ff_hold_steps == 25
    cmd.env.episode_length_buf = torch.tensor([0, 24, 25, 60])

    FmTrackingCommand._update_freeform_command(cmd)
    held, live = cmd._ff_cmd[:2], cmd._ff_cmd[2:]
    assert torch.all(held == 0.0)
    torch.testing.assert_close(live[:, 0], cmd._ff_speed[2:])
    assert torch.all(live[:, 2] < 0.0)          # steering right, back toward the line
    # vy is exactly zero on every env, held or not: the deployed controller ships
    # v_y_scale: 0.0 and the library measures |vy| max = 0.0000 over 3500 stair clips.
    assert FREEFORM_VY == 0.0
    assert torch.all(cmd._ff_cmd[:, 1] == FREEFORM_VY)


def test_a_zero_hold_commands_the_speed_from_the_first_step():
    """``freeform_cmd_hold_s = 0`` is the off switch for the hold, not a one-step hold."""
    cmd = _cmd(freeform=True)
    cmd._ff_hold_steps = 0
    cmd._ff_speed = torch.full((N_ENVS,), 1.7)
    FmTrackingCommand._update_freeform_command(cmd)
    torch.testing.assert_close(cmd._ff_cmd[:, 0], cmd._ff_speed)


def test_the_speed_is_drawn_once_per_episode_and_the_command_zeroed():
    """One constant per episode, inside the band -- and the previous episode's speed is gone.

    ``episode_length_buf`` is not cleared until every manager has been reset, so a consumer
    reading between the reset and the next ``_update_command`` would otherwise see the old
    episode's speed rather than the opening hold.
    """
    cmd = _cmd(freeform=True, freeform_speed_min=0.6, freeform_speed_max=2.6)
    cmd._ff_cmd = torch.full((N_ENVS, 3), 9.0)
    ids = torch.tensor([0, 2])
    FmTrackingCommand._reset_freeform_command(cmd, ids)
    assert torch.all((cmd._ff_speed[ids] >= 0.6) & (cmd._ff_speed[ids] <= 2.6))
    assert torch.all(cmd._ff_cmd[ids] == 0.0)
    # Untouched envs keep theirs: the draw is per env, at that env's own reset.
    assert torch.all(cmd._ff_cmd[torch.tensor([1, 3])] == 9.0)


def test_setting_a_section_is_refused_when_the_command_is_off():
    """A silent no-op is the wrong answer to a caller that meant to steer."""
    cmd = _cmd(freeform=False)
    with pytest.raises(RuntimeError, match="freeform_command=True"):
        FmTrackingCommand.set_freeform_section(
            cmd, torch.tensor([0]), torch.zeros(1, 2), torch.zeros(1))


def test_a_declared_section_is_what_the_controller_steers_to():
    """``set_freeform_section`` is the interface the terrain lane drives; it must bite."""
    cmd = _cmd(freeform=True)
    cmd._ff_hold_steps = 0
    cmd._ff_speed = torch.full((N_ENVS,), 1.0)
    # Put every env's centreline one metre to the LEFT of where they stand, running +x.
    FmTrackingCommand.set_freeform_section(
        cmd, torch.arange(N_ENVS),
        torch.tensor([[0.0, 1.0]]).repeat(N_ENVS, 1), torch.zeros(N_ENVS))
    FmTrackingCommand._update_freeform_command(cmd)
    # Robot at the origin, line at y = +1 -> steer LEFT, by atan2(+1, L).
    assert torch.allclose(cmd._ff_cmd[:, 2],
                          torch.full((N_ENVS,), math.atan2(1.0, LOOKAHEAD)))


def test_the_command_is_rebuilt_before_anything_reads_it():
    """``_update_command`` builds the free-form command FIRST.

    The clip half advances the step and the generator reads its command horizon; both run
    inside the same ``_update_command``, so a command rebuilt after them would condition the
    generator on the PREVIOUS step's value while the observation carried this one.
    """
    order: list[str] = []
    cmd = _cmd(freeform=True)
    cmd.fm_enabled = False
    cmd._update_freeform_command = lambda: order.append("freeform")
    cmd._build_clip_reference = lambda: order.append("clip")
    cmd._apply_source = lambda source: order.append("apply")
    cmd.compute_measured_output = lambda pos, quat: order.append("measured")
    cmd._zeros3 = cmd._ident_quat = None
    cmd.y_act = cmd.y_des = cmd.dy_act = cmd.dy_des = None
    cmd.clf = SimpleNamespace(compute_vdot=lambda *a: (0.0, 0.0))

    FmTrackingCommand._update_command(cmd)
    assert order[:2] == ["freeform", "clip"]
