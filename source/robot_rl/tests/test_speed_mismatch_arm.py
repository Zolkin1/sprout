# Copyright 2026 Zachary Olkin. All rights reserved.
"""The speed-mismatch conditioning arm.

The knob offsets the command speed the GENERATOR is conditioned on while the robot is still
asked for the nominal one, so the generator has to plan a box approach at a speed other than
the one it is travelling at. On the free-form path an episode draws one speed and holds it,
so the command channel is constant for a whole run and is close to a redundant input -- and
it is the input a deployed joystick moves most.

The failure that matters is not the arm failing to fire; it is the override LEAKING into
`_ff_cmd`, which is a single buffer feeding the policy's `command_window` observation, the
tracking rewards, the eval's tracking target and the recorded `cmd_future`. That would change
what the robot is ASKED to do rather than what the generator is CONDITIONED on, silently, and
would make the recording's stored conditioning a lie.

The draw and the override are tested NUMERICALLY by binding the real methods to a stub -- they
touch only the config, the four arm buffers and the eligibility gate -- so this needs no
simulator. The placement invariants, which cannot be run without one, are checked against the
source the way the other two arms' are.
"""

from __future__ import annotations

import pathlib
import types

import numpy as np
import pytest
import torch

_CMD = (pathlib.Path(__file__).resolve().parents[3] / "source" / "robot_rl" / "robot_rl"
        / "tasks" / "manager_based" / "robot_rl" / "mdp" / "commands" / "traj_tracking"
        / "fm_tracking_cmd.py")


def _src() -> str:
    """The command module's source, for the placement invariants."""
    return _CMD.read_text()


def _method(name: str):
    """Pull one method's source out of the class and exec it in isolation.

    Binding the REAL implementation rather than restating it is the whole point: a test that
    re-implemented the draw would pass while the shipped one was wrong.
    """
    s = _src()
    start = s.index(f"    def {name}(")
    end = s.index("\n    def ", start + 1)
    body = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in s[start:end].splitlines())
    ns: dict = {"torch": torch}
    exec(body, ns)                                                   # noqa: S102
    return ns[name]


class _Cfg:
    """Just the fields the arm reads."""

    def __init__(self, frac=1.0, rng=0.75, holds=(2, 3)):
        self.fm_speed_mismatch_fraction = frac
        self.fm_speed_mismatch_range_mps = rng
        self.fm_speed_mismatch_replans = holds


class _Stub:
    """A command-shaped object carrying only the arm's own state."""

    def __init__(self, n=64, frac=1.0, rng=0.75, holds=(2, 3), eligible=True, seed=0):
        self.num_envs = n
        self.device = "cpu"
        self.cfg = _Cfg(frac, rng, holds)
        self._speed_mismatch = torch.zeros(n, dtype=torch.bool)
        self._speed_mismatch_offset = torch.zeros(n)
        self._speed_mismatch_left = torch.zeros(n, dtype=torch.long)
        self._perturb_eligible = torch.full((n,), bool(eligible))
        self._speed_gen = torch.Generator(device="cpu")
        self._speed_gen.manual_seed(seed)
        for name in ("_draw_speed_mismatch", "_override_cmd_speed"):
            setattr(self, name, types.MethodType(_method(name), self))

    def all_ids(self) -> torch.Tensor:
        return torch.arange(self.num_envs)


# ------------------------------------------------------------------- the override

def test_the_override_reaches_the_CONDITIONING_and_nothing_else():
    """The override must be applied to `fut_cmd` inside `_run_fm`, AFTER the gather that
    makes it a copy -- never inside `get_command_velocity_horizon`, which is what the policy
    observation, the rewards and the recorded `cmd_future` all read."""
    s = _src()
    run = s[s.index("        fut_cmd = self.get_command_velocity_horizon("):]
    run = run[:run.index("cmd_tokens = build_cmd_tokens")]
    assert "_override_cmd_speed" in run, (
        "the override is not applied where the generator is conditioned")
    horizon = s[s.index("    def get_command_velocity_horizon("):]
    horizon = horizon[:horizon.index("\n    def ")]
    assert "_speed_mismatch" not in horizon, (
        "the override leaked into get_command_velocity_horizon, which serves the POLICY: "
        "that changes what the robot is asked to do, not what the generator is shown")


def test_the_override_does_not_write_through_to_its_input():
    """The copy-vs-view invariant, asserted numerically. `_ff_cmd` is one buffer shared by
    four consumers, so an in-place write here would corrupt the command the robot is given
    and the command the recording stores."""
    st = _Stub(n=8, frac=1.0)
    st._speed_mismatch[:] = True
    st._speed_mismatch_offset[:] = 0.5
    src = torch.ones(8, 4, 3)
    keep = src.clone()
    out = st._override_cmd_speed(src, st.all_ids())
    torch.testing.assert_close(src, keep)                 # the input is untouched
    assert out is not src
    torch.testing.assert_close(out[..., 0], torch.full((8, 4), 1.5))


def test_the_override_touches_only_the_FORWARD_channel_and_only_flagged_envs():
    """It is a speed mismatch, so `vy` and `wz` must be untouched -- and an env the draw did
    not select must come back bit-identical, or the control population is contaminated."""
    st = _Stub(n=6, frac=1.0)
    st._speed_mismatch[:] = torch.tensor([True, False, True, False, False, True])
    st._speed_mismatch_offset[:] = torch.tensor([0.3, 9.9, -0.4, 9.9, 9.9, 0.1])
    src = torch.zeros(6, 3, 3)
    src[..., 0] = 1.0
    out = st._override_cmd_speed(src, st.all_ids())
    np.testing.assert_allclose(out[:, 0, 0].numpy(), [1.3, 1.0, 0.6, 1.0, 1.0, 1.1], atol=1e-6)
    torch.testing.assert_close(out[..., 1], src[..., 1])
    torch.testing.assert_close(out[..., 2], src[..., 2])


def test_the_override_is_a_no_op_when_the_arm_is_off():
    """Off by default, and off means the same tensor back -- not a silent clone, which would
    make every run pay for a knob nobody enabled."""
    st = _Stub(n=4, frac=0.0)
    src = torch.ones(4, 2, 3)
    assert st._override_cmd_speed(src, st.all_ids()) is src


# ----------------------------------------------------------------------- the draw

def test_an_active_hold_is_never_re_rolled_mid_persistence():
    """The hold is the point of the arm -- a one-window mismatch is corrected before the body
    has moved. So an env with a live countdown must keep its offset, unchanged, until it
    expires; re-rolling every re-plan would make the hold length meaningless."""
    st = _Stub(n=1, frac=1.0, holds=(3, 3))
    st._draw_speed_mismatch(st.all_ids())
    assert bool(st._speed_mismatch[0])
    off0 = float(st._speed_mismatch_offset[0])
    left0 = int(st._speed_mismatch_left[0])
    assert left0 == 3
    for k in range(1, left0):
        st._draw_speed_mismatch(st.all_ids())
        assert bool(st._speed_mismatch[0]), f"the hold ended early at tick {k}"
        assert float(st._speed_mismatch_offset[0]) == off0, "the offset was re-rolled"
        assert int(st._speed_mismatch_left[0]) == left0 - k


def test_an_expired_hold_clears_the_flag_even_when_it_does_not_reactivate():
    """An env whose re-roll misses must come back pristine. Leaving the flag set would keep
    it corrupted by a draw that did not select it, and the recording would say so."""
    st = _Stub(n=1, frac=1.0, holds=(1, 1))
    st._draw_speed_mismatch(st.all_ids())
    assert bool(st._speed_mismatch[0])
    st.cfg.fm_speed_mismatch_fraction = 1e-12          # every re-roll now misses
    st._draw_speed_mismatch(st.all_ids())
    assert not bool(st._speed_mismatch[0])
    assert float(st._speed_mismatch_offset[0]) == 0.0
    assert int(st._speed_mismatch_left[0]) == 0


def test_the_hold_length_stays_inside_its_declared_range():
    """`fm_speed_mismatch_replans` is inclusive at both ends, and both ends must be reachable
    -- a randint that dropped the top would silently shorten every hold."""
    st = _Stub(n=4096, frac=1.0, holds=(2, 3))
    st._draw_speed_mismatch(st.all_ids())
    live = st._speed_mismatch_left[st._speed_mismatch]
    assert live.numel() > 0
    assert int(live.min()) == 2 and int(live.max()) == 3


def test_the_offset_is_symmetric_and_inside_the_declared_range():
    """A one-sided offset would be a curriculum -- it would systematically tell the generator
    to go slower (or faster) than the body is, and the arm is meant to be a corruption."""
    st = _Stub(n=8192, frac=1.0, rng=0.75)
    st._draw_speed_mismatch(st.all_ids())
    off = st._speed_mismatch_offset[st._speed_mismatch].numpy()
    assert off.size > 1000
    assert np.abs(off).max() <= 0.75 + 1e-6
    assert abs(float(off.mean())) < 0.05, "the offset distribution is not centred"
    assert (off < 0).mean() > 0.4 and (off > 0).mean() > 0.4


def test_the_arm_is_GATED_by_the_outer_eligibility():
    """`fm_perturb_fraction` guarantees a clean control population of a stated size, in the
    same run and therefore on the same terrain and command distribution. An arm that ignored
    it would contaminate that control."""
    st = _Stub(n=256, frac=1.0, eligible=False)
    st._draw_speed_mismatch(st.all_ids())
    assert not bool(st._speed_mismatch.any()), "an ineligible env was perturbed"


def test_the_draw_is_random_never_strided():
    """Envs map to tiles through a structured grid lookup, so every n-th env risks putting the
    whole arm on one tile and confounding it with terrain -- the same reason the other two
    arms are drawn this way."""
    draw = _src()
    draw = draw[draw.index("    def _draw_speed_mismatch"):]
    draw = draw[:draw.index("\n    def ")]
    assert "torch.rand" in draw
    for strided in ("% ", "arange(0,", "::"):
        assert strided not in draw.replace("torch.arange(self.num_envs", ""), (
            f"the draw looks strided ({strided!r})")


def test_the_arm_is_off_by_default():
    """Every diagnostic arm ships off, so a run that does not ask for it is unaffected."""
    cfg = (_CMD.parent / "fm_tracking_cmd_cfg.py").read_text()
    blk = cfg[cfg.index("fm_speed_mismatch_fraction"):]
    assert blk.split("\n")[0].strip().endswith("= 0.0")


def test_reset_CLEARS_the_hold_rather_than_carrying_it_across_an_episode():
    """A hold surviving a reset would corrupt the opening plans of an env that may no longer
    be eligible -- and the opening plans are the ones that choose the approach mode."""
    s = _src()
    reset = s[s.index("    def reset(self, env_ids=None):"):]
    reset = reset[:reset.index("\n    def ")]
    assert "_speed_mismatch_left" in reset and "= 0" in reset, (
        "reset must clear the countdown")
    # A CALL, not a mention: the comment in `reset` names the function deliberately, to say
    # where the activation does happen.
    assert "self._draw_speed_mismatch(" not in reset, (
        "reset must not DRAW the arm -- activation belongs on the re-plan, where the "
        "countdown is ticked")
    # And the re-plan block must call it, or the arm never fires at all.
    rp = s[s.index("replan_ids = infer_mask.nonzero"):]
    rp = rp[:rp.index("self._run_fm(")]
    assert "self._draw_speed_mismatch(replan_ids)" in rp, (
        "nothing ticks the countdown, so a hold would never expire and the arm would never "
        "activate")


def test_the_recorder_stores_the_flag_AND_the_offset():
    """The flag alone cannot say what the generator was shown, because this corruption is a
    magnitude. `cmd_future` deliberately keeps the TRUE command, so the pair is what makes
    the conditioning recoverable."""
    rec = (pathlib.Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl"
           / "record_rollouts.py").read_text()
    assert "cmd.speed_mismatched()" in rec
    assert "cmd.speed_mismatch_offset()" in rec
