# Copyright 2026 Zachary Olkin. All rights reserved.
"""``_progress_terms`` against a synthetic recording whose answer is known in closed form.

The gate and the reducer are covered by ``test_progress_reward.py``; this covers the GRADER,
which is where every priority-one failure mode of this reward actually lives:

* the successor join -- a velocity is a difference of consecutive ``anchor_pos``, and rows
  arrive in ENV-MAJOR order, so joining by row position rather than by ``(uid, rep + 1)``
  differences two unrelated episodes and returns a plausible speed;
* the yaw frame -- ``anchor_quat`` is XYZW, and reading it as WXYZ gives a rotation that is
  wrong by a bounded amount, i.e. a velocity error that looks like a mediocre policy;
* ``dt`` -- one re-plan is ``replan_k * CONTROL_DT`` = 0.24 s, and a wrong dt rescales every
  velocity uniformly, which no distribution check would flag;
* the last re-plan of every episode, which has NO successor and must be dropped rather than
  differenced against a fabricated displacement;
* ``is_last``, which decides where the one-off terminal bonus lands.

The recording is built so the right answer is exactly zero error: the robot walks at the
commanded speed, along its own heading, on both episodes.
"""

from __future__ import annotations

import importlib.util
import pathlib

import h5py
import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_itr", _HERE / "scripts" / "rsl_rl" / "inspect_terrain_reward.py")


@pytest.fixture(scope="module")
def itr():
    """The grader module, loaded by path (``scripts/`` is not a package)."""
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def _xyzw(yaw: float) -> list[float]:
    """A yaw-only quaternion in XYZW -- the order ``anchor_quat`` is stored in."""
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]


@pytest.fixture(scope="module")
def rollout(tmp_path_factory, itr):
    """Two episodes, 4 re-plans each, rows deliberately in env-major (scrambled) order.

    Episode 0 walks +x at yaw 0; episode 1 walks +y at yaw +90 deg. Both travel exactly
    ``speed * dt`` per re-plan, so a correct grader reports zero velocity error on both --
    and episode 1 only does so if the yaw rotation and the XYZW read are both right.
    """
    dt = itr.DEFAULT_REPLAN_K * itr.CONTROL_DT
    speed = 1.25
    step = speed * dt
    uid, rep, pos, quat, cmd = [], [], [], [], []
    # Env-major: all of env 0's re-plan 0, then env 1's re-plan 0, then re-plan 1, ...
    for r in range(4):
        for e in (0, 1):
            uid.append(e); rep.append(r)
            if e == 0:
                pos.append([r * step, 0.0, 0.0]); quat.append(_xyzw(0.0))
            else:
                pos.append([0.0, r * step, 0.0]); quat.append(_xyzw(np.pi / 2.0))
            cmd.append(speed)
    path = tmp_path_factory.mktemp("prog") / "rollout.h5"
    n = len(uid)
    with h5py.File(path, "w") as f:
        # A real recording pre-allocates and declares how many rows it actually wrote.
        f.attrs["num_windows_written"] = n
        f.create_dataset("anchor_pos", data=np.asarray(pos, dtype=np.float32))
        f.create_dataset("anchor_quat", data=np.asarray(quat, dtype=np.float32))
        f.create_dataset("cmd_future",
                         data=np.asarray(cmd, dtype=np.float32).reshape(n, 1, 1))
        f.create_dataset("episode_uid", data=np.asarray(uid, dtype=np.int64))
        f.create_dataset("replan_index", data=np.asarray(rep, dtype=np.int64))
    return str(path), np.asarray(uid, np.int64), np.asarray(rep, np.int64), speed


def _grade(itr, rollout, *, reached=(1.0, 0.0), success_const=10.0):
    """Grade every row of the fixture with a perfect terrain reward."""
    path, uid, rep, _speed = rollout
    rows = np.arange(uid.size)
    goal = np.where(uid == 0, reached[0], reached[1]).astype(np.float64)
    return itr._progress_terms(
        path, rows, uid, rep,
        reward_terrain=np.ones(uid.size, dtype=np.float64),
        reached_goal=goal,
        n_replans=np.full(uid.size, 4, dtype=np.float64),
        sharpness=2.0, success_sharpness=4.0, vel_sigma=0.5,
        success_const=success_const, replan_k=itr.DEFAULT_REPLAN_K)


def test_a_robot_walking_at_the_commanded_speed_has_zero_velocity_error(itr, rollout):
    """The closed-form case. Non-zero here means the join, the frame or dt is wrong -- and
    episode 1 (walking +y at yaw +90) is the one that catches a WXYZ read of `anchor_quat`
    or a missing yaw rotation, since its world displacement is entirely lateral."""
    got = _grade(itr, rollout)
    v = got["valid"].astype(bool)
    assert v.any()
    np.testing.assert_allclose(got["vel_err_mps"][v], 0.0, atol=1e-6)
    np.testing.assert_allclose(got["vel_track"][v], 1.0, atol=1e-6)


def test_the_last_replan_of_every_episode_is_dropped(itr, rollout):
    """It has no successor, so its displacement would be fabricated. Exactly one row per
    episode must be invalid, and it must be the one at `rep == n_replans - 1`."""
    _p, uid, rep, _s = rollout
    got = _grade(itr, rollout)
    v = got["valid"].astype(bool)
    assert int((~v).sum()) == 2, f"expected one dropped row per episode, got {(~v).sum()}"
    assert set(rep[~v].tolist()) == {3}
    assert set(uid[~v].tolist()) == {0, 1}


def test_the_successor_is_joined_BY_KEY_not_by_row_order(itr, rollout):
    """The fixture is env-major, so row i+1 belongs to the OTHER episode. A positional join
    would difference episode 0's position against episode 1's and report a large error on
    every row -- which reads as a policy that cannot track command."""
    _p, uid, rep, _s = rollout
    got = _grade(itr, rollout)
    v = got["valid"].astype(bool)
    # Row 0 is (uid 0, rep 0) and row 1 is (uid 1, rep 0). Positionally differencing them
    # gives |(0,step,0) - (0,0,0)| / dt = the full speed laterally, hence vx ~ 0 and an
    # error of the whole commanded speed. Assert we are nowhere near that.
    assert got["vel_err_mps"][v].max() < 1e-6


def test_the_terminal_bonus_lands_only_on_the_last_replan_of_a_SUCCESSFUL_episode(itr, rollout):
    """`is_last` comes from the episodes table's `n_replans`, never from the graded rows --
    counting chunks from a sample measured 3 against a true ~48 and made the per-chunk bonus
    16x too large. Episode 0 reaches the goal and episode 1 does not."""
    _p, uid, rep, _s = rollout
    got = _grade(itr, rollout)
    fired = got["success"] > 0.0
    assert fired.tolist() == ((uid == 0) & (rep == 3)).tolist()
    # A perfect terrain reward means the worst chunk is perfect, so the gate is open and the
    # bonus is the full constant.
    assert got["success"][fired][0] == pytest.approx(10.0, rel=1e-6)


def test_the_success_gate_collapses_on_the_episode_WORST_chunk(itr, rollout):
    """One destructive re-plan must cost the success bonus almost all of its value, even
    though every other chunk was perfect -- that is what `min` over the episode buys, and it
    is why stairs-down dismounts stopped reading as successes."""
    path, uid, rep, _s = rollout
    rows = np.arange(uid.size)
    terrain = np.ones(uid.size, dtype=np.float64)
    terrain[(uid == 0) & (rep == 1)] = float(np.exp(-2.0))       # one bad chunk, penalty 2
    got = itr._progress_terms(
        path, rows, uid, rep, reward_terrain=terrain,
        reached_goal=(uid == 0).astype(np.float64),
        n_replans=np.full(uid.size, 4, dtype=np.float64),
        sharpness=2.0, success_sharpness=4.0, vel_sigma=0.5,
        success_const=10.0, replan_k=itr.DEFAULT_REPLAN_K)
    fired = got["success"] > 0.0
    assert fired.any()
    assert got["success"][fired][0] < 0.1, "the worst chunk did not gate the bonus"


def test_the_velocity_error_is_against_the_NOMINAL_command(itr, rollout):
    """Halving the commanded speed while the robot keeps walking must show up as an error of
    exactly the difference -- the reward tracks what was ASKED, so a speed-mismatch data knob
    is an input corruption to reject rather than a new command to follow."""
    path, uid, rep, speed = rollout
    with h5py.File(path, "r+") as f:
        f["cmd_future"][:, 0, 0] = speed * 0.5
    try:
        got = _grade(itr, rollout)
        v = got["valid"].astype(bool)
        np.testing.assert_allclose(got["vel_err_mps"][v], speed * 0.5, atol=1e-6)
    finally:
        with h5py.File(path, "r+") as f:
            f["cmd_future"][:, 0, 0] = speed
