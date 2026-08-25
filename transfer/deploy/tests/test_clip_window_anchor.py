# Copyright 2026 Zachary Olkin. All rights reserved.

"""The clip path must serve the anchor convention the TRACKER declares.

``base_pose``'s position half is measured from different things under the two conventions
(``robot_rl/flow_matching/window.py``): ``robot_now`` measures against the robot this step,
so the block is a tracking error; ``plan`` measures against the robot's root at the last
latch, so the block is the reference's position in the plan frame -- a sawtooth that ramps
over ``fm_update_every`` ticks with the robot's own commanded displacement and resets.

The two arms are **identical in width and in every other block**, so a tracker trained under
one loads clean against the other and merely steers wrong. The generator path has always
honoured the declaration (``_serve_nodes``); the clip path used the measured root
unconditionally, which meant every clip-driven run of a ``plan`` tracker -- the obelisk debug
mode and ``gen_tracker_runner.py --clip`` alike -- served a window the tracker never trained
on. These tests pin the fix.

The invariance test is the load-bearing one: it is what makes C++ clip playback possible at
all, since the C++ deploy runtime has no odometry.

Numpy + h5py only, no ONNX -- the policy is built attribute-by-attribute, because everything
under test is arithmetic that runs long before a graph is consulted.

Run: ``pytest transfer/deploy/tests/test_clip_window_anchor.py``
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types

import h5py
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # transfer/
from deploy import gen_tracker as gt  # noqa: E402
from deploy.clip_reference import ClipReference  # noqa: E402

BASE = "pelvis_link"
JOINTS = ["j0", "j1"]
OFFSETS = [0, 1, 4]
K = 12  #: the shipped pair's fm_update_every


def _quat_yaw(yaw: float) -> np.ndarray:
    """Pure-yaw ``(x, y, z, w)`` quaternion."""
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=np.float64)


def _make_clip(path: str, T: int = 60, dt: float = 0.02) -> None:
    """A clip that actually goes somewhere: marching downrange while turning."""
    pos_names = ([f"{BASE}:pos_{a}" for a in "xyz"]
                 + [f"{BASE}:ori_{a}" for a in "xyzw"]
                 + [f"joint:{n}" for n in JOINTS])
    vel_names = ([f"{BASE}:pos_{a}" for a in "xyz"]
                 + [f"{BASE}:ori_{a}" for a in "xyz"]
                 + [f"joint:{n}" for n in JOINTS])
    y = np.zeros((1, T, len(pos_names)), dtype=np.float64)
    dy = np.zeros((1, T, len(vel_names)), dtype=np.float64)
    for s in range(T):
        y[0, s, 0:3] = [0.02 * s, 0.004 * s, 0.78 + 0.001 * s]
        y[0, s, 3:7] = _quat_yaw(0.01 * s)
        y[0, s, 7:9] = [0.05 * s, -0.03 * s]
        dy[0, s, 0:3] = [1.0, 0.2, 0.0]
        dy[0, s, 3:6] = [0.0, 0.0, 0.5]
    with h5py.File(path, "w") as f:
        f.create_dataset("y_des", data=y)
        f.create_dataset("dy_des", data=dy)
        f.create_dataset("target_vel", data=np.tile([1.0, 0.0, 0.5], (1, T, 1)))
        f.attrs["pos_output_names"] = json.dumps(pos_names)
        f.attrs["vel_output_names"] = json.dumps(vel_names)
        f.attrs["dt"] = json.dumps(dt)


def _policy(window_anchor: str = "plan", k: int = K):
    """A GenTrackerPolicy carrying only what the clip window path reads.

    Built without ``load()`` on purpose: every line under test is arithmetic that runs long
    before a graph is opened, and stubbing an ONNX session would only add ways for the test
    to pass for the wrong reason.
    """
    p = object.__new__(gt.GenTrackerPolicy)
    p._window_anchor = window_anchor
    p._fm_update_every_from_meta = True
    p.fm_update_every = k
    p._clip_step = -1
    p._clip_anchor_pos = None
    p._clip_anchor_quat = None
    p._fm_to_isaac = list(range(len(JOINTS)))
    p.reference_blocks = ("base_pose", "base_vel", "joint_pos")
    p.ctx = types.SimpleNamespace(target_num_extra_frames=0)
    p.tracker_onnx = "tracker.onnx"
    return p


def _serve(clip: ClipReference, policy, robot, steps: int, ego_every: int) -> np.ndarray:
    """Drive the clip exactly as a consumer does, and return the served windows.

    Args:
        clip: The player.
        policy: The bare policy from :func:`_policy`.
        robot: ``t -> (pos, quat_xyzw)`` the robot's measured world pose.
        steps: How many control steps to serve.
        ego_every: The clip's own re-anchor cadence.

    Returns:
        ``[steps, len(OFFSETS) * D]`` served reference windows.
    """
    out = []
    for t in range(steps):
        pos, quat = robot(t)
        if clip.ego_anchor and t % ego_every == 0:
            clip.reanchor_ego(t, pos, quat)
        anchor = policy._clip_window_anchor({"root_pos": pos, "root_quat": quat})
        out.append(policy._reference_window(clip.reference_at(t, OFFSETS)["window"], *anchor))
    return np.stack(out)


def _straight(t):
    """A robot walking downrange, turning as it goes."""
    return np.array([0.02 * t, 0.004 * t, 0.80]), _quat_yaw(0.01 * t)


def _elsewhere(t):
    """The SAME motion, 100 m away and facing another direction.

    A RIGID transform of :func:`_straight`, so it varies the world frame without varying the
    robot's motion within it. Weaker than :func:`_stalled` as a probe, and kept for exactly
    that reason: a window that failed this one would be broken in a much more basic way.
    """
    p, _ = _straight(t)
    yaw0 = 1.3
    c, s = math.cos(yaw0), math.sin(yaw0)
    return (np.array([100.0 + c * p[0] - s * p[1], -50.0 + s * p[0] + c * p[1], p[2]]),
            _quat_yaw(yaw0 + 0.01 * t))


def _stalled(t):
    """A robot that never moves -- the reference walks away from it.

    The DIFFERENT-motion probe, and the one that separates the two cadences. A rigid transform
    of the walking trajectory leaves the robot's step-to-step displacement unchanged, so the
    egomotion a mismatched cadence leaks in is the same for both and the comparison comes back
    equal for the wrong reason. Stalling changes that displacement, so it does not.
    """
    return np.array([0.0, 0.0, 0.80]), _quat_yaw(0.0)


@pytest.fixture()
def clip_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tiny.h5")
        _make_clip(path)
        yield path


def _clip(path: str, every: int) -> ClipReference:
    return ClipReference(path, JOINTS, [], BASE, ego_anchor=True, ego_anchor_every=every)


def test_plan_window_is_odometry_free_at_the_matched_cadence(clip_path):
    """THE property that lets the C++ runtime play clips: no world pose reaches the window.

    With the clip's placement and the window's anchor both solved from the robot on the same
    step, the robot's position and heading cancel exactly and ``base_pose`` reduces to
    ``Rz(-yaw_latch) (p_clip(s+o) - p_clip(s_latch))`` -- a pure function of the clip. Two
    robots 100 m apart and facing different directions must therefore be served bit-identical
    windows.
    """
    ref = _serve(_clip(clip_path, K), _policy(), _straight, 3 * K, K)
    for name, robot in (("rigidly displaced", _elsewhere), ("not moving at all", _stalled)):
        got = _serve(_clip(clip_path, K), _policy(), robot, 3 * K, K)
        assert np.allclose(ref, got, atol=1e-12, rtol=0.0), (
            f"the plan-anchored clip window must not depend on the robot ({name}); worst "
            f"difference {np.abs(ref - got).max():.3e}")


def test_a_mismatched_cadence_leaks_the_robot_back_in(clip_path):
    """The negative control -- otherwise the test above could pass for the wrong reason.

    Re-anchoring the clip every step while latching the window every K puts up to K-1 steps
    of the robot's own egomotion into the block. It looks exactly like a healthy window,
    which is why the cadences are checked rather than trusted.
    """
    a = _serve(_clip(clip_path, 1), _policy(), _straight, 3 * K, 1)
    b = _serve(_clip(clip_path, 1), _policy(), _stalled, 3 * K, 1)
    worst = np.abs(a - b).max()
    assert worst > 0.05, (
        "a mismatched cadence must NOT be odometry-free -- the guard would be pointless. "
        f"Worst difference {worst:.3e} m over {K - 1} steps of leaked egomotion.")


def test_plan_base_pose_is_a_sawtooth_not_a_zero(clip_path):
    """What the tracker actually trained on: ramp over K ticks, reset at the latch.

    Under the old behaviour (measured root every step, clip re-anchored every step) offset 0's
    position sat at zero forever. A tracker trained on the ramp reads that as "the reference is
    exactly on me", every step, however far behind it actually is.
    """
    served = _serve(_clip(clip_path, K), _policy(), _straight, 2 * K + 1, K)
    pos0 = served[:, 0:3]                       # base_pose position at offset 0
    assert np.allclose(pos0[0], 0.0, atol=1e-12), "the latch step must sit at the origin"
    assert np.allclose(pos0[K], 0.0, atol=1e-12), "and so must the next latch"
    ramp = np.linalg.norm(pos0[1:K], axis=1)
    assert np.all(np.diff(ramp) > 0.0), "between latches it must GROW, monotonically"
    assert ramp[-1] > 0.1, f"the ramp must be visible, not numerical: reached {ramp[-1]:.4f} m"


def test_robot_now_still_measures_against_the_robot(clip_path):
    """The other arm is unchanged: no latch, no cadence, the measured root every step."""
    p = _policy(window_anchor="robot_now")
    served = _serve(_clip(clip_path, 1), p, _straight, 5, 1)
    assert p._clip_anchor_pos is None, "robot_now must not latch anything"
    assert np.allclose(served[:, 0:3], 0.0, atol=1e-12), (
        "with the clip re-anchored onto the robot every step, the tracking error at offset 0 "
        "is zero by construction -- which is exactly why serving this to a plan tracker hides "
        "the ramp it expects")


def test_a_disagreeing_cadence_is_refused(clip_path):
    """A clip whose ego cadence is not the tracker's must raise, naming the fix."""
    p = _policy()
    with pytest.raises(ValueError, match="ego_anchor_every"):
        p.assert_clip_reference_compatible(_clip(clip_path, 1))
    p.assert_clip_reference_compatible(_clip(clip_path, K))          # the matched one is fine


def test_a_world_anchored_clip_is_allowed_but_warned(clip_path, capsys):
    """It is training-matched (a non-ego env sees exactly this) but needs a base POSITION."""
    clip = ClipReference(clip_path, JOINTS, [], BASE, ego_anchor=False)
    _policy().assert_clip_reference_compatible(clip)
    assert "WARN" in capsys.readouterr().out


def test_robot_now_does_not_constrain_the_cadence(clip_path):
    """Under robot_now the anchor is the measured root every step, so any cadence is fine."""
    _policy(window_anchor="robot_now").assert_clip_reference_compatible(_clip(clip_path, 7))


def test_a_plan_tracker_without_a_cadence_is_refused():
    """The latch PERIOD is part of the observation contract, so it cannot be defaulted."""
    p = _policy()
    p._fm_update_every_from_meta = False
    with pytest.raises(ValueError, match="fm_update_every"):
        p._check_clip_anchor()


def test_a_robot_now_tracker_without_a_cadence_is_fine():
    """Nothing latches there, so the cadence genuinely does not matter."""
    p = _policy(window_anchor="robot_now")
    p._fm_update_every_from_meta = False
    p._check_clip_anchor()
