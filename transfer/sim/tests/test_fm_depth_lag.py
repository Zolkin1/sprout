# Copyright 2026 Zachary Olkin. All rights reserved.

"""The generator must see a depth frame as old as the one it trained on.

The FM trains on frames delayed 20-60 ms -- the depth pipeline's compute plus the ROS hop --
sampled per window from ``DEPTH_LATENCY_PROBS``. Both simulators fed it a LIVE frame, so
every sim2sim and play run sat off-distribution in exactly the variable that model was fitted
to. Hardware needs no synthetic lag at all: its latency is physical and already in the
frames, so adding one there would double-count. Hence the sim runners, not ``transfer/deploy``.

What these pin:

* the image and its POSE move together -- a lagged image unprojected with the CURRENT pose is
  displaced by the egomotion, the registration error the xyz channels exist to remove;
* lag 0 is byte-identical to the behaviour before any of this existed, so a graph with no
  latency distribution (every one exported before the model, including the pair on
  HuggingFace) is unaffected;
* the warm-up repeats the oldest frame rather than zero-filling, because a zero depth row is
  a VALID reading -- a surface at the near plane -- and would show the generator a wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SIM.parent))

from sim.gen_tracker_runner import FmDepthLag  # noqa: E402
from deploy.fm_session import default_depth_lag_steps  # noqa: E402


def _state(i: int) -> dict:
    """A state dict whose depth and pose both encode the step, so a swap is visible."""
    return {"depth": np.full((2, 4), float(i)),
            "fm_camera_pose": [(np.full(3, float(i)), np.eye(3) * float(i))]}


def test_lag_zero_leaves_the_state_untouched():
    """The default for a pre-latency graph must cost nothing and change nothing."""
    lag = FmDepthLag(0)
    st = _state(7)
    lag.apply(st)
    assert "fm_depth" not in st, (
        "lag 0 introduced an fm_depth key that was not there; _fm_depth_reading treats that "
        "as authoritative, so this would silently reroute the generator's input"
    )
    assert st["fm_camera_pose"][0][0].tolist() == [7.0, 7.0, 7.0]


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_the_generator_is_served_the_frame_from_lag_steps_ago(steps):
    """The actual claim: frame at t IS the frame built at t - lag, pose with it."""
    lag = FmDepthLag(steps)
    for i in range(12):
        st = _state(i)
        lag.apply(st)
        want = max(i - steps, 0)          # clamped during the warm-up
        assert st["fm_depth"][0, 0] == want, (
            f"step {i} with lag {steps}: served frame {st['fm_depth'][0, 0]}, wanted {want}")
        assert st["fm_camera_pose"][0][0][0] == want, (
            "the POSE did not travel with the image -- unprojecting a lagged frame with the "
            "live pose displaces every pixel by the egomotion")


def test_the_warmup_repeats_the_oldest_frame_and_never_zero_fills():
    """A zero depth row is a surface at the near plane, not 'no data'."""
    lag = FmDepthLag(3)
    st = _state(1)                        # first ever step, buffer holds one frame
    lag.apply(st)
    assert st["fm_depth"][0, 0] == 1.0, "warm-up must repeat the oldest frame held"
    assert not np.allclose(st["fm_depth"], 0.0), "zero-filled: that is a wall at the near plane"


def test_the_tracker_keeps_the_live_frame():
    """Only the generator is delayed; state['depth'] is the tracker's own observation."""
    lag = FmDepthLag(2)
    for i in range(6):
        st = _state(i)
        lag.apply(st)
        assert st["depth"][0, 0] == i, "the tracker's own depth observation was delayed too"


def test_a_split_sampling_delays_the_generators_own_grid():
    """With a second cast the generator's array is fm_depth, and that is what must lag."""
    lag = FmDepthLag(2)
    for i in range(6):
        st = _state(i)
        st["fm_depth"] = np.full((2, 9), float(100 + i))     # different width on purpose
        lag.apply(st)
        assert st["fm_depth"][0, 0] == 100 + max(i - 2, 0)
        assert st["depth"][0, 0] == i


@pytest.mark.parametrize("probs,want", [
    (None, 0), ((), 0), ((1.0, 0.0, 0.0, 0.0), 0),
    ((0.0, 0.10, 0.40, 0.50), 2),        # the shipped distribution: 48 ms -> 2 steps
    ((0.0, 0.0, 0.0, 1.0), 3),
])
def test_the_default_lag_is_the_rounded_mean_of_the_trained_probs(probs, want):
    assert default_depth_lag_steps(probs) == want


def test_the_two_derivations_agree():
    """robot_rl's copy and the torch-free deploy mirror must not drift."""
    torch_side = pytest.importorskip("robot_rl.flow_matching.config",
                                     reason="needs the isaaclab env")
    for probs in (None, (), (0.0, 0.10, 0.40, 0.50), (0.0, 1.0), (0.0, 0.0, 0.0, 1.0)):
        assert torch_side.default_depth_lag_steps(probs) == default_depth_lag_steps(probs)
