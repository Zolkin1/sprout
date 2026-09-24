# Copyright 2026 Zachary Olkin. All rights reserved.
"""The eval's tilt must measure TILT, not yaw, whichever slot order the buffer arrives in.

``cos(tilt)`` is the world-z component of the body's up-axis, ``1 - 2(x^2 + y^2)`` for a unit
quaternion -- so the answer depends entirely on which two slots hold ``x`` and ``y``.
``(w, x, y, z)`` puts them at 1 and 2, ``(x, y, z, w)`` at 0 and 1, and reading the wrong pair
returns ``1 - 2(y^2 + z^2)``, which for a pure yaw of theta is exactly ``cos(theta)``.

**This was live and it cost a whole family's success rate.** ``robot.data`` hands several
buffers back as warp arrays and the shim converts them without reordering, so the slot order is
a property of the buffer, not of the reader. Measured on a v6 clip-arm eval: ``flat`` -- the
only family whose robot actually rotates, terrain traversals being world-pinned replays --
reported a **42.7%** fall rate against a **1.0%** height-based one, with median "tilt" 58.9 deg,
p95 179.5 deg, and **correlation 0.893** between the recorded angle and the clip's own commanded
yaw. Success on flat read 0.375 where arrival was 0.750. Every terrain family was unaffected,
which is exactly why it survived: the contaminated families were the ones nobody was watching.

It cannot be fixed by picking a convention and asserting it in a comment, because the buffer's
order is not this file's to choose. So it is RESOLVED at runtime against
``projected_gravity_b`` -- a vector, hence unambiguous -- and a sample that matches neither
reading is refused rather than guessed.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

_EVAL = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "rsl_rl" / "eval"
sys.path.insert(0, str(_EVAL))

import accumulate  # noqa: E402


class _FakeData:
    """The two buffers the resolution needs, and nothing else."""

    def __init__(self, quat: torch.Tensor, gravity: torch.Tensor) -> None:
        self.root_quat_w = quat
        self.projected_gravity_b = gravity


def _yaw_quat(theta: float, order: str) -> torch.Tensor:
    """A pure YAW rotation -- zero real tilt -- in the requested slot order."""
    w, z = math.cos(theta / 2.0), math.sin(theta / 2.0)
    q = (w, 0.0, 0.0, z) if order == "wxyz" else (0.0, 0.0, z, w)
    return torch.tensor([q], dtype=torch.float32)


def _pitch_quat(theta: float, order: str) -> torch.Tensor:
    """A pure PITCH rotation -- real tilt of ``theta`` -- in the requested slot order."""
    w, y = math.cos(theta / 2.0), math.sin(theta / 2.0)
    q = (w, 0.0, y, 0.0) if order == "wxyz" else (0.0, y, 0.0, w)
    return torch.tensor([q], dtype=torch.float32)


def _upright_gravity() -> torch.Tensor:
    """Gravity in the body frame of an UPRIGHT robot: straight down, whatever its yaw."""
    return torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)


def _pitched_gravity(theta: float) -> torch.Tensor:
    """Gravity in the body frame of a robot pitched by ``theta``."""
    return torch.tensor([[math.sin(theta), 0.0, -math.cos(theta)]], dtype=torch.float32)


def _acc() -> accumulate.EpisodeAccumulator:
    """An accumulator with only the field the resolution touches."""
    a = object.__new__(accumulate.EpisodeAccumulator)
    a._quat_wxyz = None
    return a


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_a_yawing_upright_robot_reads_zero_tilt_in_either_order(order):
    """THE bug: 143 deg of yaw reported 179.7 deg of "tilt" and scored as a fall."""
    a = _acc()
    d = _FakeData(_yaw_quat(math.radians(143.0), order), _upright_gravity())
    tilt = math.degrees(float(torch.acos(a._upright_cos(d))[0]))
    assert tilt == pytest.approx(0.0, abs=1.0), f"{order}: yaw leaked into tilt ({tilt:.1f} deg)"


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_a_real_tilt_is_still_measured(order):
    """The fix must not buy yaw-immunity by making the column blind -- 70 deg is a FALL."""
    a = _acc()
    want = math.radians(70.0)
    d = _FakeData(_pitch_quat(want, order), _pitched_gravity(want))
    tilt = math.degrees(float(torch.acos(a._upright_cos(d))[0]))
    assert tilt == pytest.approx(70.0, abs=1.0), f"{order}: real tilt lost ({tilt:.1f} deg)"


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_the_resolution_is_sticky(order):
    """Resolved ONCE. Re-deciding per step would let one bad frame flip the whole column."""
    a = _acc()
    a._upright_cos(_FakeData(_yaw_quat(math.radians(143.0), order), _upright_gravity()))
    assert a._quat_wxyz is (order == "wxyz")
    before = a._quat_wxyz
    a._upright_cos(_FakeData(_yaw_quat(math.radians(20.0), order), _upright_gravity()))
    assert a._quat_wxyz is before


def test_the_undecidable_sample_is_a_90_degree_pitch_not_an_upright_robot():
    """Worth pinning because the intuition runs the other way.

    The two readings coincide exactly when ``q[0]^2 == q[2]^2``, which for ``(w,x,y,z)`` means
    ``|w| == |y|`` -- a ~90 deg pitch. An UPRIGHT robot is maximally discriminating (+1 one way,
    -1 the other), so the first frame of an episode resolves immediately. The undecidable case
    is a robot already on its side.
    """
    a = _acc()
    a._upright_cos(_FakeData(_pitch_quat(math.radians(90.0), "wxyz"), _pitched_gravity(math.radians(90.0))))
    assert a._quat_wxyz is None, "a 90 deg pitch cannot name the convention"

    b = _acc()
    b._upright_cos(_FakeData(_yaw_quat(0.0, "wxyz"), _upright_gravity()))
    assert b._quat_wxyz is True, "an upright sample DOES name it"


def test_a_buffer_matching_neither_reading_is_refused():
    """A non-quaternion in that slot would otherwise produce plausible tilt numbers forever."""
    a = _acc()
    # Discriminating (q0 != q2) but far from upright under BOTH readings, while gravity says
    # upright -- so neither slot order explains it.
    d = _FakeData(torch.tensor([[0.7, 0.5, 0.5, 0.1]], dtype=torch.float32), _upright_gravity())
    with pytest.raises(RuntimeError, match="NEITHER quaternion convention"):
        a._upright_cos(d)


def test_the_yaw_misread_really_does_return_the_yaw_angle():
    """Pins the MECHANISM, so the next reader does not have to rediscover why flat was 42.7%.

    Reading `(x, y, z, w)` at the `(w, x, y, z)` slots evaluates `1 - 2(y^2 + z^2)`, which for a
    pure yaw of theta is cos(theta) exactly -- the recorded "tilt" IS the yaw.
    """
    for deg in (30.0, 74.0, 143.0):
        q = _yaw_quat(math.radians(deg), "xyzw")
        wrong = float((1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1.0, 1.0)[0])
        assert math.degrees(math.acos(wrong)) == pytest.approx(deg, abs=0.5)
