# Copyright 2026 Zachary Olkin. All rights reserved.
"""The strict-tracking termination is EXECUTED here, against WARP-backed robot data.

It ends episodes during collection, and the whole test suite passed on a version of it that
raised `TypeError` on its first call: `robot.data`'s buffers are warp arrays in a termination,
and the accessor names were copied from `eval.accumulate`, which reads them through a shim
that converts on every attribute access. Nothing that merely imports the function or reads the
config can see that -- so this drives it.

Cost of not driving it, measured: two 5-iteration fine-tunes launched on two GPUs, collected
nothing, and died at the grade stage on a missing episode table.
"""

from __future__ import annotations

import pytest
import torch
import warp as wp

from robot_rl.tasks.manager_based.robot_rl.mdp.terminations import terminations as term
from robot_rl.tasks.manager_based.robot_rl.mdp.tracking_error import BOOTSTRAP_SPEC

# `wp.from_torch` needs a warp device table, which only exists after init. Harmless if the
# module was already initialised by another import.
wp.init()

_N = 4
_NJ = 29


class _WarpData:
    """``robot.data`` as a termination really sees it: warp arrays, not tensors."""

    def __init__(self, base_z: torch.Tensor) -> None:
        """Store every buffer the termination reads, as warp."""
        pos = torch.zeros(_N, 3)
        pos[:, 2] = base_z
        quat = torch.zeros(_N, 4)
        quat[:, 3] = 1.0                                   # xyzw identity
        self.root_pos_w = wp.from_torch(pos, dtype=wp.vec3)
        self.root_quat_w = wp.from_torch(quat, dtype=wp.vec4)
        self.root_link_lin_vel_w = wp.from_torch(torch.zeros(_N, 3), dtype=wp.vec3)
        self.root_link_ang_vel_w = wp.from_torch(torch.zeros(_N, 3), dtype=wp.vec3)
        self.joint_pos = wp.from_torch(torch.zeros(_N, _NJ))


class _Robot:
    """Articulation stand-in."""

    def __init__(self, base_z: torch.Tensor) -> None:
        """Hold the warp-backed data view."""
        self.data = _WarpData(base_z)


class _Cmd:
    """Command stand-in serving a flat reference at ``ref_z`` and a validity mask."""

    def __init__(self, ref_z: float, valid: torch.Tensor | None) -> None:
        """Store the served node and the validity gate."""
        self._ref_z = ref_z
        if valid is not None:
            self._pred_valid = valid

    def get_future_world_base(self, offsets):
        """``([N,K,3], [N,K,4], [N,K,3], [N,K,3])`` -- one node, flat, at ``ref_z``."""
        k = int(offsets.numel())
        pos = torch.zeros(_N, k, 3)
        pos[..., 2] = self._ref_z
        quat = torch.zeros(_N, k, 4)
        quat[..., 3] = 1.0
        return pos, quat, torch.zeros(_N, k, 3), torch.zeros(_N, k, 3)

    def get_future_joint_state(self, offsets):
        """``([N,K,nj], [N,K,nj])``."""
        k = int(offsets.numel())
        return torch.zeros(_N, k, _NJ), torch.zeros(_N, k, _NJ)


class _Env:
    """The three attributes the termination touches."""

    def __init__(self, robot, cmd) -> None:
        """Wire the command manager and the scene lookup."""
        self.num_envs = _N
        self.device = "cpu"
        self.command_manager = type("M", (), {"get_term": lambda _s, _n: cmd})()
        self.scene = {"robot": robot}


def _run(base_z, ref_z=0.80, valid=None):
    """Drive the termination once and return its verdict."""
    z = torch.as_tensor(base_z, dtype=torch.float32)
    env = _Env(_Robot(z), _Cmd(ref_z, valid))
    return term.fm_tracking_violation_termination(env)


def test_it_RUNS_on_warp_backed_data() -> None:
    """The regression. A `TypeError` here is the bug that wasted two GPU launches."""
    out = _run([0.80, 0.80, 0.80, 0.80], valid=torch.ones(_N, dtype=torch.bool))
    assert out.shape == (_N,) and out.dtype == torch.bool
    assert not bool(out.any()), "a robot exactly on its reference must not terminate"


def test_it_fires_past_the_height_bound_and_not_inside_it() -> None:
    """0.12 m is the bound; 0.11 must pass and 0.13 must fire."""
    b = float(BOOTSTRAP_SPEC.max_height_m)
    assert b == pytest.approx(0.12)
    out = _run([0.80, 0.80 + b - 0.01, 0.80 + b + 0.01, 0.80 - b - 0.01],
               valid=torch.ones(_N, dtype=torch.bool))
    assert out.tolist() == [False, False, True, True], out.tolist()


def test_the_validity_gate_suppresses_it() -> None:
    """Before the first re-plan there is no served node to be far from."""
    far = [0.80 + 0.30] * _N
    assert _run(far, valid=torch.ones(_N, dtype=torch.bool)).all()
    assert not _run(far, valid=torch.zeros(_N, dtype=torch.bool)).any()


def test_a_command_with_no_validity_gate_is_REFUSED() -> None:
    """The clip arm never allocates `_pred_valid`; skipping the gate would fire from step 0."""
    with pytest.raises(ValueError, match="no `_pred_valid`"):
        _run([0.80] * _N, valid=None)
