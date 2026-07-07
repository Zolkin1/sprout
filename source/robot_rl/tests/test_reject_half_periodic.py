# Copyright 2026 Zachary Olkin. All rights reserved.

"""The offline clip generator must reject half-periodic libraries.

Half-periodic strides (one stored step + a mirrored half) break the offline
per-full-stride re-anchor and snap/teleport; the generator refuses them with a
clear, trajectory-naming error.  Uses the on-disk half-periodic merged library
(skipped if absent, e.g. CI without the HF cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

_HALF_LIB = Path("hf/trajectories/retargeted/2026-05-11_12-34-50_merged")
pytestmark = pytest.mark.skipif(
    not _HALF_LIB.exists(), reason=f"half-periodic library {_HALF_LIB} not present"
)

DEVICE = "cpu"


@dataclass
class _B:
    lin_vel_x: tuple = (0.0, 1.0)
    lin_vel_y: tuple = (0.0, 0.0)
    ang_vel_z: tuple = (0.0, 0.0)
    heading: tuple | None = None


def test_half_periodic_library_is_rejected():
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
        MultiSkillManager,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_clip_generator import (
        OfflineClipGenerator,
    )

    m = MultiSkillManager(str(_HALF_LIB), DEVICE, env=None)
    skills = [s.name for s in m.skills]
    sampler = OfflineVelocitySampler(
        {n: _B() for n in skills}, skills, [1.0 / len(skills)] * len(skills),
        resampling_time_range=(1.5, 4.0), max_acc=2.0, command_dt=0.02, device=DEVICE, seed=0,
    )
    with pytest.raises(ValueError, match="half-periodic"):
        OfflineClipGenerator(m, sampler, dt=0.02, clip_length_s=2.0, device=DEVICE)
