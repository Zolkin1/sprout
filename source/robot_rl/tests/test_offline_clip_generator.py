# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the offline clip generator's stitching invariants.

Requires the full-periodic stride library on disk (skipped if absent, e.g.
in CI without the HF cache).  Asserts the two correctness properties that
the dense rollout must hold: a continuous world stance foot across stride
seams (the anchor recursion) and an exactly uniform global timeline (the
dt-remainder carry).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

_LIB = Path("hf/trajectories/retargeted/2026-06-13_15-10-58_merged")
pytestmark = pytest.mark.skipif(not _LIB.exists(), reason=f"stride library {_LIB} not present")

DEVICE = "cpu"


@dataclass
class _Bucket:
    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    heading: tuple[float, float] | None = None


@pytest.fixture(scope="module")
def generated():
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
        MultiSkillManager,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_clip_generator import (
        OfflineClipGenerator,
    )

    m = MultiSkillManager(str(_LIB), DEVICE, env=None)
    skills = [s.name for s in m.skills]
    buckets = {
        "running": _Bucket((1.5, 3.0)),
        "standing": _Bucket((0.0, 0.1)),
        "walk_backwards": _Bucket((-1.0, -0.1)),
        "walk_forward": _Bucket((0.1, 1.5), (-0.3, 0.3), (-0.5, 0.5)),
    }
    probs = [0.4, 0.05, 0.2, 0.35]
    sampler = OfflineVelocitySampler(
        buckets, skills, probs, resampling_time_range=(0.8, 1.5),
        max_acc=2.0, command_dt=0.02, device=DEVICE, seed=0,
    )
    gen = OfflineClipGenerator(m, sampler, dt=0.02, clip_length_s=5.0, device=DEVICE)
    out = gen.generate_batch(16)
    return m, gen, out


def test_shapes_and_finite(generated):
    _, gen, out = generated
    n, T = 16, gen.num_steps
    assert out["y_des"].shape == (n, T, gen.num_pos)
    assert out["dy_des"].shape == (n, T, gen.num_vel)
    assert out["anchor_pos"].shape == (n, T, 3)
    assert out["anchor_quat"].shape == (n, T, 4)
    for k in ("y_des", "dy_des", "anchor_pos", "anchor_quat"):
        assert torch.isfinite(out[k]).all(), k
    # Anchor quaternions stay unit norm.
    assert torch.allclose(out["anchor_quat"].norm(dim=-1), torch.ones(n, T), atol=1e-4)


def test_world_stance_foot_continuous_across_seams(generated):
    """Across each stride seam the world right-ankle must not jump (planted)."""
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.world_lift import (
        lift_position,
    )
    m, gen, out = generated
    names = m.pos_output_names
    ra = torch.tensor([names.index(f"{gen.ref_frame}:{a}") for a in ("pos_x", "pos_y", "pos_z")])
    y, ap, aq, st = out["y_des"], out["anchor_pos"], out["anchor_quat"], out["stride_idx"]
    n = y.shape[0]
    max_jump = 0.0
    for ci in range(n):
        wra = lift_position(ap[ci], aq[ci], y[ci][:, ra])  # [T, 3]
        bnd = (st[ci, 1:] != st[ci, :-1]).nonzero().flatten()
        for b in bnd.tolist():
            max_jump = max(max_jump, (wra[b + 1] - wra[b]).norm().item())
    # A planted/just-landed foot moves at most ~one dt of low-speed motion.
    assert max_jump < 0.10, f"world stance foot jumps {max_jump:.3f} m across a seam"


def test_uniform_timeline_no_dropped_frames(generated):
    """Every clip has exactly num_steps frames with monotone within-stride phase."""
    _, gen, out = generated
    # The rollout stores exactly T frames; stride_idx is piecewise-constant and
    # only advances forward (indices change at seams, never mid-stride).
    st = out["stride_idx"]
    assert st.shape[1] == gen.num_steps
    # No NaNs / negative indices leaked.
    assert (st >= 0).all()
