# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-segment clip placement at the command level, without IsaacLab or a simulator.

Drives the REAL :class:`MotionSequenceCommand` re-anchor path -- ``_advance_step`` /
``_reanchor_ego_envs`` / ``_reanchor_clip_ids`` / ``_refresh_current_placement`` /
``_window_placement`` -- on a ``SimpleNamespace`` carrying only the attributes those methods touch,
over a synthetic 3-segment clip (``[flat approach, terrain-pinned, flat exit]``, the shape every MB /
DP terrain clip is stamped with).

The behavior under test: a clip has ONE terrain-cell anchor but a mask that varies along its
timeline, so each segment needs its own placement.  Previously a single placement was solved from
the CURRENT step's mask and applied to every frame, so an ego-anchored approach dragged the
terrain-pinned frames off the terrain, and after a segment boundary the terrain frames kept the
approach's drifted placement until the next cadence tick.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from motion_gen.world_lift import _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking import motion_sequence_cmd as msc
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (
    MotionSequenceCommand,
)

DEVICE = "cpu"
T_CLIP = 60
SEAM_IN, SEAM_OUT = 20, 40          # terrain-pinned segment [20, 40)
PERIOD = 12                         # ego re-anchor cadence (cfg.ego_anchor_every)
CELL = torch.tensor([[3.0, 5.0, 0.0]])


def _mask_schedule() -> torch.Tensor:
    """``[1, T]`` per-step mask: flat approach -> terrain-pinned interaction -> flat exit."""
    m = torch.full((1, T_CLIP), _ANCHOR_MASK_FLAT, dtype=torch.uint8)
    m[0, SEAM_IN:SEAM_OUT] = _ANCHOR_MASK_BOX
    return m


def _fake_cmd(mask_t: torch.Tensor, base_traj: torch.Tensor):
    """A ``MotionSequenceCommand`` stand-in wired to a one-clip synthetic library.

    ``base_traj`` is ``[T, 3]``, the clip-world base position per step (identity orientation).
    """
    n = 1
    # The command reads the robot through warp; hand it plain torch tensors.
    msc.wp = SimpleNamespace(to_torch=lambda t: t)

    ident = torch.zeros(T_CLIP, 4)
    ident[:, 3] = 1.0
    # y_des layout: [pos(3), ori xyzw(4)] -- the only channels the placement path reads.
    y_des = torch.cat([base_traj, ident], dim=-1).unsqueeze(0)          # [1, T, 7]

    def _gather(tensor, clip_id, step):
        return tensor[clip_id, step]

    manager = SimpleNamespace(
        num_steps=T_CLIP,
        anchor_mask_t=mask_t,
        lib=SimpleNamespace(y_des=y_des),
        gather=_gather,
        gather_desired=lambda cid, step: (y_des[cid, step], None),
        gather_anchor_mask_at_step=lambda cid, step: mask_t[cid, step],
        phase=torch.zeros(n),
    )
    base_idx = {"pos": torch.tensor([0, 1, 2]), "ori": torch.tensor([3, 4, 5, 6])}
    fake = SimpleNamespace(
        cfg=SimpleNamespace(ego_anchor_every=PERIOD),
        env=SimpleNamespace(episode_length_buf=torch.zeros(n, dtype=torch.long)),
        manager=manager,
        device=DEVICE,
        clip_id=torch.zeros(n, dtype=torch.long),
        start_step=torch.zeros(n, dtype=torch.long),
        _step=torch.zeros(n, dtype=torch.long),
        _base_idx=base_idx,
        _ego_mask=torch.ones(n, dtype=torch.bool),
        _snap_mask=torch.ones(n, dtype=torch.bool),
        _reanchor_active=torch.zeros(n, dtype=torch.bool),
        clip_world_pos=CELL.clone(),
        clip_world_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        _reset_world_pos=CELL.clone(),
        _reset_world_yaw=torch.zeros(n),
        _mask_frame_yaw=torch.zeros(n),
        _snap_base_pos=base_traj[0:1].clone(),
        _snap_base_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        _snap_target_pos=(CELL + base_traj[0:1]).clone(),
        _snap_target_quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        robot=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]))),
    )
    for name in ("_reanchor_clip_ids", "_segment_placement", "_window_placement",
                 "_refresh_current_placement"):
        setattr(fake, name, MethodType(getattr(MotionSequenceCommand, name), fake))
    return fake


def _run(drift_per_step: float, mask_t: torch.Tensor | None = None):
    """Step the command through the clip with the robot lagging downrange; log placements.

    Returns ``(placements[T, 3], masks[T])`` -- the current-step placement at each step.
    """
    mask_t = _mask_schedule() if mask_t is None else mask_t
    base_traj = torch.zeros(T_CLIP, 3)
    base_traj[:, 0] = torch.arange(T_CLIP, dtype=torch.float32) * 0.05   # walks downrange in +x
    base_traj[:, 2] = 0.8
    fake = _fake_cmd(mask_t, base_traj)

    out = torch.zeros(T_CLIP, 3)
    for s in range(T_CLIP):
        fake.env.episode_length_buf = torch.tensor([s], dtype=torch.long)
        MotionSequenceCommand._advance_step(fake)
        # Robot = the current world reference, displaced downrange by the accumulated drift.
        ref_w = fake.clip_world_pos + base_traj[s:s + 1]
        robot = ref_w.clone()
        robot[:, 0] -= drift_per_step * s
        fake.robot.data.root_pos_w = robot
        MotionSequenceCommand._reanchor_ego_envs(fake)
        MotionSequenceCommand._refresh_current_placement(fake)
        out[s] = fake.clip_world_pos[0]
    return out, mask_t[0], fake


def test_terrain_segment_is_welded_at_every_step_including_the_cadence_lag():
    """The terrain-pinned segment's placement is the cell anchor at EVERY one of its steps.

    The assertion that failed before per-segment placements: the ego cadence fires on the global step
    counter, so a boundary that lands between ticks (seam 20, ticks at 12 and 24) left the terrain
    frames playing through the approach's drifted placement for up to ``ego_anchor_every - 1`` steps.
    """
    placements, masks, _ = _run(drift_per_step=0.01)
    terrain = placements[SEAM_IN:SEAM_OUT]
    # Downrange (x) + height (z) pinned to the terrain cell for the whole segment.
    assert torch.allclose(terrain[:, 0], CELL[0, 0].expand(SEAM_OUT - SEAM_IN), atol=1e-6)
    assert torch.allclose(terrain[:, 2], CELL[0, 2].expand(SEAM_OUT - SEAM_IN), atol=1e-6)
    # Specifically the lag window right after the seam (before the next tick at 24).
    for s in range(SEAM_IN, SEAM_IN + PERIOD):
        assert abs(float(placements[s, 0]) - float(CELL[0, 0])) < 1e-6, f"step {s} not welded"


def test_approach_still_rides_the_robot():
    """The flat approach remains ego-anchored -- the fix must not silently pin the whole clip."""
    placements, _, _ = _run(drift_per_step=0.01)
    approach = placements[:SEAM_IN, 0]
    # It drifts away from the cell anchor, monotonically downrange-backwards with the robot.
    assert float(approach[-1]) < float(approach[0]) - 1e-3
    assert float(approach[0]) == pytest.approx(float(CELL[0, 0]), abs=1e-6)   # no drift yet at step 0


def test_mask_constant_clip_placement_is_unchanged_by_the_rework():
    """A clip whose mask never changes is placed identically at every step (the safety net).

    55% of the terrain library and every flat / distractor library are mask-constant, so this is the
    invariant that says the rework cannot regress them: with one segment there is nothing to
    differentiate, and the placement is exactly the single-step ego solve.
    """
    flat_only = torch.full((1, T_CLIP), _ANCHOR_MASK_FLAT, dtype=torch.uint8)
    placements, _, fake = _run(drift_per_step=0.01, mask_t=flat_only)
    # Re-solving the same snapshot for the same mask reproduces the last step exactly.
    again = fake._segment_placement(flat_only[:, -1])[0]
    assert torch.equal(again[0], fake.clip_world_pos[0])
    # And the placement moved with the robot throughout (pure ego, never pinned).
    assert float(placements[-1, 0]) < float(placements[0, 0]) - 1e-3


def test_window_shows_the_terrain_segment_on_its_terrain_during_the_approach():
    """While the robot is still on the approach, the future window places terrain frames on terrain.

    This is what the debug ribbon and the teacher's ``future_base_pose`` observation read. Before the
    rework the whole window rode the approach's placement, so the reference stairs / box appeared
    displaced by the accumulated drift for the entire approach.
    """
    _, _, fake = _run(drift_per_step=0.01)
    # Rewind to mid-approach, keeping the drifted snapshot from the end of the run.
    mid = SEAM_IN - 5
    fake._step = torch.tensor([mid], dtype=torch.long)
    steps = torch.arange(mid, SEAM_OUT, dtype=torch.long).unsqueeze(0)      # [1, K] spans the seam
    cid = torch.zeros_like(steps)
    wpos, _ = fake._window_placement(cid, steps)
    is_terrain = (steps[0] >= SEAM_IN) & (steps[0] < SEAM_OUT)

    # Terrain frames: welded to the cell anchor downrange, regardless of the approach's drift.
    assert torch.allclose(wpos[0, is_terrain, 0], CELL[0, 0].expand(int(is_terrain.sum())), atol=1e-6)
    # Approach frames: displaced from it (the ego correction is real and non-zero here).
    assert (wpos[0, ~is_terrain, 0] - CELL[0, 0]).abs().min() > 1e-3
