# Copyright 2026 Zachary Olkin. All rights reserved.

"""End-to-end terrain clip generation test (needs the flat + stair libraries).

Builds a temporary combined library (flat skills + a ``stair_up`` skill) by
symlinking the on-disk retargeted folders, then exercises the terrain rollout
through the orchestrator's geometry builder and the offline generator:

* a terrain-only clip starts on the first step and climbs its (big, ~transition_length)
  staircase for the whole clip without topping out; and
* the flat rollout still produces flat clips tagged ``terrain_id == -1``.

Skipped when either source library is absent (e.g. CI without the HF cache).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

_MERGED = Path("hf/trajectories/retargeted/2026-06-13_15-10-58_merged")
_STAIRS = Path("hf/trajectories/retargeted/2026-06-13_18-38-09_stairs_up")
pytestmark = pytest.mark.skipif(
    not (_MERGED.exists() and _STAIRS.exists()),
    reason=f"need both {_MERGED} and {_STAIRS}",
)

DEVICE = "cpu"


@pytest.fixture(scope="module")
def combined_lib(tmp_path_factory):
    """A temp library root: flat skills (symlinked) + a stair_up skill folder."""
    root = tmp_path_factory.mktemp("g1_terrain_lib")
    for skill in ("walk_forward", "running", "standing", "walk_backwards"):
        src = (_MERGED / skill).resolve()
        if src.is_dir():
            (root / skill).symlink_to(src)
    stair_dir = root / "stair_up"
    stair_dir.mkdir()
    for yf in sorted(_STAIRS.glob("stairs_up_solution_*.yaml")):
        (stair_dir / yf.name).symlink_to(yf.resolve())
    return str(root)


@pytest.fixture(scope="module")
def built(combined_lib):
    from dataclasses import dataclass

    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
        MultiSkillManager,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_clip_generator import (
        OfflineClipGenerator,
    )
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        build_terrain,
        sample_terrain_specs,
    )

    @dataclass
    class B:
        lin_vel_x: tuple = (0.0, 0.0)
        lin_vel_y: tuple = (0.0, 0.0)
        ang_vel_z: tuple = (0.0, 0.0)
        heading: tuple | None = None

    m = MultiSkillManager(combined_lib, DEVICE, env=None)
    skills = [s.name for s in m.skills]
    stair = next(s.name for s in m.skills if s.has_terrain)
    probs = [1.0 / len(skills)] * len(skills)

    # Per-skill library-derived buckets (real vx/vy/wz ranges) so the
    # generator's bucket table discriminates skills AND can place the
    # PD-injected lateral/yaw command -- exactly what the CLI does.
    def _lib_buckets(forward_only: bool) -> dict:
        cond = m._global_conditioning
        out = {}
        for s in m.skills:
            off = m._skill_traj_offset[m.skill_name_to_idx[s.name]]
            blk = cond[off:off + s.num_trajectories, :3]
            lo, hi = blk.min(0).values.tolist(), blk.max(0).values.tolist()
            if forward_only:
                out[s.name] = B((lo[0], hi[0]), (0.0, 0.0), (0.0, 0.0))
            else:
                out[s.name] = B((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]))
        return out

    build_sampler = OfflineVelocitySampler(_lib_buckets(False), skills, probs, resampling_time_range=(1.5, 4.0),
                                           max_acc=2.0, command_dt=0.02, device=DEVICE, seed=0)
    gen = OfflineClipGenerator(m, build_sampler, dt=0.02, clip_length_s=10.0,
                               exclude_skills=[stair], device=DEVICE)
    # Forward-only sampler at STAIR-range speed (probs on the stair skill, like
    # the CLI's terrain/transition samplers) so clips climb at ~0.2-0.8 m/s and
    # don't overshoot short test terrains; vy/wz come from the PD or are 0.
    stair_probs = [1.0 if n == stair else 0.0 for n in skills]
    fwd_sampler = OfflineVelocitySampler(_lib_buckets(True), skills, stair_probs, resampling_time_range=(1.5, 4.0),
                                         max_acc=2.0, command_dt=0.02, device=DEVICE, seed=1)
    return m, gen, fwd_sampler, stair


def _hd_grid(m):
    """(step_height, step_depth) pairs present in the stair skill's conditioning."""
    s = next(s for s in m.skills if s.has_terrain)
    ct = s.conditioning_tensor  # cols: [vx, vy, vyaw, width, height, length]
    return [(h, d) for h, d in {(round(float(r[4]), 4), round(float(r[5]), 4)) for r in ct}]


def test_terrain_only_clip_climbs(built):
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        build_terrain,
        sample_terrain_specs,
    )
    m, gen, sampler, stair = built
    rng = np.random.default_rng(0)
    specs = sample_terrain_specs(
        rng, n_terrain=1, n_transition=0, hd_grid=_hd_grid(m), width_range=(0.777, 0.777),
        num_steps_range=(5, 5), lead_in_range=(0.0, 0.0), end_platform=1.0,
        num_transitions_range=(1, 1), inter_flat_range=(3.0, 3.0),
        nav_width_range=(0.777, 0.777), flat_width_margin=0.3, transition_length=14.0,
    )
    # terrain-only is JUST the staircase (no flat base / top) sized to span
    # ~transition_length -- one big staircase, same length as a transition.
    assert len(specs[0].stair_steps) == 1
    bt = build_terrain(specs[0], geom_seed=1, device=DEVICE, size_y_floor=1.2)
    assert abs(bt.manifest["size_x"] - specs[0].stair_steps[0] * specs[0].step_depth) < 1e-3
    assert abs(bt.manifest["size_x"] - 14.0) <= specs[0].step_depth, "staircase not sized to transition_length"
    assert all(b["block_type"] == "StairBlock" for b in bt.manifest["bands"]), "terrain-only should have no flat bands"
    gen.clear_terrain_context()
    gen.sampler = sampler   # forward-only vx
    gen.set_terrain_context(
        projector=bt.projector, terrain_dim_lookup=bt.dim_lookup,
        spawn_anchor_pos=bt.spawn_anchor_pos, spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        spawn_on_stairs=bt.spawn_on_stairs, spawn_ground_z=bt.spawn_ground_z,
        band_exit_z=bt.band_exit_z, terrain_id=0, terrain_type="terrain",
    )
    out = gen.generate_batch(4)
    gen.clear_terrain_context()

    names = m.pos_output_names
    pz = names.index("pelvis_link:pos_z"); raz = names.index("right_ankle_roll_link:pos_z")
    y = out["y_des"]
    assert torch.isfinite(y).all()
    assert (out["terrain_id"] == 0).all()

    h = specs[0].step_height
    exit_z = bt.band_exit_z[-1]  # top of a big (~transition_length) staircase
    for ci in range(y.shape[0]):
        # Starts ON the first step: the stance foot begins at ~tread-0 height,
        # not down on the ambient ground (z~0) before the first riser.
        assert float(y[ci, 0, raz]) > 0.5 * h, "did not start on the first step"
        # Climbs during the clip.
        assert y[ci, -1, pz] - y[ci, 0, pz] > 0.3, "clip did not climb"
        # Big staircase: never tops out / falls back to flat -- stays well below
        # the top for the whole (short) test clip.
        assert float(y[ci, :, pz].max()) < exit_z, "unexpectedly reached the top of a big staircase"


def test_approach_masked_by_skill_is_forward(built):
    """The approach slow-down candidate is restricted BY SKILL (extension point).

    plan_slowdown_step picks the longest eligible stride whose forward step
    length fits the gap to the stair; if a BACKWARD stride (vx < 0, but vy =
    vyaw = 0 so it looks "straight") were eligible, then when the gap is shorter
    than the shortest forward step the only fitting strides are backward ones and
    the robot walks backwards on approach.  ``set_terrain_context(approach_skills=
    (...,))`` restricts the candidate to those skills; guard that with
    ``approach_skills=("walk_forward",)`` every approach-eligible trajectory is in
    walk_forward, is forward (vx > 0), and straight (|vy|, |vyaw| ~ 0).
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        build_terrain,
        sample_terrain_specs,
    )
    m, gen, sampler, stair = built
    rng = np.random.default_rng(1)
    specs = sample_terrain_specs(
        rng, n_terrain=1, n_transition=0, hd_grid=_hd_grid(m), width_range=(0.777, 0.777),
        num_steps_range=(5, 5), lead_in_range=(1.5, 4.0), end_platform=1.0,
        num_transitions_range=(1, 1), inter_flat_range=(3.0, 3.0),
        nav_width_range=(0.777, 0.777), flat_width_margin=0.3, transition_length=14.0,
    )
    bt = build_terrain(specs[0], geom_seed=1, device=DEVICE, size_y_floor=1.2)
    gen.set_terrain_context(
        projector=bt.projector, terrain_dim_lookup=bt.dim_lookup,
        spawn_anchor_pos=bt.spawn_anchor_pos, spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        spawn_on_stairs=bt.spawn_on_stairs, spawn_ground_z=bt.spawn_ground_z,
        band_exit_z=bt.band_exit_z, terrain_id=0, terrain_type="terrain",
        approach_skills=("walk_forward",),
    )
    elig = gen._ctx_approach_eligible
    gen.clear_terrain_context()

    assert elig is not None and bool(elig.any()), "no approach-eligible strides at all"
    wf = m.skill_name_to_idx["walk_forward"]
    assert bool((gen._traj_skill_idx[elig] == wf).all()), "approach candidate not restricted to walk_forward"
    cond = m._global_conditioning.to(elig.device)  # [T, >=3]
    assert float(cond[elig, 0].min()) > 0.0, "a BACKWARD/zero stride is approach-eligible"
    assert float(cond[elig, 1].abs().max()) < 0.05, "a strafing stride is approach-eligible"
    assert float(cond[elig, 2].abs().max()) < 0.05, "a turning stride is approach-eligible"
    # And an unknown skill name is rejected loudly.
    import pytest as _pytest
    with _pytest.raises(ValueError):
        gen.set_terrain_context(
            projector=bt.projector, terrain_dim_lookup=bt.dim_lookup,
            spawn_anchor_pos=bt.spawn_anchor_pos, spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            spawn_on_stairs=bt.spawn_on_stairs, spawn_ground_z=bt.spawn_ground_z,
            band_exit_z=bt.band_exit_z, terrain_id=0, terrain_type="terrain",
            approach_skills=("no_such_skill",),
        )
    gen.clear_terrain_context()


def test_transition_multi_band_straight_climb(built):
    """A multi-staircase transition clip walks straight up the centerline + climbs.

    PD steering is disabled: build_terrain gives a centered spawn and
    steer_to_goal=False, so vy/wz stay 0.  Checks the pelvis makes forward
    progress, stays near the centerline (no lateral drift), and climbs the
    first staircase; also that a 2-transition terrain really has 2 stair bands.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        build_terrain,
        sample_terrain_specs,
    )
    m, gen, sampler, stair = built
    rng = np.random.default_rng(3)
    specs = sample_terrain_specs(
        rng, n_terrain=0, n_transition=1, hd_grid=_hd_grid(m), width_range=(0.777, 0.777),
        num_steps_range=(3, 3), lead_in_range=(3.0, 3.0), end_platform=1.0,
        num_transitions_range=(2, 2), inter_flat_range=(3.0, 3.0),
        nav_width_range=(2.5, 2.5), flat_width_margin=0.4, transition_length=14.0,
    )
    bt = build_terrain(specs[0], geom_seed=2, device=DEVICE, size_y_floor=1.2)
    assert not bt.steer_to_goal and bt.nav_goal_xy is None    # PD disabled
    assert len(bt.band_exit_z) == 2, "expected 2 stair bands"
    assert abs(bt.manifest["size_x"] - 14.0) < 1e-3, "transition terrain should be the fixed length"
    cy = bt.manifest["size_y"] / 2.0
    assert abs(float(bt.spawn_anchor_pos[1]) - cy) < 1e-5, "transition should spawn centered"
    gen.clear_terrain_context()
    gen.sampler = sampler   # forward-only stair-range vx; vy/wz = 0
    sampler._gen.manual_seed(1)   # deterministic regardless of test order (shared module sampler)
    gen.set_terrain_context(
        projector=bt.projector, terrain_dim_lookup=bt.dim_lookup,
        spawn_anchor_pos=bt.spawn_anchor_pos, spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        spawn_on_stairs=bt.spawn_on_stairs, spawn_ground_z=bt.spawn_ground_z,
        band_exit_z=bt.band_exit_z, terrain_id=0, terrain_type="transition",
        steer_to_goal=bt.steer_to_goal, nav_goal_xy=bt.nav_goal_xy,
        # Non-steered terrains walk straight up the corridor (matches ``orchestrate``,
        # which sets ``straight_only = not steer_to_goal``); the per-step re-anchor
        # freezes the lateral anchor so the corridor walk stays on the centerline.
        straight_only=True,
        flat_exclude_skills=("running",), approach_skills=("walk_forward",),
    )
    out = gen.generate_batch(4)
    gen.clear_terrain_context()

    names = m.pos_output_names
    px = names.index("pelvis_link:pos_x"); py = names.index("pelvis_link:pos_y")
    pz = names.index("pelvis_link:pos_z")
    y = out["y_des"]
    assert torch.isfinite(y).all()
    # Commanded vy/wz are exactly 0 the whole clip (no steering).
    assert float(out["target_vel"][:, :, 1].abs().max()) == 0.0, "vy should be 0"
    assert float(out["target_vel"][:, :, 2].abs().max()) == 0.0, "vyaw should be 0"
    # No running on the flat bands (excluded via flat_exclude_skills).
    run_id = [i for i, s in enumerate(m.skills) if s.name == "running"][0]
    assert int((out["skill_id"] == run_id).sum()) == 0, "running skill leaked into a terrain clip"
    # No BACKWARD walking anywhere (the approach must be forward-only).
    back = [i for i, s in enumerate(m.skills) if s.name == "walk_backwards"]
    if back:
        assert int((out["skill_id"] == back[0]).sum()) == 0, "walk_backwards leaked into a transition clip"
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        report_off_terrain,
    )
    # Real requirement: no reference FOOT leaves the corridor / treads (the
    # per-step re-anchor keeps the lateral drift bounded).
    _off, _exc = report_off_terrain(out, bt.manifest)
    assert float(_exc.max()) < 0.10, f"foot {float(_exc.max()):.3f} m off the corridor"
    for ci in range(y.shape[0]):
        assert float(y[ci, -1, px]) > float(y[ci, 0, px]) + 1.0, "no forward progress"
        # Climbs at least the first staircase.
        assert float(y[ci, :, pz].max()) - float(y[ci, 0, pz]) > 0.4 * bt.band_exit_z[0], "did not climb"
        # Pelvis stays roughly on the centerline (loose -- includes the reference
        # gait's natural lateral sway, which the per-step re-anchor shows in full).
        assert float((y[ci, :, py] - cy).abs().max()) < 0.45, "pelvis drifted off the centerline"


def _frame_yaw(y: torch.Tensor, names: list[str], frame: str) -> torch.Tensor:
    """Heading yaw (rad) of ``frame`` from its ``ori_{x,y,z,w}`` columns; ``y`` [..., P]."""
    qx = y[..., names.index(f"{frame}:ori_x")]
    qy = y[..., names.index(f"{frame}:ori_y")]
    qz = y[..., names.index(f"{frame}:ori_z")]
    qw = y[..., names.index(f"{frame}:ori_w")]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def test_straight_only_zeros_anchor_yaw_and_records_diag(built):
    """straight_only lays every stride at global 0 yaw + records the discarded yaw.

    A straight (non-steered) transition/terrain clip should NOT carry the previous
    reference's landing heading forward: every stride's anchor is world-aligned, so
    the anchor (right-ankle) heading never accumulates and the clip cannot curve off
    the corridor.  Also checks the per-clip yaw-correction diagnostic is populated
    and that ``steer_to_goal`` + ``straight_only`` is rejected.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        build_terrain,
        sample_terrain_specs,
    )
    m, gen, sampler, stair = built
    rng = np.random.default_rng(3)
    specs = sample_terrain_specs(
        rng, n_terrain=0, n_transition=1, hd_grid=_hd_grid(m), width_range=(0.777, 0.777),
        num_steps_range=(3, 3), lead_in_range=(3.0, 3.0), end_platform=1.0,
        num_transitions_range=(2, 2), inter_flat_range=(3.0, 3.0),
        nav_width_range=(2.5, 2.5), flat_width_margin=0.4, transition_length=14.0,
    )
    bt = build_terrain(specs[0], geom_seed=2, device=DEVICE, size_y_floor=1.2)
    gen.clear_terrain_context()
    gen.sampler = sampler   # forward-only vx; vy/wz = 0
    sampler._gen.manual_seed(1)   # deterministic regardless of test order (shared module sampler)

    # steer_to_goal + straight_only is contradictory -> rejected.
    with pytest.raises(ValueError):
        gen.set_terrain_context(
            projector=bt.projector, terrain_dim_lookup=bt.dim_lookup,
            spawn_anchor_pos=bt.spawn_anchor_pos, spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            spawn_on_stairs=bt.spawn_on_stairs, spawn_ground_z=bt.spawn_ground_z,
            band_exit_z=bt.band_exit_z, terrain_id=0, terrain_type="transition",
            steer_to_goal=True, nav_goal_xy=[10.0, 0.0], straight_only=True,
        )

    gen.set_terrain_context(
        projector=bt.projector, terrain_dim_lookup=bt.dim_lookup,
        spawn_anchor_pos=bt.spawn_anchor_pos, spawn_anchor_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        spawn_on_stairs=bt.spawn_on_stairs, spawn_ground_z=bt.spawn_ground_z,
        band_exit_z=bt.band_exit_z, terrain_id=0, terrain_type="transition",
        straight_only=True, flat_exclude_skills=("running",), approach_skills=("walk_forward",),
    )
    assert gen._straight_only is True
    out = gen.generate_batch(4)

    # Diagnostic populated: every clip crossed >=1 seam, so a discarded-yaw value
    # and a valid source stride are recorded per clip.
    assert gen._diag_yaw_stride is not None and gen._diag_yaw_stride.shape == (4,)
    assert bool((gen._diag_yaw_stride >= 0).all()), "a clip recorded no yaw correction"
    assert torch.isfinite(gen._diag_yaw_signed).all()

    names = m.pos_output_names
    y = out["y_des"]
    # Anchor (right-ankle) heading does NOT accumulate: net change over the whole
    # clip is small (each stride resets to world-aligned; only bounded within-stride
    # roll remains).  Read at t=0 vs the last full-stride start to avoid the
    # mid-stride sway of a truncated final stride.
    rank_yaw = _frame_yaw(y, names, "right_ankle_roll_link")            # [n, T]
    net = (rank_yaw[:, -1] - rank_yaw[:, 0]).abs()
    assert float(net.max()) < np.radians(12.0), "anchor heading accumulated (yaw not zeroed)"

    # The clip stays on the corridor.  Real requirement: no reference FOOT leaves
    # the corridor / treads (the per-step re-anchor freezes the straight-only
    # lateral anchor, so drift stays bounded).
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.terrain_clip_orchestrator import (
        report_off_terrain,
    )
    _off, _exc = report_off_terrain(out, bt.manifest)
    assert float(_exc.max()) < 0.10, f"foot {float(_exc.max()):.3f} m off the corridor"
    # Pelvis stays roughly centered (loose -- includes the natural lateral sway
    # of the reference gait, shown in full by the per-step re-anchor).
    py = names.index("pelvis_link:pos_y")
    cy = bt.manifest["size_y"] / 2.0
    assert float((y[:, :, py] - cy).abs().max()) < 0.45, "pelvis drifted off the centerline"

    gen.clear_terrain_context()
    assert gen._straight_only is False and gen._diag_yaw_stride is None


def test_flat_rollout_still_flat(built):
    m, gen, sampler, stair = built
    from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.offline_velocity_sampler import (
        OfflineVelocitySampler,
    )
    from dataclasses import dataclass

    @dataclass
    class B:
        lin_vel_x: tuple = (0.2, 0.8)
        lin_vel_y: tuple = (0.0, 0.0)
        ang_vel_z: tuple = (0.0, 0.0)
        heading: tuple | None = None

    skills = [s.name for s in m.skills]
    flat_probs = [0.0 if n == stair else (1.0 if n == "walk_forward" else 0.0) for n in skills]
    gen.clear_terrain_context()
    gen.sampler = OfflineVelocitySampler({n: B() for n in skills}, skills, flat_probs,
                                         resampling_time_range=(1.5, 4.0), max_acc=2.0,
                                         command_dt=0.02, device=DEVICE, seed=1)
    out = gen.generate_batch(4)
    names = m.pos_output_names
    pz = names.index("pelvis_link:pos_z")
    y = out["y_des"]
    assert (out["terrain_id"] == -1).all()
    # Flat: pelvis stays near a constant standing height (no climb).
    assert float(y[:, :, pz].max()) - float(y[:, :, pz].min()) < 0.3
