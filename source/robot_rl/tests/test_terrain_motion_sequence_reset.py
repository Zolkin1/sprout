# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the terrain-aware clip reset's pure selection + placement logic.

Exercises the two building blocks of
:func:`...mdp.events.resets.reset_on_terrain_motion_sequence` without an IsaacLab
sim: :func:`sample_terrain_clips` (per-env clip assignment matching the env's tile
``terrain_id``) and :func:`...terrains.clip_terrain_loader.clip_frame_origin` (the
tile placement math), plus the flat-clip bbox-centering invariant the reset uses.
"""

from __future__ import annotations

import pytest
import torch

from types import SimpleNamespace

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.motion_sequence_cmd import (
    MotionSequenceCommand,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.events.resets import (
    ClipCycler,
    sample_terrain_clips,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
    clip_frame_origin,
    default_num_flat_tiles,
)


def _groups() -> dict[int, torch.Tensor]:
    """{-1: flat clips, 0: one terrain clip, 1: two terrain clips}."""
    return {
        -1: torch.tensor([0, 1, 2], dtype=torch.long),
        0: torch.tensor([3], dtype=torch.long),
        1: torch.tensor([4, 5], dtype=torch.long),
    }


def test_sample_matches_env_terrain():
    """Every env gets a clip whose terrain_id matches its tile."""
    groups = _groups()
    env_tid = torch.tensor([0, -1, 1, 1, -1, 0], dtype=torch.long)
    torch.manual_seed(0)
    clip_id = sample_terrain_clips(env_tid, groups)

    assert clip_id.shape == env_tid.shape
    assert clip_id.dtype == torch.long
    for cid, tid in zip(clip_id.tolist(), env_tid.tolist()):
        assert cid in groups[tid].tolist(), f"clip {cid} not in terrain {tid}'s group"


def test_sample_covers_multi_clip_group():
    """A terrain with multiple clips has all of them reachable across draws."""
    groups = _groups()
    env_tid = torch.full((200,), 1, dtype=torch.long)  # all on terrain 1 (clips 4, 5)
    torch.manual_seed(1)
    picks = set(sample_terrain_clips(env_tid, groups).tolist())
    assert picks == {4, 5}


def test_sample_raises_on_missing_terrain():
    """A tile whose terrain has no clips is a fatal misconfig (loud abort)."""
    groups = _groups()
    env_tid = torch.tensor([0, 2], dtype=torch.long)  # terrain 2 has no clips
    with pytest.raises(ValueError, match="terrain id"):
        sample_terrain_clips(env_tid, groups)


def test_sample_raises_on_empty_group():
    """An empty clip list for a present terrain id also aborts."""
    groups = {0: torch.tensor([], dtype=torch.long)}
    env_tid = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="terrain id"):
        sample_terrain_clips(env_tid, groups)


def _fake_lib(flat_clip_count: int, gen_counts: dict | None):
    """Minimal stand-in for ClipLibraryData for default_num_flat_tiles."""
    terrain_id = torch.full((flat_clip_count + 3,), -1, dtype=torch.long)
    terrain_id[:3] = torch.tensor([0, 1, 2])  # a few non-flat clips
    attrs = {} if gen_counts is None else {"generator_config": {"counts": gen_counts}}
    return SimpleNamespace(terrain_id=terrain_id, attrs=attrs)


def test_default_num_flat_tiles_from_generation():
    """With generation metadata, flat tiles = the FLAT-TERRAIN count (not flat-clip
    count), so tile counts match the generation split regardless of clips-per-terrain."""
    lib = _fake_lib(flat_clip_count=3600, gen_counts={"flat": 9, "terrain": 8, "transition": 8})
    assert default_num_flat_tiles(lib) == 9


def test_default_num_flat_tiles_fallback_without_metadata():
    """Pre-generator_config libraries fall back to the flat-clip count."""
    lib = _fake_lib(flat_clip_count=8, gen_counts=None)
    assert default_num_flat_tiles(lib) == 8


def test_clip_frame_origin_is_the_tile_corner():
    """A clip is baked TILE-LOCAL, so placement is a pure translation onto the tile corner.

    The tile's mesh is emitted in the same local frame and translated to the corner, so anything else --
    notably centring the clip in its CELL -- shifts the motion off its own geometry (a cell is sized to the
    largest tile of its class, so the error is half the class-vs-tile size difference).
    """
    corner = torch.tensor([[10.0, 4.0, 0.5]])
    x_off = torch.tensor([0.3])
    out = clip_frame_origin(corner, x_off)

    assert torch.allclose(out[:, 0], corner[:, 0] + x_off)   # x: corner + legacy back-buffer
    assert torch.allclose(out[:, 1], corner[:, 1])           # y: the corner, NOT the cell centre
    assert torch.allclose(out[:, 2], corner[:, 2])           # z: tile ground

    # Invariant: a clip point at local (a, b) lands on the tile's geometry at local (a, b).
    for a, b in ((0.0, 0.0), (2.5, 1.2), (7.0, 3.4)):
        assert torch.allclose(out[:, 0] - x_off + a, corner[:, 0] + a)
        assert torch.allclose(out[:, 1] + b, corner[:, 1] + b)


def test_clip_frame_origin_ignores_the_cell_size():
    """Two tiles of different size in the same class place their clips identically on their own corners."""
    corner = torch.tensor([[10.0, 4.0, 0.0], [10.0, 20.0, 0.0]])
    out = clip_frame_origin(corner, torch.zeros(2))
    assert torch.allclose(out[:, :2], corner[:, :2])


def test_snap_decoupled_from_reanchor():
    """The re-anchor cadence emits FM snaps for all ``_snap_mask`` envs (flat ego +
    world-anchored stairs) but re-places the clip only for ``_ego_mask`` (flat ego)
    envs -- so stair envs get a training window without moving off their treads."""
    period = 10
    n = 6
    placed: dict[str, list[int]] = {}

    def _stub_reanchor_clip_ids(ids: torch.Tensor) -> None:
        placed["ids"] = ids.tolist()

    # env: 0,1 = flat ego (snap + re-place); 2,3 = stair (snap, NO re-place);
    #      4 = flat non-ego (never snaps); 5 = stair off-tick (won't fire this step).
    fake = SimpleNamespace(
        cfg=SimpleNamespace(ego_anchor_every=period, fm_snap_every=period),
        env=SimpleNamespace(episode_length_buf=torch.tensor([0, 10, 0, 20, 0, 5]),
                            common_step_counter=0),
        _snap_mask=torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool),
        _ego_mask=torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool),
        _reanchor_active=torch.zeros(n, dtype=torch.bool),
        _last_reanchor_step=-1,
        _resampling=False,
        _reanchor_clip_ids=_stub_reanchor_clip_ids,
    )
    MotionSequenceCommand._reanchor_ego_envs(fake)

    tick = fake.env.episode_length_buf % period == 0  # [T, T, T, T, T, F]
    # Snap flag fires exactly on snap_mask & tick (envs 0,1,2,3 here; env 5 off-tick).
    assert fake._reanchor_active.tolist() == (fake._snap_mask & tick).tolist()
    assert fake._reanchor_active.tolist() == [True, True, True, True, False, False]
    # Clip is re-placed only for ego & tick (the flat ego envs 0,1).
    assert placed["ids"] == [0, 1]
    # Stair envs 2,3 snap (FM window) but are NOT re-placed (stay world-locked).
    assert fake._reanchor_active[2] and fake._reanchor_active[3]
    assert 2 not in placed["ids"] and 3 not in placed["ids"]


def _snap_fixture(ego_every: int, snap_every: int, step: int, counter: int = 0):
    """Fake command with independent ego / snap cadences at a given episode step."""
    n = 2
    placed: dict[str, list[int]] = {}
    fake = SimpleNamespace(
        cfg=SimpleNamespace(ego_anchor_every=ego_every, fm_snap_every=snap_every),
        env=SimpleNamespace(episode_length_buf=torch.full((n,), step, dtype=torch.long),
                            common_step_counter=counter),
        _snap_mask=torch.ones(n, dtype=torch.bool),
        _ego_mask=torch.ones(n, dtype=torch.bool),
        _reanchor_active=torch.zeros(n, dtype=torch.bool),
        _last_reanchor_step=-1,
        _resampling=False,
        _reanchor_clip_ids=lambda ids: placed.setdefault("ids", ids.tolist()),
    )
    return fake, placed


def test_snap_and_anchor_cadences_are_independent():
    """The FM snap cadence must NOT follow the (now per-step) re-anchor cadence.

    With every-step anchoring the two sharing a timer would emit one FM training window
    per env per step -- 12x the dataset, and a target that encodes no placement decision
    because node 0 always coincides with the robot.
    """
    # Step 3: the clip re-anchors (every step) but no snap fires (snap every 12).
    fake, placed = _snap_fixture(ego_every=1, snap_every=12, step=3)
    MotionSequenceCommand._reanchor_ego_envs(fake)
    assert placed["ids"] == [0, 1], "per-step anchoring must re-place every env"
    assert not fake._reanchor_active.any(), "no FM snap off the snap cadence"

    # Step 12: both fire.
    fake, placed = _snap_fixture(ego_every=1, snap_every=12, step=12)
    MotionSequenceCommand._reanchor_ego_envs(fake)
    assert placed["ids"] == [0, 1]
    assert fake._reanchor_active.all()


def test_reanchor_snapshot_is_once_per_step():
    """Isaac Lab computes observations twice per step; the snapshot must not double-take.

    At the old 1-in-12 cadence this was incidental; at every-step anchoring it is
    load-bearing on every step, so it is enforced against ``common_step_counter``.
    """
    fake, placed = _snap_fixture(ego_every=1, snap_every=1, step=4, counter=7)
    MotionSequenceCommand._reanchor_ego_envs(fake)
    assert placed["ids"] == [0, 1]

    placed.clear()
    MotionSequenceCommand._reanchor_ego_envs(fake)      # same step -> must be a no-op
    assert placed == {}

    fake.env.common_step_counter = 8                     # next step -> runs again
    MotionSequenceCommand._reanchor_ego_envs(fake)
    assert placed["ids"] == [0, 1]


def test_flat_bbox_centering_invariant():
    """The reset re-centers a wandering flat clip's bbox on its tile: world placement
    = cell_center - bbox_center, so the reference bbox center lands on the tile center."""
    # A wandering flat clip whose pelvis xy is NOT centered on the clip origin.
    pel_xy = torch.tensor([[[0.0, 0.0], [4.0, 2.0], [10.0, -1.0]]])  # [1, T, 2]
    bbox_center = 0.5 * (pel_xy.amax(dim=1) + pel_xy.amin(dim=1))    # [1, 2] -> (5.0, 0.5)
    assert torch.allclose(bbox_center, torch.tensor([[5.0, 0.5]]))

    cell_center = torch.tensor([[20.0, 7.0]])
    world_xy = cell_center - bbox_center                            # reset's flat placement
    world_bbox_center = world_xy + bbox_center                      # bbox center lifted to world
    assert torch.allclose(world_bbox_center, cell_center)


def test_reanchor_sits_out_the_resample_call_without_stamping_the_guard():
    """The reset-time ``_update_command`` must not take the re-anchor snapshot.

    Isaac Lab's ``_reset_idx`` calls ``command_manager.reset()`` -- which the base command
    implements as a full ``_update_command()`` -- BEFORE it zeroes ``episode_length_buf``.
    During that call ``_step`` is the NEW ``start_step`` plus the DYING episode's length, so
    a snapshot taken there samples the clip base far past the spawn.  Because the snapshot
    is guarded to once per ``common_step_counter``, taking it there also stamps the guard
    and suppresses the correct retake from the following ``command_manager.compute()``.

    The placement was then wrong by the clip's travel over that bogus span -- planar only,
    since the anchor mask pins z -- which put every tracked body ~0.57 m off its reference
    on the first step of an episode and tripped ``ee_pose_error`` on ~38% of resets while
    never once firing mid-episode.

    So: skip while resampling, and critically LEAVE THE GUARD UNSET.
    """
    fake, placed = _snap_fixture(ego_every=1, snap_every=1, step=7)
    fake._resampling = True
    MotionSequenceCommand._reanchor_ego_envs(fake)
    assert "ids" not in placed, "no snapshot may be taken during the resample call"
    assert fake._last_reanchor_step == -1, (
        "the guard must stay unset, or the corrected retake this same step is suppressed"
    )

    # The real re-anchor still happens later in the SAME step.
    fake._resampling = False
    MotionSequenceCommand._reanchor_ego_envs(fake)
    assert placed["ids"] == [0, 1]
    assert fake._last_reanchor_step == fake.env.common_step_counter


def test_reanchor_still_guarded_once_per_step_after_a_resample():
    """Sitting out the resample must not weaken the twice-per-step idempotency guard."""
    fake, placed = _snap_fixture(ego_every=1, snap_every=1, step=4)
    fake._resampling = True
    MotionSequenceCommand._reanchor_ego_envs(fake)          # skipped
    fake._resampling = False
    MotionSequenceCommand._reanchor_ego_envs(fake)          # the real one
    placed["ids"] = None                                     # detect any further take
    MotionSequenceCommand._reanchor_ego_envs(fake)          # must be a no-op
    assert placed["ids"] is None


def test_base_termination_measures_only_the_pinned_axes():
    """A FREE anchor DOF is not measured; a PINNED one is, at one tight bound.

    A free DOF is one the clip was slid along to sit on the robot, so error there is zero by
    construction and invisible to the policy -- bounding it only truncates episodes, and it
    was 90% of base terminations at 2.6k iterations. On flat that leaves height + tilt, which
    is exactly what SONIC terminates on; stairs/boxes additionally pin downrange and yaw.
    """
    from motion_gen.world_lift import _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT
    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations import (
        base_pose_error_termination,
    )

    lim, big = 0.3, 0.5          # big is over the bound
    n = 4
    # 0,1 = flat (downrange free); 2,3 = box (downrange pinned).
    # Envs 0,2 are displaced DOWNRANGE; envs 1,3 are displaced in HEIGHT.
    masks = torch.tensor([_ANCHOR_MASK_FLAT, _ANCHOR_MASK_FLAT,
                          _ANCHOR_MASK_BOX, _ANCHOR_MASK_BOX], dtype=torch.uint8)
    root = torch.zeros(n, 3)
    root[0, 0] = big; root[2, 0] = big        # downrange offset
    root[1, 2] = big; root[3, 2] = big        # height offset
    ident = torch.zeros(n, 4); ident[:, 3] = 1.0
    state = torch.cat([torch.zeros(n, 3), ident, torch.zeros(n, 6)], dim=-1)

    cmd = SimpleNamespace(
        termination_reference_world_state=lambda: state,
        clip_id=torch.zeros(n, dtype=torch.long), _step=torch.zeros(n, dtype=torch.long),
        _mask_frame_yaw=torch.zeros(n),
        manager=SimpleNamespace(gather_anchor_mask_at_step=lambda cid, st: masks),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=root, root_quat_w=ident))},
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
    )
    import robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations as T
    old_wp, T.wp = T.wp, SimpleNamespace(to_torch=lambda t: t)
    try:
        done = base_pose_error_termination(env, max_pos_dist=lim, max_ori_rad=0.8)
        # flat + downrange  -> IGNORED (free);   flat + height     -> caught (pinned)
        # box  + downrange  -> caught (pinned);  box  + height     -> caught (pinned)
        assert done.tolist() == [False, True, True, True]
    finally:
        T.wp = old_wp


def test_base_termination_yaw_only_counts_where_yaw_is_pinned():
    """Yaw is free on flat and pinned on terrain; tilt is pinned in both."""
    from motion_gen.world_lift import _ANCHOR_MASK_BOX, _ANCHOR_MASK_FLAT
    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations import (
        base_pose_error_termination,
    )

    n = 2
    masks = torch.tensor([_ANCHOR_MASK_FLAT, _ANCHOR_MASK_BOX], dtype=torch.uint8)
    ident = torch.zeros(n, 4); ident[:, 3] = 1.0
    # Reference yawed ~90 deg from the robot; positions identical.
    ref_q = torch.zeros(n, 4); ref_q[:, 2] = 0.7071; ref_q[:, 3] = 0.7071
    state = torch.cat([torch.zeros(n, 3), ref_q, torch.zeros(n, 6)], dim=-1)
    cmd = SimpleNamespace(
        termination_reference_world_state=lambda: state,
        clip_id=torch.zeros(n, dtype=torch.long), _step=torch.zeros(n, dtype=torch.long),
        _mask_frame_yaw=torch.zeros(n),
        manager=SimpleNamespace(gather_anchor_mask_at_step=lambda cid, st: masks),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(n, 3), root_quat_w=ident))},
        command_manager=SimpleNamespace(get_term=lambda name: cmd),
    )
    import robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations as T
    old_wp, T.wp = T.wp, SimpleNamespace(to_torch=lambda t: t)
    try:
        done = base_pose_error_termination(env, max_pos_dist=0.3, max_ori_rad=0.8)
        # A pure yaw error is ignored on flat (yaw free) and caught on the box (yaw pinned).
        assert done.tolist() == [False, True]
    finally:
        T.wp = old_wp


def test_tracked_frame_ori_bound_is_per_frame_and_none_disables_it():
    """``None`` drops a frame's ORIENTATION bound while keeping its POSITION bound.

    The wrists carry orientation error the terrain clips ask for and the policy cannot
    follow -- 286 of left_wrist_yaw's 290 orientation trips were on terrain, and raising the
    bound 0.8 -> 1.0 rad barely moved it. Disabling per frame beats setting a number large
    enough to be meaningless, but the position bound must survive.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations import (
        tracked_frame_error_termination,
    )

    frames = ("a", "b")
    n = 4
    # env 0 clean; 1 = 'b' over ori only; 2 = 'b' over pos only; 3 = 'a' over ori.
    act_p = torch.zeros(n, 2, 3)
    des_p = torch.zeros(n, 2, 3)
    act_p[2, 1, 0] = 1.0                                     # 'b' position error
    act_q = torch.zeros(n, 2, 4); act_q[..., 3] = 1.0
    des_q = torch.zeros(n, 2, 4); des_q[..., 3] = 1.0
    # ~pi/2 about z on the offending frames.
    for env_i, frm in ((1, 1), (3, 0)):
        des_q[env_i, frm] = torch.tensor([0.0, 0.0, 0.7071, 0.7071])

    cmd = SimpleNamespace(tracked_frame_pose=lambda names: (act_p, act_q, des_p, des_q))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: cmd))

    # 'b' orientation disabled -> env 1 survives, but its POSITION bound still bites (env 2),
    # and 'a' keeps its orientation bound (env 3).
    done = tracked_frame_error_termination(
        env, frame_names=frames, max_pos_dist=0.5, max_ori_rad=(0.8, None))
    assert done.tolist() == [False, False, True, True]

    # Both bounds active -> env 1 terminates too.
    done_both = tracked_frame_error_termination(
        env, frame_names=frames, max_pos_dist=0.5, max_ori_rad=(0.8, 0.8))
    assert done_both.tolist() == [False, True, True, True]

    with pytest.raises(ValueError, match="max_ori_rad has 3 entries"):
        tracked_frame_error_termination(
            env, frame_names=frames, max_pos_dist=0.5, max_ori_rad=(0.8, 0.8, 0.8))


# --- ClipCycler: without-replacement clip assignment ------------------------------------


def test_cycler_uses_every_clip_before_repeating_any():
    """A full pass over a group is a permutation -- no repeat until it completes."""
    cycler = ClipCycler(_groups(), seed=0)
    drawn = cycler.take(-1, 3).tolist()
    assert sorted(drawn) == [0, 1, 2]


def test_cycler_second_pass_covers_the_group_again():
    """Wrapping re-shuffles rather than continuing off the end."""
    cycler = ClipCycler(_groups(), seed=0)
    two_passes = cycler.take(-1, 6).tolist()
    assert sorted(two_passes[:3]) == [0, 1, 2]
    assert sorted(two_passes[3:]) == [0, 1, 2]
    assert cycler.coverage()[-1]["passes"] == 2


def test_cycler_take_for_respects_group_membership():
    """Every env still gets a clip belonging to its own terrain group."""
    groups = _groups()
    env_tid = torch.tensor([0, -1, 1, 1, -1, 0], dtype=torch.long)
    clip_id = ClipCycler(groups, seed=1).take_for(env_tid)
    for tid, cid in zip(env_tid.tolist(), clip_id.tolist()):
        assert cid in groups[tid].tolist()


def test_cycler_take_for_does_not_repeat_within_one_draw():
    """Two envs on the same tile in one reset get DIFFERENT clips while the queue holds."""
    groups = {0: torch.arange(8, dtype=torch.long)}
    clip_id = ClipCycler(groups, seed=2).take_for(torch.zeros(8, dtype=torch.long))
    assert sorted(clip_id.tolist()) == list(range(8))


def test_cycler_coverage_tracks_partial_progress():
    """Before a pass completes, coverage is the cursor -- not the whole group."""
    cycler = ClipCycler({0: torch.arange(10, dtype=torch.long)}, seed=0)
    cycler.take(0, 4)
    cov = cycler.coverage()[0]
    assert cov == {"clips": 10, "distinct_used": 4, "passes": 0}


def test_cycler_is_deterministic_for_a_seed():
    """Same seed, same order -- so a re-run of a recording draws the same clips."""
    a = ClipCycler(_groups(), seed=7).take(-1, 6).tolist()
    b = ClipCycler(_groups(), seed=7).take(-1, 6).tolist()
    assert a == b
    assert ClipCycler(_groups(), seed=8).take(-1, 6).tolist() != a


def test_cycler_rejects_an_empty_group():
    with pytest.raises(ValueError, match="no clips"):
        ClipCycler({0: torch.arange(3), 1: torch.empty(0, dtype=torch.long)})


def test_cycler_reports_a_tile_with_no_clips():
    """The same fatal library/terrain mismatch sample_terrain_clips raises on."""
    cycler = ClipCycler({0: torch.arange(3, dtype=torch.long)}, seed=0)
    with pytest.raises(ValueError, match="NO clips in the library"):
        cycler.take_for(torch.tensor([0, 5], dtype=torch.long))
