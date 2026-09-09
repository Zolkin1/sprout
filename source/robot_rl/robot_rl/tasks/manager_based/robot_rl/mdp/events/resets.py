# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import os

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from ..commands.traj_tracking.world_lift_isaac import (
    lift_position,
    yaw_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

# TODO: Break the key parts of this into another function that can be called from the play script easily.
#   Then I can more easily log initial conditions.

def reset_on_motion_sequence(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    base_frame_name: str = "pelvis_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pin_clip_ids=None,
    pin_start_step: int | None = None,
    spawn_z_offset: float = 0.0,
    clip_draw: str = "random",
):
    """Reset each env onto a randomly chosen pre-generated motion clip.

    Picks a clip (and, if ``cfg.random_start_phase``, a random start step),
    places the clip in the world so its start-step stance anchor sits at the
    env origin (re-zeroed so a mid-clip start still spawns at the origin, not
    metres downrange), and writes the clip's reference base pose / velocity /
    joint state into the sim.  Sets the command's per-env clip tracking state
    so the first ``compute`` reproduces the spawn pose (zero initial error).

    Unlike :func:`reset_on_reference`, there is no velocity conditioner, no
    skill bucket, no contact-gate state, and every env starts on a clip.

    Args:
        env: The environment.
        env_ids: Env indices to reset.
        command_name: The :class:`MotionSequenceCommand` term name.
        base_frame_name: Trajectory base frame (must match the command cfg).
        asset_cfg: Robot asset config.
        pin_clip_ids: Evaluation only -- pin env ``i`` to a stated clip instead of sampling
            (see :func:`resolve_pinned_clips`).
        pin_start_step: Start every env at this clip step instead of sampling one. Negative
            (the default) means sample; ``0`` matches a MuJoCo ``--clip_start_time 0``
            playback, and is what ``record_data.py`` sets so recorded windows cover clip
            phase uniformly (a sampled start run to the clip end is a ramp toward late
            phases -- see that script's ``--start_step``).
        spawn_z_offset: Diagnostic -- metres to raise the robot's spawn height above the
            clip's own root height, without moving the clip. 0.0 (the default) is off.
        clip_draw: How the clip is chosen. ``"random"`` (the default, and what training
            wants) samples with replacement; ``"cycle"`` draws from a shuffled queue so
            every clip in the library is used once before any repeats -- see
            :class:`ClipCycler` for why a recording wants that and training does not.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    if env_ids.numel() == 0:
        return
    k = env_ids.shape[0]
    device = env.device
    num_clips = cmd.manager.num_clips

    # 1. Sample a clip and a start step per env (or take the pinned evaluation set).
    if clip_draw not in ("random", "cycle"):
        raise ValueError(f"clip_draw must be 'random' or 'cycle'; got {clip_draw!r}.")
    clip_id = resolve_pinned_clips(env_ids, pin_clip_ids, device)
    if clip_id is None:
        if clip_draw == "cycle":
            # One group: a flat teacher has no tiles, so every clip is drawable by every env.
            cycler = _clip_cycler(
                cmd, {0: torch.arange(num_clips)}, seed=int(getattr(env.cfg, "seed", 0) or 0)
            )
            clip_id = cycler.take_for(torch.zeros(k, dtype=torch.long, device=device))
        else:
            clip_id = torch.randint(0, num_clips, (k,), device=device)
    elif bool((clip_id >= num_clips).any()):
        raise ValueError(
            f"pin_clip_ids indexes clip {int(clip_id.max())} but the library has "
            f"{num_clips} clips."
        )
    start_step = (sample_start_step(cmd, clip_id, device, cap_at_terrain_end=False)
                  if pin_start_step is None or int(pin_start_step) < 0
                  else torch.full((k,), int(pin_start_step), dtype=torch.long, device=device))

    # 2. World placement of the clip frame (optional random yaw): re-zero only the
    #    downrange (x, y) placement so a mid-clip start spawns the base at the env
    #    origin, but KEEP the clip's base height (z).
    world_quat = torch.zeros(k, 4, device=device)
    world_quat[:, 3] = 1.0  # identity (xyzw, w last -- Isaac Lab convention)
    if getattr(cmd.cfg, "randomize_heading", False):
        yaw = torch.empty(k, device=device).uniform_(-torch.pi, torch.pi)
        zeros = torch.zeros_like(yaw)
        world_quat = quat_from_euler_xyz(zeros, zeros, yaw)
    # Clip-world base position at the start step (clips are stored clip-world, so
    # there is no per-step anchor schedule). Rotate it by the placement heading;
    # re-zero only x/y (keep z) so the base spawns at the env origin at its
    # natural clip height, with the feet resting on the ground.
    y0, _ = cmd.manager.gather_desired(clip_id, start_step)
    base_pos_cf = y0[:, cmd._base_idx["pos"]]  # [k, 3]
    rotated = lift_position(torch.zeros(k, 3, device=device), world_quat, base_pos_cf)
    rotated[:, 2] = 0.0
    world_pos = env.scene.env_origins[env_ids] - rotated

    # 3-5. Ego re-anchor mask (clip envs are ego-anchored with probability
    #      ``ego_anchor_fraction``, else world-anchored; ego-anchored envs get their
    #      clip placement periodically snapped onto the robot root during the episode,
    #      see MotionSequenceCommand._reanchor_ego_envs -- the closed-loop FM tracker
    #      sets ego_anchor_fraction=1.0), then compute + write the reference spawn
    #      state and commit the command's per-env clip state via the shared tail.
    ego_frac = float(getattr(cmd.cfg, "ego_anchor_fraction", 0.0))
    ego_mask = torch.rand(k, device=device) < ego_frac
    _commit_clip_reset(cmd, asset, env_ids, clip_id, start_step, world_pos, world_quat, ego_mask,
                       spawn_z_offset=spawn_z_offset)


def _commit_clip_reset(
    cmd,
    asset: "Articulation",
    env_ids: torch.Tensor,
    clip_id: torch.Tensor,
    start_step: torch.Tensor,
    world_pos: torch.Tensor,
    world_quat: torch.Tensor,
    ego_mask: torch.Tensor,
    snap_mask: torch.Tensor | None = None,
    mask_frame_yaw: torch.Tensor | None = None,
    spawn_z_offset: float = 0.0,
) -> None:
    """Shared tail of the clip resets: reference world state -> sim + command state.

    Given a per-env clip assignment (``clip_id`` / ``start_step``) and world placement
    (``world_pos`` / ``world_quat``), computes the reference base pose / velocity /
    joint state at that placement, writes it to the sim, and commits the
    :class:`MotionSequenceCommand`'s per-env clip tracking state so the first
    ``compute`` (at ``episode_length_buf == 0``) reproduces the spawn pose with zero
    initial error.

    Args:
        cmd: The :class:`MotionSequenceCommand` term.
        asset: The robot articulation.
        env_ids: Env indices being reset.
        clip_id: ``[k]`` clip index per resetting env.
        start_step: ``[k]`` start step within the clip per resetting env.
        world_pos: ``[k, 3]`` world placement of each clip's frame.
        world_quat: ``[k, 4]`` world placement heading (xyzw, w last) per env.
        ego_mask: ``[k]`` bool, which envs are ego-anchored (clip re-placed onto the
            robot root on the cadence) this episode.
        snap_mask: ``[k]`` bool, which envs emit an FM snap (re-inference) event on the
            cadence. Defaults to ``ego_mask`` -- a window is recorded exactly when the clip
            is RE-ANCHORED, on whatever axes that clip's per-step mask frees. Flat and
            terrain use identical logic; the parameter stays for callers that want to
            decouple the two, but nothing does today.
        mask_frame_yaw: ``[k]`` heading (rad) of each env's TERRAIN frame, which the anchor
            mask's downrange / lateral bits resolve in. ``None`` -> 0 (world-axis masking),
            correct for flat ground and for the axis-aligned terrain tiles built today.
    """
    if snap_mask is None:
        snap_mask = ego_mask
    state = cmd.reference_world_state_at(clip_id, start_step, world_pos, world_quat)
    nj = len(asset.joint_names)
    # xyzw: what ``write_root_pose_to_sim_index`` expects (see world_lift_isaac).
    base_pose = state[:, 0:7]          # [pos(3), quat xyzw(4)]
    base_vel = state[:, 7:13]          # [lin(3), ang(3)]
    joint_pos = state[:, 13:13 + nj]
    joint_vel = state[:, 13 + nj:13 + 2 * nj]

    cmd.clip_id[env_ids] = clip_id
    cmd.start_step[env_ids] = start_step
    cmd.clip_world_pos[env_ids] = world_pos
    cmd.clip_world_quat[env_ids] = world_quat
    # Reset-time placement = the clip's terrain-cell anchor. FIXED anchor-mask DOFs
    # pin to this on the ego re-anchor cadence (so terrain-fixed axes stay put while
    # FREE axes track the robot). For flat, world_pos.z is the env-origin ground, so
    # a FIXED-z clip keeps its natural height above the ground rather than the robot's.
    cmd._reset_world_pos[env_ids] = world_pos
    cmd._reset_world_yaw[env_ids] = yaw_from_quat(world_quat)
    cmd._ego_mask[env_ids] = ego_mask
    cmd._snap_mask[env_ids] = snap_mask
    # Terrain frame the anchor mask's downrange / lateral bits resolve in (0 = world axes).
    cmd._mask_frame_yaw[env_ids] = (
        torch.zeros_like(cmd._reset_world_yaw[env_ids]) if mask_frame_yaw is None
        else mask_frame_yaw.to(cmd._mask_frame_yaw.dtype)
    )
    # Seed the re-anchor snapshot from the pose actually written to the sim, so every segment's
    # placement resolves to the terrain-cell anchor until the first ego re-anchor fires.
    cmd.seed_reanchor_snapshot(env_ids, clip_id, start_step, base_pose)

    if spawn_z_offset:
        # Lift ONLY the robot, not the clip placement above: the clip's frame-0 root height
        # can disagree with its own frame-0 joint pose, putting the foot soles below the
        # ground so PhysX ejects them and the episode opens with a kick. Leaving the
        # reference where it is keeps the offset visible as a small initial tracking error
        # rather than hiding it by moving the target too.
        # TODO(spawn-z): a workaround for a CLIP-GENERATION bug -- see the measurements and
        #   the removal condition on ``reset_on_clip`` in g1_teacher_env_cfg.py.
        base_pose = base_pose.clone()
        base_pose[:, 2] += spawn_z_offset
    asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim_index(root_velocity=base_vel, env_ids=env_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def resolve_pinned_clips(
    env_ids: torch.Tensor,
    pin_clip_ids,
    device: torch.device,
) -> torch.Tensor | None:
    """Deterministic clip assignment for evaluation: env ``i`` always gets the same clip.

    Both reset events otherwise draw ``clip_id`` uniformly at random, which makes a run
    un-reproducible at the clip level and makes any cross-sim or cross-``num_envs``
    comparison unsound -- the survival sweep's own docstring warns that "a different env
    count samples different clips out of the 1500". Pinning removes that: the clip set is
    stated, not sampled, so the same 100 clips can be played in Isaac Lab and in MuJoCo and
    the two numbers mean the same thing.

    Assignment is ``pinned[env_id % len(pinned)]`` -- keyed on the ENV INDEX, not on the
    position within ``env_ids``, so a partial reset (only some envs terminating) keeps every
    env on its own clip instead of reshuffling.

    Args:
        env_ids: Env indices being reset.
        pin_clip_ids: OFF as ``None`` or ``""`` (sample as usual). Otherwise a sequence of
            clip indices, a comma-separated string of them, or a path to a JSON file holding
            the sequence. The file form exists because a hydra override of 100+ ids is
            unwieldy; the empty-string "off" exists because Isaac Lab's configclass types a
            param from its default, so the event's default cannot be ``None``.
        device: Env device for the returned tensor.

    Returns:
        ``[len(env_ids)]`` long tensor of clip indices, or ``None`` to sample as usual.

    Raises:
        ValueError: If the pin list is empty, or holds a negative index.
    """
    if pin_clip_ids is None or (isinstance(pin_clip_ids, str) and not pin_clip_ids.strip()):
        return None
    if isinstance(pin_clip_ids, str):
        text = pin_clip_ids.strip()
        if os.path.isfile(text):
            import json  # noqa: PLC0415
            with open(text) as f:
                pin_clip_ids = json.load(f)
        else:
            pin_clip_ids = [p for p in text.replace(",", " ").split() if p]
    ids = torch.as_tensor([int(c) for c in pin_clip_ids], dtype=torch.long, device=device)
    if ids.numel() == 0:
        raise ValueError("pin_clip_ids is empty; pass None to sample clips as usual.")
    if bool((ids < 0).any()):
        raise ValueError(f"pin_clip_ids holds a negative clip index: {ids.min().item()}")
    return ids[env_ids.to(device) % ids.numel()]


def sample_start_step(
    cmd,
    clip_id: torch.Tensor,
    device: torch.device,
    cap_at_terrain_end: bool,
) -> torch.Tensor:
    """Draw each env's start step within its clip.

    Shared by both reset events so the sampling rule lives in one place.  Uniform over
    ``[0, hi)`` where ``hi`` is:

    * ``clip_len - random_start_min_remaining_s``, per clip, using each clip's OWN real
      length rather than the padded ``num_steps`` -- otherwise a short clip's random start
      lands in (or past) its repeat-last padding tail;
    * additionally capped at ``terrain_end_step + 1`` when ``cap_at_terrain_end`` and
      ``cfg.random_start_before_terrain_end``, so a start can land anywhere up to and
      including the last frame of the clip's terrain-pinned run, but never after it.

    ``clip_len`` / ``terrain_end_step`` live on the clip-library device (may be CPU via
    ``--clip_device``), so both are gathered there and moved to the env device.

    Args:
        cmd: The :class:`MotionSequenceCommand` term (read for ``cfg`` and ``manager``).
        clip_id: ``[k]`` long, each resetting env's clip.
        device: Env device the result must land on.
        cap_at_terrain_end: Whether the terrain cap applies (terrain reset only; a flat
            teacher has no terrain to miss).

    Returns:
        ``[k]`` long start step per env, zeros when ``random_start_phase`` is False.
    """
    k = clip_id.shape[0]
    if not cmd.cfg.random_start_phase:
        return torch.zeros(k, dtype=torch.long, device=device)
    margin = int(round(cmd.cfg.random_start_min_remaining_s / cmd.manager.dt))
    clip_len = cmd.manager.clip_len
    hi = (clip_len[clip_id.to(clip_len.device)].to(device) - margin).clamp(min=1)
    if cap_at_terrain_end and getattr(cmd.cfg, "random_start_before_terrain_end", False):
        terrain_end = cmd.manager.gather_terrain_end_step(clip_id).to(device)
        hi = torch.minimum(hi, terrain_end + 1).clamp(min=1)
    if getattr(cmd.cfg, "random_start_before_terrain_start", False):
        # Open on flat. terrain_start is -1 on a flat clip (no terrain at all), which must not
        # cap anything -- hence the where() rather than a plain minimum.
        t_start = cmd.manager.gather_terrain_start_step(clip_id).to(device)
        hi = torch.where(t_start >= 0, torch.minimum(hi, t_start), hi)
    # A command that seeds its generator history from the clip needs that history to exist,
    # so it reserves the first few steps. The clip-step gather clamps only at the TOP
    # (`_future_step_indices`), so a start inside the reserve would index backwards and read
    # the clip's END as history -- silently. The pad is what makes the seed safe.
    lo = int(getattr(cmd, "min_start_step", 0))
    if lo:
        # The reserve is what lets a clip-seeded history exist: the clip-step gather clamps
        # only at the TOP, so a start inside it would index backwards and read the clip's END
        # as its history, silently.
        #
        # When the caps leave no room above the reserve -- a clip whose terrain starts at step
        # 0, which is exactly the case seeding exists for -- start AT the reserve rather than
        # falling back to 0. Those episodes open on terrain by necessity; opening them without
        # a history as well would combine the two things this is trying to avoid.
        hi = torch.maximum(hi, torch.full_like(hi, lo + 1))
        lo_t = torch.full_like(hi, lo)
        return (lo_t + (torch.rand(k, device=device) * (hi - lo_t).float()).long()).long()
    return (torch.rand(k, device=device) * hi.float()).long()


def sample_terrain_clips(env_tid: torch.Tensor, groups: dict[int, torch.Tensor]) -> torch.Tensor:
    """Sample, per env, a random clip whose ``terrain_id`` matches the env's tile.

    Pure-tensor helper (no sim), shared by :func:`reset_on_terrain_motion_sequence`
    and its unit test.

    Args:
        env_tid: ``[k]`` long, the terrain id of each env's tile (``-1`` = flat).
        groups: ``{terrain_id: clip_indices}`` (from
            :func:`...clips.clip_io.group_clips_by_terrain`); each value is a 1-D long
            tensor of clip indices for that terrain.

    Returns:
        ``[k]`` long clip index per env, on ``env_tid``'s device.

    Raises:
        ValueError: if any env's terrain id has no clips -- that tile can never be
            assigned a matching clip (a fatal library / terrain mismatch).
    """
    device = env_tid.device
    missing = sorted(
        {int(t) for t in env_tid.tolist() if int(t) not in groups or groups[int(t)].numel() == 0}
    )
    if missing:
        raise ValueError(
            f"terrain id(s) {missing} have an env tile but NO clips in the library -- "
            "envs on those tiles cannot be assigned a matching clip. Regenerate the "
            "terrain clip library so every terrain has clips."
        )
    clip_id = torch.empty(env_tid.shape[0], dtype=torch.long, device=device)
    for t in torch.unique(env_tid).tolist():
        mask = env_tid == t
        g = groups[int(t)].to(device)
        pick = torch.randint(0, int(g.numel()), (int(mask.sum()),), device=device)
        clip_id[mask] = g[pick]
    return clip_id


class ClipCycler:
    """Without-replacement clip draw: every clip of a group is used before any repeats.

    :func:`sample_terrain_clips` draws with replacement, so ``E`` episodes over a
    ``C``-clip group touch only ``C * (1 - exp(-E/C))`` distinct clips -- 63% at
    ``E == C``, and the misses are a random 37% of the library.  That is harmless for
    TRAINING (an unseen clip is just a clip the policy did not get that run) and wrong
    for RECORDING, where the dataset's clip coverage IS the deliverable.  Drawing from a
    shuffled permutation instead makes coverage exactly ``min(E, C)``, and a second pass
    starts only once the first is complete.

    One queue per terrain group, because an env may only be handed a clip matching its
    tile; the flat teacher passes a single group holding every clip.  The env -> tile map
    is fixed for a run (no task here has a terrain curriculum), so each group's queue is
    consumed only by the envs sitting on that group's tiles -- and groups whose clips are
    longer therefore advance through their queue more slowly in wall-clock, which is why
    :meth:`coverage` reports per group rather than one number.

    Args:
        groups: ``{group_id: clip_indices}``, each a 1-D long tensor of clip indices.
        seed: Seed for the per-group shuffles. Fixed by default so a re-run of the same
            recording command draws the same clips in the same order.

    Raises:
        ValueError: If ``groups`` is empty or any group holds no clips.
    """

    def __init__(self, groups: dict[int, torch.Tensor], seed: int = 0) -> None:
        if not groups:
            raise ValueError("ClipCycler needs at least one group of clips.")
        empty = sorted(int(g) for g, c in groups.items() if c.numel() == 0)
        if empty:
            raise ValueError(f"ClipCycler group(s) {empty} hold no clips.")
        # Queues live on the CPU: they are drawn from a few times per reset and moved to
        # the env device on the way out, which is cheaper than keeping a permutation of a
        # 10k-clip library resident on the GPU.
        self._clips = {int(g): c.detach().to("cpu").reshape(-1) for g, c in groups.items()}
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(seed))
        self._order: dict[int, torch.Tensor] = {}
        self._cursor: dict[int, int] = {}
        self._passes: dict[int, int] = {g: 0 for g in self._clips}
        for g in self._clips:
            self._shuffle(g)

    def _shuffle(self, group_id: int) -> None:
        """Re-draw ``group_id``'s permutation and rewind its cursor."""
        clips = self._clips[group_id]
        self._order[group_id] = clips[torch.randperm(clips.numel(), generator=self._gen)]
        self._cursor[group_id] = 0

    def take(self, group_id: int, k: int) -> torch.Tensor:
        """Take the next ``k`` clips from ``group_id``'s queue, refilling as needed.

        Args:
            group_id: The group to draw from.
            k: How many clips to take (may exceed the group size; the queue re-shuffles
                and the pass counter advances).

        Returns:
            ``[k]`` long tensor of clip indices, on the CPU.

        Raises:
            KeyError: If ``group_id`` has no queue (a tile whose terrain has no clips).
        """
        if group_id not in self._order:
            raise KeyError(f"no clips for group {group_id}")
        out: list[torch.Tensor] = []
        while k > 0:
            order, cur = self._order[group_id], self._cursor[group_id]
            n = min(k, order.numel() - cur)
            out.append(order[cur:cur + n])
            self._cursor[group_id] = cur + n
            k -= n
            if self._cursor[group_id] >= order.numel():
                self._passes[group_id] += 1
                self._shuffle(group_id)
        return torch.cat(out)

    def take_for(self, group_id_per_env: torch.Tensor) -> torch.Tensor:
        """Draw one clip per env, each from its own group's queue.

        Args:
            group_id_per_env: ``[k]`` long, the group (terrain id) of each resetting env.

        Returns:
            ``[k]`` long clip index per env, on ``group_id_per_env``'s device.

        Raises:
            ValueError: If any env's group has no clips -- the same fatal library /
                terrain mismatch :func:`sample_terrain_clips` reports.
        """
        device = group_id_per_env.device
        missing = sorted({int(t) for t in group_id_per_env.tolist() if int(t) not in self._order})
        if missing:
            raise ValueError(
                f"terrain id(s) {missing} have an env tile but NO clips in the library -- "
                "envs on those tiles cannot be assigned a matching clip. Regenerate the "
                "terrain clip library so every terrain has clips."
            )
        clip_id = torch.empty(group_id_per_env.shape[0], dtype=torch.long, device=device)
        for t in torch.unique(group_id_per_env).tolist():
            mask = group_id_per_env == int(t)
            clip_id[mask] = self.take(int(t), int(mask.sum())).to(device)
        return clip_id

    def coverage(self) -> dict[int, dict[str, int]]:
        """Per-group progress through the library.

        Returns:
            ``{group_id: {"clips", "distinct_used", "passes"}}`` -- how many clips the
            group holds, how many of them have been handed out at least once, and how
            many complete passes have finished.  ``distinct_used == clips`` means the
            recording has spanned that group.
        """
        return {
            g: {
                "clips": int(self._clips[g].numel()),
                "distinct_used": (int(self._clips[g].numel()) if self._passes[g] > 0
                                  else int(self._cursor[g])),
                "passes": int(self._passes[g]),
            }
            for g in sorted(self._clips)
        }


def _clip_cycler(cmd, groups: dict[int, torch.Tensor], seed: int) -> ClipCycler:
    """Fetch (or build once) the command's :class:`ClipCycler` over ``groups``.

    Cached on the command term because the queue is run state, not config: it must
    survive every reset for "no repeat before a full pass" to mean anything.

    Args:
        cmd: The :class:`MotionSequenceCommand` term.
        groups: ``{group_id: clip_indices}`` to cycle over.
        seed: Shuffle seed, used only on the first call.

    Returns:
        The cached cycler.
    """
    cycler = getattr(cmd, "_clip_cycler", None)
    if cycler is None:
        cycler = ClipCycler(groups, seed=seed)
        cmd._clip_cycler = cycler
        total = sum(int(c.numel()) for c in groups.values())
        print(f"[INFO] clip_draw='cycle': drawing WITHOUT replacement from {len(groups)} "
              f"group(s) / {total} clips (seed {seed}); every clip is used once before any "
              "repeats.")
    return cycler


def reset_on_terrain_motion_sequence(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    base_frame_name: str = "pelvis_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pin_clip_ids=None,
    pin_start_step: int | None = None,
    spawn_z_offset: float = 0.0,
    clip_draw: str = "random",
):
    """Reset each env onto a clip whose terrain matches the env's terrain tile.

    Terrain-aware counterpart to :func:`reset_on_motion_sequence`.  The scene terrain
    is a :class:`ClipTerrainImporter` built from the same clip library the command
    tracks, so each env is pinned to a tile with a known ``terrain_id`` and tile
    geometry.  For each resetting env this:

    * samples a clip uniformly from the clips whose per-clip ``terrain_id`` matches
      the env's tile (flat tiles -> flat clips, ``terrain_id == -1``);
    * places the clip on the tile with
      :func:`...terrains.clip_terrain_loader.clip_frame_origin` (identity heading --
      clips are baked axis-aligned with the staircase), so the reference feet ride the
      treads.  Wandering flat clips are re-centered on their flat tile (their bbox is
      not centered on the clip origin);
    * restricts ego re-anchoring to flat clips: re-anchoring snaps the clip onto the
      robot root, which would decouple a terrain clip from the physical staircase, so
      terrain / transition clips are always world-anchored.

    Args:
        env: The environment (``env.scene.terrain`` must be a ClipTerrainImporter).
        env_ids: Env indices to reset.
        command_name: The :class:`MotionSequenceCommand` term name.
        base_frame_name: Trajectory base frame (must match the command cfg).
        asset_cfg: Robot asset config.
        pin_clip_ids: Evaluation only -- pin env ``i`` to a stated clip instead of sampling
            (see :func:`resolve_pinned_clips`). Every pinned clip must sit on its env's
            tile, which a flat-only library guarantees.
        pin_start_step: Start every env at this clip step. Negative (the default) means
            sample; see :func:`reset_on_motion_sequence` for why the recorder pins it to 0.
        clip_draw: ``"random"`` (default) or ``"cycle"``; see :class:`ClipCycler`. Cycling
            is PER TERRAIN GROUP here, since an env may only be handed a clip whose
            terrain matches its tile.
    """
    from motion_gen.clips.clip_io import (
        group_clips_by_terrain,
    )
    from robot_rl.tasks.manager_based.robot_rl.terrains.clip_terrain_loader import (
        clip_frame_origin,
    )

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    if env_ids.numel() == 0:
        return
    k = env_ids.shape[0]
    device = env.device
    importer = env.scene.terrain
    if not hasattr(importer, "env_terrain_id"):
        raise TypeError(
            "reset_on_terrain_motion_sequence requires the scene terrain to be a "
            "ClipTerrainImporter (exposing env_terrain_id / env_cell_*). Use "
            "reset_on_motion_sequence for the flat-ground teacher."
        )

    # Cache the terrain -> clip-indices grouping on the command (built once, on device).
    groups = getattr(cmd, "_terrain_groups", None)
    if groups is None:
        # One-time single-source-of-truth check. The env cfg cannot do this (its __post_init__
        # runs BEFORE the --clip override), and a mismatch is silent-but-fatal: the tiles carry
        # one library's terrain ids while the clips carry another's.
        importer_path = getattr(importer.cfg, "clip_library_path", None)
        if importer_path != cmd.cfg.path:
            raise ValueError(
                "terrain clip library mismatch: the ClipTerrainImporter built its mesh from "
                f"'{importer_path}' but the tracking command reads '{cmd.cfg.path}'. Both must "
                "be the SAME file (tile terrain ids come from the mesh, clip terrain ids from "
                "the library). Pass one --clip <lib.h5>, which sets both."
            )
        groups = {t: idx.to(device) for t, idx in group_clips_by_terrain(cmd.manager.lib).items()}
        cmd._terrain_groups = groups

    # 1. Per-env terrain id -> sample a matching clip + a start step.
    env_tid = importer.env_terrain_id[env_ids].to(torch.long)
    if clip_draw not in ("random", "cycle"):
        raise ValueError(f"clip_draw must be 'random' or 'cycle'; got {clip_draw!r}.")
    clip_id = resolve_pinned_clips(env_ids, pin_clip_ids, device)
    if clip_id is None:
        if clip_draw == "cycle":
            clip_id = _clip_cycler(
                cmd, groups, seed=int(getattr(env.cfg, "seed", 0) or 0)
            ).take_for(env_tid)
        else:
            clip_id = sample_terrain_clips(env_tid, groups)
    else:
        # A pinned clip still has to belong on the env's TILE: placement below puts it on
        # that tile's corner, so a terrain clip on the wrong tile would ride treads that are
        # not there. Harmless on a flat-only library (every tile and clip is terrain_id -1),
        # which is the case this exists for; loud everywhere else.
        pinned_tid = cmd.manager.lib.terrain_id.to(device)[clip_id].to(torch.long)
        bad = pinned_tid != env_tid
        if bool(bad.any()):
            i = int(torch.nonzero(bad)[0])
            raise ValueError(
                f"pin_clip_ids put clip {int(clip_id[i])} (terrain_id {int(pinned_tid[i])}) on "
                f"env {int(env_ids[i])}, whose tile is terrain_id {int(env_tid[i])}. Pinned "
                "evaluation needs a library whose clips all match their tiles -- use a "
                "flat-only library, or pin ids grouped by terrain."
            )
    # Capped at the end of the clip's terrain-pinned run: a start past it would spawn the
    # robot AFTER the stair climb / box mount, on the flat top or walking away, so the
    # episode never sees the terrain the tile exists for.
    start_step = (sample_start_step(cmd, clip_id, device, cap_at_terrain_end=True)
                  if pin_start_step is None or int(pin_start_step) < 0
                  else torch.full((k,), int(pin_start_step), dtype=torch.long, device=device))

    # 2. Place each clip on its tile (identity heading -- clips are baked axis-aligned
    #    with the staircase, so alignment is a pure translation). Terrain / transition
    #    clips are baked in the tile-LOCAL corner frame, so they go straight on the
    #    corner; flat clips land there too and are re-centered by bbox below.
    world_quat = torch.zeros(k, 4, device=device)
    world_quat[:, 3] = 1.0  # identity (xyzw, w last)
    is_flat = env_tid == -1
    world_pos = clip_frame_origin(
        importer.env_cell_corner[env_ids],
        importer.env_clip_x_offset[env_ids],
    )

    # Flat clips WANDER (turning walks up to tens of metres) and their bbox is not
    # centered on the clip origin, so anchoring that origin at the tile corner would
    # overhang the flat patch (sized to the max flat bbox). Center each flat clip's
    # base-frame bbox on its tile instead so the whole path stays on the tile.
    if bool(is_flat.any()):
        bidx = cmd._base_idx["pos"]
        bx, by = int(bidx[0]), int(bidx[1])
        y_des = cmd.manager.lib.y_des
        flat_ids = clip_id[is_flat].to(y_des.device)
        pel_xy = y_des[flat_ids][:, :, [bx, by]].to(device)             # [m, T, 2]
        bbox_center = 0.5 * (pel_xy.amax(dim=1) + pel_xy.amin(dim=1))    # [m, 2]
        cell_center_flat = importer.env_cell_center[env_ids][is_flat]
        world_pos[is_flat, 0] = cell_center_flat[:, 0] - bbox_center[:, 0]
        world_pos[is_flat, 1] = cell_center_flat[:, 1] - bbox_center[:, 1]

    # 3-5. Ego re-anchor (clip re-placement) applies to ALL envs -- the re-placement is
    #      per-DOF MASKED (see MotionSequenceCommand._reanchor_clip_ids), so a terrain env
    #      re-anchors only its FREE axes onto the robot while FIXED axes stay pinned to the
    #      terrain cell: a stair clip snaps only y (keeping x/z/yaw on the treads), a box
    #      clip follows its per-step mask (y-free-pinned approach/jump, ego recovery). So
    #      terrain no longer detaches, and every env is partial-ego with probability
    #      ego_frac.
    #
    #      Snap logic is IDENTICAL to flat: ``snap_mask`` defaults to ``ego_mask``, so an FM
    #      training window is emitted exactly when the clip is actually RE-ANCHORED, on
    #      whatever axes that clip's mask frees.  (Before the per-DOF mask existed, terrain
    #      clips could only be world-pinned or fully detached, so this passed
    #      ``ego_mask | ~is_flat`` to force a window on every non-flat env -- otherwise the
    #      FM saw no terrain at all.  With masks, a non-ego terrain env is genuinely NOT
    #      re-anchored, and recording it hands the FM accumulated tracking drift instead of a
    #      placement decision.)  Then compute + write the spawn.
    ego_frac = float(getattr(cmd.cfg, "ego_anchor_fraction", 0.0))
    ego_mask = torch.rand(k, device=device) < ego_frac
    # The tile's AUTHORED terrain frame: the anchor mask's downrange / lateral bits resolve in it, so
    # "free lateral" is across the staircase / box whatever heading the terrain carries -- rather than
    # world y, which silently became the traversal axis for a family baked on a rotated heading.
    frame_yaw = importer.env_terrain_yaw[env_ids].to(device)
    _commit_clip_reset(cmd, asset, env_ids, clip_id, start_step, world_pos, world_quat, ego_mask,
                       mask_frame_yaw=frame_yaw, spawn_z_offset=spawn_z_offset)


# ----------------------------------------------------------------------
# The CLIP-FREE runtime: free-form sections.
#
# The free-form terrain set ships as a MANIFEST-ONLY library (zero clip datasets), because
# under the FM fine-tune the reference is GENERATED and nothing places a clip on a tile.
# Every reset above is keyed on one, and fails loudly rather than quietly on such a library:
# ``sample_terrain_clips`` raises the moment a tile's terrain id has no clips -- correct for a
# clip library, fatal for this one. What follows reads the spawn off the manifest instead.
#
# ASSUMED MANIFEST INTERFACE (the free-form terrain generator owns the producing half): each
# section's manifest entry carries a ``route`` block, which the loader copies onto
# ``ClipTerrainLayout.terrain_route`` and the importer re-exposes as
# ``ClipTerrainImporter.terrain_route: dict[int, dict[str, float]]``, keyed by ``terrain_id`` --
# the same per-terrain-id dict idiom as ``terrain_frame_yaw`` / ``terrain_type_code`` /
# ``clip_size_y``.
# ----------------------------------------------------------------------

#: The route fields every free-form section's manifest entry must declare, in the TILE-LOCAL
#: CORNER frame -- the frame :func:`...clip_terrain_loader.clip_frame_origin` places geometry
#: in, so a route and the mesh it describes share one datum and cannot drift. Lengths in m,
#: angles in rad, ``x`` downrange and ``y`` lateral within the tile's own ``frame_yaw``.
#:
#: * ``start_x`` / ``start_y`` -- where the robot spawns.
#: * ``start_z`` -- the WALKABLE SURFACE the robot stands on there, not a base height: the
#:   top of the start platform. 0.0 for flat, and for the stairs family by the geometry rule
#:   that offsets the whole structure so its LOWEST platform top sits at z = 0 (a descending
#:   section otherwise goes below the world floor and the absolute-z fall termination fires on
#:   a perfectly good descent).
#: * ``start_yaw`` -- spawn heading relative to the tile frame; 0.0 = straight down the section.
#: * ``goal_x`` -- the downrange goal line: past it, the section has been walked.
#: * ``goal_y`` / ``goal_half_width`` -- the goal band's lateral centre and half-extent.
FREEFORM_ROUTE_FIELDS: tuple[str, ...] = (
    "start_x", "start_y", "start_z", "start_yaw", "goal_x", "goal_y", "goal_half_width",
)


def _freeform_route_table(importer) -> tuple[torch.Tensor, int]:
    """Build (once) and cache the terrain-id -> route lookup, validating it on the way.

    A row per terrain id in ``[lo, hi]``, so a per-env lookup is one gather rather than a
    python loop over unique tile ids -- which matters because the GOAL TERMINATION reads this
    every step for every env, not once per reset.

    Everything that can be checked is checked HERE rather than per step, for the same reason:
    the table is built once, so a bounds check costs one host sync at startup instead of one
    every control step.

    Args:
        importer: The :class:`ClipTerrainImporter` (needs ``terrain_route`` and
            ``env_terrain_id``).

    Returns:
        ``(table, lo)`` -- ``[hi - lo + 1, len(FREEFORM_ROUTE_FIELDS)]`` float table on the
        importer's device, and the terrain id its row 0 stands for.

    Raises:
        TypeError: If the terrain carries no ``terrain_route`` at all (a clip library, or a
            free-form library built before the route block existed).
        ValueError: If a route is missing a field, if a tile in the scene has no route, or if
            a route's goal is not downrange of its own start.
    """
    cached = getattr(importer, "_freeform_route_cache", None)
    if cached is not None:
        return cached

    routes = getattr(importer, "terrain_route", None)
    if not routes:
        raise TypeError(
            "the free-form runtime needs ClipTerrainImporter.terrain_route (per-terrain-id "
            f"route dicts carrying {list(FREEFORM_ROUTE_FIELDS)}), and this terrain has none. "
            "It is a clip library, or a free-form library whose manifest entries carry no "
            "'route' block -- regenerate it."
        )
    device = importer.env_terrain_id.device
    ids = sorted(int(t) for t in routes)
    lo, hi = ids[0], ids[-1]
    table = torch.full((hi - lo + 1, len(FREEFORM_ROUTE_FIELDS)), float("nan"), device=device)
    for tid, route in routes.items():
        missing = [f for f in FREEFORM_ROUTE_FIELDS if f not in route]
        if missing:
            raise ValueError(
                f"terrain {int(tid)}'s route is missing {missing}; every free-form section "
                f"must declare all of {list(FREEFORM_ROUTE_FIELDS)}."
            )
        table[int(tid) - lo] = torch.tensor(
            [float(route[f]) for f in FREEFORM_ROUTE_FIELDS], device=device
        )

    # Every tile the scene actually placed must have a route: a tile without one would spawn
    # the robot at NaN, which PhysX turns into a far-away crash rather than an error here.
    scene_ids = {int(t) for t in importer.env_terrain_id.unique().tolist()}
    orphans = sorted(t for t in scene_ids if not (lo <= t <= hi) or t not in routes)
    if orphans:
        raise ValueError(
            f"terrain id(s) {orphans} have an env tile but NO route in the manifest -- envs on "
            "those tiles have nowhere to spawn and no goal. Regenerate the free-form terrain."
        )
    # A goal at or behind the start truncates every episode on its first step, which reads as
    # a policy that never walks rather than as a broken manifest.
    start_x = table[:, FREEFORM_ROUTE_FIELDS.index("start_x")]
    goal_x = table[:, FREEFORM_ROUTE_FIELDS.index("goal_x")]
    half_w = table[:, FREEFORM_ROUTE_FIELDS.index("goal_half_width")]
    bad = [int(r) + lo for r in torch.nonzero(goal_x <= start_x).flatten().tolist()]
    if bad:
        raise ValueError(
            f"terrain id(s) {bad} declare a goal_x at or BEHIND their own start_x; the goal "
            "line is downrange (+x in the tile frame), so every episode there would truncate "
            "on its first step."
        )
    bad_w = [int(r) + lo for r in torch.nonzero(half_w <= 0.0).flatten().tolist()]
    if bad_w:
        raise ValueError(f"terrain id(s) {bad_w} declare a non-positive goal_half_width.")

    importer._freeform_route_cache = (table, lo)
    return table, lo


def freeform_route(importer, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Each env's section route, plus the tile frame it is expressed in.

    One gather off the cached table (see :func:`_freeform_route_table`), so it is cheap
    enough for the per-step goal test as well as the reset.

    Args:
        importer: The :class:`ClipTerrainImporter`.
        env_ids: Env indices to look up.

    Returns:
        Every name in :data:`FREEFORM_ROUTE_FIELDS` as a ``[k]`` tensor, plus ``corner``
        (``[k, 3]`` world position of the tile's local (0, 0, 0)) and ``frame_yaw`` (``[k]``
        heading of the tile's own frame), all on the importer's device.
    """
    table, lo = _freeform_route_table(importer)
    rows = table[importer.env_terrain_id[env_ids].to(torch.long) - lo]        # [k, F]
    out = {name: rows[:, i] for i, name in enumerate(FREEFORM_ROUTE_FIELDS)}
    out["corner"] = importer.env_cell_corner[env_ids]
    out["frame_yaw"] = importer.env_terrain_yaw[env_ids].to(rows.dtype)
    return out


def tile_local_xy(
    corner: torch.Tensor, frame_yaw: torch.Tensor, pos_w: torch.Tensor,
) -> torch.Tensor:
    """World xy -> the tile's own ``(downrange, lateral)`` coordinates.

    The inverse of the placement in :func:`reset_on_freeform_section`, and the same rotation
    ``base_pose_error_termination`` applies to split an error into downrange / lateral: the
    terrain frame is a yaw about world z, so this is a translation to the tile corner and a
    rotation by ``-frame_yaw``.

    Args:
        corner: ``[k, 3]`` world position of each tile's local (0, 0, 0).
        frame_yaw: ``[k]`` heading (rad) of each tile's own frame.
        pos_w: ``[k, 3]`` world positions (only xy is read).

    Returns:
        ``[k, 2]`` tile-local ``(x, y)``.
    """
    d = pos_w[:, :2] - corner[:, :2]
    cos_y, sin_y = torch.cos(frame_yaw), torch.sin(frame_yaw)
    return torch.stack([cos_y * d[:, 0] + sin_y * d[:, 1],
                        -sin_y * d[:, 0] + cos_y * d[:, 1]], dim=-1)


#: Columns of the cached spawn-area table, matching ``motion_gen.terrain.freeform``'s
#: ``SPAWN_AREA_FIELDS``.  Declared on both sides on purpose, the same way
#: :data:`FREEFORM_ROUTE_FIELDS` mirrors that module's ``ROUTE_FIELDS``: the authoring package
#: is IsaacLab-free and cannot be imported here.
FREEFORM_SPAWN_FIELDS: tuple[str, ...] = ("x", "y", "z", "half_x", "half_y", "yaw", "weight")


def _freeform_spawn_table(importer, pos_jitter_m: tuple[float, float]
                          ) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build (once) the terrain-id -> spawn-area lookup.

    Two levels, because areas are per tile and variable in number where the route block is a
    fixed set of scalars: a flat ``[A, 7]`` table of every area in the scene, plus a
    ``[n_tiles, 2]`` ``(offset, count)`` index, so a per-env draw is two gathers rather than a
    python loop over tiles.

    **A tile that declares no areas gets one synthesised from its single start**, with the
    jitter half-widths as its extents.  That makes the fallback the same code path as a
    declared area rather than a branch, and it is exactly the old behaviour: uniform in
    ``+/- pos_jitter_m`` about ``start_x/start_y`` at ``start_z``.  Every library authored
    before spawn areas existed therefore behaves identically, which is what allows this to
    land without regenerating any clip.

    Args:
        importer: The ``ClipTerrainImporter``, carrying ``terrain_route``.
        pos_jitter_m: Half-widths for the synthesised single-start area.

    Returns:
        ``(areas, index, lo)`` -- ``[A, 7]`` float, ``[hi - lo + 1, 2]`` long, and the terrain
        id row 0 of the index stands for.

    Raises:
        ValueError: If a declared area is missing a field.
    """
    cached = getattr(importer, "_freeform_spawn_cache", None)
    if cached is not None and cached[0] == tuple(float(v) for v in pos_jitter_m):
        return cached[1], cached[2], cached[3]

    routes = getattr(importer, "terrain_route", None) or {}
    device = importer.env_terrain_id.device
    ids = sorted(int(t) for t in routes)
    lo, hi = ids[0], ids[-1]
    rows: list[list[float]] = []
    index = torch.zeros((hi - lo + 1, 2), dtype=torch.long, device=device)
    for tid in ids:
        route = routes[tid]
        areas = route.get("spawn_areas") or []
        if not areas:
            areas = [{"x": route["start_x"], "y": route["start_y"], "z": route["start_z"],
                      "half_x": float(pos_jitter_m[0]), "half_y": float(pos_jitter_m[1]),
                      "yaw": route["start_yaw"], "weight": 1.0}]
        index[tid - lo, 0] = len(rows)
        index[tid - lo, 1] = len(areas)
        for a in areas:
            missing = [f for f in FREEFORM_SPAWN_FIELDS if f not in a]
            if missing:
                raise ValueError(
                    f"terrain {tid}'s spawn area is missing {missing}; every area must "
                    f"declare all of {list(FREEFORM_SPAWN_FIELDS)}.")
            rows.append([float(a[f]) for f in FREEFORM_SPAWN_FIELDS])
    table = torch.tensor(rows, dtype=torch.float32, device=device)
    importer._freeform_spawn_cache = (tuple(float(v) for v in pos_jitter_m), table, index, lo)
    return table, index, lo


def _draw_spawn_areas(importer, env_ids: torch.Tensor,
                      pos_jitter_m: tuple[float, float]) -> torch.Tensor:
    """Pick one spawn area per env, weighted, and return its row.

    Args:
        importer: The ``ClipTerrainImporter``.
        env_ids: Envs being reset.
        pos_jitter_m: Half-widths for a synthesised single-start area.

    Returns:
        ``[k, 7]`` the drawn areas' :data:`FREEFORM_SPAWN_FIELDS`.
    """
    k = env_ids.shape[0]
    table, index, lo = _freeform_spawn_table(importer, pos_jitter_m)
    tid = importer.env_terrain_id.reshape(-1)[env_ids].long()
    off = index[tid - lo, 0]
    cnt = index[tid - lo, 1]
    # `index` is DENSE over [lo, hi], so a tile id inside that range carrying no route leaves
    # `cnt = 0` -- and the draw below would then hand that env row `off + max_n - 1`, i.e.
    # ANOTHER TILE'S spawn area, silently and plausibly. Today this is unreachable because
    # `freeform_route` raises on an orphan before this is called, but that is an ordering
    # dependency between two functions rather than a property of either, and a caller that
    # reordered them would get wrong spawns with no error.
    if k_bad := int((cnt <= 0).sum().item()):
        orphan = sorted({int(t) for t in tid[cnt <= 0].tolist()})
        raise ValueError(
            f"{k_bad} env(s) sit on terrain {orphan}, which declares no spawn area and no "
            "route to synthesise one from. Their draw would silently return a neighbouring "
            "tile's area.")
    # A weighted draw without a python loop: build the per-env cumulative weight over its own
    # slice by masking the flat table. `cnt` is small (3 on the two-box course), so the mask
    # is cheap and stays on device.
    max_n = int(cnt.max().item()) if k else 1
    ar = torch.arange(max_n, device=table.device)[None, :].expand(k, -1)
    valid = ar < cnt[:, None]
    idx = (off[:, None] + ar).clamp_(0, max(table.shape[0] - 1, 0))
    w = table[idx, 6] * valid.to(table.dtype)
    cdf = w.cumsum(dim=1)
    u = torch.rand(k, 1, device=table.device) * cdf[:, -1:].clamp_min(1e-12)
    pick = (cdf < u).sum(dim=1).clamp_(max=max_n - 1)
    return table[off + pick]


def reset_on_freeform_section(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spawn_z_offset: float = 0.005,
    pos_jitter_m: tuple[float, float] = (0.25, 0.25),
    yaw_jitter_rad: float = 0.2,
) -> None:
    """Spawn the robot standing at its section's start pose -- no clip anywhere in it.

    The clip resets above pose the robot FROM the reference they are about to track. There is
    no reference here at reset: the generator has not run yet, and the library carries no clip
    to borrow a pose from. So the robot is placed standing, in the articulation's own default
    joint configuration and at its default base height, on the walkable surface the manifest
    declares at the section's start (``start_z``, the top of the start platform).

    **The command is deliberately untouched.** A clip reset also commits the command's per-env
    clip state so the first ``compute`` reproduces the spawn with zero error; there is no such
    state to commit without a clip, and the FM's plan anchor is latched by its own re-plan on
    the first step. Anything the clip-free command needs at reset belongs in that command's
    ``reset``, not here.

    Args:
        env: The environment (``env.scene.terrain`` must be a ``ClipTerrainImporter`` carrying
            ``terrain_route``).
        env_ids: Env indices to reset.
        asset_cfg: Robot asset config.
        spawn_z_offset: Metres to raise the spawn above the declared surface (default 0.005).
            NOT a diagnostic here, unlike the clip resets' identically-named argument: with the
            soles exactly on the surface PhysX ejects the robot and every episode opens with a
            kick. 0.785 (the G1's ``init_state`` base height) + this 0.005 is the 0.79 m
            ``mjeval.terrains.SPAWN_BASE_Z`` records as the measured spawn height.
        pos_jitter_m: Half-widths (downrange, lateral) of the uniform spawn-position jitter, in
            the TILE frame.  **Used only for a tile that declares no ``spawn_areas``**, where
            it becomes the extents of the single implicit area; a declared area carries its
            own, because on a box top the extents are what keep the soles on the surface. Without it every episode on a tile starts from one pose, so the
            generator's first plan is conditioned on one image per tile and the whole set is
            degenerate in exactly the variable the fine-tune is trying to broaden. The box must
            fit inside the start platform, which the 1.219 m platform minimum affords at the
            default.
        yaw_jitter_rad: Half-width of the uniform spawn-heading jitter (0.2 rad = 11.5 deg).
            Small on purpose: the section is walked downrange, and a large heading error at
            spawn asks the yaw controller to recover before the robot has taken a step.

    Raises:
        TypeError: If the scene terrain is not a route-carrying ``ClipTerrainImporter``.
    """
    if env_ids.numel() == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    importer = env.scene.terrain
    if not hasattr(importer, "env_terrain_id"):
        raise TypeError(
            "reset_on_freeform_section requires the scene terrain to be a ClipTerrainImporter "
            "(exposing env_terrain_id / env_cell_corner / terrain_route)."
        )
    k = env_ids.shape[0]
    device = env.device
    route = freeform_route(importer, env_ids)

    # One spawn AREA per env, drawn from what the tile declares -- a rectangle the robot may
    # start anywhere inside, which on the two-box course includes each box's TOP. A tile that
    # declares none gets one synthesised from its single start with `pos_jitter_m` as the
    # extents, so an existing library is unchanged and none needs regenerating.
    #
    # **A tile that DOES declare areas ignores `pos_jitter_m` entirely**, since the area's own
    # half-extents replace it. That is deliberate -- on a box top they are what keeps both
    # soles on the surface -- but it means an authored GROUND area must be sized at
    # `freeform.FREEFORM_SPAWN_JITTER_M` or its spawns stop matching the unauthored path.
    area = _draw_spawn_areas(importer, env_ids, pos_jitter_m)      # [k, 7]
    # Uniform WITHIN the area, in the TILE frame, so "downrange" means the same thing on every
    # tile. The area's own half-extents replace `pos_jitter_m`: for a box top they are what
    # keeps both soles on the surface, and adding the jitter on top would walk the robot off
    # the edge the buffer exists to hold it back from.
    jitter = (torch.rand(k, 3, device=device) * 2.0 - 1.0)
    local_x = area[:, 0] + jitter[:, 0] * area[:, 3]
    local_y = area[:, 1] + jitter[:, 1] * area[:, 4]
    yaw_f = route["frame_yaw"].to(device)
    yaw = yaw_f + area[:, 5] + jitter[:, 2] * float(yaw_jitter_rad)

    # Tile-local -> world: a yaw about z and a translation onto the tile's corner. The corner's
    # own z is the tile ground, so the surface the robot stands on is corner.z + the AREA's
    # own z (0 on the ground, the box height on a box top), and the base sits a nominal
    # standing height above THAT.
    cos_f, sin_f = torch.cos(yaw_f), torch.sin(yaw_f)
    nominal = wp.to_torch(asset.data.default_root_pose)[env_ids]           # [k, 7], pos + xyzw
    pos = route["corner"].to(device).clone()
    pos[:, 0] += cos_f * local_x - sin_f * local_y
    pos[:, 1] += sin_f * local_x + cos_f * local_y
    # The AREA's surface height, not the section start's: a box-top spawn stands 0.40 m up.
    pos[:, 2] += area[:, 2] + nominal[:, 2] + float(spawn_z_offset)

    zeros = torch.zeros_like(yaw)
    root_pose = torch.cat([pos, quat_from_euler_xyz(zeros, zeros, yaw)], dim=-1)   # xyzw
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    asset.write_root_link_velocity_to_sim_index(
        root_velocity=torch.zeros(k, 6, device=device), env_ids=env_ids,
    )
    # The DEFAULT joint state, which is also the PD target's rest point -- so the robot opens
    # the episode holding the pose it is being asked for, rather than with a step input.
    # ``add_joint_default_pos`` (a startup DR term) writes that same buffer, so a randomised
    # default is spawned as well as commanded, which is what makes the two consistent.
    asset.write_joint_position_to_sim_index(
        position=wp.to_torch(asset.data.default_joint_pos)[env_ids], env_ids=env_ids,
    )
    asset.write_joint_velocity_to_sim_index(
        velocity=wp.to_torch(asset.data.default_joint_vel)[env_ids], env_ids=env_ids,
    )
