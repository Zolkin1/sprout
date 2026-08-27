# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import os

import torch
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
