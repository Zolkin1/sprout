# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac side: roll one grid cell and write its episode table.

Imported only by ``evaluate.py --stage cell|grid``, so the plan and report stages stay
simulator-free.

Three things here are load-bearing and none of them is obvious:

* **The env->tile map is overwritten after the env is built.**  IsaacLab spreads envs evenly
  over tile columns, which leaves the 20 s flat lane setting the wall clock while the 6 s
  distractor lanes idle.  ``terrain_levels`` / ``terrain_types`` / ``env_origins`` are plain
  attributes, and ``_refresh_env_arrays`` recomputes every derived per-env array from them,
  so a post-build reassignment is exact -- provided it happens before the first reset, which
  ``RslRlVecEnvWrapper.__init__`` performs as its last line.
* **The clip reset is WRAPPED, not appended to.**  Event terms run in ``cfg.__dict__`` order,
  so a term added by ``setattr`` lands after ``reset_on_clip`` -- too late to choose the clip
  it is about to spawn, and too late to read the state of the episode that just ended.
  Wrapping the term's ``func`` is order-free.  The wrapper must restate the full signature:
  ``ManagerBase._resolve_common_term_cfg`` matches parameter names against the term's params
  statically, and rejects a ``**kwargs`` function.
* **``clip_end`` is added to the terminations.**  The tracker task deliberately has none, so
  without it every episode runs the full 20 s regardless of clip length: a 300-frame
  distractor clip would be 6 s of motion plus 14 s of holding a frozen pose, and in the FM
  arm 14 s of being commanded the final frame's velocity -- walking off a finite tile and
  falling for reasons that have nothing to do with the policy.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

import accumulate
import dr_presets
import sampler
import schema


def load_run(out_dir: str) -> tuple[dict, dict]:
    """Read ``plan.json`` and ``manifest.json`` from a run directory."""
    out = pathlib.Path(out_dir)
    plan = json.loads((out / "plan.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    if int(plan.get("schema_version", -1)) != schema.SCHEMA_VERSION:
        raise SystemExit(
            f"{out}/plan.json is schema_version {plan.get('schema_version')}, but this code "
            f"implements {schema.SCHEMA_VERSION}; re-run --stage plan."
        )
    return plan, manifest


def cell_dir(out_dir: str, key: str) -> pathlib.Path:
    """Directory for one cell's artifacts."""
    return pathlib.Path(out_dir) / "cells" / key


def is_cell_complete(out_dir: str, key: str, plan_hash: str) -> bool:
    """Whether a cell is finished AND was produced by this plan and schema.

    ``complete`` is written last, after the table is closed and renamed, so a killed run is
    detectable rather than quietly short.
    """
    meta = cell_dir(out_dir, key) / "cell.json"
    if not meta.is_file():
        return False
    try:
        info = json.loads(meta.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(info.get("complete")) and info.get("plan_hash") == plan_hash \
        and int(info.get("schema_version", -1)) == schema.SCHEMA_VERSION


def assign_env_tiles(importer, levels, types) -> None:
    """Overwrite the env->tile mapping and refresh every array derived from it.

    Safe only between ``gym.make`` and the wrapper's first reset: ``env_origins`` is read at
    reset time (``root_pose[:, :3] += env_origins[env_ids]``), and USD cloning used an
    unrelated grid, so nothing has consumed the old mapping yet.
    """
    import torch  # noqa: PLC0415

    dev = importer.device
    importer.terrain_levels = torch.as_tensor(levels, device=dev, dtype=torch.long)
    importer.terrain_types = torch.as_tensor(types, device=dev, dtype=torch.long)
    importer.env_origins[:] = importer.terrain_origins[
        importer.terrain_levels, importer.terrain_types]
    importer._refresh_env_arrays()


def prove_coverage(env, plan: dict, env_columns: np.ndarray, pins: list[int]) -> dict:
    """Assert the env->tile->clip wiring is exactly what the plan says, before stepping.

    Every one of these would otherwise surface from inside a reset, minutes in, with a much
    less specific message -- or, for the column order, not at all: envs would simply track
    clips belonging to other tiles.

    Raises:
        SystemExit: On the first mismatch, naming what differs.
    """
    u = env.unwrapped
    importer = u.scene.terrain
    tiles = importer._tile_terrain_id
    live_cols = [int(t) for t in tiles[0].tolist()]
    if live_cols != list(plan["tile_columns"]):
        raise SystemExit(
            "tile column order differs from the plan.\n"
            f"  plan: {plan['tile_columns'][:12]} ...\n  live: {live_cols[:12]} ...\n"
            "The importer's class ordering changed; sampler.TERRAIN_CLASS_ORDER must match."
        )
    for row in range(tiles.shape[0]):
        if [int(t) for t in tiles[row].tolist()] != live_cols:
            raise SystemExit(f"terrain row {row} is not a replica of row 0")
    env_tid = importer.env_terrain_id.detach().to("cpu").numpy()
    want_tid = np.array([live_cols[c] for c in env_columns], dtype=np.int64)
    bad = np.flatnonzero(env_tid != want_tid)
    if bad.size:
        e = int(bad[0])
        raise SystemExit(
            f"env {e} sits on terrain {env_tid[e]} but the allocation put it on column "
            f"{env_columns[e]} (terrain {want_tid[e]})"
        )
    clip_tid = np.array([plan["clips"][p]["terrain_id"] for p in pins], dtype=np.int64)
    bad = np.flatnonzero(clip_tid != env_tid)
    if bad.size:
        e = int(bad[0])
        raise SystemExit(
            f"env {e} is pinned to clip {pins[e]} on terrain {clip_tid[e]} but stands on "
            f"terrain {env_tid[e]}. The terrain reset would refuse this."
        )
    visited = sorted(set(int(c) for c in env_columns))
    return {"columns_total": len(live_cols), "columns_with_envs": len(visited),
            "envs_per_column_min": int(np.bincount(env_columns).min()) if len(env_columns) else 0,
            "envs_per_column_max": int(np.bincount(env_columns).max()) if len(env_columns) else 0}


class EvalResetHook:
    """Wraps the clip reset: close out the ended episodes, then hand each env its next clip.

    Runs inside ``_reset_idx``, which is the only place where the terminal state of the
    episode that just ended is still readable AND the pin list can still be changed in time
    for the reset about to consume it.

    The signature restates the wrapped function's parameters explicitly because
    ``ManagerBase._resolve_common_term_cfg`` validates them statically against the term's
    ``params`` and rejects a ``**kwargs`` function.
    """

    def __init__(self, base_func, *, queue, accum, writer, plan, cell, env_ref,
                 importer=None, levels=None) -> None:
        import inspect  # noqa: PLC0415

        self.base_func = base_func
        # Take the wrapped function's OWN defaults rather than restating them. The event's
        # params dict does not carry every argument (``asset_cfg`` is left to the default),
        # so a default of None here would be passed straight through and the real function
        # would dereference it.
        self._defaults = {
            name: param.default
            for name, param in inspect.signature(base_func).parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        self.queue = queue
        self.accum = accum
        self.writer = writer
        self.plan = plan
        self.cell = cell
        self.env_ref = env_ref
        self.importer = importer
        self.levels = levels
        self.retargets = 0
        self.parked = 0
        self._repeat: dict[int, int] = {}
        self._expected_missing: list | None = None
        self.armed = False
        self.episodes = 0
        self.terminal_state_ok: bool | None = None
        self._term_names: list[str] = []
        self._pin_list: list[int] | None = None

    def __call__(self, env, env_ids, command_name="traj_ref",
                 base_frame_name="pelvis_link", asset_cfg=None, pin_clip_ids=None,
                 pin_start_step=None, spawn_z_offset=0.0, clip_draw="random"):
        """Harvest, re-dispatch, then delegate to the real clip reset."""
        import torch  # noqa: PLC0415

        if asset_cfg is None:
            asset_cfg = self._defaults.get("asset_cfg")
        if pin_start_step is None:
            pin_start_step = self._defaults.get("pin_start_step")
        if self.armed and env_ids is not None and len(env_ids) > 0:
            ids = [int(e) for e in (env_ids.tolist() if hasattr(env_ids, "tolist")
                                    else list(env_ids))]
            # Harvest ONLY envs that were running a real job. Once the queue has no work
            # left an env is parked, but the sim is synchronous so the clip reset keeps
            # respawning it; those episodes back no job and counting them inflated a
            # 104-episode cell to 309 with up to 24 rows on a single clip.
            live = [e for e in ids if bool(self.queue.env_active[e])]
            self.parked += len(ids) - len(live)
            if live:
                self._harvest(env, torch.as_tensor(live, device=env.device).long())
            if isinstance(pin_clip_ids, list):
                # Only from the SECOND reset onwards. The queue's constructor already made
                # the first assignment, and the pin list handed to the config carries it;
                # re-dispatching here would issue every env a second episode before its
                # first had run.
                self._pin_list = pin_clip_ids
                moved = False
                for e in (env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)):
                    before = int(self.queue.env_column[int(e)])
                    nxt = self.queue.next_job(int(e))
                    if nxt is not None:
                        pin_clip_ids[int(e)] = int(nxt)
                        moved = moved or int(self.queue.env_column[int(e)]) != before
                if moved and self.importer is not None:
                    # The dispatcher sent at least one env to a different tile column, so the
                    # env->tile map has to follow BEFORE the wrapped reset reads
                    # ``env_terrain_id`` -- otherwise the clip and the tile disagree and the
                    # reset refuses the pin (which is exactly what it is there to catch).
                    assign_env_tiles(self.importer, self.levels, self.queue.env_column)
                    self.retargets += 1
        self.armed = True
        return self.base_func(env, env_ids, command_name=command_name,
                              base_frame_name=base_frame_name, asset_cfg=asset_cfg,
                              pin_clip_ids=pin_clip_ids, pin_start_step=pin_start_step,
                              spawn_z_offset=spawn_z_offset, clip_draw=clip_draw)

    def _harvest(self, env, env_ids, *, count_completion: bool = True) -> None:
        """Emit one row per ended episode and zero those accumulators.

        ``count_completion=False`` is for the end-of-rollout flush: those episodes did not
        finish, so they must not count towards a clip's quota.
        """
        tm = env.termination_manager
        if not self._term_names:
            self._term_names = list(tm.active_terms)
        ids_np = env_ids.detach().to("cpu").numpy().astype(np.int64)
        cmd = env.command_manager.get_term("traj_ref")
        clip_ids = cmd.clip_id.detach().to("cpu").numpy().astype(np.int64)[ids_np]
        start_steps = cmd.start_step.detach().to("cpu").numpy().astype(np.int64)[ids_np]

        # Fold the TERMINAL frame into the running extremes before reducing. The per-step
        # fold masks out envs that reset, so without this the one frame that crossed a
        # threshold -- the frame that ENDED the episode -- is the one frame missing from
        # min_base_height_w / max_tilt_rad / first_fall_step. Measured cost of omitting it:
        # 289 of 469 base_fallen episodes read as upright, sitting just inside both bounds
        # (median 0.514 m and 45.9 deg against limits of 0.5 m and 70 deg).
        self.accum.fold_terminal(env_ids)
        red = self.accum.snapshot(env_ids)
        bits = np.zeros(ids_np.size, dtype=np.uint16)
        for b, name in enumerate(self._term_names):
            fired = tm.get_term(name).detach().to("cpu").numpy().astype(bool)[ids_np]
            bits |= (fired.astype(np.uint16) << b)
        time_outs = tm.time_outs.detach().to("cpu").numpy().astype(bool)[ids_np]

        # Through the accumulator's converting view: several ArticulationData buffers are
        # warp arrays, which do not accept a torch-style slice.
        d = self.accum.rd
        end_h = d.root_pos_w[:, 2].detach().to("cpu").numpy()[ids_np]
        grav = d.projected_gravity_b.detach().to("cpu").numpy()[ids_np]
        end_tilt = np.arccos(np.clip(-grav[:, 2], -1.0, 1.0))
        if self.terminal_state_ok is None:
            self.terminal_state_ok = self._check_terminal_state(end_h, end_tilt, bits)

        repeats = np.array([self._seen_clip(int(c)) for c in clip_ids], dtype=np.int64)
        rows = self._build_rows(ids_np, clip_ids, red, bits, time_outs, end_h, end_tilt,
                                start_steps, repeats)
        self.writer.append(rows)
        self.episodes += len(ids_np)
        if count_completion:
            for cid in clip_ids.tolist():
                self.queue.note_completed(int(cid))
        self.accum.reset_envs(env_ids)

    def _check_terminal_state(self, end_h, end_tilt, bits) -> bool:
        """Decide, once, whether the state read at reset is the TERMINAL state.

        ``_reset_idx`` calls ``scene.reset`` before the reset events, and whether that leaves
        the pose buffers holding the pre-reset value is an IsaacLab internal.  Rather than
        assume either way, this checks the one case where the answer is decidable: an episode
        that ended via ``base_fallen`` must have been beyond the fall threshold, so if the
        read shows an upright robot at spawn height the read is post-reset.
        """
        if "base_fallen" not in self._term_names:
            return True
        b = self._term_names.index("base_fallen")
        fell = (bits & np.uint16(1 << b)) != 0
        if not fell.any():
            return True
        ok = ((end_h[fell] < schema.FALL_MIN_HEIGHT_M)
              | (end_tilt[fell] > schema.FALL_TILT_RAD)).all()
        if not ok:
            print("[cell] WARNING: the state read inside the clip reset is POST-reset "
                  "(a base_fallen episode reads as upright at spawn height). "
                  "end_base_height_w / end_tilt_rad describe the new episode's spawn, not "
                  "the terminal state; min_base_height_w / max_tilt_rad are unaffected "
                  "because they are folded over the episode. Recorded as "
                  "terminal_state_source='post_reset'.")
        return bool(ok)

    def _seen_clip(self, clip_id: int) -> int:
        """0-based index of this episode among the ones recorded for ``clip_id``."""
        k = self._repeat.get(clip_id, 0)
        self._repeat[clip_id] = k + 1
        return k

    def _build_rows(self, ids_np, clip_ids, red, bits, time_outs, end_h, end_tilt,
                    start_steps, repeats) -> dict:
        """Assemble one full episode-table batch."""
        n = ids_np.size
        info = [self.plan["clips"][int(c)] for c in clip_ids]
        get = lambda k: np.array([i[k] for i in info])  # noqa: E731
        steps = red["steps_survived"]
        last_usable = get("last_usable_step")
        terr = schema.terrain_columns(
            steps, terrain_start_step=get("terrain_start_step"),
            terrain_end_step=get("terrain_end_step"), last_usable=last_usable,
            dt=float(self.plan["dt"]))
        fam_codes = self.cell["family_codes"]
        rows: dict[str, np.ndarray] = {
            "episode_id": np.arange(self.episodes, self.episodes + n, dtype=np.int64),
            "env_id": ids_np,
            "clip_id": clip_ids,
            "source_clip_id": get("source_clip_id"),
            "terrain_id": get("terrain_id"),
            "scene_tile_id": get("column"),
            "family_code": np.array([fam_codes[i["family"]] for i in info]),
            "type_code": np.zeros(n, dtype=np.uint8),
            "repeat_idx": repeats,
            # -1 by design: with DR re-drawn every episode the draw sequence is
            # not reproducible across cells, so there is no episode-level pairing
            # to claim. Cells are paired on clip_id instead.
            "draw_id": np.full(n, -1, dtype=np.int64),
            "clip_len": get("clip_len"),
            "start_step": start_steps,
            "last_usable_step": last_usable,
            # The two terrain PRIMITIVES, explicitly: ``terrain_columns`` returns only the
            # derived flags, so without these they fall through to the zero-fill below and the
            # report's independent recomputation of the degeneracy gate reads every family as
            # informative.
            "terrain_start_step": get("terrain_start_step"),
            "terrain_end_step": get("terrain_end_step"),
            "termination_bits": bits,
            "termination_step": steps,
            "event_observed": (~time_outs).astype(np.uint8),
            "admin_censored": np.zeros(n, dtype=np.uint8),
        }
        rows.update({k: v for k, v in terr.items()})
        rows.update({k: v for k, v in red.items()})
        rows["end_base_height_w"] = end_h
        rows["end_tilt_rad"] = end_tilt
        missing = [n for n, _ in schema.EPISODE_COLUMNS if n not in rows]
        if missing != self._expected_missing:
            # Not fatal -- some columns genuinely have no source yet -- but a column silently
            # filled with zeros reads downstream as a real measurement, so say which.
            print(f"[cell] NOTE: filling {len(missing)} column(s) with no source: {missing}")
            self._expected_missing = missing
        for name, dtype in schema.EPISODE_COLUMNS:
            if name not in rows:
                rows[name] = np.full(n, np.nan if np.dtype(dtype).kind == "f" else 0)
        return {k: rows[k] for k in schema.EPISODE_COLUMN_NAMES}


def reference_overrides(mode: str) -> dict:
    """The command-term fields that DEFINE a reference mode.

    Pure, so the mapping can be tested without a simulator -- and named, so "what makes an
    ``fm_tf`` cell different from an ``fm`` cell" has exactly one answer in the codebase
    instead of being spread across conditionals.

    Every field is stated for every mode rather than left alone where it "should already be
    right".  These are cfg fields with task-level defaults: a task shipping
    ``history_from='clip'`` would otherwise turn every ``fm`` cell into a teacher-forced one
    while the cell key still said ``fm``, which is a wrong number wearing the right label.

    ``ego_anchor_fraction`` is the exception and is ``None`` (leave alone) for ``fm``:

    * ``clip`` -- 1.0.  Only meaningful on the FM TRACKER task, which defaults it to 0:
      clip tracking is off-distribution there unless the reference is ego-anchored.  A
      teacher task already carries what it trained with, and overriding that would evaluate
      it under anchoring it never saw.
    * ``fm_tf`` -- 1.0, for the SAME reason, and this one is easy to miss.  Under teacher
      forcing the history rows are the CLIP's nodes, and the FM trained on ego-anchored
      history where node 0's base offset is centimetres.  At 0.0 the clip drifts off the
      robot and the "correct" context is itself off-distribution -- handicapping the very
      arm meant to show what the model can do.  ``fm_survival_sweep.py`` records the measured
      cost of getting this wrong: 18/24 instead of 22/24.
    * ``fm`` -- ``None``.  The generator's plan is the reference; the clip placement does not
      feed it, so the task's deploy default (0.0) is what should run.

    The CADENCE (``ego_anchor_every``) is never pinned here -- pinning it to 1 is what made
    base-position error identically zero in earlier sweeps.

    Args:
        mode: One of ``evaluate.REFERENCE_MODES``.

    Returns:
        Field name -> value, where ``None`` means "leave the task's value alone".

    Raises:
        ValueError: On an unknown mode, rather than silently returning the ``fm`` mapping.
    """
    if mode == "clip":
        # The generator is not loaded at all, so history_from is inert -- stated anyway so
        # the recorded cell manifest says what ran.
        return {"reference_source": "clip", "history_from": "generator",
                "ego_anchor_fraction": 1.0}
    if mode == "fm":
        return {"reference_source": "fm", "history_from": "generator",
                "ego_anchor_fraction": None}
    if mode == "fm_tf":
        # Same generator, same reference SOURCE -- only the autoregressive context differs.
        return {"reference_source": "fm", "history_from": "clip",
                "ego_anchor_fraction": 1.0}
    raise ValueError(f"unknown reference mode {mode!r}")


def _apply_env_overrides(env_cfg, args, plan: dict, manifest: dict, cell: dict) -> dict:
    """Everything the cell needs set on the config, before ``gym.make``.

    Returns:
        A dict of what was changed, for the cell manifest.
    """
    subset = plan["subset_library"]["path"]
    notes: dict = {}

    # --- the clip library: BOTH the command and the terrain, never one alone -------------
    for name, term in vars(env_cfg.commands).items():
        if hasattr(term, "path"):
            term.path = subset
        if hasattr(term, "clip_device"):
            term.clip_device = args.clip_device or None
    env_cfg.scene.terrain.clip_library_path = subset
    env_cfg.scene.terrain.num_rows = int(args.num_rows)
    env_cfg.scene.terrain.num_flat_tiles = int(plan["subset_library"]["flat_tiles"])

    # --- scene / seed --------------------------------------------------------------------
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)

    # --- the reference the policy tracks --------------------------------------------------
    traj = env_cfg.commands.traj_ref
    for field, value in reference_overrides(cell["reference_mode"]).items():
        if value is not None and hasattr(traj, field):
            setattr(traj, field, value)
    if cell["fm_checkpoint"] and hasattr(traj, "fm_checkpoint"):
        traj.fm_checkpoint = cell["fm_checkpoint"]
    elif hasattr(traj, "fm_checkpoint"):
        traj.fm_checkpoint = ""
    # Adopt the checkpoint's OBSERVATION CONVENTION, not the task's default.
    #
    # ``window_anchor`` decides the frame the reference window's base_pose block is expressed
    # in ("robot_now" = anchored on the robot, "plan" = the no-odometry variant anchored on the
    # plan draw). The block has the SAME WIDTH either way, so a mismatch loads clean and steers
    # wrong with no visible symptom -- which is exactly what ``check_run_conventions`` refuses,
    # and it would refuse every plan-anchored checkpoint on a robot_now task. Reading it off
    # the trained run makes the eval follow the policy instead of the task.
    trained_conv = (manifest.get("policy") or {}).get("conventions") or {}
    want_anchor = trained_conv.get("window_anchor")
    if want_anchor and hasattr(traj, "window_anchor") and traj.window_anchor != want_anchor:
        print(f"[cell] window_anchor: task default {traj.window_anchor!r} -> trained "
              f"{want_anchor!r} (from the run's params/conventions.yaml)")
        traj.window_anchor = want_anchor
    notes["window_anchor"] = getattr(traj, "window_anchor", None)
    notes["trained_window_anchor"] = want_anchor
    notes["reference_source"] = getattr(traj, "reference_source", None)
    # Recorded because ``reference_source`` alone cannot distinguish fm from fm_tf: both run
    # the generator, and the whole point of the pair is the difference between them.
    notes["history_from"] = getattr(traj, "history_from", None)
    notes["ego_anchor_fraction"] = getattr(traj, "ego_anchor_fraction", None)
    notes["ego_anchor_every"] = getattr(traj, "ego_anchor_every", None)
    notes["tracking_mode"] = getattr(traj, "tracking_mode", None)

    # --- episode length from the CLIP, not from the 20 s wall --------------------------
    if getattr(env_cfg.terminations, "clip_end", None) is None:
        from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: PLC0415
        from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: PLC0415

        env_cfg.terminations.clip_end = DoneTerm(
            func=mdp.clip_end_reached,
            params={"command_name": "traj_ref", "reserve_lookahead": True},
            time_out=True)
        notes["terminations_added"] = ["clip_end"]
    else:
        notes["terminations_added"] = []

    # --- the clip reset's own knobs -------------------------------------------------------
    reset = env_cfg.events.reset_on_clip
    reset.params["pin_start_step"] = 0
    reset.params["spawn_z_offset"] = float(args.spawn_z_offset)
    notes["spawn_z_offset"] = float(args.spawn_z_offset)

    # --- domain randomization -------------------------------------------------------------
    notes["dr"] = dr_presets.apply_dr_preset(
        env_cfg, cell["dr_preset"], ood_scale=args.ood_scale,
        ood_depth_noise=bool(args.ood_depth_noise), dr_resample=args.dr_resample)
    return notes


def _install_pin_list(env_cfg, pins: list[int]) -> None:
    """Seed the reset's pin list with the queue's first assignment.

    A list of exactly ``num_envs`` entries, because ``resolve_pinned_clips`` maps
    ``ids[env_id % len(ids)]`` -- at that length the list IS the per-env assignment, so the
    reset hook's job is a single store rather than any arithmetic.
    """
    env_cfg.events.reset_on_clip.params["pin_clip_ids"] = list(int(p) for p in pins)


def stage_cell(args) -> int:
    """Roll one cell and write ``cells/<key>/{episodes.h5,cell.json}``."""
    plan, manifest = load_run(args.out)
    key = args.cell
    cells = {c["key"]: c for c in plan["cells"]}
    if key not in cells:
        raise SystemExit(f"unknown cell {key!r}; the plan has {sorted(cells)}")
    cell = dict(cells[key])
    families = sorted({c["family"] for c in plan["clips"]})
    cell["family_codes"] = {f: i for i, f in enumerate(families)}

    out_cell = cell_dir(args.out, key)
    out_cell.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    launcher_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, hydra_args = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(launcher_args).app

    import gymnasium as gym  # noqa: PLC0415,E402
    import torch  # noqa: PLC0415,E402

    import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,PLC0415,E402
    import robot_rl.tasks  # noqa: F401,PLC0415,E402
    from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper,  # noqa: PLC0415,E402
                                    handle_deprecated_rsl_rl_cfg)
    from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: PLC0415,E402
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415,E402

    status = {"rc": 1}

    @hydra_task_config(cell["task"], "rsl_rl_cfg_entry_point")
    def _main(env_cfg, agent_cfg) -> None:
        """Build, verify, roll, write."""
        import importlib.metadata as md  # noqa: PLC0415

        t0 = time.time()
        notes = _apply_env_overrides(env_cfg, args, plan, manifest, cell)

        clips = [sampler.SelectedClip(c["source_clip_id"], c["terrain_id"], c["family"],
                                     c["clip_len"], c["column"]) for c in plan["clips"]]
        queue = sampler.JobQueue(clips, int(cell["robots_per_clip"]), int(args.num_envs))
        pins = [int(x) for x in queue.env_clip.tolist()]
        _install_pin_list(env_cfg, pins)

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
        # Hard refusal if the checkpoint was trained under a different observation
        # convention. The reference window's base_pose block has the same WIDTH under every
        # convention, so a stale policy loads clean and steers wrong with no visible symptom.
        from provenance import check_run_conventions  # noqa: PLC0415
        check_run_conventions(manifest["policy"]["log_dir"], env_cfg)

        env_raw = gym.make(cell["task"], cfg=env_cfg)
        u = env_raw.unwrapped

        # Reassign the env->tile map to the queue's allocation, then prove it took. Must be
        # before the wrapper, whose __init__ performs the first reset.
        levels = np.zeros(int(args.num_envs), dtype=np.int64)
        assign_env_tiles(u.scene.terrain, levels, queue.env_column)
        coverage = prove_coverage(env_raw, plan, queue.env_column, pins)
        print(f"[cell {key}] coverage {coverage}")

        dr_resolved = notes["dr"]
        dr_presets.verify_dr_terms(env_raw, dr_resolved)
        fp = dr_presets.dr_fingerprint(env_raw)
        if cell["dr_preset"] == "none":
            dr_presets.assert_no_dr_variation(fp)
            print(f"[cell {key}] dr_preset=none verified: every env is the same robot")

        env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                                device=agent_cfg.device)
        ckpt = manifest["policy"]["checkpoint"]
        if getattr(agent_cfg, "load_critic", True):
            runner.load(ckpt)
        else:
            runner.load(ckpt, load_cfg={"actor": True, "critic": False, "optimizer": False,
                                        "iteration": True, "rnd": True})
        policy = runner.get_inference_policy(device=u.device)

        dt = float(u.step_dt)
        accum = accumulate.EpisodeAccumulator(env, dt=dt)
        capacity = int(cell["robots_per_clip"]) * len(clips) + int(args.num_envs)
        table = out_cell / "episodes.h5.inprogress"
        writer = schema.EpisodeTableWriter(str(table), capacity=capacity, meta={
            "cell": key, "reference_mode": cell["reference_mode"],
            "dr_preset": cell["dr_preset"], "fm_label": cell["fm_label"],
            "robots_per_clip": cell["robots_per_clip"], "task": cell["task"],
            "family_codes": cell["family_codes"], "dt": dt,
            "termination_terms": list(u.termination_manager.active_terms),
            "plan_hash": plan["plan_hash"], "overrides": notes, "dr_fingerprint": fp,
        })
        hook = EvalResetHook(u.event_manager.get_term_cfg("reset_on_clip").func,
                             queue=queue, accum=accum, writer=writer, plan=plan,
                             cell=cell, env_ref=u, importer=u.scene.terrain, levels=levels)
        u.event_manager.get_term_cfg("reset_on_clip").func = hook

        # Attach the plan classifier, when one was given and a generator is loaded. Never on
        # the clip arm: there is no plan to classify, and a zero count there is the honest
        # record rather than a missing column.
        if getattr(args, "terrain_ref", "") and cell["fm_checkpoint"]:
            import terrain_class  # noqa: PLC0415
            accum.plan_clf = terrain_class.TerrainClassifier.load(args.terrain_ref)
            print(f"[cell] plan classifier: {accum.plan_clf.feats.shape[0]} reference "
                  f"windows from {args.terrain_ref}. Counts are per RE-PLAN; compare against "
                  "the open-loop control stairs->jump 0.059 [0.047, 0.071].")

        max_steps = int(args.max_steps) or _derived_max_steps(clips, queue, args.num_envs)
        print(f"[cell {key}] rolling: {len(clips)} clips x R={cell['robots_per_clip']} "
              f"= {len(clips) * cell['robots_per_clip']} episodes, {args.num_envs} envs, "
              f"step ceiling {max_steps}")

        obs = env.get_observations()
        steps = 0
        t_roll = time.time()
        with torch.inference_mode():
            while steps < max_steps and not queue.drained:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                accum.fold(~dones)
                steps += 1
                if steps % 500 == 0:
                    st = queue.stats()
                    print(f"[cell {key}] step {steps}: {st['episodes_done']}/"
                          f"{st['episodes_owed']} episodes, "
                          f"{st['clips_complete']}/{st['clips']} clips complete")
            # Inside inference_mode: the accumulators were last written there, which makes
            # them inference tensors, and those cannot be updated in place outside the block.
            censored = _flush_inflight(u, accum, writer, hook, queue)
        writer.close()
        table.rename(out_cell / "episodes.h5")

        wall = time.time() - t0
        info = {
            "complete": True, "schema_version": schema.SCHEMA_VERSION,
            "plan_hash": plan["plan_hash"], "cell": cell,
            "coverage": coverage, "queue": queue.stats(),
            "steps": steps, "max_steps": max_steps,
            "episodes_written": writer.num_written, "admin_censored": censored,
            "terminal_state_source": ("terminal" if hook.terminal_state_ok
                                      else "post_reset"),
            "tile_retargets": hook.retargets,
            "parked_episode_resets": hook.parked,
            "dr": dr_resolved, "dr_fingerprint": fp, "overrides": notes,
            "rollout_tracking_mode": notes.get("tracking_mode"),
            "throughput": {"wall_clock_s": wall, "rollout_s": time.time() - t_roll,
                           "steps_per_s": steps / max(1e-6, time.time() - t_roll),
                           "num_envs": int(args.num_envs), "device": str(u.device)},
            "written_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        }
        (out_cell / "cell.json").write_text(json.dumps(info, indent=2, default=str))
        print(f"[cell {key}] wrote {writer.num_written} episodes in {wall / 60:.1f} min "
              f"({info['throughput']['steps_per_s']:.1f} steps/s)")
        status["rc"] = 0

    _main()
    with contextlib.suppress(Exception):
        app.close()
    return status["rc"]


def _derived_max_steps(clips, queue, num_envs: int) -> int:
    """Step ceiling: the balanced estimate plus generous slack.

    A ceiling below the work simply truncates the cell, which reads as a pile of censored
    episodes rather than as an error -- so it is derived rather than defaulted, and a user
    value below this is refused by the caller.
    """
    total = sum(c.clip_len for c in clips) * queue.robots_per_clip
    return int(1.6 * total / max(1, num_envs)) + 2000


def _flush_inflight(u, accum, writer, hook, queue) -> int:
    """Write the episodes still running at the step ceiling, marked censored.

    They are valid right-censored observations; dropping them would bias every rate, and
    leaving them unmarked would read as failures.

    Restricted to envs the queue still considers ACTIVE.  Once the queue drains, the clip
    reset keeps respawning parked envs -- the sim is synchronous, so they go on stepping --
    and those partial episodes back no job at all.  Writing them would inflate the episode
    count by exactly ``num_envs`` and, worse, add rows that no clip owes.
    """
    import torch  # noqa: PLC0415

    live = torch.as_tensor(queue.env_active, device=accum.device, dtype=torch.bool)
    active = torch.nonzero((accum.steps > 0) & live).flatten()
    if active.numel() == 0:
        return 0
    hook._harvest(u, active, count_completion=False)
    # Re-stamp the rows just written as censored: they did not end, so no event was observed.
    with_h5 = writer._f
    n = int(active.numel())
    start = writer.num_written - n
    with_h5["admin_censored"][start:writer.num_written] = 1
    with_h5["event_observed"][start:writer.num_written] = 0
    return n


def stage_grid(args) -> int:
    """Run every incomplete cell, one subprocess each.

    One process per cell so an Isaac crash cannot poison the next cell and the GPU is fully
    released in between -- and so a multi-hour grid can be resumed.
    """
    plan, _ = load_run(args.out)
    wanted = [c.strip() for c in args.cells.split(",") if c.strip()]
    cells = [c for c in plan["cells"] if not wanted or c["key"] in wanted]
    print(f"[grid] {len(cells)} cell(s) in {args.out}")
    todo = []
    for c in cells:
        done = is_cell_complete(args.out, c["key"], plan["plan_hash"])
        print(f"    {'[skip]' if done and not args.force else '[run ]'} {c['key']:<28} "
              f"R={c['robots_per_clip']}")
        if args.force or not done:
            todo.append(c)
    if args.dry_run:
        return 0
    rc = 0
    for c in todo:
        cmd = [sys.executable, "-u", str(pathlib.Path(__file__).with_name("evaluate.py")),
               "--stage", "cell", "--out", str(args.out), "--cell", c["key"],
               "--num_envs", str(args.num_envs), "--num_rows", str(args.num_rows),
               "--seed", str(args.seed), "--ood_scale", str(args.ood_scale),
               "--dr_resample", args.dr_resample,
               "--spawn_z_offset", str(args.spawn_z_offset)]
        if args.clip_device:
            cmd += ["--clip_device", args.clip_device]
        # Forward EVERY flag the cell stage reads. This list is the whole reason `fm_tf` is a
        # reference MODE and not a flag: anything added here and forgotten reverts silently to
        # its argparse default in the subprocess. Measured, twice: --num_envs ran a grid at 256
        # while the manifest recorded 1024, and --terrain_ref produced 3950 episodes with
        # n_plans == 0 and no error anywhere.
        if getattr(args, "terrain_ref", ""):
            cmd += ["--terrain_ref", args.terrain_ref]
        if not args.ood_depth_noise:
            cmd += ["--no-ood_depth_noise"]
        if args.max_steps:
            cmd += ["--max_steps", str(args.max_steps)]
        print(f"\n[grid] === {c['key']} ===\n[grid] {' '.join(cmd)}")
        proc = subprocess.run(cmd, cwd=os.getcwd(), check=False)
        if proc.returncode != 0:
            print(f"[grid] cell {c['key']} FAILED (rc={proc.returncode}); continuing")
            rc = proc.returncode
    return rc
