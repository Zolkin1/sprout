# Copyright 2026 Zachary Olkin. All rights reserved.

"""The Isaac side of the best-of-K headroom test: roll, branch, score, write.

Imported only by ``headroom.py --stage run``, so the report stage stays simulator-free.

**The shape of the loop, and why it is not the shape the spec sketches.**  A one-step
deviation rollout normally needs an exact state save/restore, which PhysX does not offer.  It
does not need one here, because a uniformly random pick from K i.i.d. flow-matching samples
*is* an i.i.d. flow-matching sample -- so in the primary (``random``) arm the outer rollout
does not have to depend on the branches at all.  Envs are grouped as ``1 parent + K branch``,
the parent runs an ordinary deployment episode and is **never restored**, and every round the
parent's state is copied into the K branch envs which then diverge for one window.  The
deployed state distribution is therefore the parent's, by construction rather than by
argument, and the K=1 tie-out to the existing ``fm`` arm is exact.

Only ``--commit_mode argmax`` restores the parent: it takes the winning candidate's state
twelve steps in, which is what makes it a deployable oracle policy rather than a measurement.

**Four things here are load-bearing and each was chosen against an alternative.**

* **Candidate 0 inherits the parent's plan by COPY, not by re-inference.**  Its window is
  then the same computation as the parent's own next W seconds up to the state restore, so
  the difference between the two rows measures the harness noise floor -- at every branched
  state, on the real state distribution.  That replaces the spec's synthetic determinism and
  CRN tests with a continuous measurement, which is the only honest form the check can take
  in a simulator whose solver state cannot be restored.
* **The round start forces a re-plan on every parent.**  It is idempotent -- the natural
  cadence has just fired with the same armed seed, so the forced call reproduces it -- and it
  removes the loop's dependence on the cadence having landed where it was expected.  Which
  matters because it does not always: an episode that terminated mid-round restarts its step
  counter wherever the round happened to be, and the counter drives the cadence.
* **Misaligned groups are snapped DOWN to the cadence and skip one round of branching.**
  ``episode_length_buf`` moves by at most 11 steps, and only for an episode that has just
  begun; the alternative -- letting a group sit permanently off-phase -- silently drops it
  from the sample.  The skipped round is also the one whose states have a null-masked
  generator history, which is not a state worth branching from.
* **The branch envs' own tile assignment is irrelevant during a window.**  The terrain is one
  mesh for the whole scene and inter-env collisions are filtered, so writing the parent's
  absolute world pose onto a branch env stands it on the parent's tile whatever column it
  nominally owns.  Its pin still follows the parent's clip, so a mid-window reset puts it
  somewhere coherent rather than on another family's terrain.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import pathlib
import sys
import time
import types

import numpy as np

import dr_presets
import headroom_noise
import headroom_schema as hs
import headroom_score
import runner as eval_runner

#: The scoring window, in re-plan chunks.  Five chunks at K=12 control steps and dt=0.02 is
#: 1.20 s -- the spec's 1.25 s rounded to the cadence, which is what makes the round
#: self-align (the re-plan that opens the next round is the one drawn at the end of this
#: one's last step).
DEFAULT_WINDOW_CHUNKS: int = 5

#: Inventory paths the state copy MUST find.  Not the whole list -- the walk discovers
#: dozens -- but the ones whose absence would leave a branch in a different situation from
#: its parent while every shape still agreed: the plan and where it is anchored, the
#: generator's own history ring (the autoregression), the clip identity, the step counter
#: that drives the re-plan cadence, and the action the policy last emitted.
REQUIRED_STATE: tuple[str, ...] = (
    "cmd._pred_window", "cmd._pred_idx", "cmd._pred_valid", "cmd._anchor_pos",
    "cmd._anchor_quat", "cmd._prev_served_node", "cmd._history._steps_since_reset",
    "cmd.clip_id", "cmd.start_step", "env.episode_length_buf",
    "actions._action", "actions._prev_action",
)


class GroupAllocator:
    """Hands each group its next clip, and tracks the tile column that must follow it.

    One episode per job.  Deliberately not ``sampler.JobQueue``: that one dispatches per ENV
    and balances a per-clip quota against envs that finish at different times, where here a
    group is the unit and its K branch envs never own a job at all.
    """

    def __init__(self, clips: list[dict], *, episodes_per_clip: int, num_groups: int,
                 seed: int) -> None:
        """Build the work list and make the first assignment.

        Args:
            clips: ``plan["clips"]`` entries, indexed by clip id.
            episodes_per_clip: Parent episodes owed per clip.
            num_groups: Number of groups running concurrently.
            seed: Shuffle seed, so consecutive groups do not all start on one family.
        """
        self.clips = clips
        jobs = np.repeat(np.arange(len(clips), dtype=np.int64), int(episodes_per_clip))
        np.random.default_rng(int(seed)).shuffle(jobs)
        self._jobs = list(int(j) for j in jobs)
        self._cursor = 0
        self.num_groups = int(num_groups)
        self.group_clip = np.zeros(self.num_groups, dtype=np.int64)
        self.episodes_started = 0
        self.episodes_done = 0
        for g in range(self.num_groups):
            self.group_clip[g] = self._take()

    def _take(self) -> int:
        """Pop the next job, wrapping once the list is exhausted."""
        if self._cursor >= len(self._jobs):
            return int(self._jobs[-1]) if self._jobs else 0
        j = self._jobs[self._cursor]
        self._cursor += 1
        self.episodes_started += 1
        return int(j)

    @property
    def drained(self) -> bool:
        """Whether every owed episode has been STARTED (the last ones may still run)."""
        return self._cursor >= len(self._jobs)

    def advance(self, group: int) -> int:
        """Retire ``group``'s episode and give it the next clip.

        Returns:
            The new clip id.
        """
        self.episodes_done += 1
        self.group_clip[int(group)] = self._take()
        return int(self.group_clip[int(group)])

    def column_of(self, clip_id: int) -> int:
        """The tile column the clip's terrain sits in."""
        return int(self.clips[int(clip_id)]["column"])

    def stats(self) -> dict:
        """Progress, for the run log and the manifest."""
        return {"jobs": len(self._jobs), "started": self.episodes_started,
                "done": self.episodes_done, "drained": self.drained}


class HeadroomResetHook:
    """Wraps the clip reset so a group's envs always spawn on the group's own clip.

    Restates the wrapped function's signature explicitly for the same reason
    ``eval_runner.EvalResetHook`` does: ``ManagerBase._resolve_common_term_cfg`` matches
    parameter names against the term's ``params`` statically and rejects a ``**kwargs``
    function.
    """

    def __init__(self, base_func, *, alloc: GroupAllocator, group_size: int, importer,
                 levels, env_column, pin_list) -> None:
        """Bind the allocator and the per-env tile map the hook keeps in step with it."""
        import inspect  # noqa: PLC0415

        self.base_func = base_func
        self._defaults = {
            name: p.default for name, p in inspect.signature(base_func).parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        self.alloc = alloc
        self.group_size = int(group_size)
        self.importer = importer
        self.levels = levels
        self.env_column = env_column
        self.pin_list = pin_list
        self.retargets = 0
        #: One entry per PARENT reset since the list was last cleared:
        #: ``(group, ended_clip_id, termination_bits, steps_survived, was_truncation)``.
        #:
        #: A LIST and not a set, and the terminal state is read HERE rather than after the
        #: window, because a parent can reset twice inside one 60-step window -- fall,
        #: respawn, fall again -- and a set collapses that into one episode. It undercounts
        #: exactly the instant failures, which is what made the ``jump_off_front`` tie-out
        #: read 0.704 against the episode harness's 0.484 while every other family agreed.
        #:
        #: This is the only place the ended episode is still readable: ``_reset_idx`` applies
        #: the reset-mode events (this hook) BEFORE zeroing ``episode_length_buf``, and the
        #: termination manager's buffers are not recomputed, so both still describe the
        #: episode that just ended.
        self.parent_resets: list[tuple[int, int, int, int, bool]] = []
        self._term_names: list[str] = []

    def __call__(self, env, env_ids, command_name="traj_ref",
                 base_frame_name="pelvis_link", asset_cfg=None, pin_clip_ids=None,
                 pin_start_step=None, spawn_z_offset=0.0, clip_draw="random"):
        """Re-dispatch any resetting group, then delegate to the real clip reset."""
        import torch  # noqa: PLC0415

        if asset_cfg is None:
            asset_cfg = self._defaults.get("asset_cfg")
        if pin_start_step is None:
            pin_start_step = self._defaults.get("pin_start_step")
        if env_ids is not None and len(env_ids) > 0 and isinstance(pin_clip_ids, list):
            ids = [int(e) for e in (env_ids.tolist() if hasattr(env_ids, "tolist")
                                    else list(env_ids))]
            moved = False
            for e in ids:
                g, slot = divmod(e, self.group_size)
                if slot == 0:
                    # A parent reset ends the group's episode. Take the next clip and move
                    # the WHOLE group's tile map with it -- before the wrapped reset reads
                    # env_terrain_id, which is what refuses a clip/tile mismatch.
                    self.parent_resets.append(
                        (g, int(self.alloc.group_clip[g]), *self._terminal(env, e)))
                    clip = self.alloc.advance(g)
                    col = self.alloc.column_of(clip)
                    for local in range(self.group_size):
                        env_id = g * self.group_size + local
                        pin_clip_ids[env_id] = clip
                        self.pin_list[env_id] = clip
                        moved = moved or int(self.env_column[env_id]) != col
                        self.env_column[env_id] = col
                else:
                    # A branch env fell mid-window. It respawns on the group's current clip;
                    # its state is overwritten from the parent at the next round start, so
                    # this only has to be coherent, not correct.
                    pin_clip_ids[e] = int(self.alloc.group_clip[g])
            if moved:
                eval_runner.assign_env_tiles(self.importer, self.levels, self.env_column)
                self.retargets += 1
        return self.base_func(env, env_ids, command_name=command_name,
                              base_frame_name=base_frame_name, asset_cfg=asset_cfg,
                              pin_clip_ids=pin_clip_ids, pin_start_step=pin_start_step,
                              spawn_z_offset=spawn_z_offset, clip_draw=clip_draw)

    def _terminal(self, env, env_id: int) -> tuple[int, int, bool]:
        """The terminal state of the episode that just ended on ``env_id``.

        Returns:
            ``(termination_bits, steps_survived, was_truncation)``.
        """
        tm = env.termination_manager
        if not self._term_names:
            self._term_names = list(tm.active_terms)
        bits = 0
        for b, name in enumerate(self._term_names):
            if bool(tm.get_term(name)[env_id]):
                bits |= 1 << b
        return bits, int(env.episode_length_buf[env_id]), bool(tm.time_outs[env_id])

    @property
    def termination_terms(self) -> list[str]:
        """Active termination names in the bit order ``_terminal`` packs them in."""
        return list(self._term_names)


def make_cell(args, plan: dict) -> dict:
    """The ``cell`` dict ``eval_runner._apply_env_overrides`` expects.

    Reusing that function rather than restating the overrides is the point: the reference
    mode's field mapping, the ``window_anchor`` adoption from the trained run, the
    ``clip_end`` termination and the DR preset are then defined in exactly one place for both
    harnesses.

    Raises:
        SystemExit: If the requested generator label is not in the plan.
    """
    gens = plan.get("generators") or {}
    if args.fm_label not in gens:
        raise SystemExit(
            f"--fm_label {args.fm_label!r} is not in the plan's generators {sorted(gens)}. "
            "Re-run `evaluate.py --stage plan` with the checkpoint you want, or pass one of "
            "those labels."
        )
    return {
        "key": (f"headroom__{args.commit_mode}__{args.dr_preset}__{args.fm_label}"
                + ("__samenoise" if getattr(args, "same_noise", False) else "")),
        "task": plan["cells"][0]["task"] if plan.get("cells") else args.task,
        "reference_mode": "fm",
        "dr_preset": args.dr_preset,
        "fm_label": args.fm_label,
        "fm_checkpoint": gens[args.fm_label]["path"],
        "robots_per_clip": int(args.episodes_per_clip),
    }


def override_args(args, num_envs: int) -> types.SimpleNamespace:
    """An ``args``-shaped namespace for ``eval_runner._apply_env_overrides``.

    Every field that function reads is stated here, including the ones this harness does not
    expose -- a missing attribute would raise deep inside the override, and a defaulted one
    that silently differs from the episode harness would make the two incomparable.
    """
    return types.SimpleNamespace(
        num_envs=int(num_envs), num_rows=int(args.num_rows), seed=int(args.seed),
        clip_device=args.clip_device, start_step=int(args.start_step),
        spawn_z_offset=float(args.spawn_z_offset), ood_scale=float(args.ood_scale),
        ood_depth_noise=bool(args.ood_depth_noise), dr_resample=args.dr_resample,
        fall_min_height=float(args.fall_min_height),
    )


def stage_run(args) -> int:
    """Roll the branch experiment and write ``<out>/headroom/<key>/{states,episodes}.h5``."""
    plan, manifest = eval_runner.load_run(args.out)
    cell = make_cell(args, plan)
    key = cell["key"]
    families = sorted({c["family"] for c in plan["clips"]})
    family_codes = {f: i for i, f in enumerate(families)}

    group_size = int(args.k) + 1
    num_groups = max(1, int(args.num_envs) // group_size)
    num_envs = num_groups * group_size
    if num_envs != int(args.num_envs):
        print(f"[headroom] num_envs {args.num_envs} -> {num_envs} "
              f"({num_groups} groups x {group_size} envs); a partial group would give some "
              "candidate index no env.")

    out_cell = pathlib.Path(args.out) / "headroom" / key
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

    import headroom_state  # noqa: PLC0415,E402
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
        notes = eval_runner._apply_env_overrides(
            env_cfg, override_args(args, num_envs), plan, manifest, cell)

        alloc = GroupAllocator(plan["clips"], episodes_per_clip=int(args.episodes_per_clip),
                               num_groups=num_groups, seed=int(args.seed))
        env_column = np.array([alloc.column_of(alloc.group_clip[e // group_size])
                               for e in range(num_envs)], dtype=np.int64)
        pins = [int(alloc.group_clip[e // group_size]) for e in range(num_envs)]
        eval_runner._install_pin_list(env_cfg, pins)

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
        from provenance import check_run_conventions  # noqa: PLC0415
        check_run_conventions(manifest["policy"]["log_dir"], env_cfg)

        env_raw = gym.make(cell["task"], cfg=env_cfg)
        u = env_raw.unwrapped
        levels = np.zeros(num_envs, dtype=np.int64)
        eval_runner.assign_env_tiles(u.scene.terrain, levels, env_column)
        coverage = eval_runner.prove_coverage(env_raw, plan, env_column, pins)
        print(f"[headroom {key}] coverage {coverage}")

        dr_resolved = notes["dr"]
        dr_presets.verify_dr_terms(env_raw, dr_resolved)
        fp = dr_presets.dr_fingerprint(env_raw)
        if cell["dr_preset"] == "none":
            dr_presets.assert_no_dr_variation(fp)
            print(f"[headroom {key}] dr_preset=none verified: every env is the same robot, "
                  "so the K branches differ only in the flow noise and the state restore.")
        else:
            print(f"[headroom {key}] WARNING: dr_preset={cell['dr_preset']!r}. The DR draw "
                  "and the per-step observation noise are per ENV, so they differ ACROSS "
                  "the K branches of a state -- an uncontrolled difference the parent-row "
                  "noise floor will absorb and report. Read every spread against it.")

        env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
        rl = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ckpt = manifest["policy"]["checkpoint"]
        if getattr(agent_cfg, "load_critic", True):
            rl.load(ckpt)
        else:
            rl.load(ckpt, load_cfg={"actor": True, "critic": False, "optimizer": False,
                                    "iteration": True, "rnd": True})
        policy = rl.get_inference_policy(device=u.device)

        cmd = u.command_manager.get_term("traj_ref")
        assert_terminations(u)
        replan_k = int(cmd.cfg.fm_update_every)
        window_steps = int(args.window_chunks) * replan_k
        dt = float(u.step_dt)
        print(f"[headroom {key}] K={args.k} candidates, chunk {replan_k} steps "
              f"({replan_k * dt:.3f} s), window {window_steps} steps "
              f"({window_steps * dt:.3f} s), commit={args.commit_mode}, "
              f"branch stride {args.window_chunks if args.commit_mode == 'random' else 1} "
              "chunk(s)")
        if args.commit_mode == "random_k":
            print(f"[headroom {key}] RANDOM-K CONTROL for the argmax arm: the parent is "
                  "restored every chunk exactly as under argmax, but onto a UNIFORMLY RANDOM "
                  "candidate. Its episode success is what argmax has to beat -- the "
                  "difference is the oracle, and the restore cost cancels.")

        crn = headroom_noise.CrnNoise(
            num_envs=num_envs, group_size=group_size, future_len=int(cmd.ctx.F),
            state_dim=int(cmd.ctx.D), device=u.device, run_seed=int(args.seed))
        if bool(getattr(args, "same_noise", False)):
            crn.force_candidate = 0
            print(f"[headroom {key}] SAME-NOISE CONTROL: every candidate is served candidate "
                  "0's draw, so the K branches differ only in which env slot they landed in. "
                  "Any best-of-K rise here is the state restore, not the generator -- it is "
                  "the number the real arm's rise has to beat.")
        cmd.set_noise_source(crn)

        mirror = headroom_state.EnvStateMirror(u)

        scorer = headroom_score.WindowScorer(u, dt=dt)
        hook = HeadroomResetHook(
            u.event_manager.get_term_cfg("reset_on_clip").func, alloc=alloc,
            group_size=group_size, importer=u.scene.terrain, levels=levels,
            env_column=env_column, pin_list=pins)
        u.event_manager.get_term_cfg("reset_on_clip").func = hook

        meta = {
            "key": key, "cell": cell, "k": int(args.k), "group_size": group_size,
            "num_groups": num_groups, "num_envs": num_envs, "dt": dt,
            "replan_chunk_steps": replan_k, "window_steps": window_steps,
            "window_s": window_steps * dt, "commit_mode": args.commit_mode,
            "branch_stride_chunks": (int(args.window_chunks)
                                     if args.commit_mode == "random" else 1),
            "commits_parent": args.commit_mode in ("argmax", "random_k"),
            "family_codes": family_codes, "plan_hash": plan["plan_hash"],
            "termination_terms": list(u.termination_manager.active_terms),
            "overrides": notes, "dr_fingerprint": fp, "run_seed": int(args.seed),
            "state_inventory_at_build": mirror.describe(),
            "commit_score": headroom_score.COMMIT_SCALES,
            "same_noise_control": bool(getattr(args, "same_noise", False)),
        }
        writer = hs.RecordWriter(str(out_cell / "states.h5.inprogress"), meta=meta)
        ep_writer = EpisodeLog(str(out_cell / "episodes.h5.inprogress"), meta=meta)

        rc = _roll(env, u, policy, cmd, crn, mirror, scorer, hook, alloc, writer, ep_writer,
                   plan=plan, family_codes=family_codes, args=args, group_size=group_size,
                   num_groups=num_groups, num_envs=num_envs, window_steps=window_steps,
                   replan_k=replan_k, dt=dt, key=key, required=REQUIRED_STATE)

        writer.close()
        ep_writer.close()
        (out_cell / "states.h5.inprogress").rename(out_cell / "states.h5")
        (out_cell / "episodes.h5.inprogress").rename(out_cell / "episodes.h5")
        wall = time.time() - t0
        info = {"complete": True, "schema_version": hs.SCHEMA_VERSION, "meta": meta,
                "coverage": coverage, "alloc": alloc.stats(), "rows": writer.num_written,
                "states": rc["states"], "episodes": ep_writer.num_written,
                "rounds": rc["rounds"], "sim_steps": rc["steps"],
                "skipped_rounds": rc["skipped"], "cadence_snaps": rc["cadence_snaps"],
                "uncommitted_group_rounds": rc["uncommitted_rounds"],
                "tile_retargets": hook.retargets,
                "noise_calls": crn.calls,
                "throughput": {"wall_clock_s": wall, "steps_per_s": rc["steps"] / max(1e-6, wall),
                               "device": str(u.device)},
                "written_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")}
        (out_cell / "headroom.json").write_text(json.dumps(info, indent=2, default=str))
        print(f"[headroom {key}] {rc['states']} states x {args.k} candidates in "
              f"{wall / 60:.1f} min ({rc['steps']} sim steps, "
              f"{info['throughput']['steps_per_s']:.1f}/s)")
        status["rc"] = 0

    _main()
    with contextlib.suppress(Exception):
        app.close()
    return status["rc"]


def assert_terminations(u) -> None:
    """Refuse to run unless the active terminations are the ones the spec requires.

    ``base_fallen`` must be present -- it IS the survival signal -- and no reference-graded
    term may be, because every one of those fires exactly when the generator emits an
    unusual chunk and would convert "this candidate is unusual" into "this candidate
    failed".  The tracker task already ships neither (``G1TrackerTerminationCfg`` is
    deliberately not derived from the teacher's), so this is a guard against a task swap
    rather than a fix.

    Raises:
        SystemExit: Naming what is active and what is forbidden.
    """
    forbidden = {"base_pose_error", "ee_pose_error", "joint_pos_error", "base_orientation",
                 "bad_contact", "shin_contact"}
    active = list(u.termination_manager.active_terms)
    bad = [t for t in active if t in forbidden]
    if bad:
        raise SystemExit(
            f"reference-graded termination(s) active: {bad}. They fire when the generator "
            "steers somewhere the clip does not go, which is the thing being measured. "
            f"Active terms: {active}"
        )
    if "base_fallen" not in active:
        raise SystemExit(
            f"base_fallen is not an active termination ({active}); it is the survival "
            "signal, and the derived `fell` column of the episode harness is not a "
            "substitute (it thresholds total tilt where the termination limits roll and "
            "pitch separately)."
        )
    print(f"[headroom] active terminations: {active}")


class EpisodeLog:
    """The per-episode side table, for Table 4 (episode success under the commit rule).

    Small and separate: the state table's unit is a branched state, and folding an episode
    result into every one of its rows would make ``success`` a clip-and-episode-weighted
    average of itself.
    """

    COLUMNS: tuple[tuple[str, str], ...] = (
        ("episode_id", "int64"), ("group_id", "int32"), ("clip_id", "int32"),
        ("source_clip_id", "int32"), ("family_code", "uint8"), ("terrain_id", "int32"),
        ("scene_tile_id", "int32"), ("start_step", "int32"), ("steps_survived", "int32"),
        ("last_usable_step", "int32"), ("terrain_start_step", "int32"),
        ("terrain_end_step", "int32"), ("termination_bits", "uint16"),
        ("event_observed", "uint8"), ("admin_censored", "uint8"), ("n_states", "int32"),
    )

    def __init__(self, path: str, *, meta: dict) -> None:
        """Create the file and its resizable columns."""
        import h5py  # noqa: PLC0415

        self._f = h5py.File(path, "w")
        self._f.attrs["schema_version"] = hs.SCHEMA_VERSION
        self._f.attrs["meta"] = json.dumps(meta, default=str)
        for name, dtype in self.COLUMNS:
            self._f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                   chunks=(512,))
        self.num_written = 0

    def append(self, rows: dict[str, np.ndarray]) -> None:
        """Append a batch; every column must be present."""
        n = int(len(rows["episode_id"]))
        if n == 0:
            return
        for name, _ in self.COLUMNS:
            dset = self._f[name]
            dset.resize((self.num_written + n,))
            dset[self.num_written:] = np.asarray(rows[name])
        self.num_written += n

    def close(self) -> None:
        """Flush and close."""
        self._f.attrs["num_rows"] = self.num_written
        self._f.close()


def _roll(env, u, policy, cmd, crn, mirror, scorer, hook, alloc, writer, ep_writer, *,
          plan, family_codes, args, group_size, num_groups, num_envs, window_steps,
          replan_k, dt, key, required) -> dict:
    """The round loop.

    One round is: align the parents to the re-plan cadence, force their round-start plan,
    copy it into the K branch envs, re-draw all but candidate 0, roll ``window_steps``, then
    score.  Returns the run counters.
    """
    import torch  # noqa: PLC0415

    device = u.device
    k = group_size - 1
    parents = torch.arange(num_groups, device=device) * group_size
    branch = (parents[:, None] + 1 + torch.arange(k, device=device)[None, :]).reshape(-1)
    all_envs = torch.arange(num_envs, device=device)
    parents_np = parents.detach().to("cpu").numpy()
    group_of_branch = torch.repeat_interleave(torch.arange(num_groups, device=device), k)
    # Candidates 1..k-1 are re-drawn; candidate 0 keeps the parent's plan by copy.
    redraw = branch.reshape(num_groups, k)[:, 1:].reshape(-1)
    redraw_src = torch.repeat_interleave(torch.arange(num_groups, device=device),
                                         max(k - 1, 0))

    #: Per-group bookkeeping.  ``skip`` covers the round after a parent reset: the step
    #: counter restarted mid-round so the cadence is off, and those states carry a
    #: null-masked generator history besides.
    skip = np.ones(num_groups, dtype=bool)
    round_index = np.zeros(num_groups, dtype=np.int64)
    episode_id = np.arange(num_groups, dtype=np.int64)
    next_episode_id = int(num_groups)
    ep_start_step = np.zeros(num_groups, dtype=np.int64)
    ep_states = np.zeros(num_groups, dtype=np.int64)

    state_id = 0
    steps = 0
    rounds = 0
    skipped = 0
    snapped = 0
    uncommitted = 0
    max_rounds = int(args.max_rounds) or 10 ** 9
    t_roll = time.time()

    obs = env.get_observations()
    with torch.inference_mode():
        # WARM UP one full window before taking the inventory.  Several buffers do not exist
        # until the first ``command_manager.compute`` has run -- ``_prev_served_node`` is
        # None at construction and the depth latency ring is one frame long -- so an
        # inventory taken at build time would silently omit exactly the state that makes a
        # branch the same situation as its parent.  A whole window (not one step) also
        # leaves the counters on the cadence and gets the generator past its null-history
        # region, so round 0 opens on a representative state.
        for _ in range(window_steps):
            obs, _, _, _ = env.step(policy(obs))
            steps += 1
        mirror.refresh()
        gap = mirror.missing(required)
        if gap:
            raise SystemExit(
                f"the state inventory is missing {gap}. Those buffers decide the next "
                "control step, so a branch that does not carry them is not the same "
                "situation as its parent. Inventory:\n  "
                + "\n  ".join(mirror.describe()))
        print(f"[headroom {key}] state inventory: {len(mirror.describe())} per-env buffers, "
              f"after a {window_steps}-step warm-up")

        while rounds < max_rounds:
            st = alloc.stats()
            if st["drained"] and st["done"] >= st["jobs"]:
                break
            crn.set_round(None, round_index)
            mirror.refresh()

            # -- align to the cadence ----------------------------------------------------
            n_par = u.episode_length_buf[parents]
            off = n_par % replan_k
            if bool((off != 0).any()):
                u.episode_length_buf[parents] = n_par - off
                snapped += int((off != 0).sum())

            # -- the round-start plan, and its copy into the branch envs -----------------
            crn.arm_branch(0)
            _force_replan(cmd, parents)
            snap_parent = mirror.capture(parents)
            mirror.restore(branch, snap_parent, src_index=group_of_branch)
            if redraw.numel():
                _force_replan(cmd, redraw)
                # _run_fm re-latches the anchor from robot.data, whose freshness right after
                # a state write is an IsaacLab internal. Taking the parent's values makes
                # every branch plan sit in exactly the parent's frame by construction.
                _copy_anchor(cmd, snap_parent, redraw, redraw_src)

            live_groups = np.flatnonzero(~skip)
            branch_this_round = live_groups.size > 0
            state_meta = _state_metadata(u, cmd, plan, family_codes, parents_np, args, crn,
                                         round_index, episode_id, rounds)
            if branch_this_round:
                ep_start_step[live_groups] = np.where(
                    ep_states[live_groups] == 0, state_meta["start_step"][live_groups],
                    ep_start_step[live_groups])

            scorer.begin(all_envs)
            in_window = torch.ones(num_envs, device=device, dtype=torch.bool)
            hook.parent_resets.clear()
            snap12 = None

            # The observation is now STALE for every env: a branch has just been handed a
            # different robot, and every parent has just been forced to re-plan, which
            # rewrites the reference-window terms. Reusing the previous step's `obs` would
            # compute the first action of the window from the pre-restore state -- a
            # one-step perturbation on every branch, every round, in exactly the quantity
            # this harness is trying to hold fixed.
            # compute() defaults to update_history=False, so this recompute does NOT push
            # the observation history a second time; only env.step() advances that ring.
            obs = env.get_observations()

            # -- roll the window ---------------------------------------------------------
            for w in range(1, window_steps + 1):
                if w % replan_k == 0:
                    chunk = w // replan_k
                    # The compute at the END of this step draws the plan used from the next
                    # step on, so arming happens before the step, not after it.
                    if chunk >= int(args.window_chunks):
                        crn.arm_branch(1)          # this one opens the NEXT round
                    else:
                        crn.arm_continuation(chunk)
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                steps += 1
                scorer.note_terminations(in_window)
                done_b = dones.to(torch.bool).reshape(-1)
                scorer.fold(in_window & ~done_b)
                in_window &= ~done_b
                if args.commit_mode in ("argmax", "random_k") and w == replan_k:
                    # NOT gated on branch_this_round. The skip exists so the report does not
                    # average over states whose generator history is still null-masked; it is
                    # not a reason to drop the COMMIT, and dropping it was expensive: a
                    # group resets every ~4.7 rounds, an un-committed round advances the
                    # parent 60 steps against a committed round's 12, so the skipped rounds
                    # carried ~57% of every episode's progress un-oracled and Table 4 was
                    # barely an oracle at all.
                    mirror.refresh()
                    snap12 = mirror.capture(branch)
            mirror.refresh()

            # -- score and write ---------------------------------------------------------
            if branch_this_round:
                rows, n_states = _build_rows(scorer, state_meta, live_groups, parents,
                                             branch, k, state_id)
                writer.append(rows)
                state_id += n_states
                ep_states[live_groups] += 1

            # -- commit ------------------------------------------------------------------
            # Every group EXCEPT the ones whose parent reset inside this window: their
            # snap12 is of the episode that has since ended, so committing it would
            # resurrect a dead episode on a tile the group has already left.
            if args.commit_mode in ("argmax", "random_k") and snap12 is not None:
                reset_now = {g for g, *_ in hook.parent_resets}
                eligible = np.array([g for g in range(num_groups) if g not in reset_now],
                                    dtype=np.int64)
                uncommitted += num_groups - int(eligible.size)
                _commit(args.commit_mode, mirror, scorer, snap12, parents, branch, k,
                        eligible, device, crn, round_index)

            # -- close out every episode whose parent ended during this window -----------
            # One row per RESET, in the order they happened: a parent can fall, respawn and
            # fall again inside one window, and collapsing those into one row undercounts
            # exactly the instant failures.
            resets = list(hook.parent_resets)
            if resets:
                rows = []
                for g, clip, bits, steps_surv, trunc in resets:
                    rows.append({
                        "group": g, "clip": clip, "bits": bits, "steps": steps_surv,
                        "trunc": trunc, "episode_id": int(episode_id[g]),
                        "start_step": int(ep_start_step[g]), "n_states": int(ep_states[g]),
                    })
                    episode_id[g] = next_episode_id
                    next_episode_id += 1
                    ep_states[g] = 0
                    ep_start_step[g] = 0
                ep_writer.append(_episode_rows(plan, family_codes, rows, censored=False))
            ended = sorted({g for g, *_ in resets})

            skip[:] = False
            skip[np.array(ended, dtype=np.int64)] = True
            round_index += 1
            round_index[np.array(ended, dtype=np.int64)] = 0
            if not branch_this_round:
                skipped += 1
            rounds += 1
            if rounds % 25 == 0:
                st = alloc.stats()
                rate = steps / max(1e-6, time.time() - t_roll)
                print(f"[headroom {key}] round {rounds}: {state_id} states, "
                      f"{st['done']}/{st['jobs']} episodes, {steps} sim steps "
                      f"({rate:.1f}/s)")

        # Episodes still running at the end are right-censored, not failures.
        live_n = u.episode_length_buf[parents].detach().to("cpu").numpy()
        rows = [{"group": g, "clip": int(alloc.group_clip[g]), "bits": 0,
                 "steps": int(live_n[g]), "trunc": False,
                 "episode_id": int(episode_id[g]), "start_step": int(ep_start_step[g]),
                 "n_states": int(ep_states[g])}
                for g in range(num_groups) if ep_states[g] > 0]
        if rows:
            ep_writer.append(_episode_rows(plan, family_codes, rows, censored=True))
    return {"states": state_id, "steps": steps, "rounds": rounds, "skipped": skipped,
            "cadence_snaps": int(snapped), "uncommitted_rounds": int(uncommitted)}


def _force_replan(cmd, ids) -> None:
    """Re-run generator inference for ``ids`` under whatever noise phase is armed.

    Idempotent where the cadence has just fired: the conditioning has not moved since and
    the armed seed is the same, so the forced call reproduces the natural one.
    """
    if ids.numel() == 0:
        return
    cmd._run_fm(_mask(cmd, ids), cmd._history.null_mask())


def _copy_anchor(cmd, snap: dict, dst_ids, src_index) -> None:
    """Restore the plan anchor from a parent snapshot after a forced re-inference.

    Raises:
        SystemExit: If the anchor is not in the state inventory, which would leave the
            branch plans anchored on whatever ``robot.data`` held after the state write.
    """
    for name in ("cmd._anchor_pos", "cmd._anchor_quat"):
        if name not in snap["paths"]:
            raise SystemExit(f"{name} is not in the state inventory; branch plans would be "
                             "anchored on an unspecified pose.")
        saved = snap["slots"][snap["paths"].index(name)]
        getattr(cmd, name.split(".", 1)[1])[dst_ids] = saved[src_index]


def _mask(cmd, ids):
    """A ``[num_envs]`` bool mask from env indices, on the command's device."""
    import torch  # noqa: PLC0415

    m = torch.zeros(cmd.num_envs, dtype=torch.bool, device=cmd.device)
    m[ids] = True
    return m


def _state_metadata(u, cmd, plan, family_codes, parents_np, args, crn, round_index,
                    episode_id, branch_step) -> dict:
    """Per-group state identity, read off the parents at the round start."""
    g = parents_np.size
    clip_ids = cmd.clip_id.detach().to("cpu").numpy().astype(np.int64)[parents_np]
    start = cmd.start_step.detach().to("cpu").numpy().astype(np.int64)[parents_np]
    t_step = u.episode_length_buf.detach().to("cpu").numpy().astype(np.int64)[parents_np]
    info = [plan["clips"][int(c)] for c in clip_ids]
    get = lambda kk: np.array([i[kk] for i in info])  # noqa: E731
    clip_step = start + t_step
    return {
        "episode_id": np.asarray(episode_id, dtype=np.int64),
        "group_id": np.arange(g, dtype=np.int64),
        "branch_step": np.full(g, int(branch_step), dtype=np.int64),
        "t_step": t_step, "clip_step": clip_step, "clip_id": clip_ids,
        "source_clip_id": get("source_clip_id"), "family_code": np.array(
            [family_codes[i["family"]] for i in info]),
        "terrain_id": get("terrain_id"), "scene_tile_id": get("column"),
        "start_step": start, "terrain_start_step": get("terrain_start_step"),
        "terrain_end_step": get("terrain_end_step"),
        "last_usable_step": get("last_usable_step"),
        "reset_source": np.full(g, hs.RESET_SOURCE_CODES[
            "pinned" if int(args.start_step) >= 0 else "sampled"], dtype=np.uint8),
        "phase_code": hs.phase_code(clip_step, get("terrain_start_step"),
                                    get("terrain_end_step")),
        "commit_mode_code": np.full(g, hs.COMMIT_MODE_CODES[args.commit_mode],
                                    dtype=np.uint8),
        "state_seed": np.array([crn.state_seed(i, int(round_index[i])) & ((1 << 63) - 1)
                                for i in range(g)], dtype=np.int64),
    }


def _build_rows(scorer, meta: dict, live_groups: np.ndarray, parents, branch, k: int,
                state_id0: int) -> tuple[dict, int]:
    """Assemble the ``(state, candidate)`` rows for the groups branched this round."""
    import torch  # noqa: PLC0415

    device = parents.device
    lg = torch.as_tensor(live_groups, device=device, dtype=torch.long)
    n_states = int(lg.numel())
    cand = scorer.harvest(branch.reshape(-1, k)[lg].reshape(-1))
    par = scorer.harvest(parents[lg])

    rows: dict[str, np.ndarray] = {}
    per_state = k + 1
    for name, dtype in hs.STATE_COLUMNS:
        col = (np.arange(state_id0, state_id0 + n_states, dtype=np.int64)
               if name == "state_id" else np.asarray(meta[name])[live_groups])
        rows[name] = np.repeat(col, per_state).astype(dtype)

    cand_idx = np.tile(np.append(np.arange(k, dtype=np.int64), hs.PARENT_ROW), n_states)
    rows["candidate_idx"] = cand_idx.astype(np.uint8)
    rows["is_parent_rollout"] = (cand_idx == hs.PARENT_ROW).astype(np.uint8)
    for name, dtype in hs.CANDIDATE_COLUMNS:
        if name in ("candidate_idx", "is_parent_rollout"):
            continue
        rows[name] = _interleave(cand[name], par[name], n_states, k).astype(dtype)
    for chan in hs.ERROR_CHANNELS:
        rows[f"err_{chan}"] = _interleave(cand[f"err_{chan}"], par[f"err_{chan}"],
                                          n_states, k).astype("float32")
    for chan in hs.COST_CHANNELS:
        rows[chan] = _interleave(cand[chan], par[chan], n_states, k).astype("float32")
    return rows, n_states


def _interleave(cand: np.ndarray, parent: np.ndarray, n_states: int, k: int) -> np.ndarray:
    """Weave ``[n_states*k]`` candidate values with ``[n_states]`` parent values.

    Row order matches ``candidate_idx``: the ``k`` candidates, then the parent, per state.
    """
    cand = np.asarray(cand)
    out = np.empty(n_states * (k + 1), dtype=np.result_type(cand, np.asarray(parent)))
    grid = out.reshape(n_states, k + 1)
    grid[:, :k] = cand.reshape(n_states, k)
    grid[:, k] = np.asarray(parent)
    return out


def _commit(mode: str, mirror, scorer, snap12: dict, parents, branch, k: int,
            live_groups: np.ndarray, device, crn, round_index) -> None:
    """Move each group's parent onto one candidate's one-chunk-in state.

    ``argmax`` picks the best-scoring candidate; ``random_k`` picks uniformly at random from
    the same K and is otherwise identical, which is what makes it the control -- the parent is
    restored the same number of times, at the same cadence, from the same pool.

    The whole group moves, not just the parent: the branch envs are about to be overwritten
    from the parent anyway, and leaving them where they were would mean the next round's
    restore is the only thing keeping them coherent.
    """
    import torch  # noqa: PLC0415

    num_groups = int(parents.numel())
    cand = scorer.harvest(branch)
    score = headroom_score.commit_score(cand).reshape(num_groups, k)
    if mode == "random_k":
        # Derived from the same (run, group, round) address as everything else, so the
        # control's choices are reproducible and independent of call order.
        best = np.array([headroom_noise.derive(crn.run_seed, g, int(round_index[g]), 0xC0DE)
                         % k for g in range(num_groups)], dtype=np.int64)
    else:
        best = np.argmax(score, axis=1)
    lg = np.asarray(live_groups, dtype=np.int64)
    if lg.size == 0:
        return
    # snap12 was captured over ``branch`` in order, so group g's candidate j is row g*k + j.
    rows = torch.as_tensor(lg * k + best[lg], device=device, dtype=torch.long)
    lg_t = torch.as_tensor(lg, device=device, dtype=torch.long)
    dst = torch.cat([parents[lg_t], branch.reshape(-1, k)[lg_t].reshape(-1)])
    src = torch.cat([rows, torch.repeat_interleave(rows, k)])
    mirror.restore(dst, snap12, src_index=src)


def _episode_rows(plan, family_codes, rows: list[dict], *, censored: bool) -> dict:
    """One output row per entry in ``rows``, which is one per parent RESET.

    Args:
        plan: The plan dict, for the clip's family and terrain span.
        family_codes: Family name -> code.
        rows: Dicts from the round loop, each carrying the ended episode's own terminal
            state rather than a per-group latest value.
        censored: True for the end-of-run flush, where nothing was observed.

    Returns:
        The column batch for :class:`EpisodeLog`.
    """
    info = [plan["clips"][int(r["clip"])] for r in rows]
    get = lambda kk: np.array([i[kk] for i in info])  # noqa: E731
    n = len(rows)
    trunc = np.array([bool(r["trunc"]) for r in rows], dtype=bool)
    return {
        "episode_id": np.array([r["episode_id"] for r in rows], dtype=np.int64),
        "group_id": np.array([r["group"] for r in rows], dtype=np.int32),
        "clip_id": np.array([r["clip"] for r in rows], dtype=np.int32),
        "source_clip_id": get("source_clip_id").astype(np.int32),
        "family_code": np.array([family_codes[i["family"]] for i in info], dtype=np.uint8),
        "terrain_id": get("terrain_id").astype(np.int32),
        "scene_tile_id": get("column").astype(np.int32),
        "start_step": np.array([r["start_step"] for r in rows], dtype=np.int32),
        "steps_survived": np.array([r["steps"] for r in rows], dtype=np.int32),
        "last_usable_step": get("last_usable_step").astype(np.int32),
        "terrain_start_step": get("terrain_start_step").astype(np.int32),
        "terrain_end_step": get("terrain_end_step").astype(np.int32),
        "termination_bits": np.array([r["bits"] for r in rows], dtype=np.uint16),
        "event_observed": ((~trunc) & (not censored)).astype(np.uint8),
        "admin_censored": np.full(n, int(censored), dtype=np.uint8),
        "n_states": np.array([r["n_states"] for r in rows], dtype=np.int32),
    }


def stage_restore_probe(args) -> int:
    """Measure what the state copy actually reproduces, and how fast the rest amplifies.

    The spec asks for a BLOCKING determinism test: save, roll, restore, roll again, assert
    the two trajectories match.  In IsaacLab that test cannot pass -- PhysX exposes no handle
    on contact manifolds, friction anchors or solver warm-start impulses -- so asserting it
    would only be a way of not measuring it.  This measures it instead, and separates the two
    possible causes, which have completely different consequences:

    * **A missing restorable buffer** would show up as a NON-ZERO difference in the
      controller-visible state immediately after the copy, before any stepping.  That is a
      bug in this harness and it is fixable, which would restore the experiment's resolution.
    * **Unrestorable solver state** shows up as a copy that is exact at step 0 and diverges
      from ~1e-7 upward once stepping resumes, because the constraint solve starts from a
      different initial guess and a contact-rich walker amplifies it.

    Both envs are fed IDENTICAL actions throughout, so the policy cannot contribute to the
    divergence and what is measured is the dynamics alone.
    """
    plan, manifest = eval_runner.load_run(args.out)
    cell = make_cell(args, plan)
    group_size = int(args.k) + 1
    num_groups = max(1, int(args.num_envs) // group_size)
    num_envs = num_groups * group_size

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    lp = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(lp)
    la, hydra_args = lp.parse_known_args([])
    la.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(la).app

    import gymnasium as gym  # noqa: PLC0415,E402
    import torch  # noqa: PLC0415,E402

    import headroom_state  # noqa: PLC0415,E402
    import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,PLC0415,E402
    import robot_rl.tasks  # noqa: F401,PLC0415,E402
    from isaaclab_rl.rsl_rl import (RslRlVecEnvWrapper,  # noqa: PLC0415,E402
                                    handle_deprecated_rsl_rl_cfg)
    from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: PLC0415,E402
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415,E402

    status = {"rc": 1}

    @hydra_task_config(cell["task"], "rsl_rl_cfg_entry_point")
    def _main(env_cfg, agent_cfg) -> None:
        """Copy one env onto another, then watch the two drift apart."""
        import importlib.metadata as md  # noqa: PLC0415

        eval_runner._apply_env_overrides(env_cfg, override_args(args, num_envs), plan,
                                         manifest, cell)

        # PhysX knobs under test. Enhanced determinism is documented as "improved
        # determinism at the expense of performance"; the iteration counts are the
        # accuracy lever. They are DIFFERENT properties -- a more accurate solve can still
        # reduce in a different order -- so they are separately switchable rather than
        # bundled, and both default off so the measured baseline stays the shipped config.
        if args.enhanced_determinism or args.solver_position_iters:
            phx = env_cfg.sim.physics
            if args.enhanced_determinism:
                phx.enable_enhanced_determinism = True
            if args.solver_position_iters:
                phx.min_position_iteration_count = int(args.solver_position_iters)
            print(f"[probe] PHYSX: enhanced_determinism="
                  f"{getattr(phx, 'enable_enhanced_determinism', None)} "
                  f"min_position_iteration_count="
                  f"{getattr(phx, 'min_position_iteration_count', None)}")
        alloc = GroupAllocator(plan["clips"], episodes_per_clip=1, num_groups=num_groups,
                               seed=int(args.seed))
        col = np.array([alloc.column_of(alloc.group_clip[e // group_size])
                        for e in range(num_envs)], dtype=np.int64)
        eval_runner._install_pin_list(
            env_cfg, [int(alloc.group_clip[e // group_size]) for e in range(num_envs)])
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
        env_raw = gym.make(cell["task"], cfg=env_cfg)
        u = env_raw.unwrapped
        eval_runner.assign_env_tiles(u.scene.terrain, np.zeros(num_envs, dtype=np.int64), col)
        env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
        rl = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        rl.load(manifest["policy"]["checkpoint"],
                load_cfg={"actor": True, "critic": False, "optimizer": False,
                          "iteration": True, "rnd": True})
        policy = rl.get_inference_policy(device=u.device)
        robot = u.scene.articulations["robot"]
        mirror = headroom_state.EnvStateMirror(u)

        # inference_mode for the same reason ``_roll`` uses it: the policy's output otherwise
        # requires grad, and the warp bridge refuses such a tensor with an error that names
        # __cuda_array_interface__ rather than the gradient.
        ctx = torch.inference_mode()
        ctx.__enter__()
        obs = env.get_observations()
        for _ in range(int(args.probe_warmup)):
            obs, _, _, _ = env.step(policy(obs))
        mirror.refresh()

        parents = torch.arange(num_groups, device=u.device) * group_size
        branch = (parents[:, None] + 1
                  + torch.arange(group_size - 1, device=u.device)[None, :]).reshape(-1)
        src = torch.repeat_interleave(torch.arange(num_groups, device=u.device),
                                      group_size - 1)

        def _spread(a, b, ids_a, ids_b):
            """Max |a - b| over the paired envs, as a float."""
            return float((a[ids_a] - b[ids_b]).abs().max())

        snap = mirror.capture(parents)
        mirror.restore(branch, snap, src_index=src)
        if args.probe_self_restore:
            # Write the parent's own state back onto the parent: numerically a no-op, every
            # buffer already holds exactly these values. What it is NOT a no-op in is PhysX,
            # which drops its warm start -- contact manifolds, friction anchors, solver
            # impulses -- whenever pose or joint state is written. The branches all pay that
            # cost and the parent does not, which is the ONE asymmetry left between them
            # once the copy is bit-exact, and it shows up as branch-vs-branch being exact
            # while branch-vs-parent never is. Paying it on both sides removes it.
            mirror.restore(parents, snap,
                           src_index=torch.arange(int(parents.numel()), device=u.device))

        # 1. IMMEDIATELY after the copy, before any stepping: is the copy BIT-IDENTICAL?
        #    Not "small" -- exactly zero. Nothing here has been stepped, so every quantity a
        #    branch reports is either a value that was copied or a value derived from copied
        #    values by the same kernel on the same device; a floating-point difference has
        #    nowhere to come from. Anything non-zero is a buffer this harness failed to
        #    carry, which is a fixable bug rather than a property of the simulator, and it
        #    is named so it can be fixed rather than characterised.
        rd = headroom_state._t
        src_of = parents.repeat_interleave(group_size - 1)

        def _delta(a, b) -> float:
            """Difference between two aligned buffers, 0.0 exactly when identical.

            Max |a - b| for floats, and a COUNT of differing elements for bool and integer
            buffers -- subtraction is undefined on bool tensors and wraps on unsigned ints,
            and the inventory carries both (episode counters, validity masks). Both readings
            share the one contract this gate needs: zero means identical.
            """
            if a.dtype == torch.bool or not a.is_floating_point():
                return float((a != b).sum())
            # Compare for EQUALITY first, then subtract. Two non-finite values that are
            # bit-identical are identical, and subtracting them is not: `inf - inf` and
            # `nan - nan` are both nan, so a plain difference reports a perfect copy as a
            # failure. Both occur here -- y_des and the CLF metrics are NaN until the first
            # command compute, and a height-scan ray that hits nothing stores inf.
            same = (a == b) | (a.isnan() & b.isnan())
            d = torch.where(same, torch.zeros_like(a), (a - b).abs())
            return float(d.max()) if d.numel() else 0.0

        def _diff(tensor) -> float:
            """``_delta`` between every branch env and its own parent, over one buffer."""
            t = rd(tensor)
            if not isinstance(t, torch.Tensor) or t.ndim == 0 or t.shape[0] != num_envs:
                return float("nan")
            return _delta(t[branch], t[src_of])

        checks: list[tuple[str, float]] = [
            ("sim.root_pose", _diff(robot.data.root_link_pose_w)),
            ("sim.root_vel", _diff(robot.data.root_link_vel_w)),
            ("sim.joint_pos", _diff(robot.data.joint_pos)),
            ("sim.joint_vel", _diff(robot.data.joint_vel)),
            # Derived, and the reason this probe exists: joint_acc finite-differences
            # against _previous_joint_vel, and projected_gravity_b is cached behind a
            # timestamp that a sim write does not advance.
            ("derived.joint_acc", _diff(robot.data.joint_acc)),
            ("derived.projected_gravity_b", _diff(robot.data.projected_gravity_b)),
        ]
        # Every buffer the mirror claims to carry, by its own inventory path -- so a slot
        # that is copied but WRONG is as visible as one that was never found.
        checks += [(f"slot.{sl.path}", _diff(sl.tensor)) for sl in mirror._slots]
        checks += [(f"cache.{path}", _diff(t)) for path, t in mirror._derived]
        for path, buf in mirror._buffers:
            if getattr(buf, "_buffer", None) is not None:
                b = buf._buffer
                checks.append((f"hist.{path}",
                               _delta(b[:, branch, :], b[:, src_of, :])))
        # The observation itself, EVERY group -- the critic's is not the policy's, and a
        # value estimate read off a mis-restored state is exactly what the selector would
        # rank on. compute() defaults to update_history=False, so this does not push the
        # history ring.
        om = u.observation_manager
        for gname, gval in om.compute().items():
            items = gval.items() if isinstance(gval, dict) else [("", gval)]
            for tname, tval in items:
                if not isinstance(tval, torch.Tensor):
                    continue
                label = f"obs.{gname}" + (f".{tname}" if tname else "")
                checks.append((label, _diff(tval)))
                if tname:
                    continue
                # A concatenated group is one tensor, so a non-zero names the GROUP and not
                # the term -- which is the difference between "the observation is wrong" and
                # a buffer to go and fix. Slice it back into terms by the manager's own
                # widths, and only if they account for the whole width, so a layout this
                # does not understand degrades to the group-level number rather than
                # mislabelling one term as another.
                names = om._group_obs_term_names.get(gname, [])
                dims = om._group_obs_term_dim.get(gname, [])
                widths = [int(np.prod(d)) for d in dims]
                if len(names) != len(widths) or sum(widths) != int(tval.shape[-1]):
                    continue
                off = 0
                for nm, w in zip(names, widths):
                    checks.append((f"obs.{gname}::{nm}", _diff(tval[..., off:off + w])))
                    off += w

        # The two PhysX-output buffers, measured but NOT gating: they rebind .data on every
        # read, so a copy cannot stick, and their source is solver output with no setter.
        # Reported for the same reason the divergence curve below is -- to say what the
        # restore genuinely cannot do, rather than to leave it unstated.
        unrestorable = [
            ("robot.data.body_com_acc_w", _diff(robot.data.body_com_acc_w)),
            ("robot.data.body_incoming_joint_wrench_b",
             _diff(robot.data.body_incoming_joint_wrench_b)),
        ]

        bad = [(n, v) for n, v in checks if not (v == 0.0)]
        print(f"[probe] BIT-EXACTNESS OF THE COPY: {len(checks) - len(bad)}/{len(checks)} "
              "buffers identical")
        for name, val in checks:
            if name.startswith(("sim.", "derived.", "obs.")):
                mark = "OK " if val == 0.0 else "BAD"
                print(f"[probe]   {mark} {name:<34} max |parent - branch| = {val:.3e}")
        if bad:
            print(f"[probe] *** {len(bad)} BUFFER(S) NOT BIT-IDENTICAL ***")
            for name, val in sorted(bad, key=lambda kv: -kv[1])[:25]:
                print(f"[probe]     {name:<44} {val:.6e}")
        else:
            print("[probe] the copy is bit-identical in every buffer checked.")
        print("[probe] UNRESTORABLE (PhysX solver output, no setter; read by nothing on the "
              "control path):")
        for name, val in unrestorable:
            print(f"[probe]   -- {name:<44} {val:.3e}")
        pre = {name: val for name, val in checks}
        pre_bad = {name: val for name, val in bad}

        # 2. Then step both with IDENTICAL actions and watch the divergence grow. The
        #    policy is evaluated on the PARENT's observation and broadcast, so nothing the
        #    network does can contribute.
        # Is the stack COLLIDING? Every branch is written to the parent's absolute world
        # pose, so a group is N robots occupying one volume. filter_collisions defaults True,
        # but the step-1 divergence scales with group size, so the flag is checked against
        # the physics rather than trusted: a robot standing on terrain loads its FEET, and
        # any force on a torso, elbow or backpack contact sensor is something else touching
        # it. Reported before the first step and after it.
        def _nonfoot_contact() -> dict:
            """Max contact-force magnitude on each non-foot sensor, parents vs branches."""
            out = {}
            for nm, sensor in u.scene.sensors.items():
                if "foot" in nm or not hasattr(sensor, "data"):
                    continue
                f = getattr(sensor.data, "net_forces_w", None)
                if f is None:
                    continue
                f = rd(f)
                if f.ndim < 2 or f.shape[0] != num_envs:
                    continue
                mag = f.reshape(num_envs, -1, 3).norm(dim=-1).amax(dim=1)
                out[nm] = (float(mag[parents].max()), float(mag[branch].max()))
            return out

        print("[probe] NON-FOOT CONTACT (N), max over envs -- parents / branches. Non-zero "
              "means the stacked robots are touching each other:")
        for nm, (pmax, bmax) in _nonfoot_contact().items():
            print(f"[probe]   {nm:<24} parents {pmax:10.3f}   branches {bmax:10.3f}")

        print("[probe] stepping with IDENTICAL actions -- divergence is dynamics only:")
        rows = []
        for step in range(1, int(args.probe_steps) + 1):
            act = policy(obs)
            act[branch] = act[parents].repeat_interleave(group_size - 1, dim=0)
            obs, _, _, _ = env.step(act)
            # PER ENV, then percentiles across envs. A max alone cannot distinguish "every
            # branch is coming apart" from "one branch is in a contact transition and the
            # rest are exact", and those two readings have opposite consequences for whether
            # the restore is usable at all.
            def _per_env(buf):
                t = rd(buf)
                d = (t[branch] - t[parents.repeat_interleave(group_size - 1)]).abs()
                return d.reshape(int(branch.numel()), -1).max(dim=1).values

            # BRANCH vs BRANCH, the clean control. Every branch in a group is a copy of the
            # same parent, so any two of them are bit-identical at step 0 and are handed the
            # same action. They also sit at the same pose in the same stack, so they share
            # whatever collision situation the group is in. Nothing distinguishes them --
            # not state, not action, not geometry, not neighbours. Divergence between THEM
            # is therefore the simulator's own cross-env nondeterminism and nothing else,
            # where branch-vs-parent additionally carries the parent's un-restored solver
            # state and its stale contact history.
            def _bb(buf, nb=group_size - 1):
                t = rd(buf)
                if nb < 2:
                    return None
                b = t[branch].reshape(num_groups, nb, -1)
                d = (b[:, 1:, :] - b[:, :1, :]).abs()
                return d.reshape(-1, d.shape[-1]).max(dim=1).values

            bb = _bb(robot.data.joint_pos)
            if step == 1 and bb is not None:
                # PER GROUP, because the aggregate cannot tell a fixed set of chaotic groups
                # from noise sprinkled over all of them, and those imply different things:
                # the first says the floor depends on what the robot is DOING and can be
                # attributed, the second says it is irreducible. The count staying pinned at
                # 43/70 for four steps and then dropping by exactly one group's worth of
                # pairs is what suggested the group is the unit.
                nb = group_size - 1
                per = bb.reshape(num_groups, nb - 1)
                # Contact state of each group's parent, as the candidate explanation: a
                # trivially-converging solve should be reproducible, a contact-rich one
                # order-dependent.
                tot = torch.zeros(num_envs, device=u.device)
                ncon = torch.zeros(num_envs, device=u.device)
                for nm, sensor in u.scene.sensors.items():
                    f = getattr(getattr(sensor, "data", None), "net_forces_w", None)
                    if f is None:
                        continue
                    f = rd(f)
                    if f.ndim < 2 or f.shape[0] != num_envs:
                        continue
                    mag = f.reshape(num_envs, -1, 3).norm(dim=-1)
                    tot += mag.sum(dim=1)
                    ncon += (mag > 1.0).sum(dim=1)
                print("[probe] PER-GROUP determinism at step 1 "
                      "(exact pairs / contact force on the parent):")
                print(f"[probe]   {'group':>5} {'exact':>7} {'max_div':>10} "
                      f"{'contactN':>10} {'nbodies':>8}")
                for g in range(num_groups):
                    ex = int((per[g] == 0).sum())
                    print(f"[probe]   {g:>5} {ex:>3}/{nb - 1:<3} "
                          f"{float(per[g].max()):>10.3e} "
                          f"{float(tot[int(parents[g])]):>10.1f} "
                          f"{int(ncon[int(parents[g])]):>8}")
            jpe = _per_env(robot.data.joint_pos)
            rpe = _per_env(robot.data.root_link_pose_w)
            q = torch.tensor([0.5, 0.9], device=jpe.device, dtype=torch.float64)
            jq = torch.quantile(jpe.double(), q).tolist()
            rq = torch.quantile(rpe.double(), q).tolist()
            jp, rp = float(jpe.max()), float(rpe.max())
            bbq = ([float(torch.quantile(bb.double(),
                                          torch.tensor(x, device=bb.device,
                                                       dtype=torch.float64)))
                    for x in (0.5, 0.9)] + [float(bb.max()), int((bb == 0).sum()),
                                            int(bb.numel())]
                   if bb is not None else [float("nan")] * 3 + [0, 0])
            rows.append({"step": step, "joint_pos": jp, "root_pose": rp,
                         "branch_vs_branch_p50": bbq[0], "branch_vs_branch_p90": bbq[1],
                         "branch_vs_branch_max": bbq[2], "branch_vs_branch_exact": bbq[3],
                         "branch_vs_branch_n": bbq[4],
                         "joint_pos_p50": jq[0], "joint_pos_p90": jq[1],
                         "root_pose_p50": rq[0], "root_pose_p90": rq[1],
                         "joint_pos_n_exact": int((jpe == 0).sum()),
                         "n_branch": int(branch.numel())})
            if step in (1, 2, 3, 5, 10, 20, 30, 40, 50, 60):
                print(f"[probe]   step {step:>3}  BRANCH-vs-BRANCH p50 {bbq[0]:.3e} "
                      f"max {bbq[2]:.3e}  exact {bbq[3]}/{bbq[4]}")
                print(f"[probe]   step {step:>3}  d(joint_pos) p50 {jq[0]:.3e} "
                      f"p90 {jq[1]:.3e} max {jp:.3e}   d(root_pose) p50 {rq[0]:.3e} "
                      f"max {rp:.3e}   exact: {int((jpe == 0).sum())}/{int(branch.numel())}")
        out = pathlib.Path(args.out) / "headroom" / "restore_probe.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"immediately_after_copy": pre,
                                   "not_bit_identical": pre_bad,
                                   "unrestorable": dict(unrestorable),
                                   "n_buffers_checked": len(pre),
                                   "divergence": rows,
                                   "num_groups": num_groups, "k": int(args.k),
                                   "warmup": int(args.probe_warmup)}, indent=2))
        ctx.__exit__(None, None, None)
        print(f"[probe] -> {out}")
        print("[probe] READ: a non-zero 'immediately after the copy' means a buffer is "
              "MISSING from the inventory and the harness is fixable. Exact there, growing "
              "from ~1e-7 once stepping resumes, means unrestorable solver state.")
        # Non-zero exit on a non-exact copy: this stage is a GATE, not a report. Everything
        # downstream assumes a branch starts as the same robot in the same situation, and a
        # run that proceeds past a failed copy produces numbers that look ordinary.
        status["rc"] = 1 if pre_bad else 0

    _main()
    with contextlib.suppress(Exception):
        app.close()
    return status["rc"]
