# Copyright 2026 Zachary Olkin. All rights reserved.

"""Run the V(s) best-of-K selector inside IsaacLab, and measure what it is worth.

At every re-plan the generator is sampled ``K`` times from one state, each candidate is
rolled one chunk, the trained value model scores the states they reach, and the best one
becomes the group's state.  Unlike the MuJoCo selector this runs ``num_groups`` episodes at
once, and unlike ``headroom_runner`` it carries the choice forward to the end of the episode
rather than scoring a fixed window -- so what comes out is episode success, the same quantity
``report.py`` prints for the ``fm`` arm.

**The group is ``1 + K`` envs and NONE of them is a parent.**  ``headroom_runner`` branches
off a parent that is never restored, and the restore probe showed that this makes the parent a
different situation: two branches restored from one state are bit-identical to each other
(median 0.000e+00) while a branch is never bit-identical to its parent.  Here the group's
state lives in whichever env WON the last cycle, and every env -- including the winner's next
carrier -- is written at the start of every cycle, so no env is ever the odd one out.  Slot 0
is a deliberate twin of candidate 0: it draws the same noise, so its disagreement with
candidate 0 is a live per-cycle reading of the restore floor at no extra cost.

**Why the same-noise arm is not a sanity check but half the experiment.**  Isaac reproduces
exactly across envs given identical state and identical writes, but a humanoid in contact
amplifies a one-ulp difference in GPU reduction order to ~1e-2 rad within a single 20 ms step,
and about a third of candidate pairs diverge that way.  A selector cannot tell that apart from
a better chunk: it picks the survivor either way.  So ``--same_noise`` serves every candidate
ONE draw, which makes every candidate the same plan and any resulting best-of-K rise an
artifact of the simulator.  Read the real arm's rise against it, never on its own -- the
previous version of this experiment died precisely because its floor turned out to exceed its
signal, and it only found out because the control was run.
"""

from __future__ import annotations

import numpy as np
import torch


class GroupEpisodes:
    """Per-group episode bookkeeping for a selector run.

    One group runs one episode at a time across ``1 + K`` envs, so neither the env-indexed
    accumulator nor the termination manager can be read directly: they would report ``K + 1``
    outcomes for the one episode the group actually ran, all but one of them belonging to a
    candidate that was discarded a chunk later.  This tracks the CHAIN -- the sequence of
    winners -- and emits exactly one row per group episode.
    """

    def __init__(self, num_groups: int, group_size: int) -> None:
        """Start every group on its first episode.

        Args:
            num_groups: Number of concurrent groups.
            group_size: ``1 + K``; slot 0 is candidate 0's twin.
        """
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        #: Env index holding each group's live state -- the last cycle's winner.
        self.carrier = np.arange(num_groups, dtype=np.int64) * group_size
        self.cycles = np.zeros(num_groups, dtype=np.int64)
        self.episode_id = np.arange(num_groups, dtype=np.int64)
        self.rows: list[dict] = []

    def envs_of(self, group: int) -> np.ndarray:
        """Every env index belonging to one group."""
        base = group * self.group_size
        return np.arange(base, base + self.group_size, dtype=np.int64)

    def finish(self, group: int, *, verdict: str, clip_id: int, family: int,
               next_episode_id: int) -> None:
        """Record one completed group episode and start the next.

        Args:
            group: Group index.
            verdict: ``"success"``, ``"fall"`` or ``"timeout"``.
            clip_id: The clip this episode ran.
            family: Family code.
            next_episode_id: Id to hand the group's next episode.
        """
        self.rows.append({"episode_id": int(self.episode_id[group]), "group": int(group),
                          "verdict": verdict, "clip_id": int(clip_id),
                          "family_code": int(family), "cycles": int(self.cycles[group])})
        self.cycles[group] = 0
        self.episode_id[group] = int(next_episode_id)


def choose(scores: torch.Tensor, alive: torch.Tensor, num_groups: int,
           group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each group's best LIVE candidate.

    **A terminated candidate must be masked out explicitly rather than left to the value
    model.**  Isaac auto-resets an env the moment it terminates, so a candidate that fell
    during the window is, by the time it is scored, standing at a fresh spawn -- a state the
    critic has every reason to like.  Ranking on the raw score would therefore prefer exactly
    the candidates that just died.

    Args:
        scores: ``[num_envs]`` value, higher is better.
        alive: ``[num_envs]`` bool, False for any env that terminated during the window.
        num_groups: Number of groups.
        group_size: Envs per group.

    Returns:
        ``(winner_env, group_alive)`` -- the chosen env index per group, and whether the
        group had any live candidate at all.  Where it had none the winner is arbitrary and
        the caller must end that episode.
    """
    s = scores.reshape(num_groups, group_size).clone()
    a = alive.reshape(num_groups, group_size)
    group_alive = a.any(dim=1)
    # -inf, not a large negative: a masked candidate must lose to every live one regardless
    # of how the value model is scaled, and a magic constant would silently stop working if
    # the critic's output range ever moved.
    s = torch.where(a, s, torch.full_like(s, float("-inf")))
    local = s.argmax(dim=1)
    base = torch.arange(num_groups, device=scores.device) * group_size
    return base + local, group_alive


class WindowOutcome:
    """What each env did during one chunk, by TERMINATION REASON rather than by ``dones``.

    ``dones`` alone cannot run this loop.  An episode that reaches ``clip_end`` upright has
    SUCCEEDED and an episode that trips ``base_fallen`` has failed, and both arrive as a set
    bit in the same tensor -- so a selector reading ``dones`` would retire a group at the
    finish line and score it as a fall.  Isaac has also already auto-reset the env by then, so
    the reason cannot be recovered afterwards from the robot's state; it has to be latched
    during the window, on the step it fires.
    """

    def __init__(self, num_envs: int, device) -> None:
        """Allocate the per-env latches.

        Args:
            num_envs: Total envs.
            device: Torch device.
        """
        import torch  # noqa: PLC0415

        self.fell = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=device)
        #: Step within this window at which the env first entered the goal / first fell,
        #: or -1. STEP resolution, not per-window: `terminal_reward` scores on which came
        #: FIRST, and its docstring records that distinction as worth 17 points on a
        #: 298-episode rollout -- larger than any effect this harness is trying to resolve.
        #: A window-level "arrived and also fell" flag cannot express it.
        self.first_goal = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.first_fall = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._step = 0

    def note(self, env, in_goal=None) -> None:
        """Latch this step's terminations and goal membership.

        Args:
            env: The unwrapped env, for the termination manager.
            in_goal: ``[num_envs]`` bool, whether each base is inside its goal region now,
                or ``None`` to skip goal accounting.
        """
        import torch  # noqa: PLC0415

        self._step += 1
        tm = env.termination_manager
        active = set(tm.active_terms)
        if "base_fallen" in active:
            f = tm.get_term("base_fallen").reshape(-1)
            self.first_fall = torch.where((self.first_fall < 0) & f,
                                          torch.full_like(self.first_fall, self._step),
                                          self.first_fall)
            self.fell |= f
        for name in ("clip_end", "time_out"):
            if name in active:
                self.finished |= tm.get_term(name).reshape(-1)
        if in_goal is not None:
            g = in_goal.reshape(-1)
            self.first_goal = torch.where((self.first_goal < 0) & g,
                                          torch.full_like(self.first_goal, self._step),
                                          self.first_goal)

    def arrived_before_falling(self):
        """``[num_envs]`` bool: entered the goal before ever crossing the fall predicate.

        This IS ``rollout_outcomes.terminal_reward``'s rule, restated on a window rather than
        an episode: an episode that reaches the goal and then falls in the run-out did the
        task; one that falls and is later dragged over the line did not.
        """
        return (self.first_goal >= 0) & ((self.first_fall < 0)
                                         | (self.first_goal < self.first_fall))

    def alive(self):
        """Envs still eligible to be chosen: neither fallen nor finished."""
        return ~(self.fell | self.finished)


def group_verdict(outcome: WindowOutcome, group: int, group_size: int) -> str | None:
    """The group's episode verdict after one chunk, or ``None`` if it continues.

    A group's episode is over when no candidate is still running.  It SURVIVED if any
    candidate reached the end of its clip upright, and failed only if every candidate fell --
    the order matters, because a chunk in which one candidate finishes and the rest fall is a
    completed episode, not a wipeout.

    Args:
        outcome: This window's latches.
        group: Group index.
        group_size: Envs per group.
        
    Returns:
        ``"survived"``, ``"fell"``, or ``None`` to continue.
    """
    lo, hi = group * group_size, (group + 1) * group_size
    if bool(outcome.finished[lo:hi].any()):
        return "survived"
    if bool(outcome.alive()[lo:hi].any()):
        return None
    return "fell"


def slot0_verdict(outcome: WindowOutcome, group: int, group_size: int) -> str | None:
    """The verdict of candidate 0 ALONE, ignoring every other env in the group.

    The structural control needs the group's fate to be one env's fate, exactly as it is
    when ``k=0``.  Reading :func:`group_verdict` instead would still let the group live on
    whichever sibling happened to survive -- which is selection, reintroduced through the
    outcome rule rather than through the scorer.
    """
    e = group * group_size
    if bool(outcome.finished[e]):
        return "survived"
    return "fell" if bool(outcome.fell[e]) else None


class BranchOnce:
    """One branch point per episode, every candidate rolled to TERMINATION.

    This is the HEADROOM measurement, and it is a different question from the selector's.
    The selector asks whether a critic can rank the generator's samples; this asks whether
    the samples are worth ranking -- whether the distribution at a state the policy actually
    visits contains a chunk that leads to success. An oracle answers the second and bounds
    the first: no critic can beat picking the candidate that actually worked.

    **Three things make it an honest oracle rather than a leak.**

    * The candidates differ in ONE draw. Before the branch every env in the group is served
      candidate 0's noise, so they are the same rollout; at the branch they are served K
      distinct draws; after it they share ONE continuation draw per group
      (``CrnNoise.arm_continuation``). Without that last part a candidate could win on a lucky
      continuation and the number would be about downstream noise, not about the chunk.
    * Nothing is restored after the branch. Each candidate runs its own episode to its own
      termination, so its outcome IS what that chunk led to rather than an estimate of it.
    * The K=1 reference is candidate 0 OF THE SAME BRANCH, not a separate run, so best-of-K
      and its baseline are paired on the state as well as on the clip.

    Slot 0 duplicates candidate 0 by construction (``CrnNoise.candidate`` maps slots 0 and 1
    to the same index), which is kept rather than removed: two envs handed one draw from one
    state are a free per-episode reading of the restore floor.
    """

    def __init__(self, num_groups: int, group_size: int, rng) -> None:
        """Start every group pre-branch.

        Args:
            num_groups: Concurrent groups.
            group_size: ``1 + K``.
            rng: ``numpy`` generator, for the per-episode branch cycle.
        """
        self.num_groups = int(num_groups)
        self.group_size = int(group_size)
        self.rng = rng
        self.branched = np.zeros(num_groups, dtype=bool)
        #: Cycle index at which each group branches. Sampled per episode so the measurement
        #: spans the episode rather than reporting the headroom at one privileged phase.
        self.branch_at = np.zeros(num_groups, dtype=np.int64)
        self.post = np.zeros(num_groups, dtype=np.int64)
        self.done = np.zeros((num_groups, group_size), dtype=bool)
        self.ok = np.zeros((num_groups, group_size), dtype=bool)

    def arm_episode(self, group: int, lo: int = 1, hi: int = 8) -> None:
        """Choose where this group's episode will branch, and clear its outcome slate."""
        self.branched[group] = False
        self.branch_at[group] = int(self.rng.integers(lo, hi + 1))
        self.post[group] = 0
        self.done[group] = False
        self.ok[group] = False

    def result(self, group: int) -> dict:
        """Best-of-K against candidate 0, for one finished group episode.

        ``k1`` is SLOT 1, not slot 0: slots 0 and 1 share candidate 0's draw, and using the
        duplicate as the baseline would make the pair differ by simulator noise alone.
        """
        row = self.ok[group]
        cands = row[1:]
        return {"k1": bool(row[1]) if row.size > 1 else bool(row[0]),
                "best_of_k": bool(cands.any()) if cands.size else bool(row[0]),
                "n_success": int(cands.sum()), "n_cand": int(cands.size),
                "twin_agrees": bool(row[0] == row[1]) if row.size > 1 else True,
                "branch_at": int(self.branch_at[group])}


class GoalRefreshHook:
    """Re-resolve the goal anchor AFTER the clip reset has committed, not before.

    ``refresh_goal_anchor`` reads the clip's own endpoint through the command's reset-time
    placement, so calling it from OUTSIDE the sim loop -- at the moment the allocator hands a
    group its next clip -- resolves the anchor against the clip the env is still standing on.
    Every episode is then scored against its predecessor's goal. Measured: it put the
    do-nothing control at 0.788 against an arrival rate of 0.676, i.e. above a bound it
    cannot exceed, which is what exposed it.

    ``HeadroomResetHook`` solves the same ordering problem and additionally owns episode
    advancement, which this loop does itself and differently (its episode is one env's; ours
    is a chain across K). So this hook does the goal half only.

    The signature is restated explicitly rather than taken as ``**kwargs`` because
    ``ManagerBase._resolve_common_term_cfg`` matches parameter names statically and rejects a
    var-keyword function -- the same reason ``eval_runner.EvalResetHook`` does it.
    """

    def __init__(self, base_func, *, accum, cmd, family_of_clip: list[str]) -> None:
        """Wrap the clip reset.

        Args:
            base_func: The original ``reset_on_clip`` function.
            accum: The ``EpisodeAccumulator`` holding the goal columns.
            cmd: The ``FmTrackingCommand``, read for the COMMITTED clip id.
            family_of_clip: Family name per clip id, as the plan spells them.
        """
        import inspect  # noqa: PLC0415

        self.base_func = base_func
        self._defaults = {
            name: prm.default
            for name, prm in inspect.signature(base_func).parameters.items()
            if prm.default is not inspect.Parameter.empty
        }
        self.accum = accum
        self.cmd = cmd
        self.family_of_clip = list(family_of_clip)
        self.refreshes = 0

    def __call__(self, env, env_ids, command_name="traj_ref",
                 base_frame_name="pelvis_link", asset_cfg=None, pin_clip_ids=None,
                 pin_start_step=None, spawn_z_offset=0.0, clip_draw="random"):
        """Run the real reset, then resolve the goal for whatever clip it committed."""
        import torch  # noqa: PLC0415

        if asset_cfg is None:
            asset_cfg = self._defaults.get("asset_cfg")
        if pin_start_step is None:
            pin_start_step = self._defaults.get("pin_start_step")
        out = self.base_func(env, env_ids, command_name=command_name,
                             base_frame_name=base_frame_name, asset_cfg=asset_cfg,
                             pin_clip_ids=pin_clip_ids, pin_start_step=pin_start_step,
                             spawn_z_offset=spawn_z_offset, clip_draw=clip_draw)
        if getattr(self.accum, "_gr", None) is None:
            return out
        ids = (torch.arange(int(env.num_envs), device=self.cmd.clip_id.device)
               if env_ids is None else env_ids)
        if ids.numel() == 0:
            return out
        names = [self.family_of_clip[int(c)] for c in self.cmd.clip_id[ids].tolist()]
        self.accum.set_goal_families(names, env_ids=ids)
        self.accum.refresh_goal_anchor(ids)
        self.refreshes += 1
        return out


def oracle_scores(outcome: WindowOutcome):
    """Rank a survivor above a faller, and nothing finer.

    The GATE does not need the trained critic.  What it asks is how much a selector can
    manufacture out of simulator noise alone, and an oracle that keeps whichever candidate
    happened to survive is the BEST any selector could do on that noise -- so its
    same-noise rise is an upper bound on every real selector's.  If the bound is small,
    no critic can manufacture much and the experiment is safe; if it is large, the bound
    alone does not condemn V(s), and the realistic number needs V.

    Cheap in a way that matters here: it is available before the value model is wired in,
    so the go/no-go does not wait on it.
    """
    import torch  # noqa: PLC0415

    return torch.where(outcome.alive(), torch.ones_like(outcome.fell, dtype=torch.float32),
                       torch.zeros_like(outcome.fell, dtype=torch.float32))


def summarise(rows: list[dict]) -> dict:
    """Survival rate overall and per family, from the group-episode rows."""
    if not rows:
        return {"n": 0, "survived": 0.0, "by_family": {}}
    surv = [1.0 if r["verdict"] == "success" else 0.0 for r in rows]
    by: dict[int, list[float]] = {}
    for r, s in zip(rows, surv):
        by.setdefault(int(r["family_code"]), []).append(s)
    return {"n": len(rows), "survived": float(np.mean(surv)),
            "by_family": {k: {"n": len(v), "survived": float(np.mean(v))}
                          for k, v in sorted(by.items())}}


def stage_select(args) -> int:
    """Run the selector to the end of each episode and write the survival table."""
    import argparse  # noqa: PLC0415
    import contextlib  # noqa: PLC0415
    import os  # noqa: PLC0415
    import json  # noqa: PLC0415
    import pathlib  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    import headroom_runner as hr  # noqa: PLC0415
    import runner as eval_runner  # noqa: PLC0415

    plan, manifest = eval_runner.load_run(args.out)
    cell = hr.make_cell(args, plan)
    plan_fm_path = str(plan["generators"][args.fm_label]["path"])
    group_size = int(args.k) + 1
    num_groups = max(1, int(args.num_envs) // group_size)
    num_envs = num_groups * group_size
    print(f"[select] {num_groups} groups x {group_size} envs = {num_envs}; "
          f"scorer={args.scorer}  same_noise={bool(args.same_noise)}")

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    lp = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(lp)
    la, hydra_args = lp.parse_known_args([])
    la.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(la).app

    import gymnasium as gym  # noqa: PLC0415,E402
    import torch  # noqa: PLC0415,E402

    import headroom_noise  # noqa: PLC0415,E402
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
        """Boot the env, install the selector, and roll until the job list drains."""
        import importlib.metadata as md  # noqa: PLC0415

        eval_runner._apply_env_overrides(env_cfg, hr.override_args(args, num_envs), plan,
                                         manifest, cell)
        alloc = hr.GroupAllocator(plan["clips"], episodes_per_clip=int(args.episodes_per_clip),
                                  num_groups=num_groups, seed=int(args.seed))
        col = np.array([alloc.column_of(alloc.group_clip[e // group_size])
                        for e in range(num_envs)], dtype=np.int64)
        eval_runner._install_pin_list(
            env_cfg, [int(alloc.group_clip[e // group_size]) for e in range(num_envs)])
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, md.version("rsl-rl-lib"))
        env_raw = gym.make(cell["task"], cfg=env_cfg)
        u = env_raw.unwrapped
        levels = np.zeros(num_envs, dtype=np.int64)
        eval_runner.assign_env_tiles(u.scene.terrain, levels, col)
        # THE TILE MUST FOLLOW THE CLIP. `assign_env_tiles` is a one-shot map built from the
        # groups' FIRST clips; when a group finishes and takes its next one, nothing moves
        # its envs to the new clip's tile. With many groups each runs an episode or two and
        # the error is invisible; with few groups each runs a dozen, so almost every episode
        # after the first runs the right motion on the WRONG terrain. Measured without this:
        # 0.864 at 90 groups against 0.697 at 10, with selection making no difference to
        # either -- the whole gap was this.
        #
        # `HeadroomResetHook` does the retarget but ALSO owns episode advancement, and it
        # keys that on SLOT 0 RESETTING. That is right for stage_run, whose parent is the
        # episode, and wrong here: a selector's episode ends when the CHAIN ends, and slot 0
        # may fall mid-episode while a sibling carries the group on. Installing it both
        # double-advanced the allocator (132 episodes became 41) and would have retired
        # groups mid-episode. So the retarget is driven from this loop's own boundary.
        pin_params = u.event_manager.get_term_cfg("reset_on_clip").params["pin_clip_ids"]

        def retarget(group: int, clip: int) -> None:
            """Move a group's envs onto ``clip``'s tile, and pin them to it."""
            column = alloc.column_of(clip)
            moved = False
            for local in range(group_size):
                e = group * group_size + local
                pin_params[e] = int(clip)
                moved = moved or int(col[e]) != column
                col[e] = column
            if moved:
                eval_runner.assign_env_tiles(u.scene.terrain, levels, col)
            # The GOAL half deliberately does NOT happen here: the env has not reset onto
            # the new clip yet, so the anchor would describe the previous one. GoalRefreshHook
            # does it from inside the reset, once the placement is committed.
        env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
        rl = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        rl.load(manifest["policy"]["checkpoint"],
                load_cfg={"actor": True, "critic": False, "optimizer": False,
                          "iteration": True, "rnd": True})
        policy = rl.get_inference_policy(device=u.device)
        device = u.device
        cmd = u.command_manager.get_term("traj_ref")
        mirror = headroom_state.EnvStateMirror(u)
        replan_k = int(getattr(cmd, "fm_update_every", 12))

        # ---- the critic, the goal region, and the proprioception ring ------------------
        # All three come from the ROLLOUT the critic was fitted on, not from defaults: the
        # goal tolerances are per family and calibrated, the ring's width and depth are the
        # fit's, and the normalizers are the generator's. Any of them silently substituted
        # puts the critic in a different input space while every shape still agrees.
        import accumulate  # noqa: PLC0415
        import rollout_outcomes as ro  # noqa: PLC0415
        from robot_rl.tasks.manager_based.robot_rl.mdp import goal_region  # noqa: PLC0415

        spec = goal_region.GoalRegionSpec.load(args.goal_regions)
        accum = accumulate.EpisodeAccumulator(u, dt=float(u.step_dt),
                                              fall_min_height=float(args.fall_min_height))
        accum.attach_goal_region(spec)
        ring = ro.ProprioRing(num_envs, device, ro.PROPRIO_HISTORY, ro.PROPRIO_DIM)

        # Installed AFTER attach_goal_region, so the hook sees a live goal spec. `cmd.clip_id`
        # indexes the eval SUBSET, which is the same order as plan["clips"], so a clip id maps
        # straight to a family name.
        goal_hook = GoalRefreshHook(
            u.event_manager.get_term_cfg("reset_on_clip").func, accum=accum, cmd=cmd,
            family_of_clip=[c["family"] for c in plan["clips"]])
        u.event_manager.get_term_cfg("reset_on_clip").func = goal_hook

        class _GoalView:
            """The two attributes ``OnlineValueScorer`` reads off a rollout tracker.

            The full ``RolloutEpisodeTracker`` also owns episode advancement and outcome
            harvesting, both of which this loop does itself and differently -- its episode is
            one env's, ours is a chain across K. So only the accumulator is shared.
            """

            def __init__(self, a) -> None:
                self.accum = a

        tracker = _GoalView(accum)

        # The value scorer is GONE, and `--scorer` offers `oracle` only. It read
        # `FlowValueModel.p_fail`, the critic's survival head, which no longer exists: the
        # fine-tune drops that head's loss, so it was untrained in every reported result and
        # was deleted with the rest of the two-head architecture.
        #
        # The ORACLE scorer is untouched, and it is the one that carries the measurement --
        # "keep whichever candidate survived" is the upper bound on what any selector could
        # manufacture from simulator noise, which is what this harness exists to bound.
        #
        # TODO(headroom-v): rebuild it on the surviving VALUE head (`FlowValueModel.quality`,
        # a return regression) if a learned scorer is wanted again. That is a different score
        # from P(fail) and not a drop-in: it ranks by expected return, not by failure risk.

        if args.steer_kp != 0.0:
            # Installed on the COMMAND, so the generator's conditioning, the reward and the
            # command_window observation all see the same turn. See set_yaw_rate_source.
            def _steer_source():
                """Yaw rate toward each env's goal, zero once inside the region."""
                down, lat, hz = goal_region.goal_frame_offsets(
                    accum.rd.root_pos_w, accum._goal_pos_w, accum._goal_yaw)
                return goal_region.steer_yaw_rate(
                    accum.rd.root_pos_w, accum.rd.root_quat_w, accum._goal_pos_w,
                    goal_region.in_goal(down, lat, hz, accum._goal_tol),
                    kp=float(args.steer_kp), wz_max=float(args.steer_wz_max))

            cmd.set_yaw_rate_source(_steer_source)
            print(f"[select] STEERING ON: kp={args.steer_kp} wz_max={args.steer_wz_max}. "
                  "This is OFF-DISTRIBUTION on the stair families, which the training "
                  "library never commands a turn on -- a steered stair number is a "
                  "different measurement, not a cleaner one.")

        crn = headroom_noise.CrnNoise(
            num_envs=num_envs, group_size=group_size, future_len=int(cmd.ctx.F),
            state_dim=int(cmd.ctx.D), device=device, run_seed=int(args.seed))
        if args.same_noise:
            # THE CONTROL. Every candidate is served candidate 0's draw, so every candidate
            # is the SAME PLAN and there is nothing to select between by construction. Any
            # best-of-K rise this arm shows was manufactured by the simulator, not by the
            # generator, and the real arm's rise means nothing until it is read against it.
            crn.force_candidate = 0
        cmd.set_noise_source(crn)

        all_envs = torch.arange(num_envs, device=device)
        group_of_env = torch.arange(num_envs, device=device) // group_size
        ge = GroupEpisodes(num_groups, group_size)
        # Same mapping stage_run uses, so a family code means the same thing in both tables.
        families = sorted({c["family"] for c in plan["clips"]})
        family_codes = {f: i for i, f in enumerate(families)}
        round_index = np.zeros(num_groups, dtype=np.int64)
        #: Groups whose work is done; they keep stepping (Isaac steps every env) but their
        #: episodes are no longer recorded, so the table holds one row per owed job.
        retired = np.zeros(num_groups, dtype=bool)
        #: Has the CHAIN entered the goal region before ever falling? Latched along the
        #: winner, because the chain's state is the winner's -- an arrival by a candidate
        #: that was then discarded is not something the robot did.
        chain_arrived = np.zeros(num_groups, dtype=bool)
        bo = None
        if args.branch_once:
            bo = BranchOnce(num_groups, group_size, np.random.default_rng(int(args.seed)))
            for g in range(num_groups):
                bo.arm_episode(g)
            # Per-ENV latches: after the branch each candidate runs its OWN episode to its own
            # termination, so arrival is no longer a property of a chain.
            arrived_env = np.zeros(num_envs, dtype=bool)
            done_env = np.zeros(num_envs, dtype=bool)
            print(f"[select] BRANCH-ONCE oracle: one branch point per episode, all "
                  f"{group_size - 1} candidates rolled to termination under a SHARED "
                  "continuation draw.")
        next_episode_id = int(num_groups)
        t0 = time.time()
        cycles = 0

        obs = env.get_observations()
        with torch.inference_mode():
            # One full chunk before the first branch: several buffers do not exist until the
            # first command compute has run, so an inventory taken at build time would omit
            # exactly the state that makes a candidate the same situation as its carrier.
            for _ in range(replan_k):
                obs, _, _, _ = env.step(policy(obs))
            mirror.refresh()
            print(f"[select] inventory: {len(mirror.describe())} per-env buffers")
            # SEED THE GOAL FOR EVERY GROUP'S FIRST EPISODE. `retarget` resolves the anchor
            # and the per-family tolerances, but it only runs when a group ADVANCES -- so
            # without this the first episode of every group is scored against a goal at the
            # world origin with zero tolerance, which can never be reached. At 45 groups over
            # 132 episodes that silently marked 34% of the sample as failures and put the
            # whole run 15 points under the episode harness on identical inputs.
            for g in range(num_groups):
                retarget(g, int(alloc.group_clip[g]))
            print(f"[select] goal anchors resolved for {num_groups} group(s)")

            while True:
                # `drained` means every owed episode has STARTED; the last ones are still
                # running, so both conditions are needed or the run stops with groups
                # mid-episode and their rows missing from the table.
                st = alloc.stats()
                if bool(retired.all()) or (st["drained"] and st["done"] >= st["jobs"]):
                    break
                if cycles >= int(args.max_cycles):
                    print(f"[select] STOPPING at the --max_cycles cap of {args.max_cycles}; "
                          f"{st['done']}/{st['jobs']} episodes done. The table is TRUNCATED, "
                          "not complete -- raise the cap or lower --episodes_per_clip.")
                    break
                mirror.refresh()
                # PER CYCLE. CrnNoise addresses a draw by (group, round, candidate), so a
                # round index that only moved when an EPISODE ended served the generator the
                # identical x0 at every re-plan for the whole episode -- a rollout no
                # training episode resembles, in every arm at once, with nothing in the
                # shapes to show for it.
                crn.set_round(None, round_index)
                crn.arm_branch(0)
                if bo is not None:
                    pre = ~bo.branched
                    # BEFORE the branch every candidate is the same rollout, so they are all
                    # served candidate 0's draw; AT the branch they are served their own.
                    # --same_noise pins force_candidate to 0 throughout, which is the control:
                    # the candidates are then identical at the branch too, so any best-of-K
                    # rise is the simulator and not the generator.
                    if not args.same_noise:
                        crn.force_candidate = 0 if pre.all() else None
                    if not pre.all():
                        # AFTER the branch, one shared draw per group per re-plan. Without it
                        # a candidate can win on a lucky continuation and the number stops
                        # being about the chunk.
                        post_any = int(bo.post.max())
                        if post_any > 0:
                            crn.arm_continuation(post_any)

                # Every env in the group is written from the group's carrier, INCLUDING the
                # carrier itself. The restore probe showed an env that is never written is a
                # different situation from one that is -- branch-vs-branch is bit-identical
                # while branch-vs-parent never is -- so leaving the carrier untouched would
                # make it systematically unlike the candidates it is compared against.
                if args.align_cadence:
                    # SNAP TO THE CADENCE, as `_roll` does. The generator re-plans naturally
                    # on `episode_length_buf % fm_update_every == 0`; forcing one at the top
                    # of a cycle whose counter sits at another phase means it re-plans TWICE
                    # per cycle, at two phases, and the reference the tracker follows is one
                    # no training episode produced. `_force_replan` is only idempotent where
                    # the cadence has just fired.
                    n = u.episode_length_buf
                    off = n % replan_k
                    if bool((off != 0).any()):
                        u.episode_length_buf = n - off

                carrier = torch.as_tensor(ge.carrier, device=device)
                snap = mirror.capture(carrier)
                if bo is not None and bo.branched.any():
                    # A branched group is NOT resynchronised: each candidate must carry its
                    # own consequences to termination, which is the whole measurement.
                    keep = torch.as_tensor(np.repeat(~bo.branched, group_size),
                                           device=device)
                    restore_ids = all_envs[keep]
                else:
                    restore_ids = all_envs
                if args.no_restore:
                    # Nothing is written to the simulator at all. At k=0 there is nothing to
                    # branch, so this is a pure no-op that must not change the outcome --
                    # any difference it makes IS the restore's cost.
                    if not args.no_force_replan:
                        hr._force_replan(cmd, all_envs)
                elif args.isolate:
                    # Restore ONLY the carrier onto itself, so its siblings are never moved
                    # to its pose and it stands alone. Everything else -- env count, group
                    # count, tile assignment, cadence, the self-restore itself -- is
                    # unchanged, so this isolates the one thing the K-env arms do that the
                    # k=0 arm does not: put K robots in the same cubic metre.
                    mirror.restore(carrier, snap,
                                   src_index=torch.arange(num_groups, device=device))
                    hr._force_replan(cmd, carrier)
                elif restore_ids.numel():
                    mirror.restore(restore_ids, snap, src_index=group_of_env[restore_ids])
                    if not args.no_force_replan:
                        hr._force_replan(cmd, restore_ids)
                # After the restore AND the forced re-plan, the previous step's observation
                # describes neither the robot nor the plan any more.
                obs = env.get_observations()

                outcome = WindowOutcome(num_envs, device)
                for _ in range(replan_k):
                    obs, _, dones, _ = env.step(policy(obs))
                    ring.push(accum.rd)
                    ring.invalidate(dones)
                    down, lat, hz = goal_region.goal_frame_offsets(
                        accum.rd.root_pos_w, accum._goal_pos_w, accum._goal_yaw)
                    outcome.note(u, in_goal=goal_region.in_goal(down, lat, hz, accum._goal_tol))
                cycles += 1
                ge.cycles += 1
                round_index += 1

                if bo is not None:
                    ok_now = outcome.arrived_before_falling().detach().to("cpu").numpy()
                    ended = ((outcome.fell | outcome.finished)
                             .detach().to("cpu").numpy())
                    # Latch only while an env is still LIVE. Isaac auto-resets a terminated
                    # env, so a dead candidate goes on to run a junk episode -- and can
                    # "arrive" in it. Latching that would score a chunk by what the harness
                    # did after it failed.
                    live = ~done_env
                    arrived_env |= ok_now & live
                    done_env |= ended & live
                    for g in range(num_groups):
                        lo, hi = g * group_size, (g + 1) * group_size
                        if not bo.branched[g]:
                            if int(ge.cycles[g]) >= int(bo.branch_at[g]):
                                bo.branched[g] = True
                            continue
                        bo.post[g] += 1
                        bo.done[g] = done_env[lo:hi]
                        bo.ok[g] = arrived_env[lo:hi]
                    # A branched group's episode is over only when EVERY candidate has
                    # terminated; the slowest one sets the pace.
                    ge.carrier = np.array([g * group_size for g in range(num_groups)])

                # ORACLE only. There used to be a learned arm here reading the critic's
                # survival head, whose score was P(fail) and so needed its sign flipped
                # before the argmax. The head is gone, `--scorer` offers `oracle` alone, and
                # keeping the branch meant a careful sign comment guarding nothing.
                scores = oracle_scores(outcome)
                if args.no_selection:
                    # THE STRUCTURAL CONTROL. Nine envs are still built, stacked, restored
                    # and rolled exactly as in the selector arms -- but the group follows
                    # candidate 0 whatever the others do, so no choice is ever made. It must
                    # therefore reproduce the k=0 baseline. Where it does not, the gap
                    # belongs to the K-env STRUCTURE and not to selection, and no selector
                    # number measured in that structure means anything yet.
                    winner = (torch.arange(num_groups, device=device) * group_size)
                else:
                    winner, _ = choose(scores, outcome.alive(), num_groups, group_size)
                ge.carrier = winner.detach().to("cpu").numpy()
                ok = outcome.arrived_before_falling().detach().to("cpu").numpy()
                chain_arrived |= ok[ge.carrier]

                for g in range(num_groups):
                    if bo is not None:
                        if not (bo.branched[g] and bo.done[g].all()):
                            continue
                        verdict = "success" if bo.result(g)["best_of_k"] else "fail"
                    else:
                        verdict = (slot0_verdict(outcome, g, group_size) if args.no_selection
                                   else group_verdict(outcome, g, group_size))
                    if verdict is None or retired[g]:
                        continue
                    clip = int(alloc.group_clip[g])
                    fam = int(family_codes[plan["clips"][clip]["family"]])
                    # `rollout_outcomes.terminal_reward`, restated: reaching the goal before
                    # falling is the ONLY success. Surviving without arriving is a failure,
                    # or standing still would score like climbing the staircase -- which is
                    # what the survival metric this replaces actually did, reading 0.864
                    # against the episode harness's ~0.67.
                    verdict = "success" if chain_arrived[g] else "fail"
                    chain_arrived[g] = False
                    ge.finish(g, verdict=verdict, clip_id=clip, family=fam,
                              next_episode_id=next_episode_id)
                    if bo is not None:
                        ge.rows[-1].update(bo.result(g))
                        lo, hi = g * group_size, (g + 1) * group_size
                        arrived_env[lo:hi] = False
                        done_env[lo:hi] = False
                        bo.arm_episode(g)
                    next_episode_id += 1
                    if alloc.drained:
                        # `_take` WRAPS to the last job rather than stopping, so a group
                        # that finishes after the list drains would re-run one clip over and
                        # over. Harmless in the episode harness, fatal here: the number of
                        # groups differs between the arms being compared, so each arm's tail
                        # is a different pile of duplicates and the arms stop being paired
                        # on clip id. Retire the group instead.
                        retired[g] = True
                        continue
                    retarget(g, alloc.advance(g))
                    # The group restarts from its slot-0 env. Which env carries the state is
                    # arbitrary once the episode is over -- all of them are about to be reset
                    # onto the new clip -- but it must be a member of THIS group, or the next
                    # cycle would seed the group from a neighbour's robot.
                    ge.carrier[g] = g * group_size
                    round_index[g] += 1
                if cycles % 25 == 0:
                    st = alloc.stats()
                    print(f"[select] cycle {cycles}: {len(ge.rows)} episodes, "
                          f"{st['done']}/{st['jobs']} jobs, "
                          f"{cycles * replan_k / max(time.time() - t0, 1e-9):.1f} steps/s",
                          flush=True)

        summary = summarise(ge.rows)
        # Every flag that changes what was run belongs in the name. Twice now an arm has
        # overwritten another's json because the name encoded only some of them, and the
        # two were then indistinguishable after the fact.
        arm = "real"
        if args.no_selection:
            arm = "nosel"
        elif args.same_noise:
            arm = "samenoise"
        if args.isolate:
            arm += "_isolate"
        if args.steer_kp != 0.0:
            arm += f"_steer{args.steer_kp:g}"
        for flag, tag in ((args.no_restore, "norestore"),
                          (args.no_force_replan, "noforce"),
                          (args.align_cadence, "aligned")):
            if flag:
                arm += f"_{tag}"
        out = pathlib.Path(args.out) / "select" / f"select__k{args.k}__{args.scorer}__{arm}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "rows": ge.rows,
                                   "k": int(args.k), "same_noise": bool(args.same_noise),
                                   "scorer": args.scorer, "num_groups": num_groups,
                                   "steer_kp": float(args.steer_kp),
                                   "steer_wz_max": float(args.steer_wz_max),
                                   "replan_k": replan_k}, indent=2))
        print(f"[select] {arm} k={args.k}: SUCCESS {summary['survived']:.3f} "
              f"over {summary['n']} episodes -> {out}")
        status["rc"] = 0

    _main()
    with contextlib.suppress(Exception):
        app.close()
    return status["rc"]
