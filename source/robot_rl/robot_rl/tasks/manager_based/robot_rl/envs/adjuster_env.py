# Copyright 2026 Zachary Olkin. All rights reserved.

"""One env step = one re-plan interval, with the tracker frozen inside it.

The RL agent here is the ADJUSTER: it emits a residual on the generator's predicted reference
window, and that residual is only consumed when the generator re-plans -- every
``fm_update_every`` control steps, 0.24 s at K=12. So the agent's decision rate is the plan's
rate, not the control rate, and one ``step`` covers the whole interval: the residual is
applied once, then the frozen tracker drives the robot through K control steps and nothing can
be done about it until the next plan. From the adjuster's point of view the tracker is part of
the robot's dynamics.

## The counters stay in CONTROL steps, and that is what keeps everything else unchanged

``decimation``, ``sim.dt`` and therefore ``step_dt`` are untouched, and
``episode_length_buf`` / ``common_step_counter`` are incremented INSIDE the inner loop. So
``time_out``'s ``max_episode_length`` still counts control steps, the clip cursor still
advances 1:1 with them, the ego re-anchor keeps its cadence, and
``_snapshot_and_infer``'s de-duplication on ``common_step_counter`` stays exactly as tight as
it is on a per-control-step task. The one thing that cannot be shared is the RE-PLAN phase,
which is why this env owns :attr:`fm_cadence_buf` and the command term prefers it.

## Ordering, which is the whole design

IsaacLab's convention is that the observation returned by step *t* feeds the action of step
*t+1*. Under that convention the generator re-plans at the END of an interval, so the
observation the adjuster acts on already describes the fresh plan -- and the residual it
returns is applied to that same plan before any physics. Applying it inside ``_run_fm``
instead would make it a function of the state before the plan was drawn: a state-conditioned
bias, not a correction to a plan the agent has seen.

The post-loop half mirrors ``ManagerBasedRLEnv.step`` deliberately, including the order
rewards-then-reset: the residual L2 reward reads the residual that was applied, and
``FmTrackingCommand.reset`` zeroes it, so folding rewards after the reset would silently
drop the penalty on every terminating interval.

## What is deliberately NOT here

* **No action term for the residual.** The env's action manager keeps holding the tracker's
  ``JointPositionAction``, fed by the frozen policy inside the loop -- so ``mdp.last_action``
  still returns the 29 joint targets the tracker's own ``actions`` observation term expects,
  and no observation had to be re-pointed. The residual reaches the command term directly.
  The consequence is that ``action_manager.total_action_dim`` is the tracker's width, not the
  agent's, which is why the rsl-rl wrapper overrides ``num_actions``.
* **No mid-interval reset.** A done is LATCHED and the reset fires at the interval boundary.
  Breaking the loop instead would collapse it to one control step at any realistic env count
  -- something terminates almost every step at 1024 envs -- so the agent's step would stop
  being 0.24 s. Resetting mid-loop is worse: the transition would span two episodes and the
  value bootstrap would be taken against a state from a different clip and tile.
"""

from __future__ import annotations

import json

import torch
from isaaclab.envs import ManagerBasedRLEnv

from robot_rl.flow_matching.window import residual_scale_vector
from robot_rl.rl.frozen_tracker import FrozenTracker
from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error
from robot_rl.tasks.manager_based.robot_rl.mdp.terminations.terminations import TrackingViolation

from .adjuster_env_cfg import AdjusterEnvCfg


class AdjusterEnv(ManagerBasedRLEnv):
    """A manager-based RL env whose step spans one generator re-plan interval.

    Attributes:
        replan_period: ``K``, control steps per agent step, taken from the command term.
        fm_cadence_buf: ``[N]`` control steps since reset, which the command term's re-plan
            cadence keys on. Distinct from ``episode_length_buf`` only in that this env is
            free to zero it independently; today they move together.
        track_mean: ``[N]`` the interval's mean tracking reward, in ``(0, 1]``. The dense
            reward term reads it.
    """

    cfg: AdjusterEnvCfg

    def __init__(self, cfg: AdjusterEnvCfg, **kwargs) -> None:
        """Build the env, then load the frozen tracker against it.

        Args:
            cfg: The env cfg.
            **kwargs: Forwarded to :class:`ManagerBasedRLEnv`.

        Raises:
            ValueError: If ``tracker_run_dir`` or ``tracking_spec_path`` is unset, or if the
                command term's re-plan cadence is not a usable interval length.
        """
        if not cfg.tracker_run_dir:
            raise ValueError(
                "AdjusterEnvCfg.tracker_run_dir is unset. The tracker drives the robot from "
                "inside this env, so there is nothing to run without it -- and a default "
                "would be a path to an artifact, which rots quietly."
            )
        if not cfg.tracking_spec_path:
            raise ValueError(
                "AdjusterEnvCfg.tracking_spec_path is unset. The dense reward's kernel widths "
                "come from a fitted spec, and BOOTSTRAP_SPEC's are placeholders roughly 2x "
                "blunter -- so there is no safe default."
            )
        super().__init__(cfg, **kwargs)

        cmd = self.command_manager.get_term(cfg.command_name)
        self.replan_period = int(cmd.cfg.fm_update_every)
        if self.replan_period < 1:
            raise ValueError(
                f"fm_update_every must be >= 1 to be an interval length, got "
                f"{self.replan_period}."
            )
        self.fm_cadence_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.track_mean = torch.zeros(self.num_envs, device=self.device)
        # WHICH TERM ended each episode, recorded per term so a sparse reward can name its
        # cause. These are real terminations -- the episode ends and the env resets. This
        # exists only because the reset is deferred to the interval boundary: an env can
        # fail at control step 3 of 12, and resetting there would split one agent step
        # across two episodes (the residual belongs to the old one, the remaining control
        # steps would run under its plan, and one reward would cover both). The
        # aggregates (`terminated` / `time_outs`) cannot: reaching the goal and running out of
        # time are both truncations here, and they must be priced differently -- arriving
        # collects the value bootstrap and nothing else, running out collects a penalty.
        # `mdp.is_terminated_term` is NOT usable for this: it reads the manager's per-term
        # state, which is recomputed every control step, so a term that fired early in the
        # interval and stopped holding would read as never having fired.
        self._warned_no_attribution = False
        # Per-check violation rates, folded over control steps and reported once per agent
        # step. See the fold in `step` for why they are never pooled into one number.
        self._violation_counts = {
            name: torch.zeros((), device=self.device)
            for name in tracking_error.VIOLATION_CHECK_NAMES
        }
        self._violation_steps = torch.zeros((), device=self.device)
        self.ended_by: dict[str, torch.Tensor] = {
            name: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in self.termination_manager.active_terms
        }
        self._track_offsets = torch.zeros(1, dtype=torch.long, device=self.device)
        # ONE spec, and this ENFORCES it rather than assuming it. The dense reward's kernel
        # widths and the failure termination's bounds have to come from the same file, or the
        # reward grades against one calibration while the episode ends on another -- silently,
        # since both are valid specs. `TrackingViolation` prefers its own `spec_path` param
        # (hydra-overridable) and only falls back to the env cfg, so the two CAN diverge.
        # Take the term's resolved answer where there is one; that way the override still
        # works and both halves move together.
        spec_path = str(cfg.tracking_spec_path)
        violation = None
        for name in self.termination_manager.active_terms:
            term = self.termination_manager.get_term_cfg(name).func
            if isinstance(term, TrackingViolation):
                violation = term
                override = str(term.cfg.params.get("spec_path") or "")
                if override and override != spec_path:
                    print(f"[INFO] AdjusterEnv: tracking spec taken from the "
                          f"'{name}' term ({override}), overriding the env cfg's "
                          f"{spec_path} -- the reward and the termination stay on one file.")
                    spec_path = override
                break
        with open(spec_path) as f:
            self._track_spec = tracking_error.TrackingErrorSpec.from_dict(json.load(f))
        if violation is not None and violation.spec != self._track_spec:
            raise ValueError(
                "AdjusterEnv: the dense reward and the tracking_violation termination "
                f"resolved DIFFERENT tracking specs ({spec_path} vs the term's). They grade "
                "the same quantity, so a split calibration means the reward's kernel widths "
                "and the failure bounds disagree with no symptom."
            )
        # Per-channel residual authority, resolved once from the generator's own layout.
        self._residual_scale = torch.tensor(
            residual_scale_vector(cmd.ctx.nj, cmd.ctx.target_num_extra_frames,
                                  cmd.ctx.reference_blocks),
            device=self.device,
        )

        # The two stashes are enabled by the COMMAND CFG, not from here, and this asserts it
        # rather than doing it: `ObservationManager._prepare_terms` calls every term once at
        # construction to discover its shape, and that happened inside `super().__init__()`
        # above -- so enabling them here would already be too late and the obs term would have
        # raised. Found by the first smoke run, with the stash guard naming the cause.
        missing = [name for name, flag in (("stash_conditioning", cmd.cfg.stash_conditioning),
                                           ("stash_flow", cmd.cfg.stash_flow)) if not flag]
        if missing:
            raise ValueError(
                f"AdjusterEnv needs {missing} set on the command cfg: its observation groups "
                "read the generator's conditioning and flow field, which are only captured "
                "when the command term is told to at construction."
            )

        self._frozen_tracker = FrozenTracker(
            self, cfg.tracker_run_dir, cfg.tracker_checkpoint,
            fm_meta=cmd.ctx.recording_meta,
            allow_mismatch=bool(cfg.allow_tracker_mismatch),
        )
        print(f"[INFO] AdjusterEnv: 1 agent step = {self.replan_period} control steps = "
              f"{self.replan_period * self.step_dt:.3f} s; "
              f"episode <= {self.max_episode_length // self.replan_period} agent steps")

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids) -> None:
        """Reset, and zero the re-plan phase for those envs.

        Args:
            env_ids: Environments to reset.
        """
        super()._reset_idx(env_ids)
        # Zero AFTER the parent: its command-manager reset is what invalidates the prediction,
        # and a fresh phase plus an invalid prediction is what makes the next command compute
        # re-plan for these envs exactly once (``reinfer | ~_pred_valid``).
        self.fm_cadence_buf[env_ids] = 0

    def reset(self, seed=None, env_ids=None, options=None):
        """Reset, then draw a first plan so the first residual has something to correct.

        ``ManagerBasedEnv.reset`` does NOT call ``command_manager.compute`` -- it resets the
        managers and computes observations. On a per-control-step task that is harmless: the
        first ``step`` computes the command before anything reads a plan. Here the first thing
        ``step`` does is ADD A RESIDUAL to the live plan, so without this the first interval of
        a run would correct a window of zeros and serve it.

        Args:
            seed: Forwarded.
            env_ids: Forwarded.
            options: Forwarded.

        Returns:
            ``(obs, extras)``.
        """
        super().reset(seed=seed, env_ids=env_ids, options=options)
        self.command_manager.compute(dt=self.step_dt)
        # update_history=False: the parent already pushed a history row for this instant, and
        # pushing a second would advance every history-stacked term by one step of nothing.
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, self.extras

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """Apply one residual, then run a whole re-plan interval under the frozen tracker.

        Args:
            action: ``[N, F * D]`` the dimensionless per-node residual.

        Returns:
            ``(obs, reward, terminated, truncated, extras)`` -- the interval's, not a control
            step's.
        """
        cmd = self.command_manager.get_term(self.cfg.command_name)
        # The phase check, and it is cheap: under a locked cadence every env re-planned on the
        # control step that ended the previous interval, so this is all-True at every boundary.
        # Asserting it turns a cadence drift into a failure on step one rather than a
        # slowly-wrong run whose only symptom is two re-plans inside one agent step.
        if not bool(cmd.fm_reinfer_mask.all()):
            raise RuntimeError(
                "AdjusterEnv.step: not every env re-planned at the interval boundary "
                f"({int(cmd.fm_reinfer_mask.sum())}/{self.num_envs} did). The re-plan cadence "
                "has drifted out of phase with the agent's step, so the residual is being "
                "applied to a partly-walked plan. Check fm_cadence_buf and fm_update_every."
            )

        residual = self._residual_from_action(action)
        cmd.apply_plan_residual(residual)
        # The tracker must see the CORRECTED plan on the interval's very first control step.
        # Only the reference group is rebuilt: the policy group's depth is unchanged by a
        # residual and re-drawing it would only resample its observation noise.
        self.obs_buf["reference"] = self.observation_manager.compute_group(
            "reference", update_history=False
        )

        rd = self.scene["robot"].data
        is_rendering = self.sim.is_rendering
        track_sum = torch.zeros(self.num_envs, device=self.device)
        live = torch.zeros(self.num_envs, device=self.device)
        done_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ended_terminated = torch.zeros_like(done_any)
        ended_truncated = torch.zeros_like(done_any)
        for buf in self.ended_by.values():
            buf.zero_()
        for buf in self._violation_counts.values():
            buf.zero_()
        self._violation_steps.zero_()

        for inner in range(self.replan_period):
            # `FrozenTracker.__call__` is already `@torch.no_grad`. Deliberately NOT
            # `inference_mode` on top: that would mark the output an inference tensor, which
            # cannot later enter an autograd graph, and buys nothing a frozen model needs.
            self.recorder_manager.record_pre_step()
            joint_action = self._frozen_tracker(self.obs_buf)
            self.action_manager.process_action(joint_action.to(self.device))

            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)
            self.recorder_manager.record_post_physics_decimation_step()

            self.episode_length_buf += 1
            self.common_step_counter += 1
            self.fm_cadence_buf += 1

            # -- the dense reward, folded at the CONTROL rate and reduced to one number ----
            # Over LIVE steps only: an env that fails a third of the way in must not have its
            # reward diluted by two thirds zeros, or the return would encode WHEN in the
            # interval it failed rather than that it did.
            err, valid = tracking_error.served_tracking_error(cmd, rd, self._track_offsets)
            alive = (~done_any) & valid
            track_sum += tracking_error.step_reward(err, self._track_spec) * alive
            # WHICH bound is doing the work, never pooled. Lateral and height are chunk-local
            # and re-zeroed at every re-plan; tilt is measured against gravity and is not. A
            # single violation rate cannot tell a spawn transient (tilt, in the opening control
            # steps) from a bad plan (lateral/height, accumulating over the interval), and
            # those call for opposite responses. Folded off the SAME `err` the reward uses, so
            # the diagnostic and the reward cannot disagree about what happened.
            for name, hit in tracking_error.violation_checks(err, self._track_spec).items():
                self._violation_counts[name] += (hit & alive).sum()
            self._violation_steps += alive.sum()
            live += alive

            # -- terminations, recorded at the FIRST crossing -------------------------------
            # First, so an env that reaches the goal at inner 3 and then falls at inner 7 is
            # reported as arriving. `terminated` is absorbing under GAE while `time_outs` gets
            # the value bootstrap, so mislabelling arrival as failure would teach the policy
            # not to arrive.
            dones = self.termination_manager.compute()
            fresh = dones & ~done_any
            ended_terminated |= fresh & self.termination_manager.terminated
            ended_truncated |= fresh & self.termination_manager.time_outs
            for name in self.ended_by:
                self.ended_by[name] |= fresh & self.termination_manager.get_term(name)
            done_any |= dones

            if inner < self.replan_period - 1:
                # Advances the plan cursor by one node. The re-plan does NOT fire here: the
                # cadence is `fm_cadence_buf % K == 0`, which the post-loop compute sees.
                self.command_manager.compute(dt=self.step_dt)
                if "interval" in self.event_manager.available_modes:
                    self.event_manager.apply(mode="interval", dt=self.step_dt)
                self.obs_buf = self.observation_manager.compute(update_history=True)

        # -- the interval's outcome ---------------------------------------------------------
        self.track_mean = track_sum / live.clamp_min(1.0)
        self.reset_terminated = ended_terminated
        self.reset_time_outs = ended_truncated
        self.reset_buf = ended_terminated | ended_truncated
        # dt=1.0: every term here is per-DECISION (the tracking mean, the residual penalties,
        # the sparse outcomes), not a rate, so the configured weights are the reward
        # magnitudes. Multiplying by a step time would make them a function of the cadence.
        self.reward_buf = self.reward_manager.compute(dt=1.0)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        self._restore_termination_attribution(ended_terminated | ended_truncated)

        # Rewards BEFORE the reset, as the base class does -- and here it is load-bearing
        # rather than incidental: `FmTrackingCommand.reset` zeroes the applied residual, so
        # folding afterwards would drop the L2 penalty on every terminating interval.
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        # THE re-plan for the next interval: `fm_cadence_buf` is now K (or 0 for a reset env),
        # so `% K == 0` fires for every env and the adjuster's next observation describes a
        # fresh plan.
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # AFTER the reset: `_reset_idx` rebuilds `extras["log"]`, so writing these earlier
        # would silently drop them on any step where something terminated -- which is most.
        log = self.extras.setdefault("log", {})
        denom = self._violation_steps.clamp_min(1.0)
        for name, count in self._violation_counts.items():
            log[f"Diagnostics/violation_{name}"] = (count / denom).item()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    # ------------------------------------------------------------------
    def _restore_termination_attribution(self, ended: torch.Tensor) -> None:
        """Write the FIRST-crossing terms back over the manager's per-episode attribution.

        ``TerminationManager.compute`` rewrites ``_last_episode_dones`` on every call where
        any term fires, and this env calls it K times per agent step. Left alone, the row
        logged at reset is the LAST control step's terms rather than the recorded first
        crossing -- so an env that reaches the goal at inner 3 and then trips
        ``tracking_violation`` at inner 7 is logged as a violation.

        The RL signal never had this problem (``reset_terminated`` / ``reset_time_outs`` /
        ``episode_ended_by`` all read the record), so this is purely about what is REPORTED --
        which happens to be ``Episode_Termination/goal_reached``, the arrival rate this whole
        reward design is built to move, and the go/no-go signature for the run. Under-reported
        by exactly the fraction of arrivals followed by any other termination inside the
        remaining K-1 control steps.

        Guarded on the private names because they are private: a rename should cost a loud
        message and correct training, not silent mis-attribution.

        Args:
            ended: ``[N]`` bool, the envs whose episode ended this interval.
        """
        tm = self.termination_manager
        if not (hasattr(tm, "_term_dones") and hasattr(tm, "_last_episode_dones")):
            if not self._warned_no_attribution:
                print("[WARN] AdjusterEnv: TerminationManager has no _term_dones / "
                      "_last_episode_dones, so Episode_Termination/* reports the LAST "
                      "control step of each interval rather than the recorded first "
                      "crossing. Arrival rates will be under-reported.")
                self._warned_no_attribution = True
            return
        if not bool(ended.any()):
            return
        rows = ended.nonzero(as_tuple=False).squeeze(-1)
        for i, name in enumerate(tm.active_terms):
            buf = self.ended_by.get(name)
            if buf is not None:
                tm._term_dones[rows, i] = buf[rows]
        tm._last_episode_dones[rows] = tm._term_dones[rows]

    # ------------------------------------------------------------------
    def _residual_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """Turn the agent's dimensionless action into a RAW-unit ``[N, F, D]`` residual.

        Three stages, and each is here rather than in the model for a reason:

        1. ``tanh``, not a clamp. A hard clamp gives zero gradient outside the box and lets
           the Gaussian's mean wander arbitrarily far, which makes ``init_std`` and the KL
           trust region meaningless.
        2. the per-channel scale from
           :func:`~robot_rl.flow_matching.window.residual_scale_vector`, which is where the
           physical calibration lives -- one table beside the noise model that stands in for
           the same error.
        3. nothing else: the rotation half's ANGLE bound is applied by
           ``apply_plan_residual``, after the projection, because it needs a real rotation.

        Applied HERE and not in the actor because ``MLPModel.forward`` returns
        ``distribution.sample()`` and ``PPO.update`` scores those stored actions against a
        freshly updated distribution -- so anything applied after the sample would score the
        same value under a density it did not come from. Same division of labour as
        ``JointPositionActionCfg(scale=...)``.

        Args:
            action: ``[N, F * D]`` from the policy.

        Returns:
            ``[N, F, D]`` in raw units.

        Raises:
            ValueError: If the action width is not ``F * D``.
        """
        cmd = self.command_manager.get_term(self.cfg.command_name)
        nodes, dim = int(cmd.ctx.F), int(cmd.ctx.D)
        if action.shape[-1] != nodes * dim:
            raise ValueError(
                f"AdjusterEnv expects an action of width F * D = {nodes} * {dim} = "
                f"{nodes * dim}, got {int(action.shape[-1])}."
            )
        shaped = action.to(self.device).view(-1, nodes, dim)
        return torch.tanh(shaped) * self._residual_scale
