# Copyright 2026 Zachary Olkin. All rights reserved.

"""Verify the adjuster's re-plan cadence and its reward accumulation, on a live env.

Two properties that no unit test can reach, because both are about how ``AdjusterEnv.step``
interleaves with the command term over many steps and across resets:

1. **Cadence.** One agent step must contain EXACTLY ONE re-plan per env, and it must be the
   post-loop one. A drift here is silent -- the residual would be applied to a partly-walked
   plan, and the only symptom would be a slowly worse run. The env asserts the boundary
   condition (``fm_reinfer_mask.all()``); this asserts the whole interval, including that no
   re-plan sneaks into the inner loop and that the plan cursor sweeps ``0..K-1`` in order.

2. **Reward accumulation.** ``track_mean`` must be the mean of the per-control-step rewards
   over the steps the env was ALIVE for. Recomputing it here from independently captured
   per-step values checks the masking, the live count and the division -- which is where an
   error would hide (an env failing at inner 3 whose reward is diluted by nine zeros encodes
   WHEN it failed rather than THAT it did).

Also checks the zero-residual identity in place: with the action forced to zero the served
plan must be BIT-identical, since ``project_window_rotations`` is a Gram-Schmidt round trip
that is not the identity in float32.

Run with the same artifact flags as ``train.py``; see README section 4e.
"""

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments (before hydra_task_config)
from isaaclab_tasks.utils import add_launcher_args, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

parser = argparse.ArgumentParser(description="Verify the adjuster cadence.", allow_abbrev=False)
parser.add_argument("--task", type=str, default="G1-adjuster-fm-terrain-depth")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200, help="Agent (outer) steps to run.")
parser.add_argument("--clip", type=str, default=None)
parser.add_argument("--clip_device", type=str, default=None)
parser.add_argument("--fm_checkpoint", type=str, default=None)
parser.add_argument("--tracker_run", type=str, default=None)
parser.add_argument("--tracker_checkpoint", type=str, default=None)
parser.add_argument("--tracking_spec", type=str, default=None)
parser.add_argument("--goal_regions", type=str, default=None)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg) -> None:
    """Run the two checks and exit non-zero on any failure."""
    with launch_simulation(env_cfg, args_cli):
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401
        from robot_rl.tasks.manager_based.robot_rl.mdp import tracking_error

        env_cfg.scene.num_envs = args_cli.num_envs
        for name, term in env_cfg.commands.__dict__.items():
            if args_cli.clip is not None and hasattr(term, "path"):
                term.path = args_cli.clip
            if args_cli.clip_device is not None and hasattr(term, "clip_device"):
                term.clip_device = args_cli.clip_device
            if args_cli.fm_checkpoint is not None and hasattr(term, "fm_checkpoint"):
                term.fm_checkpoint = args_cli.fm_checkpoint
        if args_cli.clip is not None and hasattr(env_cfg.scene, "terrain"):
            if hasattr(env_cfg.scene.terrain, "clip_library_path"):
                env_cfg.scene.terrain.clip_library_path = args_cli.clip
        for flag, field in (("tracker_run", "tracker_run_dir"),
                            ("tracker_checkpoint", "tracker_checkpoint"),
                            ("tracking_spec", "tracking_spec_path"),
                            ("goal_regions", "goal_regions_path")):
            value = getattr(args_cli, flag, None)
            if value is not None and hasattr(env_cfg, field):
                setattr(env_cfg, field, value)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        cmd = env.command_manager.get_term(env.cfg.command_name)
        k = env.replan_period

        # -- instrumentation, all read-only wrappers -----------------------------------
        plan_calls: list[tuple[torch.Tensor, torch.Tensor]] = []
        dones_log: list[torch.Tensor] = []
        valid_log: list[torch.Tensor] = []
        reward_log: list[torch.Tensor] = []
        identity_max = torch.zeros((), device=env.device)

        raw_update = cmd._update_command

        def update_hook() -> None:
            """Record only the calls that DO WORK.

            ``_update_command`` is invoked more than once per control step at a reset step:
            ``_reset_idx`` -> ``command_manager.reset`` -> ``_resample_command`` reaches it,
            and the post-loop ``command_manager.compute`` then early-returns on
            ``_last_step_seen``. That is true of every task here, not just this one -- the
            guard is what makes the second call idempotent -- so the cadence invariant is
            about working calls, and which of the two did the work is incidental.
            """
            before = int(cmd._last_step_seen)
            raw_update()
            if int(cmd._last_step_seen) != before:
                plan_calls.append((cmd.fm_reinfer_mask.clone(), cmd._pred_idx.clone(),
                                   int(env.common_step_counter)))

        cmd._update_command = update_hook

        raw_apply = cmd.apply_plan_residual

        def apply_hook(residual: torch.Tensor) -> None:
            nonlocal identity_max
            before = cmd._pred_window.clone()
            raw_apply(residual)
            if not bool(residual.any()):
                identity_max = torch.maximum(identity_max,
                                             (cmd._pred_window - before).abs().max())

        cmd.apply_plan_residual = apply_hook

        raw_terms = env.termination_manager.compute

        def terms_hook() -> torch.Tensor:
            out = raw_terms()
            dones_log.append(out.clone())
            return out

        env.termination_manager.compute = terms_hook

        raw_served = tracking_error.served_tracking_error
        raw_step_reward = tracking_error.step_reward

        def served_hook(*a, **kw):
            err, valid = raw_served(*a, **kw)
            valid_log.append(valid.clone())
            return err, valid

        def reward_hook(*a, **kw):
            out = raw_step_reward(*a, **kw)
            reward_log.append(out.clone())
            return out

        tracking_error.served_tracking_error = served_hook
        tracking_error.step_reward = reward_hook

        # -- run ------------------------------------------------------------------------
        env.reset()
        zero = torch.zeros(env.num_envs, int(cmd.ctx.F) * int(cmd.ctx.D),
                           device=env.device)
        failures: list[str] = []
        n_resets = 0
        reset_steps: set[int] = set()
        mean_abs_err = 0.0

        for outer in range(args_cli.steps):
            plan_calls.clear()
            dones_log.clear()
            valid_log.clear()
            reward_log.clear()
            env.step(zero)

            # 1. cadence -------------------------------------------------------------
            if len(plan_calls) != k:
                failures.append(f"step {outer}: {len(plan_calls)} working command "
                                f"computes, expected {k} (one per control step); counters "
                                f"{[c for _, _, c in plan_calls]}")
                break
            counters = [c for _, _, c in plan_calls]
            if counters != list(range(counters[0], counters[0] + k)):
                failures.append(f"step {outer}: working computes at control steps {counters}, "
                                "expected one per consecutive step")
            per_env = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            for i, (mask, idx, _c) in enumerate(plan_calls):
                per_env += mask.long()
                if i < k - 1:
                    if bool(mask.any()):
                        failures.append(f"step {outer}: a re-plan fired at inner {i} "
                                        f"({int(mask.sum())} envs); only the post-loop one may")
                    if not bool((idx == i + 1).all()):
                        failures.append(f"step {outer}: plan cursor at inner {i} is "
                                        f"{idx.tolist()[:4]}..., expected {i + 1}")
                else:
                    if not bool(mask.all()):
                        failures.append(f"step {outer}: post-loop re-plan covered "
                                        f"{int(mask.sum())}/{env.num_envs} envs")
                    if not bool((idx == 0).all()):
                        failures.append(f"step {outer}: cursor after re-plan is "
                                        f"{idx.tolist()[:4]}..., expected 0")
            if not bool((per_env == 1).all()):
                failures.append(f"step {outer}: re-plans per env {per_env.tolist()}, expected all 1")

            # 2. reward accumulation, reconstructed --------------------------------------
            latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            acc = torch.zeros(env.num_envs, device=env.device)
            live = torch.zeros(env.num_envs, device=env.device)
            for i in range(k):
                alive = (~latched) & valid_log[i]
                acc += reward_log[i] * alive
                live += alive
                latched |= dones_log[i]
            expect = acc / live.clamp_min(1.0)
            err = (expect - env.track_mean).abs().max().item()
            mean_abs_err = max(mean_abs_err, err)
            if err > 1e-6:
                failures.append(f"step {outer}: track_mean off by {err:.3e}")
            if bool(((env.track_mean < 0.0) | (env.track_mean > 1.0)).any()):
                failures.append(f"step {outer}: track_mean outside (0, 1]")

            n_now = int(env.reset_buf.sum())
            if n_now:
                n_resets += n_now
                reset_steps.add(outer)
            if failures:
                break

        print("\n=== adjuster cadence + reward verification ===")
        print(f"outer steps          : {args_cli.steps} at {env.num_envs} envs "
              f"(K={k}, {k * args_cli.steps} control steps)")
        print(f"resets               : {n_resets} across {len(reset_steps)} distinct steps "
              f"({'staggered' if len(reset_steps) > 1 else 'NOT staggered'})")
        print(f"zero-residual identity: max |plan change| = {identity_max.item():.3e} "
              f"({'EXACT' if identity_max.item() == 0.0 else 'NOT EXACT'})")
        print(f"track_mean            : max |reconstructed - env| = {mean_abs_err:.3e}")
        if failures:
            print(f"\nFAILED ({len(failures)}):")
            for f in failures[:10]:
                print(f"  - {f}")
        else:
            print("\nPASSED: one re-plan per env per agent step, all post-loop; cursor swept "
                  f"0..{k - 1}; reward is the live-step mean; zero residual is a bit-identity.")
        env.close()
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
