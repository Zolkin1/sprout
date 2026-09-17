# Copyright 2026 Zachary Olkin. All rights reserved.

"""rsl-rl wrapper for an env whose agent is not the thing driving the joints.

:class:`~robot_rl.tasks.manager_based.robot_rl.envs.adjuster_env.AdjusterEnv` keeps a frozen
tracker's ``JointPositionAction`` in its action manager -- that is what lets the tracker's own
``actions`` observation term keep working unchanged -- while the RL agent's action is a
residual on the generator's plan. So ONE of the base wrapper's derived numbers is wrong:

* ``num_actions`` comes from ``action_manager.total_action_dim``, which is the tracker's 29
  joints. Left alone it would size the adjuster's head to 29 and the env would refuse the
  action on the first step -- loudly, but for a reason that reads as a config error rather
  than as a wrapper assumption.

``max_episode_length`` is deliberately left alone, and the reason is worth stating because
the opposite looks right: its only rsl-rl reader seeds ``episode_length_buf``, and on this
env that buffer counts CONTROL steps, not agent steps. Converting it here divided the
randomization range by ``replan_period``.
"""

from __future__ import annotations

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class RslRlAdjusterVecEnvWrapper(RslRlVecEnvWrapper):
    """:class:`RslRlVecEnvWrapper` with the agent's action width and step rate corrected."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        """Wrap ``env`` and re-derive the two numbers the base class gets from the wrong place.

        Args:
            env: An :class:`AdjusterEnv` (or anything exposing ``replan_period`` and a command
                term whose generator states ``F`` and ``D``).
            clip_actions: Forwarded. Note the residual's own bound is applied inside the env,
                in physical units and per channel, so this is a second and much blunter clip;
                leaving it ``None`` is the default for a reason.

        Raises:
            ValueError: If the env is not an interval-stepping one -- so a plain
                ``ManagerBasedRLEnv`` wrapped by mistake fails here rather than training a
                29-dimensional "residual".
        """
        unwrapped = env.unwrapped
        if not hasattr(unwrapped, "replan_period"):
            raise ValueError(
                f"{type(self).__name__} expects an AdjusterEnv (it needs `replan_period` to "
                f"convert the episode length), got {type(unwrapped).__name__}. Use the base "
                "RslRlVecEnvWrapper for a per-control-step task."
            )
        super().__init__(env, clip_actions=clip_actions)

        cmd = unwrapped.command_manager.get_term(unwrapped.cfg.command_name)
        # K * D, not F * D: the action is the residual's KNOTS and the env lifts them to
        # the full window. Read from the env cfg so this and the actor cannot disagree
        # silently -- and if they do, the env's action-width check names it.
        self.num_actions = int(env.unwrapped.cfg.num_knots) * int(cmd.ctx.D)
        # `max_episode_length` is deliberately NOT overridden. It has exactly one reader in
        # rsl-rl -- `OnPolicyRunner.learn`'s `init_at_random_ep_len`, which seeds
        # `episode_length_buf` from it -- and on this env that buffer counts CONTROL steps
        # (AdjusterEnv increments it inside the inner loop, on purpose, so `time_out` and the
        # clip cursor keep working unchanged). Reporting agent steps here therefore seeded
        # every env into the first 1/K of the episode: 12x under-randomized, which is the
        # opposite of what that flag is for. An earlier version of this file named that very
        # call as the justification for the override, which was backwards.
        agent_steps = unwrapped.max_episode_length // int(unwrapped.replan_period)
        # Re-run only if it did something: it is a no-op at clip_actions=None, and it reads
        # `num_actions`, which was the tracker's width when the base class called it.
        if clip_actions is not None:
            self._modify_action_space()
        print(f"[INFO] {type(self).__name__}: agent action width {self.num_actions} "
              f"(F x D), episode <= {agent_steps} agent steps "
              f"({self.max_episode_length} control steps)")
