# Copyright 2026 Zachary Olkin. All rights reserved.

"""One env step = one re-plan interval, and the AGENT'S ACTION IS THE PLAN.

The RL sibling of :class:`~.adjuster_env.AdjusterEnv`. That env keeps the generator in the
loop and learns a residual on the plan it drew; here the tokenized planner *is* the policy, so
the action is the ``N`` token ids of a whole plan and the decoded window replaces the
generator's output outright.

Everything else is inherited deliberately and unchanged: the plan-cadence step, the frozen
tracker inside the interval, the counters staying in CONTROL steps, the ordering of
rewards-before-reset, the termination attribution, the steering install. Those were the hard
parts and they are already right -- see ``AdjusterEnv``'s module docstring, all of which
applies here.

## What differs, and why each difference is forced

* **The action is ``int64`` tokens, not a float residual.** ``step`` takes ``[N_envs, N]``
  ids, decodes them through the frozen tokenizer, and calls ``apply_plan_window`` rather than
  ``apply_plan_residual``. The write lands in the same buffer at the same point in the step,
  so every consumer -- tracker observation, CLF, tracking error, history row, ribbon -- sees
  the agent's plan the way it saw the adjuster's correction.
* **The generator's own sample is discarded.** ``FmTrackingCommand`` still runs its sampler
  at the tail of the interval, because that is what populates ``_pred_valid``, the anchor and
  the conditioning stash; its window is then overwritten. Wasteful by one forward pass and
  worth it: suppressing the call instead would mean re-deriving the anchor latch and the
  history push here, which is exactly the kind of second implementation this file exists to
  avoid.
* **The conditioning is exposed.** The policy has to sample tokens BEFORE the env can step,
  so :meth:`planner_conditioning` hands out the same normalized ``(history, extero, cmd)``
  rows the command term stashed -- the ones that produced the discarded plan, so the policy
  conditions on exactly what the runtime built rather than on a re-derivation.
* **No residual rewards.** ``plan_residual_l2`` / ``plan_residual_rate_l2`` price a
  correction against a reference plan, and there is no reference plan here. A task cfg that
  leaves them in is refused rather than silently pricing zeros.
"""

from __future__ import annotations

import torch

from .adjuster_env import AdjusterEnv
from .token_planner_env_cfg import TokenPlannerEnvCfg


class TokenPlannerEnv(AdjusterEnv):
    """Plan-cadence env whose action is a plan's token ids.

    Args:
        cfg: The env cfg.
        **kwargs: Forwarded to :class:`AdjusterEnv`.
    """

    #: No flow field to stash: this generator's intermediate state is a distribution over
    #: tokens, not a direction in window space.
    REQUIRED_STASHES: tuple[str, ...] = ("stash_conditioning",)

    def __init__(self, cfg: TokenPlannerEnvCfg, **kwargs) -> None:
        """Build the env and bind the frozen tokenizer used to decode actions."""
        super().__init__(cfg, **kwargs)
        cmd = self.command_manager.get_term(cfg.command_name)
        if getattr(cmd.cfg, "generator_kind", "flow_matching") != "token_planner":
            raise ValueError(
                "TokenPlannerEnv needs generator_kind='token_planner' on the command: the "
                "action is decoded through the CHECKPOINT'S OWN frozen tokenizer, and a flow "
                "checkpoint carries none. Set "
                "env.commands.traj_ref.generator_kind=token_planner.")
        if getattr(cfg, "zero_residual", False):
            raise ValueError(
                "zero_residual has no meaning here: there is no residual to zero, so it "
                "would silently do nothing while reading as the bare-generator control. The "
                "control arm for this env is the REFERENCE planner sampling normally.")
        for banned in ("residual_l2", "residual_rate_l2"):
            if hasattr(self.cfg.rewards, banned):
                raise ValueError(
                    f"reward term '{banned}' prices a residual against the plan the generator "
                    "drew, and this env has no such reference -- the agent's tokens ARE the "
                    "plan. Remove it from the reward cfg rather than leaving it to price "
                    "zeros, which would read as a well-behaved agent.")
        self._cmd = cmd
        #: The planner context the command loaded: model, frozen tokenizer, normalizers.
        self.planner_ctx = cmd.ctx
        self.num_tokens = int(cmd.ctx.model.num_tokens)
        self.vocab_size = int(cmd.ctx.model.vocab_size)
        print(f"[TokenPlannerEnv] action = {self.num_tokens} tokens over V={self.vocab_size}; "
              f"{self.replan_period} control steps per agent step.")

    # ------------------------------------------------------------------
    def planner_conditioning(self) -> torch.Tensor:
        """The normalized ``(history | extero | cmd | null_flag)`` rows for this re-plan.

        The policy samples from these, and they are the rows the command term ACTUALLY built
        for the plan it just drew -- not a re-derivation. Rebuilding them here would read a
        different instant (after this interval's physics) and a different depth lag.

        Returns:
            ``[N_envs, cond_dim]``.
        """
        return self._cmd.plan_conditioning()

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """Serve one decoded plan, then run a whole re-plan interval under the tracker.

        Args:
            action: ``[N_envs, N]`` int64 token ids.

        Returns:
            ``(obs, reward, terminated, truncated, extras)`` for the interval.
        """
        if action.dtype not in (torch.int64, torch.int32):
            raise TypeError(
                f"TokenPlannerEnv.step expects int64 token ids, got {action.dtype}. A float "
                "action here would be silently floored into the vocabulary.")
        ids = action.to(self.device, dtype=torch.long)
        if ids.shape != (self.num_envs, self.num_tokens):
            raise ValueError(
                f"action {tuple(ids.shape)} != ({self.num_envs}, {self.num_tokens}).")
        with torch.no_grad():
            window = self.planner_ctx.model.decode(ids, self.planner_ctx.state_norm)
        self._pending_window = window
        return super().step(window)

    # ------------------------------------------------------------------
    def _apply_plan(self, cmd, residual: torch.Tensor) -> None:
        """REPLACE the plan with the agent's decoded window.

        Args:
            cmd: The ``FmTrackingCommand``.
            residual: The decoded ``[N_envs, F, D]`` window (named for the base signature).
        """
        cmd.apply_plan_window(residual)

    def _residual_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """Not a residual: hand the decoded window straight through.

        :class:`AdjusterEnv.step` calls this and then ``apply_plan_residual``. Overriding the
        pair (here and in :meth:`_apply_plan`) is what turns the inherited step from "correct
        the generator's plan" into "serve the agent's plan", without restating the interval
        loop, the termination attribution or the reward ordering.

        Args:
            action: The decoded ``[N_envs, F, D]`` window, passed through by :meth:`step`.

        Returns:
            The window, unchanged.
        """
        return action
