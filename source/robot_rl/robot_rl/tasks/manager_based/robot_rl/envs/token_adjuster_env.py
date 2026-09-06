# Copyright 2026 Zachary Olkin. All rights reserved.

"""One env step = one re-plan, and the agent's action is a DISCRETE correction to the plan.

The third env in this family, and it is the cross of the other two:
:class:`~.adjuster_env.AdjusterEnv` keeps the flow-matching generator in the loop and learns a
continuous residual on the plan it drew; :class:`~.token_planner_env.TokenPlannerEnv` replaces
the generator with a token policy. This one keeps the generator AND makes the correction
discrete, which is the whole point of the stack: a categorical policy has an exact per-token
log-probability and entropy, so PPO needs no reparameterisation trick and no surrogate for the
sampling density.

## Why the discrete correction, and not either sibling

Measured on the shipped pair, all on the same held-out windows in metres of base position:

* the generator alone is 0.0431, and the absolute token planner is WORSE than it;
* the residual codebook can represent the correction -- 0.0187 given the true residual;
* but the supervised adjuster only predicts its way to 0.0427, because the conditional over
  residual tokens is broad and its mode is "no correction".

So behaviour cloning converges on doing nothing safely, and only an outcome-driven objective
can do better. That is this env.

## What differs from ``AdjusterEnv``, and why each difference is forced

* **The action is ``int64`` ids, not a float residual.** ``[N_envs, N]`` ids are decoded
  through the frozen RESIDUAL tokenizer, denormalized into metres, scaled by ``alpha`` and
  handed to the inherited ``apply_plan_residual``. Everything after that point -- the
  projection, the geodesic tilt bound, the ``_prev_served_node`` re-take, the interval loop,
  the termination attribution, the reward ordering -- is the base class's and is untouched.
* **Two normalizers, and they are not interchangeable.** The plan is in the GENERATOR's state
  space; the residual is in the tokenizer's own. Both are ``(B, H, 44)`` of order-one floats,
  so decoding a residual with the state normalizer is a silent per-channel rescale of every
  correction. The env holds the residual one and never sees the other.
* **The plan is exposed.** The policy conditions on the window it is correcting, so
  :meth:`plan_window` hands out the generator's own prediction for this re-plan -- the one the
  command term actually latched, not a re-derivation, for the same reason
  :meth:`planner_conditioning` exists.
* **``residual_alpha = 0`` is the identity, exactly.** Not approximately: a zero residual
  returns the plan bit-identically, so an agent initialised there reproduces the bare
  generator and the first RL iteration measures the baseline rather than a perturbation of it.
"""

from __future__ import annotations

import os

import torch

from .adjuster_env import AdjusterEnv
from .token_adjuster_env_cfg import TokenAdjusterEnvCfg


class TokenAdjusterEnv(AdjusterEnv):
    """Plan-cadence env whose action is the token ids of a correction to the plan.

    Args:
        cfg: The env cfg.
        **kwargs: Forwarded to :class:`AdjusterEnv`.

    Raises:
        ValueError: If the command's generator is not the flow-matching one (there would be
            no plan to correct), or if no residual tokenizer was named.
    """

    def __init__(self, cfg: TokenAdjusterEnvCfg, **kwargs) -> None:
        """Build the env and bind the frozen residual tokenizer that decodes actions."""
        super().__init__(cfg, **kwargs)
        cmd = self.command_manager.get_term(cfg.command_name)
        kind = getattr(cmd.cfg, "generator_kind", "flow_matching")
        if kind == "token_planner":
            raise ValueError(
                "TokenAdjusterEnv corrects a FLOW-MATCHING plan, and the command is serving a "
                "token planner. Correcting a token planner's own output with a second token "
                "model is a different experiment and would need its own residual dataset -- "
                "the one on disk was built against the generator.")
        self._cmd = cmd
        self._tokenizer, self._res_norm = self._load_residual_tokenizer(
            str(cfg.residual_tokenizer_path))
        self.num_tokens = int(self._tokenizer.num_tokens)
        self.vocab_size = int(self._tokenizer.vocab_size)
        self.residual_alpha = float(cfg.residual_alpha)
        f, d = int(cmd.ctx.F), int(cmd.ctx.D)
        if int(self._tokenizer.cfg.chunk_len) != f or int(self._tokenizer.cfg.node_dim) != d:
            raise ValueError(
                f"the residual tokenizer covers {self._tokenizer.cfg.chunk_len} x "
                f"{self._tokenizer.cfg.node_dim} nodes but the generator's plan is {f} x {d}. "
                "A correction has to be the same shape as the thing it corrects.")
        print(f"[TokenAdjusterEnv] action = {self.num_tokens} residual tokens over "
              f"V={self.vocab_size}; alpha={self.residual_alpha}; "
              f"{self.replan_period} control steps per agent step.")

    # ------------------------------------------------------------------
    def _load_residual_tokenizer(self, path: str):
        """Load the frozen residual tokenizer and its normalizer.

        Args:
            path: A ``train_tokenizer.py --residual`` checkpoint.

        Returns:
            ``(tokenizer, residual_normalizer)``, the tokenizer in eval mode and out of the
            autograd graph -- PPO's optimizer must not be able to reach it, or the meaning of
            every id would drift under the policy that is emitting them.

        Raises:
            ValueError: If no path was given, or the checkpoint is an ABSOLUTE tokenizer.
        """
        from robot_rl.flow_matching.normalizer import Normalizer  # noqa: PLC0415
        from robot_rl.token_planner.config import TokenizerConfig  # noqa: PLC0415
        from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: PLC0415

        if not path or not os.path.exists(path):
            raise ValueError(
                "TokenAdjusterEnv needs env.residual_tokenizer_path to name a residual "
                f"tokenizer checkpoint; got {path!r}. The action is token ids and they mean "
                "nothing without the codebook that defines them.")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        tok_cfg = TokenizerConfig.from_dict(ck["config"])
        tok = TrajectoryTokenizer(tok_cfg)
        tok.load_state_dict(ck.get("model_ema") or ck["model"])
        tok = tok.to(self.device).eval().requires_grad_(False)
        norm = Normalizer.from_state_dict(ck["state_normalizer"]).to(self.device)
        # An absolute tokenizer's normalizer standardizes only some channels and carries a
        # non-zero mean; a residual one pins the mean to zero and standardizes all of them.
        # Decoding an absolute codebook here would produce a TRAJECTORY where a correction is
        # expected, and adding it to the plan would double the motion at the right shape.
        if bool(norm.mean.abs().max() > 0) or not bool(norm.standardize_mask.all()):
            raise ValueError(
                f"{path} looks like an ABSOLUTE tokenizer: its normalizer has a non-zero mean "
                "and/or an identity channel. A residual tokenizer pins the mean to zero and "
                "standardizes every channel. Composing an absolute codebook's decode onto a "
                "plan adds two trajectories.")
        return tok, norm

    # ------------------------------------------------------------------
    def planner_conditioning(self) -> torch.Tensor:
        """The normalized ``(history | extero | cmd | null_flag)`` rows for this re-plan.

        The rows the command term ACTUALLY built for the plan now on the buffer -- rebuilding
        them here would read a different instant and a different depth lag.

        Returns:
            ``[N_envs, cond_dim]``.
        """
        return self._cmd.plan_conditioning()

    def plan_window(self) -> torch.Tensor:
        """The generator's own prediction for this re-plan, in RAW units.

        The window the agent is correcting, straight off the command's buffer, so the policy
        conditions on the plan that will actually be adjusted rather than on a re-sample of
        the same conditioning -- which would differ, the generator being stochastic in
        general and its buffer being what every other consumer reads.

        Returns:
            ``[N_envs, F, D]``.
        """
        return self._cmd.plan_window()

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """Compose one decoded correction onto the plan, then run the whole interval.

        Args:
            action: ``[N_envs, N]`` int64 residual token ids.

        Returns:
            ``(obs, reward, terminated, truncated, extras)`` for the interval.
        """
        if action.dtype not in (torch.int64, torch.int32):
            raise TypeError(
                f"TokenAdjusterEnv.step expects int64 token ids, got {action.dtype}. A float "
                "action here would be silently floored into the vocabulary.")
        ids = action.to(self.device, dtype=torch.long)
        if ids.shape != (self.num_envs, self.num_tokens):
            raise ValueError(
                f"action {tuple(ids.shape)} != ({self.num_envs}, {self.num_tokens}).")
        return super().step(ids)

    def _residual_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """Decode residual ids into a RAW-unit correction, scaled by ``alpha``.

        Replaces the base class's ``tanh(action) * residual_scale``. The squash is gone
        because there is nothing to squash: a codebook is bounded by construction, and its
        reach is a property of the data it was fitted on rather than of a hand-set authority
        vector. The geodesic tilt bound still applies -- it is inside ``apply_plan_residual``,
        where it protects the tracker's own tilt check regardless of where the residual came
        from.

        Args:
            action: ``[N_envs, N]`` int64 ids.

        Returns:
            ``[N_envs, F, D]`` in raw units.
        """
        with torch.no_grad():
            residual = self._res_norm.denormalize(
                self._tokenizer.decode_ids(action.to(self.device, dtype=torch.long)).float())
        return residual * self.residual_alpha
