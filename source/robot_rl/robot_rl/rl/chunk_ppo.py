# Copyright 2026 Zachary Olkin. All rights reserved.

"""PPO whose importance ratio is factored along the action chunk's TIME axis.

The adjuster emits one chunk per re-plan: ``a in R^{T x D}`` with ``T = 62`` look-ahead
nodes and ``D = 44`` channels, ``N = T*D = 2728`` scalar dimensions. The policy is a
factored Gaussian, and rsl-rl sums ``log_prob`` over every one of those dimensions, so the
PPO ratio is the JOINT one::

    r = exp( sum_{t,d} l_{t,d} ),   l_{t,d} = log pi_new - log pi_old

For a small update moving dimension ``(t,d)`` by ``delta_{t,d}`` sigmas, with the action
drawn from the old policy::

    E[log r] = -KL_tot        Var[log r] = 2 * KL_tot        KL_tot = sum_{t,d} delta^2 / 2

**The variance of the log-ratio grows with the number of dimensions while ``clip_param``
does not.** At a KL target of 0.94 that is ``std(log r) ~ 1.4`` nats against a band of
``log(1.2) ~ 0.18``, and the clip stops being a trust region and becomes a gate that most
samples fail: measured ~45% of samples contributing no gradient, where 5-20% is normal.

Neither obvious dial fixes it. Dropping ``desired_kl`` to 0.01 asks for ``~4e-6`` KL per
dimension per update, which pins the adaptive learning rate to its hardcoded 1e-5 floor.
Raising ``clip_param`` to ~4.8 is worse than it looks: the clip is in RATIO space, so
``[1 - 4.8, 1 + 4.8]`` has a negative lower bound and a ratio may collapse to 0 unpenalized.

So factor the ratio along time and clip each factor::

    r_t = exp( sum_d l_{t,d} )                                       # (B, T)
    L   = (1/T) sum_t min( r_t A,  clip(r_t, 1-eps, 1+eps) A )       # A is (B, 1)

Each ``r_t`` is a 44-dimensional Gaussian ratio -- the regime rsl-rl's defaults were tuned
for -- and one shared chunk-level advantage is broadcast over the steps.

**Why this is the same objective, not a different one.** At ``theta = theta_old`` every
ratio is 1 and ``grad log r = sum_t grad log r_t``, so ``grad L = (1/T) grad L_joint``:
identical direction, a constant the learning rate absorbs. The two differ only where the
clip binds, and there this version constrains each factor rather than their product. That
is exactly the factorization GRPO and RLHF-PPO use along the TOKEN axis -- a per-token
ratio and clip under one sequence-level advantage (DeepSeekMath, arXiv:2402.03300, Eqs. 1
and 3, and Sec. 4.1.2, where every token of an output shares one advantage). Here the axis
is the chunk's look-ahead time instead of token position.

**What this does NOT change**: the MDP is still one transition per re-plan, rewards summed
within the chunk and discounted per chunk, so the value loss and GAE are untouched.

Two consequences worth stating because they are easy to get wrong:

* ``entropy_coef`` should be 0 here. The log-std is a state-independent ``(T, D)`` parameter,
  so an entropy bonus does not encourage exploration where it is useful -- it inflates every
  std uniformly. (rsl-rl's ``GaussianDistribution`` already allocates one free ``log_std``
  per output dimension, and the outputs ARE ``(t, d)`` flattened, so the per-step, per-dim,
  untied parameterization this wants already exists.)
* ``desired_kl`` and ``clip_param`` go back to their stock values, because the statistics
  they bound are now per-step rather than per-chunk.
"""

from __future__ import annotations

import torch
from torch.distributions import Normal, kl_divergence

from .aux_ppo import AuxPPO


class ChunkPPO(AuxPPO):
    """:class:`AuxPPO` with a per-step importance ratio over a ``(T, D)`` action chunk."""

    def __init__(self, *args, chunk_steps: int | None = None,
                 chunk_dim: int | None = None, ratio_mode: str = "per_step",
                 **kwargs) -> None:
        """Initialize, resolving the chunk layout from the actor when not given.

        Args:
            chunk_steps: ``T``, the number of look-ahead nodes. Defaults to the actor's
                ``num_nodes``.
            chunk_dim: ``D``, channels per node. Defaults to the actor's ``node_dim``.
            ratio_mode: ``"per_step"`` (the default) clips ``T`` ratios of ``D`` dims each.
                ``"gspo"`` keeps ONE ratio per decision, the length-normalized geometric mean
                ``exp((1/N) sum l)``, clipped once -- Group Sequence Policy Optimization
                (Zheng et al. 2025). Note its clip band must be much SMALLER, not larger: the
                ``1/N`` normalization shrinks ``std(log r)`` by ``N`` rather than ``sqrt(N)``,
                so measured here 1.109 becomes ~4e-4 and a 0.2 band would never bind at all.

        Raises:
            ValueError: If the layout cannot be resolved, if it disagrees with the actor's
                output width, or if symmetry augmentation is enabled (untested here).
        """
        super().__init__(*args, **kwargs)
        actor = self.actor
        self.chunk_steps = int(chunk_steps if chunk_steps is not None
                               else getattr(actor, "num_nodes", 0))
        self.chunk_dim = int(chunk_dim if chunk_dim is not None
                             else getattr(actor, "node_dim", 0))
        if self.chunk_steps <= 0 or self.chunk_dim <= 0:
            raise ValueError(
                "ChunkPPO needs the action's (T, D) layout. The actor exposes neither "
                "`num_nodes` nor `node_dim`, and neither was passed. Without it the ratio "
                "cannot be factored and this would silently be stock PPO."
            )
        width = self.chunk_steps * self.chunk_dim
        std = self.actor.distribution.std_param if hasattr(self.actor.distribution, "std_param") \
            else self.actor.distribution.log_std_param
        if int(std.shape[-1]) != width:
            raise ValueError(
                f"ChunkPPO layout T*D = {self.chunk_steps}*{self.chunk_dim} = {width} does "
                f"not match the policy's output width {int(std.shape[-1])}. A wrong split "
                "would mix channels across nodes and the ratio would be meaningless while "
                "every shape still checked out."
            )
        if self.symmetry is not None:
            raise ValueError(
                "ChunkPPO does not support symmetry augmentation: the augmented batch "
                "repeats advantages and old params in ways this ratio has not been checked "
                "against. Refusing rather than producing plausible numbers."
            )
        if ratio_mode not in ("per_step", "gspo"):
            raise ValueError(
                f"ratio_mode={ratio_mode!r}; expected 'per_step' or 'gspo'."
            )
        self.ratio_mode = ratio_mode
        #: Diagnostics from the most recent minibatch, merged into ``update``'s loss dict.
        self.chunk_diagnostics: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _split(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape a flat ``(B, T*D)`` tensor to ``(B, T, D)``.

        Args:
            flat: Any per-output-dimension tensor.

        Returns:
            The same values as ``(B, T, D)``.
        """
        return flat.view(flat.shape[0], self.chunk_steps, self.chunk_dim)

    def _per_step_log_prob(self, mean: torch.Tensor, std: torch.Tensor,
                           actions: torch.Tensor) -> torch.Tensor:
        """Log-probability summed over CHANNELS only, leaving the time axis intact.

        Args:
            mean: ``(B, T*D)`` distribution mean.
            std: ``(B, T*D)`` distribution standard deviation.
            actions: ``(B, T*D)`` the actions actually taken.

        Returns:
            ``(B, T)``.
        """
        dist = Normal(self._split(mean), self._split(std))
        return dist.log_prob(self._split(actions)).sum(dim=-1)

    def _kl_for_lr_control(self, batch, distribution_params) -> torch.Tensor:
        """Mean PER-STEP KL, so ``desired_kl`` keeps its stock meaning.

        Summed over the ``D`` channels of a node and then averaged over nodes and batch --
        NOT summed over the whole chunk, which is what made the stock 0.01 target unreachable
        without the learning rate collapsing to its floor.

        Args:
            batch: The minibatch, carrying ``old_distribution_params``.
            distribution_params: The current ``(mean, std)``.

        Returns:
            Scalar mean per-step KL.
        """
        old_mean, old_std = batch.old_distribution_params
        new_mean, new_std = distribution_params
        per_dim = kl_divergence(Normal(self._split(old_mean), self._split(old_std)),
                                Normal(self._split(new_mean), self._split(new_std)))
        per_step = per_dim.sum(dim=-1)                      # (B, T)
        self.chunk_diagnostics["kl_per_step"] = per_step.mean().item()
        self.chunk_diagnostics["kl_chunk_total"] = per_step.sum(dim=-1).mean().item()
        return per_step.mean()

    def _surrogate_loss(self, batch, actions_log_prob: torch.Tensor) -> torch.Tensor:
        """The clipped surrogate, with one ratio per look-ahead node.

        ``actions_log_prob`` (the joint scalar rsl-rl computes) is deliberately unused: it is
        exactly the statistic whose variance grows with the chunk width. Both the old and the
        new per-step log-probs are rebuilt from stored/live distribution parameters, so no
        change to rsl-rl's rollout storage is needed -- it already keeps ``(mean, std)`` at
        full width, and only the joint log-prob at width 1.

        Args:
            batch: The minibatch.
            actions_log_prob: Unused; see above.

        Returns:
            Scalar surrogate loss.
        """
        del actions_log_prob

        old_mean, old_std = batch.old_distribution_params
        new_mean, new_std = self.actor.output_distribution_params
        old_logp = self._per_step_log_prob(old_mean, old_std, batch.actions)
        new_logp = self._per_step_log_prob(new_mean, new_std, batch.actions)

        log_ratio_step = new_logp - old_logp                          # (B, T)
        if self.ratio_mode == "gspo":
            # ONE ratio per decision, the length-normalized geometric mean over all N dims.
            # `.mean(-1)` over T of a per-step log-prob that is already a sum over D is
            # exactly (1/N) * sum_{t,d}, since every node has the same D.
            ratio = torch.exp(log_ratio_step.mean(dim=-1, keepdim=True))   # (B, 1)
        else:
            ratio = torch.exp(log_ratio_step)                              # (B, T)
        advantages = batch.advantages.view(-1, 1)                    # (B, 1), broadcast over T
        surrogate = -advantages * ratio
        surrogate_clipped = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        loss = torch.max(surrogate, surrogate_clipped).mean()

        with torch.no_grad():
            log_ratio = log_ratio_step
            # On the ratio ACTUALLY clipped, so the number means the same thing in both
            # modes. Reporting the per-step fraction under `gspo` would describe a statistic
            # the loss never used.
            clipped = ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param))
            self.chunk_diagnostics["clip_fraction"] = clipped.float().mean().item()
            if clipped.shape[-1] == self.chunk_steps:
                # Where along the chunk it happens: early nodes execute before the next
                # re-plan, late ones mostly do not, so they need not behave alike.
                per_node = clipped.float().mean(dim=0)
                self.chunk_diagnostics["clip_fraction_first_node"] = per_node[0].item()
                self.chunk_diagnostics["clip_fraction_last_node"] = per_node[-1].item()
            self.chunk_diagnostics["log_ratio_std_per_step"] = log_ratio.std().item()
            # The statistic this class exists to avoid, kept for comparison.
            self.chunk_diagnostics["log_ratio_std_joint"] = log_ratio.sum(dim=-1).std().item()
        return loss

    def update(self) -> dict[str, float]:
        """Run the parent update and merge in the chunk-ratio diagnostics.

        Returns:
            The parent's loss dict plus ``clip_fraction``, the per-node clip fractions, the
            per-step and joint log-ratio spreads, and the per-step and whole-chunk KL.
        """
        out = super().update()
        out.update(self.chunk_diagnostics)
        return out
