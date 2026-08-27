# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with an optional auxiliary reconstruction loss on the actor's reference encoder.

rsl-rl's :class:`~rsl_rl.algorithms.PPO` assembles its total loss inline in
:meth:`~rsl_rl.algorithms.PPO.update` and offers no hook for extra terms, so
:meth:`AuxPPO.update` below is a **verbatim copy of rsl-rl 5.0.1's**
``rsl_rl/algorithms/ppo.py`` lines 211-416 with exactly two additions, both marked
``# [robot_rl]``:

1. the reconstruction term added into ``loss`` right after the PPO loss is assembled;
2. its running mean, reported as ``loss_dict["reference_recon"]``.

Copying rather than restructuring is deliberate: with
``reference_reconstruction_coeff = 0`` (the default) the arithmetic is identical to stock
PPO, so the ablation is a true no-op rather than "PPO, but also reorganised". The cost is
that this copy is pinned to rsl-rl 5.0.1 -- if the wheel is upgraded, re-diff this method
against the new upstream ``update()``.

The decoder lives on the actor module, so it is already covered by the shared optimizer
(``chain(actor.parameters(), critic.parameters())``), by the actor's gradient clipping and
by ``actor_state_dict`` in :meth:`~rsl_rl.algorithms.PPO.save`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO


class AuxPPO(PPO):
    """PPO plus an optional reference-reconstruction auxiliary loss."""

    def __init__(self, *args, reference_reconstruction_coeff: float = 0.0,
                 critic_warmup_iterations: int = 0, **kwargs) -> None:
        """Initialize PPO and record the auxiliary-loss weight and the critic warm-up.

        Args:
            *args: Positional arguments forwarded to :class:`~rsl_rl.algorithms.PPO`.
            reference_reconstruction_coeff: Weight of the actor's reference
                reconstruction loss in the total loss. ``0.0`` (the default) disables the
                term entirely, leaving stock PPO behaviour.
            critic_warmup_iterations: Fit the value function ALONE for this many iterations
                before the actor is allowed to move. ``0`` (the default) is stock PPO. See
                :meth:`_in_critic_warmup` for why this exists and what it freezes.
            **kwargs: Keyword arguments forwarded to :class:`~rsl_rl.algorithms.PPO`.

        Raises:
            ValueError: If a non-zero coefficient is requested but the actor was built
                without a reconstruction decoder, or the warm-up length is negative.
        """
        super().__init__(*args, **kwargs)
        self.reference_reconstruction_coeff = float(reference_reconstruction_coeff)
        self.critic_warmup_iterations = int(critic_warmup_iterations)
        if self.critic_warmup_iterations < 0:
            raise ValueError(
                f"critic_warmup_iterations={self.critic_warmup_iterations} must be >= 0."
            )
        #: Counts calls to :meth:`update`, i.e. iterations. Deliberately NOT the runner's
        #: ``current_learning_iteration``: on a resume that starts at the teacher's 20000 and
        #: the warm-up would be skipped entirely, silently.
        self._updates_done = 0
        if self.reference_reconstruction_coeff != 0.0 and getattr(self.actor, "decoder", None) is None:
            raise ValueError(
                "reference_reconstruction_coeff="
                f"{self.reference_reconstruction_coeff} needs a reconstruction decoder on the "
                "actor. Set agent.actor.reconstruction=true (RefEncoderModelCfg.reconstruction), "
                "or set the coefficient to 0 to disable the auxiliary loss."
            )

    @property
    def _in_critic_warmup(self) -> bool:
        """Whether the value function is still being fitted with the actor held still.

        **Why.** Resuming a converged policy into a changed observation moves the value
        target twice over: the states visited change, and so do the returns (episodes end
        sooner). The critic's error then flows into the advantages, the surrogate gradient
        becomes mostly noise, and the entropy bonus -- which has no such problem -- wins,
        so the policy diffuses instead of adapting. Fitting the critic first removes the
        noise source before the actor is allowed to react to it.

        **How the actor is frozen, and why not ``lr = 0``.** During warm-up the total loss is
        the value term ALONE. The actor and critic are separate modules, so nothing backprops
        into the actor and its ``.grad`` stays ``None`` -- and Adam SKIPS parameters with a
        ``None`` gradient, so they cannot drift on stale momentum either. Zeroing the
        gradients instead would not be equivalent: Adam would still step them from the
        moments the teacher's run left behind.

        The obvious alternative, an actor parameter group at ``lr = 0``, was rejected because
        rsl-rl builds ONE Adam over ``chain(actor, critic)``, and re-grouping it makes the
        checkpoint's ``optimizer_state_dict`` unloadable -- ``--resume`` would break, which is
        the one thing a fine-tune cannot do without.

        **The adaptive learning rate is also suspended** while this is true (see
        :meth:`update`): with the actor still, the measured KL is ~0, which reads to the
        controller as "far below target" and would ratchet the shared learning rate up toward
        its 1e-2 cap -- applying it to the critic, which is the one thing training here.
        """
        return self._updates_done < self.critic_warmup_iterations

    @property
    def _use_reconstruction(self) -> bool:
        """Whether the auxiliary reconstruction term is active this run."""
        return self.reference_reconstruction_coeff > 0.0

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # [robot_rl] Reference-reconstruction loss
        mean_recon_loss = 0 if self._use_reconstruction else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            # Suspended during the critic warm-up: a still actor measures KL ~ 0, which
            # the controller reads as "far below target" and answers by ratcheting the
            # SHARED learning rate toward its 1e-2 cap -- onto the critic.
            if (self.desired_kl is not None and self.schedule == "adaptive"
                    and not self._in_critic_warmup):
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            if self._in_critic_warmup:
                # Value term ALONE. Nothing here touches the actor, so its grads stay
                # None and Adam leaves it exactly where the checkpoint put it.
                loss = self.value_loss_coef * value_loss
            else:
                loss = (surrogate_loss + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy.mean())

            # [robot_rl] Auxiliary reference-reconstruction loss on the actor's encoder.
            # Sliced to the original batch so a revived symmetry path does not weight the
            # mirrored copies (whose reference terms have no reflection handler yet).
            if self._use_reconstruction:
                recon_loss = self.actor.reconstruction_loss(batch.observations[:original_batch_size])
                loss = loss + self.reference_reconstruction_coeff * recon_loss

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            # The actor has no gradients at all during warm-up, and clipping an empty set
            # is an error rather than a no-op.
            if not self._in_critic_warmup:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # [robot_rl] Reference-reconstruction loss
            if mean_recon_loss is not None:
                mean_recon_loss += recon_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # [robot_rl] Reference-reconstruction loss
        if mean_recon_loss is not None:
            mean_recon_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        # [robot_rl] Reference-reconstruction loss
        if mean_recon_loss is not None:
            loss_dict["reference_recon"] = mean_recon_loss

        # [robot_rl] Critic warm-up: logged as a series rather than printed once, so a run's
        # own metrics say when the actor was released. Read `Loss/surrogate` against it -- the
        # surrogate is computed and reported throughout, it just is not optimized while this
        # is 1, and a reader who does not know that would call the flat stretch a plateau.
        if self.critic_warmup_iterations:
            loss_dict["critic_warmup"] = float(self._in_critic_warmup)

        # AFTER the flag is read, so iteration `critic_warmup_iterations` is the first one the
        # actor moves on, and a warm-up of N means exactly N frozen iterations.
        self._updates_done += 1

        return loss_dict
