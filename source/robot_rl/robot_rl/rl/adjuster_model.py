# Copyright 2026 Zachary Olkin. All rights reserved.

"""The adjuster actor: a residual on the generator's plan, over the generator's own encoder.

One query token per plan node, attending to the generator's FROZEN conditioning prefix, with a
shared per-token head emitting that node's correction. A sibling of
:class:`~robot_rl.flow_matching.value.FlowValueModel`, which does the same borrow for a scalar
readout -- the difference is that the query stream is ``F`` nodes rather than one token, and
the head returns a ``D``-vector per node rather than a number.

## The borrow is EXACT, and only under ``prefix`` attention

Under ``attention_mode="prefix"`` the generator's conditioning attends only to itself and
never receives the flow-time embedding, so its per-layer representation is a function of the
conditioning ALONE. Adding query tokens beside it therefore cannot change it:
``encode_prefix`` returns bit-for-bit what the generator computes, and the new tokens use it
as keys and values while attending to each other -- structurally what ``decode_future`` does
with the future tokens, so the borrowed layers are used the way they were trained to be used.
Under ``full`` attention there is no fixed encoding to borrow at all, so a non-``prefix``
checkpoint is refused rather than approximated.

## Six layers frozen, two TRAINABLE COPIES

The last two transformer layers are the adjuster's own copies, warm-started from the
generator's weights. Sharing them would mean training them changes the generator -- which is
the generator fine-tune that was already tried and hit a measured ceiling, not a residual
adjuster. With copies the generator is untouched and the borrow stays exact, because
``prefix_cache`` holds the CONDITIONING's per-layer inputs and those come from the frozen
trunk either way.

## The head is per-token and SHARED

One ``Linear(d_model -> D)`` applied to each of the ``F`` token outputs, replacing the
parent's flat trunk. That is worth being explicit about: a flat
``Linear(F * d_model -> F * D)`` is ~86M parameters and has to LEARN that output slot
``(j, c)`` corresponds to token ``j``, from a ``F * D``-dimensional policy gradient. The
shared head has that correspondence structurally, in ~22k parameters.

## Where the generator comes from

``rsl_rl`` builds a model from ``agent_cfg.to_dict()``, which carries primitives and not
objects, so the frozen generator cannot be passed in. It is read from
:func:`~robot_rl.flow_matching.online.active_fm_context`, which the env's command term
publishes when it loads its checkpoint -- one generator per process, borrowed rather than
duplicated. It is held in a plain LIST so ``nn.Module.__setattr__`` does not register it: the
generator's 27M parameters must stay out of ``state_dict``, out of ``parameters()`` (which
PPO's single optimizer and its gradient clipping both walk) and out of every checkpoint, or a
resume against a different generator would silently restore the wrong weights.

## The action is DIMENSIONLESS

The head's output is squashed and scaled into physical units by the action term, not here.
``MLPModel.forward`` returns ``distribution.sample()`` and ``PPO.update`` scores those stored
actions against a freshly updated distribution, so anything applied after the sample breaks
the log-prob contract. Same division of labour as
``JointPositionActionCfg(scale=G1_ACTION_SCALE)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

import copy

from robot_rl.flow_matching import online


class _PerTokenHead(nn.Module):
    """Applies one shared linear map to every token, then flattens.

    Stands in for the parent's trunk so ``MLPModel.forward``'s ``self.mlp(latent)`` still
    yields ``(B, F * D)`` and the distribution machinery above it is untouched.
    """

    def __init__(self, num_nodes: int, d_model: int, node_dim: int) -> None:
        """Build the shared head.

        Args:
            num_nodes: ``F``, the plan's node count.
            d_model: Transformer width the tokens arrive in.
            node_dim: ``D``, the per-node residual width.
        """
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.d_model = int(d_model)
        self.node_dim = int(node_dim)
        self.proj = nn.Linear(self.d_model, self.node_dim)
        # Zero, so the mean residual is EXACTLY zero at initialisation and the served plan is
        # bit-identical to the bare generator's. That is the baseline every measurement of
        # this policy is against, and a carelessly initialised head over thousands of action
        # dimensions swamps the plan on iteration one.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Map ``(B, F * d_model)`` token outputs to ``(B, F * D)``.

        Args:
            latent: Flattened per-token transformer outputs.

        Returns:
            The flattened per-node residual, dimensionless.
        """
        batch = latent.shape[0]
        tokens = latent.view(batch, self.num_nodes, self.d_model)
        return self.proj(tokens).reshape(batch, self.num_nodes * self.node_dim)


class WindowAdjusterModel(MLPModel):
    """Actor emitting a per-node residual on the generator's predicted window.

    Reached from a runner config as
    ``class_name="robot_rl.rl.adjuster_model:WindowAdjusterModel"``.

    Attributes:
        num_nodes: ``F``, taken from the live generator rather than configured.
        node_dim: ``D``, likewise.
        num_frozen_layers: How many borrowed layers stay frozen; the rest are own copies.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1,),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        plan_obs_group: str = "fm_plan",
        cond_obs_group: str = "fm_cond",
        num_frozen_layers: int = 6,
    ) -> None:
        """Build the adjuster.

        Args:
            obs: A live observation dict; the group widths are read from it.
            obs_groups: Observation-set to group-list mapping.
            obs_set: Which set to build ("actor").
            output_dim: ``F * D``, checked against the generator's own layout.
            hidden_dims: Sizes the parent's trunk, which this model REPLACES -- the head is a
                single shared per-token linear map by design (see the module docstring). The
                parent always builds one and ``rsl_rl.modules.MLP`` rejects an empty
                ``hidden_dims``, so the default is the smallest shape that constructs. Raising
                it only allocates a larger module to throw away.
            activation: Likewise unused by anything that survives construction.
            obs_normalization: Must be False: both groups arrive already normalized (the plan
                and its flow field in the generator's standardized space, the conditioning as
                the generator's own tokens), so an empirical normalizer would re-standardize a
                declared scale into a dataset statistic.
            distribution_cfg: Output distribution config, forwarded to the parent.
            plan_obs_group: Group carrying the predicted window and its flow field.
            cond_obs_group: Group carrying the generator's conditioning, as it was USED.
            num_frozen_layers: Borrowed layers to keep frozen. The remainder become trainable
                copies warm-started from the generator's weights.

        Raises:
            ValueError: If the generator is not a ``prefix`` checkpoint, if ``output_dim`` does
                not match its layout, if a named group is missing, if the group widths
                disagree with the generator, or if ``obs_normalization`` is set.
        """
        ctx = online.active_fm_context()
        fm = ctx.model
        if getattr(fm, "attention_mode", "full") != "prefix":
            raise ValueError(
                f"WindowAdjusterModel needs a prefix-attention generator, got "
                f"{getattr(fm, 'attention_mode', 'full')!r}. Under 'full' attention the "
                "conditioning's representation depends on the future tokens and the flow "
                "time, so there is no fixed encoding to borrow -- and the two modes share one "
                "parameter set, so a mislabelled checkpoint would run silently under the "
                "wrong attention."
            )
        if obs_normalization:
            raise ValueError(
                "WindowAdjusterModel takes obs_normalization=False: both its groups arrive in "
                "the generator's own normalized space, and the FM normalizer is DECLARED "
                "rather than fitted, so re-standardizing would demote a contract constant to "
                "a statistic of whichever rollout filled the buffer first."
            )

        self.num_nodes = int(ctx.F)
        self.node_dim = int(ctx.D)
        if int(output_dim) != self.num_nodes * self.node_dim:
            raise ValueError(
                f"output_dim {int(output_dim)} != F * D = {self.num_nodes} * {self.node_dim} "
                f"= {self.num_nodes * self.node_dim}. The action IS the per-node residual, so "
                "the env's action term must declare exactly that width."
            )

        self._plan_group = str(plan_obs_group)
        self._cond_group = str(cond_obs_group)
        self.num_frozen_layers = int(num_frozen_layers)
        self._d_model = int(fm.d_model)
        # A LIST, not an attribute: nn.Module.__setattr__ registers Modules, and a Module held
        # inside a list is not one -- so the generator stays out of state_dict(), out of
        # parameters(), and out of every PPO checkpoint. See the module docstring.
        self._fm = [fm]

        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg,
        )

        # The per-token head replaces the parent's flat trunk. Safe to swap AFTER the parent
        # ran: `GaussianDistribution.init_mlp_weights` is a no-op (only the heteroscedastic
        # variant touches the trunk), so nothing the parent did to `self.mlp` is lost.
        self.mlp = _PerTokenHead(self.num_nodes, self._d_model, self.node_dim)

        # Query-token embedding: each node's [flow velocity | predicted node] pair.
        self.token_in = nn.Linear(2 * self.node_dim, self._d_model)
        # Learned rather than borrowed from the generator's sinusoidal timeline: node 0 is one
        # control step out and node F-1 is over a second, so what a slot MEANS here is a
        # look-ahead horizon, which is `FlowQModel`'s own argument for its `action_pos`.
        self.node_pos = nn.Parameter(torch.zeros(1, self.num_nodes, self._d_model))
        self.query_type = nn.Parameter(torch.zeros(1, 1, self._d_model))

        # Belt and braces: `load_fm_context` already freezes what it loads, and this model
        # REQUIRES that -- its parameters are not in `self.parameters()` so PPO would never
        # step them, but a live `.grad` on 27M weights is memory spent on nothing, and the
        # freeze is also what keeps this model's numerics identical to every other consumer's
        # (see the fast-path note in `online.load_fm_context`). Repeated here so a hand-built
        # context in a test cannot arrive unfrozen. Activations still carry gradient -- see
        # `get_latent`, where that distinction is the load-bearing one.
        for p in fm.parameters():
            p.requires_grad_(False)

        # Trainable copies of the trailing layers, warm-started from the generator.
        total = len(fm.encoder.layers)
        if not 0 <= self.num_frozen_layers <= total:
            raise ValueError(
                f"num_frozen_layers must be in [0, {total}], got {self.num_frozen_layers}."
            )
        self.tuned_layers = nn.ModuleList(
            copy.deepcopy(layer) for layer in fm.encoder.layers[self.num_frozen_layers:]
        )
        # Explicitly, and NOT left to the deepcopy: it inherits `requires_grad` from the
        # source, which was just frozen, so the copies would arrive frozen too -- a model that
        # trains only its embeddings and head while reporting the layer count as trainable.
        for p in self.tuned_layers.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] WindowAdjusterModel: F={self.num_nodes} D={self.node_dim} "
              f"d_model={self._d_model}, {self.num_frozen_layers} borrowed layers frozen + "
              f"{len(self.tuned_layers)} trainable copies, {trainable} trainable parameters "
              "(the generator is borrowed, not owned)")

    # ------------------------------------------------------------------
    # Group resolution
    # ------------------------------------------------------------------
    def _get_obs_dim(self, obs, obs_groups, obs_set) -> tuple[list[str], int]:
        """Resolve and check the two groups; see :meth:`MLPModel._get_obs_dim`.

        The parent concatenates its groups into one flat latent, which this model does not do
        -- so what it returns is only used for bookkeeping and for the (disabled) normalizer.
        The real work of this override is CHECKING the widths against the live generator,
        because every group here is a slice of the generator's own state and a width that
        merely happens to fit would be read as the wrong fields.

        Args:
            obs: Live observations.
            obs_groups: Observation-set to group-list mapping.
            obs_set: Which set to build.

        Returns:
            ``(groups, total_width)``.

        Raises:
            ValueError: If a group is missing or its width disagrees with the generator.
        """
        groups = list(obs_groups[obs_set])
        for name in (self._cond_group, self._plan_group):
            if name not in groups:
                raise ValueError(
                    f"WindowAdjusterModel needs observation group {name!r} in obs_groups"
                    f"[{obs_set!r}], which lists {groups}."
                )
        fm_ctx = online.active_fm_context()

        want_plan = 2 * self.num_nodes * self.node_dim
        have_plan = int(obs[self._plan_group].shape[-1])
        if have_plan != want_plan:
            raise ValueError(
                f"group {self._plan_group!r} is {have_plan} wide but the generator's plan plus "
                f"its flow field is 2 * F * D = 2 * {self.num_nodes} * {self.node_dim} = "
                f"{want_plan}."
            )

        want_cond = (fm_ctx.N * fm_ctx.D
                     + fm_ctx.extero_sensor_count * fm_ctx.extero_sensor_dim
                     + fm_ctx.num_cmd_tokens * fm_ctx.cmd_dim)
        have_cond = int(obs[self._cond_group].shape[-1])
        if have_cond != want_cond:
            raise ValueError(
                f"group {self._cond_group!r} is {have_cond} wide but this generator's "
                f"conditioning is history {fm_ctx.N}x{fm_ctx.D} + extero "
                f"{fm_ctx.extero_sensor_count}x{fm_ctx.extero_sensor_dim} + cmd "
                f"{fm_ctx.num_cmd_tokens}x{fm_ctx.cmd_dim} = {want_cond}."
            )
        self._cond_split = (
            fm_ctx.N * fm_ctx.D,
            fm_ctx.extero_sensor_count * fm_ctx.extero_sensor_dim,
            fm_ctx.num_cmd_tokens * fm_ctx.cmd_dim,
        )
        return groups, have_plan + have_cond

    def _get_latent_dim(self) -> int:
        """Input width the parent sizes its trunk from -- and that trunk is DISCARDED.

        ``__init__`` replaces ``self.mlp`` with the shared per-token head, so this value only
        decides the shape of a module that is built and thrown away. It returns ``d_model``
        rather than the honest ``F * d_model`` for exactly that reason: at ``F * d_model`` the
        parent would allocate a ``Linear(31744, 2728)`` -- 86M parameters, ~350 MB -- before
        anything replaced it.

        :meth:`get_latent` really does return ``F * d_model``; the head reshapes it.
        """
        return self._d_model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        """Run the query stream over the generator's frozen conditioning prefix.

        Args:
            obs: The observation mapping; reads only the two configured groups.
            masks: Recurrent padding mask (unused; this model is not recurrent).
            hidden_state: Unused.

        Returns:
            ``(B, F * d_model)`` per-token outputs, for the shared head.
        """
        fm = self._fm[0]
        ctx = online.active_fm_context()
        n_hist, n_extero, _ = self._cond_split

        cond = obs[self._cond_group]
        history = cond[:, :n_hist].reshape(-1, ctx.N, ctx.D)
        extero = cond[:, n_hist:n_hist + n_extero]
        cmd = cond[:, n_hist + n_extero:]
        cmd_tokens = (cmd.reshape(-1, ctx.num_cmd_tokens, ctx.cmd_dim)
                      if ctx.cmd_dim > 0 else None)

        plan = obs[self._plan_group]
        half = self.num_nodes * self.node_dim
        window = plan[:, :half].reshape(-1, self.num_nodes, self.node_dim)
        flow = plan[:, half:].reshape(-1, self.num_nodes, self.node_dim)

        # The conditioning prefix is entirely frozen, so it needs no graph. This is the one
        # expensive part of the forward and it is recomputed rather than cached because the
        # rollout runs under `inference_mode`: a tensor made there cannot enter an autograd
        # graph later, which is why stock PPO re-runs the actor from stored observations too.
        with torch.no_grad():
            prefix = fm.encode_prefix(history, extero, cmd_tokens, None)

        x = self.token_in(torch.cat([flow, window], dim=-1)) + self.node_pos + self.query_type
        # The BORROWED layers run WITH grad even though their parameters are frozen, and that
        # distinction is the whole reason `requires_grad_(False)` is the right tool here and
        # `no_grad` is not: `token_in`, `node_pos` and `query_type` are upstream of them, so
        # wrapping these in `no_grad` would sever the graph and silently train nothing but the
        # head. Frozen parameters get no gradient; the activations flowing through them must.
        for i, layer in enumerate(fm.encoder.layers[:self.num_frozen_layers]):
            x = fm._run_layer(layer, x, torch.cat([prefix[i], x], dim=1))
        for j, layer in enumerate(self.tuned_layers):
            i = self.num_frozen_layers + j
            x = fm._run_layer(layer, x, torch.cat([prefix[i], x], dim=1))
        return x.reshape(x.shape[0], self.num_nodes * self._d_model)
