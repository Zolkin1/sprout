# Copyright 2026 Zachary Olkin. All rights reserved.

"""The autoregressive token planner -- stage B.

A prefix-LM over the flow-matching generator's own conditioning: the conditioning tokens
attend bidirectionally among themselves, and ``N`` output slots attend causally to the prefix
and to the slots before them. The head is categorical over the tokenizer's ``V``-word
vocabulary, so one plan is ``N`` discrete decisions rather than a continuous vector.

.. admonition:: Why discrete, given the FM works

    Not accuracy -- the tokenizer's quantization error is measurably below the generator's own
    prediction error on most channel groups. It is that a categorical policy has an EXACT
    per-token log-probability and an EXACT entropy. PPO then needs neither a
    reparameterisation trick nor a surrogate for the sampling density, which is what makes
    fine-tuning a flow model awkward. Everything in :meth:`TokenPlanner.sample` and
    :meth:`TokenPlanner.logprobs` is shaped by that: per-token values (never a sum), one
    forward pass for a batch of ``(cond, tokens)``, and every random draw through an explicit
    generator.

.. admonition:: The prefix is encoded ONCE per plan

    Under the block mask the conditioning's representation at every layer is a function of the
    conditioning alone, so :meth:`encode_prefix` computes it once and all ``N`` decode steps
    reuse it. That is the same decomposition
    :meth:`~robot_rl.flow_matching.model.FlowMatchingTransformer.encode_prefix` uses, and for
    the same second reason: expressing the block structure as an attention MASK makes the
    fused kernel materialize ``(B, nhead, S, S)``, which at simulator batch sizes is
    gigabytes. Here the conditioning is simply not in the query set.

**Pre-norm, where the generator is post-norm.** That is the one trunk-level departure, and it
is deliberate: this is a deep decoder with a tied 12800-way embedding, where post-norm needs
its residual branches warmed up. :func:`_run_layer` therefore reproduces
``nn.TransformerEncoderLayer(norm_first=True)`` rather than the generator's default, and
``test_run_layer_matches_a_masked_full_pass`` pins it -- the op order has three separate
dropout modules and a wrong one is invisible at ``dropout=0``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import PlannerConfig
from .prefix import ConditioningPrefix
from .tokenizer import TrajectoryTokenizer


def _run_layer(
    layer: nn.Module, x: torch.Tensor, kv: torch.Tensor,
    attn_mask: torch.Tensor | None = None, kv_normed: bool = False,
) -> torch.Tensor:
    """One PRE-norm ``nn.TransformerEncoderLayer``, with keys/values given separately.

    Reproduces ``norm_first=True`` exactly, differing only in that the keys and values may
    come from a longer sequence than the queries -- ordinary cross-attention, so the fused
    projections, the head split and the SDPA scaling all stay PyTorch's.

    ``kv`` is passed RAW (the layer's input) and normed here, because in a full pre-norm pass
    the keys are ``norm1`` of the layer input and ``LayerNorm`` is per token -- so norming the
    concatenation is the same thing as concatenating the normed halves.

    Args:
        layer: One layer of the trunk, built with ``norm_first=True``.
        x: Query rows and residual stream, ``(B, Lq, d_model)``.
        kv: Key/value rows, ``(B, Lkv, d_model)``, un-normed.
        attn_mask: Optional ``(Lq, Lkv)`` bool mask; ``True`` means *not allowed to attend*.
        kv_normed: ``kv`` has already had ``norm1`` applied. Exact, not an approximation:
            ``LayerNorm`` is per token, so norming a concatenation and concatenating normed
            halves give the same tensor. Incremental decoding uses it to avoid re-norming
            every key at every step, which would be quadratic in the token count.

    Returns:
        ``(B, Lq, d_model)``.
    """
    xn = layer.norm1(x)
    kvn = kv if kv_normed else layer.norm1(kv)
    a = layer.self_attn(xn, kvn, kvn, attn_mask=attn_mask, need_weights=False)[0]
    x = x + layer.dropout1(a)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
    return x + layer.dropout2(ff)


def causal_output_mask(
    num_tokens: int, prefix_len: int, device: torch.device | None = None
) -> torch.Tensor:
    """The ``(N, P + N)`` mask: full attention to the prefix, causal among the outputs.

    Args:
        num_tokens: Output slots ``N``.
        prefix_len: Conditioning tokens ``P``.
        device: Where to build it.

    Returns:
        A bool mask, ``True`` where attention is FORBIDDEN.
    """
    out = torch.triu(
        torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=device), diagonal=1)
    return torch.cat(
        [torch.zeros(num_tokens, prefix_len, dtype=torch.bool, device=device), out], dim=1)


def _resolve_extero_channels(cfg: PlannerConfig) -> int:
    """Planes per exteroception pixel, resolved the way the rest of the repo resolves it.

    ``None`` means "resolve from the source" -- 5 for depth, 1 for a height scan, whose rays
    already are ground points. That idiom is
    :meth:`~robot_rl.flow_matching.config.FlowMatchingConfig.resolve_extero_channels`'s, and
    hard-coding a 1 here would be a fifth place the rule is written down.

    Args:
        cfg: The planner config.

    Returns:
        The channel count.
    """
    if cfg.extero_channels is not None:
        return int(cfg.extero_channels)
    return 5 if str(cfg.extero_source) == "depth" else 1


class TokenPlanner(nn.Module):
    """Prefix-LM over trajectory tokens, with a frozen tokenizer for decoding.

    Submodules are separately addressable on purpose (:attr:`prefix`, :attr:`encoder`,
    :attr:`head`), so the RL stage can freeze the conditioning encoders without surgery.

    Args:
        cfg: The planner config. Validated here.
        tokenizer: The FROZEN tokenizer whose vocabulary this planner emits. Its parameters
            are put out of the autograd graph on construction -- a planner that fine-tuned its
            own tokenizer would be emitting tokens whose meaning drifts under it, so every
            plan generated before the drift would decode to something else afterwards.
    """

    def __init__(self, cfg: PlannerConfig, tokenizer: TrajectoryTokenizer):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.num_tokens = int(cfg.num_tokens)
        self.vocab_size = int(cfg.vocab_size)
        d = int(cfg.d_model)

        self.tokenizer = tokenizer.eval().requires_grad_(False)

        self.prefix = ConditioningPrefix(
            history_dim=cfg.history_dim, num_history_states=cfg.num_history_states,
            future_len=cfg.future_len, extero_sensor_dim=cfg.extero_sensor_dim,
            extero_sensor_count=cfg.extero_sensor_count, d_model=d,
            extero_encoder=cfg.extero_encoder, extero_image_hw=cfg.extero_image_hw,
            extero_cnn_channels=cfg.extero_cnn_channels,
            extero_cnn_grid=cfg.extero_cnn_grid, extero_cnn_strides=cfg.extero_cnn_strides,
            extero_coord_extent_m=cfg.extero_coord_extent_m,
            extero_grid_ordering=cfg.extero_grid_ordering,
            extero_fourier_bands=cfg.extero_fourier_bands,
            extero_miss_sentinel=cfg.extero_miss_sentinel,
            extero_channels=_resolve_extero_channels(cfg),
            extero_camera_intrinsics=cfg.extero_camera_intrinsics,
            cmd_dim=cfg.cmd_dim, cmd_stride=cfg.cmd_stride,
        )

        # Output side. `token_emb` is tied to the head, so the vocabulary is embedded and
        # scored by one matrix; at V=12800 that is 6.6M parameters saved and, more to the
        # point, the two views of a token cannot drift apart.
        self.token_emb = nn.Embedding(self.vocab_size, d)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        # N slots, not N+1. The layout is [BOS, t_1 .. t_N], but t_N is only ever a TARGET --
        # feeding it would be feeding the answer -- so the model is only ever given
        # [BOS, t_1 .. t_{N-1}], which is N positions. An N+1st embedding would be a
        # parameter that never receives a gradient.
        self.out_pos = nn.Parameter(torch.zeros(1, self.num_tokens, d))
        # ONE type embedding, on the output tokens. A matching `prefix_type` would be exactly
        # degenerate: the prefix already tags its three streams individually, so a constant
        # added to all of them is absorbable into those three and would receive gradient that
        # only trades off against them.
        self.out_type = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.out_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        # `enable_nested_tensor=False` because the stock forward is never called -- every
        # pass here goes through `_run_layer`, which needs the key/value source separate from
        # the query. Left on, it only emits a warning about a fast path that cannot apply.
        self.encoder = nn.TransformerEncoder(
            layer, cfg.num_layers, enable_nested_tensor=False)
        self.head_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.vocab_size, bias=False)
        self.head.weight = self.token_emb.weight        # tied
        # The prefix `forward` last built, so a caller that teacher-forces and then samples
        # the same batch can hand it back to `sample(prefix_cache=...)`. Deliberately NOT
        # consulted automatically: it belongs to whatever conditioning ran last, and using
        # it implicitly would silently sample against a different batch's prefix.
        self._last_prefix_cache: tuple[torch.Tensor, ...] | None = None

    @property
    def prefix_len(self) -> int:
        """Conditioning tokens ``P``."""
        return self.prefix.prefix_len

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    def encode_prefix(
        self,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the conditioning through the trunk on its own; return its per-layer INPUTS.

        What an output slot needs at layer ``i`` is the conditioning's input to that layer
        (its keys and values there), so that is what is returned: ``L`` tensors, not ``L+1``.

        Args:
            history: ``(B, N_hist, history_dim)``.
            extero: ``(B, T, extero_sensor_dim)`` or anything reshaping to it.
            cmd_tokens: ``(B, num_cmd_tokens, cmd_dim)`` or ``None``.
            history_null_mask: ``(B,)`` bool null-history selector.

        Returns:
            One ``(B, P, d_model)`` tensor per layer, in layer order.
        """
        x = self.prefix(history, extero, cmd_tokens, history_null_mask)
        cache: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            cache.append(x)
            x = _run_layer(layer, x, x)
        return tuple(cache)

    def _embed_outputs(self, tokens_in: torch.Tensor | None, batch: int) -> torch.Tensor:
        """Embed the teacher-forced input slots ``[BOS, t_1 .. t_{N-1}]``.

        Args:
            tokens_in: ``(B, N-1)`` ids for slots 1..N-1, or ``None`` for BOS alone.
            batch: Batch size, needed when ``tokens_in`` is ``None``.

        Returns:
            ``(B, n_slots, d_model)`` with the positions and the type tag added.
        """
        bos = self.bos.expand(batch, 1, self.cfg.d_model)
        x = bos if tokens_in is None else torch.cat(
            [bos, self.token_emb(tokens_in)], dim=1)
        return x + self.out_pos[:, :x.shape[1], :] + self.out_type

    # ------------------------------------------------------------------
    # Teacher forcing
    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: torch.Tensor,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced logits for every output position, in ONE pass.

        Args:
            tokens: Target ids ``(B, N)``. Slots 1..N-1 are fed back as inputs; slot ``N-1``'s
                target is never fed anywhere.
            history: ``(B, N_hist, history_dim)``.
            extero: ``(B, T, extero_sensor_dim)``.
            cmd_tokens: ``(B, num_cmd_tokens, cmd_dim)`` or ``None``.
            history_null_mask: ``(B,)`` bool.

        Returns:
            ``(B, N, V)`` logits, position ``i`` predicting ``tokens[:, i]``.

        Raises:
            ValueError: On the wrong token count.
        """
        if tokens.shape[-1] != self.num_tokens:
            raise ValueError(
                f"expected {self.num_tokens} token(s) per plan, got {tokens.shape[-1]}.")
        cache = self.encode_prefix(history, extero, cmd_tokens, history_null_mask)
        self._last_prefix_cache = cache
        x = self._embed_outputs(tokens[:, :-1], tokens.shape[0])
        mask = causal_output_mask(self.num_tokens, self.prefix_len, tokens.device)
        for layer, prefix in zip(self.encoder.layers, cache):
            x = _run_layer(layer, x, torch.cat([prefix, x], dim=1), mask)
        return self.head(self.head_norm(x))

    def logprobs(
        self,
        tokens: torch.Tensor,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        return_logits: bool = False,
    ):
        """Per-token log-probabilities and entropies for GIVEN tokens, in one pass.

        Per token, never summed: PPO needs the ratio at each decision, and a summed
        log-probability throws away which of the ``N`` decisions the advantage belongs to.

        Args:
            tokens: ``(B, N)`` ids to score.
            history / extero / cmd_tokens / history_null_mask: The conditioning.
            temperature: Softmax temperature; must match what :meth:`sample` drew at.
            return_logits: Also return the ``(B, N, V)`` logits **already divided by
                ``temperature``**, so an exact KL to a reference policy is a log-softmax away
                and needs no second pass. Returning the raw logits beside tempered
                log-probabilities would make the two describe different distributions, and
                the caller would have to know to divide.

        Returns:
            ``(logprobs, entropy)`` each ``(B, N)``, or ``(logprobs, entropy, logits)``.

        Raises:
            ValueError: On a non-positive temperature.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        logits = self(tokens, history, extero, cmd_tokens, history_null_mask) / temperature
        logp_all = F.log_softmax(logits, dim=-1)
        logp = logp_all.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ent = -(logp_all.exp() * logp_all).sum(-1)
        return (logp, ent, logits) if return_logits else (logp, ent)

    # ------------------------------------------------------------------
    # Autoregressive decoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        greedy: bool = False,
        generator: torch.Generator | None = None,
        prefix_cache: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a plan autoregressively, with a per-layer KV cache.

        No attention mask: at step ``i`` the query is the single new slot and the keys are the
        prefix plus the slots already emitted, so causality holds by construction. That is
        what makes this agree with :meth:`forward` to float round-off (~1e-6, measured) rather
        than merely approximately -- the two sum the same numbers in a different order, one
        over a growing key set and one over the whole block. Pinned by
        ``test_sample_and_logprobs_agree_on_the_same_tokens``.

        No top-k and no nucleus truncation, deliberately: both put zero probability on tokens
        the model gave mass to, which makes the sampled log-probability disagree with the
        one :meth:`logprobs` computes and silently biases a PPO ratio.

        Args:
            history / extero / cmd_tokens / history_null_mask: The conditioning.
            temperature: Softmax temperature.
            greedy: Take the argmax instead of sampling. ``logprobs`` are still reported
                under the tempered distribution, so they remain the density of what was
                emitted.
            generator: Explicit RNG, so a rollout is reproducible.
            prefix_cache: A cache from :meth:`encode_prefix` for this same conditioning, to
                skip re-encoding it. The whole point of the decomposition is that the prefix
                is a function of the conditioning alone, so a caller that has just run
                :meth:`forward` on the same batch already has it -- and re-encoding was
                measured at 18% of a validation batch's cost, doubled by the EMA arm.

        Returns:
            ``(tokens, logprobs, entropy)`` of shapes ``(B, N)``, ``(B, N)``, ``(B, N)``.

        Raises:
            ValueError: On a non-positive temperature.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if self.training:
            # The per-token log-probability this returns is only EXACT against
            # :meth:`logprobs` in eval mode: the shipped dropout is 0.1 (the generator's is
            # 0.0), and two independent dropout draws over the same tokens disagree by up to
            # 0.087 nats -- a ~9% error in a PPO importance ratio, in the direction of noise
            # rather than signal. A rollout collected from a model left in train mode is the
            # easy way to make that happen, so it is refused rather than warned about.
            raise RuntimeError(
                "TokenPlanner.sample() requires eval mode: this model ships dropout="
                f"{self.cfg.dropout}, so sampling under train mode returns log-probabilities "
                "drawn through a different network than logprobs() would score them with. "
                "Call model.eval() first (and model.train() again afterwards to resume "
                "training)."
            )
        if (generator is not None and not greedy
                and generator.device.type != history.device.type):
            # torch.multinomial requires the two to agree, and the error it raises names
            # neither the argument nor the fix. Caught here because a rollout's whole point
            # is the explicit generator, so silently ignoring it is not an option either.
            raise ValueError(
                f"the sampling generator is on '{generator.device.type}' but the "
                f"conditioning is on '{history.device.type}'. Build it as "
                f"torch.Generator(device='{history.device.type}') -- a generator is not "
                "portable across devices, so this cannot be papered over without changing "
                "which numbers a seed produces.")
        b = history.shape[0]
        cache = (self.encode_prefix(history, extero, cmd_tokens, history_null_mask)
                 if prefix_cache is None else tuple(prefix_cache))
        if len(cache) != len(self.encoder.layers):
            raise ValueError(
                f"prefix_cache has {len(cache)} entries but the trunk has "
                f"{len(self.encoder.layers)} layers; it must carry one per layer.")
        layers = list(self.encoder.layers)
        pl, n, d = self.prefix_len, self.num_tokens, self.cfg.d_model

        # A PREALLOCATED per-layer key/value buffer, holding [prefix | slots emitted so far],
        # pre-normed. Both properties are worth their memory:
        #
        # * preallocated -- the obvious `torch.cat([prefix, *past], 1)` rebuilds a growing
        #   tensor once per layer per step, so a 16-token plan through 8 layers is 128
        #   allocate-and-copy pairs of up to (B, 145, 512). Writing into a slice and reading a
        #   VIEW of it is zero copies.
        # * pre-normed -- `_run_layer` would otherwise apply `norm1` to every key again at
        #   every step, which is quadratic in the token count for a result that cannot change.
        #
        # It costs `L * B * (P + N) * d * 4` bytes: 2.4 MB at batch 1, where the model is
        # dispatch-bound and removing 128 allocations per plan is worth far more than the
        # memory -- but 608 MB at batch 256 and 2.4 GB at batch 1024, which is the regime an
        # RL rollout and `benchmark_planner.py --batches 1024` both run in. Quoting only the
        # batch-1 figure would make that look free.
        kv = [torch.empty(b, pl + n, d, device=history.device, dtype=cache[0].dtype)
              for _ in layers]
        for i, layer in enumerate(layers):
            kv[i][:, :pl] = layer.norm1(cache[i])

        tokens, logps, ents = [], [], []
        x = self._embed_outputs(None, b)                       # (B, 1, d): the BOS slot
        for step in range(self.num_tokens):
            h = x
            for i, layer in enumerate(layers):
                kv[i][:, pl + step:pl + step + 1] = layer.norm1(h)
                h = _run_layer(layer, h, kv[i][:, :pl + step + 1], kv_normed=True)
            logits = self.head(self.head_norm(h))[:, -1, :]     # (B, V)
            logp_all = F.log_softmax(logits / temperature, dim=-1)
            if greedy:
                nxt = logp_all.argmax(dim=-1)
            else:
                nxt = torch.multinomial(logp_all.exp(), 1, generator=generator).squeeze(-1)
            tokens.append(nxt)
            logps.append(logp_all.gather(-1, nxt.unsqueeze(-1)).squeeze(-1))
            ents.append(-(logp_all.exp() * logp_all).sum(-1))
            if step + 1 < self.num_tokens:
                x = (self.token_emb(nxt).unsqueeze(1)
                     + self.out_pos[:, step + 1:step + 2, :] + self.out_type)
        return (torch.stack(tokens, 1), torch.stack(logps, 1), torch.stack(ents, 1))

    # ------------------------------------------------------------------
    # Decoding to a trajectory
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(self, tokens: torch.Tensor, normalizer=None,
               project_rotations: bool = True) -> torch.Tensor:
        """Frozen tokenizer decoder, denormalization, and the rotation projection.

        The projection is not cosmetic. ``base_pose``'s rotation half is six free numbers,
        and a decoder emitting them produces something only APPROXIMATELY a rotation -- so a
        served reference would not be a real pose, and the autoregressive history would drift
        off the manifold it was trained on. The flow-matching generator projects at exactly
        the same point (``online.sample_window``), which is what lets everything downstream,
        including the torch-free deploy port, assume the window is valid.

        Args:
            tokens: ``(B, N)`` ids.
            normalizer: The state :class:`~robot_rl.flow_matching.normalizer.Normalizer`. When
                ``None`` the nodes are returned in NORMALIZED space -- useful in a test, wrong
                for anything that hands them to a tracker.
            project_rotations: Re-orthonormalize the rot6d block. Only turn this off to
                MEASURE how far off the manifold the raw decode is; serving an unprojected
                window is never correct. Requires ``normalizer``, since the projection is
                defined in physical space.

        Returns:
            ``(B, H, node_dim)`` nodes.
        """
        nodes = self.tokenizer.decode_ids(tokens)
        if normalizer is None:
            return nodes
        nodes = normalizer.denormalize(nodes)
        if project_rotations:
            from robot_rl.flow_matching.window import (  # noqa: PLC0415
                project_window_rotations,
            )

            nodes = project_window_rotations(
                nodes, int(self.cfg.num_joints_for_layout),
                int(self.cfg.num_ee_frames_for_layout),
                list(self.cfg.tokenizer_cfg.reference_blocks),
            )
        return nodes
