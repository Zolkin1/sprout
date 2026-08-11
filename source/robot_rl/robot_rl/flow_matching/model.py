# Copyright 2026 Zachary Olkin. All rights reserved.

"""Single-transformer flow-matching model: robot-state-conditioned future prediction.

One transformer (encoder-only, full self-attention) operates on a single token
sequence assembled from four streams::

    [ history(N) | extero(T*S) | cmd(Ncmd) | future(F) ]

- **history**: ``N`` real-robot-state steps (the recent past, in the xyz+yaw ground
  frame at the anchor = step ``N-1``). Pure conditioning, never noised. Its only job
  is to supply *phase*. Embedded by its own ``history_in`` MLP + a learned
  ``hist_type``. A learned ``null_history`` embedding replaces the history content
  when ``history_null_mask`` is set (classifier-free-guidance / phase-agnostic mode).
- **extero**: ``T`` sensor readings of ``extero_sensor_dim`` values each, becoming ``S``
  tokens apiece (so ``T*S`` in the sequence), all carrying ``extero_type`` plus a
  per-sensor ``extero_id`` when T > 1. Pure conditioning. Two encoders:

  * ``extero_encoder="mlp"`` -- ``S = 1``, one token per reading through a shared
    ``extero_in`` MLP. The height scan's only option (T=1, 176 rays), and the A/B
    baseline for the cameras.
  * ``extero_encoder="cnn"`` -- ``S`` = a conv grid's cells (16 on the 30x26 pair), one
    trunk PER CAMERA, each cell also carrying a learned spatial-slot embedding. The flat
    path is permutation-invariant over pixels, so nothing in it can attend to WHERE in a
    frame a step edge is -- which is the shape of the measured registration failure
    (``scripts/rsl_rl/fm_registration_probe.py``). See
    :mod:`~robot_rl.flow_matching.extero_encoder`.

- **cmd**: the strided future velocity-command profile, via ``cmd_in`` + ``cmd_type``,
  each token positioned at its future timestep. Pure conditioning.
- **future**: the ``F`` state tokens the flow denoises -- ALL future (no past echo).

State/history/cmd tokens carry a sinusoidal positional encoding over a ``N + F``
timeline (history at ``0..N-1``, future at ``N..N+F-1``, each cmd at ``N + its step``).
The extero tokens carry no TIMELINE position -- the reading is "now" -- though under the
CNN they do carry a spatial one within their frame. Flow-time ``tau`` is a sinusoidal
embedding through a small MLP, added to every token. The head reads only the trailing
``F`` future-token outputs and predicts the flow vector field. There is NO inpainting /
known-mask: the conditioning is separate tokens, not held-clean slots of the denoised
window.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .extero_encoder import ExteroImageEncoderSet


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding, shape ``(1, seq_len, d_model)``.

    Standard formulation from "Attention is All You Need" (Vaswani et al., 2017).
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _timestep_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of a continuous flow-time scalar.

    Args:
        tau: Flow time in ``[0, 1]``, shape ``(B,)``.
        dim: Embedding dimension.

    Returns:
        Embedding tensor, shape ``(B, dim)``. ``tau`` is scaled to a ``[0, 1000]``
        range so the low-frequency components vary meaningfully over ``[0, 1]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float) / max(half, 1)
    )
    args = tau.float().unsqueeze(-1) * 1000.0 * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:  # odd dim
        emb = torch.cat([emb, torch.zeros(emb.shape[0], dim - emb.shape[-1], device=emb.device)], dim=-1)
    return emb


def _token_mlp(in_dim: int, d_model: int) -> nn.Sequential:
    """Two-layer token-embedding MLP ``in_dim -> d_model -> d_model``."""
    return nn.Sequential(nn.Linear(in_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))


class FlowMatchingTransformer(nn.Module):
    """Single-transformer conditional flow-matching model (robot-state conditioned).

    Args:
        state_dim: Per-step state width ``D`` (the flow operates in this space; also
            the per-step width of the history tokens).
        future_len: Future window length ``F`` (number of denoised state tokens).
        num_history_states: Number of history (phase) conditioning tokens ``N``.
        history_dim: Per-step history-token width (defaults to ``state_dim``).
        extero_sensor_dim: Width of ONE exteroception token (176 = height-scan rays,
            780 = one depth image).
        extero_sensor_count: Number of exteroception tokens (1 = height scan, 2 = both cameras).
        cmd_dim: Per-step command width; ``0`` disables command conditioning.
        cmd_stride: Keep every ``cmd_stride``-th step of the command profile as a token.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Transformer depth.
        dim_feedforward: Feedforward dimension in each transformer layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        state_dim: int,
        future_len: int,
        num_history_states: int = 2,
        history_dim: int | None = None,
        extero_sensor_dim: int = 1,
        extero_sensor_count: int = 1,
        cmd_dim: int = 0,
        cmd_stride: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        extero_encoder: str = "mlp",
        extero_image_hw: Sequence[Sequence[int]] | None = None,
        extero_cnn_channels: Sequence[int] = (16, 32, 64),
        extero_cnn_grid: Sequence[int] | None = None,
        attention_mode: str = "full",
    ):
        super().__init__()
        if attention_mode not in ("full", "prefix"):
            raise ValueError(
                f"unknown attention_mode {attention_mode!r}; expected 'full' or 'prefix'."
            )
        self.attention_mode = str(attention_mode)
        self.d_model = d_model
        self.state_dim = state_dim
        self.future_len = future_len
        self.num_history_states = num_history_states
        self.history_dim = history_dim if history_dim is not None else state_dim
        self.extero_sensor_dim = extero_sensor_dim
        self.extero_sensor_count = extero_sensor_count
        self.extero_encoder = str(extero_encoder)
        self.cmd_dim = cmd_dim
        self.cmd_stride = cmd_stride

        # Future (denoised) state tokens.
        self.state_in = _token_mlp(state_dim, d_model)
        # History (phase) conditioning tokens + a learned null-history embedding.
        self.history_in = _token_mlp(self.history_dim, d_model)
        self.hist_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        # Exteroception conditioning, one of two shapes -- never both, so a state dict cannot
        # load into the wrong one:
        #
        #   "mlp" -- ONE token per sensor through a shared 2-layer MLP. The height scan's
        #            only option (176 rays -> one token), and the A/B baseline for depth.
        #   "cnn" -- a conv trunk PER CAMERA emitting a grid of tokens, so attention can
        #            weight "under the foot" against "two metres ahead" inside one frame.
        #            The flat path cannot: it is permutation-invariant over pixels.
        if self.extero_encoder == "cnn":
            if not extero_image_hw:
                raise ValueError(
                    "extero_encoder='cnn' needs extero_image_hw (one (H, W) per sensor); the "
                    "dataset derives it from the recording's depth_camera_spec. The height "
                    "scan has no image shape in an FM recording -- use 'mlp' for it."
                )
            shapes = [(int(h), int(w)) for h, w in extero_image_hw]
            if len(shapes) != extero_sensor_count:
                raise ValueError(
                    f"extero_image_hw has {len(shapes)} shape(s) but extero_sensor_count="
                    f"{extero_sensor_count}."
                )
            bad = [hw for hw in shapes if hw[0] * hw[1] != extero_sensor_dim]
            if bad:
                raise ValueError(
                    f"extero_image_hw {bad} does not unfold the {extero_sensor_dim}-wide "
                    "sensor row."
                )
            self.extero_in = None
            self.extero_cnn = ExteroImageEncoderSet(
                shapes, d_model, extero_cnn_channels,
                None if extero_cnn_grid is None else (int(extero_cnn_grid[0]),
                                                      int(extero_cnn_grid[1])),
            )
            self.tokens_per_sensor = int(self.extero_cnn.tokens_per_sensor)
        elif self.extero_encoder == "mlp":
            self.extero_in = _token_mlp(extero_sensor_dim, d_model)
            self.extero_cnn = None
            self.tokens_per_sensor = 1
        else:
            raise ValueError(f"unknown extero_encoder {self.extero_encoder!r}.")
        self.extero_type = nn.Parameter(torch.zeros(1, 1, d_model))
        # Per-SENSOR id, so a camera's tokens are distinguishable from the other camera's
        # (under the CNN it is shared by all of that camera's grid slots -- the slot
        # embedding separates those). Only created for T > 1: at T=1 the parameter set stays
        # exactly as it was, so existing height-scan checkpoints load unchanged.
        self.extero_id = (
            nn.Parameter(torch.zeros(1, extero_sensor_count, d_model))
            if extero_sensor_count > 1 else None
        )

        if cmd_dim > 0:
            self.cmd_in = _token_mlp(cmd_dim, d_model)
            # Learned type embedding so a command token is distinguishable from the
            # future state token sharing its timestep / positional encoding.
            self.cmd_type = nn.Parameter(torch.zeros(1, 1, d_model))
            cmd_steps = list(range(0, future_len, cmd_stride))
            self.register_buffer("cmd_steps", torch.tensor(cmd_steps, dtype=torch.long))
        else:
            self.cmd_in = None
            self.cmd_type = None
            self.register_buffer("cmd_steps", torch.zeros(0, dtype=torch.long))

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # Timeline of length N + F: history at 0..N-1, future at N..N+F-1, each cmd
        # token at N + (its future step). The extero tokens carry no TIMELINE position (the
        # reading is "now"); under the CNN they do carry a learned SPATIAL one, inside the
        # encoder.
        self.register_buffer("pos", _sinusoidal_positional_encoding(num_history_states + future_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, state_dim)

    @property
    def extero_seq_tokens(self) -> int:
        """Exteroception tokens in the SEQUENCE (sensors x tokens per sensor).

        Deliberately not called ``extero_sensor_count``: that is how many physical readings
        there are, and the deploy-side camera checks read it. The two were one number back
        when a reading was necessarily one token.
        """
        return int(self.extero_sensor_count) * int(self.tokens_per_sensor)

    @property
    def sequence_len(self) -> int:
        """Total token count the transformer sees."""
        return self.prefix_len + int(self.future_len)

    @property
    def prefix_len(self) -> int:
        """Conditioning tokens: history + exteroception + command.

        The leading, contiguous block of the sequence -- which is what lets ``prefix`` mode
        split it off without reordering anything.
        """
        return (int(self.num_history_states) + self.extero_seq_tokens
                + int(self.cmd_steps.numel()))

    # ------------------------------------------------------------------
    # Token assembly, shared by both attention modes.
    # ------------------------------------------------------------------
    def _embed_prefix(
        self,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None,
        history_null_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed the conditioning streams into ``(B, prefix_len, d_model)``.

        Carries NO flow-time embedding: under ``prefix`` attention the conditioning must not
        be a function of the noise level (that is the point of the mode), and under ``full``
        attention :meth:`forward` adds ``t_emb`` to the whole sequence afterwards, exactly as
        it always did.

        Args:
            history: Raw history (phase) conditioning, ``(B, N, history_dim)``.
            extero: Exteroception, ``(B, T, extero_sensor_dim)`` / ``(B, T*dim)`` / ``(B,)``.
            cmd_tokens: Strided command tokens, ``(B, num_cmd_tokens, cmd_dim)`` or ``None``.
            history_null_mask: Optional ``(B,)`` bool; where True the history content is
                replaced by the learned ``null_history`` embedding.

        Returns:
            The assembled conditioning tokens.

        Raises:
            ValueError: If the model was built with ``cmd_dim > 0`` and no ``cmd_tokens``.
        """
        b = history.shape[0]
        n = self.num_history_states

        # History tokens (content from history_in, or the learned null), positioned.
        h_content = self.history_in(history)  # (B, N, d_model)
        if history_null_mask is not None:
            nm = history_null_mask.view(b, 1, 1)
            h_content = torch.where(nm, self.null_history.expand(b, n, self.d_model), h_content)
        h = h_content + self.pos[:, :n, :] + self.hist_type  # (B, N, d_model)

        # Exteroception tokens. No TIMELINE position -- the reading is "now" -- so the only
        # identity they carry is the type, the per-sensor id and (CNN) the spatial slot.
        # Accepts (B, T*dim), (B, dim) or (B,); all reshape to (B, T, dim).
        #
        # Both encoders are put through ONE set of identity adds by giving the MLP a
        # length-1 slot axis. On that path the adds are (1,1,1,d) and (1,T,1,d) broadcasts
        # followed by a reshape back to (B, T, d), so it stays numerically identical to the
        # pre-CNN code and the trained baseline's outputs remain bit-comparable.
        ev = extero.reshape(b, self.extero_sensor_count, self.extero_sensor_dim)
        et = (self.extero_cnn(ev) if self.extero_cnn is not None
              else self.extero_in(ev).unsqueeze(2))        # (B, T, S, d_model); S=1 for mlp
        et = et + self.extero_type.unsqueeze(1)            # (1, 1, 1, d_model)
        if self.extero_id is not None:
            et = et + self.extero_id.unsqueeze(2)          # (1, T, 1, d) -> every slot of a cam
        # Camera-major, slot-minor -- the order ``extero_id``/``spatial`` were added in.
        ht = et.reshape(b, self.extero_seq_tokens, self.d_model)

        parts = [h, ht]
        if self.cmd_in is not None:
            if cmd_tokens is None:
                raise ValueError("Model built with cmd_dim > 0 requires cmd_tokens.")
            # Each cmd token aligns in time with its future state token (pos at N + step).
            c = self.cmd_in(cmd_tokens) + self.pos[:, n + self.cmd_steps, :] + self.cmd_type
            parts.append(c)
        return torch.cat(parts, dim=1)  # (B, N + T + Ncmd, d_model)

    def _embed_future(self, x_tau: torch.Tensor) -> torch.Tensor:
        """Embed the denoised state tokens, positioned at ``N .. N+F-1``.

        Args:
            x_tau: Future state tokens at flow-time tau, ``(B, F, state_dim)``.

        Returns:
            ``(B, F, d_model)``, without the flow-time embedding.
        """
        n, f = self.num_history_states, self.future_len
        return self.state_in(x_tau) + self.pos[:, n:n + f, :]

    def _time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """Flow-time embedding, ``(B, 1, d_model)``, broadcastable over a token axis."""
        return self.time_mlp(_timestep_embedding(tau, self.d_model)).unsqueeze(1)

    @staticmethod
    def _run_layer(layer: nn.Module, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """One ``nn.TransformerEncoderLayer``, with the key/value source given separately.

        Reproduces the stock POST-norm layer exactly -- ``norm_first=False`` is the default
        this model builds with -- differing only in that the keys and values may come from a
        longer sequence than the queries. That is ordinary cross-attention, so the fused
        in/out projections, the head split and the SDPA scaling all stay PyTorch's.

        Op order is load-bearing and there are THREE distinct dropout modules on the way
        through (``dropout1`` after attention, ``dropout`` inside the feedforward, ``dropout2``
        after it); at the shipped ``dropout=0.0`` a mix-up is invisible, which is why
        ``test_prefix_attention_matches_a_masked_full_pass`` exists.

        Args:
            layer: One layer of ``self.encoder``.
            x: Query rows, ``(B, Lq, d_model)`` -- also the residual stream.
            kv: Key/value rows, ``(B, Lkv, d_model)``.

        Returns:
            ``(B, Lq, d_model)``.
        """
        a = layer.self_attn(x, kv, kv, need_weights=False)[0]
        x = layer.norm1(x + layer.dropout1(a))
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        return layer.norm2(x + layer.dropout2(ff))

    # ------------------------------------------------------------------
    # ``prefix`` attention: encode the conditioning once, decode per step.
    # ------------------------------------------------------------------
    def encode_prefix(
        self,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Run the conditioning through the stack on its own; return the per-layer inputs.

        Under ``prefix`` attention the conditioning attends only to itself, so its
        representation at every layer is a function of the conditioning ALONE -- not of the
        future tokens and not of the flow time. It can therefore be computed once per re-plan
        and reused by every Euler step, which is the whole reason the mode exists.

        What the future needs at layer ``i`` is the conditioning's *input* to that layer (its
        keys and values there), so that is what is returned: ``L`` tensors, not ``L+1``. The
        conditioning's final output is never read -- nothing downstream consumes it.

        Returned as a tuple rather than stashed on ``self`` deliberately: ``_FmSamplerOnnx``
        is TorchScript-traced, and a cache held as module state would be constant-folded into
        the exported graph.

        Args:
            history: Raw history (phase) conditioning, ``(B, N, history_dim)``.
            extero: Exteroception, ``(B, T, extero_sensor_dim)`` / ``(B, T*dim)`` / ``(B,)``.
            cmd_tokens: Strided command tokens, or ``None``.
            history_null_mask: Optional ``(B,)`` bool null-history selector.

        Returns:
            One ``(B, prefix_len, d_model)`` tensor per layer, in layer order.

        Raises:
            RuntimeError: If the model was built with ``attention_mode="full"``.
        """
        if self.attention_mode != "prefix":
            raise RuntimeError(
                "encode_prefix needs attention_mode='prefix'; a 'full'-attention model's "
                "conditioning representation depends on the future tokens and the flow time, "
                "so there is nothing stable to cache."
            )
        x = self._embed_prefix(history, extero, cmd_tokens, history_null_mask)
        cache = []
        for layer in self.encoder.layers:
            cache.append(x)
            x = self._run_layer(layer, x, x)
        return tuple(cache)

    def decode_future(
        self, x_tau: torch.Tensor, tau: torch.Tensor,
        prefix_cache: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Predict the flow field for the future window against a cached conditioning prefix.

        The future tokens attend to the conditioning AND to each other, so at layer ``i`` the
        keys and values are ``cat(prefix_cache[i], future)``. No attention mask is needed:
        the conditioning is simply not in the query set, which is the same thing a prefix mask
        expresses and considerably cheaper than expressing it.

        The flow-time embedding is added HERE, to the future tokens only -- the property that
        makes ``prefix_cache`` reusable across Euler steps.

        Args:
            x_tau: Future state tokens at flow-time ``tau``, ``(B, F, state_dim)``.
            tau: Flow time in ``[0, 1]``, ``(B,)``.
            prefix_cache: Per-layer conditioning inputs from :meth:`encode_prefix`.

        Returns:
            Predicted velocity field, ``(B, F, state_dim)``.

        Raises:
            ValueError: If the cache does not have one entry per layer.
        """
        if len(prefix_cache) != len(self.encoder.layers):
            raise ValueError(
                f"prefix_cache has {len(prefix_cache)} entries but the encoder has "
                f"{len(self.encoder.layers)} layers; it must carry one per layer."
            )
        x = self._embed_future(x_tau) + self._time_embedding(tau)
        for layer, prefix in zip(self.encoder.layers, prefix_cache):
            x = self._run_layer(layer, x, torch.cat([prefix, x], dim=1))
        return self.output_head(x)

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow vector field for the future state window.

        Args:
            x_tau: Future state tokens at flow-time ``tau`` (the only noised tokens),
                shape ``(B, F, state_dim)``.
            tau: Flow time in ``[0, 1]``, shape ``(B,)``.
            history: Clean history (phase) conditioning, shape ``(B, N, history_dim)``.
            extero: Exteroception conditioning, shape ``(B, T, extero_sensor_dim)``,
                ``(B, T*extero_sensor_dim)`` or ``(B,)``.
            cmd_tokens: Strided future command tokens (normalized), shape
                ``(B, num_cmd_tokens, cmd_dim)``. Required when built with
                ``cmd_dim > 0``; ignored otherwise.
            history_null_mask: Optional ``(B,)`` bool; where True, the history content
                is replaced by the learned ``null_history`` embedding (phase-agnostic /
                classifier-free-guidance branch).

        Returns:
            Predicted velocity field over the F future tokens, shape ``(B, F, state_dim)``.
        """
        f = self.future_len
        prefix = self._embed_prefix(history, extero, cmd_tokens, history_null_mask)

        if self.attention_mode == "prefix":
            # One implementation, not two: the single-shot path (training, and any caller that
            # does not hold a cache) is the cached path with a cache of one use. So there is no
            # second arithmetic to keep in step with the Euler loop's.
            cache = []
            x = prefix
            for layer in self.encoder.layers:
                cache.append(x)
                x = self._run_layer(layer, x, x)
            return self.decode_future(x_tau, tau, cache)

        # ``full``: every token attends to every other, and the flow-time embedding lands on
        # all of them. Kept byte-identical -- this is what every checkpoint trained before
        # attention_mode existed was trained under, and the mask adds no parameters, so the
        # config is the ONLY thing that distinguishes such a checkpoint from a prefix one.
        tokens = torch.cat([prefix, self._embed_future(x_tau)], dim=1)  # (B, S, d_model)
        tokens = tokens + self._time_embedding(tau)
        tokens = self.encoder(tokens)  # full self-attention over all tokens
        return self.output_head(tokens[:, -f:, :])  # trailing F future-token outputs
