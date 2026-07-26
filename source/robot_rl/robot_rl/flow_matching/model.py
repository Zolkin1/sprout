# Copyright 2026 Zachary Olkin. All rights reserved.

"""Single-transformer flow-matching model: robot-state-conditioned future prediction.

One transformer (encoder-only, full self-attention) operates on a single token
sequence assembled from four streams::

    [ history(N) | extero(T) | cmd(Ncmd) | future(F) ]

- **history**: ``N`` real-robot-state steps (the recent past, in the xyz+yaw ground
  frame at the anchor = step ``N-1``). Pure conditioning, never noised. Its only job
  is to supply *phase*. Embedded by its own ``history_in`` MLP + a learned
  ``hist_type``. A learned ``null_history`` embedding replaces the history content
  when ``history_null_mask`` is set (classifier-free-guidance / phase-agnostic mode).
- **extero**: ``T`` exteroception tokens of ``height_dim`` values each, through one
  shared ``height_in`` + ``height_type`` (+ a per-token ``extero_id`` when T > 1). Pure
  conditioning. T=1 is the pelvis height scan; T=2 is one token per depth camera.
- **cmd**: the strided future velocity-command profile, via ``cmd_in`` + ``cmd_type``,
  each token positioned at its future timestep. Pure conditioning.
- **future**: the ``F`` state tokens the flow denoises -- ALL future (no past echo).

State/history/cmd tokens carry a sinusoidal positional encoding over a ``N + F``
timeline (history at ``0..N-1``, future at ``N..N+F-1``, each cmd at ``N + its step``);
the extero tokens are global (no positional encoding). Flow-time ``tau`` is a sinusoidal
embedding through a small MLP, added to every token. The head reads only the trailing
``F`` future-token outputs and predicts the flow vector field. There is NO inpainting /
known-mask: the conditioning is separate tokens, not held-clean slots of the denoised
window.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


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
        height_dim: Width of ONE exteroception token (176 = height-scan rays,
            780 = one depth image).
        extero_tokens: Number of exteroception tokens (1 = height scan, 2 = both cameras).
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
        height_dim: int = 1,
        extero_tokens: int = 1,
        cmd_dim: int = 0,
        cmd_stride: int = 5,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.future_len = future_len
        self.num_history_states = num_history_states
        self.history_dim = history_dim if history_dim is not None else state_dim
        self.height_dim = height_dim
        self.extero_tokens = extero_tokens
        self.cmd_dim = cmd_dim
        self.cmd_stride = cmd_stride

        # Future (denoised) state tokens.
        self.state_in = _token_mlp(state_dim, d_model)
        # History (phase) conditioning tokens + a learned null-history embedding.
        self.history_in = _token_mlp(self.history_dim, d_model)
        self.hist_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))
        # Exteroception conditioning: ``extero_tokens`` tokens of ``height_dim`` values
        # each, through ONE shared MLP.  The height scan is the T=1 case (176 rays -> one
        # token); depth cameras are T=2 (one 780-value image per camera), so attention can
        # weight near against far instead of squeezing both through a single token.
        self.height_in = _token_mlp(height_dim, d_model)
        self.height_type = nn.Parameter(torch.zeros(1, 1, d_model))
        # Per-token id, so tokens sharing the type embedding are still distinguishable.
        # Only created for T > 1: at T=1 the parameter set stays exactly as it was, so
        # existing height-scan checkpoints load unchanged.
        self.extero_id = (
            nn.Parameter(torch.zeros(1, extero_tokens, d_model)) if extero_tokens > 1 else None
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
        # token at N + (its future step). The extero tokens are global (no position).
        self.register_buffer("pos", _sinusoidal_positional_encoding(num_history_states + future_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_head = nn.Linear(d_model, state_dim)

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        history: torch.Tensor,
        height: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow vector field for the future state window.

        Args:
            x_tau: Future state tokens at flow-time ``tau`` (the only noised tokens),
                shape ``(B, F, state_dim)``.
            tau: Flow time in ``[0, 1]``, shape ``(B,)``.
            history: Clean history (phase) conditioning, shape ``(B, N, history_dim)``.
            height: Exteroception conditioning, shape ``(B, T, height_dim)``,
                ``(B, T*height_dim)`` or ``(B,)``.
            cmd_tokens: Strided future command tokens (normalized), shape
                ``(B, num_cmd_tokens, cmd_dim)``. Required when built with
                ``cmd_dim > 0``; ignored otherwise.
            history_null_mask: Optional ``(B,)`` bool; where True, the history content
                is replaced by the learned ``null_history`` embedding (phase-agnostic /
                classifier-free-guidance branch).

        Returns:
            Predicted velocity field over the F future tokens, shape ``(B, F, state_dim)``.
        """
        b, f = x_tau.shape[0], self.future_len
        n = self.num_history_states

        # History tokens (content from history_in, or the learned null), positioned.
        h_content = self.history_in(history)  # (B, N, d_model)
        if history_null_mask is not None:
            nm = history_null_mask.view(b, 1, 1)
            h_content = torch.where(nm, self.null_history.expand(b, n, self.d_model), h_content)
        h = h_content + self.pos[:, :n, :] + self.hist_type  # (B, N, d_model)

        # Exteroception tokens (global; no positional encoding). Accepts (B, T*dim),
        # (B, dim) or (B,) -- all reshape to (B, T, dim).
        ht = self.height_in(height.view(b, self.extero_tokens, self.height_dim))
        ht = ht + self.height_type                                    # (B, T, d_model)
        if self.extero_id is not None:
            ht = ht + self.extero_id

        # Future (denoised) state tokens, positioned at N..N+F-1.
        s = self.state_in(x_tau) + self.pos[:, n:n + f, :]  # (B, F, d_model)

        parts = [h, ht]
        if self.cmd_in is not None:
            if cmd_tokens is None:
                raise ValueError("Model built with cmd_dim > 0 requires cmd_tokens.")
            # Each cmd token aligns in time with its future state token (pos at N + step).
            c = self.cmd_in(cmd_tokens) + self.pos[:, n + self.cmd_steps, :] + self.cmd_type
            parts.append(c)
        parts.append(s)
        tokens = torch.cat(parts, dim=1)  # (B, N + T + Ncmd + F, d_model)

        t_emb = self.time_mlp(_timestep_embedding(tau, self.d_model)).unsqueeze(1)  # (B,1,d_model)
        tokens = tokens + t_emb
        tokens = self.encoder(tokens)  # full self-attention over all tokens
        return self.output_head(tokens[:, -f:, :])  # trailing F future-token outputs
