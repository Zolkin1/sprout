# Copyright 2026 Zachary Olkin. All rights reserved.

"""PARC-style single-transformer flow-matching model for future-state prediction.

One transformer (encoder-only, full self-attention -- no separate history encoder)
operates on a single token sequence: ``ceil(F / cmd_stride)`` command tokens followed
by ``F`` state tokens. States are tokenized by a shared MLP (``state_in``); the strided
future-command profile is tokenized by a separate MLP (``cmd_in``). State tokens get a
sinusoidal positional encoding over the F window positions; each command token gets the
positional encoding at *its* source timestep (so it aligns in time with the matching
state token) plus a learned command-type embedding. Flow-time ``tau`` is a sinusoidal
embedding through a small MLP, added to every token. The head reads only the F
state-token outputs and predicts the flow vector field.

The history-conditioning is *inpainting* handled entirely in ``flow.py``: the first
``num_history_states`` state tokens are held at their (observed) clean values during
integration. The model is mask-agnostic -- it infers conditioning from the fact that
those tokens do not move with ``tau``.
TODO(is-known-flag): if the conditioned vs unconditional (history-dropout) modes prove
hard to separate, append a per-token is-known scalar to ``state_in``'s input.
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
    """Single-transformer conditional flow-matching model (PARC-style inpainting).

    Args:
        state_dim: Per-step state width ``D`` (the flow operates in this space).
        future_len: Window length ``F`` (number of state tokens).
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
        self.cmd_dim = cmd_dim
        self.cmd_stride = cmd_stride

        self.state_in = _token_mlp(state_dim, d_model)
        if cmd_dim > 0:
            self.cmd_in = _token_mlp(cmd_dim, d_model)
            # Learned type embedding so a command token is distinguishable from the
            # state token sharing its timestep / positional encoding.
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
        self.register_buffer("pos", _sinusoidal_positional_encoding(future_len, d_model))

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
        cmd_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow vector field for the state window.

        Args:
            x_tau: State tokens at flow-time ``tau`` (known tokens already sit at
                their clean values), shape ``(B, F, state_dim)``.
            tau: Flow time in ``[0, 1]``, shape ``(B,)``.
            cmd_tokens: Strided command tokens (normalized), shape
                ``(B, num_cmd_tokens, cmd_dim)``. Required when built with
                ``cmd_dim > 0``; ignored otherwise.

        Returns:
            Predicted velocity field over the state tokens, shape ``(B, F, state_dim)``.
        """
        s = self.state_in(x_tau) + self.pos  # (B, F, d_model)
        tokens = s
        n_cmd = 0
        if self.cmd_in is not None:
            if cmd_tokens is None:
                raise ValueError("Model built with cmd_dim > 0 requires cmd_tokens.")
            c = self.cmd_in(cmd_tokens) + self.pos[:, self.cmd_steps, :] + self.cmd_type
            n_cmd = c.shape[1]
            tokens = torch.cat([c, s], dim=1)  # (B, n_cmd + F, d_model)
        t_emb = self.time_mlp(_timestep_embedding(tau, self.d_model)).unsqueeze(1)  # (B,1,d_model)
        tokens = tokens + t_emb
        tokens = self.encoder(tokens)  # full self-attention over command + state tokens
        return self.output_head(tokens[:, n_cmd:, :])  # state-token outputs only
