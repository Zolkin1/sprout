# Copyright 2026 Zachary Olkin. All rights reserved.

"""Flow-matching encoder-decoder transformer for future robot-state prediction.

Encoder self-attends over the history tokens to produce memory. The decoder
self-attends (FULL, non-causal) over the noisy future tokens at flow-time ``tau``
and cross-attends to the memory, predicting the flow vector field for every
future step in one pass. Flow-time conditioning is an additive sinusoidal
embedding, mirroring the positional-encoding pattern used elsewhere in the repo.
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


class FlowMatchingTransformer(nn.Module):
    """Conditional flow-matching transformer (history encoder, future decoder).

    Args:
        history_in_dim: Per-step history token width (state + command + perception).
        state_dim: Per-step future state width (the flow operates in this space).
        history_len: Number of history steps ``H``.
        future_len: Number of future steps ``F``.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Encoder depth.
        num_decoder_layers: Decoder depth.
        dim_feedforward: Feedforward dimension in each transformer layer.
        dropout: Dropout rate.
        cmd_dim: Per-step future-command width injected into the decoder (0 disables
            future-command conditioning).
    """

    def __init__(
        self,
        history_in_dim: int,
        state_dim: int,
        history_len: int,
        future_len: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        cmd_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.history_len = history_len
        self.future_len = future_len
        self.cmd_dim = cmd_dim

        self.history_proj = nn.Linear(history_in_dim, d_model)
        self.future_proj = nn.Linear(state_dim, d_model)
        # Per-future-step command embedding, added to the decoder token like the
        # positional / time embeddings (only when conditioning on a command horizon).
        self.cmd_proj = nn.Linear(cmd_dim, d_model) if cmd_dim > 0 else None
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )

        self.register_buffer("enc_pos", _sinusoidal_positional_encoding(history_len, d_model))
        self.register_buffer("dec_pos", _sinusoidal_positional_encoding(future_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.output_proj = nn.Linear(d_model, state_dim)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Encode the history window into memory.

        Args:
            history: History tokens, shape ``(B, H, history_in_dim)``.

        Returns:
            Encoder memory, shape ``(B, H, d_model)``.
        """
        h = self.history_proj(history) + self.enc_pos
        return self.encoder(h)

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        memory: torch.Tensor,
        future_cmd: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the flow vector field for the future trajectory.

        Args:
            x_tau: Noisy future states at flow-time ``tau``, shape ``(B, F, state_dim)``.
            tau: Flow time in ``[0, 1]``, shape ``(B,)``.
            memory: Encoder memory, shape ``(B, H, d_model)``.
            future_cmd: Per-future-step command (normalized), shape ``(B, F, cmd_dim)``.
                Required when the model was built with ``cmd_dim > 0``; ignored otherwise.

        Returns:
            Predicted velocity field, shape ``(B, F, state_dim)``.
        """
        h = self.future_proj(x_tau) + self.dec_pos
        if self.cmd_proj is not None:
            if future_cmd is None:
                raise ValueError("Model built with cmd_dim > 0 requires future_cmd.")
            h = h + self.cmd_proj(future_cmd)
        t_emb = self.time_mlp(_timestep_embedding(tau, self.d_model))  # (B, d_model)
        h = h + t_emb.unsqueeze(1)
        # FULL self-attention over future tokens (no tgt_mask) + cross-attention to memory.
        h = self.decoder(tgt=h, memory=memory)
        return self.output_proj(h)
