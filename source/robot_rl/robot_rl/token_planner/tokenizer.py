# Copyright 2026 Zachary Olkin. All rights reserved.

"""The trajectory tokenizer: a chunk-local autoencoder with an FSQ bottleneck.

It maps one plan-length chunk of ``H`` reference-window nodes to ``N = H / D`` discrete
tokens and back. Everything here operates in NORMALIZED node space -- the same
:class:`~robot_rl.flow_matching.normalizer.Normalizer` the flow-matching generator's state
tokens go through -- so the tokenizer never sees a metre and the normalizer travels beside
it in the checkpoint rather than inside it.

.. admonition:: Chunk-local is a correctness property, not a simplification

    The encoder and decoder see exactly one chunk and hold no cross-chunk state. That makes
    "tokenize a training chunk" and "decode a sequence the planner generated" the *same*
    operation, byte for byte. A tokenizer with context would be trained on real neighbours
    and deployed on generated ones, so its reconstruction error at deploy would be a
    quantity nothing in training ever measured.

Shapes, once:

* nodes -- ``(B, H, node_dim)``, normalized, time-major (the ``(B, C, T)`` conv layout is
  internal and never crosses this module's boundary);
* raw latents -- ``(B, N, L_DIM)``, unbounded (:class:`~.fsq.FSQ` owns the ``tanh``);
* token ids -- ``(B, N)``, ``int64`` in ``[0, V)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import TokenizerConfig
from .fsq import FSQ


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL axis of a ``(B, C, T)`` tensor.

    ``nn.LayerNorm`` normalizes trailing dimensions, which on a conv layout would normalize
    over TIME -- pooling a chunk's own temporal statistics into every activation and making
    a block's output depend on the rest of the chunk. Normalizing over channels at each
    timestep independently is what the spec's "LayerNorm over channels" means and what keeps
    the stack local in time.

    Args:
        channels: Channel count ``C``.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``(B, C, T)`` over ``C``."""
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class ResidualConvBlock(nn.Module):
    """Pre-norm residual block of two 1-D convolutions along time.

    ``x + conv2(gelu(conv1(norm(x))))``. Pre-norm rather than post-norm so a deep stack's
    residual path stays an identity at initialisation, which is what lets ``num_blocks_*``
    be raised without re-tuning the learning rate.

    Args:
        channels: Channel count ``C``, unchanged by the block.
        kernel_size: Odd kernel size along time.
        pad_mode: ``nn.Conv1d`` padding mode; see :attr:`TokenizerConfig.pad_mode`.
    """

    def __init__(self, channels: int, kernel_size: int, pad_mode: str):
        super().__init__()
        pad = int(kernel_size) // 2
        self.norm = ChannelLayerNorm(channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, padding_mode=pad_mode
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=pad, padding_mode=pad_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``(B, C, T)``."""
        return x + self.conv2(F.gelu(self.conv1(self.norm(x))))


def _stride_schedule(downsample: int) -> list[int]:
    """Factor ``D`` into per-layer strides.

    A power of two becomes ``log2(D)`` stride-2 layers, so each halving gets its own
    non-linearity; anything else becomes a single strided layer, because an arbitrary
    factorisation would make the architecture depend on ``D``'s prime factors in a way
    nothing else in the stack does.

    Args:
        downsample: Nodes per token ``D`` (``>= 1``).

    Returns:
        Per-layer strides, in order; empty when ``D == 1``.
    """
    d = int(downsample)
    if d == 1:
        return []
    if d & (d - 1) == 0:
        return [2] * int(d.bit_length() - 1)
    return [d]


class TrajectoryEncoder(nn.Module):
    """``(B, H, node_dim) -> (B, N, L_DIM)`` raw latents.

    Args:
        cfg: The tokenizer config; :meth:`TokenizerConfig.validate` must already have run.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        c, k, pm = cfg.channels, cfg.kernel_size, cfg.pad_mode
        self.cfg = cfg
        self.node_in = nn.Linear(cfg.node_dim, c)
        self.high = nn.ModuleList(
            ResidualConvBlock(c, k, pm) for _ in range(cfg.num_blocks_high)
        )
        # Strided convolutions carry the downsample. They keep the same padding mode, so the
        # chunk boundary is treated identically at every rate.
        self.down = nn.ModuleList(
            nn.Conv1d(c, c, k, stride=s, padding=k // 2, padding_mode=pm)
            for s in _stride_schedule(cfg.downsample)
        )
        self.low = nn.ModuleList(
            ResidualConvBlock(c, k, pm) for _ in range(cfg.num_blocks_low)
        )
        self.norm_out = ChannelLayerNorm(c)
        # NO tanh here: FSQ owns the bound. See fsq.py's module docstring for what a second
        # squashing costs (two of eight codes unreachable per axis, silently).
        self.latent_out = nn.Linear(c, cfg.latent_dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """Encode a chunk.

        Args:
            nodes: Normalized nodes ``(B, H, node_dim)``.

        Returns:
            Raw latents ``(B, N, L_DIM)``.

        Raises:
            ValueError: On a chunk of the wrong length or width -- both are silent
                otherwise, since a conv stack happily consumes any length.
        """
        if nodes.shape[-2:] != (self.cfg.chunk_len, self.cfg.node_dim):
            raise ValueError(
                f"expected nodes of shape (B, {self.cfg.chunk_len}, {self.cfg.node_dim}), "
                f"got {tuple(nodes.shape)}."
            )
        x = self.node_in(nodes).transpose(1, 2)          # (B, C, H)
        for blk in self.high:
            x = blk(x)
        for conv in self.down:
            x = F.gelu(conv(x))
        for blk in self.low:
            x = blk(x)
        return self.latent_out(self.norm_out(x).transpose(1, 2))   # (B, N, L_DIM)


class TrajectoryDecoder(nn.Module):
    """``(B, N, L_DIM) -> (B, H, node_dim)``; the mirror of :class:`TrajectoryEncoder`.

    Args:
        cfg: The tokenizer config.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        c, k, pm = cfg.channels, cfg.kernel_size, cfg.pad_mode
        self.cfg = cfg
        self.latent_in = nn.Linear(cfg.latent_dim, c)
        self.low = nn.ModuleList(
            ResidualConvBlock(c, k, pm) for _ in range(cfg.num_blocks_low)
        )
        strides = list(reversed(_stride_schedule(cfg.downsample)))
        self.up_strides = strides
        if cfg.upsample_mode == "transpose":
            self.up = nn.ModuleList(
                nn.ConvTranspose1d(
                    c, c, kernel_size=2 * s, stride=s, padding=s // 2
                )
                for s in strides
            )
        else:
            # nearest-upsample + same-padding conv: no stride-aligned checkerboard, which on
            # a trajectory would read as jitter at exactly the token period -- the one
            # artefact the tracker is most likely to notice.
            self.up = nn.ModuleList(
                nn.Conv1d(c, c, k, padding=k // 2, padding_mode=pm) for _ in strides
            )
        self.high = nn.ModuleList(
            ResidualConvBlock(c, k, pm) for _ in range(cfg.num_blocks_high)
        )
        self.norm_out = ChannelLayerNorm(c)
        self.node_out = nn.Linear(c, cfg.node_dim)   # no output activation

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a token-rate latent sequence back to nodes.

        Args:
            latent: Dequantized latents ``(B, N, L_DIM)``.

        Returns:
            Normalized nodes ``(B, H, node_dim)``.

        Raises:
            ValueError: On the wrong token count or latent width.
        """
        if latent.shape[-2:] != (self.cfg.num_tokens, self.cfg.latent_dim):
            raise ValueError(
                f"expected latents of shape (B, {self.cfg.num_tokens}, "
                f"{self.cfg.latent_dim}), got {tuple(latent.shape)}."
            )
        x = self.latent_in(latent).transpose(1, 2)       # (B, C, N)
        for blk in self.low:
            x = blk(x)
        for stride, layer in zip(self.up_strides, self.up):
            if self.cfg.upsample_mode == "transpose":
                x = F.gelu(layer(x))
            else:
                x = F.gelu(layer(F.interpolate(x, scale_factor=float(stride), mode="nearest")))
        for blk in self.high:
            x = blk(x)
        out = self.node_out(self.norm_out(x).transpose(1, 2))
        # A transposed convolution's output length is only equal to stride*in for particular
        # (kernel, padding) triples; assert rather than trusting the arithmetic.
        if out.shape[1] != self.cfg.chunk_len:
            raise RuntimeError(
                f"decoder produced {out.shape[1]} nodes, expected {self.cfg.chunk_len}; "
                f"upsample_mode={self.cfg.upsample_mode!r} strides={self.up_strides}."
            )
        return out


class TrajectoryTokenizer(nn.Module):
    """Encoder + FSQ + decoder, addressable separately so the planner can freeze it.

    Args:
        cfg: The tokenizer config. Validated here, so a bad ``(H, D)`` pair fails at
            construction rather than in the middle of the first forward pass.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = TrajectoryEncoder(cfg)
        self.quantizer = FSQ(cfg.fsq_levels)
        self.decoder = TrajectoryDecoder(cfg)

    @property
    def vocab_size(self) -> int:
        """Token vocabulary ``V``."""
        return self.quantizer.vocab_size

    @property
    def num_tokens(self) -> int:
        """Tokens per chunk ``N``."""
        return self.cfg.num_tokens

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct a chunk and report the tokens it went through.

        Args:
            nodes: Normalized nodes ``(B, H, node_dim)``.

        Returns:
            ``(nodes_hat, ids)``: the reconstruction ``(B, H, node_dim)``, carrying the
            straight-through gradient, and the token ids ``(B, N)``.
        """
        latent, ids = self.quantizer(self.encoder(nodes))
        return self.decoder(latent), ids

    @torch.no_grad()
    def encode_to_ids(self, nodes: torch.Tensor) -> torch.Tensor:
        """Token ids for a chunk -- the dataset-export path.

        Args:
            nodes: Normalized nodes ``(B, H, node_dim)``.

        Returns:
            Token ids ``(B, N)``, ``int64``.
        """
        return self.quantizer(self.encoder(nodes))[1]

    @torch.no_grad()
    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Reconstruct a chunk from token ids -- the planner's decode path.

        Args:
            ids: Token ids ``(B, N)`` in ``[0, V)``.

        Returns:
            Normalized nodes ``(B, H, node_dim)``.

        Raises:
            ValueError: On the wrong token count, or an id outside the vocabulary.
        """
        if ids.shape[-1] != self.cfg.num_tokens:
            raise ValueError(
                f"expected {self.cfg.num_tokens} token(s) per chunk, got {ids.shape[-1]}."
            )
        return self.decoder(self.quantizer.id_to_latent(ids).to(
            next(self.decoder.parameters()).dtype
        ))
