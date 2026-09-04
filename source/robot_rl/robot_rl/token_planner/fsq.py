# Copyright 2026 Zachary Olkin. All rights reserved.

"""Finite Scalar Quantization -- the tokenizer's bottleneck.

FSQ (Mentzer et al., 2023) replaces a learned codebook with a fixed lattice: each of
``L_DIM`` latent channels is squashed into ``[-1, 1]`` and rounded to one of ``levels[i]``
evenly spaced values, and the token id is the mixed-radix index of the resulting integer
tuple. The vocabulary is ``prod(levels)`` and it is *implicit* -- there are no codebook
parameters, no commitment loss, no EMA updates and no dead-code re-initialisation, which is
the whole reason to prefer it here: the failure mode a VQ-VAE has (most of the codebook
unused, and the used part drifting during training) cannot occur, so codebook utilisation
becomes a property of the ENCODER's output distribution rather than of the optimiser's luck.

.. admonition:: The bound lives HERE, not in the encoder

    The squashing ``tanh`` is part of this module and the encoder must emit RAW
    pre-activations. Applying ``tanh`` in both places composes to ``tanh(tanh(z))``, whose
    range is ``(-0.7616, 0.7616)`` rather than ``(-1, 1)``. That is not a mild rescaling: on
    an 8-level axis the reachable integers become ``1..6``, so **two of the eight codes are
    unreachable by construction** and the ``[8, 8, 8, 5, 5]`` vocabulary silently collapses
    from 12800 to 5400 (measured; the 5-level axes happen to survive, since
    ``2 * (1 - tanh 1) < 1``, so it is the three 8-level axes that lose their ends). Nothing
    about the shapes changes, so the only symptom would be a utilisation number that looks
    like a training problem. :func:`quantize` therefore owns the bound and :class:`FSQ`
    exposes no option to move it.

Grid convention, stated once because three helpers have to agree on it:

* ``latent`` -- the dequantized value the decoder consumes, in ``[-1, 1]``.
* ``codes``  -- the per-dim integer in ``[0, levels[i] - 1]``.
* ``id``     -- ``sum_i codes[i] * basis[i]`` with ``basis = cumprod([1, *levels[:-1]])``,
  i.e. **dim 0 is the fastest-varying digit**. Token ids are therefore contiguous in
  ``[0, V)`` and every id round-trips.

The straight-through estimator passes the decoder's gradient to ``tanh(z)`` unchanged; the
rounding itself contributes nothing, which is exactly the intent.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def codes_to_id(codes: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Mixed-radix pack of per-dim integer codes into token ids.

    Args:
        codes: Integer codes ``(..., L_DIM)``, each in ``[0, levels[i] - 1]``.
        levels: Per-dim level counts ``(L_DIM,)``.

    Returns:
        Token ids ``(...)``, dtype ``int64``, in ``[0, prod(levels))``.
    """
    basis = torch.cumprod(
        torch.cat([torch.ones(1, dtype=torch.long, device=levels.device), levels[:-1]]), dim=0
    )
    return (codes.to(torch.long) * basis).sum(dim=-1)


def id_to_codes(ids: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Unpack token ids into per-dim integer codes -- the inverse of :func:`codes_to_id`.

    Args:
        ids: Token ids ``(...)``, in ``[0, prod(levels))``.
        levels: Per-dim level counts ``(L_DIM,)``.

    Returns:
        Integer codes ``(..., L_DIM)``, dtype ``int64``.
    """
    basis = torch.cumprod(
        torch.cat([torch.ones(1, dtype=torch.long, device=levels.device), levels[:-1]]), dim=0
    )
    return (ids.to(torch.long).unsqueeze(-1) // basis) % levels


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    """``round(x)`` on the forward pass, identity on the backward pass."""
    return x + (torch.round(x) - x).detach()


class FSQ(nn.Module):
    """The finite scalar quantizer. Parameter-free; holds only its level table.

    Args:
        levels: Per-dim level counts, e.g. ``(8, 8, 8, 5, 5)`` for a 12800-token vocabulary.

    Raises:
        ValueError: If ``levels`` is empty or any entry is below 2 (a 1-level axis carries
            no information and would make the vocabulary a lie).
    """

    def __init__(self, levels: tuple[int, ...] | list[int]):
        super().__init__()
        lv = [int(v) for v in levels]
        if not lv:
            raise ValueError("FSQ needs at least one level entry.")
        bad = [v for v in lv if v < 2]
        if bad:
            raise ValueError(f"FSQ levels must all be >= 2; got {bad} in {lv}.")
        self.levels_list: tuple[int, ...] = tuple(lv)
        self.latent_dim: int = len(lv)
        self.vocab_size: int = math.prod(lv)
        self.register_buffer("levels", torch.tensor(lv, dtype=torch.long), persistent=False)

    def extra_repr(self) -> str:
        return f"levels={list(self.levels_list)}, vocab_size={self.vocab_size}"

    # ------------------------------------------------------------------
    # The lattice, in one place.
    # ------------------------------------------------------------------
    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bound, round and dequantize raw encoder outputs.

        Args:
            z: RAW encoder pre-activations ``(..., L_DIM)``. Do not pre-squash these; see
                the module docstring for what a second ``tanh`` costs.

        Returns:
            ``(latent, codes)`` where ``latent`` is the dequantized value in ``[-1, 1]``
            carrying a straight-through gradient to ``tanh(z)``, and ``codes`` are the
            detached integer codes ``(..., L_DIM)``.

        Raises:
            ValueError: If the trailing dimension is not ``L_DIM``.
        """
        if z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"FSQ expected a trailing dim of {self.latent_dim}, got {z.shape[-1]}."
            )
        # FLOAT32, whatever `z` arrives as. The lattice is a fixed set of constants, and a
        # constant that changes with the caller's autocast dtype is not a constant: under
        # bf16 autocast the training path's latent differed from `id_to_latent`'s by 3.3e-3
        # and 5.2% of the emitted TOKENS differed from the fp32 answer -- so the decoder was
        # trained against one lattice and served another, and the planner's targets depended
        # on whether autocast happened to be on. Nothing had the wrong shape and the reported
        # metrics (computed outside autocast) moved by 0.04%, so no number in a run could
        # have shown it.
        #
        # Returning fp32 costs nothing: under autocast the next matmul re-casts anyway.
        z32 = z.float()
        lv = self.levels.to(z.device)
        bounded = torch.tanh(z32)                                 # (-1, 1)
        half = (lv.float() - 1.0) * 0.5
        codes_f = _round_ste(half * (bounded + 1.0))              # [0, l-1], STE
        codes_f = torch.clamp(codes_f, torch.zeros_like(half), 2.0 * half)
        latent = codes_f / half - 1.0                             # back to [-1, 1]
        return latent, torch.round(codes_f).detach().to(torch.long)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`quantize`, but returning ``(latent, ids)`` -- the training-loop shape.

        Args:
            z: Raw encoder pre-activations ``(..., L_DIM)``.

        Returns:
            ``(latent, ids)``: the dequantized latent and the packed token ids ``(...)``.
        """
        latent, codes = self.quantize(z)
        return latent, codes_to_id(codes, self.levels)

    def id_to_latent(self, ids: torch.Tensor) -> torch.Tensor:
        """Dequantized latent for token ids -- what the decoder needs at inference.

        Args:
            ids: Token ids ``(...)`` in ``[0, vocab_size)``.

        Returns:
            ``(..., L_DIM)`` latents in ``[-1, 1]``, float32.

        Raises:
            ValueError: If any id is outside the vocabulary. Silently wrapping would decode
                a plausible trajectory for a token the model never emitted.
        """
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size):
            raise ValueError(
                f"token id out of range [0, {self.vocab_size}): "
                f"min={int(ids.min())} max={int(ids.max())}."
            )
        # fp32, matching :meth:`quantize` exactly -- see the dtype note there. These two are
        # the same lattice written twice and must agree bit for bit, which is what
        # ``test_the_lattice_does_not_depend_on_the_input_dtype`` requires.
        codes = id_to_codes(ids, self.levels)
        half = (self.levels.to(torch.float32) - 1.0) * 0.5
        return codes.to(torch.float32) / half - 1.0
