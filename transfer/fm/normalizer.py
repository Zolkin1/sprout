# Copyright 2026 Zachary Olkin. All rights reserved.

"""Per-dimension standardization with a selectable identity mask.

Dimensions whose ``standardize_mask`` entry is ``False`` are passed through
unchanged (mean 0, std 1). This is used to leave the 6D orientation block raw,
since standardizing it would break its near-unit-scale / Gram-Schmidt-able
structure.
"""

from __future__ import annotations

import torch


class Normalizer:
    """Affine per-dimension standardizer ``(x - mean) / std`` with an identity mask."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, standardize_mask: torch.Tensor):
        """Store the standardization statistics.

        Args:
            mean: Per-dimension mean, shape ``(D,)`` (0 where not standardized).
            std: Per-dimension std, shape ``(D,)`` (1 where not standardized).
            standardize_mask: Boolean mask, shape ``(D,)``; ``False`` => identity.
        """
        self.mean = mean
        self.std = std
        self.standardize_mask = standardize_mask.to(torch.bool)

    @classmethod
    def fit(cls, x: torch.Tensor, standardize_mask: torch.Tensor, eps: float = 1e-6) -> "Normalizer":
        """Fit per-dimension mean/std over the sample axis.

        Args:
            x: Data, shape ``(N, D)``.
            standardize_mask: Boolean mask, shape ``(D,)``; ``False`` dims become identity.
            eps: Lower bound on std to avoid division by zero.

        Returns:
            A fitted :class:`Normalizer`.
        """
        mask = standardize_mask.to(torch.bool)
        mean = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(eps)
        mean = torch.where(mask, mean, torch.zeros_like(mean))
        std = torch.where(mask, std, torch.ones_like(std))
        return cls(mean, std, mask)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize ``x`` (broadcasts over leading dims)."""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize`."""
        return x * self.std + self.mean

    def to(self, device: torch.device | str) -> "Normalizer":
        """Move the statistics to ``device`` in place and return self."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.standardize_mask = self.standardize_mask.to(device)
        return self

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a serializable dict of the statistics (on CPU for portability)."""
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "standardize_mask": self.standardize_mask.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, d: dict[str, torch.Tensor]) -> "Normalizer":
        """Reconstruct a :class:`Normalizer` from :meth:`state_dict`."""
        return cls(d["mean"], d["std"], d["standardize_mask"])
