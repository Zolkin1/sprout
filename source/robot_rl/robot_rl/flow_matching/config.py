# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class FlowMatchingConfig:
    """Hyperparameters for the flow-matching model, data pipeline, and training.

    ``state_dim`` and ``history_in_dim`` are resolved from the dataset at build
    time (they depend on the joint count and the perceptive-input width), so the
    defaults here are placeholders.
    """

    # -- data / windowing --
    history_len: int = 25
    future_len: int = 25
    stride: int = 1
    val_fraction: float = 0.1
    precompute: bool = True

    # -- dimensions (resolved from the dataset at build time) --
    state_dim: int = 57
    history_in_dim: int | None = None

    # -- model --
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.0

    # -- flow --
    num_sampling_steps: int = 8
    ori6d_aux_weight: float = 0.0

    # -- training --
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 200
    grad_clip: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of the config (for checkpointing)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowMatchingConfig":
        """Build a config from a dict, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
