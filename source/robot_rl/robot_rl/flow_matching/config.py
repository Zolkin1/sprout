# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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

    # Future-target source: "actual" predicts the recorded robot state; "reference"
    # predicts the tracked reference (requires ``ref_state`` in the recording).
    # History/conditioning is always the actual recorded state.
    target_source: str = "actual"

    # Dedicated recorded command array (e.g. "target_vel") to condition on over the
    # *whole* window: past steps in the encoder history token, future steps injected
    # into the decoder (future-horizon command conditioning). ``None`` means no
    # command conditioning. This is the sole command path.
    cmd_array: str | None = None

    # Extra body frames (feet/hands) the *history*/encoder token is built from.
    # Default ``[]`` -> the history uses NO extra frames: this is the deployment
    # design, so the generator needs NO measured end-effectors from the robot while
    # the target still PREDICTS every recorded frame to feed the tracker's ``ee_ref``
    # obs. ``None`` -> symmetric: the history uses every recorded extra frame (legacy
    # behaviour). An explicit list (a subset of the recording's ``state_frame_names``)
    # lets the encoder consume a chosen subset. The two resolved lists
    # (history_frame_names / target_frame_names) are persisted in the checkpoint meta;
    # the target set is always every recorded extra frame. ``state_dim`` is the target
    # width and ``history_in_dim`` the (possibly narrower) history-token width.
    history_frame_names: list[str] | None = field(default_factory=list)

    # -- dimensions (resolved from the dataset at build time) --
    state_dim: int = 57
    history_in_dim: int | None = None
    # Per-step future-command width fed to the decoder (0 disables future-command
    # conditioning); resolved from the dataset (= cmd width when ``cmd_array`` is set).
    cmd_dim: int = 0

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
    num_epochs: int = 100
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
