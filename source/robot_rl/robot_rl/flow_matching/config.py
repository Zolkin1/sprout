# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the flow-matching future-state prediction model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class FlowMatchingConfig:
    """Hyperparameters for the flow-matching model, data pipeline, and training.

    ``state_dim`` and ``cmd_dim`` are resolved from the dataset at build time (they
    depend on the joint/extra-frame count and the command width), so the defaults
    here are placeholders.

    Architecture (PARC-style single transformer, inpainting): the model denoises a
    window of ``future_len`` state tokens. The ``num_history_states`` most-recent
    measured states are dropped in as *clean* conditioning tokens (held fixed at
    every flow-time and excluded from the loss); the rest of the window is the
    predicted future. Effective novel horizon is ``future_len - num_history_states``.
    During training the history is dropped with probability ``history_dropout_prob``
    (the whole window is denoised unconditionally), which regularizes away history
    over-reliance and lets the model predict from step 0 after a reset.
    """

    # -- data / windowing --
    # Window length F: the full denoised state sequence. The first ``num_history_states``
    # tokens are the recent past (conditioning), the rest are the predicted future. The
    # relative-frame anchor is the most recent history step (index num_history_states-1).
    future_len: int = 25
    num_history_states: int = 2
    stride: int = 1
    val_fraction: float = 0.1
    precompute: bool = True

    # Future-target source: "actual" predicts the recorded robot state; "reference"
    # predicts the tracked reference (requires ``ref_state`` in the recording). The
    # conditioning (the clean history tokens) is always the actual recorded state.
    target_source: str = "actual"

    # Dedicated recorded command array (e.g. "target_vel") giving the per-step
    # commanded velocity over the window. It is sampled every ``cmd_stride`` steps
    # (indices 0, cmd_stride, ...) and each sample becomes its own conditioning token
    # (a separate command-embedding MLP). ``None`` means no command conditioning.
    cmd_array: str | None = None
    cmd_stride: int = 5

    # -- dimensions (resolved from the dataset at build time) --
    # Per-step state width of every token: 2*nj + 15*(1 + num_extra_frames). History
    # and predicted states share this single representation (no asymmetry).
    state_dim: int = 57
    # Per-step command width (0 disables command conditioning).
    cmd_dim: int = 0

    # -- model (single transformer; no separate history encoder) --
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.0
    # Probability that a training sample drops its history (all tokens denoised
    # unconditionally). Regularizes history over-reliance; enables step-0 prediction.
    history_dropout_prob: float = 0.8

    # -- observation noise (train-time augmentation on the clean history tokens) --
    # Mirrors the RL teacher/tracker obs corruption (Unoise per term): the inputs the
    # generator conditions on at deployment come from noisy sensors, so the first
    # ``num_history_states`` (conditioning) state tokens are perturbed during training.
    # Noise is applied ONLY to those conditioning tokens, never to the prediction
    # target; it is resampled every access and disabled for val/eval/deployment.
    # ``obs_noise`` is the master flag; ``noise_scale`` scales every amplitude at once.
    # Amplitudes are physical, uniform half-widths (the sample is U(-amp, amp)).
    obs_noise: bool = True
    noise_scale: float = 1.0
    noise_joint_pos: float = 0.01   # rad        (RL match)
    noise_joint_vel: float = 0.5    # rad/s      (RL match)
    noise_pos: float = 0.02         # m          base + extra-frame positions (estimator)
    noise_lin_vel: float = 0.1      # m/s        base + extra-frame lin vel (estimator)
    noise_ori: float = 0.05         # ori6d cols (RL: projected_gravity)
    noise_ang_vel: float = 0.2      # rad/s      (RL: gyro)

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

    @property
    def num_cmd_tokens(self) -> int:
        """Number of strided command tokens (0 when not command-conditioned)."""
        if self.cmd_dim <= 0:
            return 0
        return math.ceil(self.future_len / self.cmd_stride)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict copy of the config (for checkpointing)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlowMatchingConfig":
        """Build a config from a dict, ignoring unknown keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
