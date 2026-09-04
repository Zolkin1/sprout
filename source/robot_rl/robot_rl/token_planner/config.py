# Copyright 2026 Zachary Olkin. All rights reserved.

"""Configuration for the tokenized autoregressive trajectory planner.

Two configs, one per stage, and they are deliberately separate objects: the tokenizer is
FROZEN before the planner is trained, so a planner checkpoint embeds the tokenizer's config
as a nested record of what its targets meant rather than sharing mutable fields with it.

Every field that describes the DATA rather than the architecture -- ``node_dim``,
``chunk_len``, ``num_joints``, ``reference_blocks`` -- is resolved from the flow-matching
dataset at build time and written back, so a checkpoint states the numbers it trained under
instead of inheriting whatever the defaults mean on the day it is loaded. That idiom is
copied from :class:`~robot_rl.flow_matching.config.FlowMatchingConfig`, which records five
separate occasions on which an absent key silently picked up a later default.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

#: FSQ level table for the shipped tokenizer: ``prod = 12800`` tokens over 5 latent dims.
#:
#: The ablation named in the spec is ``(7, 5, 5, 5, 5)`` = 4375. Both are in the regime FSQ
#: was characterised in (~2^12 to 2^14); the choice trades reconstruction fidelity against
#: how peaked a categorical head over ``V`` classes can get from a fixed number of samples.
FSQ_LEVELS: tuple[int, ...] = (8, 8, 8, 5, 5)

#: Temporal downsample factor ``D``: nodes per token.
#:
#: ``D = 4`` at the FM's 0.02 s node spacing makes one token **0.08 s of motion**, so a token
#: is a short primitive rather than a pose -- which is the point. It also has to DIVIDE the
#: chunk: the shipped horizon is ``future_len = 64``, so 4 gives 16 tokens per plan. Note
#: that the *older* 62-node horizon (``FlowMatchingConfig.future_len``'s default) is
#: ``2 x 31`` and admits only ``D in {1, 2, 31, 62}``; :meth:`TokenizerConfig.validate`
#: refuses the mismatch rather than padding, because a padded chunk would make the tokens at
#: the seam mean something different from the tokens inside it.
DOWNSAMPLE_D: int = 4


@dataclass
class TokenizerConfig:
    """Hyperparameters for the chunk-local trajectory tokenizer.

    The autoencoder is CHUNK-LOCAL: encoder and decoder only ever see one plan-length chunk
    of ``chunk_len`` nodes and carry no cross-chunk state. That is what makes tokenizing a
    training chunk and decoding a generated token sequence the same operation, which in turn
    is what lets the planner's output be decoded without the tokenizer ever having seen a
    context it was not trained on.
    """

    # -- data (resolved from the FM dataset at build time) --
    #: Width of one trajectory node. Resolved from the recording's reference-block
    #: selection, NOT assumed: the shipped ``nojumpoff`` generator predicts
    #: ``(base_pose, base_vel, joint_pos)`` = 9 + 6 + 29 = 44, while a five-block window is
    #: 94 wide. A tokenizer trained on one and fed the other is a scrambled input of the
    #: wrong width, so this travels in the checkpoint.
    node_dim: int = 44
    #: Nodes per chunk ``H``. Must equal the generator's ``future_len`` -- the tokenizer's
    #: chunk IS one plan, so a mismatch means the planner cannot express a whole plan.
    chunk_len: int = 64
    #: Which reference-window blocks ``node_dim`` is made of, in packing order. Descriptive
    #: (the widths are what the maths uses) but load-bearing for the per-group metrics and
    #: for the agreement check against a generator's own selection.
    reference_blocks: list[str] = field(
        default_factory=lambda: ["base_pose", "base_vel", "joint_pos"]
    )
    #: Robot joint count and end-effector frame count, for splitting reconstruction error by
    #: channel group. Resolved from the dataset.
    num_joints: int = 29
    num_ee_frames: int = 0
    #: Node spacing in seconds, carried so ``chunk_len``/``downsample`` can be read as times.
    dt: float = 0.02
    #: Re-plan period ``K`` the paired tracker runs at. Not used by the autoencoder; recorded
    #: so a checkpoint is self-describing about which planner cadence it belongs to.
    replan_period: int = 12

    # -- bottleneck --
    #: Per-dim FSQ level counts. ``latent_dim`` and ``vocab_size`` are derived from this.
    fsq_levels: list[int] = field(default_factory=lambda: list(FSQ_LEVELS))
    #: Nodes per token. Must divide :attr:`chunk_len`.
    downsample: int = DOWNSAMPLE_D

    # -- architecture --
    #: Trunk width ``C``.
    channels: int = 256
    #: Residual conv blocks at the FULL node rate, before downsampling.
    num_blocks_high: int = 4
    #: Residual conv blocks at the TOKEN rate, after downsampling.
    num_blocks_low: int = 2
    #: Conv kernel size along time (odd, so the padding is symmetric).
    kernel_size: int = 3
    #: Padding at the chunk boundary. ``"replicate"`` and ``"reflect"`` both avoid teaching
    #: the model that a chunk is surrounded by zeros -- a zero-padded chunk end is a
    #: discontinuity the decoder has to spend capacity undoing, and the per-position error
    #: curve is where that shows up. ``"zeros"`` is kept only as the ablation.
    pad_mode: str = "replicate"
    #: How the decoder returns to the node rate. ``"nearest"`` upsamples then convolves;
    #: ``"transpose"`` uses a strided transposed convolution. Nearest+conv avoids the
    #: checkerboard artefacts a transposed conv puts at stride-aligned positions, which on a
    #: trajectory would read as periodic jitter at the token period.
    upsample_mode: str = "nearest"

    # -- loss --
    #: Weight on the first-difference (velocity) reconstruction term. ``0`` = off, which is
    #: v1: the spec gates these on the tracker gating test actually failing on jitter, and
    #: turning them on pre-emptively would make the first number unattributable.
    lambda_vel: float = 0.0
    #: Weight on the second-difference (acceleration) reconstruction term. ``0`` = off.
    lambda_acc: float = 0.0

    # -- training --
    lr: float = 3e-4
    lr_min: float = 3e-5
    weight_decay: float = 0.01
    betas: list[float] = field(default_factory=lambda: [0.9, 0.99])
    warmup_steps: int = 2000
    batch_size: int = 256
    num_epochs: int = 100
    grad_clip: float = 1.0
    seed: int = 0
    #: Exponential-moving-average decay for the evaluated weights, or ``0`` to evaluate the
    #: raw ones. The FM trainer's EMA is what the shipped generator was selected on, so the
    #: same instrument is used here.
    ema_decay: float = 0.999

    @property
    def latent_dim(self) -> int:
        """FSQ latent width ``L_DIM`` -- the number of level entries."""
        return len(self.fsq_levels)

    @property
    def vocab_size(self) -> int:
        """Token vocabulary ``V = prod(fsq_levels)``."""
        return math.prod(int(v) for v in self.fsq_levels)

    @property
    def num_tokens(self) -> int:
        """Tokens per chunk ``N = chunk_len / downsample``."""
        return int(self.chunk_len) // int(self.downsample)

    @property
    def token_seconds(self) -> float:
        """How much motion one token spans, in seconds."""
        return float(self.downsample) * float(self.dt)

    def validate(self) -> None:
        """Check the constraints that would otherwise fail deep inside a conv stack.

        Raises:
            ValueError: If the downsample does not divide the chunk, if the kernel is even,
                or if a size is non-positive.
        """
        if self.chunk_len <= 0 or self.node_dim <= 0 or self.channels <= 0:
            raise ValueError(
                f"chunk_len/node_dim/channels must be positive; got {self.chunk_len}, "
                f"{self.node_dim}, {self.channels}."
            )
        if self.downsample <= 0:
            raise ValueError(f"downsample must be positive; got {self.downsample}.")
        if self.chunk_len % self.downsample:
            raise ValueError(
                f"downsample D={self.downsample} does not divide chunk_len "
                f"H={self.chunk_len}. Pick a divisor ("
                f"{sorted(d for d in range(1, self.chunk_len + 1) if self.chunk_len % d == 0)}"
                ") rather than padding the chunk: a padded chunk makes the tokens at the "
                "seam mean something different from the tokens inside it, and the planner "
                "would have to learn that distinction from the position embedding alone."
            )
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {self.kernel_size}.")
        if self.pad_mode not in ("replicate", "reflect", "zeros"):
            raise ValueError(f"unknown pad_mode {self.pad_mode!r}.")
        if self.upsample_mode not in ("nearest", "transpose"):
            raise ValueError(f"unknown upsample_mode {self.upsample_mode!r}.")
        if self.pad_mode == "reflect" and self.chunk_len // self.downsample < self.kernel_size:
            raise ValueError(
                f"reflect padding needs a sequence longer than the kernel; the token-rate "
                f"stage is {self.num_tokens} long with kernel {self.kernel_size}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict copy, for checkpointing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenizerConfig":
        """Build a config from a dict, ignoring unknown keys.

        Args:
            d: A saved :meth:`to_dict`.

        Returns:
            The config. No legacy-key shims: this is a new artifact, and the moment one is
            needed it belongs here with the reason written down, the way
            ``FlowMatchingConfig.from_dict`` records its five.
        """
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
