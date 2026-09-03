# Copyright 2026 Zachary Olkin. All rights reserved.

"""The trackability critic: ``V(s) = -P(fail | s)`` over the generator's own conditioning.

With a terminal-only reward, ``gamma = 1`` and a unit failure cost, the return is binary and
the value function is exactly a **Bernoulli failure classifier**.  That is simpler than a
distributional regression over bins, trains in minutes, validates with standard classification
metrics, and lands natively in ``(-1, 0)`` with nothing to normalize.

**Warm-started from the generator, and it reuses the generator's own encoder.**  The state a
value function needs here IS the FM's conditioning -- four history nodes, the exteroception
row it was served, and the command over the horizon -- and ``FlowMatchingTransformer`` already
has a trained encoder for exactly that.  Under ``attention_mode="prefix"`` the conditioning
attends only to itself, so its representation is a function of the conditioning ALONE: a fixed
statement about the robot's past, what it can see and what it was asked to do, computed
without reference to any future or to a noise level.  That is precisely a state encoder, and
it is already fitted.

**Privileged inputs enter as TOKENS, not by widening a projection.**  The encoder is a
transformer, so the natural way to add a signal is another token in the sequence -- the scan
becomes a handful of patch tokens through its own small conv, proprioception becomes one token
per timestep, and the base twist plus the goal offsets become a single scalar-context token.
Concatenating them onto a pooled vector instead would put every privileged signal through one
linear layer with no ability to attend, and would waste the one thing the borrowed encoder is
good at.

**The value is read from a READOUT token.**  One extra learned embedding, carrying no data,
prepended to the query stream: it flows through the transformer attending to everything, and
its output alone feeds the head.  This is BERT's ``[CLS]`` and ViT's class token, and the
reason it beats pooling is attention -- a token's output is a weighted sum over the others and
the weights come from ITS query, so the readout learns *what to look at* by gradient descent
on failure.  Mean-pooling 129 tokens gives every one of them equal weight, which for a
129-token sequence whose 112 camera tokens carry the spatial structure the CNN exists to
preserve is close to throwing that structure away.

**The generator's prefix comes over for free, and that is a consequence of ``prefix``
attention rather than an approximation.**  Under that mode the FM's conditioning attends only
to itself, so adding tokens beside it cannot change its representation: ``encode_prefix``
returns exactly what the generator computes, bit for bit, and the new tokens use it as keys
and values while attending to each other as well.  That is structurally the same thing
``decode_future`` does with the future tokens, so the borrowed layers are being used the way
they were trained to be used.

Two variants, and the gap between them is a measurement rather than a convenience:

* ``V_onboard`` -- the FM's conditioning, plus the goal offsets and the ANGULAR half of the
  base twist. Both of those are measurable on the robot (the angular half from an IMU, via the
  same roll/pitch the depth unprojection already uses), and the goal is not optional: with a
  sparse terminal goal reward, two states identical to the generator are worth different
  amounts depending on whether the terrain is one metre ahead or eight, so a critic without it
  is unpredictable BY CONSTRUCTION and its poor fit would read as the method failing rather
  than as a missing input.
* ``V_priv`` -- plus the dense height scan, proprioception, and the LINEAR half of the twist,
  none of which the generator sees and the last of which needs odometry.

**The prediction gap between them measures how much outcome-relevant information the FM cannot
see.**  That is the bias hypothesis, obtained from data being collected anyway.  Advantage
variance explained only by privileged information is variance the FM cannot act on: two chunks
identical from its point of view receiving opposite labels is pure label noise.  So if the gap
is large and the results disappoint, switching the labels to ``V_onboard`` is the first thing
to try -- which is why both are trained from the start and which one produced a label is
recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .extero_encoder import ExteroImageEncoder
from .model import _token_mlp


@dataclass
class ValueInputs:
    """One batch, normalized the way the generator normalizes its own conditioning.

    Attributes:
        history: ``(B, N, D)`` history tokens -- the nodes the generator emitted.
        extero: ``(B, T * K)`` the exteroception row it was served.
        cmd_tokens: ``(B, n_cmd, cmd_dim)`` strided command, or ``None``.
        history_null: ``(B,)`` bool; True while the ring had not filled.
        scan: ``(B, H, W)`` privileged height scan, or ``None``.
        proprio: ``(B, P, proprio_dim)`` per-timestep proprioception, or ``None``.
        context: ``(B, C)`` scalar context -- goal offsets and the base twist.
    """

    history: torch.Tensor
    extero: torch.Tensor
    cmd_tokens: torch.Tensor | None
    history_null: torch.Tensor | None
    scan: torch.Tensor | None = None
    proprio: torch.Tensor | None = None
    context: torch.Tensor | None = None


#: Width of the goal block in the scalar-context token: ``goal_downrange_m``,
#: ``goal_lateral_m``, ``goal_height_err_m``, each on the declared metre scale.
GOAL_BLOCK_WIDTH: int = 3


def context_width(*, privileged: bool, goal_context: bool = True) -> int:
    """Width of the scalar-context token, the ONE place that arithmetic is done.

    Four sites build this vector -- the fit, the mjeval reader, the Isaac online scorer and
    the out-of-fold predictor -- and every one of them used to restate ``3 + (6 if priv else
    3)``. With ``goal_context`` now a per-checkpoint property that is four chances for the
    stamp and the built width to disagree, and a disagreement is a linear layer fed the wrong
    columns rather than a shape error: drop the goal from the batch but not from the model and
    the twist slides into the goal's slots, which is finite and plausible.

    Args:
        privileged: The ``V_priv`` variant, which gets the FULL base twist. ``V_onboard`` gets
            the angular half alone, since nothing on the robot measures the linear half
            without odometry.
        goal_context: Whether the goal offsets are in the token at all. ``False`` is the
            ablation arm: it asks whether the critic can predict failure with no idea how far
            the goal is, which the module docstring above predicts it cannot.

    Returns:
        The width, in scalars.
    """
    return (GOAL_BLOCK_WIDTH if goal_context else 0) + (6 if privileged else 3)


def two_hot(target: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Distribute each scalar target across the two support bins that bracket it.

    The standard two-hot encoding for a distributional value head: a cross-entropy against
    this has the same MEAN as the scalar target while training far more stably than an MSE on
    returns, whose gradient scales with the error and whose target moves as the critic does.

    Exact in the mean by construction, which is the property the head's ``quality`` mean
    accessor relies on -- and it is asserted in the tests rather than argued here.

    Args:
        target: ``(B,)`` values; clamped into the support's range rather than dropped, since a
            value outside it means the reward's own bound was wrong and silently discarding
            those rows would hide it.
        support: ``(bins,)`` ascending bin centres.

    Returns:
        ``(B, bins)`` non-negative, summing to 1 along the last axis.
    """
    bins = support.numel()
    if bins < 2:
        raise ValueError(f"a two-hot target needs at least 2 bins, got {bins}")
    lo, hi = float(support[0]), float(support[-1])
    width = (hi - lo) / (bins - 1)
    pos = ((target.clamp(lo, hi) - lo) / width)
    low = pos.floor().clamp(0, bins - 2).long()
    frac = (pos - low.to(pos.dtype)).clamp(0.0, 1.0)
    out = torch.zeros(target.shape[0], bins, device=target.device, dtype=pos.dtype)
    out.scatter_(1, low.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    out.scatter_add_(1, (low + 1).unsqueeze(1), frac.unsqueeze(1))
    return out


def quality_loss(logits: torch.Tensor, target: torch.Tensor,
                 support: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of ``logits`` against the two-hot encoding of ``target``.

    Args:
        logits: ``(B, bins)`` from :meth:`FlowValueModel.quality_logits`.
        target: ``(B,)`` normalized returns in the support's range.
        support: ``(bins,)`` bin centres.

    Returns:
        Scalar mean loss.
    """
    return -(two_hot(target, support) * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


class FlowValueModel(nn.Module):
    """A failure-probability head reading one readout token over the generator's encoder.

    The query stream is ``[readout | scan patches | proprio steps | context]``; the keys and
    values are that stream concatenated with the generator's own prefix, which
    :meth:`FlowMatchingTransformer.encode_prefix` supplies unchanged. So the new tokens see
    everything -- the generator's conditioning and each other -- while the generator's
    conditioning sees nothing new, which is what makes reusing its representation exact rather
    than approximate.

    Args:
        fm: A trained :class:`FlowMatchingTransformer`, used as encoder and as the layer stack
            the value tokens run through.
        scan_hw: ``(H, W)`` height-scan grid, or ``None`` to omit the scan tokens.
        proprio_dim: Width of one proprioception row; 0 omits those tokens.
        proprio_steps: How many proprioception rows become tokens.
        context_dim: Width of the scalar-context vector; 0 omits that token.
        scan_channels: Conv widths for the scan trunk.
        scan_grid: Pooled token grid for the scan, i.e. how many patch tokens it becomes.
        hidden: Head width.
        freeze_trunk: Keep the generator's weights fixed. The default, and not merely the
            cheap option: it asks the sharper question first -- is the generator's own
            representation predictive of failure -- and answers it in minutes.
        quality_bins: Bins for the QUALITY head's distribution over the normalized tracking
            return, or 0 to omit it. 0 reproduces the survival-only model exactly, so an
            existing checkpoint loads and scores unchanged.
    """

    def __init__(self, fm, *, scan_hw: tuple[int, int] | None = None,
                 proprio_dim: int = 0, proprio_steps: int = 0, context_dim: int = 0,
                 scan_channels: Sequence[int] = (16, 32), scan_grid: tuple[int, int] = (2, 3),
                 hidden: int = 128, freeze_trunk: bool = True,
                 quality_bins: int = 0) -> None:
        super().__init__()
        if getattr(fm, "attention_mode", "full") != "prefix":
            raise ValueError(
                f"FlowValueModel needs a prefix-attention generator, got "
                f"attention_mode={getattr(fm, 'attention_mode', '?')!r}. Under `full` "
                "attention the conditioning's representation depends on the future tokens "
                "and the flow time, so there is no fixed encoding to borrow and the tokens "
                "added here would silently change what the generator's own prefix means.")
        self.fm = fm
        self.freeze_trunk = bool(freeze_trunk)
        if self.freeze_trunk:
            for p in self.fm.parameters():
                p.requires_grad_(False)
        d = int(fm.d_model)
        self.d_model = d

        # The readout. Carries no data; its whole job is to have a QUERY that gradient
        # descent points at whatever predicts failure. Small init so it starts as a neutral
        # probe rather than a strong opinion about where to look.
        self.readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.scan_enc = None
        if scan_hw is not None:
            self.scan_enc = ExteroImageEncoder(
                (int(scan_hw[0]), int(scan_hw[1])), d, tuple(scan_channels),
                grid=(int(scan_grid[0]), int(scan_grid[1])), row_channels=1)
        self.proprio_steps = int(proprio_steps)
        self.proprio_in = (_token_mlp(int(proprio_dim), d) if proprio_dim > 0 else None)
        self.context_in = (_token_mlp(int(context_dim), d) if context_dim > 0 else None)

        # One learned type embedding per stream, so the encoder can tell a proprio token from
        # a scan patch. Without them the streams are distinguishable only by content, and the
        # borrowed layers were never trained to make that distinction.
        self.readout_type = nn.Parameter(torch.zeros(1, 1, d))
        self.scan_type = nn.Parameter(torch.zeros(1, 1, d))
        self.proprio_type = nn.Parameter(torch.zeros(1, 1, d))
        self.context_type = nn.Parameter(torch.zeros(1, 1, d))
        # Proprioception is a time series, so its tokens get a position; the others do not
        # (a scan patch's position is the CNN's learned spatial slot, and readout and context
        # are single tokens).
        self.proprio_pos = nn.Parameter(torch.zeros(1, max(self.proprio_steps, 1), d))

        self.head = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        # ---- the QUALITY head, optional and additive ----------------------------------
        #
        # A SECOND readout rather than a second head on the shared one. The readout's whole
        # argument (above) is that attention lets it learn WHAT TO LOOK AT from its own query
        # -- and "will this fall" and "how cleanly is it tracking" are not the same question,
        # so sharing one query would force one compromise representation for both. One extra
        # token against a 129-token sequence.
        self.quality_bins = int(quality_bins)
        self.quality_readout = None
        self.quality_head = None
        if self.quality_bins > 0:
            self.quality_readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.quality_type = nn.Parameter(torch.zeros(1, 1, d))
            self.quality_head = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                                              nn.Linear(hidden, self.quality_bins))
            # The support is DECLARED, not fitted, and that is a property of the reward rather
            # than a convenience: a step reward is bounded in (0, 1] because the DeepMimic
            # weights sum to 1, and the normalized return carries the (1 - gamma) factor, so
            # G lands in [0, 1] by construction. Nothing here has to track a running scale,
            # and two checkpoints fitted on different data are directly comparable.
            self.register_buffer("quality_support",
                                 torch.linspace(0.0, 1.0, self.quality_bins))

    def value_tokens(self, x: ValueInputs) -> torch.Tensor:
        """Assemble the query stream, readout FIRST.

        Args:
            x: The batch.

        Returns:
            ``(B, 1 + n_scan + n_proprio + n_context, d_model)``.

        Raises:
            ValueError: When a stream the model was built with is missing from the batch. A
                model built with a scan encoder and handed no scan would score every state as
                if the ground were flat, and nothing about the shapes would object.
            """
        b = x.history.shape[0]
        parts = [self.readout.expand(b, 1, self.d_model) + self.readout_type]
        # Index 1 when present. The survival readout stays at 0 so a survival-only
        # checkpoint's head keeps reading the same token.
        if self.quality_readout is not None:
            parts.append(self.quality_readout.expand(b, 1, self.d_model) + self.quality_type)
        if self.scan_enc is not None:
            if x.scan is None:
                raise ValueError("this model has scan tokens but the batch carries no scan; "
                                 "it would score every state as if the ground were flat.")
            parts.append(self.scan_enc(x.scan.reshape(b, -1)) + self.scan_type)
        if self.proprio_in is not None:
            if x.proprio is None:
                raise ValueError("this model has proprio tokens but the batch carries none.")
            parts.append(self.proprio_in(x.proprio) + self.proprio_pos + self.proprio_type)
        if self.context_in is not None:
            if x.context is None:
                raise ValueError(
                    "this model has a context token but the batch carries none; the goal "
                    "offsets live there, and without them P(fail) is not a function of the "
                    "input at all.")
            parts.append(self.context_in(x.context).unsqueeze(1) + self.context_type)
        return torch.cat(parts, dim=1)

    def forward(self, x: ValueInputs) -> torch.Tensor:
        """Predict the failure LOGIT from the readout token.

        Args:
            x: The batch.

        Returns:
            ``(B,)`` logits; ``sigmoid`` gives ``P(fail | s)`` and ``V = -P(fail | s)``.
        """
        # Exactly what the generator computes for its own conditioning -- `prefix` attention
        # means the tokens added below cannot perturb it.
        return self.head(self._encode(x)[:, 0]).squeeze(-1)   # the survival readout, only

    def _encode(self, x: ValueInputs) -> torch.Tensor:
        """Run the token stream through the borrowed layers once.

        Factored out so the two heads share ONE forward pass. Reading them from separate
        calls would double the cost of every selector score for no gain, since the readouts
        differ only in which token the head takes.

        Args:
            x: The batch.

        Returns:
            ``(B, n_tokens, d_model)``.
        """
        cache = self.fm.encode_prefix(x.history, x.extero, x.cmd_tokens, x.history_null)
        h = self.value_tokens(x)
        for layer, prefix in zip(self.fm.encoder.layers, cache):
            h = self.fm._run_layer(layer, h, torch.cat([prefix, h], dim=1))
        return h

    def quality_logits(self, x: ValueInputs) -> torch.Tensor:
        """Logits over the normalized-return bins, from the QUALITY readout.

        Args:
            x: The batch.

        Returns:
            ``(B, quality_bins)``.

        Raises:
            ValueError: If this model has no quality head. Not a zero fallback: a zero return
                is the worst possible value, so a silent one would rank every candidate as
                maximally bad and read as the selector being broken rather than absent.
        """
        if self.quality_head is None:
            raise ValueError(
                "this critic has no quality head (quality_bins=0), so there is no tracking "
                "return to read. Fit one with --quality_bins, or score on survival alone.")
        return self.quality_head(self._encode(x)[:, 1])

    def quality(self, x: ValueInputs) -> torch.Tensor:
        """``G_hat(s)`` in ``[0, 1]`` -- the expected normalized tracking return.

        The distribution's MEAN over the declared support, not its argmax: the head is
        distributional for training stability, but the selector needs a scalar to order
        candidates by and the mean is the one that matches the regression target.

        Args:
            x: The batch.

        Returns:
            ``(B,)``.
        """
        probs = torch.softmax(self.quality_logits(x), dim=-1)
        return (probs * self.quality_support).sum(dim=-1)

    def p_fail(self, x: ValueInputs) -> torch.Tensor:
        """``P(fail | s)`` in ``[0, 1]``."""
        return torch.sigmoid(self.forward(x))

    def heads(self, x: ValueInputs) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Both heads from ONE encoder pass: ``(fail_logit, quality_logits or None)``.

        What training and the selector should call. Two separate calls would run the borrowed
        layer stack twice for the same tokens.

        Args:
            x: The batch.

        Returns:
            ``((B,) survival logit, (B, bins) quality logits or None)``.
        """
        h = self._encode(x)
        fail = self.head(h[:, 0]).squeeze(-1)
        qual = None if self.quality_head is None else self.quality_head(h[:, 1])
        return fail, qual

    def combined_value(self, x: ValueInputs, *, w_fail: float = 1.0) -> torch.Tensor:
        """``V(s) = w_track * G_hat(s) - w_fail * P(fail | s)``, larger is better.

        ONE definition, used by the advantage label and by the selector, so the thing being
        maximised at run time is the thing the fine-tune's label is computed from -- by
        construction rather than by two matching implementations.

        ``w_track = 1 - w_fail``, so ``w_fail = 1`` is exactly the survival-only
        ``V = -P(fail)`` this class shipped with and is therefore the control arm rather than
        a new configuration.

        Args:
            x: The batch.
            w_fail: Weight on the survival term, in ``[0, 1]``.

        Returns:
            ``(B,)``.

        Raises:
            ValueError: If ``w_fail < 1`` on a model with no quality head -- asking for a
                tracking term that does not exist.
        """
        if not 0.0 <= float(w_fail) <= 1.0:
            raise ValueError(f"w_fail must be in [0, 1], got {w_fail!r}")
        fail_logit, qual_logits = self.heads(x)
        v = -float(w_fail) * torch.sigmoid(fail_logit)
        if float(w_fail) >= 1.0:
            return v
        if qual_logits is None:
            raise ValueError(
                f"w_fail={w_fail} asks for the tracking term, but this critic has no quality "
                "head. Fit one, or score at w_fail=1.")
        probs = torch.softmax(qual_logits, dim=-1)
        return v + (1.0 - float(w_fail)) * (probs * self.quality_support).sum(dim=-1)

    def value(self, x: ValueInputs) -> torch.Tensor:
        """``V(s) = -P(fail | s)``, in ``(-1, 0)``."""
        return -self.p_fail(x)

    def trainable_parameters(self):
        """The parameters an optimizer should see."""
        return [p for p in self.parameters() if p.requires_grad]
