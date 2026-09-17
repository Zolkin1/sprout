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

from .model import _token_mlp


@dataclass
class ValueInputs:
    """One batch, normalized the way the generator normalizes its own conditioning.

    Attributes:
        history: ``(B, N, D)`` history tokens -- the nodes the generator emitted.
        extero: ``(B, T * K)`` the exteroception row it was served.
        cmd_tokens: ``(B, n_cmd, cmd_dim)`` strided command, or ``None``.
        history_null: ``(B,)`` bool; True while the ring had not filled.
        context: ``(B, C)`` scalar context -- goal offsets and the base twist.
    """

    history: torch.Tensor
    extero: torch.Tensor
    cmd_tokens: torch.Tensor | None
    history_null: torch.Tensor | None
    context: torch.Tensor | None = None


#: Width of the goal block in the scalar-context token: ``goal_downrange_m``,
#: ``goal_lateral_m``, ``goal_height_err_m``, each on the declared metre scale.
GOAL_BLOCK_WIDTH: int = 3


#: Width of the worst-penalty-so-far block: one scalar.
#:
#: Naming: the FLAG is ``worst_penalty_context`` and the PIECE is ``worst_penalty``, mirroring
#: ``goal_context`` / ``goal``.  They must differ -- :func:`build_context` takes the pieces as
#: ``**kwargs``, so a flag sharing a piece's name is a syntax error at every call site that
#: passes both.
WORST_BLOCK_WIDTH: int = 1


def context_layout(*, privileged: bool, goal_context: bool = True,
                   worst_penalty_context: bool = False) -> tuple[tuple[str, int], ...]:
    """The scalar-context token's blocks, in order, as ``(name, width)``.

    **The layout is DATA, and it is the only statement of it.**  Five sites build this vector
    -- the fit, ``fit_q``, the mjeval reader, the Isaac online scorer and the out-of-fold
    predictor -- and each used to assemble it by hand, so the ORDER lived five times while the
    width lived once.  That is the dangerous shape: a disagreement is a linear layer fed the
    wrong columns rather than a shape error.  Drop the goal from the batch but not from the
    model and the twist slides into the goal's slots, which is finite and plausible and shows
    up as a critic that merely predicts a bit worse.

    So both :func:`context_width` and :func:`build_context` read this, and a caller supplies
    NAMED pieces rather than a concatenation order.  Adding a block is one edit here.

    Order note: ``goal`` is first because every existing checkpoint has it in columns 0:3, and
    new blocks are APPENDED so those positions never move.

    Args:
        privileged: The ``V_priv`` variant, which gets the FULL base twist. ``V_onboard`` gets
            the angular half alone, since nothing on the robot measures the linear half
            without odometry.
        goal_context: Whether the goal offsets are in the token at all. ``False`` is the
            ablation arm: it asks whether the critic can predict failure with no idea how far
            the goal is, which the module docstring above predicts it cannot.
        worst_penalty_context: Whether the WORST TERRAIN PENALTY SO FAR is in the token.  It is a
            property of the episode's history, and it is what lets the critic predict a
            return whose success term is gated on the episode's worst chunk -- without it,
            ``V(s_t)`` has no way to know that the bonus has already been forfeited.
            **Strictly the chunks BEFORE t**: including t's own penalty would leak part of
            ``r_t`` into ``V(s_t)`` and cancel exactly that much of the advantage.

    Returns:
        The blocks, in concatenation order.
    """
    blocks: list[tuple[str, int]] = []
    if goal_context:
        blocks.append(("goal", GOAL_BLOCK_WIDTH))
    blocks.append(("twist", 6 if privileged else 3))
    if worst_penalty_context:
        blocks.append(("worst_penalty", WORST_BLOCK_WIDTH))
    return tuple(blocks)


def context_width(*, privileged: bool, goal_context: bool = True,
                  worst_penalty_context: bool = False) -> int:
    """Width of the scalar-context token, derived from :func:`context_layout`.

    Args:
        privileged: See :func:`context_layout`.
        goal_context: See :func:`context_layout`.
        worst_penalty_context: See :func:`context_layout`.

    Returns:
        The width, in scalars.
    """
    return sum(w for _, w in context_layout(privileged=privileged,
                                            goal_context=goal_context,
                                            worst_penalty_context=worst_penalty_context))


def build_context(*, privileged: bool, goal_context: bool = True,
                  worst_penalty_context: bool = False,
                  **pieces: torch.Tensor) -> torch.Tensor:
    """Assemble the scalar-context token from NAMED pieces, in the canonical order.

    The counterpart to :func:`context_layout`: callers pass ``goal=``, ``twist=`` and
    ``worst_penalty=`` and never choose an order.  Every block the layout asks for must be
    supplied at the width it asks for, and anything extra is refused -- a silently ignored
    piece is a caller that thinks it contributed a feature and did not.

    ``twist`` is passed FULL (6 wide) and sliced here for the onboard variant, so the "keep
    only the angular half" rule lives with the layout rather than at each call site.

    Args:
        privileged: See :func:`context_layout`.
        goal_context: See :func:`context_layout`.
        worst_penalty_context: See :func:`context_layout`.
        **pieces: ``(B, w)`` tensors by block name.  ``twist`` may be ``(B, 6)`` on either
            variant.

    Returns:
        ``(B, context_width(...))``.

    Raises:
        ValueError: On a missing block, an unexpected one, or a width that does not match.
    """
    layout = context_layout(privileged=privileged, goal_context=goal_context,
                            worst_penalty_context=worst_penalty_context)
    want = {n for n, _ in layout}
    extra = sorted(set(pieces) - want)
    if extra:
        raise ValueError(
            f"build_context got {extra}, which this layout does not carry "
            f"(privileged={privileged}, goal_context={goal_context}, "
            f"worst_penalty_context={worst_penalty_context}). A piece silently dropped is a "
            "CALLER that believes it contributed a feature and did not.")
    out: list[torch.Tensor] = []
    for name, width in layout:
        if name not in pieces:
            raise ValueError(f"build_context needs {name!r} ({width} wide) for this layout.")
        t = pieces[name]
        if name == "twist" and t.shape[-1] == 6 and width == 3:
            t = t[..., 3:]          # the ANGULAR half; see context_layout
        if t.shape[-1] != width:
            raise ValueError(
                f"build_context got {name!r} {tuple(t.shape)} but the layout wants width "
                f"{width}.")
        out.append(t)
    return torch.cat(out, dim=-1)


def quality_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain MSE of a SCALAR quality head against its return target.

    The alternative to :func:`quality_loss`'s two-hot cross-entropy, and the two are kept side
    by side so the choice can be A/B'd rather than argued: the distributional head trains more
    stably in general, but it needs a declared support, and an MSE needs only that the target
    be bounded -- which the clipped window return guarantees by construction (``[-1, 0]``).

    Args:
        pred: ``(B,)`` scalar prediction.
        target: ``(B,)`` return target.

    Returns:
        Scalar mean squared error.
    """
    return torch.nn.functional.mse_loss(pred.reshape(-1), target.reshape(-1))


class FlowValueModel(nn.Module):
    """A VALUE head reading one readout token over the generator's frozen encoder.

    ``V(s)`` is a scalar regression of the normalized return. ONE head and ONE readout: this
    class used to carry a survival head predicting ``P(fail | s)`` on its own readout token,
    and a distributional (two-hot) form of the value head. Neither produced a reported result
    -- the fine-tune drops the survival loss (``--quality_only``) and regresses a scalar
    (``--quality_kind mse``) -- so both are deleted rather than described.

    The query stream is ``[readout | scan patches | proprio steps | context]``; the keys and
    values are that stream concatenated with the generator's own prefix, which
    :meth:`FlowMatchingTransformer.encode_prefix` supplies unchanged. So the new tokens see
    everything -- the generator's conditioning and each other -- while the generator's
    conditioning sees nothing new, which is what makes reusing its representation exact rather
    than approximate.

    Args:
        fm: A trained :class:`FlowMatchingTransformer`, used as encoder and as the layer stack
            the value tokens run through.
        context_dim: Width of the scalar-context vector; 0 omits that token.
        hidden: Head width.
        freeze_trunk: Keep the generator's weights fixed. The default, and not merely the
            cheap option: it asks the sharper question first -- is the generator's own
            representation predictive of failure -- and answers it in minutes.
    """

    def __init__(self, fm, *, context_dim: int = 0,
                 hidden: int = 128, freeze_trunk: bool = True) -> None:
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
        # BOTH directions, explicitly. `load_fm_context` now freezes the generator's
        # parameters at load (it has to: `requires_grad` on any argument gates
        # `nn.MultiheadAttention`'s fused fast path, so every inference consumer must agree
        # on it). That made this an asymmetric switch: `freeze_trunk=False` would arrive with
        # the trunk already frozen, `trainable_parameters()` filters on `requires_grad`, and
        # the optimizer would silently get the heads alone -- a fine-tune that runs, whose
        # loss falls, and that trains nothing it was asked to.
        for p in self.fm.parameters():
            p.requires_grad_(not self.freeze_trunk)
        d = int(fm.d_model)
        self.d_model = d

        # The readout. Carries no data; its whole job is to have a QUERY that gradient
        # descent points at whatever predicts failure. Small init so it starts as a neutral
        # probe rather than a strong opinion about where to look.
        self.readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        # NO scan and NO proprio tokens. The critic used to add a CNN trunk over the
        # privileged height scan (pooled to a 2x3 patch grid) and one token per
        # proprioception step, each with its own type embedding. Both are gone: the
        # architecture the paper describes reads the generator's own conditioning plus a
        # scalar-context token, and nothing else.
        #
        # This is the one cut here that CHANGES V rather than removing something untrained --
        # the scan and proprio tokens were live inputs the value readout attended to. Judge a
        # refit on `critic.log`'s n1 DIFFERENCE ratio (1.01 with the scan, 4.55 at round 0),
        # not on the success rate: that ratio is sd(predicted change)/sd(real change) between
        # ADJACENT chunks, which is exactly what AWR exponentiates, and a success rate can
        # stay flat for several iterations while it degrades.
        self.context_in = (_token_mlp(int(context_dim), d) if context_dim > 0 else None)

        # One learned type embedding per stream, so the borrowed layers can tell the readout
        # from the context token by more than content -- they were never trained to make that
        # distinction. Two streams now; there used to be four, with a POSITION on the proprio
        # tokens because that stream was a time series. Both single tokens need no position.
        self.readout_type = nn.Parameter(torch.zeros(1, 1, d))
        self.context_type = nn.Parameter(torch.zeros(1, 1, d))

        # ONE readout and ONE head: a scalar regression of the normalized return.
        #
        # There used to be a second, SURVIVAL head predicting P(fail | s) with its own readout
        # token -- "will this fall" and "how cleanly is it tracking" are not the same question,
        # so they did not share a query -- plus a distributional (two-hot) form of this head.
        # All of it is gone. None of it produced a reported result: the fine-tune loop passes
        # `--quality_only`, which drops the survival loss entirely, and `--quality_kind mse`.
        # Carrying them meant the architecture could not be described without also describing
        # experiments that never ran.
        self.quality_head = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(),
                                          nn.Linear(hidden, 1))

    def value_tokens(self, x: ValueInputs) -> torch.Tensor:
        """Assemble the query stream, readout FIRST.

        Args:
            x: The batch.

        Returns:
            ``(B, 1 + n_context, d_model)``.

        Raises:
            ValueError: When a stream the model was built with is missing from the batch. A
                model built with a scan encoder and handed no scan would score every state as
                if the ground were flat, and nothing about the shapes would object.
            """
        b = x.history.shape[0]
        parts = [self.readout.expand(b, 1, self.d_model) + self.readout_type]
        if self.context_in is not None:
            if x.context is None:
                raise ValueError(
                    "this model has a context token but the batch carries none; the goal "
                    "offsets live there, and without them P(fail) is not a function of the "
                    "input at all.")
            parts.append(self.context_in(x.context).unsqueeze(1) + self.context_type)
        return torch.cat(parts, dim=1)

    def forward(self, x: ValueInputs) -> torch.Tensor:
        """``V(s)`` for a batch.

        Args:
            x: The batch.

        Returns:
            ``(B,)`` expected normalized return.
        """
        # Exactly what the generator computes for its own conditioning -- `prefix` attention
        # means the tokens added below cannot perturb it.
        return self.quality(x)

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
        """The head's raw output, from the readout token.

        Args:
            x: The batch.

        Returns:
            ``(B, 1)``.
        """
        # Token 0, and the ONLY readout. It was token 1 while a survival readout occupied 0,
        # and moving it is not cosmetic: this token ATTENDS to the rest of the stream, so
        # dropping one key/value changes the representation the head reads. Every critic
        # fitted before that cut is a different function rather than a checkpoint with an
        # extra unused head -- which is why they do not load, and why numbers measured with
        # one do not describe this model.
        return self.quality_head(self._encode(x)[:, 0])

    def quality(self, x: ValueInputs) -> torch.Tensor:
        """``V(s)`` -- the expected normalized return.

        The head's single output IS the prediction and is returned RAW -- deliberately not
        clamped into the reward's range. An advantage is ``G - V``, so clamping would bias it
        wherever the critic is miscalibrated, and an out-of-range prediction is information
        about the fit rather than something to hide.

        Args:
            x: The batch.

        Returns:
            ``(B,)``.
        """
        out = self.quality_logits(x)
        return out.reshape(out.shape[0])

    def trainable_parameters(self):
        """The parameters an optimizer should see."""
        return [p for p in self.parameters() if p.requires_grad]
