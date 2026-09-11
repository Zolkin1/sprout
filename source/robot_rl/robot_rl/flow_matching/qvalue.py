# Copyright 2026 Zachary Olkin. All rights reserved.

"""``Q(s, a)`` over the generator's own samples: score the CHUNK, not just the state.

:class:`~robot_rl.flow_matching.value.FlowValueModel` is ``V(s)`` -- it scores a state.
Best-of-K needs a ranking over the K chunks drawn from ONE state, which is a question about
the ACTION, and a state-only critic answers it identically for every candidate.  This adds the
chunk as a token stream and is otherwise the same architecture, so the two are directly
comparable and the difference between them is attributable.

**The action is the emitted reference window** -- the ``(F, D)`` block the generator produced
and the tracker was then driven by.  Isaac's recorder has stored it all along as ``fm_plan``,
which is why this needs no new data collection: ``logs/rollouts/v2/`` already holds 342,711
``(s, a, outcome)`` rows from the deployed pair with ``fm_zero_noise=False``, i.e. genuinely
stochastic draws.

**The label is SUCCESS, despite being called ``failed``.**  ``rollout_outcomes`` scores an
episode ``0.0`` only if it ARRIVED and did not fall first, and ``-1.0`` otherwise, so the
complement is not "fell over" -- it is "did not do the task".  That distinction is
load-bearing and the module that defines it says why: if merely staying upright scored well,
standing still would score like climbing a staircase with none of the risk, and any method
that reshapes the generator toward high-scoring outcomes finds that immediately.

**Why this can be fitted from on-policy data at all**, without the same-state branching the
original plan called for: a Q-head does not need many actions at literally one state, it needs
the action to vary independently of the state often enough to be identifiable.  Measured on
that rollout, 65,076 ``(clip, replan)`` groups carry two or more independently drawn chunks
and 42,261 carry four or more -- the branch structure, arrived at by repetition across 128
envs rather than by simulator restore.  Those are NEAR-replicates, not replicates: two
episodes on one clip have diverged by re-plan 20 even though they share terrain, command and
start.  Early re-plans are the tight ones.

**The action tokens are STRIDED, not one per node.**  ``F`` is 64, and 64 extra query tokens
would dominate a stream whose other members number ~20.  The stride is the generator's own
``cmd_steps``, the same subsampling its command conditioning uses, so the action is
represented at the offsets the model already treats as the informative ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import _token_mlp
from .value import FlowValueModel, ValueInputs


@dataclass
class QInputs:
    """A :class:`ValueInputs` plus the action being scored.

    Attributes:
        state: Everything ``V(s)`` reads.
        action: ``(B, n_a, D)`` the emitted chunk, already strided to the scored offsets.
    """

    state: ValueInputs
    action: torch.Tensor


def _require_q(x) -> None:
    """Refuse a bare :class:`ValueInputs`, before anything touches its fields.

    Checked at the TOP of every entry point rather than deep inside token assembly: reached
    later, the first thing to run is ``x.state`` and the user gets an ``AttributeError``
    naming a field instead of the reason. And a Q model that silently scored the state alone
    would give every candidate at a state the same value, making the selector a no-op --
    which presents as a weak model rather than as a bug.

    Args:
        x: What was passed in.

    Raises:
        ValueError: If it is not a :class:`QInputs`.
    """
    if not isinstance(x, QInputs):
        raise ValueError(
            f"FlowQModel needs QInputs (state + action), got {type(x).__name__}. Scoring the "
            "state alone would give every candidate at a state the same Q and make the "
            "selector a no-op.")


class FlowQModel(FlowValueModel):
    """``V(s)``'s architecture with the action added as its own token stream.

    Subclasses rather than re-implements so the two share every other choice -- the borrowed
    generator encoder, the readout, the scan/proprio/context streams and their type
    embeddings.  A Q that beat V through some unrelated architectural difference would not be
    evidence that the ACTION carries information, which is the only thing this is built to
    measure.

    Args:
        fm: A trained ``FlowMatchingTransformer``, used as encoder and layer stack.
        action_dim: Width of one action node (``D`` of the reference window).
        action_steps: How many strided nodes become tokens.
        **kw: Forwarded to :class:`~robot_rl.flow_matching.value.FlowValueModel`.
    """

    def __init__(self, fm, *, action_dim: int, action_steps: int, **kw) -> None:
        super().__init__(fm, **kw)
        self.action_steps = int(action_steps)
        self.action_in = _token_mlp(int(action_dim), self.d_model)
        # Its own type embedding, for the same reason every other stream has one: without it
        # the borrowed layers can only tell an action token from a proprio token by content,
        # and they were never trained to make that distinction.
        self.action_type = nn.Parameter(torch.zeros(1, 1, self.d_model))
        # And a position, because the action IS a time series -- node 0 is 20 ms out and node
        # 63 is 1.26 s out, and a permutation-invariant read of it would discard the ordering
        # that makes one chunk a step and another a stumble.
        self.action_pos = nn.Parameter(torch.zeros(1, self.action_steps, self.d_model))
        # Q's OWN head, no longer inherited. `FlowValueModel` used to carry a survival head
        # (`self.head`, predicting P(fail | s)) which this class read for P(fail | s, a); that
        # head is gone from V, because the fine-tune drops its loss and no reported result
        # used it. The question Q answers is still a failure probability, so the head moves
        # here rather than disappearing -- V is a value regression, Q is a ranking score, and
        # they were only ever sharing a module by accident of inheritance.
        # `hidden` MUST match what the inherited head used, or every fitted Q checkpoint
        # becomes unloadable for a reason nothing states: at the production d_model of 512 a
        # `d_model // 2` width is (256, 512) against the real (128, 512). Same default, read
        # from the same kwarg the parent reads.
        _hidden = int(kw.get("hidden", 128))
        self.head = nn.Sequential(nn.Linear(self.d_model, _hidden), nn.SiLU(),
                                  nn.Linear(_hidden, 1))

    def value_tokens(self, x) -> torch.Tensor:  # type: ignore[override]
        """The state stream with the action tokens appended.

        Args:
            x: A :class:`QInputs`.

        Returns:
            ``(B, 1 + n_state + n_action, d_model)``.

        Raises:
            ValueError: If handed a bare :class:`ValueInputs`.  A Q model silently scoring the
                state alone would rank every candidate at a state identically -- exactly the
                failure it exists to avoid, and one that shows up as "the selector does
                nothing" rather than as an error.
        """
        _require_q(x)
        parts = [super().value_tokens(x.state)]
        b = x.action.shape[0]
        parts.append(self.action_in(x.action) + self.action_pos + self.action_type)
        return torch.cat(parts, dim=1)

    def forward(self, x) -> torch.Tensor:  # type: ignore[override]
        """Logit of ``P(fail | s, a)`` from the readout token.

        Args:
            x: A :class:`QInputs`.

        Returns:
            ``(B,)`` logits.
        """
        # Identical to FlowValueModel.forward apart from unwrapping the state: the prefix is
        # what the generator computes for its own conditioning, and under `prefix` attention
        # the tokens added here cannot perturb it.
        #
        # NOT `decode_future`, which takes a flow time and adds its embedding to the tokens.
        # That is the generator's denoising path; borrowing it here would make Q a function of
        # a flow time that has no meaning for a value estimate.
        _require_q(x)
        s = x.state
        cache = self.fm.encode_prefix(s.history, s.extero, s.cmd_tokens, s.history_null)
        h = self.value_tokens(x)
        for layer, prefix in zip(self.fm.encoder.layers, cache):
            h = self.fm._run_layer(layer, h, torch.cat([prefix, h], dim=1))
        return self.head(h[:, 0]).squeeze(-1)      # the readout token, and nothing else

    def p_fail(self, x) -> torch.Tensor:
        """``P(fail | s, a)``.

        Args:
            x: A :class:`QInputs`.

        Returns:
            ``(B,)`` in [0, 1].
        """
        return torch.sigmoid(self.forward(x))

    def q(self, x) -> torch.Tensor:
        """``Q(s, a) = -P(fail | s, a)``, so larger is better and argmax is the selector.

        Args:
            x: A :class:`QInputs`.

        Returns:
            ``(B,)``.
        """
        return -self.p_fail(x)
