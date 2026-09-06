# Copyright 2026 Zachary Olkin. All rights reserved.

"""The conditioning prefix -- the flow-matching generator's, reused unchanged.

The autoregressive planner conditions on exactly what the generator conditions on: the
robot-state history, the exteroception (two depth cameras, or a height scan), and the strided
future command. Reusing that assembly rather than inventing one is the point of the whole
comparison -- if the AR planner's closed-loop number differs from the FM's, the difference
has to be attributable to the output representation and not to a differently-wired depth
encoder.

.. admonition:: Reused, and CHECKED to be reused

    The substantial part -- :class:`~robot_rl.flow_matching.extero_encoder.ExteroImageEncoderSet`,
    the per-camera conv trunks with their angular slot embeddings -- is imported, not copied.
    What is restated here is roughly thirty lines of token assembly, because
    ``FlowMatchingTransformer._embed_prefix`` is a method on a class that also builds a flow
    trunk, and instantiating one of those to borrow a third of it would drag an unused
    parameter set into every checkpoint and optimizer.

    Restating it is only safe if it is verified, so the parameter NAMES match the generator's
    exactly and ``test_prefix_matches_the_flow_matching_model`` copies a
    :class:`~robot_rl.flow_matching.model.FlowMatchingTransformer`'s weights across and
    requires bit-identical output. A drift in the assembly -- a missing type embedding, a
    positional offset off by ``num_history_states`` -- is otherwise completely silent: the
    prefix has the right shape under any of them.

**No flow-time embedding, ever.** The generator adds ``tau`` to its future tokens only, which
is what makes its prefix cacheable across Euler steps. There is no ``tau`` here at all, and
the same cacheability carries over to autoregressive decoding: the prefix is encoded once per
re-plan and reused by all ``N`` token steps.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from robot_rl.flow_matching.extero_encoder import ExteroImageEncoderSet
from robot_rl.flow_matching.model import _sinusoidal_positional_encoding, _token_mlp


class ConditioningPrefix(nn.Module):
    """Embeds ``(history, extero, cmd)`` into ``(B, prefix_len, d_model)`` tokens.

    Args:
        history_dim: Width of one history row (the reference-window node width).
        num_history_states: History rows ``N``.
        future_len: Plan horizon ``H``. Sets the positional timeline and, with
            ``cmd_stride``, which steps carry a command token -- so it must match the
            generator's, or the command tokens land on different timesteps.
        extero_sensor_dim: Width of ONE exteroception reading (``channels * H * W``).
        extero_sensor_count: Number of readings ``T`` (2 for the camera pair).
        extero_encoder: ``"cnn"`` or ``"mlp"``.
        extero_image_hw: One ``(H, W)`` per sensor; required for ``"cnn"``.
        extero_cnn_channels / extero_cnn_grid / extero_cnn_strides: Conv trunk shape.
        extero_coord_extent_m / extero_grid_ordering / extero_fourier_bands: Metric-grid
            position embedding, for a height scan.
        extero_miss_sentinel: Normalized missed-ray value, enabling the validity channel.
        extero_channels: Planes per exteroception pixel.
        extero_camera_intrinsics: One ``(fx, fy, cx, cy)`` per camera, for the angular
            token position embedding.
        cmd_dim: Command width (0 disables command conditioning).
        cmd_stride: Command tokens land on future steps ``0, cmd_stride, ...``.
        d_model: Transformer width.

    Raises:
        ValueError: On an unknown encoder, or a ``"cnn"`` without image shapes.
    """

    def __init__(
        self,
        history_dim: int,
        num_history_states: int,
        future_len: int,
        extero_sensor_dim: int,
        extero_sensor_count: int,
        d_model: int,
        extero_encoder: str = "cnn",
        extero_image_hw: Sequence[Sequence[int]] | None = None,
        extero_cnn_channels: Sequence[int] = (16, 32, 64),
        extero_cnn_grid: Sequence[int] | None = None,
        extero_cnn_strides: Sequence[int] | None = None,
        extero_coord_extent_m: Sequence[float] | None = None,
        extero_grid_ordering: str = "xy",
        extero_fourier_bands: int = 4,
        extero_miss_sentinel: float | None = None,
        extero_channels: int = 1,
        extero_camera_intrinsics: Sequence[Sequence[float]] | None = None,
        cmd_dim: int = 0,
        cmd_stride: int = 5,
        plan_dim: int = 0,
        plan_tokens: int = 0,
        plan_stride: int = 1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.history_dim = int(history_dim)
        self.num_history_states = int(num_history_states)
        self.future_len = int(future_len)
        self.extero_sensor_dim = int(extero_sensor_dim)
        self.extero_sensor_count = int(extero_sensor_count)
        self.extero_encoder = str(extero_encoder)
        self.cmd_dim = int(cmd_dim)
        self.cmd_stride = int(cmd_stride)

        self.history_in = _token_mlp(self.history_dim, d_model)
        self.hist_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.null_history = nn.Parameter(torch.zeros(1, 1, d_model))

        if self.extero_encoder == "cnn":
            if not extero_image_hw:
                raise ValueError(
                    "extero_encoder='cnn' needs extero_image_hw (one (H, W) per sensor)."
                )
            shapes = [(int(h), int(w)) for h, w in extero_image_hw]
            self.extero_in = None
            self.extero_cnn = ExteroImageEncoderSet(
                shapes, d_model, extero_cnn_channels,
                None if extero_cnn_grid is None else (int(extero_cnn_grid[0]),
                                                      int(extero_cnn_grid[1])),
                strides=extero_cnn_strides,
                coord_extent_m=(None if extero_coord_extent_m is None
                                else (float(extero_coord_extent_m[0]),
                                      float(extero_coord_extent_m[1]))),
                ordering=str(extero_grid_ordering),
                fourier_bands=int(extero_fourier_bands),
                miss_sentinel=extero_miss_sentinel,
                row_channels=int(extero_channels),
                intrinsics=extero_camera_intrinsics,
            )
            self.tokens_per_sensor = int(self.extero_cnn.tokens_per_sensor)
        elif self.extero_encoder == "mlp":
            self.extero_in = _token_mlp(self.extero_sensor_dim, d_model)
            self.extero_cnn = None
            self.tokens_per_sensor = 1
        else:
            raise ValueError(f"unknown extero_encoder {self.extero_encoder!r}.")
        self.extero_type = nn.Parameter(torch.zeros(1, 1, d_model))
        # Only created for T > 1, matching the generator: at T=1 the parameter set is
        # identical to a single-sensor model's, which is what keeps the equivalence test
        # meaningful on both arms.
        self.extero_id = (
            nn.Parameter(torch.zeros(1, self.extero_sensor_count, d_model))
            if self.extero_sensor_count > 1 else None
        )

        if self.cmd_dim > 0:
            self.cmd_in = _token_mlp(self.cmd_dim, d_model)
            self.cmd_type = nn.Parameter(torch.zeros(1, 1, d_model))
            self.register_buffer(
                "cmd_steps",
                torch.tensor(list(range(0, self.future_len, self.cmd_stride)),
                             dtype=torch.long),
            )
        else:
            self.cmd_in = None
            self.cmd_type = None
            self.register_buffer("cmd_steps", torch.zeros(0, dtype=torch.long))

        # The PLAN stream: the generator's own prediction, as one token per output slot.
        #
        # Off by default (``plan_tokens=0``), and that default is load-bearing: with it the
        # module is byte-identical to what it was, so
        # ``test_the_prefix_matches_the_flow_matching_models_embedding`` -- which copies a
        # FlowMatchingTransformer's weights across by NAME and requires bit-identical output
        # -- keeps holding. An adjuster is the only thing that turns it on, and the generator
        # has no counterpart for it to disagree with.
        self.plan_dim = int(plan_dim)
        self.plan_tokens = int(plan_tokens)
        self.plan_stride = int(plan_stride)
        if self.plan_tokens > 0:
            if self.plan_dim <= 0:
                raise ValueError(
                    f"plan_tokens={self.plan_tokens} needs a positive plan_dim; got "
                    f"{self.plan_dim}.")
            self.plan_in = _token_mlp(self.plan_dim, d_model)
            self.plan_type = nn.Parameter(torch.zeros(1, 1, d_model))
            # Plan token i covers plan nodes [i*stride, (i+1)*stride), so it sits on the
            # TIMELINE at that node -- the same convention the command tokens use, and the
            # reason it is the right one: a plan token and the output slot that corrects it
            # describe the same instant, and the trunk can only exploit that if their
            # positions say so.
            self.register_buffer(
                "plan_steps",
                torch.tensor([i * self.plan_stride for i in range(self.plan_tokens)],
                             dtype=torch.long),
            )
        else:
            self.plan_in = None
            self.plan_type = None
            self.register_buffer("plan_steps", torch.zeros(0, dtype=torch.long))

        # The generator's timeline: history at 0..N-1, the plan at N..N+H-1, each command
        # token at N + (its future step). The exteroception carries no timeline position --
        # the reading is "now" -- only a type, a per-sensor id and (CNN) a spatial slot.
        self.register_buffer(
            "pos",
            _sinusoidal_positional_encoding(self.num_history_states + self.future_len, d_model),
        )

    @property
    def extero_seq_tokens(self) -> int:
        """Exteroception tokens in the sequence (sensors x tokens per sensor)."""
        return int(self.extero_sensor_count) * int(self.tokens_per_sensor)

    @property
    def prefix_len(self) -> int:
        """Conditioning tokens ``P``: history + exteroception + command + plan."""
        return (int(self.num_history_states) + self.extero_seq_tokens
                + int(self.cmd_steps.numel()) + int(self.plan_steps.numel()))

    def forward(
        self,
        history: torch.Tensor,
        extero: torch.Tensor,
        cmd_tokens: torch.Tensor | None = None,
        history_null_mask: torch.Tensor | None = None,
        plan: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the conditioning tokens.

        Args:
            history: ``(B, N, history_dim)`` history rows.
            extero: ``(B, T, extero_sensor_dim)``, or anything that reshapes to it.
            cmd_tokens: ``(B, num_cmd_tokens, cmd_dim)``, or ``None`` when ``cmd_dim == 0``.
            history_null_mask: ``(B,)`` bool; where True the history content is replaced by
                the learned null embedding. This is the same defence the generator has, and
                it covers the same real situation -- the first ``N`` control steps after a
                reset, when no history exists yet.

            plan: ``(B, plan_tokens, plan_dim)`` the generator's own prediction, folded
                ``plan_stride`` nodes to a token. Required exactly when the module was built
                with a plan stream, and refused when it was not -- an adjuster that silently
                ran without the plan it is supposed to be correcting would still produce
                tokens, and the only symptom would be that its corrections were worse.

        Returns:
            ``(B, prefix_len, d_model)``.

        Raises:
            ValueError: If command conditioning was built but no tokens were given, or if
                the plan stream and the ``plan`` argument disagree about existing.
        """
        b = history.shape[0]
        n = self.num_history_states

        h_content = self.history_in(history)
        if history_null_mask is not None:
            nm = history_null_mask.view(b, 1, 1)
            h_content = torch.where(
                nm, self.null_history.expand(b, n, self.d_model), h_content)
        h = h_content + self.pos[:, :n, :] + self.hist_type

        ev = extero.reshape(b, self.extero_sensor_count, self.extero_sensor_dim)
        et = (self.extero_cnn(ev) if self.extero_cnn is not None
              else self.extero_in(ev).unsqueeze(2))        # (B, T, S, d_model); S=1 for mlp
        et = et + self.extero_type.unsqueeze(1)
        if self.extero_id is not None:
            et = et + self.extero_id.unsqueeze(2)
        # Camera-major, slot-minor -- the order the ids and slots were added in.
        ht = et.reshape(b, self.extero_seq_tokens, self.d_model)

        if (self.plan_in is None) != (plan is None):
            raise ValueError(
                "the plan stream and the `plan` argument must agree: this prefix was built "
                f"{'WITH' if self.plan_in is not None else 'WITHOUT'} one and `plan` was "
                f"{'given' if plan is not None else 'not given'}.")

        parts = [h, ht]
        if self.plan_in is not None:
            if plan.shape[1:] != (int(self.plan_steps.numel()), self.plan_dim):
                raise ValueError(
                    f"plan must be (B, {int(self.plan_steps.numel())}, {self.plan_dim}), got "
                    f"{tuple(plan.shape)}.")
            parts.append(self.plan_in(plan)
                         + self.pos[:, n + self.plan_steps, :] + self.plan_type)
        if self.cmd_in is not None:
            if cmd_tokens is None:
                raise ValueError("prefix built with cmd_dim > 0 requires cmd_tokens.")
            parts.append(
                self.cmd_in(cmd_tokens) + self.pos[:, n + self.cmd_steps, :] + self.cmd_type)
        return torch.cat(parts, dim=1)
