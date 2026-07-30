# Copyright 2026 Zachary Olkin. All rights reserved.

"""Actor model with a dedicated encoder branch for the reference trajectory.

The teacher's observation splits cleanly in two: *proprioception* (what the robot felt,
history-stacked) and the *reference* (what it is being asked to do, a 1 s look-ahead
window that is by far the widest part of the input).  Feeding both to one flat MLP makes
the trunk spend its first layer on the reference; this model instead compresses the
reference with its own MLP and hands the trunk a fixed-width latent alongside the raw
proprioception.

Optionally the latent is additionally supervised by reconstructing the reference back out
of it (:meth:`RefEncoderMLPModel.reconstruction_loss`), which
:class:`~robot_rl.rl.aux_ppo.AuxPPO` adds to the PPO loss.  Off by default -- it exists to
be ablated.

Structure follows ``rsl_rl.models.cnn_model.CNNModel`` (BSD-3-Clause, ETH Zurich / NVIDIA):
partition the actor's observation groups, encode some of them, widen the trunk input.  The
export wrappers deliberately take ONE flat observation vector and split it internally, so
``scripts/rsl_rl/export_tracker_onnx.py``'s single-``obs``-input contract is unchanged --
only the vector's length and term ordering move.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState


class RefEncoderMLPModel(MLPModel):
    """MLP actor/critic whose *reference* observation groups go through an encoder.

    Observation groups named in ``encoder_obs_groups`` are concatenated, pushed through
    an encoder MLP, and the resulting latent is concatenated with the remaining
    ("direct") groups before the trunk.  With ``encoder_obs_groups=()`` this degenerates
    to a plain :class:`~rsl_rl.models.MLPModel`.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_obs_groups: tuple[str, ...] | list[str] = ("reference",),
        encoder_dims: tuple[int, ...] | list[int] = (1024, 512, 512),
        reconstruction: bool = False,
        reconstruction_dims: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Initialize the reference-encoder model.

        Args:
            obs: A live observation dictionary; dimensions are inferred from it.
            obs_groups: Mapping of observation sets to lists of observation groups.
            obs_set: Which set to build for (``"actor"`` / ``"critic"``).
            output_dim: Dimension of the model output.
            hidden_dims: Hidden dimensions of the trunk MLP.
            activation: Activation function for the trunk, encoder and decoder.
            obs_normalization: Whether to normalize observations. Applied to the direct
                groups and, separately, to the encoder input.
            distribution_cfg: Output distribution config; ``None`` for a deterministic head.
            encoder_obs_groups: Observation groups routed through the encoder. Any group
                in this model's observation set that is NOT listed here goes straight to
                the trunk.
            encoder_dims: Encoder layer widths. The LAST entry is the latent width, the
                rest are hidden layers -- so ``(1024, 512, 512)`` builds
                ``ref_dim -> 1024 -> 512 -> 512``.
            reconstruction: Whether to build the decoder used by
                :meth:`reconstruction_loss`. Required for a non-zero reconstruction
                coefficient in :class:`~robot_rl.rl.aux_ppo.AuxPPO`.
            reconstruction_dims: Decoder hidden widths. ``None`` (the default) mirrors the
                encoder -- ``reversed(encoder_dims[:-1])``, i.e. ``512 -> 512 -> 1024 ->
                ref_dim`` for the default encoder. Deriving it keeps the decoder a true
                transpose of the encoder when the encoder is retuned.

        Raises:
            ValueError: If ``encoder_dims`` has fewer than two entries, if a named
                encoder group is not in this model's observation set, or if no direct
                group is left for the trunk.
        """
        if len(encoder_dims) < 2:
            raise ValueError(
                f"encoder_dims must hold at least one hidden width plus the latent width, "
                f"got {list(encoder_dims)}."
            )

        # Partition the groups BEFORE the parent runs: the parent sizes the trunk from
        # _get_latent_dim(), which needs the encoder latent width. (Same ordering trick
        # CNNModel uses for its CNN latent.)
        self._encoder_obs_groups = list(encoder_obs_groups)
        self.encoder_latent_dim = int(encoder_dims[-1])

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # Encoder + its own normalizer (the parent's normalizer covers the direct groups
        # only, so without this the reference would bypass normalization entirely).
        self.encoder = MLP(self.encoder_obs_dim, encoder_dims[-1], list(encoder_dims[:-1]), activation)
        if obs_normalization:
            self.encoder_obs_normalizer = EmpiricalNormalization(self.encoder_obs_dim)
        else:
            self.encoder_obs_normalizer = nn.Identity()

        # Optional reconstruction decoder, mirroring the encoder unless overridden.
        if reconstruction:
            decoder_dims = (
                list(reconstruction_dims)
                if reconstruction_dims is not None
                else list(reversed(list(encoder_dims[:-1])))
            )
            self.decoder = MLP(self.encoder_latent_dim, self.encoder_obs_dim, decoder_dims, activation)
        else:
            self.decoder = None

    # ------------------------------------------------------------------
    # Group resolution
    # ------------------------------------------------------------------
    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Split the active groups into encoder-fed and direct, and size both.

        Returns the direct groups and their total width for the parent; the encoder side
        is stashed on ``self`` (``encoder_obs_groups`` / ``encoder_obs_dim``), along with
        the flat index maps the export wrappers use to split a single concatenated
        observation vector.
        """
        active_obs_groups = obs_groups[obs_set]

        missing = [g for g in self._encoder_obs_groups if g not in active_obs_groups]
        if missing:
            raise ValueError(
                f"encoder_obs_groups {missing} are not in the '{obs_set}' observation set "
                f"{active_obs_groups}. Add them to obs_groups['{obs_set}'] or drop them "
                f"from encoder_obs_groups."
            )

        direct_groups: list[str] = []
        direct_dim = 0
        encoder_groups: list[str] = []
        encoder_dim = 0
        # Flat indices into cat(active groups, in obs_groups order) -- the layout the
        # exported ONNX/JIT model receives as its single `obs` input, and the order
        # export_parameters.py writes the observation-term table in.
        direct_index: list[int] = []
        encoder_index: list[int] = []
        offset = 0

        for obs_group in active_obs_groups:
            shape = obs[obs_group].shape
            if len(shape) != 2:
                raise ValueError(
                    f"RefEncoderMLPModel only supports 1D observations, got shape {shape} "
                    f"for '{obs_group}'."
                )
            width = shape[-1]
            span = list(range(offset, offset + width))
            if obs_group in self._encoder_obs_groups:
                encoder_groups.append(obs_group)
                encoder_dim += width
                encoder_index += span
            else:
                direct_groups.append(obs_group)
                direct_dim += width
                direct_index += span
            offset += width

        if not direct_groups:
            raise ValueError(
                f"Every group in the '{obs_set}' set is routed to the encoder; the trunk "
                "would see nothing. Leave at least one group out of encoder_obs_groups."
            )

        self.encoder_obs_groups = encoder_groups
        self.encoder_obs_dim = encoder_dim
        self.flat_obs_dim = offset
        self._direct_index = direct_index
        self._encoder_index = encoder_index
        return direct_groups, direct_dim

    def _get_latent_dim(self) -> int:
        """Return the trunk input width: direct observations plus the encoder latent."""
        return self.obs_dim + self.encoder_latent_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _encoder_input(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate and normalize the encoder-fed observation groups."""
        ref = torch.cat([obs[obs_group] for obs_group in self.encoder_obs_groups], dim=-1)
        return self.encoder_obs_normalizer(ref)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the trunk input from the direct groups plus the encoded reference."""
        latent_direct = super().get_latent(obs)
        return torch.cat([latent_direct, self.encoder(self._encoder_input(obs))], dim=-1)

    def reconstruction_loss(self, obs: TensorDict) -> torch.Tensor:
        """Mean-squared error of decoding the reference back out of its latent.

        The target is the encoder's own (normalized) input, so the objective is a plain
        autoencoder on the reference window: it pushes the latent to retain the whole
        trajectory rather than only the part the current action depends on.  Note that
        input is the observation as the manager produced it -- already noised, clipped and
        scaled -- which is the right target (reconstructing the raw value would only teach
        the decoder to undo the scaling) but has two consequences:

        * The noise is unpredictable by construction, so the loss floors at the injected
          noise variance rather than at zero.  A plateau is not the encoder giving up.
        * An MSE weights each channel by its magnitude squared, so the observation term's
          per-block ``scale`` silently decides what "reconstruct the reference" means.
          TODO(scaling): as configured, joint position accounts for ~62% of this loss and
          joint velocity ~3%; see the TODO on ``mdp.REFERENCE_WINDOW_SCALE``.  If that
          balance needs fixing, add a per-block weight vector HERE -- the observation
          scale also feeds the forward pass, so it cannot be retuned independently.

        Args:
            obs: Observation dictionary containing the encoder groups.

        Returns:
            Scalar reconstruction loss.

        Raises:
            RuntimeError: If the model was built with ``reconstruction=False``.
        """
        if self.decoder is None:
            raise RuntimeError(
                "reconstruction_loss requires the decoder. Build the actor with "
                "reconstruction=True (agent.actor.reconstruction=true) when using a "
                "non-zero reference_reconstruction_coeff."
            )
        target = self._encoder_input(obs)
        return F.mse_loss(self.decoder(self.encoder(target)), target)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update both the direct-group and the encoder-input normalizers."""
        super().update_normalization(obs)
        if self.obs_normalization:
            ref = torch.cat([obs[obs_group] for obs_group in self.encoder_obs_groups], dim=-1)
            self.encoder_obs_normalizer.update(ref)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchRefEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRefEncoderModel(self, verbose)


class _FlatRefEncoderModel(nn.Module):
    """Inference core taking ONE flat observation vector and splitting it internally.

    The deploy side assembles a single concatenated vector from the exported term table,
    so the exported graph must accept that rather than one input per observation group.
    The split is done with ``index_select`` over registered index buffers: general (the
    groups need not be contiguous or ordered) and both TorchScript- and ONNX-friendly.
    """

    def __init__(self, model: RefEncoderMLPModel) -> None:
        """Copy the inference-relevant submodules and index maps out of ``model``."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_obs_normalizer = copy.deepcopy(model.encoder_obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        # The decoder is training-only and deliberately not exported.
        self.register_buffer("direct_index", torch.as_tensor(model._direct_index, dtype=torch.long))
        self.register_buffer("encoder_index", torch.as_tensor(model._encoder_index, dtype=torch.long))
        self.input_size = model.flat_obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on the concatenated observation vector."""
        direct = self.obs_normalizer(x.index_select(1, self.direct_index))
        ref = self.encoder_obs_normalizer(x.index_select(1, self.encoder_index))
        latent = torch.cat([direct, self.encoder(ref)], dim=-1)
        return self.deterministic_output(self.mlp(latent))


class _TorchRefEncoderModel(_FlatRefEncoderModel):
    """Exportable reference-encoder model for JIT."""

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for this model)."""
        pass


class _OnnxRefEncoderModel(_FlatRefEncoderModel):
    """Exportable reference-encoder model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: RefEncoderMLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a :class:`RefEncoderMLPModel`."""
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return a representative dummy input for ONNX tracing."""
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
