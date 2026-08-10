# Copyright 2026 Zachary Olkin. All rights reserved.

"""Convolutional exteroception encoder: one depth image -> a GRID of transformer tokens.

The flat alternative (``_token_mlp(sensor_dim, d_model)``, one token per camera) is
permutation-invariant over pixels: a 780-wide ``Linear`` learns a weight per pixel and the
transformer receives one vector with no notion of *where* in the frame anything is. That is
the shape of the measured failure this module exists to address -- displace the robot
downrange and the generator's predicted stair segment slides with it (0.098 / 0.168 m for a
0.10 / 0.20 m shift, where the anchor mask requires 0.000; see
``scripts/rsl_rl/fm_registration_probe.py``). Localising terrain is a spatial question and
the embedding threw the spatial structure away.

So: three stride-2 convolutions reduce the image to a small grid, each cell becomes one
token carrying a learned slot embedding, and attention gets to weight "under the foot"
against "two metres ahead" within a single frame.

**One trunk per camera, not shared.** The two G1 cameras are deliberately different optics
-- ``_NEAR`` samples 100x70 deg pitched 65 deg down (the ground about to be stepped on),
``_FAR`` 70x42 deg at 35 deg (downrange) -- so a pixel means a different solid angle in
each, and a shared trunk would have to compromise. At ~23k parameters per trunk the
separation is close to free; the whole convolutional branch is still ~5x smaller than the
one ``Linear(780, 512)`` it replaces.

**The flat row is row-major over (H, W)**, ``k = v*W + u`` with ``v = 0`` the top row. Every
producer agrees -- ``mdp.observations.depth.depth_image`` reshaping IsaacLab's ``(B, H, W)``,
``transfer/sim/depth_camera.pinhole_ray_directions``, ``transfer/deploy/depth_obs`` -- but
that agreement used to be merely tidy. A convolution makes it load-bearing: transpose it and
every kernel sees a scrambled neighbourhood while the shapes stay right and the loss still
falls. :func:`images_from_flat` is the single place that folding happens.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_grid_shape(image_hw: tuple[int, int], num_layers: int) -> tuple[int, int]:
    """Spatial grid left by ``num_layers`` stride-2, kernel-3, padding-1 convolutions.

    Each such layer maps a side length ``n`` to ``ceil(n / 2)``, so the 26x30 depth image
    goes 13x15 -> 7x8 -> 4x4 in three layers. Note ceil-halving FLOORS at 1 rather than
    collapsing, so a deep stack on a small image degenerates to a single 1x1 cell -- a
    global pooling wearing a grid's name -- rather than failing. Check the returned grid if
    that matters; only a non-positive input is an error here.

    Args:
        image_hw: ``(height, width)`` of the input image.
        num_layers: Number of stride-2 convolutions.

    Returns:
        ``(grid_h, grid_w)`` after the stack.

    Raises:
        ValueError: If either input side is not positive.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    if h < 1 or w < 1:
        raise ValueError(f"image_hw={(h, w)} must be positive on both axes.")
    for _ in range(int(num_layers)):
        h, w = math.ceil(h / 2), math.ceil(w / 2)
    return h, w


def images_from_flat(flat: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    """Fold flat sensor rows into single-channel images for a convolution.

    Args:
        flat: ``(..., height * width)`` rows, row-major with ``v = 0`` the top row (see the
            module docstring -- this ordering is the contract, not a convention).
        image_hw: ``(height, width)`` the rows unfold to.

    Returns:
        ``(N, 1, height, width)`` where ``N`` is the flattened leading dimensions.

    Raises:
        ValueError: If the row width is not ``height * width``.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    if flat.shape[-1] != h * w:
        raise ValueError(
            f"exteroception row is {flat.shape[-1]} wide but image_hw={(h, w)} needs {h * w}."
        )
    # ``-1`` rather than an explicit product: the legacy ONNX tracer then emits a Reshape
    # against a constant shape instead of a Shape/Gather/Mul chain, which is what keeps the
    # graph's batch axis dynamic (see export_fm_onnx's note on the dynamo path).
    return flat.reshape(-1, 1, h, w)


class ExteroImageEncoder(nn.Module):
    """One camera's depth image -> ``(B, tokens_per_sensor, d_model)``.

    A stack of stride-2 convolutions, then a linear projection of each surviving grid cell
    up to the transformer width -- the same "embed the raw width up to ``d_model``"
    treatment every other stream gets, applied per cell instead of per frame.

    Args:
        image_hw: ``(height, width)`` of this camera's image.
        d_model: Transformer hidden width to project each cell to.
        channels: Output channels per stride-2 convolution; ``len`` sets the depth, so it
            also sets how many times the image is halved.
        grid: Optional ``(grid_h, grid_w)`` to average-pool the trunk's output down to,
            which decouples the token count from the input resolution. ``None`` keeps
            whatever the trunk produces.
    """

    def __init__(
        self,
        image_hw: tuple[int, int],
        d_model: int,
        channels: Sequence[int] = (16, 32, 64),
        grid: tuple[int, int] | None = None,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels:
            raise ValueError("channels must name at least one convolution.")
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.sensor_dim = self.image_hw[0] * self.image_hw[1]

        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1), nn.SiLU()]
            in_ch = out_ch
        # SiLU after EVERY convolution including the last: without it the final conv and the
        # projection below compose into a single linear map and one layer is wasted.
        self.trunk = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], d_model)

        native = conv_grid_shape(self.image_hw, len(channels))
        self.grid = None if grid is None else (int(grid[0]), int(grid[1]))
        out_grid = native if self.grid is None else self.grid
        self.native_grid = native
        self.tokens_per_sensor = out_grid[0] * out_grid[1]

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        """Encode one camera's flat rows.

        Args:
            flat: ``(B, height * width)`` normalized depth for this camera.

        Returns:
            ``(B, tokens_per_sensor, d_model)``, cells in row-major order.
        """
        x = self.trunk(images_from_flat(flat, self.image_hw))
        if self.grid is not None:
            # Pool BEFORE the projection, so a smaller grid buys cheaper projection too.
            x = F.adaptive_avg_pool2d(x, self.grid)
        x = x.flatten(2).transpose(1, 2)            # (B, cells, channels), row-major cells
        return self.proj(x)


class ExteroImageEncoderSet(nn.Module):
    """Per-camera :class:`ExteroImageEncoder` trunks plus one shared spatial-slot embedding.

    The slot embedding is shared across cameras because a slot means the same *relative*
    place in a frame whichever camera produced it ("upper-left of my view"); which camera
    that is comes from the model's own ``extero_id``. The trunks are not shared -- see the
    module docstring.

    Args:
        image_hw: One ``(height, width)`` per sensor, in token order.
        d_model: Transformer hidden width.
        channels: Output channels per stride-2 convolution, shared as a SHAPE by every
            camera (the weights are separate).
        grid: Optional pooled ``(grid_h, grid_w)`` per camera.

    Raises:
        ValueError: If no sensors are given, or if the cameras do not agree on how many
            tokens they produce -- they share one slot embedding and one ``(B, T, S, d)``
            tensor, so a ragged token count has nowhere to go. Pool to an explicit ``grid``
            to reconcile cameras of different resolutions.
    """

    def __init__(
        self,
        image_hw: Sequence[tuple[int, int]],
        d_model: int,
        channels: Sequence[int] = (16, 32, 64),
        grid: tuple[int, int] | None = None,
    ):
        super().__init__()
        if not len(image_hw):
            raise ValueError("ExteroImageEncoderSet needs at least one sensor.")
        self.encoders = nn.ModuleList(
            [ExteroImageEncoder(hw, d_model, channels, grid) for hw in image_hw]
        )
        counts = {int(enc.tokens_per_sensor) for enc in self.encoders}
        if len(counts) != 1:
            raise ValueError(
                f"cameras produce different token counts {sorted(counts)} from image shapes "
                f"{[tuple(e.image_hw) for e in self.encoders]}; they share one spatial "
                "embedding, so pass an explicit grid= to pool them to a common one."
            )
        self.tokens_per_sensor = counts.pop()
        self.d_model = int(d_model)
        # Zero-init, like every other learned embedding in ``model.py``: at init the encoder
        # is exactly its convolutional output and the slot embedding only earns its way in.
        self.spatial = nn.Parameter(torch.zeros(1, 1, self.tokens_per_sensor, d_model))

    @property
    def num_sensors(self) -> int:
        """Number of sensors (cameras) this set encodes."""
        return len(self.encoders)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Encode every sensor's image into its own grid of tokens.

        Args:
            rows: ``(B, T, sensor_dim)`` normalized exteroception, one row per sensor in
                token order.

        Returns:
            ``(B, T, tokens_per_sensor, d_model)`` with the spatial-slot embedding added.

        Raises:
            ValueError: If the sensor count does not match the number of trunks.
        """
        if rows.shape[1] != len(self.encoders):
            raise ValueError(
                f"got {rows.shape[1]} sensor row(s) but this set has {len(self.encoders)} "
                "per-camera encoder(s)."
            )
        per_sensor = [enc(rows[:, i]) for i, enc in enumerate(self.encoders)]
        return torch.stack(per_sensor, dim=1) + self.spatial
