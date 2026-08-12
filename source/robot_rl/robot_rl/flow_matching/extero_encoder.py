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


def conv_grid_shape(
    image_hw: tuple[int, int], num_layers: int, strides: Sequence[int] | None = None
) -> tuple[int, int]:
    """Spatial grid left by a stack of kernel-3, padding-1 convolutions.

    A stride-``s`` layer maps a side length ``n`` to ``ceil(n / s)``, so the 26x30 depth image
    goes 13x15 -> 7x8 -> 4x4 in three stride-2 layers, and a 121x121 scan goes
    121 -> 61 -> 31 -> 16 under strides ``(1, 2, 2, 2)``. Note ceil-division FLOORS at 1 rather
    than collapsing, so a deep stack on a small image degenerates to a single 1x1 cell -- a
    global pooling wearing a grid's name -- rather than failing. Check the returned grid if
    that matters; only a non-positive input is an error here.

    Args:
        image_hw: ``(height, width)`` of the input image.
        num_layers: Number of convolutions, used when ``strides`` is ``None``.
        strides: Per-layer stride. ``None`` means stride 2 on every layer, which is the
            shipped depth behaviour and keeps every existing call site exact.

    Returns:
        ``(grid_h, grid_w)`` after the stack.

    Raises:
        ValueError: If either input side is not positive, if a stride is below 1, or if
            ``strides`` is given and disagrees with ``num_layers``.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    if h < 1 or w < 1:
        raise ValueError(f"image_hw={(h, w)} must be positive on both axes.")
    if strides is None:
        per_layer = [2] * int(num_layers)
    else:
        per_layer = [int(s) for s in strides]
        if len(per_layer) != int(num_layers):
            raise ValueError(
                f"strides has {len(per_layer)} entries but there are {int(num_layers)} "
                "convolutions; one stride per layer."
            )
    for stride in per_layer:
        if stride < 1:
            raise ValueError(f"conv stride must be >= 1, got {stride}.")
        h, w = math.ceil(h / stride), math.ceil(w / stride)
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


def pixel_coords_m(
    image_hw: tuple[int, int], extent_m: tuple[float, float], ordering: str = "xy"
) -> torch.Tensor:
    """Each pixel's ``(x_forward, y_lateral)`` offset from the grid centre, in METRES.

    Built from the grid's physical extents rather than from pixel indices, so the values are
    the real thing a convolution can condition on -- the point of feeding them in at all. A
    robot-centred grid spans ``[-extent/2, +extent/2]`` inclusive on both ends, which is
    exactly what ``linspace`` over the pixel count reproduces (IsaacLab's grid pattern uses an
    inclusive arange).

    Args:
        image_hw: ``(height, width)`` of the folded image.
        extent_m: ``(size_x, size_y)`` full extents in metres.
        ordering: ``"xy"`` (rows are lateral, columns forward) or ``"yx"`` (transposed).

    Returns:
        ``(2, height, width)`` -- channel 0 is forward, channel 1 is lateral.

    Raises:
        ValueError: On an unknown ``ordering``.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    sx, sy = float(extent_m[0]), float(extent_m[1])
    if ordering == "xy":
        n_x, n_y = w, h
    elif ordering == "yx":
        n_x, n_y = h, w
    else:
        raise ValueError(f"unknown ordering {ordering!r}; expected 'xy' or 'yx'.")
    x = torch.linspace(-sx / 2.0, sx / 2.0, n_x)
    y = torch.linspace(-sy / 2.0, sy / 2.0, n_y)
    if ordering == "xy":
        x_grid = x.view(1, -1).expand(h, w)
        y_grid = y.view(-1, 1).expand(h, w)
    else:
        x_grid = x.view(-1, 1).expand(h, w)
        y_grid = y.view(1, -1).expand(h, w)
    return torch.stack([x_grid, y_grid], dim=0).contiguous()


def fourier_features(coords: torch.Tensor, bands: int) -> torch.Tensor:
    """Sinusoidal features of normalized coordinates, ``(..., 2 * axes * bands)``.

    Band ``b`` uses frequency ``2**b * pi``, so the embedding resolves position at a
    geometric range of scales rather than one -- a single linear map of ``(x, y)`` can only
    express a plane, which is too smooth to distinguish neighbouring tokens.

    Args:
        coords: ``(..., axes)`` coordinates, expected roughly in ``[-1, 1]``.
        bands: Number of frequency bands per axis. ``0`` returns the raw coordinates.

    Returns:
        ``(..., axes * 2 * bands)``, or ``coords`` unchanged when ``bands == 0``.
    """
    if bands <= 0:
        return coords
    freqs = (2.0 ** torch.arange(bands, dtype=coords.dtype, device=coords.device)) * math.pi
    scaled = coords.unsqueeze(-1) * freqs            # (..., axes, bands)
    feats = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
    return feats.flatten(-2)


class ExteroImageEncoder(nn.Module):
    """One camera's depth image -> ``(B, tokens_per_sensor, d_model)``.

    A stack of stride-2 convolutions, then a linear projection of each surviving grid cell
    up to the transformer width -- the same "embed the raw width up to ``d_model``"
    treatment every other stream gets, applied per cell instead of per frame.

    A metric height scan additionally gets **positional information**, which a plain
    convolution structurally cannot have: a shared kernel is translation-*equivariant*, so it
    can report that a step edge is present but not WHERE it is -- and "where, relative to the
    robot" is the entire quantity the generator is failing to register. When ``coord_extent_m``
    is given, each pixel's ``(x, y)`` offset from the robot in METRES is appended as two input
    channels (CoordConv), and each output token additionally receives a Fourier embedding of
    its own cell-centre coordinate. Depth cameras pass ``None`` and are bit-identical to
    before -- no extra channels, no extra parameters.

    Args:
        image_hw: ``(height, width)`` of this sensor's image.
        d_model: Transformer hidden width to project each cell to.
        channels: Output channels per convolution; ``len`` sets the depth of the stack.
        grid: Optional ``(grid_h, grid_w)`` to average-pool the trunk's output down to,
            which decouples the token count from the input resolution. ``None`` keeps
            whatever the trunk produces.
        strides: Per-layer convolution stride. ``None`` means stride 2 everywhere (the depth
            behaviour). A leading stride-1 layer buys a full-resolution first stage, which is
            how the dense scan keeps 0.025 m detail while still landing on 16x16 tokens.
        coord_extent_m: ``(size_x, size_y)`` full extents in metres of a robot-centred metric
            grid. Enables the coordinate channels and the token position embedding.
        ordering: ``GridPatternCfg.ordering`` of the flat row, deciding which image axis
            carries forward and which carries lateral.
        fourier_bands: Frequency bands per axis in the token position embedding.
        miss_sentinel: The exact normalized value a MISSED ray takes. When given, a validity
            channel is appended: 1 where the pixel is real geometry, 0 where the reading was
            a miss or saturated at the band edge. Without it a hole and a real drop-off are
            the same number, and the convolution cannot tell them apart.
    """

    def __init__(
        self,
        image_hw: tuple[int, int],
        d_model: int,
        channels: Sequence[int] = (16, 32, 64),
        grid: tuple[int, int] | None = None,
        strides: Sequence[int] | None = None,
        coord_extent_m: tuple[float, float] | None = None,
        ordering: str = "xy",
        fourier_bands: int = 4,
        miss_sentinel: float | None = None,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels:
            raise ValueError("channels must name at least one convolution.")
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.sensor_dim = self.image_hw[0] * self.image_hw[1]
        self.strides = ((2,) * len(channels) if strides is None
                        else tuple(int(s) for s in strides))
        if len(self.strides) != len(channels):
            raise ValueError(
                f"strides has {len(self.strides)} entries but channels names "
                f"{len(channels)} convolutions; one stride per layer."
            )
        self.miss_sentinel = None if miss_sentinel is None else float(miss_sentinel)
        self.coord_extent_m = (None if coord_extent_m is None
                               else (float(coord_extent_m[0]), float(coord_extent_m[1])))

        # Input channels: the reading, plus (x, y) metres, plus validity -- each only when
        # asked for, so the depth path's Conv2d(1, ...) weights are unchanged.
        in_ch = 1 + (2 if self.coord_extent_m is not None else 0)
        in_ch += 1 if self.miss_sentinel is not None else 0
        self.in_channels = in_ch

        layers: list[nn.Module] = []
        prev = in_ch
        for out_ch, stride in zip(channels, self.strides):
            layers += [nn.Conv2d(prev, out_ch, kernel_size=3, stride=stride, padding=1),
                       nn.SiLU()]
            prev = out_ch
        # SiLU after EVERY convolution including the last: without it the final conv and the
        # projection below compose into a single linear map and one layer is wasted.
        self.trunk = nn.Sequential(*layers)
        self.proj = nn.Linear(channels[-1], d_model)

        native = conv_grid_shape(self.image_hw, len(channels), self.strides)
        self.grid = None if grid is None else (int(grid[0]), int(grid[1]))
        out_grid = native if self.grid is None else self.grid
        self.native_grid = native
        self.out_grid = out_grid
        self.tokens_per_sensor = out_grid[0] * out_grid[1]

        if self.coord_extent_m is None:
            self.pos_embed = None
            self.fourier_bands = 0
        else:
            self.fourier_bands = int(fourier_bands)
            coords = pixel_coords_m(self.image_hw, self.coord_extent_m, ordering)
            # (1, 2, H, W), in metres -- concatenated to the reading as CoordConv channels.
            self.register_buffer("coord_channels", coords.unsqueeze(0), persistent=False)
            # Each token's cell-MEAN coordinate, obtained by pooling the same grid to the
            # output shape. Derived, never typed: this is exact for any stride/resolution
            # combination, where analytic stride/padding arithmetic is easy to get wrong.
            token_xy = F.adaptive_avg_pool2d(coords.unsqueeze(0), out_grid)
            token_xy = token_xy.flatten(2).transpose(1, 2)          # (1, tokens, 2)
            self.register_buffer("token_xy_m", token_xy, persistent=False)
            half = torch.tensor([self.coord_extent_m[0] / 2.0, self.coord_extent_m[1] / 2.0])
            feats = fourier_features(token_xy / half, self.fourier_bands)
            self.register_buffer("token_pos_features", feats, persistent=False)
            self.pos_embed = nn.Linear(feats.shape[-1], d_model)

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        """Encode one sensor's flat rows.

        Args:
            flat: ``(B, height * width)`` normalized reading for this sensor.

        Returns:
            ``(B, tokens_per_sensor, d_model)``, cells in row-major order.
        """
        img = images_from_flat(flat, self.image_hw)
        extra = []
        if self.miss_sentinel is not None:
            # Exact equality is meaningful here: sanitize clamps a miss to precisely the band
            # edge and the metric affine is a fixed constant, so the sentinel arrives as an
            # exact value. Real readings saturated at the same edge are also marked invalid,
            # which is intended -- both are "no usable geometry here".
            extra.append((~torch.isclose(
                img, torch.full_like(img, self.miss_sentinel), atol=1e-6)).to(img.dtype))
        if self.coord_extent_m is not None:
            extra.append(self.coord_channels.expand(img.shape[0], -1, -1, -1).to(img.dtype))
        if extra:
            img = torch.cat([img, *extra], dim=1)
        x = self.trunk(img)
        if self.grid is not None:
            # Pool BEFORE the projection, so a smaller grid buys cheaper projection too.
            x = F.adaptive_avg_pool2d(x, self.grid)
        x = x.flatten(2).transpose(1, 2)            # (B, cells, channels), row-major cells
        tokens = self.proj(x)
        if self.pos_embed is not None:
            # Metric position, added rather than concatenated so the token width is unchanged.
            tokens = tokens + self.pos_embed(self.token_pos_features.to(tokens.dtype))
        return tokens


class ExteroImageEncoderSet(nn.Module):
    """Per-camera :class:`ExteroImageEncoder` trunks plus one shared spatial-slot embedding.

    The slot embedding is shared across cameras because a slot means the same *relative*
    place in a frame whichever camera produced it ("upper-left of my view"); which camera
    that is comes from the model's own ``extero_id``. The trunks are not shared -- see the
    module docstring.

    Args:
        image_hw: One ``(height, width)`` per sensor, in token order.
        d_model: Transformer hidden width.
        channels: Output channels per convolution, shared as a SHAPE by every
            camera (the weights are separate).
        grid: Optional pooled ``(grid_h, grid_w)`` per camera.
        strides: Per-layer convolution stride, shared by every sensor.
        coord_extent_m: ``(size_x, size_y)`` metric extents for a metric grid, enabling the
            coordinate channels and token position embedding. ``None`` for depth.
        ordering: Flat-row ordering of the metric grid.
        fourier_bands: Frequency bands per axis in the token position embedding.
        miss_sentinel: Normalized value a missed ray takes, enabling the validity channel.

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
        strides: Sequence[int] | None = None,
        coord_extent_m: tuple[float, float] | None = None,
        ordering: str = "xy",
        fourier_bands: int = 4,
        miss_sentinel: float | None = None,
    ):
        super().__init__()
        if not len(image_hw):
            raise ValueError("ExteroImageEncoderSet needs at least one sensor.")
        self.encoders = nn.ModuleList([
            ExteroImageEncoder(
                hw, d_model, channels, grid, strides=strides,
                coord_extent_m=coord_extent_m, ordering=ordering,
                fourier_bands=fourier_bands, miss_sentinel=miss_sentinel,
            )
            for hw in image_hw
        ])
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
