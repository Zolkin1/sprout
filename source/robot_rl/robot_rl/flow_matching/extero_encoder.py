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
``transfer/deploy/depth_obs.pinhole_ray_directions`` and its unprojection -- but
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


def images_from_flat(
    flat: torch.Tensor, image_hw: tuple[int, int], in_channels: int = 1
) -> torch.Tensor:
    """Fold flat sensor rows into images for a convolution.

    The row is CHANNEL-MAJOR: all of channel 0's pixels, then all of channel 1's, each
    row-major over ``(height, width)``. That is the layout ``conv2d`` wants and the one
    ``mdp.observations.depth.depth_xyz_stack`` and ``deploy.depth_obs`` both emit -- an
    interleaved row would fold into a scrambled image with every shape still correct.

    Args:
        flat: ``(..., in_channels * height * width)`` rows, row-major with ``v = 0`` the top
            row (see the module docstring -- this ordering is the contract, not a
            convention).
        image_hw: ``(height, width)`` the rows unfold to.
        in_channels: Planes per pixel. 1 = depth only; 4 = depth plus the pixel's xyz in the
            root's yaw-aligned frame.

    Returns:
        ``(N, in_channels, height, width)`` where ``N`` is the flattened leading dimensions.

    Raises:
        ValueError: If the row width is not ``in_channels * height * width``.
    """
    h, w, c = int(image_hw[0]), int(image_hw[1]), int(in_channels)
    if flat.shape[-1] != c * h * w:
        raise ValueError(
            f"exteroception row is {flat.shape[-1]} wide but image_hw={(h, w)} with "
            f"{c} channel(s) needs {c * h * w}."
        )
    # ``-1`` rather than an explicit product: the legacy ONNX tracer then emits a Reshape
    # against a constant shape instead of a Shape/Gather/Mul chain, which is what keeps the
    # graph's batch axis dynamic (see export_fm_onnx's note on the dynamo path).
    return flat.reshape(-1, c, h, w)


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


def pixel_angles_rad(
    image_hw: tuple[int, int], fx: float, fy: float, cx: float, cy: float
) -> torch.Tensor:
    """Each pixel's ``(azimuth, elevation)`` from the optical axis, in RADIANS.

    The camera's answer to :func:`pixel_coords_m`. A depth pixel is a RAY, not a ground
    point, so it has no planar coordinate to hand a convolution -- but it does have a
    direction, and ``atan((u - cx) / fx)`` is that direction exactly, for the tangent-linear
    pinhole the training sensor uses.

    Why angles and not pixel indices, which carry the same information for a fixed lens: the
    angle is a property of the LENS rather than of the sampling, so it is unchanged when the
    generator re-samples the same camera on its own grid, and it is directly comparable
    between the G1's two cameras -- which are 100x70 deg and 70x42 deg, so the same pixel
    index means quite different directions in each.

    Args:
        image_hw: ``(height, width)`` of the folded image.
        fx: Horizontal focal length in pixels.
        fy: Vertical focal length in pixels.
        cx: Horizontal principal point in pixels.
        cy: Vertical principal point in pixels.

    Returns:
        ``(2, height, width)`` -- channel 0 is azimuth (+ toward image left, matching the
        camera frame's +y), channel 1 elevation (+ toward the top row, matching +z).
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    u = torch.arange(w, dtype=torch.float32) + 0.5
    v = torch.arange(h, dtype=torch.float32) + 0.5
    az = torch.atan((float(cx) - u) / float(fx)).view(1, -1).expand(h, w)
    el = torch.atan((float(cy) - v) / float(fy)).view(-1, 1).expand(h, w)
    return torch.stack([az, el], dim=0).contiguous()


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
        intrinsics: ``(fx, fy, cx, cy)`` of THIS camera, enabling the angular token position
            embedding. Mutually exclusive with ``coord_extent_m``; per sensor, because the
            G1's two cameras are different lenses and one shared extent would mislabel the
            narrower one's tokens.
        row_channels: Planes per pixel carried by the ROW itself (``cfg.extero_channels``).
            1 = depth alone; 4 = depth plus the pixel's xyz in the root's yaw-aligned frame,
            which is what lets a kernel see where a step edge is in metres rather than in
            pixels. Distinct from ``in_channels``, which counts these PLUS the planes the
            encoder derives for itself (coordinates, validity) and that never travel in the
            row.
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
        row_channels: int = 1,
        intrinsics: tuple[float, float, float, float] | None = None,
    ):
        super().__init__()
        channels = tuple(int(c) for c in channels)
        if not channels:
            raise ValueError("channels must name at least one convolution.")
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
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
        self.intrinsics = (None if intrinsics is None
                           else tuple(float(v) for v in intrinsics))
        if self.coord_extent_m is not None and self.intrinsics is not None:
            raise ValueError(
                "a sensor is either a metric ground grid (coord_extent_m) or a camera "
                "(intrinsics), not both; they parameterise position in different units."
            )

        # TWO channel counts, and conflating them is a silent shape bug:
        #   row_channels -- what the ROW carries, so what ``cfg.extero_sensor_dim`` counts and
        #                   what ``images_from_flat`` folds. The producer decides it.
        #   in_channels  -- what the CONVOLUTION sees: the row's planes plus the ones this
        #                   encoder derives itself (metric coordinates, validity), which are
        #                   computed here and never travel in the row.
        self.row_channels = int(row_channels)
        in_ch = self.row_channels + (2 if self.coord_extent_m is not None else 0)
        in_ch += 1 if self.miss_sentinel is not None else 0
        self.in_channels = in_ch
        # The row width the model receives, channels included -- what ``cfg.extero_sensor_dim``
        # counts, so a mismatch shows up here rather than inside the convolution.
        self.sensor_dim = self.row_channels * self.image_hw[0] * self.image_hw[1]

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
        if self.grid is not None and (self.grid[0] > native[0] or self.grid[1] > native[1]):
            # ``adaptive_avg_pool2d`` UPSAMPLES silently when asked for more cells than it
            # has, so the run would report (and pay for) tokens carrying no more information
            # than the smaller grid held -- a token count that is a claim, not a fact.
            raise ValueError(
                f"extero_cnn_grid {self.grid} is larger than the trunk's native grid "
                f"{native} for a {self.image_hw} image under strides {self.strides}; "
                "pooling up would invent tokens. Lower a stride to make the native grid "
                "bigger, or ask for a grid no larger than it."
            )
        out_grid = native if self.grid is None else self.grid
        self.native_grid = native
        self.out_grid = out_grid
        self.tokens_per_sensor = out_grid[0] * out_grid[1]

        if self.coord_extent_m is None and self.intrinsics is None:
            self.pos_embed = None
            self.fourier_bands = 0
        elif self.coord_extent_m is None:
            # CAMERA path: the position embedding is TOKEN-LEVEL only -- no input channels.
            # A ray has a direction rather than a planar coordinate, and the direction is
            # constant per pixel, so stating it once per token is where it belongs; feeding it
            # as a CoordConv plane would spend two input channels restating what the token
            # embedding already says. (The metric grid below does add channels, because there
            # a pixel IS a ground point and its coordinate is what the kernel localises.)
            self.fourier_bands = int(fourier_bands)
            ang = pixel_angles_rad(self.image_hw, *self.intrinsics)          # (2, H, W)
            token_ang = F.adaptive_avg_pool2d(ang.unsqueeze(0), out_grid)
            token_ang = token_ang.flatten(2).transpose(1, 2)                 # (1, tokens, 2)
            self.register_buffer("token_angles_rad", token_ang, persistent=False)
            # Normalised by this camera's OWN half-angles, so both cameras' embeddings span
            # the same [-1, 1] whatever their field of view -- otherwise the narrow one would
            # only ever excite the low-frequency bands.
            half = ang.flatten(1).abs().amax(dim=1).clamp_min(1e-6)          # (2,)
            feats = fourier_features(token_ang / half, self.fourier_bands)
            self.register_buffer("token_pos_features", feats, persistent=False)
            self.pos_embed = nn.Linear(feats.shape[-1], d_model)
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
            flat: ``(B, row_channels * height * width)`` normalized reading for this sensor,
                folded CHANNEL-MAJOR (all of plane 0, then plane 1, ...).

        Returns:
            ``(B, tokens_per_sensor, d_model)``, cells in row-major order.
        """
        img = images_from_flat(flat, self.image_hw, self.row_channels)
        extra = []
        if self.miss_sentinel is not None:
            # Tested against the DEPTH plane alone: the xyz planes are metres in the
            # root frame and have no sentinel, so comparing them against one would
            # mark ordinary geometry invalid wherever a coordinate equalled it.
            #
            # Exact equality is meaningful here: sanitize clamps a miss to precisely the band
            # edge and the metric affine is a fixed constant, so the sentinel arrives as an
            # exact value. Real readings saturated at the same edge are also marked invalid,
            # which is intended -- both are "no usable geometry here".
            #
            # Written out rather than ``torch.isclose``: that operator has no ONNX lowering
            # at opset 18 (``UnsupportedOperatorError: aten::isclose``), so the whole
            # generator failed to EXPORT the moment this channel existed -- a training-time
            # detail that silently made a model undeployable. ``other`` is a constant, so
            # isclose's band ``atol + rtol * |other|`` is itself constant and the two agree
            # exactly on every finite input. They differ only on NaN (isclose says
            # "not close" -> valid, this says invalid), which cannot arrive: every reading is
            # sanitised to a finite value on both the dataset and the deploy side.
            tol = 1e-6 + 1e-5 * abs(float(self.miss_sentinel))
            extra.append(((img[:, :1] - float(self.miss_sentinel)).abs() > tol).to(img.dtype))
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
        row_channels: Planes per pixel in the row; see :class:`ExteroImageEncoder`.
        intrinsics: One ``(fx, fy, cx, cy)`` PER SENSOR, or ``None``. Per sensor rather than
            shared: the G1's cameras are 100x70 deg and 70x42 deg, so a single extent would
            label the far camera's tokens with angles 1.43x too wide -- silently, since every
            shape stays correct.

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
        row_channels: int = 1,
        intrinsics: Sequence[Sequence[float]] | None = None,
    ):
        super().__init__()
        if not len(image_hw):
            raise ValueError("ExteroImageEncoderSet needs at least one sensor.")
        self.row_channels = int(row_channels)
        if intrinsics is not None and len(intrinsics) != len(image_hw):
            raise ValueError(
                f"intrinsics names {len(intrinsics)} camera(s) but there are "
                f"{len(image_hw)} sensor(s); they are per sensor, not shared."
            )
        self.encoders = nn.ModuleList([
            ExteroImageEncoder(
                hw, d_model, channels, grid, strides=strides,
                coord_extent_m=coord_extent_m, ordering=ordering,
                fourier_bands=fourier_bands, miss_sentinel=miss_sentinel,
                row_channels=row_channels,
                intrinsics=(None if intrinsics is None else intrinsics[i]),
            )
            for i, hw in enumerate(image_hw)
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
