# Copyright 2026 Zachary Olkin. All rights reserved.

"""Depth-camera noise: plane tilt + per-pixel noise + Gaussian blur + dropout.

Ported from legged_locomotion_rl.  Deliberately NOT an IsaacLab
:class:`~isaaclab.utils.noise.noise_model.NoiseModel`: attaching a custom noise
model as an :class:`~isaaclab.managers.ObservationTermCfg` ``noise`` field is
broken for us, because hydra's ``class_to_dict`` / ``update_class_from_dict``
round-trip rewrites ``class_type`` into a :class:`~isaaclab.utils.string.ResolvableString`
(a ``str`` subclass) and ``observation_manager.py`` then calls
``issubclass(class_type, NoiseModel)`` on it without resolving, raising
``TypeError: issubclass() arg 1 must be a class``.

Instead the noiser is driven explicitly:

- :func:`~robot_rl.tasks.manager_based.robot_rl.mdp.events.depth.resample_depth_noise`
  (an event term with ``mode="reset"``) builds it lazily and re-samples the
  per-episode plane bias and blur kernel for the resetting envs.
- :func:`~robot_rl.tasks.manager_based.robot_rl.mdp.observations.depth.depth_image`
  applies whatever noiser is registered for its sensor.

Keeping the re-sample on the reset event -- rather than every step -- preserves the
intended semantics: plane tilt and blur sigma are per-episode camera properties,
while per-pixel noise and dropout are re-drawn every frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
import torch.nn.functional as F

from isaaclab.utils import configclass
from isaaclab.utils.noise.noise_cfg import NoiseCfg, UniformNoiseCfg

# Attribute under which the per-sensor noisers are cached on the env.
_ENV_ATTR = "_robot_rl_depth_noisers"


@configclass
class DepthNoiseCfg:
    """Parameters for :class:`DepthNoiser`."""

    noise_cfg: NoiseCfg = UniformNoiseCfg(n_min=-0.015, n_max=0.015, operation="add")
    """Per-pixel noise applied after the plane bias."""

    max_plane_deviation: float = 0.1
    """Maximum absolute value of the per-env plane bias ``a*x + b*y + c``."""

    sigma_noise_cfg: NoiseCfg = UniformNoiseCfg(n_min=0.0, n_max=0.33, operation="abs")
    """Distribution the Gaussian-blur sigma is drawn from, per env, on reset."""

    dropout_prob: float = 0.025
    """Per-pixel dropout probability."""

    dropout_val: float = 0.0
    """Pre-normalisation value assigned to dropped pixels."""

    height: int = MISSING
    """Image height in pixels (must match the depth camera)."""

    width: int = MISSING
    """Image width in pixels (must match the depth camera)."""

    offset: float = MISSING
    """``offset`` from the matching observation term, used to normalise the dropout value."""

    scale: float = MISSING
    """``scale`` from the matching observation term, used to normalise the dropout value."""


class DepthNoiser:
    """Composite depth-image noise with per-episode camera state.

    Per-episode, re-sampled by :meth:`reset`:

    1. Plane bias ``a*x + b*y + c`` whose peak magnitude is drawn uniformly from
       ``[0, max_plane_deviation]`` -- a mis-calibrated depth plane.
    2. A 3x3 Gaussian blur whose ``sigma`` is drawn from ``sigma_noise_cfg``.

    Per frame, in :meth:`__call__`:

    3. Per-pixel noise from ``noise_cfg``.
    4. Random pixel dropout, set to ``(dropout_val - offset) * scale``.

    Accepts flat ``(B, H*W)`` or image ``(B, 1, H, W)`` input and returns the same
    shape.
    """

    def __init__(self, cfg: DepthNoiseCfg, num_envs: int, device: str):
        """Allocate the per-env noise state.

        Args:
            cfg: Noise parameters.
            num_envs: Number of environments.
            device: Torch device the buffers live on.
        """
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._H = cfg.height
        self._W = cfg.width

        self._sigma = torch.zeros((num_envs, 1), device=device)

        ax = torch.arange(-1, 2, dtype=torch.float, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        self._dist_sq = (xx ** 2 + yy ** 2).reshape(1, 9)
        self._kernel = torch.zeros((num_envs, 1, 9), device=device)

        ys = torch.linspace(-1, 1, self._H, device=device)
        xs = torch.linspace(-1, 1, self._W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self._grid_x = xx.reshape(1, -1)
        self._grid_y = yy.reshape(1, -1)
        self._plane = torch.zeros((num_envs, self._H * self._W), device=device)
        self._plane_coeffs = torch.zeros((num_envs, 3), device=device)

        self.reset()

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Re-sample the per-episode blur kernel and plane bias.

        Args:
            env_ids: Environments to re-sample. ``None`` re-samples all.
        """
        if env_ids is None:
            env_ids = slice(None)

        # Blur kernel.
        self._sigma[env_ids] = self.cfg.sigma_noise_cfg.func(self._sigma[env_ids], self.cfg.sigma_noise_cfg)
        sigma_sq = (self._sigma[env_ids] ** 2).unsqueeze(1).clamp(min=1e-12)
        kernel = torch.exp(-self._dist_sq / (2 * sigma_sq))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)
        self._kernel[env_ids] = kernel

        # Plane coefficients, rescaled so the peak deviation matches the target.
        n = self._plane[env_ids].shape[0]
        coeffs = torch.randn(n, 3, device=self.device)
        max_dev = coeffs.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target = torch.rand(n, 1, device=self.device) * self.cfg.max_plane_deviation
        coeffs = coeffs * (target / max_dev)
        self._plane_coeffs[env_ids] = coeffs
        a, b, c = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
        self._plane[env_ids] = a * self._grid_x + b * self._grid_y + c

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to a normalised depth image.

        Args:
            data: ``(B, H*W)`` or ``(B, 1, H, W)`` normalised depth.

        Returns:
            The corrupted depth, same shape as ``data``.
        """
        is_4d = data.dim() == 4
        if is_4d:
            orig_shape = data.shape
            data = data.reshape(data.shape[0], -1)

        data = data + self._plane
        data = self.cfg.noise_cfg.func(data, self.cfg.noise_cfg)

        # 3x3 Gaussian blur, per-env via a grouped conv.
        data = F.conv2d(
            data.reshape(1, self.num_envs, self._H, self._W),
            self._kernel.reshape(self.num_envs, 1, 3, 3),
            padding=1,
            groups=self.num_envs,
        ).reshape(self.num_envs, -1)

        mask = torch.rand(data.shape, device=self.device) < self.cfg.dropout_prob
        data[mask] = (self.cfg.dropout_val - self.cfg.offset) * self.cfg.scale

        if is_4d:
            data = data.reshape(orig_shape)
        return data


def ensure_depth_noiser(env, name: str, cfg: DepthNoiseCfg) -> DepthNoiser:
    """Fetch the noiser registered for ``name``, constructing it on first use.

    Args:
        env: The environment instance.
        name: Scene name of the depth camera the noiser belongs to.
        cfg: Noise parameters, used only when constructing.

    Returns:
        The cached :class:`DepthNoiser`.
    """
    registry = getattr(env, _ENV_ATTR, None)
    if registry is None:
        registry = {}
        setattr(env, _ENV_ATTR, registry)
    if name not in registry:
        registry[name] = DepthNoiser(cfg, env.num_envs, env.device)
    return registry[name]


def get_depth_noiser(env, name: str) -> DepthNoiser | None:
    """Return the noiser registered for ``name``, or ``None`` if there is none.

    Args:
        env: The environment instance.
        name: Scene name of the depth camera.

    Returns:
        The registered noiser, or ``None`` when noise is disabled for this sensor.
    """
    return getattr(env, _ENV_ATTR, {}).get(name)
