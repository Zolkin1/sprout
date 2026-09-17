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

**Every frame is drawn ONCE and shared by every consumer of that camera.**  One physical
camera is now read by more than one observation term in the same control step -- the policy
reads the near camera, the generator's ``fm_depth`` group reads both, and
``mdp.depth_xyz_stack`` reads it again to unproject it -- where the hardware hands all of
them literally one image.  Redrawing per call would put an independent 2.5% dropout pattern
and an independent additive draw between two views of one lens, and would make
``--dump_replay`` incoherent, since the C++ runtime rebuilds the generator's extero row from
the tracker's recorded ``depth_raw``.  See :meth:`DepthNoiser.__call__`.
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

#: Identity element of each ``NoiseCfg.operation``, used to REALISE one frame's draw.
#:
#: Every ``isaaclab.utils.noise`` function is ``data <op> r`` for a single elementwise draw
#: ``r`` -- ``data + r`` (add), ``data * r`` (scale), ``r`` (abs) -- so evaluating the cfg's
#: own ``func`` on the operation's identity hands back ``r`` itself.  That is what lets a
#: frame's noise be drawn once and applied to several readings of the same camera without
#: restating the DISTRIBUTION here: its parameters stay on the cfg, where the task set them,
#: and only the COMBINATION rule (three lines below) is written out.
_NOISE_IDENTITY: dict[str, float] = {"add": 0.0, "scale": 1.0, "abs": 0.0}


def _draw_noise(cfg: NoiseCfg, like: torch.Tensor) -> torch.Tensor:
    """Realise one frame's per-pixel noise, shaped like ``like``.

    Args:
        cfg: The noise configuration to draw from.
        like: Tensor whose shape, device and dtype the draw takes.

    Returns:
        The draw ``r``, same shape as ``like``.

    Raises:
        ValueError: If ``cfg.operation`` is not one this module can decompose.
    """
    identity = _NOISE_IDENTITY.get(cfg.operation)
    if identity is None:
        raise ValueError(
            f"depth noise operation '{cfg.operation}' is not one of "
            f"{sorted(_NOISE_IDENTITY)}; the per-frame draw is recovered by evaluating the "
            "noise on its operation's identity, which needs the operation to be known."
        )
    return cfg.func(torch.full_like(like, identity), cfg)


def _apply_noise(cfg: NoiseCfg, data: torch.Tensor, draw: torch.Tensor) -> torch.Tensor:
    """Apply an already-realised draw the way ``cfg.operation`` says.

    Args:
        cfg: The noise configuration the draw came from.
        data: The image to corrupt.
        draw: A :func:`_draw_noise` result for this frame.

    Returns:
        The corrupted image, same shape as ``data``.

    Raises:
        ValueError: If ``cfg.operation`` is not one this module can decompose.
    """
    if cfg.operation == "add":
        return data + draw
    if cfg.operation == "scale":
        return data * draw
    if cfg.operation == "abs":
        return draw
    raise ValueError(f"depth noise operation '{cfg.operation}' is not supported.")


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

    height: int | None = None
    """Image height in pixels. ``None`` = resolve from the camera (the default).

    Resolved by :func:`ensure_depth_noiser` off the sensor's own ``pattern_cfg``, so ONE
    cfg can serve cameras of different resolutions and no task restates a number the
    sensor already carries. An explicit value is still honoured, and still has to match."""

    width: int | None = None
    """Image width in pixels. ``None`` = resolve from the camera. See :attr:`height`."""

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
        if cfg.height is None or cfg.width is None:
            raise ValueError(
                "DepthNoiser needs a concrete height/width; build it through "
                "ensure_depth_noiser, which resolves them from the camera."
            )
        self._H = int(cfg.height)
        self._W = int(cfg.width)

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

        # This frame's realised per-pixel noise + dropout mask, and the control step they
        # belong to. Set before reset(), which clears them.
        self._frame_id: int | None = None
        self._frame_draws: tuple[torch.Tensor, torch.Tensor] | None = None

        self.reset()

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        plane_coeffs: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
    ) -> None:
        """Re-sample the per-episode blur kernel and plane bias.

        ``plane_coeffs`` / ``sigma`` let a caller ADOPT another camera's draw instead of
        making its own, which is what makes two sampled views of one physical camera share
        a mis-calibration. The plane is then re-derived on THIS noiser's own grid: the bias
        is ``a*x + b*y + c`` over a NORMALIZED grid, so identical coefficients are the same
        physical tilted plane whatever the resolution. ``sigma`` is in pixels, so sharing it
        means the same kernel width and therefore a slightly different angular blur extent
        -- defensible (blur is a per-pixel sensor/ISP property) but a modelling choice.

        Args:
            env_ids: Environments to re-sample. ``None`` re-samples all.
            plane_coeffs: ``(len(env_ids), 3)`` coefficients to adopt, or ``None`` to draw.
            sigma: ``(len(env_ids), 1)`` blur sigma to adopt, or ``None`` to draw.
        """
        if env_ids is None:
            env_ids = slice(None)

        # A reset re-samples the camera the frame would have been drawn against, and lands
        # BEFORE both command and observation compute, so no consumer of this step has read
        # it yet. Dropping the cache keeps "one frame per step" from spanning an episode
        # boundary.
        self._frame_id, self._frame_draws = None, None

        # Blur kernel.
        if sigma is None:
            self._sigma[env_ids] = self.cfg.sigma_noise_cfg.func(
                self._sigma[env_ids], self.cfg.sigma_noise_cfg
            )
        else:
            self._sigma[env_ids] = sigma.to(self.device)
        sigma_sq = (self._sigma[env_ids] ** 2).unsqueeze(1).clamp(min=1e-12)
        kernel = torch.exp(-self._dist_sq / (2 * sigma_sq))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)
        self._kernel[env_ids] = kernel

        # Plane coefficients, rescaled so the peak deviation matches the target.
        if plane_coeffs is None:
            n = self._plane[env_ids].shape[0]
            coeffs = torch.randn(n, 3, device=self.device)
            max_dev = coeffs.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
            target = torch.rand(n, 1, device=self.device) * self.cfg.max_plane_deviation
            coeffs = coeffs * (target / max_dev)
        else:
            coeffs = plane_coeffs.to(self.device)
        self._plane_coeffs[env_ids] = coeffs
        a, b, c = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
        self._plane[env_ids] = a * self._grid_x + b * self._grid_y + c

    def episode_state(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-episode draw, for another noiser to adopt via :meth:`reset`.

        Args:
            env_ids: Environments to read. ``None`` reads all.

        Returns:
            ``(plane_coeffs, sigma)`` clones for those environments.
        """
        idx = slice(None) if env_ids is None else env_ids
        return self._plane_coeffs[idx].clone(), self._sigma[idx].clone()

    def _draws_for(
        self, data: torch.Tensor, frame_id: int | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This frame's per-pixel noise and dropout mask, drawn once per ``frame_id``.

        Args:
            data: The flattened image being corrupted; only its shape/device/dtype are read.
            frame_id: Control-step token. ``None`` forces a fresh draw.

        Returns:
            ``(noise_draw, dropout_mask)``, both the shape of ``data``.
        """
        if frame_id is not None and frame_id == self._frame_id and self._frame_draws is not None:
            return self._frame_draws
        draws = (
            _draw_noise(self.cfg.noise_cfg, data),
            torch.rand(data.shape, device=self.device) < self.cfg.dropout_prob,
        )
        self._frame_id, self._frame_draws = frame_id, draws
        return draws

    def __call__(self, data: torch.Tensor, frame_id: int | None = None) -> torch.Tensor:
        """Apply the noise to a normalised depth image.

        **One physical camera yields ONE frame per control step.** The near camera is read by
        the policy's own observation term AND by the generator's ``fm_depth`` term, and again
        by ``mdp.depth_xyz_stack`` when the generator unprojects it; on hardware those are one
        image. So the two per-FRAME draws -- the per-pixel noise and the dropout mask -- are
        realised once per ``frame_id`` and reused, and the per-EPISODE plane and blur kernel
        were already shared by construction. Given the same input this then returns a
        bit-identical image, which is what ``--dump_replay`` needs: the C++ runtime rebuilds
        the generator's extero row from the tracker's recorded ``depth_raw``, so a second,
        independent dropout pattern would show up in ``compare_replay.py`` as a divergence
        that is the harness's fault.

        ``frame_id`` is the env's ``common_step_counter``, which advances once per env step
        and BEFORE both ``command_manager.compute`` and ``observation_manager.compute``, so
        every consumer within one step agrees on it. ``None`` redraws on every call -- the
        standalone/test behaviour, and all an env carrying no such counter can do.

        A diagnostic that drives ``sim.step`` itself rather than going through ``env.step``
        never advances that counter and therefore holds ONE frame for its whole run. That is
        what the pose-sweep checkers want (they feed a single image to two code paths on
        purpose), but it is not a place to read a noise statistic from.

        Args:
            data: ``(B, H*W)`` or ``(B, 1, H, W)`` normalised depth.
            frame_id: Control-step token shared by every reader of this camera.

        Returns:
            The corrupted depth, same shape as ``data``.
        """
        is_4d = data.dim() == 4
        if is_4d:
            orig_shape = data.shape
            data = data.reshape(data.shape[0], -1)

        noise, mask = self._draws_for(data, frame_id)

        data = data + self._plane
        data = _apply_noise(self.cfg.noise_cfg, data, noise)

        # 3x3 Gaussian blur, per-env via a grouped conv.
        data = F.conv2d(
            data.reshape(1, self.num_envs, self._H, self._W),
            self._kernel.reshape(self.num_envs, 1, 3, 3),
            padding=1,
            groups=self.num_envs,
        ).reshape(self.num_envs, -1)

        data[mask] = (self.cfg.dropout_val - self.cfg.offset) * self.cfg.scale

        if is_4d:
            data = data.reshape(orig_shape)
        return data


def ensure_depth_noiser(env, name: str, cfg: DepthNoiseCfg) -> DepthNoiser:
    """Fetch the noiser registered for ``name``, constructing it on first use.

    The image resolution is resolved from the CAMERA when ``cfg`` leaves it unset, so one
    shared cfg can serve cameras that sample at different resolutions -- and so a task
    cannot restate a resolution and have it silently disagree with the sensor it belongs to.

    Args:
        env: The environment instance.
        name: Scene name of the depth camera the noiser belongs to.
        cfg: Noise parameters, used only when constructing.

    Returns:
        The cached :class:`DepthNoiser`.

    Raises:
        ValueError: If ``cfg`` states a resolution that the camera contradicts.
    """
    registry = getattr(env, _ENV_ATTR, None)
    if registry is None:
        registry = {}
        setattr(env, _ENV_ATTR, registry)
    if name not in registry:
        pattern = env.scene[name].cfg.pattern_cfg
        cam_h, cam_w = int(pattern.height), int(pattern.width)
        if cfg.height is not None and cfg.width is not None:
            if (int(cfg.height), int(cfg.width)) != (cam_h, cam_w):
                raise ValueError(
                    f"DepthNoiseCfg for '{name}' says {cfg.height}x{cfg.width} but the camera "
                    f"samples {cam_h}x{cam_w}. Leave height/width unset to take the camera's."
                )
        else:
            cfg = cfg.replace(height=cam_h, width=cam_w)
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
