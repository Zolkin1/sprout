# Copyright 2026 Zachary Olkin. All rights reserved.

"""How far the robot is from the reference it was served -- the IsaacLab half.

Torch twin of :mod:`transfer.deploy.tracking_error`, which carries the full rationale: one
producer for the error row, read by the failure latch, by the dense tracking reward, and by the
recorders, so the three cannot disagree about what a stumble is.

**Why there are two copies rather than one import.**  ``robot_rl`` must not import ``transfer``
(the deploy runtime is torch-free and ships without IsaacLab), and this side folds the row every
control step over up to 1024 envs, so a numpy round-trip would be a device sync per step.  The
copies are held equal by ``transfer/deploy/tests/test_tracking_error_parity.py``, which checks
BOTH the constants and the functions -- the same "duplicate and pin" rule
``transfer/sim/robots/g1/make_g1_mjcf.py`` states for the actuator table.

**Three of the six channels can end an episode: lateral, height, tilt.**  Downrange and yaw are
measured and rewarded and never fatal -- downrange because a terrain-matched re-plan moves the
reference along the direction of travel ON PURPOSE and the tracker then cannot be anything but
behind it, yaw because the anchor divides heading out in full.  The numpy module carries the
measurements behind both exclusions.

The surviving checks are also not the same kind of quantity: under ``window_anchor="plan"``
position error is re-zeroed at every re-plan while TILT is measured against gravity and is not.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi, yaw_quat

#: Fixed channels of an error row, before the per-joint block.  Mirrors
#: ``transfer.deploy.tracking_error.ERROR_CHANNEL_NAMES``.
ERROR_CHANNEL_NAMES: tuple[str, ...] = (
    "d_downrange", "d_lateral", "d_height",
    "theta_tilt", "theta_yaw",
    "dv_x", "dv_y", "dv_z",
    "dw_x", "dw_y", "dw_z",
)

#: Width of the fixed part; a full row is ``NUM_FIXED_CHANNELS + num_joints``.
NUM_FIXED_CHANNELS: int = len(ERROR_CHANNEL_NAMES)

_POS = slice(0, 3)
_TILT = 3
_YAW = 4
_LIN = slice(5, 8)
_ANG = slice(8, 11)
_JOINT = slice(NUM_FIXED_CHANNELS, None)


def error_channel_names(num_joints: int) -> tuple[str, ...]:
    """Every channel of an error row, in order.

    Args:
        num_joints: Robot joint count; the trailing block is one channel per joint in the
            robot's ISAAC joint order.

    Returns:
        ``NUM_FIXED_CHANNELS + num_joints`` names.
    """
    return ERROR_CHANNEL_NAMES + tuple(f"dq_{i}" for i in range(int(num_joints)))


@dataclass(frozen=True)
class TrackingErrorSpec:
    """Thresholds and kernel widths.  Field-for-field mirror of the deploy-side spec.

    See :class:`transfer.deploy.tracking_error.TrackingErrorSpec` for what each number means
    and why the defaults are a BOOTSTRAP that publishing code should refuse.  There is
    deliberately no downrange bound and no yaw bound -- the fields do not exist, so nothing can
    quietly start reading one.
    """

    max_lateral_m: float = 0.2
    max_height_m: float = 0.2
    max_tilt_rad: float = math.radians(30.0)

    sigma_pos_m: float = 0.10
    sigma_ori_rad: float = math.radians(15.0)
    sigma_lin_vel: float = 0.5
    sigma_ang_vel: float = 1.0
    sigma_joint_rad: float = 0.15

    weights: tuple[float, float, float, float] = (0.25, 0.25, 0.10, 0.40)
    gamma: float = 0.95

    provenance: dict[str, Any] = field(default_factory=lambda: {"bootstrap": True})

    def __post_init__(self) -> None:
        """Refuse a spec that cannot mean what it says.

        Raises:
            ValueError: On a non-positive width or bound, weights that do not sum to 1, or a
                ``gamma`` outside ``(0, 1)``.  The weight check is load-bearing: a step reward
                is bounded in ``(0, 1]`` only because they sum to 1, and the quality head's
                support is declared on that basis.
        """
        for name in ("max_lateral_m", "max_height_m",
                     "max_tilt_rad", "sigma_pos_m", "sigma_ori_rad", "sigma_lin_vel",
                     "sigma_ang_vel", "sigma_joint_rad"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if len(self.weights) != 4 or any(w < 0.0 for w in self.weights):
            raise ValueError(f"weights must be four non-negative values, got {self.weights}")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError(
                f"weights must sum to 1 so a step reward is bounded in (0, 1]. Got "
                f"{self.weights}, sum {sum(self.weights)!r}.")
        if not 0.0 < float(self.gamma) < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {self.gamma!r}")

    @property
    def is_bootstrap(self) -> bool:
        """Whether these numbers are the un-fitted placeholders."""
        return bool(self.provenance.get("bootstrap", False))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe dict, for stamping into a manifest."""
        d = asdict(self)
        d["weights"] = list(self.weights)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrackingErrorSpec":
        """Rebuild a spec from :meth:`to_dict`.

        Args:
            d: The mapping.

        Returns:
            The spec.

        Raises:
            ValueError: On an unrecognised key -- a renamed field falling back to its default
                would silently score an artifact under numbers it was not fitted with.
        """
        known = set(cls.__dataclass_fields__)                              # noqa: SLF001
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"TrackingErrorSpec got unknown keys {sorted(unknown)}; expected a subset of "
                f"{sorted(known)}.")
        kw = dict(d)
        if "weights" in kw:
            kw["weights"] = tuple(float(v) for v in kw["weights"])
        return cls(**kw)


#: The un-fitted spec, for code that has to construct one before the thresholds are measured.
BOOTSTRAP_SPEC = TrackingErrorSpec()


def tracking_error(
    ref_pos_w: torch.Tensor,
    ref_quat_w: torch.Tensor,
    ref_lin_vel_w: torch.Tensor,
    ref_ang_vel_w: torch.Tensor,
    ref_joint_pos: torch.Tensor,
    act_pos_w: torch.Tensor,
    act_quat_w: torch.Tensor,
    act_lin_vel_w: torch.Tensor,
    act_ang_vel_w: torch.Tensor,
    act_joint_pos: torch.Tensor,
) -> torch.Tensor:
    """One error row per env: how the robot is displaced from the reference it was served.

    Every argument is WORLD frame and describes the SAME instant -- the served node (offset 0 at
    the current cursor) and the robot as the policy saw it.  Position, linear and angular
    velocity are differenced in world and then rotated into the REFERENCE's yaw frame, signed
    ``actual - reference``, so ``+d_downrange`` means the robot is ahead of where it should be.

    Args:
        ref_pos_w: ``[N, 3]`` reference base position.
        ref_quat_w: ``[N, 4]`` reference base orientation, xyzw.
        ref_lin_vel_w: ``[N, 3]`` reference base linear velocity.
        ref_ang_vel_w: ``[N, 3]`` reference base angular velocity.
        ref_joint_pos: ``[N, nj]`` reference joint angles, ISAAC order.
        act_pos_w: ``[N, 3]`` measured base position.
        act_quat_w: ``[N, 4]`` measured base orientation, xyzw.
        act_lin_vel_w: ``[N, 3]`` measured base linear velocity, world.
        act_ang_vel_w: ``[N, 3]`` measured base angular velocity, world.
        act_joint_pos: ``[N, nj]`` measured joint angles, ISAAC order.

    Returns:
        ``[N, NUM_FIXED_CHANNELS + nj]``, laid out as :func:`error_channel_names` says.
    """
    # The reference's HEADING frame. Yaw-only for the same reason the window is: rolling the
    # error frame with the reference's own tilt would mix a vertical displacement into
    # `d_downrange` exactly when the reference is pitched, which on stairs is always.
    ref_yaw = yaw_quat(ref_quat_w)

    dp = quat_apply_inverse(ref_yaw, act_pos_w - ref_pos_w)
    dv = quat_apply_inverse(ref_yaw, act_lin_vel_w - ref_lin_vel_w)
    dw = quat_apply_inverse(ref_yaw, act_ang_vel_w - ref_ang_vel_w)

    tilt = tilt_between(ref_quat_w, act_quat_w)
    yaw = yaw_error(ref_quat_w, act_quat_w)
    dq = act_joint_pos - ref_joint_pos

    return torch.cat([dp, tilt.unsqueeze(-1), yaw.unsqueeze(-1), dv, dw, dq], dim=-1)


def tilt_between(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    """Angle between two orientations' body-z axes, in radians.

    Yaw-independent by construction -- a pure heading change moves neither z axis -- so this and
    :func:`yaw_error` do not double-count and neither needs a frame resolved.  Same quantity
    ``terminations.base_pose_error_termination`` calls ``tilt``; only the datum differs.

    Args:
        quat_a: ``[N, 4]`` xyzw.
        quat_b: ``[N, 4]`` xyzw.

    Returns:
        ``[N]`` angle in ``[0, pi]``.
    """
    up = torch.tensor([0.0, 0.0, 1.0], device=quat_a.device,
                      dtype=quat_a.dtype).expand(quat_a.shape[0], 3)
    cos = (quat_apply(quat_a, up) * quat_apply(quat_b, up)).sum(dim=-1)
    return torch.acos(cos.clamp(-1.0, 1.0))


def yaw_error(ref_quat_w: torch.Tensor, act_quat_w: torch.Tensor) -> torch.Tensor:
    """Signed heading difference ``act - ref``, wrapped to ``(-pi, pi]``.

    Args:
        ref_quat_w: ``[N, 4]`` xyzw.
        act_quat_w: ``[N, 4]`` xyzw.

    Returns:
        ``[N]`` radians.
    """
    _, _, ref_yaw = euler_xyz_from_quat(ref_quat_w, wrap_to_2pi=False)
    _, _, act_yaw = euler_xyz_from_quat(act_quat_w, wrap_to_2pi=False)
    return wrap_to_pi(act_yaw - ref_yaw)


#: The channels that can end an episode, in report order.  DOWNRANGE AND YAW ARE ABSENT BY
#: DESIGN -- both are measured, both are rewarded, neither is fatal.
VIOLATION_CHECK_NAMES: tuple[str, ...] = ("lateral", "height", "tilt")


def violation_checks(err: torch.Tensor,
                    spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> dict[str, torch.Tensor]:
    """Which check fired, so a report can say more than "it failed".

    Three checks, named individually rather than pooled: lateral and height are chunk-local and
    re-zeroed at every re-plan, tilt is measured against gravity and is not, so a single rate
    hides which threshold is doing the work and which one to move first.

    Args:
        err: ``[N, n]`` rows from :func:`tracking_error`.
        spec: The bounds.

    Returns:
        ``{"lateral", "height", "tilt"}`` -> ``[N]`` bool tensors; a row can set several.
    """
    return {
        "lateral": err[..., 1].abs() > float(spec.max_lateral_m),
        "height": err[..., 2].abs() > float(spec.max_height_m),
        "tilt": err[..., _TILT] > float(spec.max_tilt_rad),
    }


def violates(err: torch.Tensor, spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> torch.Tensor:
    """Whether an error row is outside the failure bounds.

    The union of :func:`violation_checks`, and it goes through that function rather than
    restating the three tests, so the predicate and the breakdown cannot disagree.

    Args:
        err: ``[N, n]`` rows from :func:`tracking_error`.
        spec: The bounds.

    Returns:
        ``[N]`` bool.
    """
    checks = violation_checks(err, spec)
    out = checks[VIOLATION_CHECK_NAMES[0]]
    for name in VIOLATION_CHECK_NAMES[1:]:
        out = out | checks[name]
    return out


def step_reward(err: torch.Tensor, spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> torch.Tensor:
    """DeepMimic-style tracking reward for one control step, in ``(0, 1]``.

    Four exponential kernels over exactly the blocks the generator emits -- base pose, base
    twist, joint angles.  No end effectors and no joint velocities: the live
    ``REFERENCE_WINDOW_BLOCKS`` selection does not carry them, so a term reading either would
    grade against something the model never predicted.

    Args:
        err: ``[N, n]`` rows from :func:`tracking_error`.
        spec: The kernel widths and weights.

    Returns:
        ``[N]`` in ``(0, 1]``.
    """
    w_pos, w_ori, w_vel, w_joint = (float(v) for v in spec.weights)

    pos_sq = err[..., _POS].pow(2).sum(dim=-1)
    ori_sq = err[..., _TILT].pow(2) + err[..., _YAW].pow(2)
    lin_sq = err[..., _LIN].pow(2).sum(dim=-1)
    ang_sq = err[..., _ANG].pow(2).sum(dim=-1)
    # RMS rather than a sum, so the width survives a change in joint count.
    joint_ms = err[..., _JOINT].pow(2).mean(dim=-1)

    r_pos = torch.exp(-pos_sq / spec.sigma_pos_m ** 2)
    r_ori = torch.exp(-ori_sq / spec.sigma_ori_rad ** 2)
    r_vel = torch.exp(-(lin_sq / spec.sigma_lin_vel ** 2 + ang_sq / spec.sigma_ang_vel ** 2))
    r_joint = torch.exp(-joint_ms / spec.sigma_joint_rad ** 2)

    return w_pos * r_pos + w_ori * r_ori + w_vel * r_vel + w_joint * r_joint


def chunk_reward(err: torch.Tensor, spec: TrackingErrorSpec = BOOTSTRAP_SPEC) -> torch.Tensor:
    """Mean :func:`step_reward` over a chunk's control steps, in ``(0, 1]``.

    The MEAN rather than a sum, so the value is independent of ``fm_update_every`` and stays
    commensurate with the per-step reward it is built from.

    Args:
        err: ``[..., K, n]`` -- control steps on the second-to-last axis.
        spec: The kernel widths and weights.

    Returns:
        ``[...]``.
    """
    return step_reward(err, spec).mean(dim=-1)


def normalized_return(chunk_rewards: Sequence[float] | torch.Tensor,
                      spec: TrackingErrorSpec = BOOTSTRAP_SPEC,
                      *, failed: bool) -> float:
    """``G = (1 - gamma) * sum gamma^i R_i + tail``, in ``[0, 1]``.

    **How the episode ENDED decides the tail.**  ``failed=True`` gives a zero tail -- nothing
    good follows a fall or a latch, so failing early is worth much less than failing late, which
    is what couples this head to survival.  ``failed=False`` (arrived, or the budget ran out)
    bootstraps ``gamma^T * R_{T-1}``: absorbing there would punish arriving QUICKLY, making a
    short flawless run score 0.46 against a long one's 0.95 for reasons that are entirely about
    episode length.

    Args:
        chunk_rewards: ``[T]`` per-chunk rewards, earliest first, truncated at the terminal
            event (for a latched episode, at the latch).
        spec: Supplies ``gamma``.
        failed: Whether the truncation point is a FAILURE.  Keyword-only with no default: the
            two answers differ by more than 2x on a short episode.

    Returns:
        The normalized return.
    """
    r = torch.as_tensor(chunk_rewards, dtype=torch.float64).reshape(-1)
    if r.numel() == 0:
        return 0.0
    g = float(spec.gamma)
    disc = g ** torch.arange(r.numel(), dtype=torch.float64)
    head = (1.0 - g) * float((r * disc).sum())
    tail = 0.0 if failed else float(g ** r.numel() * r[-1])
    return head + tail
