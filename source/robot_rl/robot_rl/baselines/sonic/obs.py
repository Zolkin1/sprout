# Copyright 2026 Zachary Olkin. All rights reserved.

"""Numpy assembly of SONIC's two ONNX input vectors, ported from its C++ deploy reference.

The upstream source of truth is ``GetObservationRegistry()`` and the ``Gather*`` methods in
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp``. The three details
that a reimplementation gets wrong silently, and that are pinned here:

* **Frame-major packing.** Every stacked block is written as ``offset + frame_idx * per_frame``
  (``GatherMotionJointPositionsMultiFrame``, ~line 772), i.e. all 29 joints of frame 0, then all
  29 of frame 1 -- not joint-major.
* **Histories run oldest-first.** The registry calls ``GatherHis*(..., newest_first=false)``,
  documented in ``state_logger.hpp:198`` as ``[oldest, ..., newest]``, and IsaacLab's
  ``CircularBuffer.buffer`` matches ("oldest entry at the beginning"). :class:`HistoryBuffer`
  also reproduces IsaacLab's warm start: the first append after a reset fills the whole depth
  with that sample (``circular_buffer.py:136``), so the first ten steps are not zero-padded.
* **The 6D orientation is two COLUMNS flattened row-wise**, i.e. ``[R00, R01, R10, R11, R20,
  R21]`` of ``R = mat(q_robot^-1 (x) q_ref)`` (``GatherMotionAnchorOrientationMutiFrame``,
  ~line 679, matching python's ``matrix_from_quat(...)[..., :2].reshape(-1)``).

Upstream works in wxyz and this repo in xyzw. **Every quaternion argument below is xyzw and the
caller must convert** -- feeding wxyz arrays into these helpers silently computes a different
rotation. (An earlier version of this docstring claimed the 6D block was convention-agnostic
because conjugate/mul/to-matrix are "convention-consistent". That is false: relabelling the
components produces a genuinely different quaternion, and the resulting 6D block differs by up
to 1.6. It happened not to bite because both of ``SonicEnvBridge``'s sources -- ``root_quat_w``
and ``get_future_world_base`` -- are already xyzw under IsaacLab 3.0/Newton, verified from a
recorded rollout: the scalar sits in component 3.) ``sonic_trace_replay.py`` converts its
wxyz trace with ``_wxyz_to_xyzw`` for the same reason. Parity-tested against
``transfer/deploy/quat_math_np.py``.

Free of torch, isaaclab and onnxruntime on purpose: this is where the fiddly indexing lives, so
it has to be testable headless and without the third-party weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import TOKEN_STATE, Block, SonicSpec

# Field names carried in the robot-state history, and the observation bases that read them.
_HISTORY_FIELDS: dict[str, str] = {
    "base_angular_velocity": "base_ang_vel",
    "his_base_angular_velocity": "base_ang_vel",
    "body_joint_positions": "joint_pos_rel",
    "his_body_joint_positions": "joint_pos_rel",
    "body_joint_velocities": "joint_vel",
    "his_body_joint_velocities": "joint_vel",
    "last_actions": "last_action",
    "his_last_actions": "last_action",
    "gravity_dir": "gravity_dir",
    "his_gravity_dir": "gravity_dir",
}


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate ``[-x, -y, -z, w]`` of an xyzw quaternion ``(..., 4)``."""
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., :3] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a (x) b`` of xyzw quaternions (broadcasts over leading dims)."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``(..., 3, 3)`` from an xyzw quaternion ``(..., 4)``."""
    x, y, z, w = (q[..., i] for i in range(4))
    xx, yy, zz = x * x, y * y, z * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def anchor_ori_6d(robot_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
    """SONIC's ``motion_anchor_ori_b`` block: reference-vs-robot root orientation as 6D.

    Uses the robot's **full** pelvis quaternion as the left operand (upstream
    ``orientation_mode == 0``), so the block carries the absolute roll/pitch/yaw mismatch to the
    reference rather than a heading-corrected one.

    Args:
        robot_quat: ``[N, 4]`` robot pelvis orientation, world frame, xyzw.
        ref_quat: ``[N, F, 4]`` reference pelvis orientation per future frame, world frame, xyzw.

    Returns:
        ``[N, F, 6]`` -- per frame, the first two columns of the relative rotation matrix,
        flattened row-wise.
    """
    rel = quat_mul(quat_conjugate(robot_quat)[:, None, :], ref_quat)
    mat = matrix_from_quat(rel)                                    # [N, F, 3, 3]
    return mat[..., :2].reshape(*mat.shape[:-2], 6)


@dataclass
class MotionReference:
    """The reference window SONIC's g1 encoder mode consumes, in Isaac DOF order.

    Frame 0 is the current reference frame; frame ``k`` is ``k * step`` control steps ahead,
    clamped at the clip's last frame (upstream holds the final pose rather than running off the
    end -- ``g1_deploy_onnx_ref.cpp:763``, mirroring ``commands.py``'s ``future_time_steps``).

    Attributes:
        joint_pos: ``[N, F, 29]`` absolute reference joint angles in radians -- **not** offset by
            the default pose, unlike the robot-state history.
        joint_vel: ``[N, F, 29]`` reference joint velocities, rad/s.
        root_quat: ``[N, F, 4]`` reference pelvis orientation, world frame, xyzw.
        root_z: ``[N, F]`` reference pelvis world height. Only needed by checkpoint variants
            whose encoder mode requires ``motion_root_z_position*``; the released g1 mode does
            not (verified: perturbing those slices leaves the token bit-identical).
    """

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_quat: np.ndarray
    root_z: np.ndarray | None = None


@dataclass
class RobotState:
    """One control step of robot state, in Isaac DOF order.

    Everything except :attr:`root_quat` is a history field that feeds the decoder; ``root_quat``
    is world-frame and is used only as the left operand of the encoder's orientation block.

    Attributes:
        root_quat: ``[N, 4]`` robot pelvis orientation, world frame, xyzw.
        base_ang_vel: ``[N, 3]`` root angular velocity in the body frame.
        joint_pos_rel: ``[N, 29]`` ``q - default_q`` using SONIC's default pose.
        joint_vel: ``[N, 29]`` joint velocities, rad/s, unscaled.
        gravity_dir: ``[N, 3]`` gravity direction in the body frame (IsaacLab's
            ``projected_gravity_b``, which equals upstream's
            ``quat_apply(quat_inv(pelvis_quat), [0, 0, -1])``).
        last_action: ``[N, 29]`` the previous step's **raw, pre-scale** network output. Leave it
            None and :class:`~robot_rl.baselines.sonic.policy.SonicPolicy` fills it from its own
            previous output, which is what a rollout wants; set it explicitly only to replay a
            recorded trace.
    """

    root_quat: np.ndarray
    base_ang_vel: np.ndarray
    joint_pos_rel: np.ndarray
    joint_vel: np.ndarray
    gravity_dir: np.ndarray
    last_action: np.ndarray | None = None

    def with_last_action(self, last_action: np.ndarray) -> RobotState:
        """A copy with :attr:`last_action` set."""
        return RobotState(
            root_quat=self.root_quat,
            base_ang_vel=self.base_ang_vel,
            joint_pos_rel=self.joint_pos_rel,
            joint_vel=self.joint_vel,
            gravity_dir=self.gravity_dir,
            last_action=last_action,
        )


class HistoryBuffer:
    """Per-env ring of proprioception samples, read back oldest-first.

    Reproduces IsaacLab's ``CircularBuffer`` semantics exactly, because that is what the policy
    was trained against: on the first append after a reset the whole depth is filled with that
    sample, so a freshly reset env sees a constant history rather than zeros.

    Args:
        num_envs: Batch size.
        depth: Samples to retain. Must cover the deepest block, ``(frames - 1) * step + 1``.
        dims: ``{field: width}`` for every field the layout reads.
    """

    def __init__(self, num_envs: int, depth: int, dims: dict[str, int]) -> None:
        self._n = int(num_envs)
        self._depth = int(depth)
        self._buf = {k: np.zeros((self._depth, self._n, int(d)), np.float64)
                     for k, d in dims.items()}
        self._ptr = -1
        self._pushes = np.zeros(self._n, np.int64)

    @property
    def depth(self) -> int:
        """Number of retained samples."""
        return self._depth

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Clear the fill state for ``env_ids`` (all envs when None).

        The stored values are left alone: the next append overwrites the whole depth for any env
        whose counter is zero, which is what makes the warm start work.
        """
        if env_ids is None:
            self._pushes[:] = 0
        elif len(env_ids):
            self._pushes[np.asarray(env_ids, dtype=np.int64)] = 0

    def append(self, state: RobotState) -> None:
        """Push one control step for every env."""
        self._ptr = (self._ptr + 1) % self._depth
        first = self._pushes == 0
        for name, buf in self._buf.items():
            data = np.asarray(getattr(state, name), dtype=np.float64)
            if data.shape != (self._n, buf.shape[2]):
                raise ValueError(f"history field '{name}': got {data.shape}, "
                                 f"expected {(self._n, buf.shape[2])}")
            buf[self._ptr] = data
            if first.any():
                buf[:, first] = data[first]
        self._pushes += 1

    def stack(self, name: str, frames: int, step: int) -> np.ndarray:
        """``[N, frames, D]`` samples ending at the newest, strided by ``step``, oldest-first.

        Raises:
            KeyError: If ``name`` is not a tracked field.
            ValueError: If the requested window is deeper than the buffer.
        """
        if name not in self._buf:
            raise KeyError(f"history field '{name}' is not tracked (have {sorted(self._buf)})")
        need = (frames - 1) * step + 1
        if need > self._depth:
            raise ValueError(f"'{name}' needs {need} samples but the buffer holds {self._depth}")
        idx = [(self._ptr - k * step) % self._depth for k in range(frames)][::-1]
        return np.transpose(self._buf[name][idx], (1, 0, 2))


def history_depth(spec: SonicSpec) -> int:
    """Samples the decoder layout needs retained: ``max((frames - 1) * step + 1)``."""
    return max((b.frames - 1) * b.step + 1 for b in spec.decoder_blocks)


def history_dims(spec: SonicSpec) -> dict[str, int]:
    """``{field: width}`` for every history field the decoder layout reads."""
    dims: dict[str, int] = {}
    for b in spec.decoder_blocks:
        if b.base in _HISTORY_FIELDS:
            dims[_HISTORY_FIELDS[b.base]] = b.per_frame
    return dims


def _write(out: np.ndarray, block: Block, values: np.ndarray) -> None:
    """Write ``[N, ...]`` values into ``block``'s slice, flattening frame-major."""
    flat = np.asarray(values, dtype=np.float64).reshape(out.shape[0], -1)
    if flat.shape[1] != block.dim:
        raise ValueError(f"block '{block.name}': built {flat.shape[1]} values, "
                         f"layout wants {block.dim}")
    out[:, block.offset:block.stop] = flat


def build_encoder_input(spec: SonicSpec, mode: str, ref: MotionReference,
                        robot_quat: np.ndarray) -> np.ndarray:
    """Assemble the encoder input for one control step.

    Only the blocks ``mode`` requires are filled; every other slice stays zero, which is what
    upstream's ``GatherEncoderObservations`` does and what the graph's one-hot mode selection
    makes safe.

    Args:
        spec: The resolved layout.
        mode: Encoder mode name, e.g. ``"g1"``.
        ref: The reference window, sampled at ``spec.reference_frames(mode)``.
        robot_quat: ``[N, 4]`` robot pelvis orientation, world frame, xyzw.

    Returns:
        ``[N, spec.encoder_dim]`` float32.

    Raises:
        KeyError: If ``mode`` is not in the config.
        NotImplementedError: If the mode requires a block this port cannot build.
        ValueError: If a required block needs reference data that was not supplied.
    """
    if mode not in spec.modes:
        raise KeyError(f"encoder mode '{mode}' not in {spec.path} (have {sorted(spec.modes)})")
    n = robot_quat.shape[0]
    out = np.zeros((n, spec.encoder_dim), np.float64)
    for b in spec.mode_blocks(mode):
        if b.base in ("encoder_mode", "encoder_mode_4"):
            # The mode id occupies element 0; the remaining slots are zero padding.
            out[:, b.offset] = float(spec.modes[mode].mode_id)
        elif b.base == "motion_joint_positions":
            _write(out, b, ref.joint_pos[:, :b.frames])
        elif b.base == "motion_joint_velocities":
            _write(out, b, ref.joint_vel[:, :b.frames])
        elif b.base == "motion_anchor_orientation":
            _write(out, b, anchor_ori_6d(robot_quat, ref.root_quat[:, :b.frames]))
        elif b.base == "motion_root_z_position":
            if ref.root_z is None:
                raise ValueError(
                    f"mode '{mode}' requires '{b.name}' but MotionReference.root_z is None. "
                    "Supply the reference pelvis world height for this checkpoint variant."
                )
            _write(out, b, ref.root_z[:, :b.frames])
        else:
            raise NotImplementedError(
                f"encoder mode '{mode}' requires block '{b.name}', which this port does not "
                "build. Only the g1 (robot-motion-reference) mode is implemented -- the teleop "
                "and SMPL modes would need VR or SMPL inputs this repo has no source for."
            )
    return out.astype(np.float32)


def build_decoder_input(spec: SonicSpec, token: np.ndarray, hist: HistoryBuffer) -> np.ndarray:
    """Assemble the decoder input for one control step.

    Args:
        spec: The resolved layout.
        token: ``[N, spec.token_dim]`` latent from the encoder.
        hist: History buffer, already holding the current step.

    Returns:
        ``[N, spec.decoder_dim]`` float32.

    Raises:
        NotImplementedError: If the layout names a decoder block this port cannot build.
    """
    n = token.shape[0]
    out = np.zeros((n, spec.decoder_dim), np.float64)
    for b in spec.decoder_blocks:
        if b.name == TOKEN_STATE:
            _write(out, b, token)
        elif b.base in _HISTORY_FIELDS:
            _write(out, b, hist.stack(_HISTORY_FIELDS[b.base], b.frames, b.step))
        else:
            raise NotImplementedError(
                f"decoder block '{b.name}' is not built by this port. Add it to "
                "robot_rl.baselines.sonic.obs._HISTORY_FIELDS if it is a robot-state history, or "
                "implement it explicitly."
            )
    return out.astype(np.float32)
