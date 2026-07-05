# Copyright 2026 Zachary Olkin. All rights reserved.

"""Torch-free bend-cycle deploy policy: a single ONNX actor with an external phase.

The bend-cycle policy is a single RL actor whose only non-proprioceptive input is
the sin/cos of the trajectory phasing variable ``phi in [0, 1]`` (``phi 0 -> ~0.47``
bends over, ``~0.47 -> 1.0`` bends back up; ``phi 0 == phi 1.0 == standing``).
Unlike the walking policies there is no phase *clock* here -- ``phi`` is supplied
externally each step (driven by joystick buttons in the sim2sim runner), so this
class simply assembles the observation around the given phase and runs the actor.

Reuses the metadata-driven ONNX loader :class:`TrackerSession`: the obs-term
order/scales, joint order, PD gains, and default pose all come from the ``.onnx``
metadata -- no YAML. No torch / isaaclab / robot_rl imports (numpy + onnxruntime).
"""

from __future__ import annotations

import numpy as np

from .quat_math_np import quat_apply, quat_inv
from .tracker_session import TrackerSession, convert_joint_order

# Matches the training ``actions`` observation clip (-20, 20) in
# G1ClfTrackingObservationsCfg. Other bend-cycle terms are unclipped.
_ACTION_CLIP = 20.0


class BendCyclePolicy:
    """Single-actor bend-cycle policy with an externally supplied phasing variable."""

    def __init__(self, tracker_onnx: str, num_threads: int = 0):
        """Load ``tracker.onnx`` and its embedded metadata via :class:`TrackerSession`.

        Args:
            tracker_onnx: Path to the exported ``tracker.onnx`` (raw obs -> actions,
                normalizer folded in, ``policy_parameters`` embedded).
            num_threads: onnxruntime intra-op threads (0 = auto).
        """
        self._trk = TrackerSession(tracker_onnx, num_threads)
        self._obs_terms = self._trk.get_obs_terms()
        self._default = self._trk.default_joint_angles
        self._num_obs = self._trk.get_num_obs()

        tt = self._trk.params.get("total_time")
        if tt is None or float(tt) <= 0.0:
            raise ValueError(
                "tracker.onnx metadata has no positive 'total_time'; re-export the "
                "bend-cycle policy with export_parameters.py + export_tracker_onnx.py."
            )
        self.total_time = float(tt)

        # Phase the policy holds "bent" at (the bend_down->bend_up domain boundary,
        # e.g. ~0.474), embedded at export time. Falls back to 0.5 for policies
        # exported before this metadata existed. The runner sweeps the "bend over"
        # button to this phase so the hold matches what the policy trained on.
        self.bent_phi = float(self._trk.params.get("bent_hold_phi", 0.5))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def create_obs(
        self,
        base_quat_xyzw: np.ndarray,
        base_ang_vel: np.ndarray,
        joint_pos_mujoco: np.ndarray,
        joint_vel_mujoco: np.ndarray,
        joint_names_mujoco: list[str],
        phi: float,
    ) -> np.ndarray:
        """Assemble the policy observation ``[1, num_obs]`` for the given phase.

        Args:
            base_quat_xyzw: Base (pelvis) orientation quaternion ``(x, y, z, w)``.
            base_ang_vel: Base angular velocity in the base frame ``(3,)`` (MuJoCo
                free-joint ``qvel[3:6]`` equals Isaac ``base_ang_vel``).
            joint_pos_mujoco: Joint positions in MuJoCo joint order.
            joint_vel_mujoco: Joint velocities in MuJoCo joint order.
            joint_names_mujoco: MuJoCo joint names (to reorder into the policy order).
            phi: Phasing variable in ``[0, 1]`` (externally driven).

        Returns:
            Float32 observation array of shape ``[1, num_obs]``.
        """
        isaac = self._trk.get_joint_names()
        jp = convert_joint_order(np.asarray(joint_pos_mujoco), joint_names_mujoco, isaac)
        jv = convert_joint_order(np.asarray(joint_vel_mujoco), joint_names_mujoco, isaac)
        inv_root = quat_inv(np.asarray(base_quat_xyzw, dtype=np.float64))
        two_pi_phi = 2.0 * np.pi * float(phi)

        # Term values (pre-scale). ``projected_gravity`` = world -Z rotated into the
        # base frame, matching mdp.projected_gravity (and the FM tracker's numpy path).
        # ``velocity_commands`` is identically zero -- the bend cycle is stationary.
        values = {
            "base_ang_vel": np.asarray(base_ang_vel, dtype=np.float64),
            "projected_gravity": quat_apply(inv_root, np.array([0.0, 0.0, -1.0])),
            "velocity_commands": np.zeros(3),
            "joint_pos": jp - self._default,
            "joint_vel": jv,
            "actions": np.asarray(self._trk.action_isaac, dtype=np.float64),
            "sin_phase": np.array([np.sin(two_pi_phi)]),
            "cos_phase": np.array([np.cos(two_pi_phi)]),
        }
        return self._assemble(values)

    def obs_term_slice(self, name: str) -> tuple[int, int]:
        """Return the ``(start, length)`` slice of obs term ``name`` in the assembled obs.

        Lets a caller read the exact (scaled) values fed to the actor for a term --
        e.g. ``sin_phase`` / ``cos_phase`` -- straight from the observation vector,
        instead of reconstructing them from ``phi``.

        Args:
            name: Observation term name (as in the embedded ``observation_terms.policy``).

        Returns:
            ``(start_index, length)`` into the flat ``[num_obs]`` observation.
        """
        idx = 0
        for term in self._obs_terms:
            if term["name"] == name:
                return idx, term["shape"]
            idx += term["shape"]
        raise KeyError(f"Obs term '{name}' not found in the policy observation.")

    def _assemble(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Order, clip, and scale terms per the embedded ``observation_terms.policy``."""
        out = np.zeros(self._num_obs, dtype=np.float32)
        idx = 0
        for term in self._obs_terms:
            name, shape, scale = term["name"], term["shape"], term["scale"]
            if term["history_length"] > 0:
                raise NotImplementedError(
                    f"Obs term '{name}' has history_length>0; the bend-cycle policy "
                    "expects flat (non-history) terms."
                )
            if name not in values:
                raise NotImplementedError(f"Bend-cycle obs term '{name}' not implemented.")
            v = np.atleast_1d(values[name].astype(np.float32))
            if v.shape[0] != shape:
                raise ValueError(f"Term '{name}': built dim {v.shape[0]} != metadata dim {shape}.")
            if name == "actions":
                v = np.clip(v, -_ACTION_CLIP, _ACTION_CLIP)
            out[idx:idx + shape] = v * scale
            idx += shape
        return out[None]

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def get_action(self, obs: np.ndarray, joint_names_out: list[str]) -> np.ndarray:
        """Run the actor; return joint position targets in ``joint_names_out`` order.

        Also updates ``self._trk.action_isaac`` (the raw action fed back into the
        ``actions`` observation term next step).
        """
        return self._trk.get_action(obs, joint_names_out)

    # ------------------------------------------------------------------
    # Robot-compat passthroughs (used by Robot.set_pd_gains_from_policy /
    # set_initial_condition, which the runner reuses).
    # ------------------------------------------------------------------
    def get_dt(self) -> float:
        """Control timestep (s)."""
        return self._trk.get_dt()

    def get_kp(self) -> list[float]:
        """Per-joint proportional gains (isaac order)."""
        return self._trk.get_kp()

    def get_kd(self) -> list[float]:
        """Per-joint derivative gains (isaac order)."""
        return self._trk.get_kd()

    def get_joint_names(self) -> list[str]:
        """Policy (isaac) joint order."""
        return self._trk.get_joint_names()

    def convert_joint_order(self, values, names_in, names_out):
        """Reorder per-joint values between name orderings (delegates to the session)."""
        return self._trk.convert_joint_order(values, names_in, names_out)

    def get_valid_ic_pos(self) -> np.ndarray | None:
        """Valid initial pose ``[base_pos(3), base_quat(4), joint_pos(N)]`` or None."""
        return self._trk.get_valid_ic_pos()

    def get_valid_ic_vel(self) -> np.ndarray | None:
        """Valid initial velocity ``[base_lin(3), base_ang(3), joint_vel(N)]`` or None."""
        return self._trk.get_valid_ic_vel()
