# Copyright 2026 Zachary Olkin. All rights reserved.

"""Drive :class:`~robot_rl.baselines.sonic.policy.SonicPolicy` from a manager-based env.

The only torch/isaaclab module in the SONIC baseline. It reads the robot state and the reference
window out of the environment, hands them to the numpy adapter, and returns actions shaped like
any other policy's output -- so a rollout loop can swap this in for
``runner.get_inference_policy()`` without further changes.

What matters here:

* **Joint order -- no permutation.** The clip LIBRARY stores joint channels in
  ``G1_JOINT_NAMES`` order, but the command term does not hand them over that way: its
  ``_base_idx["joint_pos"]`` is built by iterating ``robot.joint_names``, so
  ``get_future_joint_state`` already answers in the articulation's DOF order -- which is also
  SONIC's and the action term's. Reordering again here scrambles a correct vector, and that is
  exactly what it did until 2026-07-30. :meth:`SonicEnvBridge._verify_reference_joint_order`
  asserts the contract instead of assuming it.
* **Reference source.** ``get_future_joint_state`` / ``get_future_world_base`` are the command
  term's own API, so the same bridge works for the clip teacher's
  :class:`~...motion_sequence_cmd.MotionSequenceCommand` and for
  :class:`~...fm_tracking_cmd.FmTrackingCommand`, whose overrides return the flow-matching
  prediction instead. That is what keeps "SONIC on clips" and "SONIC on FM" honest.

Resets are detected from ``episode_length_buf == 0`` rather than from a caller-supplied done
mask: IsaacLab resets terminated envs inside ``env.step``, and the SONIC history has to be
warm-restarted for exactly those envs.
"""

from __future__ import annotations

import numpy as np
import torch

from .obs import MotionReference, RobotState
from .policy import SonicPolicy


def _torch(x) -> torch.Tensor:
    """Read an IsaacLab data field as a torch tensor (Newton hands back warp arrays)."""
    if isinstance(x, torch.Tensor):
        return x
    import warp as wp
    return wp.to_torch(x)


def _np(x) -> np.ndarray:
    """Read an IsaacLab data field as a host numpy array."""
    return _torch(x).detach().cpu().numpy()



class SonicEnvBridge:
    """A drop-in policy callable that runs SONIC against a manager-based env.

    Args:
        env: The unwrapped :class:`ManagerBasedRLEnv`.
        model_dir: Directory with the released ONNX pair and ``observation_config.yaml``.
        command_name: Name of the tracking command term supplying the reference.
        mode: SONIC encoder mode; only ``"g1"`` is supported.
        asset_name: Scene entity name of the robot.
    """

    def __init__(self, env, model_dir, command_name: str = "traj_ref", mode: str = "g1",
                 asset_name: str = "robot") -> None:
        self.env = env
        self.robot = env.scene[asset_name]
        self.cmd = env.command_manager.get_term(command_name)
        self.device = env.device
        self.policy = SonicPolicy(model_dir, num_envs=env.num_envs, mode=mode).load()

        # SONIC's observation and action vectors are in the simulator's breadth-first DOF order of
        # ITS urdf. If the spawned articulation resolves to any other order, every reference
        # channel and every action is silently permuted -- so check rather than trust.
        from robot_rl.assets.robots.g1_sonic import G1_SONIC_ISAAC_JOINT_ORDER
        got, want = list(self.robot.joint_names), list(G1_SONIC_ISAAC_JOINT_ORDER)
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            raise ValueError(
                "the spawned robot's DOF order does not match the order SONIC was trained in; "
                f"they first differ at index {first}: got {got[first:first + 3]}, "
                f"expected {want[first:first + 3]}. Run "
                "'python scripts/rsl_rl/inspect_sonic_asset.py' to see the full order."
            )

        # NO joint permutation is applied to the reference. ``get_future_joint_state`` already
        # answers in the ARTICULATION's DOF order, because the command term's
        # ``_base_idx["joint_pos"]`` is built by iterating ``robot.joint_names`` (see
        # ``base_trajectory_cmd._reference_world_indices``). Permuting again scrambled SONIC's
        # entire reference: the values were right and the arrangement was not, so it commanded a
        # ~1 rad elbow error from step 1, collapsed into a one-legged stance and never locomoted,
        # while every value-wise check (block magnitudes, the trace-replay oracle, the clip
        # comparison) passed -- the oracle is blind to it because it feeds a reference IN and
        # compares the packing, never exercising this path.
        self._verify_reference_joint_order()

        frames, step = self.policy.reference_frames
        self._offsets = torch.arange(frames, device=self.device, dtype=torch.long) * step
        # ``[offsets, offsets + 1]`` in one gather, so the velocity block can be the forward
        # difference their motion_lib defines (see :meth:`_reference`). The command term clamps
        # past the clip end, so the last frames difference to zero rather than reading out of range.
        self._fd_offsets = torch.cat([self._offsets, self._offsets + 1])
        self._first_call = True
        self._last_obs: dict[str, np.ndarray] = {}
        print(f"[INFO] SONIC baseline: mode='{mode}', reference window {frames} frames "
              f"x step {step} ({frames * step * env.step_dt:.2f} s lookahead), "
              f"encoder {self.policy.spec.encoder_dim} -> token {self.policy.spec.token_dim} -> "
              f"{self.policy.spec.decoder_dim} -> 29 actions")

    @property
    def num_actions(self) -> int:
        """Action width, for symmetry with the rsl-rl policies."""
        return 29

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Warm-restart the SONIC history and action feedback for ``env_ids``."""
        self.policy.reset(env_ids)

    def _sync_resets(self) -> None:
        """Clear history for envs the environment reset since the last call."""
        buf = _torch(self.env.episode_length_buf)
        ids = torch.nonzero(buf == 0, as_tuple=False).squeeze(-1)
        if self._first_call:
            self.policy.reset(None)
            self._first_call = False
        elif ids.numel():
            self.policy.reset(ids.detach().cpu().numpy())

    def _verify_reference_joint_order(self) -> None:
        """Fail loudly if the command term stops answering in the articulation's DOF order.

        The bridge relies on that contract and applies no permutation, so a change upstream would
        otherwise silently scramble every reference SONIC sees -- a failure mode that looks like
        bad policy behaviour, not a wiring bug.

        Raises:
            ValueError: If the command's resolved joint index is not ``robot.joint_names`` order.
        """
        idx = getattr(self.cmd, "_base_idx", {}).get("joint_pos")      # noqa: SLF001
        if idx is None:
            return                                                     # nothing to check against
        names = list(self.cmd.ordered_pos_output_names)
        want = [names.index(f"joint:{n}") for n in self.robot.joint_names]
        got = [int(i) for i in _torch(idx).detach().cpu().tolist()]
        if got != want:
            raise ValueError(
                "the tracking command no longer returns future joint channels in the "
                "articulation's DOF order, which SonicEnvBridge relies on (it applies no "
                "permutation). Resolve the order explicitly here before using this baseline.\n"
                f"  expected index (robot order): {want[:6]} ...\n"
                f"  actual index               : {got[:6]} ..."
            )

    def _reference(self) -> MotionReference:
        """The reference window SONIC's g1 mode consumes, in simulator DOF order.

        Velocities are the one-step FORWARD DIFFERENCE of the reference positions, not the clip's
        stored ``dy_des``. That is how the reference velocity SONIC trained on is defined: their
        motion_lib sets ``dof_vels = (dof_pos[1:] - dof_pos[:-1]) / dt`` after resampling
        (``torch_humanoid_batch.fk_batch``). Our ``dy_des`` is the generator's own velocity, which
        agrees at the first frame but drifts from the difference of the stored positions further
        out (measured up to 0.24 rad/s against their env, on a signal averaging 0.44 rad/s), so
        feeding it would hand the policy an out-of-distribution velocity block.
        """
        # Already in articulation DOF order -- see :meth:`_verify_reference_joint_order`.
        jp_all, _ = self.cmd.get_future_joint_state(self._fd_offsets)    # [N, 2K, nj], robot order
        k = self._offsets.numel()
        jp, jp_next = jp_all[:, :k], jp_all[:, k:]
        jv = (jp_next - jp) / self.env.step_dt
        _, quat_w, _, _ = self.cmd.get_future_world_base(self._offsets)  # [N, K, 4] xyzw
        return MotionReference(
            joint_pos=_np(jp),
            joint_vel=_np(jv),
            root_quat=_np(quat_w),
        )

    def _state(self) -> RobotState:
        """This step's robot state, matching SONIC's observation definitions."""
        d = self.robot.data
        return RobotState(
            root_quat=_np(d.root_quat_w),
            base_ang_vel=_np(d.root_ang_vel_b),
            joint_pos_rel=_np(d.joint_pos) - _np(d.default_joint_pos),
            joint_vel=_np(d.joint_vel),
            gravity_dir=_np(d.projected_gravity_b),
        )

    @property
    def last_obs(self) -> dict[str, np.ndarray]:
        """The vectors fed to SONIC on the most recent step, for offline validation.

        The reference blocks of the encoder input depend only on the clip phase, not on the robot
        state, so they can be compared against a trace recorded from NVIDIA's own env at the same
        clip step even though the two rollouts diverge dynamically. That is the only way to check
        this class's reference sourcing -- the trace-replay oracle bypasses it by supplying a
        recorded reference.
        """
        return dict(self._last_obs)

    def __call__(self, obs=None) -> torch.Tensor:
        """One control step. ``obs`` is ignored: SONIC builds its own observation.

        Returns:
            ``[N, 29]`` raw actions on the env device, ready for ``env.step``. The env's action
            term applies SONIC's own scale and default-pose offset.
        """
        self._sync_resets()
        state, ref = self._state(), self._reference()
        action = self.policy.act(state, ref)
        self._last_obs = {
            "encoder_input": self.policy.last_encoder_input,
            "decoder_input": self.policy.last_decoder_input,
            # The clip step each env is on, so a comparison can be phase-aligned.
            "clip_step": _np(self.cmd._step),                              # noqa: SLF001
            "clip_id": _np(self.cmd.clip_id),
            "ref_joint_pos_future": ref.joint_pos,
            "ref_root_quat_future": ref.root_quat,
            "robot_root_quat": state.root_quat,
        }
        return torch.as_tensor(action, device=self.device)
