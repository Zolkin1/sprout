# Copyright 2026 Zachary Olkin. All rights reserved.

"""Torch-free generator + tracker policy (numpy + onnxruntime).

Shared deployment runtime for the MuJoCo sim2sim runner and the hardware
controller. Numpy + onnxruntime port of ``transfer/sim/gen_tracker_policy.py``
(the torch version, validated against Isaac to ~1e-7): runs the flow-matching
generator (``fm.onnx``) online and feeds its world-frame prediction into the RL
tracker (``tracker.onnx``), reproducing the ``GeneratorTrackingCommand`` coupling.
No torch / isaaclab / robot_rl imports.

Same per-step contract as the torch version: ``create_obs(state)`` builds the
tracker observation from the FM prediction + measured proprioception;
``get_action(obs, joint_names_out)`` returns joint position targets.
"""

from __future__ import annotations

import numpy as np

from .fm_session import build_cmd_tokens, build_cond_states, load_fm_context, sample_world
from .history import BatchedHistory
from .quat_math_np import quat_apply, quat_box_minus, quat_inv
from .tracker_session import TrackerSession

# Obs-term clip bounds set in the env cfg but not carried in the metadata.
_TERM_CLIP = {"actions": 20.0, "joint_vel_ref": 20.0}


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a ``(w, x, y, z)`` quaternion (MuJoCo) to ``(x, y, z, w)`` (FM/Isaac)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


class GenTrackerPolicy:
    """Drive the RL tracker with the online flow-matching generator (numpy + ORT)."""

    def __init__(
        self,
        tracker_onnx: str,
        fm_onnx: str,
        fm_update_every: int = 2,
        future_horizon_s: float = 0.5,
        future_stride: int = 5,
        base_frame_name: str = "pelvis_link",
        num_threads: int = 0,
        seed: int | None = 0,
    ):
        """Configure the generator + tracker pair from two ONNX artifacts."""
        self.tracker_onnx = tracker_onnx
        self.fm_onnx = fm_onnx
        self.fm_update_every = max(int(fm_update_every), 1)
        self.future_horizon_s = future_horizon_s
        self.future_stride = future_stride
        self.base_frame_name = base_frame_name
        self.num_threads = num_threads
        self._rng = np.random.default_rng(seed)
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading + setup
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the tracker + FM ONNX sessions and build the orderings/buffers."""
        self._trk = TrackerSession(self.tracker_onnx, self.num_threads)
        # Prefer the cadence baked into the tracker metadata (the value the policy trained
        # with) over the constructor/CLI default, so sim2sim + hardware can't silently drift
        # from training. Falls back to the constructor value when the export predates it.
        meta_every = self._trk.params.get("fm_update_every")
        if meta_every is not None:
            self.fm_update_every = max(int(meta_every), 1)
        print(f"[INFO GenTrackerPolicy] fm_update_every={self.fm_update_every} "
              f"({'from tracker metadata' if meta_every is not None else 'from constructor default'})")
        self.ctx = load_fm_context(self.fm_onnx, self.num_threads)
        if self.ctx.cmd_dim <= 0:
            raise ValueError("GenTrackerPolicy expects a command-conditioned fm.onnx.")
        self._build_orderings()
        self._build_future_offsets()
        F, nj, k = self.ctx.F, self.ctx.nj, self.ctx.target_num_extra_frames
        self._history = BatchedHistory(1, self.ctx.N)
        self._pred = {
            "root_pos": np.zeros((1, F, 3)), "root_quat": np.zeros((1, F, 4)),
            "root_lin": np.zeros((1, F, 3)), "root_ang": np.zeros((1, F, 3)),
            "jp": np.zeros((1, F, nj)), "jv": np.zeros((1, F, nj)),
            "frame_pos": np.zeros((1, F, k, 3)), "frame_quat": np.zeros((1, F, k, 4)),
            "frame_lin": np.zeros((1, F, k, 3)), "frame_ang": np.zeros((1, F, k, 3)),
        }
        self._pred_idx = 0
        self._pred_valid = False
        self._ctrl_step = -1
        self._future_cmd_override = None  # set by clip→FM handoff to feed the actual future cmd
        self._loaded = True

    def _build_orderings(self) -> None:
        """Resolve joint perms between the FM (isaac) order and the CLF output order."""
        isaac = list(self._trk.get_joint_names())
        if list(self.ctx.joint_names) != isaac:
            raise ValueError(
                f"FM joint order {self.ctx.joint_names} != tracker joint_names_isaac {isaac}."
            )
        self.joint_names_isaac = isaac
        joint_out = self._trk.get_joint_out_names()
        if joint_out is None:
            print(
                "[WARN GenTrackerPolicy] tracker.onnx metadata has no 'joint_out_names'; falling "
                "back to joint_names_isaac for joint_pos_ref/joint_vel_ref (WRONG if the CLF output "
                "joint order differs). Re-export the tracker with the updated export_parameters.py."
            )
            joint_out = list(isaac)
        self.joint_out_names = joint_out
        # FM (isaac order) -> CLF output joint order.
        self._fm_to_out = np.array([isaac.index(n) for n in joint_out], dtype=np.int64)

    def _build_future_offsets(self) -> None:
        """Strided future step offsets covering the look-ahead window (>=1 offset)."""
        dt = self.get_dt()
        horizon_steps = max(1, int(round(self.future_horizon_s / dt)))
        stride = max(1, int(self.future_stride))
        offs = list(range(stride, horizon_steps + 1, stride))
        self.future_offsets = offs or [horizon_steps]

    # ------------------------------------------------------------------
    # Snapshot + generator inference
    # ------------------------------------------------------------------
    def _make_snapshot(self, state: dict) -> dict[str, np.ndarray]:
        """Pack one control step of world state into the FM history layout (1 env)."""
        a = lambda x: np.asarray(x, dtype=np.float64)[None]
        snap = {
            "root_pos": a(state["root_pos"]), "root_quat": a(state["root_quat"]),
            "root_lin": a(state["root_lin_vel_w"]), "root_ang": a(state["root_ang_vel_w"]),
            "jp": a(state["joint_pos"]), "jv": a(state["joint_vel"]), "cmd": a(state["command"]),
        }
        if self.ctx.num_extra_frames > 0:
            frames = np.concatenate(
                [state["frame_pos"], state["frame_quat"], state["frame_lin"], state["frame_ang"]], axis=-1
            )
            snap["frames"] = frames[None]
        return snap

    def warm_start(self, state: dict) -> None:
        """Pre-fill the history window with a (standing) snapshot so the FM is valid at step 0."""
        self._history.fill(self._make_snapshot(state))
        self._pred_valid = False
        self._pred_idx = 0

    def observe(self, state: dict) -> None:
        """Append a real snapshot to the history without running the generator.

        Used during the stable-stand warmup: the window fills with measured frames
        while the FM stays dormant (``_pred_valid`` False), so the first live
        ``create_obs`` cleanly triggers the first FM sample anchored on a settled
        stand rather than on a static repeated snapshot.
        """
        self._ctrl_step += 1
        self._history.append(self._make_snapshot(state))

    def _update_generator(self, state: dict) -> None:
        """Append the snapshot and (re-)run or advance the generator prediction."""
        self._ctrl_step += 1
        self._history.append(self._make_snapshot(state))
        valid = bool(self._history.valid()[0])
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        due = (self._ctrl_step % self.fm_update_every) == 0
        if valid and (due or not self._pred_valid or exhausted):
            self._run_fm()
        elif self._pred_valid:
            self._pred_idx = min(self._pred_idx + 1, self.ctx.F - 1)

    def _run_fm(self) -> None:
        """Sample an F-step world trajectory anchored at the current base pose."""
        ctx = self.ctx
        rp = self._history.stack("root_pos")
        rq = self._history.stack("root_quat")
        rl = self._history.stack("root_lin")
        ra = self._history.stack("root_ang")
        jp = self._history.stack("jp")
        jv = self._history.stack("jv")
        cmd = self._history.stack("cmd")
        frames = self._history.stack("frames") if ctx.num_extra_frames > 0 else None

        cond, known = build_cond_states(ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        if self._future_cmd_override is not None:
            # Actual future-command horizon (clip→FM handoff) -- matches IsaacLab's
            # get_command_velocity_horizon. Without it the deploy holds the current
            # command, which is right for joystick (future unknown) but diverges from
            # IsaacLab during command ramps.
            fut_cmd = np.asarray(self._future_cmd_override, dtype=np.float64).reshape(1, ctx.F, ctx.cmd_dim)
        else:
            fut_cmd = np.broadcast_to(cmd[:, -1:], (1, ctx.F, ctx.cmd_dim)).copy()  # held command
        cmd_tokens = build_cmd_tokens(ctx, fut_cmd)
        noise = self._rng.standard_normal((1, ctx.F, ctx.D))
        world = sample_world(ctx, cond, known, rp[:, -1], rq[:, -1], cmd_tokens, noise)

        self._pred["root_pos"] = world["root_pos_w"]
        self._pred["root_quat"] = world["root_quat_w"]
        self._pred["root_lin"] = world["root_link_lin_vel_w"]
        self._pred["root_ang"] = world["root_link_ang_vel_w"]
        self._pred["jp"] = world["joint_pos"]
        self._pred["jv"] = world["joint_vel"]
        if ctx.target_num_extra_frames > 0:
            self._pred["frame_pos"] = world["frame_pos_w"]
            self._pred["frame_quat"] = world["frame_quat_w"]
            self._pred["frame_lin"] = world["frame_lin_vel_w"]
            self._pred["frame_ang"] = world["frame_ang_vel_w"]
        # Frames 0..N-1 reproduce the conditioning; novel future starts at N, so the
        # tracked reference index begins one step ahead of "now" (= N-1).
        self._pred_idx = ctx.N
        self._pred_valid = True

    def _pred_at(self, idx: int) -> dict[str, np.ndarray]:
        """Gather the generator prediction at horizon index ``idx`` (squeezed to 1 env)."""
        idx = min(max(idx, 0), self.ctx.F - 1)
        return {k: v[0, idx] for k, v in self._pred.items()}

    def _reference_from_fm(self) -> dict:
        """Assemble the current + future world reference from the FM prediction buffers."""
        cur = self._pred_at(self._pred_idx)
        fut = [self._pred_at(self._pred_idx + o) for o in self.future_offsets]
        return {"current": cur, "future": fut}

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------
    def create_obs(self, state: dict) -> np.ndarray:
        """Step the generator and build the tracker policy observation ``[1, num_obs]``."""
        if not self._loaded:
            raise RuntimeError("Call load() before create_obs().")
        self._update_generator(state)
        return self._build_obs(state, reference=None)

    def _build_obs(self, state: dict, reference: dict | None) -> np.ndarray:
        """Build the policy obs from proprioception + the world reference (numpy).

        Mirrors ``mdp/observations/observations.py`` exactly (base-frame-rotated
        orientation errors; futures from the FM horizon; held future command).
        """
        ref = reference if reference is not None else self._reference_from_fm()
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        root_pos = f64(state["root_pos"])
        root_quat = f64(state["root_quat"])
        command = f64(state["command"])
        default = self._trk.default_joint_angles
        inv_root = quat_inv(root_quat)

        cur = ref["current"]
        ref_pos = f64(cur["root_pos"]); ref_quat = f64(cur["root_quat"])
        ref_jp = f64(cur["jp"]); ref_jv = f64(cur["jv"])

        pos_err_b = quat_apply(inv_root, ref_pos - root_pos)
        ori_err_b = quat_apply(inv_root, quat_box_minus(root_quat, ref_quat))
        base_pose_error = np.concatenate([pos_err_b, ori_err_b], axis=-1)

        joint_pos_ref = ref_jp[self._fm_to_out]
        joint_vel_ref = ref_jv[self._fm_to_out]

        ee_parts = []
        if self.ctx.target_num_extra_frames > 0:
            inv_ref = quat_inv(ref_quat)
            frame_pos = f64(cur["frame_pos"])
            for kf in range(self.ctx.target_num_extra_frames):
                ee_parts.append(quat_apply(inv_ref, frame_pos[kf] - ref_pos))
        ee_ref = np.concatenate(ee_parts, axis=-1) if ee_parts else np.zeros(0)

        fcmd, fpose, fvel = [], [], []
        for step in ref["future"]:
            # Held current command (FM: no per-step command) or the clip's actual
            # future target_vel when the reference source supplies it (parity with
            # IsaacLab's future_command_velocity, which reads the clip lookahead).
            fcmd.append(f64(step["command"]) if "command" in step else command)
            fp = f64(step["root_pos"]); fq = f64(step["root_quat"])
            fl = f64(step["root_lin"]); fa = f64(step["root_ang"])
            fpose.append(np.concatenate(
                [quat_apply(inv_root, fp - root_pos), quat_apply(inv_root, quat_box_minus(root_quat, fq))], axis=-1
            ))
            fvel.append(np.concatenate([quat_apply(inv_root, fl), quat_apply(inv_root, fa)], axis=-1))

        values = {
            "base_ang_vel": f64(state["base_ang_vel"]),
            "projected_gravity": quat_apply(inv_root, np.array([0.0, 0.0, -1.0])),
            "joint_pos": f64(state["joint_pos"]) - default,
            "joint_vel": f64(state["joint_vel"]),
            "actions": np.asarray(self._trk.action_isaac, dtype=np.float64),
            "base_pose_error": base_pose_error,
            "joint_pos_ref": joint_pos_ref,
            "joint_vel_ref": joint_vel_ref,
            "ee_ref": ee_ref,
            "command_vel": command,
            "future_command_vel": np.concatenate(fcmd, axis=-1),
            "future_base_pose": np.concatenate(fpose, axis=-1),
            "future_base_vel": np.concatenate(fvel, axis=-1),
        }
        return self._assemble(values)

    def _assemble(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Order, scale, and clip the terms per the embedded observation_terms.policy."""
        out = np.zeros(self.get_num_obs(), dtype=np.float32)
        idx = 0
        for term in self._trk.get_obs_terms():
            name, shape, scale = term["name"], term["shape"], term["scale"]
            if name not in values:
                raise NotImplementedError(f"Generator-tracker obs term '{name}' not implemented.")
            v = np.atleast_1d(values[name].astype(np.float32))
            if v.shape[0] != shape:
                raise ValueError(f"Term '{name}': built dim {v.shape[0]} != metadata dim {shape}.")
            if name in _TERM_CLIP:
                v = np.clip(v, -_TERM_CLIP[name], _TERM_CLIP[name])
            out[idx:idx + shape] = v * scale
            idx += shape
        return out[None]

    # ------------------------------------------------------------------
    # Action + passthroughs
    # ------------------------------------------------------------------
    def get_action(self, obs: np.ndarray, joint_names_out: list[str]) -> np.ndarray:
        """Run the tracker; return joint position targets in ``joint_names_out`` order."""
        return self._trk.get_action(obs, joint_names_out)

    def set_future_command(self, future_cmd: np.ndarray | None) -> None:
        """Override the decoder's future-command horizon for subsequent FM samples.

        ``future_cmd`` is ``(F, cmd_dim)`` (the actual upcoming command, e.g. from a
        clip); ``None`` restores the held-current-command default (joystick deploy).
        """
        self._future_cmd_override = None if future_cmd is None else np.asarray(future_cmd, dtype=np.float64)

    def reset(self) -> None:
        """Reset the rolling history + prediction state (the tracker is stateless)."""
        if self._loaded:
            self._history = BatchedHistory(1, self.ctx.N)
            self._pred_idx = 0
            self._pred_valid = False
            self._ctrl_step = -1
            self._future_cmd_override = None

    def get_dt(self) -> float:
        return self._trk.get_dt()

    def get_num_obs(self) -> int:
        return self._trk.get_num_obs()

    def get_num_actions(self) -> int:
        return self._trk.get_num_actions()

    def get_kp(self):
        return self._trk.get_kp()

    def get_kd(self):
        return self._trk.get_kd()

    def get_joint_names(self) -> list[str]:
        return self._trk.get_joint_names()
