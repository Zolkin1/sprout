# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generator + tracker policy for sim2sim / hardware deployment.

Runs the flow-matching **generator** (vendored ``transfer/fm``) online and feeds
its world-frame prediction into the RL **tracker** policy, reproducing the
``FmTrackingCommand`` coupling outside Isaac Lab (no isaaclab / robot_rl, no clip
library). Pure-FM: the velocity command comes from a joystick or a scripted
profile, and every reference observation term is sourced from the FM prediction
(the future terms from the FM horizon).

Per control step (mirrors ``fm_tracking_cmd._snapshot_and_infer`` +
``_update_command`` for a single env):

1. Append the LIVE MEASURED robot state to a rolling history (the N phase-history
   tokens; the measured stand seeds it during the warmup).
2. Every ``fm_update_every`` steps, sample an ``F``-step world-frame trajectory
   conditioned on that robot-state history, a root-height token, and the (held)
   command horizon, anchored on the robot's current base pose (node 0 = "now").
   Otherwise advance the read index through the existing prediction.
3. Build the tracker's policy observation from the prediction (current step +
   future window) and the measured proprioception.

The generator is **robot-state-conditioned**: the N history tokens are the real
robot state (phase), NOT the FM's own past output -- this fixes the autoregressive
z-drift / exposure bias. The FM conditions on AND predicts the same end-effector
frames (``ctx.num_extra_frames``); the history carries the measured frames. The FM
joint order need not equal the robot's isaac order (the FM is trained in the
clip-library order). The class is fully checkpoint-driven.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

from rl_policy import RLPolicy

# Vendored, isaaclab-free flow-matching inference (transfer/ on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fm import BatchedHistory, load_fm_context  # noqa: E402
from fm.online import build_cmd_tokens, build_height_token, build_history_tokens, sample_world  # noqa: E402
from fm.quat_math import quat_apply, quat_box_minus, quat_inv  # noqa: E402

# Observation clip bounds that are set on the obs terms in the env cfg
# (G1TeacherObservationsCfg) but not carried in policy_parameters.yaml.
_TERM_CLIP = {"actions": 20.0, "joint_vel_ref": 20.0}


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a ``(w, x, y, z)`` quaternion (MuJoCo) to ``(x, y, z, w)`` (FM/Isaac)."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


class GeneratorTrackerPolicy:
    """Drive the RL tracker with the online flow-matching generator (FM-only)."""

    def __init__(
        self,
        policy_params_path: str,
        policy_path: str,
        fm_checkpoint: str,
        device: str = "cpu",
        fm_device: str | None = None,
        fm_update_every: int = 2,
        num_sampling_steps: int | None = None,
        future_horizon_s: float = 0.5,
        future_stride: int = 5,
        base_frame_name: str = "pelvis_link",
    ):
        """Configure the generator + tracker pair.

        Args:
            policy_params_path: Path to the tracker's ``policy_parameters.yaml``.
            policy_path: Path to the tracker's TorchScript ``policy.pt``.
            fm_checkpoint: Path to the flow-matching generator ``.pt`` checkpoint.
            device: Device for the tracker policy + observation math.
            fm_device: Device for the generator (defaults to ``device``).
            fm_update_every: Re-run the generator every N control steps.
            num_sampling_steps: Override the generator's Euler step count.
            future_horizon_s: Look-ahead window for the future obs terms.
            future_stride: Sub-sampling stride (in control steps) within the window.
            base_frame_name: Base/root link the CLF + FM are anchored on.
        """
        self._rl = RLPolicy(policy_params_path, policy_path)  # param handling + get_action
        self.policy_path = policy_path
        self.policy_params_path = policy_params_path
        self.fm_checkpoint = fm_checkpoint
        self.device = torch.device(device)
        self.fm_device = torch.device(fm_device or device)
        self.fm_update_every = max(int(fm_update_every), 1)
        self.num_sampling_steps = num_sampling_steps
        self.future_horizon_s = future_horizon_s
        self.future_stride = future_stride
        self.base_frame_name = base_frame_name

        self.ctx = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading + setup
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the tracker TorchScript policy and the flow-matching generator."""
        self._rl.policy = torch.jit.load(self.policy_path, map_location=self.device)
        self._rl.policy.eval()
        self._rl.reset()

        self.ctx = load_fm_context(
            self.fm_checkpoint, device=self.fm_device,
            num_sampling_steps=self.num_sampling_steps,
        )
        if self.ctx.cmd_dim <= 0:
            raise ValueError(
                "GeneratorTrackerPolicy expects a command-conditioned (cmd_array='target_vel') "
                "flow-matching checkpoint; the loaded one has no command conditioning."
            )

        self._build_orderings()
        self._build_future_offsets()
        n, F, nj, k = 1, self.ctx.F, self.ctx.nj, self.ctx.target_num_extra_frames
        self._history = BatchedHistory(n, self.ctx.N, self.fm_device)
        self._pred = {
            "root_pos": torch.zeros(n, F, 3, device=self.fm_device),
            "root_quat": torch.zeros(n, F, 4, device=self.fm_device),
            "root_lin": torch.zeros(n, F, 3, device=self.fm_device),
            "root_ang": torch.zeros(n, F, 3, device=self.fm_device),
            "jp": torch.zeros(n, F, nj, device=self.fm_device),
            "jv": torch.zeros(n, F, nj, device=self.fm_device),
            "frame_pos": torch.zeros(n, F, k, 3, device=self.fm_device),
            "frame_quat": torch.zeros(n, F, k, 4, device=self.fm_device),
            "frame_lin": torch.zeros(n, F, k, 3, device=self.fm_device),
            "frame_ang": torch.zeros(n, F, k, 3, device=self.fm_device),
        }
        self._pred_idx = 0
        self._pred_valid = False
        self._ctrl_step = -1
        self._warm = False
        self._loaded = True

    def _build_orderings(self) -> None:
        """Resolve the joint/frame permutations between the robot (isaac), the FM, and the CLF.

        - ``joint_names_isaac`` is the robot joint order -- the tracker's
          ``joint_pos``/``joint_vel``/``actions`` order.
        - ``ctx.joint_names`` is the FM joint order; it need NOT equal the isaac order
          (a clip-conditioned FM is trained in the clip-library order). ``_isaac_to_fm``
          permutes the measured joints into the FM order for the robot-state history.
        - ``joint_out_names`` is the CLF output joint order (clip
          ``ordered_pos_output_names``); ``joint_pos_ref``/``joint_vel_ref`` use it.
          ``_fm_to_out`` permutes the FM prediction (FM order) into it. It must be
          exported in ``policy_parameters.yaml``; falls back to the isaac order.
        - ``ee_ref`` frame order == the FM's predicted (target) frame order.
        """
        isaac = list(self._rl.get_joint_names())
        self.joint_names_isaac = isaac
        fm = list(self.ctx.joint_names)
        joint_out = self._resolve_joint_out_names(isaac)
        self.joint_out_names = joint_out
        # Measured (isaac order) -> FM order, for the robot-state history input.
        self._isaac_to_fm = torch.tensor(
            [isaac.index(n) for n in fm], dtype=torch.long, device=self.fm_device
        )
        # FM prediction (FM order) -> CLF output joint order, for joint_pos_ref / joint_vel_ref.
        self._fm_to_out = torch.tensor(
            [fm.index(n) for n in joint_out], dtype=torch.long, device=self.fm_device
        )
        # ee_ref frames == FM target frames (sorted ee), already in prediction order.
        self.ee_frame_names = list(self.ctx.target_frame_names)

    def _resolve_joint_out_names(self, isaac: list[str]) -> list[str]:
        """Get the CLF-output joint order from the exported params, else fall back."""
        params = self._rl.policy_params
        if "joint_out_names" in params:
            return list(params["joint_out_names"])
        if "ordered_pos_output_names" in params:
            names = params["ordered_pos_output_names"]
            if isinstance(names, str):
                names = json.loads(names)
            return [n.split(":", 1)[1] for n in names if n.startswith("joint:")]
        print(
            "\n" + "=" * 78 + "\n[WARN GeneratorTrackerPolicy] policy_parameters.yaml has no "
            "'joint_out_names'/'ordered_pos_output_names'; falling back to joint_names_isaac "
            "for joint_pos_ref/joint_vel_ref. This is WRONG if the CLF output joint order "
            "differs from the isaac order (it does for the g1 clips). Export it from "
            "export_parameters.py (cmd.ordered_pos_output_names).\n" + "=" * 78 + "\n"
        )
        return list(isaac)

    def _build_future_offsets(self) -> None:
        """Strided future step offsets covering the look-ahead window (>=1 offset)."""
        dt = self.get_dt()
        horizon_steps = max(1, int(round(self.future_horizon_s / dt)))
        stride = max(1, int(self.future_stride))
        offs = list(range(stride, horizon_steps + 1, stride))
        self.future_offsets = offs or [horizon_steps]

    # ------------------------------------------------------------------
    # Per-step state snapshot + generator inference
    # ------------------------------------------------------------------
    def _make_snapshot(self, state: dict) -> dict[str, torch.Tensor]:
        """Pack one MEASURED world state into the FM history layout (1 env).

        Appended to the rolling history every control step (and used to warm-start
        it): the robot-state-conditioned FM's history IS the measured robot state.
        ``state`` keys (numpy, robot/isaac joint order, world frame, xyzw quats):
        ``root_pos``,
        ``root_quat``, ``root_lin_vel_w``, ``root_ang_vel_w``, ``joint_pos``,
        ``joint_vel``, ``command``, and -- only when the history token carries extra
        frames -- ``frame_pos``/``frame_quat``/``frame_lin``/``frame_ang`` each
        ``(K_hist, ·)``. Measured joints are permuted isaac->FM order.
        """
        tf = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.fm_device)
        jp, jv = tf(state["joint_pos"]), tf(state["joint_vel"])
        if self._isaac_to_fm is not None:
            jp, jv = jp[self._isaac_to_fm], jv[self._isaac_to_fm]
        snap = {
            "root_pos": tf(state["root_pos"])[None],
            "root_quat": tf(state["root_quat"])[None],
            "root_lin": tf(state["root_lin_vel_w"])[None],
            "root_ang": tf(state["root_ang_vel_w"])[None],
            "jp": jp[None],
            "jv": jv[None],
            "cmd": tf(state["command"])[None],
        }
        if self.ctx.num_extra_frames > 0:
            frames = np.concatenate(
                [state["frame_pos"], state["frame_quat"], state["frame_lin"], state["frame_ang"]], axis=-1
            )  # (K, 13)
            snap["frames"] = tf(frames)[None]
        return snap

    def warm_start(self, state: dict) -> None:
        """Pre-fill the history window with a (standing) snapshot so the FM is valid at step 0."""
        self._history.fill(self._make_snapshot(state))
        self._pred_valid = False
        self._pred_idx = 0
        self._warm = True

    def _update_generator(self, state: dict) -> None:
        """Append the LIVE robot-state snapshot and (re-)run or advance the prediction.

        The appended snapshot is always the MEASURED robot state (permuted into FM
        joint/frame order) -- the robot-state-conditioned FM conditions on the real
        robot history (phase), not its own past output. This mirrors
        ``FmTrackingCommand._snapshot_and_infer`` for a single env.
        """
        self._ctrl_step += 1
        self._history.append(self._make_snapshot(state))

        valid = bool(self._history.valid()[0].item())
        exhausted = self._pred_idx >= (self.ctx.F - 1)
        due = (self._ctrl_step % self.fm_update_every) == 0
        if valid and (due or not self._pred_valid or exhausted):
            self._run_fm(state)
        elif self._pred_valid:
            self._pred_idx = min(self._pred_idx + 1, self.ctx.F - 1)

    @torch.no_grad()
    def _run_fm(self, state: dict) -> None:
        """Sample an F-step world trajectory anchored on the robot's current base pose.

        Conditions the FM on the N most-recent MEASURED robot states (phase history),
        a root-height token (live root z), and the held future velocity command, then
        lifts the F-step ego-anchored prediction back to the world using the robot's
        current base pose as the xyz+yaw anchor (node 0 = "now", base offset ~= 0).
        Mirrors ``FmTrackingCommand._run_fm`` for a single env.
        """
        ctx = self.ctx
        rp = self._history.stack("root_pos")
        rq = self._history.stack("root_quat")
        rl = self._history.stack("root_lin")
        ra = self._history.stack("root_ang")
        jp = self._history.stack("jp")
        jv = self._history.stack("jv")
        cmd = self._history.stack("cmd")
        frames = self._history.stack("frames") if ctx.num_extra_frames > 0 else None

        history = build_history_tokens(ctx, rp, rq, rl, ra, jp, jv, frames=frames)
        height = build_height_token(ctx, rp[:, -1, 2])
        # Held command over the future horizon (no clip to read ahead from).
        fut_cmd = cmd[:, -1:].expand(1, ctx.F, ctx.cmd_dim).clone()
        cmd_tokens = build_cmd_tokens(ctx, fut_cmd)

        # The robot NOW is the anchor: the most-recent history step IS the live robot
        # state, so its base pose is the xyz+yaw ground frame the FM was trained on.
        p_ref = rp[:, -1].clone()
        q_ref = rq[:, -1].clone()
        world = sample_world(ctx, history, height, cmd_tokens, p_ref, q_ref)

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
        # Node 0 = "now" (base offset ~= 0); the applied index advances from here.
        self._pred_idx = 0
        self._pred_valid = True

    def _pred_at(self, idx: int) -> dict[str, torch.Tensor]:
        """Gather the generator prediction at horizon index ``idx`` (1 env -> squeezed)."""
        idx = min(max(idx, 0), self.ctx.F - 1)
        return {k: v[0, idx] for k, v in self._pred.items()}

    # ------------------------------------------------------------------
    # Observation construction (the 207-dim tracker policy vector)
    # ------------------------------------------------------------------
    def create_obs(self, state: dict) -> torch.Tensor:
        """Step the generator and build the tracker's policy observation ``[1, num_obs]``.

        Args:
            state: World-state dict (see :meth:`_make_snapshot`) plus ``base_ang_vel``
                (base-frame angular velocity, the Isaac ``base_ang_vel`` term).

        Returns:
            Observation tensor ``[1, num_obs]`` on ``self.device``.
        """
        if not self._loaded:
            raise RuntimeError("Call load() before create_obs().")
        self._update_generator(state)
        return self._build_obs(state, reference=None)

    def _reference_from_fm(self) -> dict:
        """Assemble the current + future world reference from the FM prediction buffers."""
        cur = self._pred_at(self._pred_idx)
        fut = [self._pred_at(self._pred_idx + o) for o in self.future_offsets]
        return {"current": cur, "future": fut}

    def _build_obs(self, state: dict, reference: dict | None) -> torch.Tensor:
        """Build the 207-dim policy obs from proprioception + the world reference.

        ``reference`` may be injected (parity testing); otherwise it is read from
        the live FM prediction. All math is in float64 on ``fm_device`` then moved
        to ``device`` -- it mirrors ``mdp/observations/observations.py`` exactly,
        using the vendored xyzw quaternion ops.
        """
        ref = reference if reference is not None else self._reference_from_fm()
        dev = self.fm_device

        def t(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float64, device=dev)

        root_pos = t(state["root_pos"])            # (3,)
        root_quat = t(state["root_quat"])          # (4,) xyzw
        joint_pos = t(state["joint_pos"])          # (nj,) isaac order
        joint_vel = t(state["joint_vel"])
        command = t(state["command"])              # (3,)
        default = t(self._rl.default_joint_angles)

        cur = ref["current"]
        ref_pos = cur["root_pos"].to(torch.float64)
        ref_quat = cur["root_quat"].to(torch.float64)
        ref_jp = cur["jp"].to(torch.float64)
        ref_jv = cur["jv"].to(torch.float64)
        inv_root = quat_inv(root_quat)

        # -- term 6: base_pose_error -- pos + ori error, BOTH in the current base
        # frame (heading-invariant). Matches observations.base_pose_error: the
        # world-frame quat_box_minus rotation vector is rotated into the base frame.
        pos_err_b = quat_apply(inv_root, ref_pos - root_pos)
        ori_err_b = quat_apply(inv_root, quat_box_minus(root_quat, ref_quat))
        base_pose_error = torch.cat([pos_err_b, ori_err_b], dim=-1)

        # -- terms 7/8: joint refs in the CLF-output joint order --
        joint_pos_ref = ref_jp[self._fm_to_out]
        joint_vel_ref = ref_jv[self._fm_to_out]

        # -- term 9: ee_ref -- predicted ee positions in the reference base frame --
        ee_parts = []
        if self.ctx.target_num_extra_frames > 0:
            inv_ref_quat = quat_inv(ref_quat)
            frame_pos = cur["frame_pos"].to(torch.float64)   # (K, 3)
            for kf in range(self.ctx.target_num_extra_frames):
                ee_parts.append(quat_apply(inv_ref_quat, frame_pos[kf] - ref_pos))
        ee_ref = torch.cat(ee_parts, dim=-1) if ee_parts else torch.zeros(0, device=dev, dtype=torch.float64)

        # -- term 10: base_vel_ref -- reference base lin+ang velocity (world frame)
        # rotated into the current base frame (observations.ref_base_velocity_local).
        # Guarded so an injected reference without velocity channels (parity tests)
        # simply omits the term.
        base_vel_ref = None
        if "root_lin" in cur:
            ref_lin = cur["root_lin"].to(torch.float64)
            ref_ang = cur["root_ang"].to(torch.float64)
            base_vel_ref = torch.cat([quat_apply(inv_root, ref_lin), quat_apply(inv_root, ref_ang)], dim=-1)

        # -- terms 11/12/13: future window. future_base_pose/vel come from the FM
        # horizon (matches GeneratorTrackingCommand.get_future_world_base, now
        # FM-sourced); future_command_vel is the held command (deployment has no
        # clip to read ahead from -- a known, intended difference from training,
        # which still reads target_vel ahead from the clip).
        fut = ref["future"]
        fcmd, fpose, fvel = [], [], []
        for step in fut:
            fcmd.append(command)  # held command over the horizon
            fp = step["root_pos"].to(torch.float64)
            fq = step["root_quat"].to(torch.float64)
            fl = step["root_lin"].to(torch.float64)
            fa = step["root_ang"].to(torch.float64)
            fpose.append(torch.cat(
                [quat_apply(inv_root, fp - root_pos), quat_apply(inv_root, quat_box_minus(root_quat, fq))], dim=-1
            ))
            fvel.append(torch.cat([quat_apply(inv_root, fl), quat_apply(inv_root, fa)], dim=-1))
        future_command_vel = torch.cat(fcmd, dim=-1)
        future_base_pose = torch.cat(fpose, dim=-1)
        future_base_vel = torch.cat(fvel, dim=-1)

        last_action = t(self._rl.action_isaac)

        values = {
            "base_ang_vel": t(state["base_ang_vel"]),
            "projected_gravity": self._projected_gravity(root_quat),
            "joint_pos": joint_pos - default,
            "joint_vel": joint_vel,
            "actions": last_action,
            "base_pose_error": base_pose_error,
            "joint_pos_ref": joint_pos_ref,
            "joint_vel_ref": joint_vel_ref,
            "ee_ref": ee_ref,
            "command_vel": command,
            "future_command_vel": future_command_vel,
            "future_base_pose": future_base_pose,
            "future_base_vel": future_base_vel,
        }
        if base_vel_ref is not None:
            values["base_vel_ref"] = base_vel_ref
        return self._assemble(values)

    def _projected_gravity(self, root_quat_xyzw: torch.Tensor) -> torch.Tensor:
        """Gravity ``[0,0,-1]`` rotated into the base frame (the Isaac projected_gravity)."""
        g = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64, device=root_quat_xyzw.device)
        return quat_apply(quat_inv(root_quat_xyzw), g)

    def _assemble(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        """Order, scale, and clip the terms per the exported observation_terms.policy."""
        terms = self._rl.get_obs_terms()
        out = np.zeros(self.get_num_obs(), dtype=np.float32)
        idx = 0
        for term in terms:
            name, shape, scale = term["name"], term["shape"], term["scale"]
            if name not in values:
                raise NotImplementedError(f"Generator-tracker obs term '{name}' not implemented.")
            v = values[name].detach().to("cpu").numpy().astype(np.float32)
            if v.shape[0] != shape:
                raise ValueError(f"Term '{name}': built dim {v.shape[0]} != exported dim {shape}.")
            if name in _TERM_CLIP:
                v = np.clip(v, -_TERM_CLIP[name], _TERM_CLIP[name])
            out[idx:idx + shape] = v * scale
            idx += shape
        return torch.from_numpy(out).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Action + passthroughs to the composed RLPolicy
    # ------------------------------------------------------------------
    def get_action(self, obs: torch.Tensor, joint_names_out: list[str]) -> np.ndarray:
        """Run the tracker; return joint position targets in ``joint_names_out`` order.

        Runs on ``self.device`` (the jit is loaded there) and caches the raw action
        on the composed RLPolicy so the next step's ``actions`` obs term reads it.
        """
        with torch.no_grad():
            raw = self._rl.policy(obs.to(self.device)).detach().cpu().numpy().squeeze()
        self._rl.action_isaac = raw
        target = raw * self._rl.action_scale + self._rl.default_joint_angles
        return self._rl.convert_joint_order(target, self._rl.get_joint_names(), joint_names_out)

    def reset(self) -> None:
        """Reset the tracker recurrent state, history buffer, and prediction buffers."""
        self._rl.reset()
        if self._loaded:
            self._history = BatchedHistory(1, self.ctx.N, self.fm_device)
            self._pred_idx = 0
            self._pred_valid = False
            self._ctrl_step = -1
            self._warm = False

    def get_dt(self) -> float:
        return self._rl.get_dt()

    def get_num_obs(self) -> int:
        return self._rl.get_num_obs()

    def get_num_actions(self) -> int:
        return self._rl.get_num_actions()

    def get_kp(self):
        return self._rl.get_kp()

    def get_kd(self):
        return self._rl.get_kd()

    def get_joint_names(self) -> list[str]:
        return self._rl.get_joint_names()
