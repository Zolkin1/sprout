#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage-1 harness: the Motion Bricks agent plus the pinning hooks every MB driver installs.

Everything the Motion-Bricks-side drivers (``mb_encode.py`` flat and
``mb_terrain_segments.py`` terrain approach/exit) share lives here, so a driver is only its schedule:

* :func:`build_demo_args` / :class:`MBAgent` -- construct the live ``full_navigation_agent`` and install
  the three hooks that turn it into a *pinned* generator: the target-pose override
  (:func:`target_pose_override`, our optimized pose instead of Motion Bricks' style clip),
  ``BYPASS_SPRING_MODEL`` (so our exact per-frame target position wins over the spring model), and the
  regime-gated ``predict`` wrapper (drop the target-pose pins through velocity transitions so the model
  in-betweens instead of snapping).
* :class:`StrideTargeter` / :func:`load_bank` -- velocity-indexed lookup over the optimized stride bank.
* :func:`integrate_lookahead` / :func:`walk_control` -- the exact-velocity root pin: because the pinned
  target positions advance at precisely the command, the achieved velocity at the pins IS the command.
* :func:`load_schema_module` -- import the numpy-only Stage-1/Stage-2 contract modules
  (``raw_schema.py``, ``terrain_plan.py``, ``terrain_families.py``) straight from their source files, so
  both environments share one definition without the MB env needing the ``motion_gen`` package.

Imports the external ``motionbricks`` package -- this module only runs in the ``motionbricks`` conda env.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch as t

_EPS = 1e-4
WALK_MODE = 2                 # index of 'walk' in the G1 CLIPS dict
STAND_MODE = 0                # index of 'idle'/'stand'

# The 29 joints of Motion Bricks' g1.xml, in qpos[7:36] order (== repo G1_JOINT_NAMES).  Hard-coded here
# (Stage 1 has no motion_gen import); Stage 2 reindexes by name, so a mismatch fails loudly.
G1_JOINT_NAMES = (
    [f"{s}_{j}_joint" for s in ("left", "right")
     for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right")
       for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                 "wrist_roll", "wrist_pitch", "wrist_yaw")]
)


# ---- numpy-only contract modules (loaded by path, not imported as a package) ---------------------
def load_schema_module(stem: str):
    """Import ``motion_gen/motionbricks/<stem>.py`` as a standalone (numpy-only) module.

    Avoids importing the ``motion_gen`` package (unavailable in the Motion Bricks env) while keeping a
    single on-disk definition of each cross-environment contract.  Only modules with no intra-package
    imports can be loaded this way -- ``raw_schema``, ``terrain_plan`` and ``terrain_families`` are written
    to that constraint on purpose.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(here, "..", "..", "source", "motion_gen", "motion_gen",
                                        "motionbricks", f"{stem}.py"))
    mod_name = f"mb_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod          # register so @dataclass type resolution finds the module
    spec.loader.exec_module(mod)
    return mod


def load_bank(path: str):
    """Load a stride / stair bank ``.npz`` -> ``(list of qpos[Fi,36], steady_vel[N,3], meta)``.

    Banks (``export_stride_bank.py`` / ``export_stair_bank.py``) store one flat ``qpos_all`` plus per-entry
    offsets because entries have different cycle lengths.
    """
    d = np.load(path, allow_pickle=False)
    qpos_all, offsets = d["qpos_all"], d["offsets"]
    entries = [qpos_all[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    vel_key = "steady_vel" if "steady_vel" in d else "climb_speed"
    vel = d[vel_key].astype(np.float32)
    if vel.ndim == 1:                                          # stair bank: forward climb speed only
        vel = np.stack([vel, np.zeros_like(vel), np.zeros_like(vel)], axis=-1)
    return entries, vel, json.loads(str(d["meta"]))


class StrideTargeter:
    """Velocity-indexed lookup + phase-advancing target windows over an optimized stride bank."""

    def __init__(self, strides, steady_vel, nfpt: int):
        self.strides = strides
        self.steady_vel = steady_vel                          # [N,3]
        self.nfpt = nfpt
        self._phase: dict[int, int] = {}                      # stride_idx -> running phase (frames)

    def nearest(self, cmd: np.ndarray) -> int:
        """Index of the stride whose steady_vel is nearest ``cmd=(vx,vy,wz)`` (stillest on ties)."""
        d = np.sqrt(((self.steady_vel - np.asarray(cmd, np.float32)[None]) ** 2).sum(1))
        return int(np.argmin(d))

    def window(self, idx: int, advance: int = 1) -> np.ndarray:
        """A ``[nfpt, 36]`` window of stride ``idx``, advancing its phase by ``advance`` frames (periodic)."""
        s = self.strides[idx]
        span = max(1, s.shape[0] - self.nfpt)
        ph = (self._phase.get(idx, 0) + advance) % span
        self._phase[idx] = ph
        return s[ph:ph + self.nfpt]


# ---- pose / command helpers ----------------------------------------------------------------------
def cur_pose(ctx: t.Tensor) -> tuple[float, float, float, float]:
    """``(x, y, z, yaw)`` of context frame 0 (world) from a ``[1, nfpt, 36]`` context qpos buffer."""
    c0 = ctx[0, 0].detach().cpu().numpy()
    qw, qx, qy, qz = (float(v) for v in c0[3:7])
    yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return float(c0[0]), float(c0[1]), float(c0[2]), yaw


def integrate_lookahead(x0: float, y0: float, yaw0: float, cmd, times, substep: float = 0.02):
    """World (MuJoCo) root positions ``[K,3]`` + headings ``[K]`` reached by HOLDING a body-frame command.

    Integrates ``(vx forward, vy left, wz yaw-rate)`` from ``(x0, y0, yaw0)`` and samples at each lookahead
    time in ``times`` (increasing).  This is the exact-velocity rollout used as the pinned target: because
    the pinned positions advance at precisely the command, the achieved velocity at the pins IS the command.
    """
    vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
    pos, head = [], []
    x, y, th, tprev = x0, y0, yaw0, 0.0
    for tk in times:
        while tprev < tk - 1e-9:
            h = min(substep, tk - tprev)
            x += (np.cos(th) * vx - np.sin(th) * vy) * h
            y += (np.sin(th) * vx + np.cos(th) * vy) * h
            th += wz * h
            tprev += h
        pos.append([x, y, 0.0])
        head.append(th)
    return np.asarray(pos, np.float32), np.asarray(head, np.float32)


def cmd_at(segs: list[tuple], t_now: float) -> np.ndarray:
    """Piecewise-constant command ``(vx,vy,wz)`` at time ``t_now`` for a ``(duration, vx, vy, wz)`` list."""
    acc = 0.0
    for dur, vx, vy, wz in segs:
        acc += dur
        if t_now < acc:
            return np.array([vx, vy, wz], dtype=np.float32)
    vx, vy, wz = segs[-1][1:]
    return np.array([vx, vy, wz], dtype=np.float32)


def regime_weight(bounds, t_now: float, ramp_s: float = 0.4) -> float:
    """Steady-weight in ``[0,1]`` (smoothstep): 1 far from any command change, 0 within ``ramp_s`` of one.

    Mirrors the motion-matching ``_precompute_regime``: used to relax the target-POSE pin through velocity
    transitions (the root stays pinned, the pose is free to blend), pinning densely only deep in a steady
    hold.  ``bounds`` are the times at which the command changes.
    """
    b = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if b.size == 0:
        return 1.0
    x = min(max(float(np.min(np.abs(t_now - b))) / ramp_s, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def schedule_bounds(segs: list[tuple]) -> np.ndarray:
    """Interior segment boundaries (command-change times) of a ``(duration, vx, vy, wz)`` schedule."""
    return np.cumsum([d for d, *_ in segs])[:-1] if len(segs) > 1 else np.zeros(0)


def walk_control(ctx: t.Tensor, cmd, lookahead_times, *, start=None, standing: bool | None = None,
                 target: tuple[float, float, float] | None = None) -> dict:
    """Build one exact-velocity control dict for ``generate_new_frames``.

    The root is pinned to the exact-velocity rollout of ``cmd`` (``specific_target_positions`` /
    ``_headings``, honoured because the drivers set ``BYPASS_SPRING_MODEL``); ``movement_direction`` carries
    the SIGN of the command (so a negative ``vx`` walks backwards) while ``facing_direction`` keeps the
    heading, and ``target_vel`` the speed magnitude.

    ``target`` replaces that rollout with a FIXED world pose ``(x, y, yaw)`` held at every lookahead time.
    Use it when the destination is known -- a terrain boundary sitting at a definite place on the tile --
    rather than implied by a command.  The difference matters more than it looks: the rollout is anchored
    on the robot's CURRENT pose, so the target travels along with the robot and always sits
    ``|cmd| * horizon`` ahead of it, which is why a decelerating command is what stops the robot (the
    target collapses onto it) and why the terrain's real location never reaches the model.  A fixed target
    inverts that -- the gap closes because the robot arrives, so the deceleration is a consequence of the
    geometry instead of something prescribed.

    Args:
        ctx: the agent's current context qpos ``[1, nfpt, 36]`` (world frame).
        cmd: ``(vx, vy, wz)`` command in the body frame.
        lookahead_times: the target times (s) to pin, increasing.
        start: optional ``(x, y, yaw)`` to integrate from instead of the context pose.
        standing: force stand / walk mode; by default a ~zero command stands.
        target: optional fixed world ``(x, y, yaw)`` to pin instead of integrating ``cmd``.
    """
    cx, cy, _cz, cyaw = cur_pose(ctx) if start is None else (start[0], start[1], 0.0, start[2])
    cmd = np.asarray(cmd, np.float32)
    if target is None:
        spec_pos, spec_head = integrate_lookahead(cx, cy, cyaw, cmd, lookahead_times)
    else:
        k = len(lookahead_times)
        spec_pos = np.tile(np.array([target[0], target[1], 0.0], np.float32), (k, 1))
        spec_head = np.full(k, float(target[2]), np.float32)
    speed = float(np.hypot(cmd[0], cmd[1]))
    if standing is None:
        standing = speed < 1e-3 and abs(float(cmd[2])) < 1e-3
    mv = (np.array([cmd[0], cmd[1], 0.0], np.float32) / speed) if speed > _EPS else np.zeros(3, np.float32)
    face = np.array([np.cos(spec_head[-1]), np.sin(spec_head[-1]), 0.0], np.float32)
    return {
        "mode": t.tensor([[STAND_MODE if standing else WALK_MODE]], dtype=t.int64),
        "movement_direction": t.from_numpy(mv).view([1, 3]),
        "facing_direction": t.from_numpy(face).view([1, 3]),
        "target_vel": t.tensor([speed], dtype=t.float32),
        "specific_target_positions": t.from_numpy(spec_pos).view([1, -1, 3]),
        "specific_target_headings": t.from_numpy(spec_head).view([1, -1]),
        "has_specific_target": t.tensor([[1]], dtype=t.int32),
        "context_mujoco_qpos": ctx,
    }


# ---- the target-pose override ---------------------------------------------------------------------
def target_pose_override(self, input: dict):
    """Replacement ``_generate_target_joint_transforms`` -- target pose from ``self._active_target_qpos``.

    Mirrors the clip-holder pose convention + the ``target_root_realignment`` / ``source_root_realignment``
    blocks of the original (``full_agent.py:321-391``), swapping only the pose SOURCE.
    ``self._active_target_qpos`` is a ``[1, nfpt, 36]`` MuJoCo qpos window (root quat w-first) that the
    driver sets each step: our optimized stride for a steady hold, or the terrain replay's own frames when
    a terrain boundary is being pinned.
    """
    from motionbricks.geometry.quaternions import matrix_to_quaternion
    from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix

    dev = self._device
    my_qpos = self._active_target_qpos.to(dev)                                   # [1,nfpt,36]
    gjp, gjr = self._converter.convert_mujoco_qpos_to_motion_transforms(my_qpos)  # [1,n,34,3],[1,n,34,3,3]

    # clip-holder convention: root ground-projection (height 0); joints root-relative in x/z, absolute height.
    ground = t.tensor([[1.0, 0.0, 1.0]], device=dev)
    global_root_positions = (gjp[0, :, 0] * ground).clone()[None]                  # [1,n,3]
    global_joint_positions = (gjp - global_root_positions[:, :, None, :]).clone()   # [1,n,34,3]
    global_joint_rotations = gjr.clone()                                            # [1,n,34,3,3]
    z_fwd = t.tensor([0.0, 0.0, 1.0], device=dev).view([1, 1, 3, 1])
    root_direction = t.matmul(global_joint_rotations[:, :, 0, :, :], z_fwd)[..., 0]  # [1,n,3]
    global_headings = t.atan2(root_direction[..., 0], root_direction[..., 2])       # [1,n]

    # --- target realignment (full_agent.py:347-359, target_root_realignment=True) ---
    diff_heading = (input['target_root_headings'] - global_headings + t.pi) % (2 * t.pi) - t.pi
    corrective_mat = angle_to_Y_rotation_matrix(diff_heading).float()               # [1,n,3,3]
    global_headings = global_headings + diff_heading
    global_joint_rotations = t.matmul(corrective_mat[:, :, None], global_joint_rotations)
    global_joint_positions = t.matmul(corrective_mat[:, :, None],
                                      global_joint_positions[:, :, :, :, None])[..., 0]
    global_root_positions[:, :, [0, 2]] = input['target_root_positions'].transpose(1, 2).float()

    # --- source (context) realignment (full_agent.py:377-389) ---
    if self._source_root_realignment:
        context_headings = t.atan2(input['context_global_joint_rotations'][:, :, 0, 0, 2],
                                   input['context_global_joint_rotations'][:, :, 0, 2, 2])
        cmat = angle_to_Y_rotation_matrix(input['start_root_headings'] - context_headings).float()
        input['context_global_joint_rotations'] = t.matmul(
            cmat[:, :, None, :, :], input['context_global_joint_rotations'])
        input['context_global_joint_positions'] = t.matmul(
            cmat[:, :, None, :, :], input['context_global_joint_positions'][:, :, :, :, None])[..., 0]
        input['context_global_joint_positions'][:, :, :, [0, 2]] = \
            input['context_global_joint_positions'][:, :, :, [0, 2]] - \
            input['context_global_joint_positions'][:, :, :1, [0, 2]] + \
            input['start_root_positions'].transpose(1, 2)[:, :, None, :].float()

    # Record the pinned target keyframe (for the viser overlay).  This override is called only on an actual
    # replan, so one keyframe is captured per replan.  We take the furthest target frame (the ~1 s lookahead
    # "goal" the model is asked to reach) and rebuild its MuJoCo qpos: joints are the target's (rigid
    # realignment leaves joint ANGLES unchanged), root pos/rot are the realigned target mapped Y-up motion
    # space -> Z-up MuJoCo via the converter's own matrices.
    if getattr(self, "_capture_kf", False):
        k = self.NUM_FRAMES_PER_TOKEN - 1
        m2j = self._converter.motion_to_mujoco_matrix.to(dev).float()
        j2m = self._converter.mujoco_to_motion_matrix.to(dev).float()
        motion_root_pos = global_root_positions[0, k] + global_joint_positions[0, k, 0]   # (x, height, z)
        motion_root_rot = global_joint_rotations[0, k, 0]                                  # [3,3]
        mj_pos = m2j @ motion_root_pos                                                     # canonical MuJoCo pos
        mj_rot = m2j @ motion_root_rot @ j2m                                               # canonical MuJoCo rot
        # These transforms live in the per-replan CANONICAL frame (context re-centred: current root ->
        # origin, facing +x); MB only un-canonicalizes the final decoded output.  Apply the inverse
        # canonicalization ourselves (world = R_heading @ canonical + first_frame_position, the exact inverse
        # of _canonicalize_mujoco_qpos) so the keyframe lands at its true WORLD placement -- otherwise every
        # keyframe piles up near the trajectory start.
        if self.FORCE_CANONICALIZATION and "first_frame_heading_angle" in input:
            h = input["first_frame_heading_angle"].reshape(-1)[0]
            c, s, z, o = t.cos(h), t.sin(h), t.zeros((), device=dev), t.ones((), device=dev)
            r_head = t.stack([t.stack([c, -s, z]), t.stack([s, c, z]), t.stack([z, z, o])])  # Z-up heading rot
            p0 = input["first_frame_position"].reshape(-1)[:3].to(dev)
            mj_pos = r_head @ mj_pos + p0
            mj_rot = r_head @ mj_rot
        mj_quat = matrix_to_quaternion(mj_rot[None])[0]                                    # wxyz
        joints = self._active_target_qpos[0, k, 7:]
        kf = t.cat([mj_pos, mj_quat, joints]).detach().cpu().numpy().astype(np.float32)
        self._captured_kf.append((float(getattr(self, "_active_time_s", 0.0)), kf,
                                  str(getattr(self, "_active_target_kind", "stride"))))

    return global_joint_positions, global_joint_rotations, global_root_positions


def enter_mb_root(args, path_attrs: tuple[str, ...]) -> None:
    """Absolutize the listed path arguments, then ``chdir`` into the Motion Bricks root.

    Motion Bricks' own configs reference assets by paths relative to its package root (e.g.
    ``out/motionbricks_pose/version_1/skeleton/joints.p``), so the agent can only be constructed with that
    root as the working directory.  Resolving OUR paths first means a driver still works when invoked from
    anywhere (the repo root, a worktree, a cron job) with repo-relative arguments.
    """
    for attr in path_attrs:
        val = getattr(args, attr, None)
        if val:
            setattr(args, attr, os.path.abspath(val))
    args.mb_root = os.path.abspath(args.mb_root)
    os.chdir(args.mb_root)


def build_demo_args(args) -> SimpleNamespace:
    """Assemble the argument namespace ``navigation_demo`` expects (mirrors ``interactive_demo_g1.py``)."""
    mb_root = os.path.abspath(args.mb_root)
    skel = os.path.join(mb_root, "assets", "skeletons", "g1")
    return SimpleNamespace(
        humanoid_scene_xml=os.path.join(skel, "scene_29dof.xml"),
        skeleton_xml=os.path.join(skel, "g1.xml"),
        result_dir=os.path.join(mb_root, "out"),
        data_root=os.path.join(mb_root, "datasets"),
        clips_ckpt=os.path.join(mb_root, "out", "G1-clip.ckpt"),
        explicit_dataset_folder=os.path.join(mb_root, "datasets", "motionbricks-G1"),
        reprocess_clips=0,
        controller=getattr(args, "controller", "random"),
        lookat_movement_direction=0,
        pre_filter_qpos=1,
        source_root_realignment=1,
        target_root_realignment=1,
        force_canonicalization=1,
        skip_ending_target_cond=0,
        random_speed_scale=0,
        speed_scale=[1.0, 1.0],
        generate_dt=2.0,
        use_qpos=1,
        planner="default",
        allowed_mode=getattr(args, "allowed_mode", None),
        clips="G1",
        return_model_configs=True,
        return_dataloader=True,
        recording_dir=None,
        EXP="default",
    )


class MBAgent:
    """A live Motion Bricks navigation agent set up as an EXACTLY-PINNED generator.

    Wraps ``navigation_demo``'s ``full_agent`` and installs the three hooks the pinned generation depends
    on (see the module docstring).  Drivers then only script the per-step target: set the target pose
    window (:meth:`set_target_pose`), optionally relax the pose pin (:meth:`set_regime`), and step with a
    control dict from :func:`walk_control`.
    """

    def __init__(self, args, *, capture_keyframes: bool = False, pose_pin_thresh: float = 0.5):
        from motionbricks.motion_backbone.demo.utils import navigation_demo

        self.demo = navigation_demo(build_demo_args(args))
        self.agent = self.demo.full_agent
        self.agent._generate_target_joint_transforms = types.MethodType(target_pose_override, self.agent)
        self.agent._capture_kf = bool(capture_keyframes)
        self.agent._captured_kf = []
        self.agent.BYPASS_SPRING_MODEL = True          # our exact per-frame target position, not the spring
        self.agent._regime_weight = 1.0
        self.agent._regime_pose_pin_thresh = float(pose_pin_thresh)
        nfpt = self.agent.NUM_FRAMES_PER_TOKEN
        _orig_predict = self.agent._inferencer.predict

        def _predict_regime(global_root_values, has_g, local_root_values, has_lr, local_poses, has_lp,
                            num_tokens, *a, **kw):
            """Drop the TARGET-pose pins (keep context + root) while the command is in transition."""
            if float(getattr(self.agent, "_regime_weight", 1.0)) < self.agent._regime_pose_pin_thresh:
                has_lp = has_lp.clone()
                has_lp[:, nfpt:] = False
            return _orig_predict(global_root_values, has_g, local_root_values, has_lr, local_poses, has_lp,
                                 num_tokens, *a, **kw)

        self.agent._inferencer.predict = _predict_regime
        self.fps = float(self.agent._fps)
        self.dt = 1.0 / self.fps
        self.nfpt = int(nfpt)
        self.gen_dt = self.demo.controller.get_controller_dt() * self.demo.args.generate_dt
        self.lookahead_times = [float(self.agent.DEFAULT_PLANING_HORIZON + i / self.fps)
                                for i in range(self.nfpt)]

    def describe(self) -> str:
        """One-line summary of the generation cadence (printed by the drivers)."""
        return (f"fps={self.fps} replan_dt={self.gen_dt:.3f}s (regen every ~{self.gen_dt * self.fps:.0f} "
                f"frames) lookahead={self.lookahead_times[0]:.2f}-{self.lookahead_times[-1]:.2f}s")

    def reset(self, seed: int | None = None) -> None:
        """Reset the agent buffer (and seed both RNGs) before a clip."""
        if seed is not None:
            np.random.seed(int(seed))
            t.manual_seed(int(seed))
        self.agent.reset()
        self.agent._captured_kf = []

    def set_target_pose(self, window: np.ndarray, kind: str = "stride") -> None:
        """Set the ``[nfpt, 36]`` target-pose window the override feeds the model this replan.

        ``kind`` labels where the window came from -- ``"stride"`` (our optimized steady stride) or
        ``"terrain"`` (the replay's own frames, under a hard boundary pin).  It is recorded with the
        captured keyframe purely for diagnostics: the two look identical in the overlay otherwise, and
        the distinction matters because a terrain window's ROOT XY is discarded (see
        :func:`target_pose_override`) -- it pins a pose, not a place.
        """
        self.agent._active_target_qpos = t.as_tensor(np.asarray(window, np.float32),
                                                     dtype=t.float32, device=self.agent._device)[None]
        self.agent._active_target_kind = str(kind)

    def set_regime(self, weight: float, t_now: float | None = None) -> None:
        """Set the steady-weight (``< thresh`` drops the target-pose pins) and the keyframe timestamp."""
        self.agent._regime_weight = float(weight)
        if t_now is not None:
            self.agent._active_time_s = float(t_now)

    def context(self) -> t.Tensor:
        """The agent's current context qpos ``[1, nfpt, 36]`` (world frame, un-canonicalized buffer)."""
        return self.agent.get_context_mujoco_qpos()

    def next_frame(self) -> np.ndarray:
        """Pop the next generated frame ``[36]`` from the buffer."""
        return np.asarray(self.agent.get_next_frame(), dtype=np.float32).copy()

    def step(self, control: dict) -> None:
        """Advance generation with ``control`` (replans only when the buffer needs it)."""
        with t.no_grad():
            self.agent.generate_new_frames(control, self.gen_dt)

    def prime(self, control: dict) -> None:
        """FORCE one generation from ``control``'s context -- how a segment is context-seeded.

        With ``context_mujoco_qpos`` set to a supplied history (e.g. the terrain replay's last frames), the
        forced generation fills the buffer with motion that continues FROM that history, so the frames
        ``next_frame`` then pops start on the terrain instead of wherever the agent happened to be.
        """
        with t.no_grad():
            self.agent.generate_new_frames(control, self.gen_dt, force_generation=True)

    def captured_keyframes(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """The pinned target keyframes recorded this clip: ``(t[K], qpos[K, 36], kind[K])``.

        ``kind`` is ``"stride"`` or ``"terrain"`` per :meth:`set_target_pose`.
        """
        kf = list(self.agent._captured_kf)
        if not kf:
            return np.zeros((0,), np.float32), np.zeros((0, 36), np.float32), []
        return (np.array([x[0] for x in kf], dtype=np.float32), np.stack([x[1] for x in kf], 0),
                [x[2] for x in kf])
