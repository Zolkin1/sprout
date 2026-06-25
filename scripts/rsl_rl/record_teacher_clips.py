#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Record the teacher's *achieved* motion into a dense clip library.

The teacher policy (``G1-teacher-clf``) tracks the hand-stitched kinematic
clips in ``--source-clip`` open-loop.  Those clips are not perfectly
dynamically realizable, so this script rolls the trained teacher over every
source clip and records the motion it *actually achieved* -- the measured
trajectory-command outputs ``y_act``/``dy_act`` -- back into the **same** HDF5
clip schema (``clips/clip_io.py``).  Because the output format is byte-identical
to a generated library, the result is selected exactly like any other clip
library: point ``traj_ref.path`` at it (the tracker's ``clip_fraction`` still
controls the clip-vs-flow-matching mix).  Tracking the teacher's own achieved
motion gives the tracker an on-distribution, physically consistent reference.

Each source clip is recorded **exactly once**, deterministically: the source
clip ids are tiled across the envs in contiguous batches via the command's
``forced_clip_id`` override (instead of the default random, with-replacement
draw), every termination is disabled so no env auto-resets to a different clip
mid-rollout, and a covered-set check asserts full, non-overlapping coverage.

Example::

    python scripts/rsl_rl/record_teacher_clips.py \\
        --task G1-teacher-clf-play \\
        --load_run 2026-06-22_16-05-10_TM1_teacher_updated_sym \\
        --source-clip clips/g1_flat_v1.h5 \\
        --out clips/g1_teacher_v1.h5 \\
        --num_envs 256 --headless --clip_device cpu

Smoke-test a handful of clips first with ``--max-clips 16 --num_envs 16``.
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_mul, quat_inv

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.clips.clip_io import (
    ClipLibraryWriter,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.world_lift import (
    unbake_outputs_from_world,
)

# local imports
import cli_args  # isort: skip
from recorded_data_io import sha256_file  # isort: skip
from provenance import auto_clip_device, git_snapshot  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Roll the trained teacher over its source clips and record its achieved motion."
)
parser.add_argument("--task", type=str, default="G1-teacher-clf-play", help="Teacher (clip-tracking) task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--num_envs", type=int, default=256, help="Number of parallel envs (clips per batch).")
parser.add_argument(
    "--source-clip", type=str, default="clips/g1_flat_v1.h5",
    help="Clip library the teacher tracks (the clips it was trained on).",
)
parser.add_argument("--out", type=str, required=True, help="Output .h5 teacher-clip library path.")
parser.add_argument(
    "--max-clips", type=int, default=None,
    help="Record only the first N source clips (default: all). Use for smoke tests.",
)
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions stochastically. Default is deterministic (mean) actions.",
)
parser.add_argument(
    "--fail-pos-thresh", type=float, default=0.75,
    help="Max base position drift (m) before a recorded clip is flagged invalid.",
)
parser.add_argument(
    "--fail-ori-thresh", type=float, default=0.8,
    help="Max base orientation error (rad) before a recorded clip is flagged invalid.",
)
parser.add_argument(
    "--clip_device", type=str, default=None,
    help="Device to hold the source clip library on (e.g. 'cpu' to keep a multi-GB "
         "library off a small GPU; gathers copy only the per-env slice). Default: env device.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_resume_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the teacher checkpoint path the same way play.py / record_data.py do.

    Args:
        agent_cfg: The RSL-RL agent configuration (already CLI-updated).

    Returns:
        Absolute path to the checkpoint file to load.
    """
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _disable_terminations(env_cfg) -> None:
    """Disable every termination so no env auto-resets while we record.

    Resets are driven manually, one per batch, so an env must run its assigned
    clip from start to finish without a clip-end / drift / fall reset switching
    it onto a different clip mid-rollout (which would break exactly-once
    coverage).  ``time_out`` is left in place but neutralised by a long
    ``episode_length_s`` (set by the caller).
    """
    for name in ("base_pose_error", "clip_end", "base_fallen"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)


def _base_orientation_error(q_act: torch.Tensor, q_des: torch.Tensor) -> torch.Tensor:
    """Geodesic angle (rad) between two batches of xyzw quaternions, ``[N]``."""
    # angle of the relative rotation q_act^{-1} * q_des; |w| -> 2*acos for the angle.
    q_rel = quat_mul(quat_inv(q_act), q_des)
    w = q_rel[:, 3].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)


def _gather_label(manager, tensor: torch.Tensor, clip_id: torch.Tensor, step: torch.Tensor) -> torch.Tensor | None:
    """Gather a per-step label tensor (skill_id / stride_idx) if present, else None."""
    if tensor is None or tensor.numel() == 0:
        return None
    return manager.gather(tensor, clip_id, step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load the teacher checkpoint, roll it over every source clip, and write the achieved motion."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    resume_path = _resolve_resume_path(agent_cfg)
    log_dir = os.path.dirname(resume_path)

    with launch_simulation(env_cfg, args_cli):
        # Recursive contact sensor patch (matches play.py / record_data.py).
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = log_dir

        # Record the teacher over its training clips, deterministically: source
        # clips, start at step 0, no auto-reset, long episode so time_out never
        # fires inside a clip-length window.
        env_cfg.commands.traj_ref.path = args_cli.source_clip
        env_cfg.commands.traj_ref.random_start_phase = False
        # Never render during recording; the -play configs enable debug markers,
        # whose set_debug_vis path NameErrors on a headless build with no
        # visualizer (omni not imported).  Force it off before the env is built.
        env_cfg.commands.traj_ref.debug_vis = False
        _disable_terminations(env_cfg)

        # Hold the source library off-GPU when large; gathers copy only the per-env slice.
        auto_clip_device(env_cfg, args_cli.clip_device)

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Long enough that time_out never fires within one recorded clip.
        cmd = env.unwrapped.command_manager.get_term("traj_ref")
        T = int(cmd.manager.num_steps)
        dt = float(env.unwrapped.step_dt)
        env.unwrapped.cfg.episode_length_s = (2 * T + 100) * dt

        print(f"[INFO]: Loading teacher checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        runner.alg.eval_mode()
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # rsl-rl < 4.0.0: the inference policy lacks reset(); use the network instead.
        use_policy_reset = version.parse(installed_version) >= version.parse("4.0.0")
        if use_policy_reset:
            policy_nn = None
        elif version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        def _reset_policy(dones=None):
            target = policy if use_policy_reset else policy_nn
            target.reset() if dones is None else target.reset(dones)

        # --- Library layout (copied from the source so the schema is identical) ---
        device = env.unwrapped.device
        n_envs = env.num_envs
        pos_names = list(cmd.ordered_pos_output_names)
        vel_names = list(cmd.ordered_vel_output_names)
        contact_frames = list(cmd.manager.contact_frames)
        P, V, C = len(pos_names), len(vel_names), len(contact_frames)
        num_src = int(cmd.manager.num_clips)
        n_out = num_src if args_cli.max_clips is None else min(num_src, int(args_cli.max_clips))
        if n_out <= 0:
            raise ValueError(f"Nothing to record: n_out={n_out} (source has {num_src} clips).")

        meta = {
            "dt": dt,
            "num_steps": T,
            "clip_length_s": T * dt,
            "pos_output_names": pos_names,
            "vel_output_names": vel_names,
            "contact_frames": contact_frames,
            "ref_frame": cmd.manager.ref_frame,
            "skill_list": list(cmd._clip_lib.meta.get("skill_list", [])),
        }

        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)) or ".", exist_ok=True)
        writer = ClipLibraryWriter(args_cli.out, n_out, meta)
        writer.set_attr("source_clip_path", os.path.abspath(args_cli.source_clip))
        writer.set_attr("source_clip_fingerprint", cmd._clip_lib.attrs.get("clip_fingerprint"))
        writer.set_attr("teacher_clip_provenance", {
            "checkpoint_path": os.path.abspath(resume_path),
            "checkpoint_sha256": sha256_file(resume_path),
            "task_name": args_cli.task,
            "experiment_name": agent_cfg.experiment_name,
            "run_name": os.path.basename(log_dir),
            "num_envs": int(n_envs),
            "stochastic": bool(args_cli.stochastic),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        })

        # Deterministic clip assignment: force the per-env clip id / start step.
        cmd.forced_clip_id = torch.zeros(n_envs, dtype=torch.long, device=device)
        cmd.forced_start_step = torch.zeros(n_envs, dtype=torch.long, device=device)

        base_pos_idx = cmd._base_idx["pos"]
        base_ori_idx = cmd._base_idx["ori"]
        lib = cmd.manager.lib
        has_tvel = cmd.manager.has_target_vel()
        valid_mask = torch.ones(n_out, dtype=torch.bool)
        covered = torch.zeros(n_out, dtype=torch.bool)

        print(f"[INFO] Recording {n_out}/{num_src} source clips x {T} steps @ dt={dt}s "
              f"(P={P}, V={V}, C={C}) on {n_envs} envs -> {args_cli.out}")

        b0 = 0
        while b0 < n_out:
            n = min(n_envs, n_out - b0)
            ids = torch.arange(b0, b0 + n, device=device, dtype=torch.long)
            # Pad spare envs (last partial batch) with a valid dummy id; only the
            # first n envs are written, so the dummy is never recorded.
            cmd.forced_clip_id[:n] = ids
            cmd.forced_clip_id[n:] = 0

            # Reset under inference_mode: the command-term metric tensors are
            # touched inside the step loop's inference_mode below, so zeroing
            # them in CommandTerm.reset must also run under inference_mode
            # (inplace update to an inference tensor is illegal outside it).
            with torch.inference_mode():
                obs, _ = env.reset()
                _reset_policy()

            y_buf = torch.zeros(n_envs, T, P)
            dy_buf = torch.zeros(n_envs, T, V)
            contact_buf = torch.zeros(n_envs, T, C, dtype=torch.uint8)
            tvel_buf = torch.zeros(n_envs, T, 3)
            skill_buf = torch.zeros(n_envs, T, dtype=torch.uint8)
            stride_buf = torch.zeros(n_envs, T, dtype=torch.int32)
            max_pos_err = torch.zeros(n_envs, device=device)
            max_ori_err = torch.zeros(n_envs, device=device)

            with torch.inference_mode():
                for t in range(T):
                    actions = policy(obs, stochastic_output=args_cli.stochastic)
                    obs, _, dones, _ = env.step(actions)
                    _reset_policy(dones)  # no-op: terminations disabled, but keeps state clean

                    # Measured (achieved) world-frame outputs -> clip-world via the
                    # inverse of the per-env placement; store as the new clip's y_des/dy_des.
                    y_cw, dy_cw = unbake_outputs_from_world(
                        cmd.y_act, cmd.dy_act, cmd.clip_world_pos, cmd.clip_world_quat,
                        pos_names, vel_names,
                    )
                    y_buf[:, t] = y_cw.cpu()
                    dy_buf[:, t] = dy_cw.cpu()

                    # Source-aligned labels at the current step (copied through verbatim).
                    cid, step = cmd.clip_id, cmd._step
                    contact_buf[:, t] = cmd.manager.gather_contact(cid, step).to(torch.uint8).cpu()
                    if has_tvel:
                        tvel_buf[:, t] = cmd.manager.gather_target_vel(cid, step).cpu()
                    skill = _gather_label(cmd.manager, lib.skill_id, cid, step)
                    if skill is not None:
                        skill_buf[:, t] = skill.to(torch.uint8).cpu()
                    stride = _gather_label(cmd.manager, lib.stride_idx, cid, step)
                    if stride is not None:
                        stride_buf[:, t] = stride.to(torch.int32).cpu()

                    # Drift tracking (world frame) for the per-clip valid flag.
                    pos_err = (cmd.y_act[:, base_pos_idx] - cmd.y_des[:, base_pos_idx]).norm(dim=-1)
                    ori_err = _base_orientation_error(cmd.y_act[:, base_ori_idx], cmd.y_des[:, base_ori_idx])
                    max_pos_err = torch.maximum(max_pos_err, pos_err)
                    max_ori_err = torch.maximum(max_ori_err, ori_err)

            batch = {
                "y_des": y_buf[:n], "dy_des": dy_buf[:n], "contact": contact_buf[:n],
                "skill_id": skill_buf[:n], "stride_idx": stride_buf[:n], "target_vel": tvel_buf[:n],
            }
            writer.write_batch(b0, batch)

            good = (max_pos_err[:n] < args_cli.fail_pos_thresh) & (max_ori_err[:n] < args_cli.fail_ori_thresh)
            valid_mask[b0:b0 + n] = good.cpu()
            covered[b0:b0 + n] = True
            n_bad = int((~good).sum().item())
            print(f"  recorded clips [{b0}, {b0 + n})  failed={n_bad}  "
                  f"max_pos_err={float(max_pos_err[:n].max()):.3f}m", flush=True)
            b0 += n

        # Exactly-once coverage check, then persist the valid mask and close.
        assert bool(covered.all()), "coverage gap: some clip ids were never recorded"
        assert int(covered.sum()) == n_out, "coverage overlap: a clip id was recorded twice"
        writer.set_attr("valid_clip_mask", valid_mask.tolist())
        writer.close()

        n_valid = int(valid_mask.sum())
        print(f"[INFO] Wrote {n_out} teacher clips to {args_cli.out} "
              f"({n_valid} valid, {n_out - n_valid} flagged > drift thresholds).")
        # Provenance sidecar (git snapshot) next to the output library.
        git_snapshot(os.path.splitext(os.path.abspath(args_cli.out))[0] + ".provenance")

        env.close()


if __name__ == "__main__":
    main()
