# Copyright 2026 Zachary Olkin. All rights reserved.

"""Capture the spawn states that trip ``ee_pose_error``, for offline inspection in viser.

``ee_pose_error`` never fires mid-episode (0 of 381k steps measured) -- every trip is on the
first step of an episode, where the robot has just been written onto its clip and ought to
have ~0 tracking error.  Something about the spawn disagrees with the body reference by
~0.13 m and this dumps the evidence so it can be looked at rather than inferred.

WHAT INSTANT IS CAPTURED.  Isaac Lab's order is

    terminations -> rewards -> _reset_idx() -> command_manager.compute() -> observations

so the command term recomputes ``y_act`` / ``y_des`` at the END of the reset step, and the
NEXT step's termination judges exactly that pair.  At the top of the following loop iteration
``episode_length_buf == 0``, and what ``tracked_frame_pose`` returns there is precisely what
the termination is about to act on.  So: snapshot every env sitting at age 0, step, and keep
the snapshots of whichever envs then tripped.

Three positions are stored per tracked frame so the viewer can separate the candidate causes:

* ``act``  -- from ``y_act``, i.e. what the CLF and the termination compare.
* ``live`` -- ``body_pos_w`` read at capture time. ``act`` vs ``live`` is measurement lag;
  measured at 0.043 m, so this is NOT the problem.
* ``des``  -- from ``y_des``: the stored ``@root`` frames lifted through the masked
  placement. ``live`` vs ``des`` is the real 0.163 m gap.

Pair with ``scripts/rsl_rl/view_reset_states_viser.py``, which needs no simulator.

Usage (from the repo root, with the venv active)::

    python scripts/rsl_rl/capture_reset_states.py --clip clips/g1_mb_v5_small_body.h5 \\
        --cases 12 --out reset_states.npz
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-teacher-clf-terrain-body", help="Gym task id.")
parser.add_argument("--clip", required=True, help="Clip library .h5.")
parser.add_argument("--checkpoint", default=None,
                    help="model_*.pt to drive the robot. Omit for zero actions -- the spawn "
                         "mismatch shows up either way, since it happens before the policy "
                         "has done anything.")
parser.add_argument("--num_envs", type=int, default=128, help="Envs to sample from.")
parser.add_argument("--cases", type=int, default=12, help="How many tripping spawns to keep.")
parser.add_argument("--steps", type=int, default=400, help="Max rollout steps to search.")
parser.add_argument("--all_spawns", action="store_true",
                    help="Keep every spawn, not only the ones that trip. Useful as a control: "
                         "the clean ones should show ~0 separation.")
parser.add_argument("--clip_device", default="cpu", help="Device to hold the clip library on.")
parser.add_argument("--out", default="reset_states.npz", help="Output .npz.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import robot_rl  # noqa: F401,E402  (registers the gym envs)
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def build_env():
    """Construct the training-condition env pointed at the requested clip library."""
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.commands.traj_ref.path = args.clip
    cfg.commands.traj_ref.clip_device = args.clip_device
    if hasattr(cfg.scene, "terrain") and hasattr(cfg.scene.terrain, "clip_library_path"):
        cfg.scene.terrain.clip_library_path = args.clip
    return gym.make(args.task, cfg=cfg).unwrapped


def load_policy(env):
    """Return an action callable: the checkpoint's policy, or zero actions."""
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    if args.checkpoint is None:
        return lambda _obs: zero
    import inspect

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.utils import resolve_callable

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cfg_d = agent_cfg.to_dict()
    for key in ("actor", "critic"):          # drop deprecated cfg fields hydra would strip
        sec = cfg_d.get(key)
        if isinstance(sec, dict) and "class_name" in sec:
            ok = set(inspect.signature(resolve_callable(sec["class_name"]).__init__).parameters)
            for k in [k for k in sec if k != "class_name" and k not in ok]:
                sec.pop(k)
    runner = OnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions),
                            cfg_d, log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    return runner.get_inference_policy(device=env.device)


def main() -> None:
    """Roll out, snapshot age-0 states, and keep the ones whose next step trips."""
    env = build_env()
    policy = load_policy(env)
    cmd = env.command_manager.get_term("traj_ref")
    asset = env.scene["robot"]
    obs, _ = env.reset()

    # Every tracked body frame, not just the four the termination watches -- if the spawn is
    # wrong it is worth seeing whether it is wrong everywhere or only distally.  Falls back
    # to the four end effectors in joint mode, where there are no @root frames.
    from motion_gen.joint_names import strip_root_marker

    frames = [strip_root_marker(n) for n in getattr(cmd, "root_body_frames", []) or []]
    frames = [f for f in frames if f in asset.body_names] or [
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    ]
    body_idx = [asset.body_names.index(f) for f in frames]
    print(f"[capture_reset_states] tracking {len(frames)} frames: {frames}")
    ee_lim = env.termination_manager.get_term_cfg("ee_pose_error").params["max_pos_dist"]

    cases: list[dict] = []
    pending: dict | None = None
    for _ in range(args.steps):
        if len(cases) >= args.cases:
            break
        with torch.inference_mode():
            age0 = (env.episode_length_buf == 0)
            pending = None
            if age0.any():
                act_p, act_q, des_p, des_q = cmd.tracked_frame_pose(frames)
                pending = {
                    "envs": torch.nonzero(age0, as_tuple=False).squeeze(-1).clone(),
                    "joint_pos": wp.to_torch(asset.data.joint_pos).clone(),
                    "root_pos": wp.to_torch(asset.data.root_pos_w).clone(),
                    "root_quat": wp.to_torch(asset.data.root_quat_w).clone(),
                    "live_p": wp.to_torch(asset.data.body_pos_w)[:, body_idx, :].clone(),
                    "act_p": act_p.clone(), "act_q": act_q.clone(),
                    "des_p": des_p.clone(), "des_q": des_q.clone(),
                    "clip_id": cmd.clip_id.clone(), "start_step": cmd.start_step.clone(),
                }
            obs, _, terminated, truncated, _ = env.step(policy(obs))

        if pending is None:
            continue
        tripped = env.termination_manager.get_term("ee_pose_error")
        keep = tripped if not args.all_spawns else (terminated | truncated)
        sel = [int(e) for e in pending["envs"] if bool(keep[int(e)])]
        for e in sel:
            if len(cases) >= args.cases:
                break
            cases.append({k: (v[e].cpu().numpy() if torch.is_tensor(v) and k != "envs" else None)
                          for k, v in pending.items()} | {"env_id": e})

    if not cases:
        raise SystemExit("[capture_reset_states] no cases captured -- try more --steps.")

    stack = lambda k: np.stack([c[k] for c in cases])          # noqa: E731
    np.savez_compressed(
        args.out,
        frame_names=np.array(frames),
        joint_names=np.array(list(asset.joint_names)),
        joint_pos=stack("joint_pos"), root_pos=stack("root_pos"), root_quat=stack("root_quat"),
        live_p=stack("live_p"), act_p=stack("act_p"), act_q=stack("act_q"),
        des_p=stack("des_p"), des_q=stack("des_q"),
        clip_id=stack("clip_id"), start_step=stack("start_step"),
        env_id=np.array([c["env_id"] for c in cases]), ee_limit=np.array(ee_lim),
    )
    worst = np.linalg.norm(stack("act_p") - stack("des_p"), axis=-1).max(axis=-1)
    print(f"\n[capture_reset_states] {len(cases)} cases -> {args.out}")
    print(f"  worst act-vs-des separation per case: "
          f"{np.array2string(worst, precision=3, max_line_width=100)}")
    print(f"  ee_pose_error limit = {ee_lim} m")
    simulation_app.close()


if __name__ == "__main__":
    main()
