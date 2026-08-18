#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Is the generator's terrain segment REGISTERED to the terrain, or carried with the robot?

The clip library already states the answer it should give. ``anchor_mask`` marks a stair or
box segment ``2`` -- FREE = ``['lateral']`` -- meaning downrange and yaw are FIXED TO THE
TERRAIN, while flat walking is ``35`` (downrange/lateral/yaw all free, i.e. carried with the
robot). So displacing the robot gives a falsifiable prediction with no ground truth to
regenerate:

===================  ==========================  ==================================
displace the robot   flat segment (mask 35)      terrain segment (mask 2)
===================  ==========================  ==================================
laterally            moves with the robot        moves with the robot (lateral FREE)
downrange            moves with the robot        **must NOT move in world**
===================  ==========================  ==================================

The downrange case is the test. If the predicted stair climb slides with the robot, the
generator is not localising the terrain at all -- it is replaying a stair-shaped motion at a
fixed distance ahead, which is exactly the "the trajectory is in the wrong place" symptom and
would put the foot on the riser.

Why this and not a prediction-vs-clip error: displacing the robot leaves gait phase, speed and
the commanded velocity untouched, so any world-frame movement of the pinned segment is pure
registration error. An earlier per-node position metric conflated all three and was unusable.

Every env is reset to the SAME clip at the SAME step and differs ONLY by its world offset, so
the depth image is the only changed input. Probing right after reset (null history) isolates
it further: with no history the prediction is a function of exteroception and command alone.

    python scripts/rsl_rl/fm_registration_probe.py --headless \\
        --checkpoint <tracker>.pt --fm_checkpoint <fm>.pt \\
        --clip clips/_tile_stairs_up.h5 --clip_id 0 --start_step 130
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-tracker-clf-fm-terrain-depth-play")
parser.add_argument("--checkpoint", required=True, help="Tracker policy .pt.")
parser.add_argument("--fm_checkpoint", required=True, help="Flow-matching .pt.")
parser.add_argument("--clip", required=True, help="Clip library (use a SINGLE-TILE subset).")
parser.add_argument("--clip_device", default="cpu")
parser.add_argument("--clip_id", type=int, default=0, help="Clip every env is pinned to.")
parser.add_argument("--start_step", type=int, default=-1,
                    help="Clip step to start at (-1 = clip start). Pick one whose horizon "
                         "contains the mask boundary.")
parser.add_argument("--downrange", default="0.0,0.10,0.20,-0.10,-0.20",
                    help="Downrange offsets (m). The terrain segment must NOT move for these.")
parser.add_argument("--lateral", default="0.0,0.10,0.20,-0.10,-0.20",
                    help="Lateral offsets (m). The terrain segment SHOULD move with these.")
parser.add_argument("--settle_steps", type=int, default=0,
                    help="Sim steps before probing. KEEP AT 0 unless you also raise the "
                         "re-plan period: this probe runs at K=1 (see below), and the FM was "
                         "trained with history rows spaced fm_snap_every=12 apart, so a "
                         "FILLED history at K=1 is spaced 1 step and out of distribution. At "
                         "0 the history is null (the learned null embedding) and K cannot "
                         "affect the forward pass at all.")
parser.add_argument("--out", default=None, help="Output .npz (default: alongside the FM run).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Recursive contact-sensor traversal. MUST come after the SimulationApp exists (same note as
# play.py); without it the G1's per-link contact sensors find no bodies and env construction
# dies with "could not find any bodies with contact reporter API".
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402


def main() -> int:
    """Probe world-frame registration of the predicted terrain segment under robot offsets."""
    dr = [float(v) for v in args_cli.downrange.split(",") if v.strip()]
    lat = [float(v) for v in args_cli.lateral.split(",") if v.strip()]
    # env layout: [downrange sweep at lateral 0] + [lateral sweep at downrange 0]
    offsets = [(d, 0.0) for d in dr] + [(0.0, y) for y in lat]
    n_env = len(offsets)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n_env)
    env_cfg.commands.traj_ref.path = args_cli.clip
    env_cfg.commands.traj_ref.clip_device = args_cli.clip_device
    if hasattr(env_cfg.scene, "terrain") and hasattr(env_cfg.scene.terrain, "clip_library_path"):
        env_cfg.scene.terrain.clip_library_path = args_cli.clip
    env_cfg.commands.traj_ref.fm_checkpoint = args_cli.fm_checkpoint
    # K=1 for the probe. This measures the PREDICTION, not the serving policy, and at K=12
    # reading the whole horizon is refused outright: serving offset F-1 needs F >= 62+11 = 73
    # because the cursor runs up to K-1 nodes past the re-plan. K is a serving choice and does
    # not change what the generator predicts from a given observation.
    env_cfg.commands.traj_ref.fm_update_every = 1
    # Same clip, same step, every env -- the offset is the ONLY difference.
    env_cfg.events.reset_on_clip.params["pin_clip_ids"] = " ".join([str(args_cli.clip_id)] * n_env)
    if args_cli.start_step >= 0:
        env_cfg.events.reset_on_clip.params["pin_start_step"] = int(args_cli.start_step)

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    robot = u.scene["robot"]
    cmd = u.command_manager.get_term("traj_ref")

    u.reset()
    # Displace each env in the WORLD after the clip reset has placed it. Yaw is untouched, so
    # "downrange" here is the terrain's traversal axis only insofar as the clip starts aligned
    # with it -- the offsets are applied in the robot's own yaw frame to keep that exact.
    def _torch(x):
        """Whatever the sim hands back (torch / warp / numpy) as a torch tensor.

        IsaacLab's data views are warp-backed on the GPU pipeline, so ``.clone()`` is not
        guaranteed and the type differs by backend.
        """
        if hasattr(x, "clone") and isinstance(x, torch.Tensor):
            return x.clone()
        try:
            import warp as wp
            return wp.to_torch(x).clone()
        except Exception:
            return torch.as_tensor(np.asarray(x), device=u.device)

    root = _torch(robot.data.root_state_w)
    q = root[:, 3:7]
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    off = torch.tensor(offsets, device=root.device, dtype=root.dtype)     # (n, 2) (dr, lat)
    dx = off[:, 0] * torch.cos(yaw) - off[:, 1] * torch.sin(yaw)
    dy = off[:, 0] * torch.sin(yaw) + off[:, 1] * torch.cos(yaw)
    root[:, 0] += dx
    root[:, 1] += dy
    robot.write_root_state_to_sim(root)
    u.sim.step(render=True)
    u.scene.update(u.physics_dt)

    obs = env.get_observations() if hasattr(env, "get_observations") else None
    for _ in range(max(0, args_cli.settle_steps)):
        with torch.inference_mode():
            env.step(torch.zeros(n_env, u.action_manager.total_action_dim, device=u.device))

    with torch.inference_mode():
        u.command_manager.compute(dt=u.step_dt)
        y_win, _ = cmd.predicted_outputs_window()          # (n, F, P) WORLD frame

    pred = _torch(y_win)[:, :, :3].detach().cpu().numpy()   # pelvis world xyz
    anchor = _torch(robot.data.root_pos_w).detach().cpu().numpy()
    out = args_cli.out or os.path.join(
        os.path.dirname(os.path.abspath(args_cli.fm_checkpoint)), "fm_registration_probe.npz")
    np.savez_compressed(out, pred=pred, anchor=anchor, offsets=np.asarray(offsets),
                        n_downrange=len(dr), n_lateral=len(lat),
                        clip_id=args_cli.clip_id, start_step=args_cli.start_step,
                        clip=args_cli.clip)
    print(f"[probe] wrote {out}  pred={pred.shape}")

    base = pred[0]                                          # the (0, 0) offset env
    print(f"\n=== world-frame drift of the PREDICTED path vs the unshifted env ===")
    print(f"  {'offset':>22s} {'mean |dx| (m)':>14s} {'mean |dy| (m)':>14s}   expectation")
    for i, (d, y) in enumerate(offsets):
        if d == 0.0 and y == 0.0:
            continue
        ddx = float(np.abs(pred[i, :, 0] - base[:, 0]).mean())
        ddy = float(np.abs(pred[i, :, 1] - base[:, 1]).mean())
        kind = "downrange" if y == 0.0 else "lateral"
        exp = "should stay ~0 (pinned)" if kind == "downrange" else f"should track {abs(y):.2f}"
        print(f"  {kind:>10s} {d if y == 0.0 else y:+6.2f} m {ddx:14.4f} {ddy:14.4f}   {exp}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
