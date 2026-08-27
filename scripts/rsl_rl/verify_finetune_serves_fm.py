#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Verify the fine-tune task serves the GENERATOR's reference window and grades the CLIP.

``G1-teacher-clf-terrain-depth-finetune`` rests on one property that no config assertion can
establish: at observation time the reference window comes from the generator, and by the time
control returns, the applied reference is the clip again -- so the reward, the critic and the
clip-graded terminations never see the swap.

Both halves of that fail SILENTLY if they break.  A generator that quietly falls back serves a
window identical to the clip, which trains a policy that is merely the teacher again -- no
error, no shape change, and a fine-tune that does nothing.  A swap that leaks the other way
leaves the CLF grading the generator, which is the reward re-pointing this task exists to
avoid.  So both are measured here rather than assumed.

Run (from the repo root, ``env_isaaclab`` active)::

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/rsl_rl/verify_finetune_serves_fm.py \\
        --clip clips/g1_mb_v5_body_mode11_decor.h5 --clip_device cpu \\
        --fm_checkpoint logs/flow_matching/<run>/best.pt

Exits non-zero if the served window matches the clip (a silent fallback), if the clip serve is
not restored, or if any served value is non-finite.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-teacher-clf-terrain-depth-finetune")
parser.add_argument("--clip", required=True, help="clip library .h5 (no default, by design)")
parser.add_argument("--clip_device", default="cpu")
parser.add_argument("--fm_checkpoint", required=True, help="the generator to serve")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--atol", type=float, default=1e-4,
                    help="below this the served window is treated as a silent clip fallback")
AppLauncher.add_app_launcher_args(parser)
# parse_known_args + the argv rewrite below, exactly as train.py does it: hydra_task_config
# re-parses sys.argv itself and rejects anything it does not own, so this script's own flags
# have to be taken out of the way first.
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AFTER SimulationApp, exactly as train.py does it: the contact sensors this task declares
# need the recursive-traversal patch, and importing it earlier is a no-op.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: E402,F401

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import robot_rl.tasks  # noqa: E402,F401  (registers the task ids)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl import mdp  # noqa: E402
from robot_rl.tasks.manager_based.robot_rl.g1.g1_teacher_env_cfg import (  # noqa: E402
    REFERENCE_OFFSETS,
    REFERENCE_WINDOW_BLOCKS,
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    """Step the fine-tune env and compare the two reference windows at identical states.

    Composed through hydra rather than ``parse_env_cfg`` so the env is built the way
    ``train.py`` builds it -- the two paths do not resolve the same config.

    Args:
        env_cfg: The task's env config, composed by hydra.
        agent_cfg: The paired rsl-rl config; unused, but the decorator supplies it.

    Raises:
        SystemExit: On a silent clip fallback, a leaked swap, or a non-finite window.
    """
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.traj_ref.path = args_cli.clip
    env_cfg.commands.traj_ref.clip_device = args_cli.clip_device
    env_cfg.scene.terrain.clip_library_path = args_cli.clip
    env_cfg.commands.traj_ref.fm_checkpoint = args_cli.fm_checkpoint

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    cmd = unwrapped.command_manager.get_term("traj_ref")
    print(f"[verify] reference_source={cmd.cfg.reference_source!r}  "
          f"history_from={cmd.cfg.history_from!r}  fm_enabled={cmd.fm_enabled}")

    # The width the critic-padding step needs. Printed from the live manager rather than
    # computed by hand, because getting it wrong pads to the wrong size and the mismatch only
    # shows up as a load error much later.
    for group, dim in unwrapped.observation_manager.group_obs_dim.items():
        width = int(dim[0]) if not isinstance(dim, int) else int(dim)
        print(f"[verify] obs group {group:12s} width {width}")
    print("[verify] pass the 'critic' width to pad_critic_for_finetune.py --new_width")

    env.reset()
    zero_action = torch.zeros(
        unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device
    )
    window_kwargs = dict(offsets=REFERENCE_OFFSETS, blocks=REFERENCE_WINDOW_BLOCKS)

    worst = 0.0
    for step in range(args_cli.steps):
        env.step(zero_action)
        served = mdp.fm_reference_window(unwrapped, "traj_ref", **window_kwargs)
        clip = mdp.reference_window(unwrapped, "traj_ref", **window_kwargs)
        if not bool(torch.isfinite(served).all()):
            raise SystemExit(
                f"[verify] step {step}: the served window is not finite. Under "
                "strict_fm_only every channel the generator does not write is NaN, so this "
                "means the window selection asks for a block the generator never fills."
            )
        worst = max(worst, float((served - clip).abs().max()))

    # The swap must not outlive the observation: re-reading the clip-served term has to give
    # the clip. If this fails the CLF is grading the generator and the reward has moved.
    restored = torch.equal(
        mdp.reference_window(unwrapped, "traj_ref", **window_kwargs), clip
    )
    env.close()

    print(f"[verify] worst |served - clip| over {args_cli.steps} steps : {worst:.5f}")
    print(f"[verify] clip serve restored after the swap              : {restored}")
    if worst <= args_cli.atol:
        raise SystemExit(
            f"[verify] FAIL: the served window matches the clip to {args_cli.atol}. The "
            "generator is not reaching the observation, and this task would train the "
            "teacher again with no error to show for it."
        )
    if not restored:
        raise SystemExit(
            "[verify] FAIL: the clip serve was not restored, so the CLF is grading the "
            "generator. The whole point of reference_source='clip' is that it does not."
        )
    print("[verify] PASS: the generator is served, and only to the observation.")


if __name__ == "__main__":
    main()
    simulation_app.close()
