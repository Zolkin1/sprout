# Copyright 2026 Zachary Olkin. All rights reserved.

"""Why a checkpoint looks better in ``play.py`` than its training curves say.

``play.py`` does not evaluate the policy under the conditions it was trained in.  The
``-play`` env cfg turns OFF observation noise (policy AND reference), pushes and friction
randomization, and rebuilds the terrain with ``num_rows = 1``; on top of that ``play.py``
runs the DETERMINISTIC policy while training rollouts SAMPLE from the action distribution.
Any one of those can turn a 70%-failure training statistic into a clean-looking rollout, so
"it looks fine in play" is not evidence the training metric is wrong.

This runs the same policy headless over many envs and reports the termination distribution,
with each of those factors individually switchable.  Defaults reproduce TRAINING (noise on,
events on, stochastic actions); ``--preset play`` reproduces what ``play.py`` shows.  Flip
one factor at a time to find which one accounts for the gap.

Shares are read from ``extras["log"]``, the very entries rsl_rl turns into
``Episode_Termination/<term>`` on the dashboard, so they are directly comparable to wandb.
They are per-term means over episodes and can sum to more than 1.0 when several terms fire
on the same step.

Usage (from the repo root, with the venv active)::

    python scripts/rsl_rl/termination_stats.py --task G1-teacher-clf-terrain-body \\
        --clip clips/g1_mb_v5_body.h5 --checkpoint logs/.../model_7400.pt --steps 3000
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-teacher-clf-terrain-body",
                    help="TRAINING task id (not the -play variant -- the point is to "
                         "reproduce training conditions and switch them off one by one).")
parser.add_argument("--clip", required=True, help="Clip library .h5 the run trained on.")
parser.add_argument("--checkpoint", required=True, help="Path to a model_*.pt.")
parser.add_argument("--num_envs", type=int, default=512, help="Envs to average over.")
parser.add_argument("--steps", type=int, default=3000, help="Rollout steps.")
parser.add_argument("--clip_device", default="cpu", help="Device to hold the clip library on.")
parser.add_argument("--preset", choices=["train", "play"], default="train",
                    help="'train' (default) = noise + events + stochastic actions. "
                         "'play' = what play.py shows: all noise/events off, deterministic.")
parser.add_argument("--deterministic", action="store_true", default=None,
                    help="Use the mean action instead of sampling. Overrides --preset.")
parser.add_argument("--no_obs_noise", action="store_true", default=None,
                    help="Disable policy AND reference observation corruption.")
parser.add_argument("--no_events", action="store_true", default=None,
                    help="Disable push_robot and ground-friction randomization.")
parser.add_argument("--num_rows", type=int, default=None,
                    help="Terrain replica rows. The -play cfg uses 1, training uses the "
                         "task default -- this changes the TERRAIN MIX, not just the count.")
parser.add_argument("--out", default="termination_stats.txt",
                    help="Report file. IsaacSim's startup often swallows stdout.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

# Resolve the preset first so explicit flags can override it.
_play = args.preset == "play"
DETERMINISTIC = _play if args.deterministic is None else args.deterministic
NO_OBS_NOISE = _play if args.no_obs_noise is None else args.no_obs_noise
NO_EVENTS = _play if args.no_events is None else args.no_events

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import robot_rl  # noqa: F401,E402  (registers the gym envs)
# The recursive activate_contact_sensors patch lives in the PRIVATE impl module and is
# applied at import time; the public shim does NOT apply it. train.py does the same.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402


def build_env():
    """Construct the env with the requested training/play factors switched on or off."""
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.commands.traj_ref.path = args.clip
    cfg.commands.traj_ref.clip_device = args.clip_device
    if hasattr(cfg.scene, "terrain") and hasattr(cfg.scene.terrain, "clip_library_path"):
        cfg.scene.terrain.clip_library_path = args.clip
    if NO_OBS_NOISE:
        # Both actor-side groups: the reference window is corrupted during training too.
        cfg.observations.policy.enable_corruption = False
        cfg.observations.reference.enable_corruption = False
    if NO_EVENTS:
        cfg.events.push_robot = None
        cfg.events.randomize_ground_contact_friction = None
    if args.num_rows is not None:
        cfg.scene.terrain.num_rows = args.num_rows
    return gym.make(args.task, cfg=cfg).unwrapped, cfg


def _strip_unaccepted(cfg: dict) -> dict:
    """Drop model-cfg keys the resolved model class does not accept.

    Building the runner straight from the registry skips the ``@hydra_task_config``
    resolution that ``train.py`` / ``play.py`` get, so ``RslRlMLPModelCfg``'s deprecated
    legacy entries survive ``to_dict()`` and reach the constructor as unexpected kwargs.
    Filtering against the signature keeps this robust as those fields come and go.
    """
    import inspect

    from rsl_rl.utils import resolve_callable

    for key in ("actor", "critic"):
        section = cfg.get(key)
        if not isinstance(section, dict) or "class_name" not in section:
            continue
        cls = resolve_callable(section["class_name"])
        accepted = set(inspect.signature(cls.__init__).parameters)
        for k in [k for k in section if k != "class_name" and k not in accepted]:
            section.pop(k)
    return cfg


def load_policy(env):
    """Load the checkpoint and return an action callable honouring ``DETERMINISTIC``."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, _strip_unaccepted(agent_cfg.to_dict()), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    def act(obs):
        # stochastic_output=True samples from the action distribution, which is what the
        # TRAINING rollouts do. play.py takes the mean, and the mean tracks much better.
        return policy(obs, stochastic_output=not DETERMINISTIC)

    return act


def main() -> None:
    """Roll out and report the per-termination episode shares."""
    env, _ = build_env()
    policy = load_policy(env)
    obs, _ = env.reset()

    # Weighted accumulation of extras["log"]: each entry is a mean over the envs that reset
    # on that step, so it has to be weighted by how many did.
    totals: dict[str, float] = {}
    episodes = 0

    # Per-frame attribution for ee_pose_error: knowing it fires is not actionable, knowing
    # it is one wrist on a long kinematic chain (rather than a foot) is.  Read the same
    # quantities the termination does, on the step it fires.
    cmd = env.command_manager.get_term("traj_ref")
    ee_cfg = env.termination_manager.get_term_cfg("ee_pose_error").params
    frames = list(ee_cfg["frame_names"])
    pos_lim, ori_lim = ee_cfg["max_pos_dist"], ee_cfg["max_ori_rad"]
    from isaaclab.utils.math import quat_box_minus
    trips = {f: [0, 0] for f in frames}          # frame -> [position trips, orientation trips]
    err_sum = {f: 0.0 for f in frames}
    err_n = 0
    # State AT the firing instant. All four frames are lifted through the anchored base, so
    # a base-level event moves them together -- these separate "the limbs drifted" from
    # "the whole robot left" and from "the reference jumped".
    import warp as wp
    fire = {"n": 0, "base_glob": 0.0, "base_anch": 0.0, "root_z": 0.0, "maxfrm": 0.0,
            "nframes": 0.0}
    # Frame error split by whether the env was RESET on the previous step.  Everything here
    # is read at the top of the loop, so it is one consistent instant, and "just reset"
    # means the robot has been teleported onto its new clip start -- where the tracking
    # error must be ~0.  If it is not, the reference is stale on the reset step.
    post = {"n": 0, "err": 0.0, "over": 0}
    stdy = {"n": 0, "err": 0.0, "over": 0}
    just_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for _ in range(args.steps):
        with torch.inference_mode():
            act_p, act_q, des_p, des_q = cmd.tracked_frame_pose(frames)
            n_e, n_f = act_p.shape[0], act_p.shape[1]
            pe = torch.norm(act_p - des_p, dim=-1)                                   # [N, F]
            oe = torch.norm(quat_box_minus(act_q.reshape(n_e * n_f, 4),
                                           des_q.reshape(n_e * n_f, 4)), dim=-1).view(n_e, n_f)
            worst = pe.max(dim=-1).values
            for tgt, m in ((post, just_reset), (stdy, ~just_reset)):
                if m.any():
                    tgt["n"] += int(m.sum())
                    tgt["err"] += float(worst[m].sum())
                    tgt["over"] += int((worst[m] > pos_lim).sum())
            obs, _, terminated, truncated, extras = env.step(policy(obs))
            just_reset = terminated | truncated
        for j, f in enumerate(frames):
            err_sum[f] += float(pe[:, j].mean())
        err_n += 1
        n = int((terminated | truncated).sum())
        if not n:
            continue
        fired = env.termination_manager.get_term("ee_pose_error")
        if fired.any():
            for j, f in enumerate(frames):
                trips[f][0] += int(((pe[:, j] > pos_lim) & fired).sum())
                trips[f][1] += int(((oe[:, j] > ori_lim) & fired).sum())
            k = int(fired.sum()); fire["n"] += k
            root = wp.to_torch(env.scene["robot"].data.root_pos_w)[fired]
            fire["base_glob"] += float(torch.norm(
                cmd.termination_reference_world_state()[fired][:, 0:3] - root, dim=-1).sum())
            fire["base_anch"] += float(torch.norm(
                cmd.reference_world_state()[fired][:, 0:3] - root, dim=-1).sum())
            fire["root_z"] += float(root[:, 2].sum())
            fire["maxfrm"] += float(pe[fired].max(dim=-1).values.sum())
            fire["nframes"] += float((pe[fired] > pos_lim).sum(dim=-1).float().sum())
        episodes += n
        log = extras.get("log") or extras.get("episode") or {}
        for key, value in log.items():
            if key.startswith("Episode_Termination/") or key == "Episode_Length":
                totals[key] = totals.get(key, 0.0) + float(value) * n

    rows = sorted(((k.split("/")[-1], v / max(episodes, 1))
                   for k, v in totals.items() if k.startswith("Episode_Termination/")),
                  key=lambda kv: -kv[1])
    factors = (f"obs_noise={'OFF' if NO_OBS_NOISE else 'ON'}  "
               f"events={'OFF' if NO_EVENTS else 'ON'}  "
               f"actions={'deterministic' if DETERMINISTIC else 'STOCHASTIC (as trained)'}")
    lines = [
        "=" * 78,
        f"TERMINATION STATS  {os.path.basename(args.checkpoint)}  envs={env.num_envs} "
        f"steps={args.steps}",
        f"  preset={args.preset}  {factors}",
        f"  terrain num_rows={args.num_rows if args.num_rows is not None else 'task default'}",
        "=" * 78,
        f"episodes completed: {episodes}",
        "",
        f"  {'termination':28} {'share of episodes':>18}",
    ]
    lines += [f"  {name:28} {share:17.3f}" for name, share in rows]
    if "Episode_Length" in totals:
        lines += ["", f"mean episode length: {totals['Episode_Length'] / max(episodes,1):.1f}"]
    lines += [
        "",
        f"ee_pose_error attribution  (limits: {pos_lim} m / {ori_lim} rad)",
        f"  {'frame':28} {'pos trips':>10} {'ori trips':>10} {'mean err':>10}",
    ]
    lines += [
        f"  {f:28} {trips[f][0]:10d} {trips[f][1]:10d} {err_sum[f] / max(err_n,1):9.3f} m"
        for f in sorted(frames, key=lambda f: -trips[f][0])
    ]
    if fire["n"]:
        k = fire["n"]
        lines += [
            "", f"AT the ee_pose_error firing instant (n={k}):",
            f"  base pos err vs GLOBAL clip   {fire['base_glob']/k:8.3f} m   (its own limit: "
            f"{env.termination_manager.get_term_cfg('base_pose_error').params['max_pos_dist']} m)",
            f"  base pos err vs ANCHORED ref  {fire['base_anch']/k:8.3f} m",
            f"  root height                   {fire['root_z']/k:8.3f} m",
            f"  worst frame error             {fire['maxfrm']/k:8.3f} m",
            f"  frames over limit             {fire['nframes']/k:8.2f} of {len(frames)}",
        ]
    lines += ["", "worst-frame position error, split by whether the env JUST RESET:"]
    for nm, d in (("first step after reset", post), ("steady (mid-episode)", stdy)):
        if d["n"]:
            lines += [f"  {nm:24} mean {d['err']/d['n']:7.3f} m   over {pos_lim} m: "
                      f"{100.0*d['over']/d['n']:5.1f}%   [n={d['n']}]"]
    lines += [
        "",
        "Shares are per-term means over episodes and may sum to >1 (several terms can fire",
        "on one step). Same quantity rsl_rl logs as Episode_Termination/<term>.",
        "=" * 78,
    ]
    report = "\n".join(lines)
    print("\n" + report + "\n")
    with open(args.out, "w") as fh:
        fh.write(report + "\n")
    print(f"[termination_stats] report written to {args.out}")
    simulation_app.close()


if __name__ == "__main__":
    main()
