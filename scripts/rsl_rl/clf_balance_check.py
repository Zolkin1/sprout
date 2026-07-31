# Copyright 2026 Zachary Olkin. All rights reserved.

"""What the CLF reward is ACTUALLY grading, measured against a trained checkpoint.

The two numbers the dashboard shows are both easy to misread:

* ``Metrics/traj_ref/v_<subgroup>`` is computed with an IDENTITY P, so it is raw squared
  error and does not reflect the Q/R weights at all -- a block can look enormous there and
  contribute little, or vice versa.
* ``Episode_Reward/clf_reward`` is an episode SUM x dt divided by ``max_episode_length_s``
  (``reward_manager.py:120,150``), so it tracks SURVIVAL as much as quality: short episodes
  read low even when the per-step reward is high.

This reports the real, P-weighted per-block share of V, plus the reward chain with every
factor spelled out -- the inputs needed to set the Q weights and ``max_eta_err``.

Run it against a CHECKPOINT (``--load_run``): a collapsing zero-action robot has wildly
inflated velocities, and tuning against those over-weights pose.

Usage (from the repo root, with the venv active)::

    python scripts/rsl_rl/clf_balance_check.py --task G1-teacher-clf-terrain-body \\
        --clip clips/g1_mb_v5_body.h5 --load_run 2026-07-29_19-01-07_body_v1
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="G1-teacher-clf-terrain-body", help="Gym task id.")
parser.add_argument("--clip", required=True, help="Clip library .h5 (widened, for body mode).")
parser.add_argument("--num_envs", type=int, default=64, help="Envs to average over.")
parser.add_argument("--steps", type=int, default=200, help="Rollout steps to accumulate.")
parser.add_argument("--load_run", default=None,
                    help="Run dir under logs/rsl_rl/<experiment>/ to load. Omit for an "
                         "UNTRAINED policy (zero-action-like; see the module docstring).")
parser.add_argument("--checkpoint", default="model_.*.pt",
                    help="Checkpoint regex within the run. NOTE the resolver sorts "
                         "ALPHABETICALLY, so model_800 beats model_1000.")
parser.add_argument("--clip_device", default="cpu", help="Device to hold the clip library on.")
parser.add_argument("--out", default="clf_balance.txt",
                    help="Report file. IsaacSim's startup often swallows stdout.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import robot_rl  # noqa: F401,E402  (registers the gym envs)
# The recursive activate_contact_sensors patch lives in the PRIVATE impl module and is
# applied at import time; the public shim does NOT apply it. train.py does the same.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402


def build_env():
    """Construct the task env pointed at the requested clip library."""
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.commands.traj_ref.path = args.clip
    cfg.commands.traj_ref.clip_device = args.clip_device
    if hasattr(cfg.scene, "terrain") and hasattr(cfg.scene.terrain, "clip_library_path"):
        cfg.scene.terrain.clip_library_path = args.clip
    return gym.make(args.task, cfg=cfg).unwrapped, cfg


def _strip_unaccepted(cfg: dict) -> dict:
    """Drop model-cfg keys the resolved model class does not accept.

    ``train.py`` / ``play.py`` reach the runner through ``@hydra_task_config``, which
    resolves the config and drops unset fields.  Building it straight from the registry
    skips that, so ``RslRlMLPModelCfg``'s deprecated legacy entries (``stochastic``, which
    renders as ``{}``, plus ``init_noise_std`` / ``noise_std_type`` /
    ``state_dependent_std``) survive ``to_dict()`` and get splatted at the model
    constructor as unexpected kwargs.

    Filtering against the constructor signature keeps this robust as those fields come and
    go, and it only affects this diagnostic -- training still goes through hydra.
    """
    import inspect
    from rsl_rl.utils import resolve_callable

    for key in ("actor", "critic"):
        section = cfg.get(key)
        if not isinstance(section, dict) or "class_name" not in section:
            continue
        cls = resolve_callable(section["class_name"])
        accepted = set(inspect.signature(cls.__init__).parameters)
        dropped = [k for k in section if k != "class_name" and k not in accepted]
        for k in dropped:
            section.pop(k)
        if dropped:
            print(f"[clf_balance_check] {key}: dropped non-constructor cfg keys {dropped}")
    return cfg


def load_policy(env):
    """Return an action callable: the trained policy, or zero actions if no run was given.

    Returns:
        ``(policy, label)`` where ``policy`` maps an observation TensorDict to actions.
    """
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    if args.load_run is None:
        return (lambda _obs: zero), "UNTRAINED (zero-ish actions -- see docstring)"

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_tasks.utils import load_cfg_from_registry

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    # ABSOLUTE: get_checkpoint_path does os.path.join(log_path, DirEntry), and a DirEntry
    # from a relative scandir carries a relative path, so a relative log_path silently
    # doubles into 'logs/rsl_rl/g1/logs/rsl_rl/g1/<run>'. play.py passes an absolute path.
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    path = get_checkpoint_path(log_root, args.load_run, args.checkpoint)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, _strip_unaccepted(agent_cfg.to_dict()), log_dir=None,
                            device=agent_cfg.device)
    runner.load(path)
    return runner.get_inference_policy(device=env.device), f"checkpoint {os.path.basename(path)}"


def block_shares(cmd):
    """Per-block share of V under the REAL P (block-diagonal, so the shares sum to V).

    Returns:
        ``(rows, v_total)`` where each row is ``(name, raw_identity_P, weighted, pct)``.
    """
    clf = cmd.clf
    subset = getattr(clf, "_subset", False)
    y_a = cmd.y_act[:, clf.pos_gather] if subset else cmd.y_act
    y_d = cmd.y_des[:, clf.pos_gather] if subset else cmd.y_des
    dy_a = cmd.dy_act[:, clf.vel_gather] if subset else cmd.dy_act
    dy_d = cmd.dy_des[:, clf.vel_gather] if subset else cmd.dy_des
    y_err = clf.compute_y_err(y_a, y_d)
    eta = torch.zeros(y_err.shape[0], clf.n_outputs, device=y_err.device)
    eta[:, 0::2] = y_err
    eta[:, 1::2] = dy_a - dy_d
    total = float(torch.einsum("bi,ij,bj->b", eta, clf.P, eta).mean())
    rows = []
    for name, idx in clf.subgroup_indices.items():
        if not idx.numel():
            continue
        sub = eta[:, idx]
        wgt = float((torch.matmul(sub, clf.P[idx][:, idx]) * sub).sum(dim=-1).mean())
        raw = float((sub * sub).sum(dim=-1).mean())
        rows.append((name, raw, wgt, 100.0 * wgt / max(total, 1e-9)))
    return rows, total


def main() -> None:
    """Roll out and report the weighting balance and the reward chain."""
    env, cfg = build_env()
    policy, label = load_policy(env)
    cmd = env.command_manager.get_term("traj_ref")
    obs, _ = env.reset()

    acc: dict[str, list[float]] = {}
    vs, live = [], 0
    for _ in range(args.steps):
        with torch.inference_mode():
            obs = env.step(policy(obs))[0]   # unwrapped env: (obs, rew, term, trunc, extras)
        rows, total = block_shares(cmd)
        vs.append(total)
        for name, raw, wgt, _ in rows:
            slot = acc.setdefault(name, [0.0, 0.0])
            slot[0] += raw
            slot[1] += wgt
        live += 1

    v_mean = sum(vs) / len(vs)
    tot_w = sum(v[1] for v in acc.values())
    lam = float(cmd.clf.lambda_max)
    mee = cfg.rewards.clf_reward.params["max_eta_err"]
    w = cfg.rewards.clf_reward.weight
    mc = lam * mee ** 2

    inst = w * torch.exp(torch.tensor(-v_mean / mc)).item()
    lines = [
        "=" * 78,
        f"CLF BALANCE  task={args.task}  policy={label}  steps={live}  envs={env.num_envs}",
        "=" * 78,
        f"eta dim={cmd.clf.n_outputs}  lambda_max={lam:.4f}  max_eta_err={mee}  max_clf={mc:.2f}",
        "",
        "per-block share of V   (raw = identity P, i.e. NOT the reward's view)",
        f"  {'block':24} {'raw':>12} {'weighted':>12} {'% of V':>8}",
    ]
    lines += [
        f"  {name:24} {raw / live:12.4f} {wgt / live:12.4f} {100.0 * wgt / tot_w:7.1f}%"
        for name, (raw, wgt) in sorted(acc.items(), key=lambda kv: -kv[1][1])
    ]
    lines += [
        "",
        f"mean V over the rollout = {v_mean:.2f}",
        f"  instantaneous clf_reward = {w} * exp(-V/max_clf) = {inst:.3f} / {w}",
        f"  half-reward sits at V    = max_clf * ln2 = {mc * 0.6931:.1f}",
        f"  to put half-reward at this V: max_eta_err = {(v_mean / (lam * 0.6931)) ** 0.5:.2f}",
        "=" * 78,
    ]
    report = "\n".join(lines)
    print("\n" + report + "\n")
    # Also to a file: IsaacSim's startup takes over stdout and routinely swallows this.
    with open(args.out, "w") as fh:
        fh.write(report + "\n")
    print(f"[clf_balance_check] report written to {args.out}")
    simulation_app.close()


if __name__ == "__main__":
    main()
