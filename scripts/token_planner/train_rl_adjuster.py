# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator PPO for the token ADJUSTER -- stage D.

The generator plans and the agent corrects it: the action is ``N`` residual token ids,
decoded through the frozen residual codebook, scaled by ``alpha`` and composed onto the plan
the flow-matching generator just drew. The frozen tracker then executes the corrected window
for one re-plan interval, so the reward is measured on motion the robot actually produced.

Why this stage exists at all is a measurement, not a hope. On the same held-out windows, base
position in metres: the generator alone is 0.0431; the residual codebook can REPRESENT the
correction at 0.0187; but the supervised adjuster only predicts its way to 0.0427, because the
conditional over residual tokens is broad and its mode is "no correction". Behaviour cloning
converges on doing nothing safely, and only an outcome-driven objective can leave that.

## Two deliberate departures from the RL spec, both argued from the above

**The advantage is GROUP-RELATIVE, not a critic's.** The learning problem here is literally
"which of several corrections to THIS plan works best", and that comparison can be made
directly: every env in a group is pinned to the same clip and the same start step, so its
group-mates run identical geometry from an identical spawn and differ only in the correction
they sampled. A baseline formed from the group's own returns has no approximation error and
no value-function bias, where a critic on robot state has both. ``--critic`` is kept as the
ablation rather than the default.

**The regularizer is NOT a KL toward the supervised policy.** In LLM RL the reference is
peaked, so a KL to it is a genuine trust region against reward hacking. This reference is not:
its predictive entropy is ~4.3 nats/token, because its conditional is broad. A KL toward a
broad reference penalizes SHARPENING -- which is precisely what RL has to do here, commit to
the correction that works. So the supervised model is the initialisation and the sampling
distribution and nothing more, and what gets bounded instead is the correction's MAGNITUDE:
``--beta_zero`` anchors probability mass on the codebook's zero code, and the env's
``residual_l2`` reward term prices the decoded correction directly. ``--beta_ref`` still
exists, defaults to 0, and should stay there unless something specific argues otherwise.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/train_rl_adjuster.py \\
        --task G1-token-adjuster-rl --clip clips/<lib>.h5 \\
        --adjuster logs/token_planner/<run>/adjuster.pt \\
        --residual_tokenizer logs/token_planner/<run>/tokenizer.pt \\
        --fm_checkpoint logs/flow_matching/<run>/best.pt \\
        --tracker_run logs/rsl_rl/g1/<tracker> --tracking_spec logs/tracking/<spec>.json \\
        --num_envs 512 --group_size 8 --run_name rladj_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

from isaaclab.app import AppLauncher

# --- CLI, before AppLauncher: hydra composes the env cfg BEFORE Kit boots ----------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-token-adjuster-rl")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--group_size", type=int, default=8,
                    help="G envs per identical-reset group. The group's own returns ARE the "
                         "baseline, so this is the number of corrections compared against "
                         "each other on one plan. --num_envs must be a multiple of it.")
parser.add_argument("--clip", required=True, help="Clip library (terrain + reference motion).")
parser.add_argument("--clip_device", default="cpu")
parser.add_argument("--adjuster", required=True,
                    help="The supervised adjuster: the INITIALISATION and the sampling "
                         "distribution. Not a KL reference -- see the module docstring.")
parser.add_argument("--residual_tokenizer", required=True,
                    help="The residual codebook the action is decoded through. Must be the "
                         "one the adjuster was fitted against; both are checked.")
parser.add_argument("--fm_checkpoint", required=True, help="The generator being corrected.")
parser.add_argument("--reference", default="",
                    help="Frozen policy for --beta_ref. Defaults to --adjuster, and is inert "
                         "while beta_ref is 0.")
parser.add_argument("--tracker_run", required=True)
parser.add_argument("--tracker_checkpoint", default="model_19999.pt")
parser.add_argument("--tracking_spec", required=True)
parser.add_argument("--goal_regions", default="")
parser.add_argument("--run_name", default="rladj_v1")
parser.add_argument("--logdir", default="logs/token_planner_rl")
parser.add_argument("--render_decor", action=argparse.BooleanOptionalAction, default=None)

parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--rollout", type=int, default=16, help="Macro-steps (re-plans) per iter.")
parser.add_argument("--ppo_epochs", type=int, default=2)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--eps", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--t_roll", type=float, default=1.0)
parser.add_argument("--beta_zero", type=float, default=0.01,
                    help="Weight on the ZERO-CODE anchor: -log pi(zero_code). Bounds how far "
                         "the correction moves the plan WITHOUT constraining how peaked the "
                         "rest of the distribution is, which is the whole reason it replaces "
                         "a KL to the supervised policy.")
parser.add_argument("--beta_ref", type=float, default=0.0,
                    help="Weight on an exact per-token KL to --reference. DEFAULT 0 and it "
                         "should stay there: the reference is broad (~4.3 nats/token), so "
                         "this term mostly penalizes the sharpening RL needs to do.")
parser.add_argument("--lambda_ent", type=float, default=0.002)
parser.add_argument("--alpha_start", type=float, default=0.25,
                    help="Residual gain at iteration 0. NOT 0: at alpha=0 the action provably "
                         "cannot affect the environment, so the policy gradient is exactly "
                         "zero and a group-relative run has no signal at all -- every group "
                         "member returns the same thing and every advantage is masked. 0.25 "
                         "is small enough to be safe and large enough to be measurable.")
parser.add_argument("--alpha_end", type=float, default=1.0)
parser.add_argument("--alpha_ramp_iters", type=int, default=100)
parser.add_argument("--critic", action="store_true",
                    help="ABLATION: replace the group baseline with a learned value function "
                         "and GAE. The group baseline is the default because the comparison "
                         "it needs is exactly the one this env can stage.")
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_every", type=int, default=25)
parser.add_argument("--wandb", action="store_true", default=True)
parser.add_argument("--no_wandb", dest="wandb", action="store_false")
parser.add_argument("--wandb_project", default="robot_rl_token_planner")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

# MUST come after SimulationApp: it patches `activate_contact_sensors` for RECURSIVE
# traversal, without which every contact sensor on a nested link path fails to find a body
# with the reporter API. `train.py` imports it at the same point for the same reason.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402  (registers the tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig  # noqa: E402
from robot_rl.token_planner.metrics import channel_group_labels  # noqa: E402
from robot_rl.token_planner.planner import TokenPlanner  # noqa: E402
from robot_rl.token_planner.residual import find_zero_code  # noqa: E402
from robot_rl.token_planner.rl_critic import Critic  # noqa: E402
from robot_rl.token_planner.rl_gae import compute_gae  # noqa: E402
from robot_rl.token_planner.rl_outcomes import EndingCounts  # noqa: E402
from robot_rl.token_planner.rl_reward import group_advantages  # noqa: E402
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
from provenance import git_snapshot  # noqa: E402


def load_adjuster(path: str, device) -> tuple[TokenPlanner, dict]:
    """Load a supervised adjuster, preferring EMA weights where it has them.

    Args:
        path: A ``train_adjuster.py`` checkpoint.
        device: Torch device.

    Returns:
        ``(model, checkpoint)``.

    Raises:
        SystemExit: If the checkpoint was not built for plan conditioning -- an absolute
            planner's codes are a trajectory, so composing them onto a plan would add two
            trajectories at exactly the right shape.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PlannerConfig.from_dict(ck["config"])
    if not cfg.plan_conditioning:
        raise SystemExit(
            f"{path} was trained WITHOUT plan_conditioning, so its tokens decode to a "
            "trajectory rather than to a correction. That is the absolute planner; train it "
            "with train_rl_sim.py, not here.")
    m = TokenPlanner(cfg, TrajectoryTokenizer(TokenizerConfig.from_dict(cfg.tokenizer)))
    m.load_state_dict(ck["model"])
    return m.to(device), ck


def build_groups(env, group_size: int):
    """Partition envs into groups that share a TERRAIN TILE.

    A group is only a valid baseline if its members face the same problem, and on this stack
    "the same problem" is bounded by geometry the env cannot change: each env is bound to a
    tile at startup, and the reset REFUSES a clip whose ``terrain_id`` is not that env's --
    the clip is placed on the tile's own corner, so a mismatched one would ride treads that
    are not there. So the grouping is a property of the terrain layout, computed once, and
    the clip is drawn per group from that tile's own clips.

    Envs left over on a tile with fewer than ``group_size`` of them are EXCLUDED rather than
    padded into a short group: a baseline over two members is mostly noise, and silently
    letting group sizes vary would make the advantage scale vary with the tile.

    Args:
        env: The unwrapped env.
        group_size: ``G``.

    Returns:
        ``(members, tiles)`` -- ``[n_groups, G]`` env indices and ``[n_groups]`` tile ids.

    Raises:
        SystemExit: If no full group can be formed. That is a statement about --num_envs
            against the tile count, and running with zero groups would train on nothing while
            looking healthy.
    """
    importer = env.scene.terrain
    tid = importer.env_terrain_id.to(torch.long)
    blocks = []
    for t in torch.unique(tid):
        members = torch.nonzero(tid == t, as_tuple=False).flatten()
        n = (members.numel() // group_size) * group_size
        if n:
            blocks.append(members[:n].view(-1, group_size))
    if not blocks:
        per_tile = int(torch.bincount(tid).max())
        raise SystemExit(
            f"no tile holds {group_size} envs (the fullest holds {per_tile}), so not one "
            f"group can be formed. Raise --num_envs, lower --group_size, or use a library "
            f"with fewer tiles: envs are spread over {int(torch.unique(tid).numel())} of them.")
    members = torch.cat(blocks)
    return members, tid[members[:, 0]]


def draw_group_clips(env, members: torch.Tensor, tiles: torch.Tensor,
                     rng: np.random.Generator) -> None:
    """Pin one clip per group, drawn from that group's own tile.

    Writes the reset event's ``pin_clip_ids``, the same hook the headroom harness uses, rather
    than reaching into the command: the clip is CHOSEN at reset and the event is where that
    happens. Also pins the start step, because two envs on the same clip at different phases
    are different problems and the group exists to remove exactly that difference.

    Envs outside every group still need a legal clip -- they reset like any other -- so they
    are given a random one from their own tile and simply never enter an advantage.

    Args:
        env: The unwrapped env.
        members: ``[n_groups, G]`` env indices.
        tiles: ``[n_groups]`` tile per group.
        rng: This iteration's draw.
    """
    from motion_gen.clips.clip_io import group_clips_by_terrain  # noqa: PLC0415

    cmd = env.command_manager.get_term(env.cfg.command_name)
    by_tile = group_clips_by_terrain(cmd.manager.lib)
    tid = env.scene.terrain.env_terrain_id.to(torch.long)
    pin = torch.zeros(env.num_envs, dtype=torch.long)
    # Everyone gets a legal clip first, then the grouped envs are overwritten with their
    # group's shared one -- so no env is ever left holding a clip from another tile.
    for t in torch.unique(tid):
        pool = by_tile[int(t)]
        who = torch.nonzero(tid == t, as_tuple=False).flatten()
        pin[who.cpu()] = pool[torch.as_tensor(
            rng.integers(0, len(pool), size=who.numel()))].cpu().to(torch.long)
    for g in range(members.shape[0]):
        pool = by_tile[int(tiles[g])]
        pin[members[g].cpu()] = int(pool[int(rng.integers(0, len(pool)))])
    term = env.event_manager.get_term_cfg("reset_on_clip")
    term.params["pin_clip_ids"] = [int(c) for c in pin]
    term.params["pin_start_step"] = 0


def alpha_at(it: int) -> float:
    """The residual gain at iteration ``it``, ramped linearly then held.

    Args:
        it: Iteration index.

    Returns:
        The gain.
    """
    if args.alpha_ramp_iters <= 0:
        return float(args.alpha_end)
    t = min(1.0, it / float(args.alpha_ramp_iters))
    return float(args.alpha_start + t * (args.alpha_end - args.alpha_start))


def main() -> None:
    """Collect groups, score them against each other, update."""
    # No --num_envs % G check: groups are formed WITHIN a tile, not from contiguous env
    # blocks, so what matters is envs-per-tile and `build_groups` reports that directly. The
    # library's tile count is the real constraint -- 75 tiles at G=8 wants 600 envs.
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.tracker_run_dir = args.tracker_run
    env_cfg.tracker_checkpoint = args.tracker_checkpoint
    env_cfg.tracking_spec_path = args.tracking_spec
    env_cfg.residual_tokenizer_path = args.residual_tokenizer
    env_cfg.residual_alpha = alpha_at(0)
    env_cfg.roll_temperature = args.t_roll
    if args.goal_regions:
        env_cfg.goal_regions_path = args.goal_regions
    terrain = getattr(env_cfg.scene, "terrain", None)
    if args.render_decor is not None and terrain is not None \
            and hasattr(terrain, "render_decor"):
        terrain.render_decor = bool(args.render_decor)
    # The importer carries its OWN copy of the library path and it must stay identical to the
    # command's: one file defines both what the robot tracks and what it walks on.
    if terrain is not None and hasattr(terrain, "clip_library_path"):
        terrain.clip_library_path = args.clip
    for _nm, _term in vars(env_cfg.commands).items():
        for _f, _v in (("path", args.clip), ("clip_device", args.clip_device),
                       ("fm_checkpoint", args.fm_checkpoint)):
            if hasattr(_term, _f):
                setattr(_term, _f, _v)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    dev = torch.device(env.device)

    policy, parent = load_adjuster(args.adjuster, dev)
    policy.eval()                    # dropout OFF: the sampled and re-scored passes must be
    policy.requires_grad_(True)      # one network, or the ratio is between two models
    policy.tokenizer.eval().requires_grad_(False)
    ref = None
    if args.beta_ref > 0.0:
        ref, _ = load_adjuster(args.reference or args.adjuster, dev)
        ref = ref.eval().requires_grad_(False)

    # The env decodes actions with ITS copy of the codebook and the policy scores them with
    # its own. A drift between the two is a silent change of what every id means, so they are
    # compared rather than assumed equal.
    res_norm = Normalizer.from_state_dict(parent["residual_normalizer"])
    for field in ("mean", "std"):
        a = getattr(res_norm, field)
        b = getattr(env.residual_normalizer, field).cpu()
        if a.shape != b.shape or not torch.equal(a, b):
            raise SystemExit(
                f"the adjuster's residual normalizer and the env's disagree on '{field}'. "
                "Both decode the same ids; disagreeing on the space they name makes the "
                "correction the policy scores a different one from the correction served.")

    names, labels = channel_group_labels(
        int(policy.cfg.num_joints_for_layout), int(policy.cfg.num_ee_frames_for_layout),
        list(policy.cfg.tokenizer_cfg.reference_blocks))
    zero_code, zero_decode, zero_rms = find_zero_code(
        policy.tokenizer.to(dev), res_norm.to(dev), labels, names)
    print(f"[rladj] ZERO_CODE {zero_code} (decode rms {zero_rms:.5f} m); the magnitude anchor "
          f"is -log pi(that) at beta_zero={args.beta_zero}, NOT a KL to the supervised "
          "policy")
    print("[rladj]   per group: "
          + "  ".join(f"{g} {v:.4f}" for g, v in zero_decode.items()))

    N, G = env.num_tokens, args.group_size
    members, tiles = build_groups(env, G)
    n_groups = int(members.shape[0])
    used = int(members.numel())
    print(f"[rladj] groups: {n_groups} x G={G} = {used} of {env.num_envs} envs on "
          f"{int(torch.unique(tiles).numel())} tile(s); "
          f"{env.num_envs - used} env(s) are ungrouped and contribute no advantage")
    critic = None
    if args.critic:
        critic = Critic(int(env.observation_manager.compute_group("critic").shape[1])).to(dev)
    params = [{"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.lr}]
    if critic is not None:
        params.append({"params": critic.parameters(), "lr": args.critic_lr})
    opt = torch.optim.AdamW(params, weight_decay=0.0)

    out_dir = os.path.join(args.logdir, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}")
    os.makedirs(out_dir, exist_ok=True)
    git_snapshot(out_dir)
    writer = SummaryWriter(log_dir=out_dir)
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(project=args.wandb_project,
                                   name=os.path.basename(out_dir), config=vars(args))
        except Exception as exc:                                       # noqa: BLE001
            print(f"[WARN] wandb off: {type(exc).__name__}: {exc}")
    print(f"[rladj] {n_groups} groups x G={G} = {env.num_envs} envs; {args.rollout} macro-steps"
          f"/iter; advantage = {'CRITIC + GAE' if critic is not None else 'GROUP-RELATIVE'}")
    print(f"[rladj] out: {out_dir}")
    _run(env, policy, ref, critic, opt, zero_code, rng, members, tiles, G, N, dev,
         writer, wandb_run, out_dir, parent)


@torch.no_grad()
def _collect(env, policy, rng, members, tiles, G, N, dev):
    """One iteration's rollout: identical-reset groups, rolled for ``--rollout`` macro-steps.

    Returns are accumulated to each env's FIRST ending and masked after it. That is what makes
    the group comparison exact: every member is scored on one episode from one shared spawn,
    so the spread within a group is the correction and nothing else. Continuing past a reset
    would mix in whatever clip the allocator handed out next, which is a different problem and
    would silently widen the baseline.

    Args:
        env: The unwrapped env.
        policy: The adjuster.
        rng: This iteration's clip draw.
        n_groups: Group count.
        G: Envs per group.
        N: Tokens per plan.
        dev: Torch device.

    Returns:
        A dict of stacked rollout tensors plus the per-env return, live mask and outcomes.
    """
    e, t_max = env.num_envs, args.rollout
    draw_group_clips(env, members, tiles, rng)
    obs, _ = env.reset()

    cond0 = env.planner_conditioning()
    C = torch.zeros(t_max, e, cond0.shape[1], device=dev)
    PW = torch.zeros(t_max, e, *env.plan_window().shape[1:], device=dev)
    I = torch.zeros(t_max, e, N, dtype=torch.long, device=dev)
    LP = torch.zeros(t_max, e, N, device=dev)
    R = torch.zeros(t_max, e, device=dev)
    LIVE = torch.zeros(t_max, e, device=dev)

    alive = torch.ones(e, dtype=torch.bool, device=dev)
    ret = torch.zeros(e, device=dev)
    counts = EndingCounts()
    disc = 1.0
    for t in range(t_max):
        cond = env.planner_conditioning()
        plan_raw = env.plan_window()
        hist, ext, cmd, null = env.split_conditioning(cond)
        plan_tok = policy.plan_tokens_from_window(env.plan_normalizer.normalize(plan_raw))
        ids, _, _ = policy.sample(hist, ext, cmd, null, temperature=args.t_roll,
                                  greedy=False, plan=plan_tok)
        # Re-scored through the TEACHER-FORCED path under the same autocast the update uses,
        # so epoch 1's ratio is exactly 1 by construction rather than by luck. The sampler's
        # own log-probabilities come from the incremental KV path and differ by a few parts in
        # a hundred at p99 under bf16, which the ratio would read as a real policy change.
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(args.bf16 and dev.type == "cuda")):
            lp, _ = policy.logprobs(ids, hist, ext, cmd, null,
                                    temperature=args.t_roll, plan=plan_tok)
        C[t], PW[t], I[t], LP[t] = cond, plan_raw, ids, lp.float()
        LIVE[t] = alive.float()

        obs, rew, term, trunc, _ = env.step(ids)
        ret += disc * rew * alive
        disc *= args.gamma
        done = (term | trunc)
        counts.update(done & alive, trunc & alive, getattr(env, "ended_by", None))
        R[t] = rew * alive
        alive = alive & ~done
    return {"C": C, "PW": PW, "I": I, "LP": LP, "R": R, "LIVE": LIVE,
            "ret": ret, "counts": counts, "finished": (~alive).float()}


def _run(env, policy, ref, critic, opt, zero_code, rng, members, tiles, G, N, dev,
         writer, wandb_run, out_dir, parent) -> None:
    """The iteration loop: collect, form group advantages, update, log, gate.

    Args:
        env: The unwrapped env.
        policy: The adjuster being trained.
        ref: A frozen reference for ``--beta_ref``, or ``None``.
        critic: Unused; the group baseline is the only advantage path.
        opt: The optimizer.
        zero_code: The codebook's near-zero id, for the magnitude anchor.
        rng: Clip draw.
        n_groups / G / N: Group count, group size, tokens per plan.
        dev: Torch device.
        writer: TensorBoard writer.
        wandb_run: wandb run or ``None``.
        out_dir: Where checkpoints go.
        parent: The supervised checkpoint, for the contract graft.
    """
    use_bf16 = bool(args.bf16 and dev.type == "cuda")
    for it in range(args.iters):
        t0 = time.perf_counter()
        env.residual_alpha = alpha_at(it)
        roll = _collect(env, policy, rng, members, tiles, G, N, dev)

        # --- the group baseline ------------------------------------------------------
        # Members share a clip, a tile and a spawn, so the spread WITHIN a group is the
        # correction. `group_advantages` z-scores against the group's own mean and returns
        # `degenerate` where the spread is below eps -- a group whose members all did the
        # same thing carries no ranking information, and a near-zero-but-nonzero advantage
        # there is noise the policy would happily chase.
        ret = roll["ret"][members]                                   # [n_groups, G]
        adv_g, degenerate = group_advantages(ret)
        # Scattered back BY MEMBERSHIP, not by position: groups are tile-contiguous, not env-
        # contiguous, so a reshape would credit the wrong envs while every shape stayed right.
        adv = torch.zeros(env.num_envs, device=dev)
        keep = torch.zeros(env.num_envs, device=dev)
        adv[members.reshape(-1)] = adv_g.reshape(-1)
        keep[members.reshape(-1)] = (~degenerate).float().unsqueeze(-1).expand_as(
            adv_g).reshape(-1)
        # ...and only on steps the env was still live for. A masked step contributed nothing
        # to the return it is being credited with.
        w = roll["LIVE"] * keep.unsqueeze(0)

        flat = lambda x: x.reshape(-1, *x.shape[2:])                  # noqa: E731
        Cf, PWf, If, LPf = (flat(roll[k]) for k in ("C", "PW", "I", "LP"))
        Af = adv.unsqueeze(0).expand_as(roll["LIVE"]).reshape(-1)
        Wf = w.reshape(-1)
        n = Cf.shape[0]
        mb = max(n // args.minibatches, 1)

        stats = {k: [] for k in ("policy", "zero", "ref_kl", "entropy", "clipped",
                                 "approx_kl", "grad")}
        for _ep in range(args.ppo_epochs):
            for idx in torch.randperm(n, device=dev).split(mb):
                cb, pwb, ib, lpb, ab, wb = (Cf[idx], PWf[idx], If[idx], LPf[idx],
                                            Af[idx], Wf[idx])
                if float(wb.sum()) == 0.0:
                    continue
                hist, ext, cmd, null = env.split_conditioning(cb)
                plan_tok = policy.plan_tokens_from_window(
                    env.plan_normalizer.normalize(pwb))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    lp_new, ent, lg = policy.logprobs(
                        ib, hist, ext, cmd, null, temperature=args.t_roll,
                        return_logits=True, plan=plan_tok)
                lp32, lg32 = lp_new.float(), lg.float()
                # PER TOKEN, never summed: the macro-step advantage is broadcast to the N
                # decisions that earned it, and a summed log-probability would throw away
                # which decision an update belongs to.
                ratio = torch.exp(lp32 - lpb)
                a = ab.unsqueeze(-1)
                surr = torch.min(ratio * a, ratio.clamp(1 - args.eps, 1 + args.eps) * a)
                m = wb.unsqueeze(-1)
                denom = m.sum() * N
                logp_all = F.log_softmax(lg32, dim=-1)
                # THE MAGNITUDE ANCHOR. Mass on the zero code, not shape against a reference:
                # the supervised policy is broad (~4.3 nats/token), so a KL toward it would
                # penalize the sharpening this stage exists to do.
                zero_nll = -logp_all[..., int(zero_code)]
                loss = (-(surr * m).sum() / denom
                        + args.beta_zero * (zero_nll * m).sum() / denom
                        - args.lambda_ent * (ent.float() * m).sum() / denom)
                if ref is not None:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                         enabled=use_bf16):
                        _, _, lg_ref = ref.logprobs(ib, hist, ext, cmd, null,
                                                    temperature=args.t_roll,
                                                    return_logits=True, plan=plan_tok)
                    kl = (logp_all.exp()
                          * (logp_all - F.log_softmax(lg_ref.float(), -1))).sum(-1)
                    loss = loss + args.beta_ref * (kl * m).sum() / denom
                    stats["ref_kl"].append(float((kl * m).sum() / denom))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                opt.step()
                policy._last_prefix_cache = None
                with torch.no_grad():
                    bound = (ratio * a > ratio.clamp(1 - args.eps, 1 + args.eps) * a)
                    stats["policy"].append(float(-(surr * m).sum() / denom))
                    stats["zero"].append(float((zero_nll * m).sum() / denom))
                    stats["entropy"].append(float((ent.float() * m).sum() / denom))
                    stats["clipped"].append(float((bound.float() * m).sum() / denom))
                    stats["approx_kl"].append(float(((lpb - lp32) * m).sum() / denom))
                    stats["grad"].append(float(gn))

        _log(it, roll, adv, degenerate, stats, env, writer, wandb_run,
             time.perf_counter() - t0)
        if it % max(args.ckpt_every, 1) == 0 or it == args.iters - 1:
            _save(os.path.join(out_dir, f"rl_{it}.pt"), policy, opt, parent, it, zero_code)


def _log(it, roll, adv, degenerate, stats, env, writer, wandb_run, secs) -> None:
    """One iteration's line, and everything the spec asks be watched.

    ``degenerate_frac`` is the one worth watching first: a group whose members all returned
    the same thing carries no ranking information, so if most groups are degenerate the run
    is spending its budget on masked gradients -- which is what ``alpha`` being too small, or
    the correction not mattering, both look like.

    Args:
        it: Iteration.
        roll: The rollout dict.
        adv: ``[E]`` group advantages.
        degenerate: ``[n_groups]`` bool.
        stats: Per-minibatch scalars.
        env: The env, for the outcome tallies.
        writer: TensorBoard writer.
        wandb_run: wandb run or ``None``.
        secs: Iteration wall time.
    """
    fmean = lambda v: (statistics.fmean(v) if v else 0.0)             # noqa: E731
    counts = roll["counts"]
    rates = counts.rates()
    resid = env.extras.get("log", {})
    sc = {
        "train/return": float(roll["ret"].mean()),
        "train/reward": float(roll["R"].sum(0).mean()),
        "train/alpha": float(env.residual_alpha),
        "train/degenerate_frac": float(degenerate.float().mean()),
        "train/adv_abs": float(adv.abs().mean()),
        "train/live_frac": float(roll["LIVE"].mean()),
        "train/finished_frac": float(roll["finished"].mean()),
        "train/policy_loss": fmean(stats["policy"]),
        "train/zero_nll": fmean(stats["zero"]),
        "train/entropy": fmean(stats["entropy"]),
        "train/clipped_frac": fmean(stats["clipped"]),
        "train/approx_kl": fmean(stats["approx_kl"]),
        "train/grad_norm": fmean(stats["grad"]),
        "train/iter_s": secs,
        "outcome/episodes": float(counts.episodes),
    }
    if stats["ref_kl"]:
        sc["train/ref_kl"] = fmean(stats["ref_kl"])
    for k, v in rates.items():
        sc[f"outcome/{k}"] = float(v)
    for k, v in resid.items():
        if "residual" in k.lower() or "violation" in k.lower():
            sc[f"diag/{k}"] = float(v)
    for k, v in sc.items():
        writer.add_scalar(k, v, it)
    if wandb_run is not None:
        wandb_run.log(sc, step=it)
    print(f"[rladj] it {it:5d}  a {sc['train/alpha']:.2f}  ret {sc['train/return']:+.3f}  "
          f"deg {100 * sc['train/degenerate_frac']:.0f}%  "
          f"ent {sc['train/entropy']:.2f}  zero {sc['train/zero_nll']:.2f}  "
          f"clip {100 * sc['train/clipped_frac']:.0f}%  "
          f"goal {rates.get('goal_rate', 0.0):.3f}  fall {rates.get('fall_rate', 0.0):.3f}  "
          f"({secs:.0f}s)", flush=True)


def _save(path, policy, opt, parent, it, zero_code) -> None:
    """Write an RL checkpoint that ``load_planner_context`` can actually load.

    The graft list includes ``data``, which the token-planner RL saver omitted -- without it
    the fallback that re-reads the contract from the recording has nothing to read, and every
    checkpoint in that chain was a file no runtime could load.

    Args:
        path: Destination.
        policy: The adjuster.
        opt: The optimizer.
        parent: The supervised checkpoint whose contract must travel.
        it: Iteration.
        zero_code: The magnitude anchor's id, recorded so a later run cannot re-derive a
            different one against the same codebook.
    """
    ck = {"model": policy.state_dict(), "optimizer": opt.state_dict(),
          "config": policy.cfg.to_dict(), "epoch": it, "zero_code": int(zero_code),
          "rl": {k: getattr(args, k) for k in vars(args) if not k.startswith("_")}}
    for k in ("residual_normalizer", "state_normalizer", "fm_provenance", "data",
              "family_names", "reference_blocks", "num_joints", "num_ee_frames"):
        if k in parent:
            ck[k] = parent[k]
    torch.save(ck, path)


if __name__ == "__main__":
    if args.critic:
        raise SystemExit(
            "--critic is not implemented. The GROUP baseline is the default and the only "
            "path here, deliberately: the comparison a critic approximates -- which of "
            "several corrections to THIS plan works best -- is one this env can stage "
            "exactly, by pinning a group to one clip. The critic arm is worth building as an "
            "ablation, and it is not built yet; refusing beats silently running the group "
            "path under a flag that says otherwise.")
    main()
    simulation_app.close()
