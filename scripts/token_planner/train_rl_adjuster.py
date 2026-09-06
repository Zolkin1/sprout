# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator PPO for the token ADJUSTER -- stage D.

The generator plans and the agent corrects it: the action is ``N`` residual token ids,
decoded through the frozen residual codebook, scaled by ``alpha`` and composed onto the plan
the flow-matching generator just drew. The frozen tracker then executes the corrected window
for one re-plan interval, so the reward is measured on motion the robot actually produced.

Why this stage exists is a measurement. On the same held-out windows, base position in
metres: the generator alone is 0.0431; the residual codebook can REPRESENT the correction at
0.0187; but the supervised adjuster only predicts its way to 0.0427, because the conditional
over residual tokens is broad and its mode is "no correction". Behaviour cloning converges on
doing nothing safely, and only an outcome-driven objective can leave that.

## This is ``train_rl_sim.py``'s loop, and deliberately so

Same critic, same GAE, same per-token clipped PPO, same arrival-wins precedence, same gates,
same ONE ``env.reset()`` before training with episodes auto-resetting inside ``step``. That
last one is not a stylistic match: a per-iteration ``env.reset()`` calls
``command_manager.compute`` at a ``common_step_counter`` the previous step already used, and
``_snapshot_and_infer`` de-duplicates on exactly that -- so the reset's history invalidation
and re-plan are silently skipped and the generator drives a teleported robot from the
previous episode's history. Measured: 2% of envs failing within 0.24 s on the first
iteration, 30-44% on every one after.

## One departure from the RL spec, argued from stage C

**The regularizer is NOT a KL toward the supervised policy.** In LLM RL the reference is
peaked, so a KL to it is a genuine trust region against reward hacking. This reference is
not: its predictive entropy is ~4.3 nats/token. A KL toward a broad reference penalizes
SHARPENING -- which is exactly what this stage has to do, commit to the correction that
works. So the supervised model is the initialisation and nothing more, and what is bounded
instead is the correction's MAGNITUDE: ``--beta_zero`` puts mass on the codebook's zero code
without constraining how peaked the rest of the distribution is, and the env's
``residual_l2`` reward terms price the decoded correction directly. ``--beta_ref`` still
exists and defaults to 0.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/train_rl_adjuster.py \\
        --task G1-token-adjuster-rl --clip clips/<lib>.h5 \\
        --adjuster logs/token_planner/<run>/adjuster.pt \\
        --residual_tokenizer logs/token_planner/<run>/tokenizer.pt \\
        --fm_checkpoint logs/flow_matching/<run>/best.pt \\
        --tracker_run logs/rsl_rl/g1/<tracker> --tracking_spec logs/tracking/<spec>.json \\
        --goal_regions logs/rollouts/<v>/goal_regions.json \\
        --num_envs 1024 --run_name rladj_v1
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
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--clip", required=True, help="Clip library (terrain + reference motion).")
parser.add_argument("--clip_device", default="cpu")
parser.add_argument("--adjuster", required=True,
                    help="The supervised adjuster: the INITIALISATION and the sampling "
                         "distribution. Not a KL reference -- see the module docstring.")
parser.add_argument("--residual_tokenizer", required=True,
                    help="The residual codebook the action decodes through. Must be the one "
                         "the adjuster was fitted against; the normalizers are compared.")
parser.add_argument("--fm_checkpoint", required=True, help="The generator being corrected.")
parser.add_argument("--reference", default="",
                    help="Frozen policy for --beta_ref. Defaults to --adjuster; inert at 0.")
parser.add_argument("--tracker_run", required=True)
parser.add_argument("--tracker_checkpoint", default="model_19999.pt")
parser.add_argument("--tracking_spec", required=True)
parser.add_argument("--goal_regions", default="")
parser.add_argument("--run_name", default="rladj_v1")
parser.add_argument("--logdir", default="logs/token_planner_rl")
parser.add_argument("--render_decor", action=argparse.BooleanOptionalAction, default=None)

parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--rollout", type=int, default=24, help="Macro-steps (re-plans) per iter.")
parser.add_argument("--warmup_iters", type=int, default=20,
                    help="Critic-only iterations. The value loss does not depend on the "
                         "policy's parameters, so the policy is frozen in effect and the "
                         "critic is fitted to the distribution it will actually score.")
parser.add_argument("--ppo_epochs", type=int, default=3)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--eps", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--gamma", type=float, default=0.985)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--target_kl", type=float, default=0.03)
parser.add_argument("--t_roll", type=float, default=1.0)
parser.add_argument("--beta_zero", type=float, default=0.01,
                    help="Weight on the ZERO-CODE anchor, -log pi(zero_code). Bounds how far "
                         "the correction moves the plan WITHOUT constraining how peaked the "
                         "rest of the distribution is, which is why it replaces a KL to the "
                         "supervised policy.")
parser.add_argument("--beta_ref", type=float, default=0.0,
                    help="Weight on an exact per-token KL to --reference. DEFAULT 0 and it "
                         "should stay there: the reference is broad (~4.3 nats/token), so "
                         "this term mostly penalizes the sharpening RL needs to do.")
parser.add_argument("--lambda_ent", type=float, default=0.002)
parser.add_argument("--alpha_start", type=float, default=0.25,
                    help="Residual gain at iteration 0. NOT 0: at alpha=0 the action provably "
                         "cannot affect the environment, so the policy gradient is exactly "
                         "zero and the iterations are dead time.")
parser.add_argument("--alpha_end", type=float, default=1.0)
parser.add_argument("--alpha_ramp_iters", type=int, default=100)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_every", type=int, default=25)
parser.add_argument("--validate_only", action="store_true",
                    help="Run the SUPERVISED policy with no training and tally the endings. "
                         "The gate to run before spending iterations: it says what the arm "
                         "being fine-tuned already does on this library.")
parser.add_argument("--validate_steps", type=int, default=200)
parser.add_argument("--wandb", action="store_true", default=True)
parser.add_argument("--no_wandb", dest="wandb", action="store_false")
parser.add_argument("--wandb_project", default="robot_rl_token_planner")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
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
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
from provenance import git_snapshot  # noqa: E402


def load_adjuster(path: str, device) -> tuple[TokenPlanner, dict]:
    """Load a supervised adjuster.

    Args:
        path: A ``train_adjuster.py`` checkpoint.
        device: Torch device.

    Returns:
        ``(model, checkpoint)``.

    Raises:
        SystemExit: If the checkpoint was not built for plan conditioning -- an absolute
            planner's codes decode to a trajectory, so composing them onto a plan would add
            two trajectories at exactly the right shape.
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
    """Collect, bootstrap, update."""
    torch.manual_seed(args.seed)
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
        print(f"[rladj] render_decor={terrain.render_decor} (visual only; physics unchanged)")
    # The importer carries its OWN copy of the library path and it must stay identical to the
    # command's: one file defines both what the robot tracks and what it walks on, and a
    # mismatch is a robot tracking one tile's motion on another's geometry.
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
    policy.eval()                    # dropout OFF: the sampled and the re-scored pass must be
    policy.requires_grad_(True)      # ONE network, or the ratio is between two models
    policy.tokenizer.eval().requires_grad_(False)
    ref = None
    if args.beta_ref > 0.0:
        ref, _ = load_adjuster(args.reference or args.adjuster, dev)
        ref = ref.eval().requires_grad_(False)

    # The env decodes actions with ITS copy of the codebook and the policy scores them with
    # its own. A drift between the two silently changes what every id means, so the spaces
    # are compared rather than assumed equal.
    res_norm = Normalizer.from_state_dict(parent["residual_normalizer"])
    for field in ("mean", "std"):
        a, b = getattr(res_norm, field), getattr(env.residual_normalizer, field).cpu()
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
          f"is -log pi(that) at beta_zero={args.beta_zero}, NOT a KL to the supervised policy")
    print("[rladj]   per group: "
          + "  ".join(f"{g} {v:.4f}" for g, v in zero_decode.items()))

    N = int(env.num_tokens)
    obs0, _ = env.reset()
    critic_obs_dim = int(obs0["critic"].shape[1])
    cond_dim = int(env.planner_conditioning().shape[1])
    plan_shape = tuple(env.plan_window().shape[1:])
    # THE CRITIC READS THE PRIVILEGED GROUP the env already computes -- dense proprio history,
    # the reference window, terrain rays and `goal_geometry`. The policy's conditioning row is
    # the wrong input for it: 97% of it is a flattened depth pair, and its depth channels are
    # unprojected in a LEVEL frame precisely so that tilt does NOT move static terrain, so a
    # critic on it cannot see the tilt that fires the -10 fall term.
    critic = Critic(critic_obs_dim).to(dev)
    opt = torch.optim.AdamW(
        [{"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.lr},
         {"params": critic.parameters(), "lr": args.critic_lr}], weight_decay=0.0)

    out_dir = os.path.join(args.logdir, f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}")
    os.makedirs(out_dir, exist_ok=True)
    git_snapshot(out_dir)
    print(f"[rladj] action = {N} residual tokens over V={policy.vocab_size}; "
          f"critic obs {critic_obs_dim}; plan {plan_shape}; out: {out_dir}")

    @torch.no_grad()
    def sample(model, temperature: float):
        """Draw one macro-step's tokens and re-score them the way the update will.

        The stored log-probabilities come from the TEACHER-FORCED path under the same
        autocast the update uses, so epoch 1's ratio is exactly 1 by construction. The
        sampler's own values come from the incremental KV path and differ by a few parts in a
        hundred at p99 under bf16 -- which the ratio would read as a real policy change.

        Args:
            model: The policy to draw from.
            temperature: Rollout temperature.

        Returns:
            ``(cond, plan_raw, ids, logprobs)``.
        """
        cond = env.planner_conditioning()
        plan_raw = env.plan_window()
        h, e, c, nul = env.split_conditioning(cond)
        plan_tok = model.plan_tokens_from_window(env.plan_normalizer.normalize(plan_raw))
        ids, _, _ = model.sample(h, e, c, nul, temperature=temperature, greedy=False,
                                 plan=plan_tok)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(args.bf16 and dev.type == "cuda")):
            lp, _ = model.logprobs(ids, h, e, c, nul, temperature=temperature, plan=plan_tok)
        return cond, plan_raw, ids, lp.float()

    # ---------------- the gate to run first --------------------------------------------
    if args.validate_only:
        counts = EndingCounts()
        obs = obs0
        for _ in range(args.validate_steps):
            _, _, ids, _ = sample(policy, args.t_roll)
            obs, _, term, trunc, _ = env.step(ids)
            counts.update(term | trunc, trunc, getattr(env, "ended_by", None))
        rt = counts.rates()
        print(f"[rladj] --validate_only over {args.validate_steps} macro-steps (no training):"
              f"\n  episodes ended {counts.episodes} | success {counts.success} "
              f"({rt['success']:.3f}) | fell {counts.fell} ({rt['fell']:.3f}) "
              f"| timeout {counts.timeout} ({rt['timeout']:.3f})"
              f"\n  unattributed {counts.unattributed} | reached the goal AND fell "
              f"{counts.goal_and_fall} (counted as success)"
              f"\n  raw endings by key (these OVERLAP): "
              + ", ".join(f"{k}={v}" for k, v in sorted(counts.by_key.items())), flush=True)
        rec = counts.as_dict()
        rec.update({"alpha": float(env.residual_alpha),
                    "adjuster": os.path.abspath(args.adjuster),
                    "render_decor": args.render_decor})
        with open(os.path.join(out_dir, "validation.json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        env.close()
        return

    writer = SummaryWriter(log_dir=out_dir)
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(project=args.wandb_project,
                                   name=os.path.basename(out_dir), config=vars(args))
        except Exception as exc:                                       # noqa: BLE001
            print(f"[WARN] wandb off: {type(exc).__name__}: {exc}")

    # ---------------- training ---------------------------------------------------------
    obs = obs0
    T = args.rollout
    use_bf16 = bool(args.bf16 and dev.type == "cuda")
    for it in range(args.iters):
        warm = it < args.warmup_iters
        env.residual_alpha = alpha_at(it)
        CO = torch.zeros(T, args.num_envs, critic_obs_dim, device=dev)
        C = torch.zeros(T, args.num_envs, cond_dim, device=dev)
        PW = torch.zeros(T, args.num_envs, *plan_shape, device=dev)
        I = torch.zeros(T, args.num_envs, N, dtype=torch.long, device=dev)
        LP = torch.zeros(T, args.num_envs, N, device=dev)
        R = torch.zeros(T, args.num_envs, device=dev)
        D = torch.zeros(T, args.num_envs, device=dev)
        TRM = torch.zeros(T, args.num_envs, device=dev)
        V = torch.zeros(T + 1, args.num_envs, device=dev)
        counts = EndingCounts()
        t0 = time.time()
        for t in range(T):
            cond, plan_raw, ids, lp = sample(policy, args.t_roll)
            co = obs["critic"]
            with torch.no_grad():
                # Updated as observations ARRIVE, so V[t] (which sets the GAE) and the value
                # targets are fit under the same normalizer state.
                critic.update_norm(co)
                V[t] = critic(co)
            CO[t], C[t], PW[t], I[t], LP[t] = co, cond, plan_raw, ids, lp
            obs, rew, term, trunc, _ = env.step(ids)
            # The reward as the env gave it; the truncation bootstrap is applied inside
            # `compute_gae`, which owns that convention. Doing it here as well is the double
            # count that diverged an earlier run of the sibling trainer.
            R[t] = rew
            # PRECEDENCE: ARRIVAL WINS. IsaacLab evaluates the two managers independently, so
            # a robot that reaches the goal and falls on the same control step fires both.
            # Reaching the goal IS the task and the episode ends there either way.
            term = term & ~trunc
            D[t] = trunc.float()
            TRM[t] = term.float()
            counts.update(term | trunc, trunc, getattr(env, "ended_by", None))
        with torch.no_grad():
            V[T] = critic(obs["critic"])
        adv, ret = compute_gae(R, V, TRM, D, args.gamma, args.lam)
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat = lambda x: x.reshape(-1, *x.shape[2:])                   # noqa: E731
        COf, Cf, PWf, If, LPf, Af, Rf = map(flat, (CO, C, PW, I, LP, advn, ret))
        n = Cf.shape[0]
        mb = max(n // args.minibatches, 1)
        stats = {k: [] for k in ("surr", "zero", "ref_kl", "ent", "vloss", "clip",
                                 "approx_kl", "gn")}
        stop = False
        for _ in range(args.ppo_epochs):
            if stop:
                break
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                cb, pwb, ib, lpb, ab, rb, cob = (Cf[idx], PWf[idx], If[idx], LPf[idx],
                                                 Af[idx], Rf[idx], COf[idx])
                h, e, c, nul = env.split_conditioning(cb)
                plan_tok = policy.plan_tokens_from_window(env.plan_normalizer.normalize(pwb))
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    lp_new, ent, lg = policy.logprobs(
                        ib, h, e, c, nul, temperature=args.t_roll, return_logits=True,
                        plan=plan_tok)
                lp32, lg32 = lp_new.float(), lg.float()
                # PER TOKEN, never summed: the macro-step advantage is broadcast to the N
                # decisions that earned it, and a summed log-probability would throw away
                # which decision an update belongs to.
                ratio = torch.exp(lp32 - lpb)
                a = ab.unsqueeze(-1)
                surr = torch.min(ratio * a, ratio.clamp(1 - args.eps, 1 + args.eps) * a).mean()
                logp_all = F.log_softmax(lg32, dim=-1)
                # THE MAGNITUDE ANCHOR: mass on the zero code, not shape against a reference.
                zero_nll = -logp_all[..., int(zero_code)].mean()
                v = critic(cob)
                vloss = F.mse_loss(v, rb)
                loss = (args.value_coef * vloss if warm else
                        -surr + args.beta_zero * zero_nll
                        - args.lambda_ent * ent.float().mean() + args.value_coef * vloss)
                if ref is not None and not warm:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                         enabled=use_bf16):
                        _, _, lg_ref = ref.logprobs(ib, h, e, c, nul,
                                                    temperature=args.t_roll,
                                                    return_logits=True, plan=plan_tok)
                    kl = (logp_all.exp()
                          * (logp_all - F.log_softmax(lg_ref.float(), -1))).sum(-1).mean()
                    loss = loss + args.beta_ref * kl
                    stats["ref_kl"].append(float(kl))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(critic.parameters()), args.grad_clip))
                opt.step()
                # `forward` stashes the per-layer prefix cache for callers that reuse it;
                # nothing here does, so it is pure retention between iterations.
                policy._last_prefix_cache = None
                if ref is not None:
                    ref._last_prefix_cache = None
                with torch.no_grad():
                    bound = (ratio * a > ratio.clamp(1 - args.eps, 1 + args.eps) * a)
                    stats["surr"].append(float(surr))
                    stats["zero"].append(float(zero_nll))
                    stats["ent"].append(float(ent.float().mean()))
                    stats["vloss"].append(float(vloss))
                    stats["clip"].append(float(bound.float().mean()))
                    stats["approx_kl"].append(float((lpb - lp32).mean()))
                    stats["gn"].append(gn)
            if not warm and abs(_fmean(stats["approx_kl"][-4:])) > args.target_kl:
                stop = True
        _report(it, warm, R, D, advn, ret, COf, Rf, critic, counts, stats, env,
                writer, wandb_run, out_dir, policy, opt, parent, zero_code, time.time() - t0)
    env.close()


def _fmean(v) -> float:
    """Mean of a possibly-empty list, as 0.0.

    Args:
        v: Values.

    Returns:
        The mean, or 0.0.
    """
    return statistics.fmean(v) if v else 0.0


def _report(it, warm, R, D, advn, ret, COf, Rf, critic, counts, stats, env,
            writer, wandb_run, out_dir, policy, opt, parent, zero_code, secs) -> None:
    """Log one iteration, check the divergence gates, and checkpoint.

    The gates are the sibling trainer's, and each caught a run that otherwise looked healthy:
    a return whose SCALE has diverged (explained variance is scale-invariant and cannot see
    it), a policy that has walked away from where its log-probabilities were collected, an
    entropy collapse, and a critic that explains less than nothing after warm-up.

    Args:
        it / warm: Iteration and whether it was critic-only.
        R / D / advn / ret / COf / Rf: Rollout tensors.
        critic: The value head, for explained variance.
        counts: The iteration's endings.
        stats: Per-minibatch scalars.
        env: The env, for alpha and the diagnostic log.
        writer / wandb_run / out_dir: Logging sinks.
        policy / opt / parent / zero_code: For the checkpoint.
        secs: Wall time.
    """
    with torch.no_grad():
        pred = critic(COf)
        ev = 1.0 - float((Rf - pred).var() / (Rf.var() + 1e-8))
    rates = counts.rates()
    sc = {
        "train/reward": float(R.sum(0).mean()),
        "train/return": float(ret.mean()),
        "train/alpha": float(env.residual_alpha),
        "train/entropy": _fmean(stats["ent"]),
        "train/zero_nll": _fmean(stats["zero"]),
        "train/value_loss": _fmean(stats["vloss"]),
        "train/clipped_frac": _fmean(stats["clip"]),
        "train/approx_kl": _fmean(stats["approx_kl"]),
        "train/grad_norm": _fmean(stats["gn"]),
        "train/explained_variance": float(ev),
        "train/done_rate": float(D.mean()),
        "train/warmup": float(warm),
        "train/adv_abs": float(advn.abs().mean()),
        "train/iter_s": secs,
        "outcome/episodes": float(counts.episodes),
    }
    if stats["ref_kl"]:
        sc["train/ref_kl"] = _fmean(stats["ref_kl"])
    for k, v in rates.items():
        sc[f"outcome/{k}"] = float(v)
    for k, v in env.extras.get("log", {}).items():
        if "residual" in k.lower() or "violation" in k.lower():
            sc[f"diag/{k}"] = float(v)
    for k, v in sc.items():
        writer.add_scalar(k, v, it)
    if wandb_run is not None:
        wandb_run.log(sc, step=it)
    # INDEXED, not .get()-with-a-default. `EndingCounts.rates()` names these `success`/`fell`;
    # an earlier version asked for keys that do not exist and printed 0.000 for both while 459
    # of 600 episodes were ending per iteration. A missing headline key must raise.
    print(f"[rladj] it {it:>4}{' (warm)' if warm else ''}  a {sc['train/alpha']:.2f}  "
          f"r {sc['train/reward']:+.3f}  ev {ev:+.3f}  ent {sc['train/entropy']:.2f}  "
          f"zero {sc['train/zero_nll']:.2f}  clip {100 * sc['train/clipped_frac']:.0f}%  "
          f"succ {rates['success']:.3f}  fell {rates['fell']:.3f}  to {rates['timeout']:.3f}  "
          f"n {counts.episodes}  ({secs:.0f}s)", flush=True)

    stop_reason = ""
    if abs(sc["train/return"]) > 1e4:
        stop_reason = (f"|return| {sc['train/return']:.3e} > 1e4 -- the value scale has "
                       "diverged, which explained variance cannot see")
    elif not warm and sc["train/entropy"] < 0.3:
        stop_reason = f"entropy {sc['train/entropy']:.3f} < 0.3 nats"
    elif not warm and it >= args.warmup_iters + 5 and ev < 0.0:
        stop_reason = f"critic explained variance {ev:.3f} < 0 after warm-up"
    if it % args.ckpt_every == 0 or stop_reason or it == args.iters - 1:
        _save(os.path.join(out_dir, f"rl_{it}.pt"), policy, opt, parent, it, zero_code)
    if stop_reason:
        print(f"[rladj] STOP: {stop_reason}", flush=True)
        with open(os.path.join(out_dir, "stop.json"), "w") as fh:
            json.dump({"iter": it, "reason": stop_reason, "scalars": sc}, fh, indent=2)
        raise SystemExit(1)


def _save(path, policy, opt, parent, it, zero_code) -> None:
    """Write an RL checkpoint that ``load_planner_context`` can actually load.

    The graft list includes ``data``, which the sibling trainer's saver omitted -- without it
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
          "rl": {k: v for k, v in vars(args).items() if not k.startswith("_")},
          "adjuster_sha": hashlib.sha256(
              open(args.adjuster, "rb").read(1 << 20)).hexdigest()[:16]}
    for k in ("residual_normalizer", "state_normalizer", "fm_provenance", "data",
              "family_names", "reference_blocks", "num_joints", "num_ee_frames"):
        if k in parent:
            ck[k] = parent[k]
    torch.save(ck, path)


if __name__ == "__main__":
    main()
    simulation_app.close()
