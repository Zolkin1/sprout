# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator PPO for the tokenized AR planner -- stage D.

The planner's action is a WHOLE PLAN: ``N`` token ids, decoded through the frozen tokenizer
and executed by the frozen tracker for ``K_EXEC`` control steps inside one env step. So the
agent decides at the plan's cadence, the tracker is part of the robot's dynamics, and the
reward is measured on motion the robot actually produced -- which is what this stage adds over
the sim-free one, where the reward was a distance to a demonstration.

The update is per-token clipped PPO with the MACRO-STEP advantage broadcast to all ``N``
tokens of the plan that earned it, plus an exact per-token KL to a frozen reference planner.
The loss is the sim-free stage's, with the demo-distance advantage replaced by GAE over a
learned critic.

Written as its own loop rather than through rsl-rl's runner: rsl-rl's PPO assumes a Gaussian
actor with one scalar log-probability per transition, and every seam here is per-token over a
12800-way categorical -- storage, ratio, entropy and KL alike. Bending the runner around that
would leave two half-true descriptions of the update instead of one.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/train_rl_sim.py \\
        --task G1-token-planner-rl --clip clips/<lib>.h5 \\
        --fm_checkpoint logs/token_planner/<run>/ckpt_N.pt \\
        --tracker_run logs/rsl_rl/g1/<tracker> --tracking_spec logs/tracking/<spec>.json \\
        --num_envs 1024 --run_name rlsim_v1
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

from isaaclab.app import AppLauncher

# --- CLI, before AppLauncher: hydra composes the env cfg BEFORE Kit boots ----------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-token-planner-rl")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--clip", required=True, help="Clip library (terrain + reference motion).")
parser.add_argument("--clip_device", default="cpu")
parser.add_argument("--fm_checkpoint", required=True, help="Supervised planner to fine-tune.")
parser.add_argument("--reference", default="", help="Frozen KL reference; default = --fm_checkpoint.")
parser.add_argument("--tracker_run", required=True)
parser.add_argument("--tracker_checkpoint", default="model_19999.pt")
parser.add_argument("--tracking_spec", required=True)
parser.add_argument("--goal_regions", default="")
parser.add_argument("--run_name", default="rlsim_v1")
parser.add_argument("--logdir", default="logs/token_planner_rl")
parser.add_argument("--seed", type=int, default=0)
# rollout / update
parser.add_argument("--rollout", type=int, default=24, help="Macro-steps per env per iteration.")
parser.add_argument("--iters", type=int, default=3000)
parser.add_argument("--ppo_epochs", type=int, default=3)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--t_roll", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=0.985)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--eps", type=float, default=0.2)
parser.add_argument("--beta", type=float, default=0.02, help="KL-to-reference weight.")
parser.add_argument("--lambda_ent", type=float, default=0.002)
parser.add_argument("--value_coef", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--target_kl", type=float, default=0.03,
                    help="Approximate-KL early stop within an iteration's epochs.")
parser.add_argument("--warmup_iters", type=int, default=20,
                    help="Critic-only iterations; rollouts from the REFERENCE planner.")
parser.add_argument("--ckpt_every", type=int, default=50)
parser.add_argument("--validate_only", action="store_true",
                    help="Milestone 1: run the reference planner with NO training and report "
                         "success rates, then exit. Required gate before any RL iteration.")
parser.add_argument("--validate_steps", type=int, default=200)
parser.add_argument("--render_decor", action=argparse.BooleanOptionalAction, default=None,
                    help="Force the terrain's VISUAL-ONLY decoration on or off. It lives on "
                         "its own prim that the depth cameras cast against and physics "
                         "ignores, so it is the one setting that changes what the generator "
                         "SEES without changing what the robot stands on. None = leave the "
                         "task's own value.")
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--wandb_project", default="robot_rl_token_planner_rl")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- everything below runs AFTER Kit is up ----------------------------------------------
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

# MUST come after SimulationApp: it patches `activate_contact_sensors` for RECURSIVE
# traversal, and without it every contact sensor on a nested link path fails to find a body
# with the reporter API -- the env dies in `sim.reset()` pointing at the sensor rather than at
# the missing patch. `train.py` imports it at the same point and for the same reason.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401,E402
import robot_rl.tasks  # noqa: F401,E402  (registers the tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig  # noqa: E402
from robot_rl.token_planner.planner import TokenPlanner  # noqa: E402
from robot_rl.token_planner.rl_gae import compute_gae  # noqa: E402
from robot_rl.token_planner.rl_outcomes import EndingCounts  # noqa: E402
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: E402


class Critic(nn.Module):
    """A plain MLP state-value head with running input and return normalization.

    Deliberately NOT the planner's trunk: the spec asks for a separate network, and sharing
    would couple the value loss's gradients into the very encoder whose drift the KL term is
    trying to bound.

    Args:
        obs_dim: Width of the critic observation.
        hidden: Hidden widths.
    """

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (512, 256, 128)) -> None:
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ELU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("_mu", torch.zeros(obs_dim))
        self.register_buffer("_var", torch.ones(obs_dim))
        self.register_buffer("_n", torch.tensor(1e-4))

    def update_norm(self, x: torch.Tensor) -> None:
        """Welford update of the input normalizer over a rollout batch."""
        b = x.shape[0]
        mu, var = x.mean(0), x.var(0, unbiased=False)
        d = mu - self._mu
        tot = self._n + b
        self._mu += d * b / tot
        self._var.copy_((self._var * self._n + var * b + d.pow(2) * self._n * b / tot) / tot)
        self._n.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Value estimate ``[B]`` for a normalized observation."""
        z = (x - self._mu) / (self._var.sqrt() + 1e-6)
        return self.net(z.clamp(-10.0, 10.0)).squeeze(-1)


def load_planner(path: str, device) -> TokenPlanner:
    """Load a planner checkpoint, preferring EMA weights.

    Args:
        path: Checkpoint path.
        device: Torch device.

    Returns:
        The model, on ``device``.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PlannerConfig.from_dict(ck["config"])
    m = TokenPlanner(cfg, TrajectoryTokenizer(TokenizerConfig.from_dict(cfg.tokenizer)))
    m.load_state_dict(ck.get("model_ema") or ck["model"])
    return m.to(device)


def main() -> None:
    """Collect, evaluate, update."""
    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.tracker_run_dir = args.tracker_run
    env_cfg.tracker_checkpoint = args.tracker_checkpoint
    env_cfg.tracking_spec_path = args.tracking_spec
    # Set on the COMMAND too: `__post_init__` copied `roll_temperature` into
    # `planner_temperature` before this line runs, so setting only the env field leaves the
    # env's own sampler at the default while the policy samples at `--t_roll`.
    env_cfg.roll_temperature = args.t_roll
    env_cfg.commands.traj_ref.planner_temperature = args.t_roll
    if args.goal_regions:
        env_cfg.goal_regions_path = args.goal_regions
    # The terrain importer carries its OWN copy of the library path and it must stay
    # identical to the command's: one file defines both what the robot tracks and what it
    # walks on, and a mismatch is a robot tracking one tile's motion on another's geometry.
    terrain = getattr(env_cfg.scene, "terrain", None)
    if args.render_decor is not None and terrain is not None \
            and hasattr(terrain, "render_decor"):
        terrain.render_decor = bool(args.render_decor)
        print(f"[rlsim] render_decor={terrain.render_decor} (visual only; physics unchanged)")
    if terrain is not None and hasattr(terrain, "clip_library_path"):
        terrain.clip_library_path = args.clip
    # Fanned out BY hasattr over every command term, exactly as `train.py` does: the field
    # is `path` (not `clip_path`), and setting it on one term while another still points at a
    # stale library is how a run tracks one library's motion on another's terrain.
    for _nm, _term in vars(env_cfg.commands).items():
        for _f, _v in (("path", args.clip), ("clip_device", args.clip_device),
                       ("fm_checkpoint", args.fm_checkpoint)):
            if hasattr(_term, _f):
                setattr(_term, _f, _v)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    dev = torch.device(env.device)   # `env.device` is a STRING on this env

    # THE POLICY IS THE ENV'S OWN PLANNER. Loading a second copy would leave two models that
    # must be kept identical by hand -- and the env decodes actions with its copy's frozen
    # tokenizer, so a drift between them is a silent change of what the tokens mean.
    policy = env.planner_ctx.model
    policy.eval()                       # dropout OFF: the ratio and the exact KL both need
    policy.requires_grad_(True)         # the sampled and re-scored passes to be one network
    policy.tokenizer.eval().requires_grad_(False)
    ref = load_planner(args.reference or args.fm_checkpoint, dev).eval().requires_grad_(False)
    # Kept for the checkpoint graft below: the frozen contract the RL result must carry.
    parent = torch.load(args.fm_checkpoint, map_location="cpu", weights_only=False)
    ref_sha = hashlib.sha256(
        open(args.reference or args.fm_checkpoint, "rb").read(1 << 20)).hexdigest()[:16]

    N = env.num_tokens
    # THE CRITIC READS THE PRIVILEGED GROUP the env already computes every step -- dense
    # proprio history, the reference window, terrain rays and `goal_geometry`. The planner's
    # conditioning row is the wrong input for it: 97% of it is a flattened depth pair, it
    # carries no MEASURED robot state at all (its history rows are the reference), and the
    # depth channels are unprojected in a LEVEL frame precisely so that tilt does not move
    # static terrain. So a critic on it cannot see the tilt that fires the -10 fall term, nor
    # the goal that the arrival bootstrap is worth -- the two things the return is made of.
    obs0, _ = env.reset()
    if "critic" not in obs0:
        raise SystemExit(
            "the env exposes no 'critic' observation group; this task is expected to carry "
            "the adjuster's privileged group (proprio history, reference window, terrain "
            "rays, goal geometry).")
    critic_obs_dim = obs0["critic"].shape[1]
    print(f"[rlsim] critic obs: {critic_obs_dim} dims (privileged group) | policy conditions "
          f"on {env.planner_conditioning().shape[1]} dims (planner stash)")
    cond_dim = env.planner_conditioning().shape[1]
    critic = Critic(critic_obs_dim).to(dev)
    opt = torch.optim.AdamW([
        {"params": [p for p in policy.parameters() if p.requires_grad], "lr": args.lr},
        {"params": critic.parameters(), "lr": args.critic_lr},
    ], weight_decay=0.0)

    run = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    out_dir = os.path.join(args.logdir, run)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)
    wb = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415
            wb = wandb.init(project=args.wandb_project, name=run, config=vars(args))
        except Exception as exc:                                      # noqa: BLE001
            print(f"[WARN] wandb off: {type(exc).__name__}: {exc}")
    print(f"[rlsim] {args.num_envs} envs, {N} tokens/plan, {env.replan_period} control "
          f"steps/agent step | ref {ref_sha} | out {out_dir}", flush=True)

    def sample(model, temperature):
        """Draw one plan per env from ``model`` on the live conditioning."""
        cond = env.planner_conditioning()
        h, e, c, nul = env.split_conditioning(cond)
        with torch.no_grad():
            ids, _, _ = model.sample(h, e, c, nul, temperature=temperature)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(args.bf16 and dev.type == "cuda")):
                # RE-SCORED through the teacher-forced path in the autocast the UPDATE uses.
                # `sample` returns log-probs from the incremental KV path in fp32; the update
                # runs the block path in bf16 and the two sum the same numbers in a different
                # order, which put |ratio-1| into the clip band's tail on the sim-free stage.
                lp, _, _ = model.logprobs(ids, h, e, c, nul, temperature=temperature,
                                          return_logits=True)
        return cond, ids, lp.detach().float()

    # ---------------- milestone 1: no-training validation ------------------------------
    if args.validate_only:
        env.reset()
        # `EndingCounts` owns the success/fall/timeout convention and keeps the RAW per-key
        # tallies, so the headline split can be re-derived offline under any convention
        # without paying for another 20-minute rollout.
        counts = EndingCounts()
        # HOW FAR IN does each ending happen? The success criterion is entry into a goal
        # REGION, and a region anchored too early is indistinguishable from a policy that
        # traverses: both report a goal. Episode age at the ending, against the clip's own
        # length, is what separates them -- a "success" a quarter of the way through a clip
        # is a mis-anchored gate, not a traverse.
        age = torch.zeros(env.num_envs, device=dev)
        age_goal, age_fall = [], []
        for _ in range(args.validate_steps):
            _, ids, _ = sample(ref, args.t_roll)
            _, _, term, trunc, extras = env.step(ids)
            age += 1.0
            d = (term | trunc)
            if d.any():
                age_goal += age[trunc & d].tolist()
                age_fall += age[term & ~trunc & d].tolist()
                age[d] = 0.0
            counts.update(d, trunc, getattr(env, "ended_by", None))
        import statistics as _st
        k = int(env.replan_period)
        clip_macro = env.max_episode_length / k
        # The clips' OWN lengths, which is what a traverse is measured against.
        _cl = getattr(getattr(env, "_cmd", None), "clip_len", None)
        if _cl is not None:
            _cl = _cl.float() / k
            clip_lo, clip_med, clip_hi = (float(_cl.min()), float(_cl.median()),
                                          float(_cl.max()))
        else:
            clip_lo = clip_med = clip_hi = float("nan")
        def _q(v):
            """Median and range of a list of macro-step ages, as a printable string."""
            if not v:
                return "n/a"
            return (f"median {_st.median(v):.1f} ({_st.median(v) * k * 0.02:.1f} s), "
                    f"range {min(v):.0f}-{max(v):.0f}")
        print(f"\n[rlsim] EPISODE AGE AT ENDING (macro-steps; the episode budget is "
              f"{clip_macro:.0f} macro-steps = {clip_macro * k * 0.02:.1f} s)")
        print(f"  at goal_reached : {_q(age_goal)}   n={len(age_goal)}")
        print(f"  at a fall       : {_q(age_fall)}   n={len(age_fall)}")
        # Compared against the CLIP's own length, not the episode budget. The budget is a
        # deliberate backstop with slack (24 s against clips of 4.0-11.6 s here), so scoring
        # goal age against it makes every genuine traverse look like an early gate -- which
        # is exactly the wrong-denominator mistake this print was added to catch.
        print(f"  (clips in this library run {clip_lo:.1f}-{clip_hi:.1f} macro-steps, "
              f"median {clip_med:.1f}; the {clip_macro:.0f}-macro-step budget is a backstop)")
        if age_goal and _st.median(age_goal) < 0.4 * clip_med:
            print("  *** goals are firing in the first 40% of the MEDIAN CLIP. On a traverse "
                  "task that suggests a mis-anchored goal region. ***")
        rt = counts.rates()
        print(f"\n[rlsim] VALIDATION ({args.validate_steps} macro-steps, reference planner, "
              f"no training):\n  episodes ended {counts.episodes} | success {counts.success} "
              f"({rt['success']:.3f}) | fell {counts.fell} ({rt['fell']:.3f}) "
              f"| timeout {counts.timeout} ({rt['timeout']:.3f})"
              f"\n  unattributed {counts.unattributed} | reached the goal AND fell "
              f"{counts.goal_and_fall} (counted as success)"
              f"\n  raw endings by termination key (these OVERLAP): "
              + ", ".join(f"{k}={v}" for k, v in sorted(counts.by_key.items())), flush=True)
        rec = counts.as_dict()
        rec["render_decor"] = args.render_decor
        rec["checkpoint"] = os.path.abspath(args.reference or args.fm_checkpoint)
        with open(os.path.join(out_dir, "validation.json"), "w") as fh:
            json.dump(rec, fh, indent=2)
        env.close()
        return

    # ---------------- training ---------------------------------------------------------
    obs = obs0
    T = args.rollout
    for it in range(args.iters):
        warm = it < args.warmup_iters
        actor = ref if warm else policy
        CO = torch.zeros(T, args.num_envs, critic_obs_dim, device=dev)   # critic input
        C = torch.zeros(T, args.num_envs, cond_dim, device=dev)          # policy conditioning
        I = torch.zeros(T, args.num_envs, N, dtype=torch.long, device=dev)
        LP = torch.zeros(T, args.num_envs, N, device=dev)
        R = torch.zeros(T, args.num_envs, device=dev)
        D = torch.zeros(T, args.num_envs, device=dev)
        TRM = torch.zeros(T, args.num_envs, device=dev)
        counts = EndingCounts()
        V = torch.zeros(T + 1, args.num_envs, device=dev)
        t0 = time.time()
        for t in range(T):
            cond, ids, lp = sample(actor, args.t_roll)
            co = obs["critic"]
            with torch.no_grad():
                # Updated as observations ARRIVE, so V[t] (which sets the GAE) and the value
                # targets are fit under the same normalizer state. Updating after the rollout
                # made iteration 0's advantages a function of raw ~8000-dim input while its
                # targets were fit on normalized input.
                critic.update_norm(co)
                V[t] = critic(co)
            CO[t], C[t], I[t], LP[t] = co, cond, ids, lp
            obs, rew, term, trunc, extras = env.step(ids)
            # THE ONE TRUNCATION IS ARRIVAL. On this cfg `time_out=True` is set on
            # `goal_reached` ALONE -- `time_out`, `clip_end` and `tracking_violation` are all
            # terminations -- and reaching the goal carries NO reward term, so its entire
            # payoff is `gamma * V(s)`. Dropping the bootstrap there makes running the clock
            # out worth more than arriving, and the optimal policy under that MDP is to track
            # well and never finish. Added into the reward exactly as rsl-rl does it (with
            # `V(s_t)`, since the env has already auto-reset past `s_{t+1}`), which is also
            # what the adjuster runs this env was built for do.
            # The reward as the env gave it. The truncation bootstrap is applied inside
            # `compute_gae`, which owns the convention -- doing it here as well is exactly
            # the double count that diverged rlsim_v2.
            R[t] = rew
            # PRECEDENCE: ARRIVAL WINS. IsaacLab evaluates the two managers independently,
            # so a step can be both terminated and truncated -- a robot that reaches the goal
            # and falls on the same control step fires `goal_reached` and `base_fallen`
            # together. Reaching the goal IS the task; the episode ends there either way, so
            # what the body does on the way in does not undo it. Masking the other way would
            # score a completed traverse as a failure.
            #
            # This is not an opening for "collapse into the goal": the goal region still has
            # to be REACHED, which means traversing the tile. `compute_gae` refuses the
            # ambiguous state rather than silently picking a side, which is how the
            # co-occurrence surfaced at all -- it never fired before `base_fallen` was added.
            both = term & trunc
            term = term & ~trunc
            D[t] = trunc.float()
            TRM[t] = term.float()
            # Same bucketing as the `--validate_only` gate, from the same object, so a
            # training curve and a validation table are never two conventions wearing one
            # set of names. Before this the two disagreed: the loop double-counted an
            # arrival that also fell (into BOTH goal and fall), while the gate scored a
            # `base_fallen`-only ending as a success.
            counts.update(term | trunc, trunc, getattr(env, "ended_by", None))
        with torch.no_grad():
            V[T] = critic(obs["critic"])
        # GAE over macro-steps
        adv, ret = compute_gae(R, V, TRM, D, args.gamma, args.lam)
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat = lambda x: x.reshape(-1, *x.shape[2:])                   # noqa: E731
        COf, Cf, If, LPf, Af, Rf = map(flat, (CO, C, I, LP, advn, ret))
        n = Cf.shape[0]
        mb = max(n // args.minibatches, 1)
        stats = {k: [] for k in ("surr", "kl", "ent", "vloss", "clip", "approx_kl", "gn")}
        stop = False
        for _ in range(args.ppo_epochs):
            if stop:
                break
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                cb, ib, lpb, ab, rb = Cf[idx], If[idx], LPf[idx], Af[idx], Rf[idx]
                cob = COf[idx]
                h, e, c, nul = env.split_conditioning(cb)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(args.bf16 and dev.type == "cuda")):
                    lp_new, ent, lg_new = policy.logprobs(
                        ib, h, e, c, nul, temperature=args.t_roll, return_logits=True)
                    with torch.no_grad():
                        _, _, lg_ref = ref.logprobs(ib, h, e, c, nul, temperature=1.0,
                                                    return_logits=True)
                lp32, lgn, lgr = lp_new.float(), lg_new.float(), lg_ref.float()
                ratio = torch.exp(lp32 - lpb)
                a = ab.unsqueeze(-1)
                surr = torch.min(ratio * a, ratio.clamp(1 - args.eps, 1 + args.eps) * a).mean()
                logp_n = F.log_softmax(lgn, dim=-1)
                kl = (logp_n.exp() * (logp_n - F.log_softmax(lgr, dim=-1))).sum(-1).mean()
                v = critic(cob)
                vloss = F.mse_loss(v, rb)
                loss = (args.value_coef * vloss if warm else
                        -surr + args.beta * kl - args.lambda_ent * ent.float().mean()
                        + args.value_coef * vloss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(critic.parameters()), args.grad_clip))
                opt.step()
                # `TokenPlanner.forward` stashes the per-layer prefix cache on the module for
                # callers that reuse it. Nothing here passes `prefix_cache=`, so it is pure
                # retention -- (B, P, d_model) x num_layers, per model, held between
                # iterations.
                policy._last_prefix_cache = None
                ref._last_prefix_cache = None
                with torch.no_grad():
                    ak = float((lpb - lp32).mean())
                    stats["surr"].append(float(surr)); stats["kl"].append(float(kl))
                    stats["ent"].append(float(ent.float().mean()))
                    stats["vloss"].append(float(vloss)); stats["gn"].append(gn)
                    bind = torch.min(ratio * a, ratio.clamp(1 - args.eps, 1 + args.eps) * a) \
                        != ratio * a
                    stats["clip"].append(float(bind.float().mean()))
                    stats["approx_kl"].append(ak)
                if not warm and abs(statistics.fmean(stats["approx_kl"][-4:])) > args.target_kl:
                    stop = True
                    break
        with torch.no_grad():
            ev = 1.0 - (Rf - critic(COf)).var() / (Rf.var() + 1e-8)
        _rt = counts.rates()
        sc = {"train/reward": float(R.mean()), "train/return": float(ret.mean()),
              "train/kl_seq_nats": statistics.fmean(stats["kl"]) * N,
              "train/entropy": statistics.fmean(stats["ent"]),
              "train/clipped_frac": statistics.fmean(stats["clip"]),
              "train/approx_kl": statistics.fmean(stats["approx_kl"]),
              "train/value_loss": statistics.fmean(stats["vloss"]),
              "train/explained_variance": float(ev),
              "train/grad_norm": statistics.fmean(stats["gn"]),
              "train/done_rate": float(D.mean()), "train/warmup": float(warm),
              # THE GO/NO-GO SIGNATURE. Without these a run doing exactly the wrong thing --
              # tracking well and never finishing -- looks like a run that is working, since
              # the dense reward RISES while arrivals fall.
              "outcome/episodes": float(counts.episodes),
              "outcome/goal_rate": _rt["success"],
              "outcome/fall_rate": _rt["fell"],
              "outcome/timeout_rate": _rt["timeout"],
              # Endings matching no key this code knows about. It is 0 unless the task grows
              # a termination term nobody classified, which is exactly when a rate silently
              # lands in the wrong bucket -- so it is logged rather than assumed away.
              "outcome/unattributed_rate": _rt["unattributed"],
              # How often arrival and a fall land on the SAME step. Logged because the env's
              # own `failed` reward still charges -10 on those steps while this counts them
              # as goals -- a disagreement that is harmless if the count is small and worth
              # fixing in the reward if it is not.
              "outcome/goal_and_fall": _rt["goal_and_fall"],
              "train/iter_s": time.time() - t0}
        for k, v2 in sc.items():
            writer.add_scalar(k, v2, it)
        if wb:
            wb.log(sc, step=it)
        if it % 5 == 0:
            print(f"[rlsim] it {it:>4}{' (warmup)' if warm else ''} r {sc['train/reward']:+.4f} "
                  f"ret {sc['train/return']:+.3f} KL {sc['train/kl_seq_nats']:6.2f} "
                  f"H {sc['train/entropy']:.3f} clip {sc['train/clipped_frac']:.3f} "
                  f"EV {float(ev):+.3f} | goal {sc['outcome/goal_rate']:.3f} "
                  f"fall {sc['outcome/fall_rate']:.3f} n={counts.episodes} "
                  f"({sc['train/iter_s']:.0f}s)", flush=True)
        # --- gates ---------------------------------------------------------------------
        stop_reason = ""
        if abs(sc["train/return"]) > 1e4:
            stop_reason = (f"|return| {sc['train/return']:.3e} > 1e4 -- the value scale is "
                           "diverging. EV cannot see this: it is scale-invariant.")
        if sc["train/kl_seq_nats"] > 20.0:
            stop_reason = f"sequence KL {sc['train/kl_seq_nats']:.1f} > 20 nats"
        if not warm and sc["train/entropy"] < 0.3:
            stop_reason = f"entropy {sc['train/entropy']:.3f} < 0.3 nats"
        if not warm and it >= args.warmup_iters + 5 and float(ev) < 0.0:
            stop_reason = f"critic explained variance {float(ev):.3f} < 0 after warm-up"
        if it % args.ckpt_every == 0 or stop_reason or it == args.iters - 1:
            torch.save({"model": policy.state_dict(), "critic": critic.state_dict(),
                        "config": parent["config"],
                        # Grafted from the supervised parent so the result can actually be
                        # SERVED: `load_planner_context` requires all of these and none is
                        # changed by RL (the tokenizer and the contract are frozen). Without
                        # them a 3000-iteration run produces a file no runtime can load.
                        "state_normalizer": parent.get("state_normalizer"),
                        "extero_normalizer": parent.get("extero_normalizer"),
                        "cmd_normalizer": parent.get("cmd_normalizer"),
                        "meta": parent.get("meta"),
                        "fm_config": parent.get("fm_config"),
                        "epoch": parent.get("epoch"),
                        "rl": {"ref_sha": ref_sha, "parent": os.path.abspath(args.fm_checkpoint),
                               "reference": os.path.abspath(args.reference or args.fm_checkpoint),
                               "clip": os.path.abspath(args.clip),
                               "tracker_run": os.path.abspath(args.tracker_run),
                               "tracking_spec": os.path.abspath(args.tracking_spec),
                               "reward_weights": {"trackability": 1.0, "failed": -10.0,
                                                  "ran_out_of_time": -2.0},
                               "lambda_ent": args.lambda_ent, "value_coef": args.value_coef,
                               "warmup_iters": args.warmup_iters, "num_envs": args.num_envs,
                               "beta": args.beta, "eps": args.eps,
                               "t_roll": args.t_roll, "gamma": args.gamma, "lam": args.lam,
                               "lr": args.lr, "seed": args.seed, "iter": it}},
                       os.path.join(out_dir, f"rl_{it}.pt"))
        if stop_reason:
            print(f"[rlsim] STOPPED at it {it}: {stop_reason}", flush=True)
            with open(os.path.join(out_dir, "stop.json"), "w") as fh:
                json.dump({"iter": it, "stop_reason": stop_reason}, fh, indent=2)
            break
    writer.close()
    if wb:
        wb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
