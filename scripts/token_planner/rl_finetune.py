# Copyright 2026 Zachary Olkin. All rights reserved.

"""Simulator-free RL fine-tuning of the tokenized AR planner -- stage C.

Fine-tunes the supervised planner on its OWN free-running samples against a reward that
needs no physics: the distance between the decoded plan and the demonstration future. It
adds no information beyond the clips. What it changes is how the planner SAMPLES -- pushing
it to draw coherent plans rather than incoherent mixes of near-duplicate codes, and reducing
the exposure bias teacher forcing leaves behind.

It is also the dress rehearsal: sampling, per-token log-probabilities, group advantages, a
clipped update and an exact KL to a frozen reference all run end to end here, with the
simulator out of the loop, so that when the simulator does go in the only new thing is the
reward.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/rl_finetune.py \\
        --planner logs/token_planner/<run>/ckpt_90.pt \\
        --fm_config logs/flow_matching/<fm run>/params/fm_config.yaml \\
        --data <recorded>.h5,<augmented>.h5 --run_name rl_v1
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from motion_gen.features import _REPO_ROOT

from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.data import load_recordings
from robot_rl.token_planner.planner import TokenPlanner
from robot_rl.token_planner.rl_reward import PlanReward, group_advantages
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer


def _trainer():
    """Import ``train_planner`` to reuse its dataset wrapper and config loader."""
    path = _REPO_ROOT / "scripts" / "token_planner" / "train_planner.py"
    spec = importlib.util.spec_from_file_location("train_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_planner"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planner", required=True, help="Supervised checkpoint to fine-tune.")
    p.add_argument("--fm_config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--run_name", default="rl_v1")
    p.add_argument("--logdir", default="logs/token_planner")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    # rollout
    p.add_argument("--contexts", type=int, default=64, help="B: contexts per iteration.")
    p.add_argument("--group", type=int, default=8, help="G: plans per context.")
    p.add_argument("--t_roll", type=float, default=1.0)
    # reward weights
    p.add_argument("--w_pos", type=float, default=1.0)
    p.add_argument("--w_rot", type=float, default=0.5)
    p.add_argument("--w_joint", type=float, default=1.0)
    p.add_argument("--w_vel", type=float, default=0.2)
    p.add_argument("--w_jit", type=float, default=1.0)
    p.add_argument("--w_jq", type=float, default=0.5)
    p.add_argument("--leave_one_out", action="store_true", default=False)
    # loss
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.05, help="KL-to-reference weight.")
    p.add_argument("--lambda_ent", type=float, default=0.0)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--freeze_prefix", action="store_true", default=False)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    # schedule
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--ckpt_every", type=int, default=250)
    p.add_argument("--val_every", type=int, default=100)
    p.add_argument("--val_contexts", type=int, default=512)
    p.add_argument("--load_rows", type=int, default=0)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", default="robot_rl_token_planner")
    return p.parse_args()


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(pol, loader, dev, rew, n_ctx: int, tok_ce_ref: float | None,
             fam_names: list[str] | None) -> dict[str, float]:
    """Free-running error under four decode modes, plus health checks.

    ``greedy`` is what deploys; ``T=0.7`` is the tempered draw; ``best8`` re-ranks eight
    samples by joint log-probability; ``mbr8`` serves the sample closest to the other seven
    (minimum Bayes risk), which is the decode rule that does not assume the model's own
    likelihood is a good judge of plan quality.

    Returns:
        A flat dict of scalars.
    """
    out: dict[str, float] = {}
    modes = ("greedy", "t07", "best8", "mbr8")
    acc = {m: {"pos": 0.0, "rot": 0.0, "joint": 0.0, "jit": 0.0, "jit_raw": 0.0}
           for m in modes}
    n, ce_sum, ce_n, div_sum, div_n = 0, 0.0, 0, 0.0, 0
    per_fam: dict[int, list[float]] = {}
    for batch in loader:
        if n >= n_ctx:
            break
        h, e, c, _ = _trainer_cond(batch, dev)
        tgt_ids = batch["tokens"].to(dev, non_blocking=True)
        b = tgt_ids.shape[0]
        target = pol.decode(tgt_ids, pol._rl_norm)
        cache = pol.encode_prefix(h, e, c, None)
        lp, _ = pol.logprobs(tgt_ids, h, e, c, None)
        ce_sum += float(-lp.mean()) * b
        ce_n += b
        for m in modes:
            if m == "greedy":
                ids, _, _ = pol.sample(h, e, c, None, greedy=True, prefix_cache=cache)
            elif m == "t07":
                ids, _, _ = pol.sample(h, e, c, None, temperature=0.7, prefix_cache=cache)
            else:
                k = 8
                hb, eb = h.repeat_interleave(k, 0), e.repeat_interleave(k, 0)
                cb = None if c is None else c.repeat_interleave(k, 0)
                sid, slp, _ = pol.sample(hb, eb, cb, None, temperature=1.0)
                w = pol.decode(sid, pol._rl_norm).view(b, k, *target.shape[1:])
                if m == "best8":
                    j = slp.sum(-1).view(b, k).argmax(-1)
                else:
                    d = (w.unsqueeze(1) - w.unsqueeze(2)).flatten(3).norm(dim=-1).mean(-1)
                    j = d.argmin(-1)
                ids = sid.view(b, k, -1)[torch.arange(b, device=dev), j]
                if m == "best8":
                    div_sum += float(
                        (w.unsqueeze(1) - w.unsqueeze(2)).flatten(3).norm(dim=-1).mean())
                    div_n += 1
            plan = pol.decode(ids, pol._rl_norm)
            _, terms = rew(plan, target)
            for key, src in (("pos", "pos"), ("rot", "rot"), ("joint", "joint"),
                             ("jit", "jitter_excess"), ("jit_raw", "jitter")):
                acc[m][key] += float(terms[src].sum())
            if m == "greedy" and fam_names is not None:
                # Split the DEPLOYED decode mode by terrain family: this stage exists to be
                # read per family, and a pooled number cannot answer that.
                fam = batch["family"].to(dev)
                for f in fam.unique().tolist():
                    per_fam.setdefault(int(f), []).append(
                        float(terms["pos"][fam == f].sum()) / max(int((fam == f).sum()), 1))
        n += b
    for m in modes:
        for k2, v in acc[m].items():
            out[f"val/{m}_{k2}"] = v / max(n, 1)
    out["val/tf_ce"] = ce_sum / max(ce_n, 1)
    out["val/diversity_t1"] = div_sum / max(div_n, 1)
    out["val/n_contexts"] = float(n)
    for f, vals in per_fam.items():
        nm = fam_names[f] if fam_names and f < len(fam_names) else f"fam{f}"
        out[f"val_family/{nm}_greedy_pos"] = float(np.mean(vals))
    if tok_ce_ref:
        out["val/tf_ce_rise"] = out["val/tf_ce"] / tok_ce_ref - 1.0
    return out


_trainer_mod = None


def _trainer_cond(batch, dev):
    """Move one batch's conditioning to ``dev`` (no null draw)."""
    return _trainer_mod._cond(batch, dev, 0.0)


def main() -> None:
    """Run the fine-tune."""
    global _trainer_mod
    args = parse_args()
    _trainer_mod = _trainer()
    tp = _trainer_mod
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ck = torch.load(args.planner, map_location="cpu", weights_only=False)
    cfg = PlannerConfig.from_dict(ck["config"])
    tok_cfg = TokenizerConfig.from_dict(cfg.tokenizer)
    state = ck.get("model_ema") or ck["model"]
    pol = TokenPlanner(cfg, TrajectoryTokenizer(tok_cfg))
    pol.load_state_dict(state)
    # EVAL MODE THROUGHOUT. This model ships dropout=0.1, and the RL stage needs the
    # sampling pass, the recomputed log-probabilities and the reference to all come from the
    # SAME deterministic network: otherwise the PPO ratio is not 1 at the on-policy first
    # step and the exact KL measures two independent dropout draws rather than drift. Only
    # the optimizer changes the weights here; `sample()` refuses train mode for this reason.
    pol = pol.to(dev).eval()
    ref = copy.deepcopy(pol).eval().requires_grad_(False)
    ref_hash = hashlib.sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()[:16]
    pol.tokenizer.eval().requires_grad_(False)          # frozen, both copies
    ref.tokenizer.eval().requires_grad_(False)

    fm = tp.load_fm_config(args.fm_config)
    from robot_rl.token_planner.data import _EXTERO_BY_COST, _recorded_data_io
    keys = [k for k in _recorded_data_io().WINDOW_ARRAY_KEYS
            if k not in set(_EXTERO_BY_COST) - {fm.extero_source}]
    if args.load_rows > 0:
        tp._patch_row_limit(args.load_rows)
    # The split fields the supervised trainer OVERRIDES before building, restated here so
    # this run rebuilds the same dataset rather than whatever the FM config happened to say.
    fm.future_len = tok_cfg.chunk_len
    fm.reference_blocks = list(tok_cfg.reference_blocks)
    data, meta = load_recordings(args.data.split(","), keys=keys)
    train_ds, val_ds = tp.build_train_val(data, meta, fm)
    if val_ds is None or not len(val_ds):
        raise SystemExit("no validation split; the gates and every val metric need one.")
    # THE CHECK THAT MAKES THE REWARD MEAN ANYTHING. The state normalizer is FITTED on the
    # train split, and the frozen tokenizer encodes futures through it. A different seed,
    # val_fraction, stride, --data order or row limit gives a different fit -- and then every
    # `batch["tokens"]` target is a different trajectory from the one the checkpoint was
    # trained against, the reward is scored against that, and nothing changes shape. The
    # supervised trainer asserts this at the same seam; skipping it here was the gap.
    if ck.get("state_normalizer") is not None:
        tp.assert_normalizers_agree(train_ds.state_normalizer, ck["state_normalizer"])
    norm = train_ds.state_normalizer.to(dev)
    pol._rl_norm = norm
    ref._rl_norm = norm
    tr_tok = tp.tokenize_split(pol.tokenizer, train_ds.state, dev)
    va_tok = tp.tokenize_split(pol.tokenizer, val_ds.state, dev)
    # REAL family codes. Zeros here made `batch["family"]` a constant and left the
    # per-terrain split -- the whole reason this stage is being run -- unanswerable.
    fam_names = [str(x) for x in (getattr(meta, "family_names", None) or ["unknown"])]
    fam_all = data.get("clip_family")

    def _fam(ds):
        """Family codes for a split, in the dataset's own filtered index space."""
        sel = getattr(ds, "window_sel", None)
        if fam_all is None or sel is None:
            return torch.zeros(len(ds), dtype=torch.long)
        return torch.as_tensor(np.asarray(fam_all)[np.asarray(sel)], dtype=torch.long)

    fam_t, fam_v = _fam(train_ds), _fam(val_ds)
    train_loader = DataLoader(tp.TupleDataset(train_ds, tr_tok, fam_t),
                              batch_size=args.contexts, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    # A FIXED RANDOM SUBSET, not the first `val_contexts` rows. The windows are stored in
    # episode order, so an unshuffled prefix is a handful of contiguous episodes on one or
    # two tiles -- which read tf_ce 2.59 at 64 contexts, 4.25 at 512 and 2.32 over the whole
    # split, purely from where the prefix stopped. Seeded, so every validation across the run
    # scores the SAME contexts and the gates compare like with like.
    _vidx = torch.randperm(len(val_ds),
                           generator=torch.Generator().manual_seed(12345))[:args.val_contexts]
    val_loader = DataLoader(
        torch.utils.data.Subset(tp.TupleDataset(val_ds, va_tok, fam_v), _vidx.tolist()),
        batch_size=args.contexts, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    rew = PlanReward(int(train_ds.num_joints), int(train_ds.num_extra_frames),
                     tok_cfg.reference_blocks, w_pos=args.w_pos, w_rot=args.w_rot,
                     w_joint=args.w_joint, w_vel=args.w_vel, w_jit=args.w_jit,
                     w_jq=args.w_jq)

    params = [p for p in pol.parameters() if p.requires_grad]
    if args.freeze_prefix:
        for nme, p in pol.named_parameters():
            if nme.startswith("prefix."):
                p.requires_grad_(False)
        params = [p for p in pol.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    run = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    out_dir = os.path.join(args.logdir, run)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=out_dir)
    wb = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415
            wb = wandb.init(project=args.wandb_project, name=run, config=vars(args))
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb off: {type(exc).__name__}: {exc}")
    print(f"[rl] policy from {os.path.basename(args.planner)} epoch {ck.get('epoch')} "
          f"(ref sha {ref_hash}) | B={args.contexts} G={args.group} T={args.t_roll} "
          f"beta={args.beta} eps={args.eps} lr={args.lr}")
    print(f"[rl] out: {out_dir}")

    # Baselines measured on the FROZEN REFERENCE -- which IS the supervised checkpoint that
    # spec 7's gates name. Taking them from the policy at it==0 baselines them one optimizer
    # step in (the update runs before that validation), mislabelling both gates.
    b0 = validate(ref, val_loader, dev, rew, args.val_contexts, None, None)
    base = {"tf_ce": b0["val/tf_ce"], "div": b0["val/diversity_t1"]}
    print(f"[rl] supervised baseline: greedy pos {b0['val/greedy_pos']:.4f} m  "
          f"tf_ce {base['tf_ce']:.4f}  diversity {base['div']:.4f}")
    it, t0, printed_terms = 0, time.time(), False
    stop_reason = ""
    while it < args.iters and not stop_reason:
        for batch in train_loader:
            if it >= args.iters or stop_reason:
                break
            h, e, c, _ = _trainer_cond(batch, dev)
            tgt_ids = batch["tokens"].to(dev, non_blocking=True)
            b, G = tgt_ids.shape[0], args.group
            # Re-keyed EVERY iteration: one generator advanced across the run makes
            # iteration k's draws unreproducible without replaying every prior iteration,
            # and shifts every subsequent draw if --contexts changes.
            g_it = torch.Generator(device=dev.type).manual_seed(args.seed * 1_000_003 + it)
            with torch.no_grad():
                target = pol.decode(tgt_ids, norm)                     # (B,H,D) physical
                hb = h.repeat_interleave(G, 0)
                eb = e.repeat_interleave(G, 0)
                cb = None if c is None else c.repeat_interleave(G, 0)
                ids, _samp_lp, ent0 = pol.sample(hb, eb, cb, None,
                                                 temperature=args.t_roll, generator=g_it)
                # `old_lp` is RE-SCORED through the same teacher-forced path, in the same
                # autocast context, that the update will use -- NOT the log-probs `sample`
                # returned. Those come from the incremental KV path in fp32; the update runs
                # the full block path under bf16, and the two sum the same numbers in a
                # different order. Measured, that alone put |ratio - 1| at a p99 of 3-7% with
                # tails reaching the 0.2 clip band, so the surrogate was being scaled by a
                # per-token dtype artefact and `clipped_frac` was reporting rounding rather
                # than policy movement. Re-scoring makes epoch 1's ratio exactly 1 by
                # construction, which is what `ppo_epochs=1` is supposed to guarantee.
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(args.bf16 and dev.type == "cuda")):
                    old_lp, _, _ = pol.logprobs(ids, hb, eb, cb, None,
                                                temperature=args.t_roll, return_logits=True)
                old_lp = old_lp.detach()
                plan = pol.decode(ids, norm)                           # (B*G,H,D)
                r, terms = rew(plan, target.repeat_interleave(G, 0))
                r = r.view(b, G)
                adv, degen = group_advantages(r, leave_one_out=args.leave_one_out)
            if not printed_terms:
                print("[rl] per-term means on the supervised model's own samples "
                      "(set weights so these are the same order):")
                for k2, v in terms.items():
                    print(f"       {k2:<14}{float(v.mean()):.5f}")
                printed_terms = True
            adv_f = adv.reshape(-1)
            keep = ~degen.repeat_interleave(G)
            n_upd = max(int(keep.sum()), 1)

            for _ in range(max(args.ppo_epochs, 1)):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(args.bf16 and dev.type == "cuda")):
                    lp_new, ent_new, lg_new = pol.logprobs(
                        ids, hb, eb, cb, None, temperature=args.t_roll, return_logits=True)
                    with torch.no_grad():
                        _, _, lg_ref = ref.logprobs(ids, hb, eb, cb, None, temperature=1.0,
                                                    return_logits=True)
                # ratio and KL in fp32: a bf16 exp() of a near-zero difference is where a
                # PPO ratio quietly stops being 1.0 at the on-policy first step.
                lp_new32, lg_new32, lg_ref32 = lp_new.float(), lg_new.float(), lg_ref.float()
                ratio = torch.exp(lp_new32 - old_lp.float())
                a = adv_f.unsqueeze(-1)
                surr = torch.min(ratio * a,
                                 ratio.clamp(1 - args.eps, 1 + args.eps) * a)
                logp_n = F.log_softmax(lg_new32, dim=-1)
                kl_tok = (logp_n.exp() * (logp_n - F.log_softmax(lg_ref32, dim=-1))).sum(-1)
                m = keep.unsqueeze(-1).float()
                loss = (-(surr * m).sum() / (n_upd * ids.shape[1])
                        + args.beta * (kl_tok * m).sum() / (n_upd * ids.shape[1])
                        - args.lambda_ent * (ent_new.float() * m).sum()
                        / (n_upd * ids.shape[1]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
                opt.step()

            with torch.no_grad():
                kl_seq = float((kl_tok * m).sum() / n_upd)             # summed over N tokens
                # The clip BINDS only when the ratio's excursion has the same sign as the
                # advantage; |ratio-1| > eps overcounts it ~2x and, unmasked, counts skipped
                # contexts whose advantage is exactly 0 and which can never clip.
                bind = (torch.min(ratio * a, ratio.clamp(1 - args.eps, 1 + args.eps) * a)
                        != ratio * a)
                clipped = float((bind & keep.unsqueeze(-1)).float().sum()
                                / max(n_upd * ids.shape[1], 1))
                # Distinct codes among the G samples OF ONE CONTEXT, averaged over
                # contexts and positions. Counting `.unique()` over the whole (B, G) slab --
                # which this did -- mixes in cross-context variety, so a policy that has
                # collapsed to ONE plan per context still reads mid-range and never moves.
                # That is the exact failure this stage risks, so the metric has to be able
                # to show it: collapsed now reads 1.0 against a ceiling of G.
                iv = ids.view(b, G, -1)
                dist_per_pos = float(
                    torch.stack([(iv[:, :, t].unsqueeze(-1) == iv[:, :, t].unsqueeze(-2))
                                 .float().mean(dim=(-1, -2)).reciprocal().mean()
                                 for t in range(iv.shape[2])]).mean())
                collapsed = float((iv == iv[:, :1]).all(dim=-1).all(dim=-1).float().mean())
            sc = {
                "train/reward": float(r.mean()),
                "train/reward_best_of_g": float(r.max(dim=1).values.mean()),
                "train/kl_seq_nats": kl_seq,
                "train/entropy": float(ent_new.detach().mean()),
                "train/frac_skipped": float(degen.float().mean()),
                "train/clipped_frac": clipped,
                "train/grad_norm": gnorm,
                "train/loss": float(loss),
                "train/codes_per_pos": dist_per_pos,
                "train/frac_collapsed": collapsed,
            }
            for k2, v in terms.items():
                sc[f"train/term_{k2}"] = float(v.mean())
            for k2, v in sc.items():
                writer.add_scalar(k2, v, it)
            if wb:
                wb.log(sc, step=it)
            if it % 25 == 0:
                print(f"[rl] it {it:>5} r {sc['train/reward']:+.4f} "
                      f"best {sc['train/reward_best_of_g']:+.4f} "
                      f"KL {kl_seq:6.2f} H {sc['train/entropy']:.3f} "
                      f"skip {sc['train/frac_skipped']:.2f} "
                      f"gn {gnorm:.2f} ({time.time() - t0:.0f}s)", flush=True)
            # --- gates -------------------------------------------------------------
            if kl_seq > 15.0:
                stop_reason = f"KL {kl_seq:.1f} > 15 nats"
            if sc["train/frac_skipped"] > 0.5:
                stop_reason = f"skipped {sc['train/frac_skipped']:.2f} > 0.5 of contexts"
            if it % args.val_every == 0:
                v = validate(pol, val_loader, dev, rew, args.val_contexts,
                             base["tf_ce"], fam_names)
                if v.get("val/tf_ce_rise", 0.0) > 0.20:
                    stop_reason = f"val teacher-forced CE rose {v['val/tf_ce_rise']:.1%} > 20%"
                if base["div"] and v["val/diversity_t1"] < 0.25 * base["div"]:
                    stop_reason = ("diversity collapsed to "
                                   f"{v['val/diversity_t1'] / base['div']:.0%} of supervised")
                for k2, val in v.items():
                    writer.add_scalar(k2, val, it)
                if wb:
                    wb.log(v, step=it)
                print(f"[rl] VAL it {it}: greedy pos {v['val/greedy_pos']:.4f} m  "
                      f"mbr8 pos {v['val/mbr8_pos']:.4f}  best8 pos {v['val/best8_pos']:.4f}  "
                      f"tf_ce {v['val/tf_ce']:.4f}  div {v['val/diversity_t1']:.4f}",
                      flush=True)
            if it % args.ckpt_every == 0 or it == args.iters - 1 or stop_reason:
                torch.save({"model": pol.state_dict(), "config": ck["config"],
                            "meta": ck.get("meta"), "epoch": ck.get("epoch"), "rl_iter": it,
                            "state_normalizer": norm.state_dict(),
                            # THIS run's fits, not the parent's: the rollout normalizes
                            # exteroception with `train_ds.extero_normalizer`, so recording
                            # the parent's would describe a different conditioning than the
                            # policy was actually fine-tuned under.
                            "extero_normalizer": train_ds.extero_normalizer.state_dict(),
                            "cmd_normalizer": (train_ds.cmd_normalizer.state_dict()
                                               if train_ds.cmd_normalizer is not None
                                               else None),
                            "rl": {"ref_sha": ref_hash, "beta": args.beta, "eps": args.eps,
                                   "t_roll": args.t_roll, "group": args.group,
                                   "weights": rew.w, "lr": args.lr, "seed": args.seed,
                                   "bf16": bool(args.bf16), "contexts": args.contexts,
                                   "ppo_epochs": args.ppo_epochs,
                                   "lambda_ent": args.lambda_ent,
                                   "leave_one_out": bool(args.leave_one_out),
                                   "freeze_prefix": bool(args.freeze_prefix)}},
                           os.path.join(out_dir, f"rl_{it}.pt"))
            it += 1
    if stop_reason:
        print(f"[rl] STOPPED at it {it}: {stop_reason}")
    with open(os.path.join(out_dir, "stop.json"), "w") as fh:
        json.dump({"iters": it, "stop_reason": stop_reason}, fh, indent=2)
    writer.close()
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
