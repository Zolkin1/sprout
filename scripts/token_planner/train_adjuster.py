# Copyright 2026 Zachary Olkin. All rights reserved.

"""Stage C: fit a token model to CORRECT the flow-matching generator, not to replace it.

The absolute planner does not beat the generator (base_pos 0.067 against 0.038, at roughly
the tokenizer's own floor). What it has that the generator does not is an exact per-token
log-probability and entropy, which is what makes PPO tractable without a reparameterisation
trick. So the generator stays the planner of record and this model rides on top of it,
emitting RESIDUAL tokens that are composed onto the generator's plan.

Two spaces, and confusing them is silent because both are ``(B, H, 44)`` of order-one floats:

* the PLAN is normalized by the GENERATOR's state normalizer -- the space it predicts in;
* the RESIDUAL is normalized by the residual normalizer -- mean pinned to zero, every channel
  standardized including rot6d.

Both travel in the residual cache, and this script never fits either.

.. admonition:: The reported error is a PREDICTION, unlike stage B's

    ``train_tokenizer.py --residual`` reports a CEILING: it is handed the true residual and
    only has to compress it. Here the residual is never shown -- the model sees the same
    depth, command and history the generator saw, plus the generator's own plan, and has to
    say what the generator got wrong. The two numbers are directly comparable and the gap
    between them is what conditioning buys.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))

from provenance import git_snapshot  # noqa: E402

from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig  # noqa: E402
from robot_rl.token_planner.metrics import channel_group_labels  # noqa: E402
from robot_rl.token_planner.planner import TokenPlanner  # noqa: E402
from robot_rl.token_planner.residual import (  # noqa: E402
    RESIDUAL_CACHE_VERSION,
    load_residual_cache,
    plan_space_report,
)
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True,
                   help="A residual build from build_residual_cache.py --stage build. It "
                        "carries the plans, the residuals, the conditioning and the split, "
                        "so this stage is a function of that one file.")
    p.add_argument("--tokenizer", required=True,
                   help="A RESIDUAL tokenizer checkpoint (train_tokenizer.py --residual). "
                        "Its normalizer must be the cache's -- the ids it emits are only "
                        "meaningful in the space it was fitted in, and the check is exact.")
    p.add_argument("--logdir", default="logs/token_planner")
    p.add_argument("--run_name", default="adjuster")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", default="robot_rl_token_planner")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--ckpt_every", type=int, default=20)

    d = PlannerConfig()
    p.add_argument("--d_model", type=int, default=d.d_model)
    p.add_argument("--nhead", type=int, default=d.nhead)
    p.add_argument("--num_layers", type=int, default=d.num_layers)
    p.add_argument("--dim_feedforward", type=int, default=d.dim_feedforward)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--null_cond_prob", type=float, default=d.null_cond_prob)
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="DEFAULT 0, unlike the absolute planner. Smoothing there is "
                        "deliberate -- it widens the exploration distribution PPO samples "
                        "from -- but stage C's job is first to establish whether the residual "
                        "is predictable at all, and a smoothed CE is not that measurement. "
                        "Turn it on for the checkpoint RL will actually start from.")
    p.add_argument("--residual_alpha", type=float, default=d.residual_alpha)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--lr_min", type=float, default=d.lr_min)
    p.add_argument("--weight_decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=d.grad_clip)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


class AdjusterDataset(Dataset):
    """One ``(conditioning, plan, residual tokens)`` triple per window.

    Args:
        chunks: The residual build.
        tokens: ``(Nw, N)`` residual token ids.
        split: ``"train"`` or ``"val"``.
        plan_norm: The GENERATOR's state normalizer, for putting the plan in the space the
            model reads it in.

    Raises:
        ValueError: If the build carries no conditioning -- an older cache predates it, and
            training on one would silently condition the model on nothing.
    """

    def __init__(self, chunks, tokens: torch.Tensor, split: str, plan_norm: Normalizer):
        hist = getattr(chunks, f"{split}_history", None)
        if hist is None:
            raise ValueError(
                f"this residual build carries no {split}_history. Rebuild it with the "
                "current build_residual_cache.py -- the adjuster conditions on what the "
                "GENERATOR saw, and a build without it cannot supply that.")
        self.history = hist
        self.extero = getattr(chunks, f"{split}_extero")
        self.cmd = getattr(chunks, f"{split}_cmd")
        # Normalized ONCE, here: the plan is a fixed input, so re-normalizing it per batch
        # would be the same arithmetic every epoch.
        self.plan = plan_norm.normalize(getattr(chunks, f"{split}_plan"))
        self.plan_raw = getattr(chunks, f"{split}_plan")
        # The TRUE normalized residual, carried for the evaluation only: `plan + denorm(this)`
        # reconstructs the clip exactly, so the gate's target needs no second copy.
        self.residual = getattr(chunks, f"{split}_nodes")
        self.tokens = tokens
        self.family = getattr(chunks, f"{split}_family")
        n = self.history.shape[0]
        for name, arr in (("extero", self.extero), ("plan", self.plan),
                          ("tokens", self.tokens), ("family", self.family)):
            if arr.shape[0] != n:
                raise ValueError(
                    f"{split}: {n} history row(s) but {arr.shape[0]} {name} row(s).")

    def __len__(self) -> int:
        return int(self.history.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """One training triple."""
        out = {
            "history": self.history[i], "extero": self.extero[i],
            "plan": self.plan[i], "plan_raw": self.plan_raw[i],
            "residual": self.residual[i],
            "tokens": self.tokens[i], "family": self.family[i],
        }
        if self.cmd is not None:
            out["cmd_tokens"] = self.cmd[i]
        return out


def load_residual_tokenizer(path: str, chunks) -> tuple[TrajectoryTokenizer, Normalizer]:
    """Load the residual tokenizer and REFUSE one fitted in a different space.

    A token id names a point in the tokenizer's own normalized residual space. Pair a
    checkpoint with a cache whose normalizer differs and every id still decodes, to a
    correction scaled wrong by a per-channel factor -- which no shape, count or loss curve
    would show.

    Args:
        path: A ``train_tokenizer.py --residual`` checkpoint.
        chunks: The residual build it must agree with.

    Returns:
        ``(tokenizer, residual_normalizer)``, the tokenizer frozen and in eval mode.

    Raises:
        SystemExit: On a normalizer or layout mismatch.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TokenizerConfig.from_dict(ck["config"])
    tok = TrajectoryTokenizer(cfg)
    tok.load_state_dict(ck.get("model_ema") or ck["model"])
    tok.eval().requires_grad_(False)

    mine = Normalizer.from_state_dict(chunks.state_normalizer_state)
    theirs = Normalizer.from_state_dict(ck["state_normalizer"])
    for field in ("mean", "std"):
        a, b = getattr(mine, field), getattr(theirs, field)
        if a.shape != b.shape or not torch.allclose(a, b, rtol=0, atol=0):
            raise SystemExit(
                f"the tokenizer's residual normalizer disagrees with the cache's on "
                f"'{field}' (max |diff| "
                f"{float((a - b).abs().max()) if a.shape == b.shape else float('nan'):.3e}). "
                "Its ids name points in ITS space; decoding them against another is a "
                "per-channel rescale of every correction, and nothing downstream would say "
                "so. Use the tokenizer trained on this cache.")
    if int(cfg.chunk_len) != int(chunks.chunk_len) or int(cfg.node_dim) != int(chunks.node_dim):
        raise SystemExit(
            f"tokenizer layout ({cfg.chunk_len} x {cfg.node_dim}) != cache layout "
            f"({chunks.chunk_len} x {chunks.node_dim}).")
    print(f"[adj] residual tokenizer {path}: H={cfg.chunk_len} D={cfg.downsample} -> "
          f"N={cfg.num_tokens} tokens, V={cfg.vocab_size}")
    return tok, mine


def build_config(args: argparse.Namespace, chunks, tok_cfg: TokenizerConfig) -> PlannerConfig:
    """Assemble the planner config from the cache's own generator provenance.

    Every conditioning field is RESOLVED from the build rather than defaulted: the adjuster
    has to read exactly the tokens the generator read, and a mismatch in extero geometry or
    command stride is width-silent for several of them.

    Args:
        args: Parsed CLI.
        chunks: The residual build.
        tok_cfg: The residual tokenizer's config.

    Returns:
        A validated :class:`PlannerConfig` with ``plan_conditioning`` on.
    """
    fm = dict(chunks.fm_provenance.get("fm_config") or {})
    if not fm:
        raise SystemExit(
            "the residual build records no fm_config, so the conditioning geometry the "
            "generator read is unknown. Rebuild the cache with the current "
            "build_residual_cache.py.")
    cfg = PlannerConfig(
        tokenizer=tok_cfg.to_dict(), tokenizer_path=os.path.abspath(args.tokenizer),
        future_len=int(chunks.chunk_len), history_dim=int(chunks.node_dim),
        num_history_states=int(fm["num_history_states"]),
        extero_sensor_dim=int(fm["extero_sensor_dim"]),
        extero_sensor_count=int(fm["extero_sensor_count"]),
        extero_encoder=str(fm["extero_encoder"]),
        extero_image_hw=fm.get("extero_image_hw"),
        extero_cnn_channels=fm.get("extero_cnn_channels", (16, 32, 64)),
        extero_cnn_grid=fm.get("extero_cnn_grid"),
        extero_cnn_strides=fm.get("extero_cnn_strides"),
        extero_coord_extent_m=fm.get("extero_coord_extent_m"),
        extero_grid_ordering=str(fm.get("extero_grid_ordering", "xy")),
        extero_fourier_bands=int(fm.get("extero_fourier_bands", 4)),
        extero_miss_sentinel=fm.get("extero_miss_sentinel"),
        extero_channels=fm.get("extero_channels"),
        extero_camera_intrinsics=fm.get("extero_camera_intrinsics"),
        cmd_dim=int(fm.get("cmd_dim", 0)), cmd_stride=int(fm.get("cmd_stride", 5)),
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        null_cond_prob=args.null_cond_prob, label_smoothing=args.label_smoothing,
        plan_conditioning=True, residual_alpha=args.residual_alpha,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, batch_size=args.batch_size,
        num_epochs=args.num_epochs, grad_clip=args.grad_clip, seed=args.seed,
        bf16=args.bf16,
    )
    cfg.validate()
    return cfg


def scheduled_lr(cfg: PlannerConfig, epoch: int, step: int, steps_per_epoch: int) -> float:
    """Per-step warmup into a per-epoch cosine decay, closed form.

    A pure function of ``(epoch, step)``, matching the tokenizer and the generator, so a
    resume or a stray scheduler call cannot desync it.

    Args:
        cfg: The config.
        epoch: Current epoch.
        step: Step within the epoch.
        steps_per_epoch: Steps per epoch, for the global step count.

    Returns:
        The learning rate.
    """
    g = epoch * steps_per_epoch + step
    if cfg.warmup_steps > 0 and g < cfg.warmup_steps:
        return cfg.lr * (g + 1) / cfg.warmup_steps
    t = epoch / max(cfg.num_epochs - 1, 1)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * t))


def _cond(batch: dict, model: TokenPlanner, device: torch.device, null_prob: float = 0.0):
    """Move one batch's conditioning to ``device`` and fold the plan into prefix tokens.

    Args:
        batch: A collated :class:`AdjusterDataset` item.
        model: The planner, for the plan fold.
        device: Target device.
        null_prob: Probability of nulling a sample's history, as the generator does.

    Returns:
        ``(history, extero, cmd_tokens, null_mask, plan_tokens)``.
    """
    hist = batch["history"].to(device, non_blocking=True)
    ext = batch["extero"].to(device, non_blocking=True)
    cmd = batch["cmd_tokens"].to(device, non_blocking=True) if "cmd_tokens" in batch else None
    plan = model.plan_tokens_from_window(batch["plan"].to(device, non_blocking=True))
    null = None
    if null_prob > 0.0:
        null = torch.rand(hist.shape[0], device=device) < null_prob
    return hist, ext, cmd, null, plan


@torch.no_grad()
def evaluate(model: TokenPlanner, loader: DataLoader, device: torch.device,
             chunks, res_norm: Normalizer, cfg: PlannerConfig) -> dict[str, Any]:
    """Score the adjuster in PLAN space, against the generator it corrects.

    The primary metric, and it is a PREDICTION: the model never sees the residual, only the
    conditioning and the plan the generator drew. Token accuracy and cross-entropy are
    reported beside it but are secondary -- a token is right or wrong, while what a tracker
    feels is where the composed window ended up.

    Two arms, both autoregressive: greedy (what a deployment serves) and one ``T=1`` draw
    (what PPO would explore with). On the absolute planner the ``T=1`` arm was 4.4x worse
    than greedy, which is the failure this is here to catch before RL inherits it.

    Args:
        model: The adjuster.
        loader: Validation loader.
        device: Where to run.
        chunks: The residual build, for the family names.
        res_norm: The residual normalizer.
        cfg: The config.

    Returns:
        A JSON-able report.
    """
    model.eval()
    ce_sum, top1_sum, ent_sum, n_tok = 0.0, 0.0, 0.0, 0
    ids: dict[str, list[torch.Tensor]] = {"greedy": [], "sample": []}
    plans, targets, fams, trues = [], [], [], []
    for batch in loader:
        hist, ext, cmd, _, plan = _cond(batch, model, device)
        tgt = batch["tokens"].to(device)
        logits = model(tgt, hist, ext, cmd, plan=plan)
        logp = F.log_softmax(logits.float(), dim=-1)
        ce_sum += float(-logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum())
        top1_sum += float((logits.argmax(-1) == tgt).float().sum())
        ent_sum += float((-(logp.exp() * logp).sum(-1)).sum())
        n_tok += int(tgt.numel())
        # The prefix `forward` just built -- the two sampled arms condition on the same
        # batch, so re-encoding it twice would be the same arithmetic three times.
        cache = model._last_prefix_cache
        g, _, _ = model.sample(hist, ext, cmd, greedy=True, prefix_cache=cache, plan=plan)
        d, _, _ = model.sample(hist, ext, cmd, temperature=1.0, greedy=False,
                               prefix_cache=cache, plan=plan)
        ids["greedy"].append(g.cpu())
        ids["sample"].append(d.cpu())
        plans.append(batch["plan_raw"])
        trues.append(batch["tokens"])
        # `plan + denorm(true residual)` IS the clip, by the definition of the residual --
        # so the target is reconstructed rather than stored a second time.
        targets.append(batch["plan_raw"] + res_norm.denormalize(batch["residual"]))
        fams.append(batch["family"])

    plan_raw, target = torch.cat(plans), torch.cat(targets)
    family, true_ids = torch.cat(fams), torch.cat(trues)
    out: dict[str, Any] = {
        "windows": int(plan_raw.shape[0]),
        "tf_ce": ce_sum / max(n_tok, 1),
        "tf_top1": top1_sum / max(n_tok, 1),
        "entropy": ent_sum / max(n_tok, 1),
    }
    for name, chunks_of_ids in ids.items():
        tok = torch.cat(chunks_of_ids)
        residual = res_norm.denormalize(
            model.tokenizer.decode_ids(tok.to(device)).float().cpu())
        rep = plan_space_report(plan_raw, target, residual, cfg.num_joints_for_layout,
                                cfg.num_ee_frames_for_layout,
                                list(cfg.tokenizer_cfg.reference_blocks), family,
                                chunks.family_names, model.vocab_size,
                                alpha=float(cfg.residual_alpha))
        out[f"{name}_plan_space"] = rep
        out[f"{name}_tok_acc"] = float((tok == true_ids).float().mean())
    # The generator alone, on the same windows: identical in both arms (it does not depend on
    # the tokens), so it is lifted out as THE baseline every row is read against.
    out["plan_alone"] = out["greedy_plan_space"]["plan"]["rmse"]
    return out


def save_checkpoint(path: str, model: TokenPlanner, opt, cfg: PlannerConfig, chunks,
                    epoch: int, report: dict[str, Any]) -> None:
    """Write a self-describing adjuster checkpoint.

    Everything needed to SERVE it travels here: the config, the embedded residual tokenizer,
    both normalizers and the generator this was fitted against. An adjuster is meaningless
    without that last one -- its output is a correction to a specific model's specific
    mistakes -- so the generator's identity is part of the artifact, not of the filename.

    Args:
        path: Destination.
        model: The adjuster.
        opt: The optimizer.
        cfg: The config.
        chunks: The residual build.
        epoch: Current epoch.
        report: The latest validation report.
    """
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "config": cfg.to_dict(), "epoch": epoch, "report": report,
        "residual_normalizer": chunks.state_normalizer_state,
        "state_normalizer": chunks.absolute_normalizer_state,
        "fm_provenance": chunks.fm_provenance,
        "data": chunks.provenance,
        "family_names": chunks.family_names,
        "reference_blocks": list(chunks.reference_blocks),
        "num_joints": chunks.num_joints, "num_ee_frames": chunks.num_ee_frames,
    }, path)


def main() -> None:
    """Entry point."""
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    chunks = load_residual_cache(
        args.cache, {"residual_version": RESIDUAL_CACHE_VERSION}, payload=payload)
    tokenizer, res_norm = load_residual_tokenizer(args.tokenizer, chunks)
    plan_norm = Normalizer.from_state_dict(chunks.absolute_normalizer_state)
    cfg = build_config(args, chunks, tokenizer.cfg)

    fm = chunks.fm_provenance
    print(f"[adj] correcting {fm.get('fm_run')} (epoch {fm.get('fm_epoch')}, "
          f"draw={fm.get('draw')}, history_source={fm.get('history_source')}, "
          f"honesty={fm.get('honesty')})")

    model = TokenPlanner(cfg, tokenizer).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[adj] prefix {model.prefix_len} tokens (+{model.num_tokens} of them are the "
          f"PLAN) -> {model.num_tokens} output slots, V={model.vocab_size}; "
          f"{trainable / 1e6:.1f}M trainable")

    # Targets, once: the tokenizer is frozen, so a window's residual ids never change.
    tok_of = {}
    for split in ("train", "val"):
        nodes = getattr(chunks, f"{split}_nodes")
        out = []
        for i in range(0, nodes.shape[0], 2048):
            out.append(tokenizer.encode_to_ids(nodes[i:i + 2048].to(device)).cpu())
        tok_of[split] = (torch.cat(out) if out
                         else torch.zeros(0, model.num_tokens, dtype=torch.long))

    train_ds = AdjusterDataset(chunks, tok_of["train"], "train", plan_norm)
    val_ds = AdjusterDataset(chunks, tok_of["val"], "val", plan_norm)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"[adj] data: {len(train_ds)} train / {len(val_ds)} val windows")

    out_dir = os.path.join(args.logdir, f"{_stamp()}_{args.run_name}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "params"), exist_ok=True)
    with open(os.path.join(out_dir, "params", "adjuster_config.yaml"), "w") as fh:
        import yaml  # noqa: PLC0415

        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
    git_snapshot(out_dir)
    print(f"[adj] out: {out_dir}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(project=args.wandb_project,
                                   name=os.path.basename(out_dir), config=cfg.to_dict())
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb off: {type(exc).__name__}: {exc}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay, betas=tuple(cfg.betas))
    use_bf16 = bool(cfg.bf16 and device.type == "cuda")
    steps = max(len(train_loader), 1)
    best = math.inf

    for epoch in range(cfg.num_epochs):
        model.train()
        loss_sum, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            for group in opt.param_groups:
                group["lr"] = scheduled_lr(cfg, epoch, step, steps)
            hist, ext, cmd, null, plan = _cond(batch, model, device, cfg.null_cond_prob)
            tgt = batch["tokens"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(tgt, hist, ext, cmd, null, plan=plan)
            loss = F.cross_entropy(
                logits.float().reshape(-1, model.vocab_size), tgt.reshape(-1),
                label_smoothing=float(cfg.label_smoothing))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            loss_sum += float(loss.detach()) * tgt.shape[0]
            seen += int(tgt.shape[0])

        line = f"[adj] epoch {epoch:4d}  train_ce {loss_sum / max(seen, 1):.4f}"
        if epoch % max(args.val_every, 1) == 0 or epoch == cfg.num_epochs - 1:
            rep = evaluate(model, val_loader, device, chunks, res_norm, cfg)
            g = rep["greedy_plan_space"]
            sel = g["plan_plus_residual"]["rmse"]["base_pos"]
            line += (f"  ce {rep['tf_ce']:.3f}  top1 {100 * rep['tf_top1']:.1f}%  "
                     f"ent {rep['entropy']:.2f}  |  P {rep['plan_alone']['base_pos']:.4f} -> "
                     f"greedy {sel:.4f} m  "
                     f"(T=1 {rep['sample_plan_space']['plan_plus_residual']['rmse']['base_pos']:.4f})")
            if wandb_run is not None:
                flat = {"train/ce": loss_sum / max(seen, 1), "val/tf_ce": rep["tf_ce"],
                        "val/tf_top1": rep["tf_top1"], "val/entropy": rep["entropy"]}
                for arm in ("greedy", "sample"):
                    for grp, v in rep[f"{arm}_plan_space"]["plan_plus_residual"]["rmse"].items():
                        flat[f"val/{arm}/{grp}"] = v
                    flat[f"val/{arm}/tok_acc"] = rep[f"{arm}_tok_acc"]
                for grp, v in rep["plan_alone"].items():
                    flat[f"val/plan_alone/{grp}"] = v
                wandb_run.log(flat, step=epoch)
            if sel < best:
                best = sel
                save_checkpoint(os.path.join(out_dir, "adjuster.pt"), model, opt, cfg,
                                chunks, epoch, rep)
                with open(os.path.join(out_dir, "gating_report.json"), "w") as fh:
                    json.dump({"epoch": epoch, "report": rep}, fh, indent=2)
                line += "  *"
        print(line, flush=True)
        if epoch % max(args.ckpt_every, 1) == 0 and epoch:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg,
                            chunks, epoch, {})

    print(f"[adj] done. best greedy plan-space base_pos {best:.5f} m -> {out_dir}")
    if wandb_run is not None:
        wandb_run.finish()


def _stamp() -> str:
    """Run-directory timestamp, matching the other trainers."""
    import datetime  # noqa: PLC0415

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":
    main()
