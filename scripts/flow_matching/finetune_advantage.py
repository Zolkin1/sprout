# Copyright 2026 Zachary Olkin. All rights reserved.

"""Advantage-conditioned fine-tune of a trained flow-matching generator.

Takes a pretrained generator, a rollout of that generator driving the tracker, and a label
saying for each emitted chunk whether it bought anything (``label_advantage.py``).  Trains
the model to reproduce its own chunks with the label as conditioning, using the SAME
supervised flow-matching objective it was pretrained with.  At inference the label is pinned
``positive``, which is the whole intervention: nothing about the loss, the sampler or the
deploy path changes.

Three properties make this safe to run against a shipped checkpoint:

* **the conversion is bit-identical.**  The label embedding is zero-initialised, so before
  the first gradient step the converted model reproduces the original exactly, under every
  label.  ``--verify_identity`` asserts it rather than assuming it.
* **the normalizers come from the pretrained checkpoint**, never refitted.  A refit would
  change what every weight means.
* **it always fine-tunes from the PRETRAINED checkpoint**, never from the previous
  iteration -- so iteration k is one well-defined step from a known model rather than the
  end of an unrepeatable chain.

The number that decides whether any of this worked is the NEGATIVE-conditioning control:
sample the same checkpoint with the label pinned negative.  If it performs the same as
positive, the model is ignoring the label and every other number in the experiment is noise.
Run it before any sweep.

Usage::

    python scripts/flow_matching/finetune_advantage.py \\
        --pretrained logs/flow_matching/<run>/best.pt \\
        --rollout logs/rollouts/v2/rollout_<...>.h5 \\
        --labels logs/value/v2/advantage.npz \\
        --epochs 6 --out logs/flow_matching/<run>_ac_v1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "source", "motion_gen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_rl.flow_matching import online  # noqa: E402
from robot_rl.flow_matching.advantage import (  # noqa: E402
    LABEL_NAMES, LABEL_NULL, label_from_name,
)
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.ema import WeightEMA  # noqa: E402
from robot_rl.flow_matching.flow import flow_matching_loss, sample_trajectory  # noqa: E402
from robot_rl.flow_matching.model import FlowMatchingTransformer  # noqa: E402
from robot_rl.flow_matching.rollout_dataset import (  # noqa: E402
    RolloutWindowDataset, join_labels,
)

#: The state-dict keys the conversion adds, per injection mode.  Anything else missing means
#: the pretrained checkpoint is not the architecture this script thinks it is.
_NEW_KEYS = {
    "time_embed": {"advantage_emb.weight"},
    # The token variant also carries its own type embedding, exactly as every other stream in
    # the prefix does.
    "prefix_token": {"advantage_emb.weight", "advantage_type"},
}


def resolved_config(path: str) -> FlowMatchingConfig:
    """The checkpoint's config, with ``extero_encoder`` resolved the way the loader does.

    ``FmOnlineContext`` does not carry the config, and ``"auto"`` must be settled before
    ``from_config`` builds anything -- so this mirrors ``online.load_fm_context``'s own two
    lines rather than re-deriving the architecture from the weights.

    Args:
        path: A generator ``.pt``.

    Returns:
        The config, encoder resolved.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    cfg.extero_encoder = cfg.resolve_extero_encoder()
    return cfg


def convert(ctx, cfg, device, inject: str | None = None) -> FlowMatchingTransformer:
    """Build the advantage-conditioned twin of a loaded checkpoint's model.

    Args:
        ctx: A loaded :class:`~robot_rl.flow_matching.online.FmOnlineContext`.
        cfg: That checkpoint's resolved config.
        device: Where to put it.
        inject: Override the injection mode ('time_embed' / 'prefix_token').

    Returns:
        A trainable :class:`FlowMatchingTransformer` with ``advantage_cond=True``, carrying
        the pretrained weights and a zero label embedding.

    Raises:
        RuntimeError: If the load is missing anything other than the label embedding, or
            carries an unexpected key.  Either would mean weights silently did not transfer.
    """
    cfg = copy.deepcopy(cfg)
    cfg.advantage_cond = True
    if inject is not None:
        cfg.advantage_inject = inject
    model = FlowMatchingTransformer.from_config(cfg).to(device)
    missing, unexpected = model.load_state_dict(ctx.model.state_dict(), strict=False)
    want = _NEW_KEYS[cfg.advantage_inject]
    if set(missing) != want or unexpected:
        raise RuntimeError(
            f"converting to advantage_cond ({cfg.advantage_inject}) should add exactly "
            f"{sorted(want)}; got missing={sorted(missing)} unexpected={sorted(unexpected)}. "
            "The pretrained checkpoint is not the architecture this config describes.")
    return model


@torch.no_grad()
def verify_identity(ctx, model, ds, device, n: int = 4) -> float:
    """Assert the freshly-converted model reproduces the original under EVERY label.

    Zero-init makes the two models mathematically identical, so any difference is float
    noise rather than a statistical near-miss.  It is not bit-identical on a GPU -- two
    module instances with the same weights can still pick different cuBLAS algorithms --
    which is why the tolerance is 1e-5 against a measured ~4e-6 on the shipped 27M model,
    four orders below the ~1e-2 the label is expected to move things once trained.  Checked
    here against REAL conditioning as well as in the unit tests, which build a toy model.

    Args:
        ctx: The loaded context (its ``model`` is the original).
        model: The converted model.
        ds: A dataset to draw conditioning from.
        device: Device.
        n: Number of windows to check.

    Returns:
        The largest absolute difference seen.

    Raises:
        RuntimeError: If any label changes the output.
    """
    items = [ds[i] for i in range(min(n, len(ds)))]
    hist = torch.stack([it["history"] for it in items]).to(device)
    ext = torch.stack([it["extero"] for it in items]).to(device)
    cmd = (torch.stack([it["cmd_tokens"] for it in items]).to(device)
           if "cmd_tokens" in items[0] else None)
    null = torch.stack([it["history_null"] for it in items]).to(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    noise = torch.randn(len(items), ctx.F, ctx.D, generator=gen).to(device)
    base = sample_trajectory(ctx.model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                             history_null_mask=null, noise=noise)
    worst = 0.0
    for label in range(len(LABEL_NAMES)):
        adv = torch.full((len(items),), label, dtype=torch.long, device=device)
        got = sample_trajectory(model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                                history_null_mask=null, noise=noise, advantage=adv)
        d = float((got - base).abs().max())
        worst = max(worst, d)
        if d > 1e-5:
            raise RuntimeError(
                f"the converted model is NOT identical to its parent under label "
                f"'{LABEL_NAMES[label]}': max |delta| = {d:.3e}. The embedding must be "
                "zero-initialised, or the conversion moved something else.")
    return worst


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of dataset items, keeping the optional keys optional."""
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    return out


def _split_rows(uid: np.ndarray, rows: np.ndarray, val_frac: float, seed: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Split rows by EPISODE, so near-duplicate windows do not straddle the split.

    Args:
        uid: ``[W]`` episode id per FILE row.
        rows: Candidate rows.
        val_frac: Fraction of episodes held out.
        seed: RNG seed.

    Returns:
        ``(train_rows, val_rows)``.
    """
    eps = np.unique(uid[rows])
    rng = np.random.default_rng(seed)
    n = max(1, int(round(val_frac * eps.size))) if val_frac > 0 else 0
    held = set(rng.choice(eps, size=min(n, eps.size), replace=False).tolist()) if n else set()
    is_val = np.array([int(u) in held for u in uid[rows]])
    return rows[~is_val], rows[is_val]


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained", required=True, help="The generator .pt to fine-tune.")
    p.add_argument("--rollout", required=True, help="Rollout window .h5.")
    p.add_argument("--labels", default="",
                   help="advantage.npz from label_advantage.py. Omit for an UNCONDITIONED "
                        "self-distillation control -- the same data and the same number of "
                        "steps with the label carrying nothing, which is what separates 'the "
                        "label helped' from 'more training on rollout data helped'.")
    p.add_argument("--out", required=True, help="Output run directory.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-5,
                   help="Deliberately well below the pretraining 3e-4: this is a fine-tune "
                        "from a shipped model, and the failure mode is losing motion quality "
                        "that no success rate will show. Applies to every parameter EXCEPT "
                        "advantage_emb when --emb_lr_multiplier != 1.")
    p.add_argument("--advantage_inject", default="time_embed",
                   choices=("time_embed", "prefix_token"),
                   help="HOW the label reaches the network. `time_embed` adds it to the "
                        "flow-time embedding (zero-init, conversion exactly inert, ONNX "
                        "shapes unchanged) and is MEASURED INADEQUATE here -- across 1x/50x "
                        "embedding lr and a 10x whole-model lr it moved the served window by "
                        "0.3-0.7% of ordinary sampling noise. `prefix_token` gives the label "
                        "its own token in the conditioning prefix, with its own attention "
                        "weights per layer, so it can modulate WHICH conditioning is attended "
                        "to; it grows prefix_len by 1, so its exported graph is not "
                        "interchangeable with a time_embed one.")
    p.add_argument("--emb_lr_multiplier", type=float, default=1.0,
                   help="advantage_emb.weight gets lr * this, every other parameter gets "
                        "lr unchanged -- a separate AdamW param group, not a global scale. "
                        "MOTIVATION, measured on the v2/n=12 run: the embedding is the only "
                        "parameter starting at exact zero (everything else is pretrained and "
                        "converged), and at multiplier=1 it moved to a row norm of ~0.03-0.04 "
                        "against the ~17-34 scale of the time_mlp output it is summed with -- "
                        "three orders of magnitude too small to matter, while flow_mse barely "
                        "moved. Raising the WHOLE model's lr to compensate would move the "
                        "converged 26.8M-param bulk just as fast, which is the motion-quality "
                        "risk --lr is conservative to avoid. A multiplier here targets only "
                        "the parameter that actually needs a bigger step. Try ~100-300x as a "
                        "starting point (an extrapolation, not a derivation) and re-check the "
                        "embedding norm against the target scale before trusting a full run.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--null_prob", type=float, default=0.3,
                   help="Per-access probability of overriding the label with null, so "
                        "classifier-free guidance over the label is available at inference.")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_windows", type=int, default=0,
                   help="Cap the number of windows used, for a smoke test.")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--drop_tile_holdout", action="store_true",
                   help="Train on training tiles only, keeping the held-out tiles clean for "
                        "evaluation. Requires --labels (the sidecar carries the split).")
    p.add_argument("--verify_identity", action=argparse.BooleanOptionalAction, default=True,
                   help="Assert the converted model matches its parent before training.")
    p.add_argument("--wandb", action="store_true", default=True)
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="robot_rl_flow_matching")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    ctx = online.load_fm_context(args.pretrained, device=args.device)
    cfg = resolved_config(args.pretrained)
    if cfg.advantage_cond:
        raise SystemExit(
            f"{args.pretrained} is ALREADY advantage-conditioned. Every iteration fine-tunes "
            "from the pretrained checkpoint, never from the previous one -- chaining makes "
            "iteration k an unrepeatable path rather than one defined step.")
    model = convert(ctx, cfg, device, args.advantage_inject)

    import h5py
    with h5py.File(args.rollout, "r") as f:
        n_win = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        uid = f["episode_uid"][:n_win]
        ridx = f["replan_index"][:n_win]
    labels = None
    keep = np.ones(n_win, dtype=bool)
    if args.labels:
        side = dict(np.load(args.labels))
        labels = join_labels(side, uid, ridx)
        meta = json.loads(str(side["meta"])) if "meta" in side else {}
        n_hit = int((labels != LABEL_NULL).sum())
        print(f"[ft] labels: {n_hit}/{n_win} windows carry a non-null label "
              f"(n={meta.get('n')}, variant={meta.get('variant')}, "
              f"folds={meta.get('num_folds')})")
        if args.drop_tile_holdout:
            hold = join_labels({"label": side["val_tile"].astype(np.int8),
                                "episode_uid": side["episode_uid"],
                                "replan_index": side["replan_index"]}, uid, ridx)
            keep = hold != 1
            print(f"[ft] dropping {int((~keep).sum())} window(s) on held-out tiles")
    elif args.drop_tile_holdout:
        raise SystemExit("--drop_tile_holdout needs --labels: the tile split lives in the "
                         "sidecar, and guessing it would silently train on the holdout.")

    rows = np.flatnonzero(keep)
    if args.max_windows and rows.size > args.max_windows:
        print(f"[ft] --max_windows: using {args.max_windows} of {rows.size} windows")
        rows = rows[:args.max_windows]
    tr_rows, va_rows = _split_rows(uid, rows, args.val_frac, args.seed)
    common = dict(ctx=ctx, label=labels, seed=args.seed)
    train_ds = RolloutWindowDataset(args.rollout, tr_rows, null_prob=args.null_prob, **common)
    val_ds = (RolloutWindowDataset(args.rollout, va_rows, null_prob=0.0, **common)
              if va_rows.size else None)
    print(f"[ft] {len(train_ds)} train / {0 if val_ds is None else len(val_ds)} val windows "
          f"over {int(np.unique(uid[rows]).size)} episodes")
    if labels is not None:
        print(f"[ft] train label counts (before null dropout): {train_ds.label_counts()}")

    if args.verify_identity and args.advantage_inject == "time_embed":
        worst = verify_identity(ctx, model, train_ds, device)
        print(f"[ft] conversion is identical to the parent under all {len(LABEL_NAMES)} "
              f"labels (max |delta| {worst:.2e})")
    elif args.verify_identity:
        # A TOKEN is a key in the attention softmax, so its presence changes every other
        # token's output at ANY initialisation -- there is no inert conversion to assert, and
        # claiming one would be false. What replaces it is measuring how far the untrained
        # conversion already sits from its parent, so a later change is read against that
        # baseline rather than against an assumed zero.
        with torch.no_grad():
            items = [train_ds[i] for i in range(min(4, len(train_ds)))]
            hist = torch.stack([it["history"] for it in items]).to(device)
            ext = torch.stack([it["extero"] for it in items]).to(device)
            cmd = (torch.stack([it["cmd_tokens"] for it in items]).to(device)
                   if "cmd_tokens" in items[0] else None)
            nullm = torch.stack([it["history_null"] for it in items]).to(device)
            noise = torch.randn(len(items), ctx.F, ctx.D,
                                generator=torch.Generator().manual_seed(0)).to(device)
            base = sample_trajectory(ctx.model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                                     history_null_mask=nullm, noise=noise)
            adv = torch.full((len(items),), 1, dtype=torch.long, device=device)
            got = sample_trajectory(model, hist, ext, cmd, ctx.F, ctx.D, ctx.num_steps,
                                    history_null_mask=nullm, noise=noise, advantage=adv)
            d = float((got - base).abs().mean())
        print(f"[ft] prefix_token: conversion is NOT inert by construction (a token changes "
              f"the attention softmax at any init). Untrained |delta| from parent: {d:.4f}")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=_collate, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = (torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=_collate) if val_ds is not None else None)

    if args.emb_lr_multiplier != 1.0:
        emb_params = list(model.advantage_emb.parameters())
        emb_ids = {id(p) for p in emb_params}
        rest = [p for p in model.parameters() if id(p) not in emb_ids]
        opt = torch.optim.AdamW(
            [{"params": rest, "lr": args.lr},
             {"params": emb_params, "lr": args.lr * args.emb_lr_multiplier}],
            weight_decay=args.weight_decay)
        print(f"[ft] advantage_emb gets its own param group: lr={args.lr * args.emb_lr_multiplier:.2e} "
              f"({args.emb_lr_multiplier}x the base {args.lr:.2e}); everything else at {args.lr:.2e}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = WeightEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    wb = None
    if args.wandb:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project,
                            name=os.path.basename(os.path.abspath(args.out)),
                            config={**vars(args), "n_train": len(train_ds)})
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}")

    def _run(batch, train: bool) -> dict:
        """One forward (and optionally backward) pass over a collated batch."""
        x1 = batch["state"].to(device)
        out = flow_matching_loss(
            model, x1, batch["history"].to(device), batch["extero"].to(device),
            cmd_tokens=(batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None),
            history_null_mask=batch["history_null"].to(device),
            advantage=(batch["advantage"].to(device) if "advantage" in batch else None),
            # No anchor mask and no auxiliary terms: the target is the model's OWN chunk, so
            # there is no clip to be off-axis from and the consistency terms would be
            # regressing the model onto its own already-consistent output.
            blocks=ctx.reference_blocks,
        )
        if train:
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
        return {k: float(v.detach()) for k, v in out.items()}

    history, best_val = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        for batch in loader:
            m = _run(batch, True)
            tot += m["flow_mse"] * batch["state"].shape[0]
            seen += batch["state"].shape[0]
        train_mse = tot / max(seen, 1)
        rec = {"epoch": epoch + 1, "train_flow_mse": train_mse,
               "secs": round(time.time() - t0, 1)}
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vt, vn = 0.0, 0
                for batch in val_loader:
                    vt += _run(batch, False)["flow_mse"] * batch["state"].shape[0]
                    vn += batch["state"].shape[0]
            rec["val_flow_mse"] = vt / max(vn, 1)
        history.append(rec)
        print(f"[ft] epoch {epoch + 1}/{args.epochs}  train {train_mse:.5f}"
              + (f"  val {rec['val_flow_mse']:.5f}" if "val_flow_mse" in rec else "")
              + f"  ({rec['secs']}s)")
        if wb is not None:
            wb.log(rec)
        _save(args, cfg, model, opt, ema, epoch + 1, history,
              os.path.join(args.out, "last.pt"))
        # `best.pt` is what every downstream consumer reaches for by convention, so it has to
        # BE the best rather than the last -- a file named for a selection nothing performed
        # is the kind of quiet claim this stack has been bitten by. With no val split there
        # is no selection to make and the last epoch is the honest answer.
        val = rec.get("val_flow_mse")
        if val is None or val < best_val:
            best_val = float("inf") if val is None else val
            _save(args, cfg, model, opt, ema, epoch + 1, history,
                  os.path.join(args.out, "best.pt"))
            if val is not None:
                rec["is_best"] = True
    # SERVABLE copies, one per arm. `best.pt` names no arm, deliberately -- a conditioned
    # checkpoint that does not state one is refused at load, so nothing (an Isaac eval, a
    # probe, a play run) can serve an arm by accident. These two say which they are, in the
    # artifact, exactly as `fm_meta.advantage_label` does on the ONNX side.
    for name in ("positive", "negative"):
        stamp_serve_label(os.path.join(args.out, "best.pt"),
                          os.path.join(args.out, f"best_{name}.pt"), name)
        print(f"[ft] wrote {args.out}/best_{name}.pt")
    with open(os.path.join(args.out, "finetune.json"), "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "label_counts": train_ds.label_counts()}, f, indent=2)
    if wb is not None:
        wb.finish()
    print(f"[ft] wrote {args.out}/best.pt")
    print("[ft] NEXT, before any sweep: run BOTH arms and compare them.\n"
          f"     Isaac:  --fm_checkpoint pos={args.out}/best_positive.pt "
          f"neg={args.out}/best_negative.pt\n"
          f"     ONNX :  export_fm_onnx.py --checkpoint {args.out}/best.pt "
          "--advantage_label positive|negative\n"
          "     If the two perform the same, the model ignored the label and every other "
          "number here is noise.")
    return 0


def stamp_serve_label(src: str, dst: str, label: str) -> None:
    """Copy a conditioned checkpoint, naming the arm it serves.

    The label is not recoverable from the weights -- one parameter set answers to all three --
    so it has to travel in the artifact, the same way ``attention_mode`` and ``extero_clip``
    already do and for the same reason. A consumer then reads which arm it holds instead of
    being told at the call site, where it could be told wrongly.

    Args:
        src: The unlabelled checkpoint.
        dst: Where to write the labelled copy.
        label: One of :data:`~robot_rl.flow_matching.advantage.LABEL_NAMES`.

    Raises:
        ValueError: On an unrecognised label, or a checkpoint with no label embedding.
    """
    label_from_name(label)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not ck["config"].get("advantage_cond"):
        raise ValueError(f"{src} is not advantage-conditioned; there is no arm to name.")
    ck["config"]["advantage_serve_label"] = label
    torch.save(ck, dst)


def _save(args, cfg, model, opt, ema, epoch: int, history: list, path: str) -> None:
    """Write a checkpoint in the pretrained file's own format.

    Everything except the model weights, the config and the provenance is copied from the
    parent -- the normalizers above all, which were NOT refitted and must travel unchanged
    or nothing downstream (the ONNX export, the deploy contract checks) will agree with the
    generator this came from.
    """
    parent = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(cfg)
    cfg.advantage_cond = True
    cfg.advantage_inject = args.advantage_inject
    ckpt = dict(parent)
    ckpt["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ckpt["config"] = cfg.to_dict()
    ckpt["optimizer"] = opt.state_dict()
    ckpt["epoch"] = epoch
    ckpt["finetune"] = {"pretrained": os.path.abspath(args.pretrained),
                        "rollout": os.path.abspath(args.rollout),
                        "labels": os.path.abspath(args.labels) if args.labels else "",
                        "lr": args.lr, "epochs": args.epochs, "history": history}
    if ema is not None:
        ckpt["model_ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
        ckpt["ema_state"] = {k: v for k, v in ema.state_dict().items() if k != "shadow"}
    else:
        # An EMA inherited from the PARENT would be the pretrained weights wearing this
        # run's name, and `load_fm_context` prefers `model_ema` -- so the fine-tune would
        # silently evaluate as the model it started from.
        ckpt.pop("model_ema", None)
        ckpt.pop("ema_state", None)
    torch.save(ckpt, path)


if __name__ == "__main__":
    raise SystemExit(main())
