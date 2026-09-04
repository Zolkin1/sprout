# Copyright 2026 Zachary Olkin. All rights reserved.

"""Train the autoregressive token planner -- stage B.

Fits a prefix-LM over a frozen tokenizer's vocabulary to the SAME ``(depth, cmd, history,
future)`` tuples the flow-matching generator was trained on, with one preprocessing step: the
future is tokenized.

    PYTHONPATH=source/robot_rl:source/motion_gen python scripts/token_planner/train_planner.py \\
        --data <recorded>.h5,<augmented>.h5 \\
        --fm_config logs/flow_matching/<fm run>/params/fm_config.yaml \\
        --tokenizer logs/token_planner/<tok run>/tokenizer.pt --run_name ar_v1

.. admonition:: ``--fm_config`` is how the conditioning stays honest

    The planner's exteroception contract, its camera intrinsics, its normalization mode, its
    camera-latency distribution and its history noise all have to be the generator's, or the
    comparison measures a differently-wired depth encoder rather than a different output
    representation. Restating twenty fields on a command line is how they drift, so the
    generator's OWN saved config is read and used. Only the fields that cannot apply are
    overridden, and each override is printed.

.. admonition:: The state normalizer must be the tokenizer's, and it is CHECKED

    The planner's targets are token ids produced by the frozen tokenizer, and the tokenizer
    decodes into the space its own normalizer defined. If this build's state normalizer
    differs -- a different split, different declared scales, a different block selection --
    then the ids mean something else and every decoded plan is quietly in the wrong units.
    Nothing about the shapes would change, so the agreement is asserted at startup.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from motion_gen.features import _REPO_ROOT

sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))

from provenance import dump_params_yaml, git_snapshot  # noqa: E402

from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import build_train_val  # noqa: E402
from robot_rl.flow_matching.ema import WeightEMA  # noqa: E402
from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402
from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig  # noqa: E402
from robot_rl.token_planner.data import load_recordings  # noqa: E402
from robot_rl.token_planner.metrics import (  # noqa: E402
    ReconAccumulator,
    channel_group_labels,
)
from robot_rl.token_planner.metrics import format_report as format_recon  # noqa: E402
from robot_rl.token_planner.planner import TokenPlanner  # noqa: E402
from robot_rl.token_planner.planner_metrics import (  # noqa: E402
    TokenAccumulator,
    sample_diversity,
)
from robot_rl.token_planner.planner_metrics import format_report as format_tokens  # noqa: E402
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer  # noqa: E402

#: Fields of the generator's saved config that CANNOT carry over, with the reason. Printed at
#: startup so an override is never silent.
_OVERRIDES: dict[str, tuple[Any, str]] = {
    "pinned_axis_loss_weight": (1.0, "a flow-loss weight; this model's loss is categorical"),
    "pinned_axis_metrics": (False, "needs a clip-library join for a metric nothing here reads"),
    "transition_fraction": (None, "a flow-loss-era sampler weight; kept uniform here"),
}


def parse_args() -> tuple[PlannerConfig, argparse.Namespace]:
    """Build the CLI and return ``(config, runtime_args)``.

    Every flag shadowing a :class:`PlannerConfig` field renders its default from the
    dataclass, never a literal -- the FM trainer's pinned convention.

    Returns:
        The planner config (data-dependent fields unresolved) and the runtime-only args.
    """
    d = PlannerConfig()
    p = argparse.ArgumentParser(description="Train the autoregressive token planner.")
    p.add_argument("--data", required=True,
                   help="Comma-separated windowed .h5 recordings, the SAME set the "
                        "generator and the tokenizer used, in the SAME order (episode ids "
                        "are offset per file, so a swap renumbers every episode).")
    p.add_argument("--tokenizer", required=True,
                   help="A tokenizer.pt from scripts/token_planner/train_tokenizer.py. "
                        "Frozen; its config and normalizer become this checkpoint's.")
    p.add_argument("--fm_config", default="",
                   help="A flow-matching run's params/fm_config.yaml. Supplies the whole "
                        "conditioning contract, so the AR-vs-FM comparison differs in the "
                        "output representation and nothing else. Strongly recommended.")
    p.add_argument("--logdir", default="logs/token_planner")
    p.add_argument("--run_name", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--ckpt_every", type=int, default=10)
    p.add_argument("--diversity_contexts", type=int, default=256,
                   help="Validation contexts to measure sample diversity over.")
    p.add_argument("--diversity_samples", type=int, default=16,
                   help="Draws per context for the diversity metric.")
    p.add_argument("--load_rows", type=int, default=0,
                   help="SMOKE TEST ONLY: load only the first N windows of each recording. "
                        "This CHANGES THE SPLIT, so nothing measured is comparable -- but it "
                        "is the only way to exercise this whole path against a tokenizer "
                        "trained the same way, which is what verifies the state-normalizer "
                        "agreement rather than assuming it.")
    p.add_argument("--wandb", action="store_true", default=True,
                   help="Log to Weights & Biases (default; use --no_wandb to disable).")
    p.add_argument("--no_wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb_project", type=str, default="robot_rl_token_planner")

    # -- data / split (must match the tokenizer's, and are checked against it) --
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--split_seed", type=int, default=0)

    # -- trunk --
    p.add_argument("--d_model", type=int, default=d.d_model)
    p.add_argument("--nhead", type=int, default=d.nhead)
    p.add_argument("--num_layers", type=int, default=d.num_layers)
    p.add_argument("--dim_feedforward", type=int, default=d.dim_feedforward)
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--null_cond_prob", type=float, default=d.null_cond_prob)

    # -- loss / training --
    p.add_argument("--label_smoothing", type=float, default=d.label_smoothing)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--lr_min", type=float, default=d.lr_min)
    p.add_argument("--weight_decay", type=float, default=d.weight_decay)
    p.add_argument("--warmup_steps", type=int, default=d.warmup_steps)
    p.add_argument("--batch_size", type=int, default=d.batch_size)
    p.add_argument("--num_epochs", type=int, default=d.num_epochs)
    p.add_argument("--grad_clip", type=float, default=d.grad_clip)
    p.add_argument("--ema_decay", type=float, default=d.ema_decay)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=d.bf16)

    args = p.parse_args()
    cfg = PlannerConfig(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        null_cond_prob=args.null_cond_prob, label_smoothing=args.label_smoothing,
        lr=args.lr, lr_min=args.lr_min, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, batch_size=args.batch_size,
        num_epochs=args.num_epochs, grad_clip=args.grad_clip, ema_decay=args.ema_decay,
        seed=args.seed, bf16=args.bf16, load_rows=args.load_rows,
    )
    return cfg, args


class TupleDataset(Dataset):
    """``(history, extero, cmd_tokens, tokens, state)`` per window.

    Wraps a built :class:`~robot_rl.flow_matching.dataset.FutureStateDataset` rather than
    copying out of it, so the history noise and the per-item exteroception normalization the
    FM applies still happen exactly where and when the FM applies them.

    Args:
        base: The built FM dataset.
        tokens: ``(Nw, N)`` token ids for each window's future.
        family: ``(Nw,)`` terrain-family codes.
    """

    def __init__(self, base, tokens: torch.Tensor, family: torch.Tensor):
        if len(base) != tokens.shape[0]:
            raise ValueError(
                f"{len(base)} window(s) but {tokens.shape[0]} token row(s).")
        self.base = base
        self.tokens = tokens
        self.family = family

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return one training tuple."""
        item = self.base[i]
        out = {
            "history": item["history"],
            "extero": item["extero"],
            "tokens": self.tokens[i],
            "state": item["state"],
            "family": self.family[i],
        }
        if "cmd_tokens" in item:
            out["cmd_tokens"] = item["cmd_tokens"]
        return out


def load_fm_config(path: str) -> FlowMatchingConfig:
    """Read a generator's saved config and apply the documented overrides.

    Args:
        path: A run's ``params/fm_config.yaml``.

    Returns:
        The config, with :data:`_OVERRIDES` applied and each one printed.
    """
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = FlowMatchingConfig.from_dict(raw)
    cfg.fk_model_path = _relocate_fk_model(cfg.fk_model_path)
    for field_name, (value, why) in _OVERRIDES.items():
        have = getattr(cfg, field_name, None)
        if have != value:
            print(f"[ar] override {field_name}: {have!r} -> {value!r} ({why})")
            setattr(cfg, field_name, value)
    return cfg


def _relocate_fk_model(path: str) -> str:
    """Re-resolve a saved ``fk_model_path`` against THIS checkout.

    A generator's config records the absolute path its training box used, and that box is
    usually not this one -- the shipped run's is under ``/mnt/zolkin_new/...``. The file it
    names is the robot, so the fix is to find the same file here rather than to fail.

    **The BASENAME must match.** Silently substituting a different robot model is precisely
    the failure CLAUDE.md's ``g1_29j.urdf`` story records: nothing errored, every joint name
    resolved, and both depth cameras landed 10 mm out on every unprojected channel. So a
    relocation that would change which file is FK'd is refused instead.

    Args:
        path: The saved path.

    Returns:
        A path that exists, or ``path`` unchanged when it already does.

    Raises:
        SystemExit: If the local candidate is a different file by name, or none exists.
    """
    if not path or os.path.exists(path):
        return path
    from motion_gen.features import DEFAULT_MJCF

    local = str(DEFAULT_MJCF if os.path.isabs(str(DEFAULT_MJCF))
                else _REPO_ROOT / str(DEFAULT_MJCF))
    if os.path.basename(local) != os.path.basename(path):
        raise SystemExit(
            f"[ar] the config's fk_model_path is {path}, which does not exist here, and this "
            f"checkout's default robot is {local} -- a DIFFERENT file. Refusing to "
            "substitute it: FK'ing the wrong robot resolves every joint name and puts the "
            "cameras a centimetre out, which no shape check can see.")
    if not os.path.exists(local):
        raise SystemExit(f"[ar] neither {path} nor {local} exists; no robot model to FK.")
    print(f"[ar] fk_model_path relocated to this checkout: {path} -> {local}")
    return local


def scheduled_lr(cfg: PlannerConfig, epoch: int, step: int) -> float:
    """Per-step warmup into a per-epoch cosine decay to :attr:`PlannerConfig.lr_min`.

    Args:
        cfg: The config.
        epoch: Zero-based epoch.
        step: Zero-based global step.

    Returns:
        The learning rate for this step.
    """
    progress = min(max(float(epoch) / max(int(cfg.num_epochs), 1), 0.0), 1.0)
    lr = cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        lr *= float(step + 1) / float(cfg.warmup_steps)
    return lr


def assert_normalizers_agree(built: Normalizer, tokenizer_state: dict) -> None:
    """Require this build's state normalizer to be the tokenizer's.

    Args:
        built: The normalizer this build fitted.
        tokenizer_state: The tokenizer checkpoint's ``state_normalizer`` state dict.

    Raises:
        SystemExit: On any disagreement, naming the worst channel. Fatal because the
            tokenizer decodes into the space ITS normalizer defined: a different one here
            means every token id denotes a different trajectory, with no shape to notice it.
    """
    want = Normalizer.from_state_dict(tokenizer_state)
    for name in ("mean", "std", "standardize_mask"):
        a, b = getattr(built, name).float(), getattr(want, name).float()
        if a.shape != b.shape:
            raise SystemExit(
                f"[ar] state normalizer '{name}' is {tuple(a.shape)} here but "
                f"{tuple(b.shape)} in the tokenizer; the node layouts differ."
            )
        if not torch.allclose(a, b, atol=1e-6):
            i = int((a - b).abs().argmax())
            raise SystemExit(
                f"[ar] the state normalizer does not match the tokenizer's: '{name}' differs "
                f"most at channel {i} ({float(a[i]):.6g} here vs {float(b[i]):.6g}).\n"
                "      The tokenizer decodes into the space its own normalizer defined, so "
                "this build's token ids would denote different trajectories.\n"
                "      Check --split_seed, --val_fraction, --data and its ORDER, and the "
                "declared state scales in --fm_config against the tokenizer's run."
            )
    print("[ar] state normalizer matches the tokenizer's on all "
          f"{built.std.numel()} channels.")


@torch.no_grad()
def tokenize_split(tokenizer: TrajectoryTokenizer, state: torch.Tensor,
                   device: torch.device, batch: int = 4096) -> torch.Tensor:
    """Token ids for every window of a split.

    Args:
        tokenizer: The frozen tokenizer.
        state: ``(Nw, H, D)`` normalized futures.
        device: Where to run.
        batch: Windows per pass.

    Returns:
        ``(Nw, N)`` int64 ids, on the CPU.
    """
    out = []
    for i in range(0, state.shape[0], batch):
        out.append(tokenizer.encode_to_ids(state[i:i + batch].to(device)).cpu())
    return torch.cat(out, dim=0) if out else torch.zeros(0, tokenizer.num_tokens,
                                                         dtype=torch.long)


def _cond(batch: dict, device: torch.device, null_prob: float = 0.0):
    """Move one batch's conditioning to ``device`` and draw the null-history mask.

    Args:
        batch: A collated :class:`TupleDataset` item.
        device: Target device.
        null_prob: Probability of replacing a sample's history with the learned null row.
            The generator's ``null_cond_prob``, kept because it covers the same real
            situation: the first ``N`` control steps after a reset.

    Returns:
        ``(history, extero, cmd_tokens, history_null_mask)``.
    """
    hist = batch["history"].to(device, non_blocking=True)
    extero = batch["extero"].to(device, non_blocking=True)
    cmd = (batch["cmd_tokens"].to(device, non_blocking=True)
           if "cmd_tokens" in batch else None)
    null = (torch.rand(hist.shape[0], device=device) < null_prob
            if null_prob > 0.0 else None)
    return hist, extero, cmd, null


@torch.no_grad()
def evaluate(model: TokenPlanner, loader: DataLoader, device: torch.device,
             cfg: PlannerConfig, recon_factory) -> tuple[dict[str, Any], dict[str, float]]:
    """Teacher-forced token metrics plus greedy-decode trajectory error.

    Both in one pass over the split, because the second needs the same batches and a second
    pass over a depth dataset is minutes.

    Args:
        model: The planner, in the weights to be measured.
        loader: A validation loader.
        device: Where to run.
        cfg: The config.
        recon_factory: Zero-argument factory for a
            :class:`~robot_rl.token_planner.metrics.ReconAccumulator`.

    Returns:
        ``(report, scalars)``.
    """
    model.eval()
    tok_acc = TokenAccumulator(num_tokens=model.num_tokens, vocab_size=model.vocab_size)
    recon = recon_factory()
    for batch in loader:
        hist, extero, cmd, _ = _cond(batch, device)
        tokens = batch["tokens"].to(device, non_blocking=True)
        tok_acc.update(model(tokens, hist, extero, cmd), tokens)
        # Reuse the prefix `forward` just built: it is a function of the conditioning alone,
        # so re-encoding it here was 18% of this loop's cost for an identical answer.
        greedy, _, _ = model.sample(hist, extero, cmd, greedy=True,
                                    prefix_cache=model._last_prefix_cache)
        recon.update(batch["state"].to(device, non_blocking=True),
                     model.decode(greedy), greedy, batch["family"])
    trep, rrep = tok_acc.report(), recon.report()
    scalars = {
        "ce": trep["ce"], "top1": trep["top1"],
        "entropy_nats": trep["entropy_nats"],
        "entropy_frac_of_uniform": trep["entropy_frac_of_uniform"],
        "argmax_utilisation": trep["argmax_utilisation"],
        "ori_geodesic_rad": rrep["ori_geodesic_rad"],
    }
    for g, v in rrep["rmse"].items():
        scalars[f"rmse/{g}"] = v
    for g, v in rrep["node_rmse_p99"].items():
        scalars[f"p99/{g}"] = v
    scalars["rmse_mean"] = float(np.mean(list(rrep["rmse"].values())))
    return {"tokens": trep, "trajectory": rrep}, scalars


def save_checkpoint(path: str, model: TokenPlanner, opt, cfg: PlannerConfig,
                    extras: dict[str, Any], epoch: int, scalars: dict[str, float],
                    report: dict[str, Any], ema: WeightEMA | None, step: int = 0) -> None:
    """Write a self-describing planner checkpoint.

    Everything needed to interpret a token id travels here: the tokenizer's config AND its
    weights (inside ``model``, which owns it), the state normalizer, the block layout and the
    split. A checkpoint that only NAMED a tokenizer file would stop being interpretable the
    moment that file moved.

    Args:
        path: Destination.
        model: The planner.
        opt: The optimizer, for a resume.
        cfg: The resolved config.
        extras: Normalizers, layout, provenance and split.
        epoch: Zero-based epoch just finished.
        scalars: Flat validation scalars.
        report: The full validation report.
        ema: The weight EMA, or ``None``.
        step: The global optimizer step; ``scheduled_lr`` is a pure function of
            ``(epoch, step)``, so a resume needs it.
    """
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg.to_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": dict(scalars),
        "report": report,
        **extras,
    }
    if ema is not None:
        payload["model_ema"] = ema.shadow
        payload["ema_state"] = {k: v for k, v in ema.state_dict().items() if k != "shadow"}
    torch.save(payload, path)


def main() -> None:
    """Train the planner and write ``planner.pt`` plus an offline report."""
    cfg, args = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(args.device)

    # -- the frozen tokenizer --
    tok_ckpt = torch.load(args.tokenizer, map_location="cpu", weights_only=False)
    tok_cfg = TokenizerConfig.from_dict(tok_ckpt["config"])
    tokenizer = TrajectoryTokenizer(tok_cfg)
    tokenizer.load_state_dict(tok_ckpt.get("model_ema") or tok_ckpt["model"])
    tokenizer = tokenizer.to(device).eval().requires_grad_(False)
    print(f"[ar] tokenizer {args.tokenizer} (epoch {tok_ckpt.get('epoch')}): "
          f"H={tok_cfg.chunk_len} -> N={tok_cfg.num_tokens} tokens, V={tok_cfg.vocab_size}, "
          f"{'EMA' if tok_ckpt.get('model_ema') else 'raw'} weights")

    # -- the dataset: the generator's, with the generator's own config --
    paths = [p for p in args.data.split(",") if p.strip()]
    fm = (load_fm_config(args.fm_config) if args.fm_config
          else FlowMatchingConfig(future_len=tok_cfg.chunk_len))
    if not args.fm_config:
        print("[ar] WARNING: no --fm_config. The conditioning contract is then this "
              "script's defaults rather than the generator's, so an AR-vs-FM difference is "
              "not attributable to the output representation.")
    fm.future_len = tok_cfg.chunk_len
    fm.reference_blocks = list(tok_cfg.reference_blocks)
    fm.seed, fm.val_fraction, fm.stride = args.split_seed, args.val_fraction, args.stride
    # Drop the exteroception array this run does not use: a recording stores BOTH, and the
    # unused one is 6.2 GB of RAM per copy at the shipped window count. Same reason
    # scripts/flow_matching/train.py subsets its load keys.
    from robot_rl.token_planner.data import _EXTERO_BY_COST, _recorded_data_io

    keys = [k for k in _recorded_data_io().WINDOW_ARRAY_KEYS
            if k not in set(_EXTERO_BY_COST) - {fm.extero_source}]
    if args.load_rows > 0:
        print(f"[ar] *** --load_rows {args.load_rows}: SMOKE TEST. The episode split is "
              "computed over a truncated recording, so nothing measured here is comparable "
              "with a full run. ***")
        _patch_row_limit(args.load_rows)
    data, meta = load_recordings(paths, keys=keys)
    train_ds, val_ds = build_train_val(data, meta, fm)
    assert_normalizers_agree(train_ds.state_normalizer, tok_ckpt["state_normalizer"])

    fam_names = [str(s) for s in (getattr(meta, "family_names", None) or ["unknown"])]
    fam_all = data.get("clip_family")

    def _family(ds) -> torch.Tensor:
        """Family codes for a split, in the dataset's own (filtered) index space."""
        if ds is None:
            return torch.zeros(0, dtype=torch.long)
        sel = getattr(ds, "window_sel", None)
        if fam_all is None or sel is None:
            return torch.zeros(len(ds), dtype=torch.long)
        return torch.as_tensor(np.asarray(fam_all)[np.asarray(sel)], dtype=torch.long)

    # -- the one preprocessing step: tokenize the futures --
    t0 = time.time()
    train_tokens = tokenize_split(tokenizer, train_ds.state, device)
    val_tokens = (tokenize_split(tokenizer, val_ds.state, device)
                  if val_ds is not None else torch.zeros(0, tok_cfg.num_tokens,
                                                         dtype=torch.long))
    print(f"[ar] tokenized {train_tokens.shape[0]} train / {val_tokens.shape[0]} val "
          f"futures in {time.time() - t0:.1f}s; "
          f"{len(torch.unique(train_tokens))} distinct ids of {tok_cfg.vocab_size}")

    # -- the planner --
    cfg.tokenizer = tok_cfg.to_dict()
    cfg.tokenizer_path = os.path.abspath(args.tokenizer)
    cfg.history_dim = int(train_ds.state_dim)
    cfg.num_history_states = int(fm.num_history_states)
    cfg.future_len = int(tok_cfg.chunk_len)
    for name in ("extero_source", "extero_encoder", "extero_sensor_count",
                 "extero_sensor_dim", "extero_image_hw", "extero_cnn_channels",
                 "extero_cnn_grid", "extero_cnn_strides", "extero_coord_extent_m",
                 "extero_grid_ordering", "extero_fourier_bands", "extero_miss_sentinel",
                 "extero_camera_intrinsics", "cmd_dim", "cmd_stride"):
        setattr(cfg, name, getattr(fm, name))
    cfg.extero_channels = fm.resolve_extero_channels()
    # RESOLVED, not copied. It works today only because `build_train_val` mutates
    # `cfg.extero_encoder` in place before the copy above; reorder those two statements, or
    # build a planner without going through the dataset, and `ConditioningPrefix` raises
    # "unknown extero_encoder 'auto'".
    cfg.extero_encoder = fm.resolve_extero_encoder()
    model = TokenPlanner(cfg, tokenizer).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_pre = sum(p.numel() for p in model.prefix.parameters())
    print(
        f"[ar] planner: prefix {model.prefix_len} tokens "
        f"({cfg.num_history_states} history + {model.prefix.extero_seq_tokens} extero + "
        f"{int(model.prefix.cmd_steps.numel())} cmd) + {model.num_tokens} output slots "
        f"= sequence {model.prefix_len + model.num_tokens}\n"
        f"[ar] trunk d={cfg.d_model} x{cfg.num_layers} heads={cfg.nhead} "
        f"ff={cfg.dim_feedforward} dropout={cfg.dropout} pre-norm, head tied to the "
        f"{cfg.vocab_size}-way embedding\n"
        f"[ar] {n_par / 1e6:.2f}M trainable ({(n_par - n_pre) / 1e6:.2f}M excluding the "
        f"conditioning encoders); label smoothing {cfg.label_smoothing}, "
        f"null_cond_prob {cfg.null_cond_prob}"
    )

    run = (f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
           if args.run_name else time.strftime("%Y-%m-%d_%H-%M-%S"))
    out_dir = os.path.join(args.logdir, run)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[ar] out: {out_dir}")
    git_snapshot(out_dir)
    dump_params_yaml(os.path.join(out_dir, "params", "planner_config.yaml"), cfg.to_dict())
    dump_params_yaml(os.path.join(out_dir, "params", "fm_config.yaml"), fm.to_dict())
    dump_params_yaml(os.path.join(out_dir, "params", "data.yaml"), {
        "data": ",".join(paths),
        "tokenizer": os.path.abspath(args.tokenizer),
        "train_windows": int(train_tokens.shape[0]),
        "val_windows": int(val_tokens.shape[0]),
        "val_episode_ids": [int(v) for v in
                            (getattr(train_ds, "val_episode_ids", None) or [])],
        "family_names": fam_names,
    })

    train_loader = DataLoader(
        TupleDataset(train_ds, train_tokens, _family(train_ds)),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = (
        DataLoader(TupleDataset(val_ds, val_tokens, _family(val_ds)),
                   batch_size=cfg.batch_size, shuffle=False,
                   num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
        if val_ds is not None and len(val_ds) else None)

    group_names, labels = channel_group_labels(
        int(train_ds.num_joints), int(train_ds.num_extra_frames), tok_cfg.reference_blocks)
    norm = train_ds.state_normalizer
    std, mean = norm.std.clone(), norm.mean.clone()

    def recon_factory() -> ReconAccumulator:
        """A fresh trajectory-error accumulator wired to this run's layout."""
        return ReconAccumulator(
            group_names=group_names, labels=labels, std=std, mean=mean,
            chunk_len=tok_cfg.chunk_len, vocab_size=tok_cfg.vocab_size,
            num_families=len(fam_names), family_names=fam_names)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(float(cfg.betas[0]), float(cfg.betas[1])))
    # `freeze=("tokenizer.",)`: the tokenizer is a submodule and therefore 5.55M of the
    # model's 38M state-dict entries. Averaging it would drift the shadow away from the
    # weights it was frozen at, and every loader prefers `model_ema` -- so the tokenizer that
    # SHIPS would not be the frozen one. Harmless for decoding; not harmless for re-encoding,
    # since FSQ rounds to a lattice.
    ema = (WeightEMA(model, decay=cfg.ema_decay, freeze=("tokenizer.",))
           if cfg.ema_decay > 0.0 else None)
    writer = SummaryWriter(log_dir=out_dir)
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(
                project=args.wandb_project, name=run,
                config={**cfg.to_dict(), "data": ",".join(paths),
                        "tokenizer": os.path.abspath(args.tokenizer),
                        "train_windows": int(train_tokens.shape[0]),
                        "sequence_len": model.prefix_len + model.num_tokens,
                        "trainable_params": n_par})
        except Exception as exc:                                          # noqa: BLE001
            print(f"[WARN] wandb logging is OFF: {type(exc).__name__}: {exc}\n"
                  f"       Training continues; scalars still go to TensorBoard in {out_dir}.")
    use_bf16 = bool(args.bf16 and device.type == "cuda")
    extras = {
        "state_normalizer": norm.state_dict(),
        "extero_normalizer": train_ds.extero_normalizer.state_dict(),
        "cmd_normalizer": (train_ds.cmd_normalizer.state_dict()
                           if train_ds.cmd_normalizer is not None else None),
        "num_joints": int(train_ds.num_joints),
        "num_ee_frames": int(train_ds.num_extra_frames),
        "reference_blocks": list(train_ds.blocks),
        "family_names": fam_names,
        "fm_config": fm.to_dict(),
        "data": ",".join(paths),
        "tokenizer_path": os.path.abspath(args.tokenizer),
        "val_episode_ids": [int(v) for v in
                            (getattr(train_ds, "val_episode_ids", None) or [])],
    }

    best, step = math.inf, 0
    for epoch in range(cfg.num_epochs):
        model.train()
        t0 = time.time()
        run_loss, run_raw, run_n, clipped = 0.0, 0.0, 0, 0
        for batch in train_loader:
            hist, extero, cmd, null = _cond(batch, device, cfg.null_cond_prob)
            tokens = batch["tokens"].to(device, non_blocking=True)
            lr = scheduled_lr(cfg, epoch, step)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(tokens, hist, extero, cmd, null)
            flat, flat_t = logits.float().reshape(-1, model.vocab_size), tokens.reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                flat, flat_t, label_smoothing=cfg.label_smoothing)
            # The UNSMOOTHED value too, because the smoothed one is not comparable with the
            # validation CE -- `TokenAccumulator` is deliberately unsmoothed (a smoothed CE
            # depends on V, and the FSQ level table is an ablation axis). Smoothing adds a
            # floor that GROWS as the model sharpens: +0.20 nats for a weak model, +3.13 for
            # a sharp one at V=12800. Logging only the smoothed number makes validation
            # appear to beat training by a widening margin, all run long, on the one figure
            # anyone reads per epoch.
            with torch.no_grad():
                raw_ce = float(torch.nn.functional.cross_entropy(flat, flat_t))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.grad_clip))
            clipped += int(gn > cfg.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model)
            run_loss += float(loss.detach()) * tokens.shape[0]
            run_raw += raw_ce * tokens.shape[0]
            run_n += int(tokens.shape[0])
            if step % 50 == 0:
                writer.add_scalar("train/loss_step", float(loss.detach()), step)
                writer.add_scalar("train/lr_step", lr, step)
                writer.add_scalar("train/grad_norm_step", gn, step)
            step += 1

        scal_epoch = {
            "train/loss_epoch": run_loss / max(run_n, 1),
            "train/ce_unsmoothed": run_raw / max(run_n, 1),
            "train/grad_clipped_frac": clipped / max(len(train_loader), 1),
            "train/epoch_time_s": time.time() - t0,
            "train/lr": scheduled_lr(cfg, epoch, step),
        }
        for k, v in scal_epoch.items():
            writer.add_scalar(k, v, epoch)
        # Both, and labelled: `smoothed` is the objective, `ce` is what compares with val.
        line = (f"[ar] epoch {epoch:>4d}  train ce {run_raw / max(run_n, 1):.4f} "
                f"(smoothed {run_loss / max(run_n, 1):.4f})  {time.time() - t0:.1f}s")

        do_val = val_loader is not None and (
            epoch % max(args.val_every, 1) == 0 or epoch == cfg.num_epochs - 1)
        if do_val:
            rep, scal = evaluate(model, val_loader, device, cfg, recon_factory)
            for k, v in scal.items():
                writer.add_scalar(f"val/{k}", v, epoch)
            scal_epoch.update({f"val/{k}": v for k, v in scal.items()})
            sel, sel_rep, tag = scal["ce"], rep, "val"
            if ema is not None:
                with ema.swapped_in(model):
                    rep_e, scal_e = evaluate(model, val_loader, device, cfg, recon_factory)
                for k, v in scal_e.items():
                    writer.add_scalar(f"val_ema/{k}", v, epoch)
                scal_epoch.update({f"val_ema/{k}": v for k, v in scal_e.items()})
                sel, sel_rep, tag, scal = scal_e["ce"], rep_e, "val_ema", scal_e
            # Indexed with `.get`: a `reference_blocks` selection without one of these
            # groups would otherwise KeyError at the end of the first validation epoch,
            # after the whole build and tokenize pass.
            line += (f"  {tag} ce {scal['ce']:.4f}  top1 {100 * scal['top1']:.1f}%  "
                     f"H {scal['entropy_nats']:.2f}")
            # The group's own name, not a prefix of it: `base_pos` and `base_ori` both
            # start with "base", so truncating makes the two indistinguishable in the one
            # line anyone reads per epoch.
            for key, label, unit in (("base_pos", "pos", "m"), ("joint_pos", "jnt", "rad")):
                if f"rmse/{key}" in scal:
                    line += f"  {label} {scal[f'rmse/{key}']:.4f} {unit}"
            if sel < best:
                best = sel
                save_checkpoint(os.path.join(out_dir, "planner.pt"), model, opt, cfg,
                                extras, epoch, scal, sel_rep, ema, step)
                with open(os.path.join(out_dir, "offline_report.json"), "w") as fh:
                    json.dump({"epoch": epoch, "scalars": scal, "report": sel_rep}, fh,
                              indent=2)
                line += "  *"
        print(line, flush=True)
        if wandb_run is not None:
            wandb_run.log({**scal_epoch, "epoch": epoch}, step=epoch)
        if epoch % max(args.ckpt_every, 1) == 0 or epoch == cfg.num_epochs - 1:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{epoch}.pt"), model, opt, cfg,
                            extras, epoch, {}, {}, ema, step)

    # -- diversity, once, at the end: it costs `diversity_samples` full decodes per context --
    div = None
    if val_loader is not None and args.diversity_samples > 1:
        batch = next(iter(val_loader))
        k = min(int(args.diversity_contexts), batch["history"].shape[0])
        hist, extero, cmd, _ = _cond(
            {key: v[:k] for key, v in batch.items()}, device)
        # Measured on the EMA weights when there are any, matching what `best` selected on.
        with (ema.swapped_in(model) if ema is not None else contextlib.nullcontext()):
            div = sample_diversity(
                model, hist, extero, cmd, num_samples=int(args.diversity_samples),
                generator=torch.Generator(device=device.type).manual_seed(cfg.seed),
                std=std.to(device))
        for key, v in div.items():
            writer.add_scalar(f"val_ema/diversity_{key}", v, cfg.num_epochs - 1)
        if wandb_run is not None:
            wandb_run.log({f"val_ema/diversity_{key}": v for key, v in div.items()},
                          step=cfg.num_epochs - 1)

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    if os.path.exists(os.path.join(out_dir, "offline_report.json")):
        with open(os.path.join(out_dir, "offline_report.json")) as fh:
            saved = json.load(fh)
        text = (format_tokens(saved["report"]["tokens"], div) + "\n\n"
                + format_recon(saved["report"]["trajectory"]))
        with open(os.path.join(out_dir, "offline_report.txt"), "w") as fh:
            fh.write(text + "\n")
        print("\n" + text)
    print(f"[ar] done. best val ce {best:.6f} -> {out_dir}/planner.pt")


def _patch_row_limit(n: int) -> None:
    """Restrict every recording read to its first ``n`` windows (smoke tests only).

    Args:
        n: Row cap.
    """
    from robot_rl.token_planner import data as tpd

    rdio = tpd._recorded_data_io()
    original = rdio.load_recorded_windows

    def limited(path, keys=None, rows=None):
        """Read only the first ``n`` rows, whatever was asked for."""
        return original(path, keys=keys, rows=np.arange(n))

    rdio.load_recorded_windows = limited


if __name__ == "__main__":
    main()
