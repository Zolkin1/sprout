# Copyright 2026 Zachary Olkin. All rights reserved.

"""Two drift canaries for a fine-tune, both scored against a FROZEN cached batch.

A filtered-BC fine-tune trains on a narrow slice -- one generator's own selected chunks, from
a handful of tiles at a handful of speeds.  The fine-tune's own held-out split
(``finetune_generator.py --val_frac``) is drawn from that same slice, so it detects
MEMORISATION of the training episodes and nothing else: a model that got very good at
reproducing selected chunks on those tiles while forgetting how to generate anything else
shows a monotonically FALLING held-out loss the whole way, because its reference point moved
with it.

These two supply reference points that do not move:

* **``pretrain_val_flow_mse``** -- the pretraining objective on a fixed batch of windows from
  the PRETRAINING recordings' own validation split.  Rising means the model is leaving the
  distribution it was fitted on.
* **``drift_l2``** -- mean L2 between this model's sampled plan and the PRETRAINED model's
  sampled plan, on fixed conditioning with fixed sampling noise.  Rising means the served
  plan has moved, measured in the space the tracker actually consumes rather than in loss.

**Freeze the randomness or the curves are a random walk.**  The flow loss is an expectation
over the flow time ``tau`` and the noise draw ``x0``; resampling those each evaluation buries
a 1% weight change under Monte-Carlo variance.  The cache therefore stores ``x0`` and ``tau``
alongside the windows, so the reported number is a deterministic function of the weights.  The
same argument applies ACROSS runs: an LR sweep whose arms score different batches produces
arms that cannot be compared, which is why this is a cache on disk and not a resample.

**Matched to ``train.evaluate`` deliberately.**  That function's conditioned ``flow_mse``
passes no null mask, so this one does not either -- which makes the step-0 value directly
comparable to the pretrain run's own val curve, and makes "does step 0 land on the pretrained
model's known val loss" a correctness check on this whole apparatus.

**What a rise does NOT mean.**  It measures drift in the training objective, not in
closed-loop behaviour, and the BC data covers four families where the pretrain val covers all
of them -- so some rise is the model correctly specialising toward the deploy families rather
than degrading.  These narrow where to spend MuJoCo eval time (the window where the fine-tune's
held-out loss has bottomed and these have not yet lifted); they do not replace it.

Build once, then score every checkpoint::

    python ft_canary.py --build \\
        --pretrained logs/flow_matching/<run>/best.pt \\
        --out logs/value/canary_v1.pt --n_windows 2048 --n_drift 256
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if os.path.join(_THIS_DIR, os.pardir, "rsl_rl") not in sys.path:
    sys.path.insert(0, os.path.join(_THIS_DIR, os.pardir, "rsl_rl"))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from recorded_data_io import load_recorded_windows  # noqa: E402
from robot_rl.flow_matching import build_train_val  # noqa: E402
from robot_rl.flow_matching.flow import sample_trajectory  # noqa: E402

#: Cache format version, stamped into the file and checked at load. A cache whose contents
#: mean something different must not be silently averaged into an existing curve.
CACHE_VERSION = 1


def _load_recordings(paths: list[str]) -> tuple[dict, object]:
    """Load and concatenate windowed recordings the way ``train.py`` does.

    Episode ids are offset per file, which is what keeps ``build_train_val``'s by-episode
    split from leaking between them -- and therefore what makes the val split reproduced here
    the SAME split the pretraining used.

    Args:
        paths: Recording ``.h5`` paths, in the checkpoint's own order.

    Returns:
        ``(data, meta)`` for the concatenation; ``meta`` is the first file's.
    """
    data, meta = load_recorded_windows(paths[0])
    ep_off = int(np.asarray(data["episode_id"]).max()) + 1
    for extra_path in paths[1:]:
        extra, _ = load_recorded_windows(extra_path)
        for k in data:
            if k == "episode_id":
                data[k] = np.concatenate([data[k], np.asarray(extra[k]) + ep_off])
            else:
                data[k] = np.concatenate([data[k], np.asarray(extra[k])])
        ep_off = int(np.asarray(data["episode_id"]).max()) + 1
    return data, meta


def build_cache(pretrained: str, out: str, n_windows: int, n_drift: int, seed: int,
                device: str, data_paths: list[str] | None = None) -> dict:
    """Build the frozen scoring batch and the pretrained model's reference plans.

    Args:
        pretrained: The generator ``.pt`` the fine-tune starts from.
        out: Cache ``.pt`` to write.
        n_windows: Validation windows for the flow-loss canary.
        n_drift: Windows (a subset of the above) for the sample-space canary.
        seed: RNG seed for the window choice, ``x0``, ``tau`` and the sampling noise.
        device: Device to build the reference plans on.
        data_paths: Override the recordings; default is the checkpoint's own ``data_path``.

    Returns:
        The cache dict (also written to ``out``).

    Raises:
        SystemExit: If the checkpoint names no recordings, or the val split is empty.
    """
    from finetune_generator import resolved_config  # noqa: PLC0415
    from robot_rl.flow_matching import online  # noqa: PLC0415

    blob = torch.load(pretrained, map_location="cpu", weights_only=False)
    paths = data_paths or [p for p in str(blob.get("data_path", "")).split(",") if p]
    if not paths:
        raise SystemExit(f"{pretrained} records no data_path; pass --data explicitly.")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit("pretraining recordings not found (this must run where they live):\n"
                         + "\n".join(f"  {p}" for p in missing))

    cfg = resolved_config(pretrained)
    print(f"[canary] recordings: {len(paths)}")
    for p in paths:
        print(f"    {p}")
    data, meta = _load_recordings(paths)
    # The SAME call the pretraining made, with the SAME cfg -- so `val_ds` is the split the
    # pretrained model was validated on, not a fresh one that merely resembles it.
    _train_ds, val_ds = build_train_val(data, meta, cfg)
    if val_ds is None or len(val_ds) == 0:
        raise SystemExit("the pretraining val split is empty; nothing to score against.")
    # The history noise is a TRAINING augmentation (it models exposure bias). A canary wants
    # the clean val loss, and wants it deterministic.
    val_ds.add_noise = False
    print(f"[canary] pretraining val split: {len(val_ds):,} windows")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(val_ds), size=min(n_windows, len(val_ds)), replace=False)
    pick = np.sort(pick)
    items = [val_ds[int(i)] for i in pick]
    batch = {k: torch.stack([it[k] for it in items])
             for k in ("state", "history", "extero")}
    if "cmd_tokens" in items[0]:
        batch["cmd_tokens"] = torch.stack([it["cmd_tokens"] for it in items])

    g = torch.Generator().manual_seed(seed)
    x1 = batch["state"]
    batch["x0"] = torch.randn(x1.shape, generator=g)
    batch["tau"] = torch.rand(x1.shape[0], generator=g)

    ctx = online.load_fm_context(pretrained, device=device)
    n_d = min(n_drift, x1.shape[0])
    drift_noise = torch.randn(n_d, ctx.F, ctx.D, generator=g)
    cond = {k: batch[k][:n_d].to(device) for k in batch
            if k in ("history", "extero", "cmd_tokens")}
    with torch.no_grad():
        ref = sample_trajectory(ctx.model, cond["history"], cond["extero"],
                                cond.get("cmd_tokens"), ctx.F, ctx.D, ctx.num_steps,
                                noise=drift_noise.to(device))
    cache = {
        "version": CACHE_VERSION,
        "pretrained": os.path.abspath(pretrained),
        "data_paths": paths,
        "seed": seed,
        "rows": pick,
        "n_val_total": len(val_ds),
        "batch": {k: v.cpu() for k, v in batch.items()},
        "drift_noise": drift_noise.cpu(),
        "drift_ref": ref.cpu(),
        "n_drift": n_d,
        "F": ctx.F, "D": ctx.D, "num_steps": ctx.num_steps,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    torch.save(cache, out)

    can = Canary(out, device=device)
    base_mse = can.flow_mse(ctx.model)
    base_drift = can.drift(ctx.model)
    print(f"[canary] wrote {out}")
    print(f"[canary] {x1.shape[0]:,} windows / {n_d} drift conditionings")
    print(f"[canary] BASELINE on the pretrained model: pretrain_val_flow_mse={base_mse:.6f}  "
          f"drift_l2={base_drift['drift_l2']:.3e} (must be ~0: same weights, same noise)")
    if base_drift["drift_l2"] > 1e-4:
        print("[canary] WARNING: the pretrained model does not reproduce its own cached "
              "plans. The reference is not a fixed point, so drift_l2 is not measuring "
              "drift. Check that sampling is deterministic given `noise`.")
    return cache


class Canary:
    """Score a model against a frozen cached batch.

    Args:
        path: A cache written by :func:`build_cache`.
        device: Device to score on.
        batch_size: Forward-pass chunk for the flow loss.

    Raises:
        SystemExit: On a cache written by a different version of this file.
    """

    def __init__(self, path: str, device: str = "cuda", batch_size: int = 256) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if int(blob.get("version", -1)) != CACHE_VERSION:
            raise SystemExit(
                f"{path} is cache version {blob.get('version')}, this is {CACHE_VERSION}. "
                "Rebuild it -- a curve must not mix two definitions of its own metric.")
        self.path = path
        self.device = device
        self.batch_size = int(batch_size)
        self.batch = {k: v.to(device) for k, v in blob["batch"].items()}
        self.drift_noise = blob["drift_noise"].to(device)
        self.drift_ref = blob["drift_ref"].to(device)
        self.n_drift = int(blob["n_drift"])
        self.F, self.D, self.num_steps = int(blob["F"]), int(blob["D"]), int(blob["num_steps"])
        self.pretrained = blob["pretrained"]

    @torch.no_grad()
    def flow_mse(self, model: torch.nn.Module) -> float:
        """The pretraining flow loss on the cached batch, at the cached ``(x0, tau)``.

        Reproduces ``flow_matching_loss``'s ``flow_mse`` term exactly -- the unweighted mean
        square of ``v - (x1 - x0)`` over every element -- but at a FIXED draw instead of a
        fresh one, and with no null mask, matching ``train.evaluate``'s conditioned metric.

        Args:
            model: The model to score (already on ``self.device``).

        Returns:
            Mean squared flow error.
        """
        was_training = model.training
        model.eval()
        b = self.batch
        tot, n = 0.0, 0
        for i in range(0, b["state"].shape[0], self.batch_size):
            sl = slice(i, i + self.batch_size)
            x1, x0 = b["state"][sl], b["x0"][sl]
            tau = b["tau"][sl]
            t = tau.view(-1, *([1] * (x1.dim() - 1)))
            x_tau = (1.0 - t) * x0 + t * x1
            v = model(x_tau, tau, b["history"][sl], b["extero"][sl],
                      b["cmd_tokens"][sl] if "cmd_tokens" in b else None)
            sq = (v - (x1 - x0)) ** 2
            tot += float(sq.mean()) * x1.shape[0]
            n += x1.shape[0]
        if was_training:
            model.train()
        return tot / max(n, 1)

    @torch.no_grad()
    def drift(self, model: torch.nn.Module) -> dict[str, float]:
        """Sample-space distance from the PRETRAINED model's plans on fixed conditioning.

        Uses the cached sampling noise, so the pretrained model reproduces its own reference
        exactly and every number is attributable to the weights.  Reported in NORMALIZED
        window units (the space the model emits), not metres: the affine is per channel, so
        one scalar in physical units would not exist.

        Args:
            model: The model to score.

        Returns:
            ``{drift_l2, drift_l2_max}`` -- mean and worst per-window RMS deviation.
        """
        was_training = model.training
        model.eval()
        b = self.batch
        n = self.n_drift
        pred = sample_trajectory(
            model, b["history"][:n], b["extero"][:n],
            b["cmd_tokens"][:n] if "cmd_tokens" in b else None,
            self.F, self.D, self.num_steps, noise=self.drift_noise)
        per_window = ((pred - self.drift_ref) ** 2).mean(dim=(1, 2)).sqrt()
        if was_training:
            model.train()
        return {"drift_l2": float(per_window.mean()),
                "drift_l2_max": float(per_window.max())}

    def score(self, model: torch.nn.Module) -> dict[str, float]:
        """Both canaries in one call.

        Args:
            model: The model to score.

        Returns:
            ``{pretrain_val_flow_mse, drift_l2, drift_l2_max}``.
        """
        return {"pretrain_val_flow_mse": self.flow_mse(model), **self.drift(model)}


def main() -> int:
    """Entry point for ``--build``."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true", required=True,
                   help="Build the cache. Scoring happens inside finetune_generator.py.")
    p.add_argument("--pairs", default="",
                   help="Build the SELECTED-vs-REJECTED pair cache from this rollout instead "
                        "of the drift caches. Needs a merge_selected.py file WITHOUT "
                        "--successful_only: a successful-only file has no `failed` group, and "
                        "that group is the only one the model never fit.")
    p.add_argument("--n_pairs", type=int, default=512, help="Triples per group (--pairs).")
    p.add_argument("--val_frac", type=float, default=0.05,
                   help="Must MATCH the fine-tune's --val_frac, or 'heldout' is not the split "
                        "that was held out.")
    p.add_argument("--split_seed", type=int, default=0,
                   help="Must MATCH the fine-tune's --seed, for the same reason.")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="",
                   help="Comma-separated recordings, overriding the checkpoint's own "
                        "data_path. Only for a checkpoint whose recordings have moved -- a "
                        "DIFFERENT recording makes the baseline incomparable to the pretrain "
                        "run's val curve, which is the check that this cache is sound.")
    p.add_argument("--n_windows", type=int, default=2048)
    p.add_argument("--n_drift", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.pairs:
        build_pair_cache(args.pairs, args.pretrained, args.out, args.n_pairs, args.seed,
                         args.val_frac, args.split_seed, args.device)
    else:
        build_cache(args.pretrained, args.out, args.n_windows, args.n_drift, args.seed,
                    args.device, [x for x in args.data.split(",") if x] or None)
    return 0



# ---------------------------------------------------------------------------------------
# The SELECTED-vs-REJECTED pair gap
# ---------------------------------------------------------------------------------------

#: Flow times the pair loss is averaged over.  A GRID rather than random draws: the flow loss
#: varies strongly with tau, so averaging a fixed grid cuts variance far more per unit compute
#: than more pairs would, and it removes the draw from the checkpoint-to-checkpoint comparison
#: entirely.  Endpoints excluded -- at tau ~ 0 the target is pure noise and at tau ~ 1 it is
#: pure data, and neither end discriminates between two chunks from the same conditional.
PAIR_TAU_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: Row groups the gap is reported over, and what each one answers.
#:   train    successful episode, TRAIN split   -- rows the model actually fit
#:   heldout  successful episode, VAL split     -- did the ranking generalize across episodes
#:   failed   failed episode, never trained on  -- an untouched control
PAIR_GROUPS = ("train", "heldout", "failed")


def build_pair_cache(rollout: str, pretrained: str, out: str, n_pairs: int, seed: int,
                     val_frac: float, split_seed: int, device: str) -> dict:
    """Freeze a set of (conditioning, selected, rejected) triples for the pair-gap metric.

    Args:
        rollout: A merge carrying candidate columns AND both outcomes -- i.e.
            ``merge_selected.py`` WITHOUT ``--successful_only``.  A successful-only file has
            no ``failed`` group and no rejected chunks from failed episodes.
        pretrained: The generator the fine-tune starts from (supplies the normalizers).
        out: Cache ``.pt``.
        n_pairs: Triples to keep PER GROUP.
        seed: RNG for the triple choice and the frozen noise.
        val_frac: Must match the fine-tune's ``--val_frac``.
        split_seed: Must match the fine-tune's ``--seed``.
        device: Device for the token building.

    Returns:
        The cache dict (also written).

    Raises:
        SystemExit: If the rollout carries no candidate columns.
    """
    import h5py  # noqa: PLC0415

    from finetune_generator import _split_rows  # noqa: PLC0415
    from robot_rl.flow_matching import online  # noqa: PLC0415
    from robot_rl.flow_matching.rollout_dataset import RolloutWindowDataset  # noqa: PLC0415

    with h5py.File(rollout, "r") as f:
        if not int(f.attrs.get("has_candidates", 0)):
            raise SystemExit(
                f"{rollout} carries no candidate columns. Re-collect with "
                "`live_select.py --record_candidates` -- the rejected chunks are the whole "
                "metric and cannot be reconstructed after the fact (they were never on disk).")
        n_win = int(f.attrs["num_windows_written"])
        uid = f["episode_uid"][:n_win]
        scores = f["cand_scores"][:n_win]
        verdict = f["cand_verdict"][:n_win]
        sel_idx = f["sel_index"][:n_win].astype(np.int64)
        success = f["episode_success"][:n_win].astype(bool)

    # The SAME split the trainer makes, over the SAME rows it trains on (successful only), so
    # `train` really means "fit" and `heldout` really means "held out" rather than resembling
    # them. `failed` rows are outside that universe entirely.
    ok_rows = np.flatnonzero(success)
    tr_rows, va_rows = _split_rows(uid, ok_rows, val_frac, split_seed)
    groups = {"train": tr_rows, "heldout": va_rows, "failed": np.flatnonzero(~success)}

    rng = np.random.default_rng(seed)
    ctx = online.load_fm_context(pretrained, device=device)
    keep: dict[str, np.ndarray] = {}
    for name, rows in groups.items():
        if rows.size == 0:
            print(f"[pairs] group '{name}' is EMPTY -- it will report nothing.")
            keep[name] = rows
            continue
        keep[name] = np.sort(rng.choice(rows, size=min(n_pairs, rows.size), replace=False))
        print(f"[pairs] group '{name}': {rows.size:,} rows -> {keep[name].size} sampled")

    rows_all = np.concatenate([keep[g] for g in PAIR_GROUPS if keep[g].size])
    group_of = np.concatenate([np.full(keep[g].size, i, dtype=np.int64)
                               for i, g in enumerate(PAIR_GROUPS) if keep[g].size])
    ds = RolloutWindowDataset(rollout, rows_all, null_prob=0.0, ctx=ctx, label=None, seed=seed)
    items = [ds[i] for i in range(len(ds))]
    cond = {k: torch.stack([it[k] for it in items])
            for k in ("history", "extero", "history_null") if k in items[0]}
    if "cmd_tokens" in items[0]:
        cond["cmd_tokens"] = torch.stack([it["cmd_tokens"] for it in items])
    x1_sel = torch.stack([it["state"] for it in items])          # already normalized

    with h5py.File(rollout, "r") as f:
        cand = f["cand_plans"][:][rows_all]                       # (R, K, F, D)
    k = cand.shape[1]
    # Same normalizer the dataset used for `state`; a different one here would make the two
    # members of a pair incomparable while both looked reasonable.
    # `ctx.state_norm` lives on the context's device; the raw plans come off disk on the CPU.
    _dev = ctx.state_norm.mean.device
    x1_cand = ctx.state_norm.normalize(
        torch.from_numpy(cand.reshape(-1, *cand.shape[2:])).float().to(_dev)
    ).reshape(cand.shape).cpu()

    # Expand rows into PAIRS: one per rejected candidate.
    sel_r, sco_r, ver_r = sel_idx[rows_all], scores[rows_all], verdict[rows_all]
    pr_row, pr_rej = [], []
    for i in range(len(rows_all)):
        for j in range(k):
            if j != int(sel_r[i]):
                pr_row.append(i)
                pr_rej.append(j)
    pr_row = np.asarray(pr_row)
    pr_rej = np.asarray(pr_rej)
    margin = (sco_r[pr_row, pr_rej] - sco_r[pr_row, sel_r[pr_row]]).astype(np.float32)
    # A candidate that fell or arrived was scored by FIAT (1.0 / -1.0), not by V. Ranking those
    # by margin would make "worst rejected" mean "fell", a different and far easier contrast.
    sentinel = ((ver_r[pr_row, pr_rej] != 0) | (ver_r[pr_row, sel_r[pr_row]] != 0))

    tau = torch.tensor(PAIR_TAU_GRID, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    # One eps per (pair, tau), SHARED between the two members of that pair: the only thing that
    # then differs between them is the target chunk itself.
    eps = torch.randn(len(pr_row), len(tau), *x1_sel.shape[1:], generator=g)

    cache = {
        "version": CACHE_VERSION, "kind": "pair_gap",
        "rollout": os.path.abspath(rollout), "pretrained": os.path.abspath(pretrained),
        "cond": {k_: v.cpu() for k_, v in cond.items()},
        "x1_sel": x1_sel.cpu(), "x1_cand": x1_cand.cpu(),
        "pair_row": pr_row, "pair_rej": pr_rej, "pair_group": group_of[pr_row],
        "margin": margin, "sentinel": sentinel, "tau": tau, "eps": eps,
        "val_frac": val_frac, "split_seed": split_seed, "seed": seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    torch.save(cache, out)
    print(f"[pairs] wrote {out}: {len(pr_row)} pairs over {len(rows_all)} re-plans, "
          f"{len(tau)} tau nodes ({sentinel.sum()} pairs touch a terminal sentinel)")

    pg = PairGapCanary(out, device=device)
    base = pg.score(ctx.model)
    print("[pairs] BASELINE on the pretrained model (expect gap ~0 and win_rate ~0.50 -- the "
          "candidates are exchangeable draws from ONE conditional, so anything else is news):")
    for kk in sorted(base):
        print(f"    {kk} = {base[kk]:+.5f}")
    return cache


class PairGapCanary:
    """Paired selected-vs-rejected flow loss, on a frozen set of triples.

    Answers the question no loss on the selected chunks alone can: is the model pulling AWAY
    from the chunks the critic rejected, or improving on all of its own samples together?

    **Both absolute levels are reported, not only the gap.**  Self-distillation on selected
    chunks will very likely pull the rejected loss down too -- they are samples from the same
    conditional at the same conditioning.  A gap alone cannot separate "rejected rose"
    (collapse onto the selected slice) from "both fell, selected faster" (the healthy case),
    and those call for opposite decisions.

    **It measures agreement with the SELECTOR, not with the truth.**  A model can learn
    perfectly to disprefer chunks the critic rejected in error; only the closed-loop eval
    catches that.

    Args:
        path: A cache from :func:`build_pair_cache`.
        device: Device to score on.
        batch_size: Chunk size for the forward passes.
    """

    def __init__(self, path: str, device: str = "cuda", batch_size: int = 256) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("kind") != "pair_gap":
            raise SystemExit(f"{path} is not a pair-gap cache (kind={blob.get('kind')!r}).")
        self.device, self.batch_size = device, int(batch_size)
        self.cond = {k: v.to(device) for k, v in blob["cond"].items()}
        self.x1_sel = blob["x1_sel"].to(device)
        self.x1_cand = blob["x1_cand"].to(device)
        self.tau = blob["tau"].to(device)
        self.eps = blob["eps"].to(device)
        self.pair_row = torch.from_numpy(blob["pair_row"]).long()
        self.pair_rej = torch.from_numpy(blob["pair_rej"]).long()
        self.group = torch.from_numpy(blob["pair_group"]).long()
        self.margin = torch.from_numpy(blob["margin"])
        self.sentinel = torch.from_numpy(blob["sentinel"])

    @torch.no_grad()
    def _losses(self, model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pair flow loss of the selected and the rejected member, averaged over tau."""
        n_p, n_t = self.pair_row.numel(), self.tau.numel()
        out = []
        for which in ("sel", "rej"):
            acc = torch.zeros(n_p, device=self.device)
            for ti in range(n_t):
                t_val = self.tau[ti]
                for i in range(0, n_p, self.batch_size):
                    sl = slice(i, i + self.batch_size)
                    rows = self.pair_row[sl].to(self.device)
                    x1 = (self.x1_sel[rows] if which == "sel"
                          else self.x1_cand[rows, self.pair_rej[sl].to(self.device)])
                    x0 = self.eps[sl, ti].to(self.device)
                    t = t_val.expand(x1.shape[0])
                    tv = t.view(-1, *([1] * (x1.dim() - 1)))
                    x_tau = (1.0 - tv) * x0 + tv * x1
                    cond = {k: v[rows] for k, v in self.cond.items()}
                    v = model(x_tau, t, cond["history"], cond["extero"],
                              cond.get("cmd_tokens"),
                              history_null_mask=cond.get("history_null"))
                    acc[sl] += ((v - (x1 - x0)) ** 2).mean(dim=(1, 2)).cpu().to(acc.device)
            out.append(acc / n_t)
        return out[0], out[1]

    def score(self, model: torch.nn.Module) -> dict[str, float]:
        """All strata in one pass.

        Args:
            model: The model to score.

        Returns:
            ``pair/<group>/{sel_loss,rej_loss,gap,win_rate,gap_worst,gap_runner_up}``.
        """
        was_training = model.training
        model.eval()
        sel, rej = self._losses(model)
        sel, rej = sel.cpu(), rej.cpu()
        gap = rej - sel
        res: dict[str, float] = {}
        for gi, name in enumerate(PAIR_GROUPS):
            m = self.group == gi
            if not bool(m.any()):
                continue
            res[f"pair/{name}/sel_loss"] = float(sel[m].mean())
            res[f"pair/{name}/rej_loss"] = float(rej[m].mean())
            res[f"pair/{name}/gap"] = float(gap[m].mean())
            res[f"pair/{name}/win_rate"] = float((gap[m] > 0).float().mean())
            # Margin strata, SENTINEL-FREE: a fallen candidate's 1.0 is not a V reading, and
            # including it would make `worst` mostly mean `fell`.
            real = m & ~self.sentinel
            if bool(real.any()):
                rows = self.pair_row[real]
                mg = self.margin[real]
                worst = torch.zeros_like(mg, dtype=torch.bool)
                runner = torch.zeros_like(mg, dtype=torch.bool)
                for r in rows.unique():
                    idx = (rows == r).nonzero(as_tuple=True)[0]
                    worst[idx[mg[idx].argmax()]] = True     # largest V gap from the chosen
                    runner[idx[mg[idx].argmin()]] = True    # the near-tie
                gr = gap[real]
                res[f"pair/{name}/gap_worst"] = float(gr[worst].mean())
                res[f"pair/{name}/gap_runner_up"] = float(gr[runner].mean())
        if was_training:
            model.train()
        return res

if __name__ == "__main__":
    raise SystemExit(main())
