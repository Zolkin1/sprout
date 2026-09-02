#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Is the critic good enough to produce ADVANTAGES, given that it was not good enough to SELECT?

Best-of-K selection has now failed to pay across several interventions.  The hypothesis this
script tests is that the two uses ask different things of the same model: a selector must order
K futures that are one chunk downstream of the SAME state and differ only by a flow-noise draw,
while advantage-weighted BC only needs to order chunks ACROSS the dataset.  The second is a far
easier ordering -- and also the one with a trap the first does not have, because an ordering
driven by "which state is this" rather than "which action was taken" reweights states, and a
policy cannot choose its state.

**Four things are measured, in the order they can kill the idea.**

1. **Is it memorised?**  Every number here uses a cross-fit reading -- the fold that did NOT
   train on this episode.  The IN-fold reading is computed too, and only so the gap can be
   quoted; it never reaches an estimator.
2. **Does the advantage vary WITHIN an episode at all?**  If its variance is almost entirely
   between episodes, advantage-weighted BC is filtered BC on successful episodes wearing a
   different name, and the critic contributed nothing (:func:`critic_eval.variance_decomposition`).
3. **Is that within-episode variation SIGNAL?**  Two cross-fit folds are two honest estimates
   of one number, so their correlation is the reliability coefficient and their disagreement is
   the noise floor (:func:`critic_eval.reliability`).  A within-episode spread that is mostly
   fold disagreement cannot order anything.
4. **Do the weights it implies keep a usable dataset?**  Effective sample size, where the mass
   lands in the episode, and how much of it sits on the chunks immediately before a failure.

**The bar to beat is named and computed**: ``A_base = G_t - mean(G)``, the trajectory-level
filtered-BC weight, which needs no critic.  An estimator that does not beat it is an argument
for deleting the critic from the pipeline, not for tuning it.

**Both heads are carried the whole way and never mixed.**  The survival head's value is
``-P(fail)`` under a terminal-only reward, so its TD residual is a difference of two readings;
the quality head's is the normalized tracking return, which is dense and supports a real GAE.
Their levels are not commensurable -- the survival head's is horizon-dependent and the quality
head's is not -- so a weighted sum of the two would be a scale choice masquerading as a method.

Usage::

    python scripts/flow_matching/critic_advantage_eval.py --stage score \\
        --rollout logs/rollouts/v4_big/rollout_2026-08-20_16-36-10_mode11_decor_plan_v1 \\
        --fm_checkpoint logs/flow_matching/2026-08-25_09-03-39_plan_v1_h64_27M_nojumpoff/best.pt \\
        --fit_dir logs/value/v4 --out logs/value/critic_eval_v4
    python scripts/flow_matching/critic_advantage_eval.py --stage metrics \\
        --out logs/value/critic_eval_v4
    python scripts/flow_matching/critic_advantage_eval.py --stage report \\
        --out logs/value/critic_eval_v4
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
for _d in (str(_HERE), str(_HERE.parent / "rsl_rl")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

#: The two critic variants every fit writes, and the order they are reported in.
VARIANTS = ("V_priv", "V_onboard")

#: GAE decays swept. ``0`` is the one-chunk TD advantage and ``1`` the Monte-Carlo one, so the
#: sweep is a sweep over how far the critic is trusted against the realized outcome.
LAMBDAS = (0.0, 0.5, 0.8, 0.95, 1.0)

#: AWR temperatures, applied AFTER standardizing the advantage by its own sd, so one grid is
#: meaningful across estimators whose raw scales differ by an order of magnitude.
BETAS = (0.1, 0.3, 1.0, 3.0)


def _jsonable(o):
    """Recursively convert numpy scalars and arrays into plain JSON types.

    Args:
        o: Any nest of dicts, lists, numpy scalars and arrays.

    Returns:
        The same structure with numpy types replaced.
    """
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


# ----------------------------------------------------------------------------------------
# stage: score
# ----------------------------------------------------------------------------------------


def _build_model(ck: dict, ctx, device):
    """Reconstruct a :class:`FlowValueModel` from what its checkpoint stamped.

    Every geometry argument is read from the checkpoint rather than from the rollout, because
    the checkpoint is the only record of what the model was BUILT with -- and the two can
    legitimately differ (a scan folded differently, a context token without the goal block).
    A disagreement here is a linear layer fed the wrong columns, which is finite and plausible
    rather than a shape error.

    Args:
        ck: A loaded ``fit_value.py`` checkpoint dict.
        ctx: The loaded FM context, whose ``model`` is the borrowed encoder.
        device: Torch device.

    Returns:
        The model, in eval mode with its weights loaded.

    Raises:
        SystemExit: If the checkpoint's stamped generator hash disagrees with the one loaded.
            The critic's inputs are normalized by the GENERATOR's normalizers, so a different
            generator is a different input space with every shape still agreeing.
    """

    from robot_rl.flow_matching import value  # noqa: PLC0415

    priv = bool(ck["privileged"])
    model = value.FlowValueModel(
        ctx.model,
        scan_hw=(tuple(int(x) for x in ck["scan_hw"]) if priv else None),
        proprio_dim=(int(ck["proprio_hw"][1]) if priv else 0),
        proprio_steps=(int(ck["proprio_hw"][0]) if priv else 0),
        context_dim=int(ck["context_dim"]),
        quality_bins=int(ck.get("quality_bins", 0) or 0),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def _score_all(model, ctx, W: dict, priv: bool, device, scan_hw, batch: int,
               goal_context: bool, want_quality: bool):
    """``P(fail)`` and, when the model has the head, ``G_hat`` over every row.

    Args:
        model: A loaded :class:`FlowValueModel`.
        ctx: The FM context supplying the normalizers.
        W: The window table, in RAM.
        priv: Whether to build the privileged batch.
        device: Torch device.
        scan_hw: Height-scan grid.
        batch: Rows per forward pass.
        goal_context: Whether the goal offsets go in the context token.
        want_quality: Whether to read the quality head.

    Returns:
        ``(p_fail, g_hat)`` -- ``[W]`` float64, the second all-NaN when the model has no
        quality head.
    """
    import torch  # noqa: PLC0415

    import fit_value as fv  # noqa: PLC0415

    n = W["episode_uid"].shape[0]
    p = np.empty(n, dtype=np.float64)
    q = np.full(n, np.nan, dtype=np.float64)
    idx = np.arange(n)
    with torch.no_grad():
        for i in range(0, n, batch):
            j = idx[i:i + batch]
            xb = fv._batch(ctx, W, j, priv, device, scan_hw, goal_context=goal_context)
            p[j] = model.p_fail(xb).cpu().numpy()
            if want_quality:
                q[j] = model.quality(xb).cpu().numpy()
            if (i // batch) % 200 == 0:
                print(f"      {i}/{n}", flush=True)
    return p, q


def stage_score(args) -> int:
    """Score every surviving window with EVERY fold of every variant, in one pass.

    Both folds score every row on purpose.  For a row on a training tile one fold held its
    episode out and the other trained on it, which is exactly the ``V_out`` / ``V_in`` pair
    the memorisation gap needs; for a row on a HELD-OUT tile neither fold ever saw it, so the
    two readings are two honest estimates of one number and their disagreement is the noise
    floor every ranking claim has to clear.

    **Fold membership is recovered by MATCHING, not by re-deriving.**  The fit's split depends
    on ``--seed`` and ``--val_frac``, which the checkpoints do not stamp, and a re-derivation
    that silently used today's defaults would mislabel every row while looking perfectly
    healthy.  The fit's own ``predictions.npz`` holds the out-of-fold reading per row, so the
    fold that produced it is identified by which of the two re-scored readings it equals --
    which doubles as an end-to-end check that this script rebuilds the model correctly.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Process exit status.

    Raises:
        SystemExit: If the row set does not match the fit's, or if the recovered folds do not
            explain the fit's own predictions.
    """
    import torch  # noqa: PLC0415

    import fit_value as fv  # noqa: PLC0415

    from robot_rl.flow_matching import online  # noqa: PLC0415

    os.makedirs(args.out, exist_ok=True)
    print(f"[score] loading {args.rollout} ...", flush=True)
    W, E, lab = fv._load(args.rollout, 0)
    # `_load` stops at the drop-after-latch cut; the CENSORED drop lives in the fit's own
    # `main`, so it has to be repeated here or the row sets differ by one episode per env
    # (8192 of 23229 on v4_big -- every env's last, still-in-flight episode). Those have no
    # outcome at all: they neither fell nor failed to arrive, they were cut when the rollout
    # hit its window ceiling.
    keep = lab["live"]
    if not keep.all():
        print(f"[score] dropping {int((~keep).sum())} window(s) from "
              f"{int(np.unique(W['episode_uid'][~keep]).size)} censored episode(s)")
        W = {k: v[keep] for k, v in W.items()}
        lab = {k: v[keep] for k, v in lab.items()}
    n = W["episode_uid"].shape[0]
    scan_hw = tuple(int(x) for x in E["attrs"]["height_scan_hw"])
    print(f"[score] {n} windows, scan {scan_hw}", flush=True)

    fit_pred = np.load(os.path.join(args.fit_dir, "predictions.npz"), allow_pickle=True)
    if fit_pred["episode_uid"].shape[0] != n:
        raise SystemExit(
            f"the fit scored {fit_pred['episode_uid'].shape[0]} windows and this load produced "
            f"{n}. The row sets must be identical or every per-row join below is off by an "
            "unknown amount -- re-run against the rollout the fit actually used.")
    if not (fit_pred["episode_uid"] == W["episode_uid"]).all():
        raise SystemExit("the fit's row ORDER differs from this load's; the join would be wrong.")

    ctx = online.load_fm_context(args.fm_checkpoint, device=args.device)
    out: dict[str, np.ndarray] = {}
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; known: {list(VARIANTS)}")
    for variant in wanted:
        for k in (0, 1):
            path = os.path.join(args.fit_dir, f"{variant}.fold{k}.pt")
            if not os.path.exists(path):
                print(f"[score] {path} absent, skipping")
                continue
            ck = torch.load(path, map_location=args.device, weights_only=False)
            # An ABSENT stamp means True: every checkpoint predating the flag did carry the
            # goal block, so defaulting it to today's value would restate a later decision as
            # a property of an earlier artifact.
            gc = bool(ck.get("goal_context", True))
            qbins = int(ck.get("quality_bins", 0) or 0)
            print(f"[score] {variant} fold{k}: context {ck['context_dim']} wide, "
                  f"goal_context={gc}, quality_bins={qbins}", flush=True)
            model = _build_model(ck, ctx, args.device)
            p, q = _score_all(model, ctx, W, bool(ck["privileged"]), args.device, scan_hw,
                              args.batch, gc, qbins > 0)
            out[f"p_{variant}_f{k}"] = p
            out[f"q_{variant}_f{k}"] = q
            del model
            torch.cuda.empty_cache()

    # ---- recover which fold held each row out, by matching the fit's own reading ----------
    ref = fit_pred["p_V_priv"].astype(np.float64)
    d0 = np.abs(out["p_V_priv_f0"] - ref)
    d1 = np.abs(out["p_V_priv_f1"] - ref)
    held_out_by = np.where(d0 <= d1, 0, 1)
    best = np.minimum(d0, d1)
    val_tile = fit_pred["val_tile"].astype(bool)
    # On a held-out tile the fit AVERAGED the two folds, so no single fold reproduces it; the
    # average must. Everywhere else exactly one fold must, to numerical noise.
    avg_err = np.abs(0.5 * (out["p_V_priv_f0"] + out["p_V_priv_f1"]) - ref)
    # THE THRESHOLD IS SET BY WHAT A REAL MISMATCH LOOKS LIKE, not by floating-point taste.
    # A model built wrong -- a different context width, a transposed scan, a re-fitted
    # normalizer -- produces probabilities that differ by an appreciable fraction of their own
    # range, O(0.1). A correctly rebuilt model re-run on a GPU differs by ~1e-4: the fit used
    # `--batch 64` and this pass uses 256, so the reductions inside every layer are ordered
    # differently, and TF32 matmuls are not associative either. A tolerance tight enough to
    # notice the second cannot distinguish it from the first, so the max is checked against a
    # bound far above fp32 noise and far below any real error, and the MEDIAN is checked too --
    # a systematic offset would move it while leaving the max unremarkable.
    err_ep, err_tile = best[~val_tile], avg_err[val_tile]
    diag = {"median": float(np.median(err_ep)), "p99": float(np.percentile(err_ep, 99)),
            "max": float(err_ep.max()), "tile_max": float(err_tile.max()),
            "tile_median": float(np.median(err_tile))}
    print(f"[score] fold recovery vs the fit's own predictions: training-tile |err| "
          f"median {diag['median']:.2e} p99 {diag['p99']:.2e} max {diag['max']:.2e}; "
          f"held-out-tile (fold average) median {diag['tile_median']:.2e} "
          f"max {diag['tile_max']:.2e}")
    # How often the recovery is genuinely ambiguous: where the two folds agree to within the
    # matching noise, which fold produced the stored value cannot be told from the value.
    amb = float((np.abs(out["p_V_priv_f0"] - out["p_V_priv_f1"])[~val_tile]
                 < 2 * diag["p99"]).mean())
    diag["ambiguous_frac"] = amb
    print(f"[score] fold assignment ambiguous on {100 * amb:.3f}% of training-tile rows "
          "(the two folds agree there to within the matching noise, so V_in and V_out are "
          "interchangeable on those rows and the memorisation gap is unaffected)")
    recovery_ok = not (diag["max"] > 1e-2 or diag["median"] > 1e-5
                       or diag["tile_max"] > 1e-2)
    # On a held-out tile NEITHER fold trained on the row, so there is no in-fold reading to
    # name and `held_out_by` is meaningless there. Marked -1 rather than left as an arbitrary
    # argmin, which would silently promote one fold's reading to "the out-of-fold one".
    held_out_by[val_tile] = -1

    # WRITTEN BEFORE THE VERDICT, on purpose. Four forward passes over 12 GB of exteroception
    # cost about an hour, and discarding them because a check failed leaves nothing to
    # diagnose the failure WITH -- which is exactly what the first version of this script did.
    # The verdict rides in the file as `recovery_ok`.
    np.savez_compressed(
        os.path.join(args.out, "scores.npz"),
        recovery_ok=bool(recovery_ok), recovery_diag=json.dumps(diag),
        episode_uid=W["episode_uid"], replan_index=W["replan_index"],
        family_code=lab["family_code"], failed=lab["failed"],
        first_fall_step=lab["first_fall_step"],
        tile=fit_pred["tile"], val_tile=val_tile, held_out_by=held_out_by,
        chunk_reward=lab["chunk_reward"], g=lab["g"], g_valid=lab["g_valid"],
        gamma=float(fit_pred["gamma"]) if "gamma" in fit_pred.files else np.nan,
        **out)
    print(f"[score] wrote {os.path.join(args.out, 'scores.npz')}")
    if not recovery_ok:
        raise SystemExit(
            "the re-scored readings do not reproduce the fit's own predictions.npz. That is "
            "not a tolerance problem -- it means this script builds a different model than "
            "the fit did (a different context width, a different scan fold, a different "
            "normalizer), and every number downstream would be about the wrong model. The "
            "scores were written anyway, so the disagreement can be inspected.")
    return 0


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("score", "metrics", "report"))
    p.add_argument("--out", required=True, help="Output directory, shared by every stage.")
    p.add_argument("--rollout", help="Rollout stem (score).")
    p.add_argument("--fm_checkpoint", help="The generator the critic borrowed (score).")
    p.add_argument("--fit_dir", help="The fit_value.py output directory (score).")
    p.add_argument("--variants", default=",".join(VARIANTS),
                   help="Which variants to re-score. Each costs a full pass over the "
                        "recording's exteroception per fold, so restricting to V_priv halves "
                        "the wall clock; the fit's own predictions.npz still supplies every "
                        "variant's OUT-of-fold reading, and only V_in and the per-fold "
                        "reliability need this pass.")
    p.add_argument("--batch", type=int, default=64,
                   help="Rows per forward pass. Defaults to the fit's own default, so the "
                        "reductions are ordered identically and the recovery check compares "
                        "like with like; a different batch shifts every probability by ~1e-4.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_boot", type=int, default=2000,
                   help="Cluster bootstrap draws for every interval reported.")
    args = p.parse_args()

    if args.stage == "score":
        missing = [f for f in ("rollout", "fm_checkpoint", "fit_dir") if not getattr(args, f)]
        if missing:
            raise SystemExit(f"--stage score needs {', '.join('--' + m for m in missing)}")
        return stage_score(args)
    if args.stage == "metrics":
        from critic_eval_metrics import stage_metrics  # noqa: PLC0415
        return stage_metrics(args)
    from critic_eval_report import stage_report  # noqa: PLC0415
    return stage_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
