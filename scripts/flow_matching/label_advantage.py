# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a critic's per-window ``P(fail)`` into the advantage label the fine-tune conditions on.

Reads ``predictions.npz`` (from ``fit_value.py``) and writes ``advantage.npz`` -- a sidecar
keyed by window index in the SAME order as the rollout ``.h5``, so the fine-tune dataset
joins it on ``(episode_uid, replan_index)`` -- see ``rollout_dataset.join_labels`` for why a
positional join would be silently wrong.

It used to also run a **G2 gate** -- the load-bearing statistic here, and the reason this was a
separate stage rather than a line inside the trainer::

    within narrow bands of V(s_t), do positive-labelled chunks fail less than
    negative-labelled ones?

G1 ("the critic predicts failure") can pass by a wide margin while G2 fails completely --
that happens exactly when the advantage is reproducing state difficulty, in which case the
label carries no information about the CHUNK and conditioning on it cannot work.  Banding on
``V(s_t)`` is what controls for difficulty; the unbanded comparison is true by construction
and says nothing.

Usage::

    python scripts/flow_matching/label_advantage.py \\
        logs/value/v2/predictions.npz --rollout logs/rollouts/v2/rollout_<...> \\
        --n 8 --out logs/value/v2/advantage.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "source", "robot_rl"), os.path.join(_ROOT, "source", "motion_gen"),
           os.path.join(_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_rl.flow_matching import advantage as adv_mod  # noqa: E402

#: Reward sources that ARE the terrain signal, so `--terrain_only` accepts them. Both are
#: per-chunk terrain-gated rewards measured by the SAME grader; `terrain_progress` adds a
#: gated velocity term and a one-off terminal bonus on top of `terrain`'s geometry, which
#: changes the reward's RANGE (and its reducer) but not what the advantage means. Restated
#: here rather than imported from `fit_value` because that module launches an argparse and
#: reads h5; the two are pinned equal by `test_label_advantage.py`.
TERRAIN_REWARD_SOURCES: tuple[str, ...] = ("terrain", "terrain_progress")

#: The progress reward specifically, whose target is bootstrapped and whose per-chunk range
#: is not `terrain`'s. Named so the one condition that differs can name it.
PROGRESS_REWARD_SOURCE: str = "terrain_progress"



def check_reward_consistency(fitted_on: str, estimator: str,
                             terrain_only: bool, fitted_hash: str = "?",
                             quality_range: tuple[float, float] = (0.0, 1.0)) -> None:
    """Refuse an advantage that would silently mix two different rewards.

    "Terrain rewards only" was three conditions; the first is now structural rather than
    checked. There is no survival half to switch off -- the critic has one head and the
    advantage is ``A = G - V`` -- so what remains is:

    1. the estimator must be ``mc``, or the quality half bootstraps on the critic's own
       prediction at ``t + n`` and stops being a difference against measured terrain reward;
    2. the critic's head must have been FITTED on the terrain reward, or the difference is
       against a prediction of a different quantity -- meaningless while looking normal.

    Args:
        fitted_on: The ``reward_source`` stamped into the predictions.
        estimator: ``"mc"`` or otherwise.
        terrain_only: Whether the caller asked for the terrain-only signal.
        fitted_hash: The reward hash, for the log line.
        quality_range: The support the value head was fitted over, for the log line.

    Raises:
        SystemExit: If ``terrain_only`` is asked for and either condition does not hold.
    """
    if terrain_only:
        if fitted_on not in TERRAIN_REWARD_SOURCES:
            raise SystemExit(
                f"--terrain_only, but this critic was fitted on {fitted_on!r}, which is not "
                f"one of {TERRAIN_REWARD_SOURCES}. Its value "
                "head predicts a different reward from the one the advantage would measure, "
                "so the difference would be meaningless while looking entirely normal. "
                "Refit with: fit_value.py --reward_npz <terrain sidecar>.")
        if estimator != "mc":
            raise SystemExit(
                "--terrain_only needs --estimator mc: an n-step quality half bootstraps on "
                "the critic's prediction at t+n, so it is not a difference against measured "
                "terrain reward.")
        if fitted_on == PROGRESS_REWARD_SOURCE:
            # The ADVANTAGE stays MC -- G is still the realized return. What changed is the
            # BASELINE: under fitted value iteration the critic's head was fitted against a
            # bootstrapped target, so V(s) estimates the discounted return rather than the
            # 3-chunk window sum. That is the intent (a terminal bonus cannot reach 3 chunks
            # back), but it means `G - V` differences a full-horizon baseline against a
            # windowed return unless G is built the same way. Stated rather than refused,
            # because `fit_value` bounds both by the same `return_max` and the loop builds G
            # with the same reducer.
            print("[label] NOTE: this critic was fitted on the PROGRESS reward, whose target "
                  "is the (1-gamma)-weighted return with a terminal bonus. G must be built "
                  "by the same reducer (bootstrapped_window_return) and bounded by the same "
                  "return_max, or the advantage differences two scales.")
        print(f"[label] TERRAIN ONLY: A = G - V, MC target, critic fitted on "
              f"{fitted_on!r} (hash {fitted_hash}), scalar head over "
              f"[{quality_range[0]:g}, {quality_range[1]:g}]")




def _commanded_speed(stem: str, uid: np.ndarray, ridx: np.ndarray) -> np.ndarray:
    """``|v_xy|`` commanded at each labelled window, joined from the recording BY KEY.

    Args:
        stem: Rollout stem (without ``.h5``).
        uid: ``[W]`` episode ids of the rows being labelled.
        ridx: ``[W]`` re-plan indices of those rows.

    Returns:
        ``[W]`` commanded planar speed in m/s.

    Raises:
        SystemExit: If any labelled row has no matching window. The fit drops censored and
            post-latch rows, so the tables differ in LENGTH -- a positional join would line
            up silently and mis-assign every speed.
    """
    import h5py  # noqa: PLC0415

    with h5py.File(stem + ".h5", "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        cmd0 = f["cmd_future"][:n, 0, :2]      # the command in force at the window
        w_uid, w_ri = f["episode_uid"][:n], f["replan_index"][:n]
    key_w = w_uid.astype(np.int64) * 100000 + w_ri.astype(np.int64)
    key_l = uid.astype(np.int64) * 100000 + ridx.astype(np.int64)
    order = np.argsort(key_w)
    pos = order[np.searchsorted(key_w, key_l, sorter=order)]
    if not (key_w[pos] == key_l).all():
        raise SystemExit(f"{stem}.h5 has no window for some labelled rows; the recording and "
                         "the predictions do not describe the same run.")
    return np.linalg.norm(cmd0[pos], axis=1)


def main() -> int:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", help="predictions.npz from fit_value.py")
    p.add_argument("--out", required=True, help="Output .npz sidecar.")
    p.add_argument("--variant", default="V_priv", choices=("V_priv",),
                   help="Which critic to label with.")
    p.add_argument("--terrain_only", action="store_true",
                   help="the advantage is the TERRAIN reward and nothing else. Requires "
                        "--estimator mc and a critic fitted on a terrain reward "
                        "sidecar; each is CHECKED, because any one missing gives a plausible "
                        "wrong label rather than an error.")
    p.add_argument("--estimator", default="mc", choices=("mc",),
                   help="How the advantage target is formed. `mc` differences the REALIZED "
                        "return against the critic's baseline: A = G - V, no bootstrap. "
                        "The `n_step` alternative bootstrapped on the critic's own prediction "
                        "at t+n and blended a survival half; both went with the survival "
                        "head, and neither produced a reported result. `--estimator n_step` is now "
                        "an argparse error rather than a silent fallback.")
    p.add_argument("--rollout", default="",
                   help="Rollout stem, needed only for --speed_bins: the commanded velocity "
                        "lives in the recording's `cmd_future`, not in predictions.npz. "
                        "Joined BY KEY on (episode_uid, replan_index), since the fit dropped "
                        "censored and post-latch rows and the two tables differ in length.")
    p.add_argument("--speed_bins", type=int, default=1,
                   help="Cross the cut with this many EQUAL-COUNT commanded-speed bins. 1 "
                        "(the default) does not stratify on speed. Without it a Monte-Carlo "
                        "advantage selects faster commands -- measured 0.33 keep rate above "
                        "1.3 m/s against 0.23 below 0.9 -- so the fine-tune would see a "
                        "different speed distribution from the recording, and a control arm "
                        "filtered by a different rule skews the other way, which confounds "
                        "the comparison with the thing being compared. Quantile edges, not "
                        "fixed ones: a fixed grid puts a handful of rows in the extreme bins "
                        "and a percentile over a handful of rows is noise.")
    p.add_argument("--progress_bins", type=int, default=1,
                   help="Cross the per-family percentile with this many NORMALIZED PROGRESS "
                        "bins. 1 (the default) is the historical family-only cut, which "
                        "over-selects the openings of surviving episodes because a "
                        "Monte-Carlo advantage is biased positive there -- the critic hedges "
                        "while the outcome is uncertain and the survivors beat the hedge. 5 "
                        "flattens the keep rate across the episode.")
    p.add_argument("--percentile", type=float, default=35.0,
                   help="Within-family percentile below which a chunk is negative.")
    p.add_argument("--null_spread", type=float, default=0.02,
                   help="A family whose advantage IQR is below this is labelled null "
                        "wholesale -- there is nothing to prefer, and thresholding would "
                        "manufacture a distinction the data does not contain.")
    p.add_argument("--exclude_tile_holdout", action="store_true",
                   help="Drop windows on held-out tiles from the label (they are the "
                        "generalisation measure; training on them spends it).")
    args = p.parse_args()

    d = np.load(args.predictions)
    # No `p_{variant}` guard: `fit_value` writes no survival prediction any more, so a check
    # for one refused every valid file. It aborted the loop at stage 7/8 on the very first
    # iteration -- the READ was deleted with the survival head and the GUARD was left behind.
    folds = int(d["num_folds"]) if "num_folds" in d.files else 1
    if folds == 1:
        print("[label] WARNING: the critic was fit with --folds 1, so most of these "
              "predictions are IN SAMPLE -- the critic has seen the outcome of the very "
              "episode it is scoring. Under memorisation V(s_t) and V(s_{t+n}) both collapse "
              "onto the realized outcome and the advantage goes to zero, so the label degrades "
              "to noise rather than becoming wrong. Refit with --folds 2 (same wall-clock).")

    # `p_<variant>` is gone: fit_value no longer writes a survival prediction.
    uid, ridx, fam = d["episode_uid"], d["replan_index"], d["family_code"]
    failed, val_tile = d["failed"].astype(bool), d["val_tile"].astype(bool)

    fitted_on = str(d["reward_source"]) if "reward_source" in d.files else "track_err"
    fitted_hash = str(d["reward_hash"]) if "reward_hash" in d.files else "?"
    q_range = (tuple(d["quality_range"].tolist()) if "quality_range" in d.files
               else (0.0, 1.0))
    check_reward_consistency(fitted_on, args.estimator,
                             bool(args.terrain_only), fitted_hash, q_range)

    qkey = f"q_{args.variant}"
    # ONE estimator. The n-step alternatives (`n_step_advantage`,
    # `combined_n_step_advantage`) differenced against the critic's own prediction at t+n and
    # blended a SURVIVAL half; both are gone with the survival head, and neither produced a
    # reported result -- the loop passes `--estimator mc`.
    if qkey not in d.files or "g" not in d.files:
        raise SystemExit(
            f"needs the value head and the realized return, but {args.predictions} has no "
            f"{qkey}/g. Refit with a value head.")
    # **`g_mc` when the file has it.** Under fitted value iteration `g` is the target that
    # round FITTED, which past round 0 is `r + gamma^N V_prev` -- so differencing it against
    # the head gives a TD advantage rather than `G_real - V`. `g_mc` is the realized Monte
    # Carlo return the same reducer produced; a file without it either predates the split or
    # was a pure MC fit, in which case `g` IS the realized return.
    g_hat = d[qkey]
    if "g_mc" in d.files:
        g = d["g_mc"]
        print("[label] G = the realized MONTE CARLO return (`g_mc`); the critic's own "
              "target was bootstrapped, which is the intended split.")
    else:
        g = d["g"]
        _boot = ("fvi_bootstrap_from" in d.files
                 and str(d["fvi_bootstrap_from"]).strip() not in ("", "None"))
        if _boot:
            raise SystemExit(
                f"{args.predictions} was fitted with --bootstrap_v but carries no "
                "`g_mc`, so the only return in it is the BOOTSTRAPPED target. "
                "Differencing that against the head gives a TD advantage, not "
                "G_real - V. Refit with a fit_value that writes `g_mc`.")
    g_valid = d["g_valid"].astype(bool)
    a, boot = adv_mod.mc_advantage(g, g_hat, valid=g_valid)
    boot = np.zeros_like(boot)          # nothing is bootstrapped by construction
    print(f"[label] MC advantage A = G - V, "
          f"{int(g_valid.sum())}/{g_valid.size} windows carry a measured return")
    # PROGRESS STRATA, crossed with the family. A Monte-Carlo advantage is systematically
    # positive early in a surviving episode (the critic hedges when the outcome is still
    # uncertain, and the survivors beat the hedge), so a family-only cut over-selects episode
    # OPENINGS and under-selects their ends -- which is where the terrain is. Computed from
    # the table's own last index per episode, not from any recorded length, because the fit
    # drops censored episodes and every window at or after the first bad step.
    strata = None
    n_prog, n_speed = int(args.progress_bins), int(args.speed_bins)
    if n_speed > 1 and not args.rollout:
        raise SystemExit("--speed_bins needs --rollout: the commanded velocity is in the "
                         "recording's `cmd_future`, not in predictions.npz.")
    if n_prog > 1:
        _, inv = np.unique(uid, return_inverse=True)
        last = np.zeros(uid.size, dtype=np.int64)
        np.maximum.at(last, inv, ridx)
        denom = np.maximum(last[inv], 1)
        prog = np.where(last[inv] > 0, ridx / denom, 0.0)
        strata = np.clip((prog * n_prog).astype(np.int64), 0, n_prog - 1)
        print(f"[label] stratifying the cut by family x {n_prog} progress bins")
    if n_speed > 1:
        speed = _commanded_speed(args.rollout, uid, ridx)
        edges = np.quantile(speed, np.linspace(0.0, 1.0, n_speed + 1))[1:-1]
        sb = np.searchsorted(edges, speed).astype(np.int64)
        strata = sb if strata is None else (strata * n_speed + sb)
        qs = ", ".join(f"{e:.2f}" for e in edges)
        print(f"[label] ... x {n_speed} equal-count speed bins (edges {qs} m/s)")
    labels, per_family = adv_mod.binarize(
        a, fam, strata=strata, percentile=args.percentile, null_spread=args.null_spread)
    if args.exclude_tile_holdout:
        labels = np.where(val_tile, adv_mod.LABEL_NULL, labels)

    counts = {name: int((labels == i).sum()) for i, name in enumerate(adv_mod.LABEL_NAMES)}
    print(f"[label] variant {args.variant}, {folds} fold(s)")
    print(f"[label] advantage: mean {a.mean():+.4f}  sd {a.std():.4f}  "
          f"|A|>0.05 on {float((np.abs(a) > 0.05).mean()):.1%} of windows  "
          f"({float(boot.mean()):.1%} bootstrapped on the outcome)")
    print(f"[label] labels: {counts}")
    for f, r in sorted(per_family.items()):
        tag = "NULL (no spread)" if r["null"] else \
            f"thr {r['threshold']:+.4f}  positive {r['positive_frac']:.1%}"
        print(f"    family {f}: n={r['n']:7d}  IQR {r['iqr']:.4f}  {tag}")

    # The G2 gate is GONE, not skipped-by-flag. It banded the label by ``P(fail|s_t)`` to ask
    # whether the advantage was merely reproducing state difficulty -- a real question, and
    # the only one of these statistics that was a test rather than true by construction. It
    # needed a trained survival head to band by, and that head no longer exists. Nothing here
    # substitutes for it: banding by ``V(s_t)`` instead would condition the label on the very
    # quantity it is built from, which is not the same test.
    #
    # TODO(g2-gate): band by a difficulty proxy the critic does not produce -- terrain class,
    # or the course's own approach distance -- and restore the test.

    # WRITTEN IN BOTH BRANCHES, and at THIS indentation deliberately. This block used to sit
    # inside the `else` above, so a `--quality_only` fit skipped the G2 gate and silently
    # skipped the WRITE with it: the run printed every statistic, returned 0, and left no
    # advantage.npz -- which a downstream fine-tune then either refused or, worse, satisfied
    # from a stale file. `meta`'s own `g2_skipped` field is the evidence it was always meant to
    # run without the gate. Pinned by test_label_advantage_writes_under_quality_only.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # No "n" and no "tail_chunks": both described the n-step estimator, and stamping them
    # into an MC label was false provenance -- a later reader saw `meta["n"] = 8` and
    # concluded the label had been bootstrapped 8 chunks ahead.
    meta = {"estimator": "mc", "variant": args.variant,
            "percentile": float(args.percentile),
            "null_spread": float(args.null_spread), "num_folds": folds,
            "progress_bins": int(args.progress_bins),
            "speed_bins": int(args.speed_bins), "rollout": args.rollout,
            "predictions": os.path.abspath(args.predictions),
            "counts": counts, "per_family": {str(k): v for k, v in per_family.items()},
            "g2_skipped": "the survival head it banded by was removed"}
    # `stratum` is the CELL binarize cut within, not a convenience: a consumer that
    # re-derives the family x progress x speed crossing from `family_code` alone gets a
    # different partition, and AWR standardizing over the wrong one puts back exactly the
    # speed skew the stratified threshold removed (0.33 keep rate above 1.3 m/s against 0.23
    # below 0.9). Saved so it is read rather than rebuilt.
    np.savez(args.out, label=labels.astype(np.int8), advantage=a.astype(np.float32),
             bootstrapped=boot, episode_uid=uid, replan_index=ridx, family_code=fam,
             stratum=adv_mod.group_ids(fam, strata), val_tile=val_tile,
             meta=json.dumps(meta))
    print(f"[label] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
