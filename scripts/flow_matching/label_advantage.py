# Copyright 2026 Zachary Olkin. All rights reserved.

"""Turn a critic's per-window ``P(fail)`` into the advantage label the fine-tune conditions on.

Reads ``predictions.npz`` (from ``fit_value.py``) and writes ``advantage.npz`` -- a sidecar
keyed by window index in the SAME order as the rollout ``.h5``, so the fine-tune dataset
joins it on ``(episode_uid, replan_index)`` -- see ``rollout_dataset.join_labels`` for why a
positional join would be silently wrong.

Also runs the plan's **G2 gate**, which is the load-bearing one and the reason this is a
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



def check_reward_consistency(fitted_on: str, w_fail: float, estimator: str,
                             terrain_only: bool, fitted_hash: str = "?",
                             quality_range: tuple[float, float] = (0.0, 1.0),
                             quality_kind: str = "two_hot",
                             quality_only: bool = False) -> None:
    """Refuse an advantage that would silently mix two different rewards.

    "Terrain rewards only" is three conditions, and **any one of them missing gives a
    plausible, wrong label rather than an error**:

    1. the SURVIVAL half must be off (``w_fail == 0``), or the advantage is part realized
       fall outcome;
    2. the estimator must be ``mc``, or the quality half bootstraps on the critic's own
       prediction at ``t + n`` and stops being a difference against measured terrain reward;
    3. the critic's quality head must have been FITTED on the terrain reward, or the advantage
       is ``G_terrain - G_hat_tracking`` -- right shape, right units, no meaning.

    (3) is the dangerous one: nothing about it shows up in the numbers, which is why it is
    stamped into ``predictions.npz`` by ``fit_value.py`` and checked here rather than trusted.

    Args:
        fitted_on: The ``reward_source`` the critic was fitted on.
        w_fail: Weight on the survival half.
        estimator: ``"n_step"`` or ``"mc"``.
        terrain_only: Whether the caller asked for the terrain signal alone.
        fitted_hash: The sidecar hash stamped at fit time, for the log line.
        quality_range: The support the critic's quality head was fitted over. Reported,
            because ``[0, 1]`` and ``[-1, 0]`` are both plausible-looking and an advantage
            formed across them is silently wrong.
        quality_kind: ``"two_hot"`` or ``"mse"``, for the log line.
        quality_only: Whether the critic was fitted with the survival head's loss DROPPED. If
            it was, ``p_fail`` was never trained -- it still produces numbers, and they look
            exactly like predictions -- so any ``w_fail > 0`` is refused outright rather than
            warned about. That is not a configuration, it is reading an untrained head.

    Raises:
        SystemExit: If ``terrain_only`` is asked for and any of the three does not hold.
    """
    if quality_only and w_fail > 0.0:
        raise SystemExit(
            f"--w_fail {w_fail} reads P(fail|s), but this critic was fitted with "
            "--quality_only: the survival head's loss was dropped, so that head is UNTRAINED "
            "and its output is noise shaped like a probability. Label at --w_fail 0, or refit "
            "without --quality_only.")
    if terrain_only:
        if fitted_on not in TERRAIN_REWARD_SOURCES:
            raise SystemExit(
                f"--terrain_only, but this critic was fitted on {fitted_on!r}, which is not "
                f"one of {TERRAIN_REWARD_SOURCES}. Its quality "
                "head predicts a different reward from the one the advantage would measure, "
                "so the difference would be meaningless while looking entirely normal. "
                "Refit with: fit_value.py --reward_npz <terrain sidecar>.")
        if w_fail != 0.0:
            raise SystemExit(
                f"--terrain_only needs --w_fail 0 so the survival half is excluded; got "
                f"{w_fail}. At w_fail > 0 the advantage mixes the realized fall outcome into "
                "what is supposed to be a terrain-only signal.")
        if estimator != "mc":
            raise SystemExit(
                "--terrain_only needs --estimator mc: the n_step quality half bootstraps on "
                "the critic's prediction at t+n, so it is not a difference against measured "
                "terrain reward.")
        if fitted_on == PROGRESS_REWARD_SOURCE:
            # The ADVANTAGE stays MC -- `--estimator mc` above is still required, and G is
            # still the realized return. What changed is the BASELINE: under fitted value
            # iteration the critic's quality head was fitted against a bootstrapped target,
            # so V(s) estimates the discounted return rather than the 3-chunk window sum.
            # That is the intent (a terminal bonus cannot reach 3 chunks back), but it means
            # `G - V` differences a full-horizon baseline against a windowed return unless
            # G is built the same way. Stated rather than refused, because `fit_value` bounds
            # both by the same `return_max` and the loop builds G with the same reducer.
            print("[label] NOTE: this critic was fitted on the PROGRESS reward, whose target "
                  "is the (1-gamma)-weighted return with a terminal bonus. G must be built "
                  "by the same reducer (bootstrapped_window_return) and bounded by the same "
                  "return_max, or the advantage differences two scales.")
        print(f"[label] TERRAIN ONLY: quality half alone, MC target, critic fitted on "
              f"{fitted_on!r} (hash {fitted_hash}), {quality_kind} head over "
              f"[{quality_range[0]:g}, {quality_range[1]:g}]"
              + (", survival head UNTRAINED (quality_only fit)" if quality_only else ""))
    elif fitted_on in TERRAIN_REWARD_SOURCES and w_fail > 0.0:
        print(f"[label] WARNING: this critic was fitted on the TERRAIN reward but --w_fail "
              f"{w_fail} mixes in the realized fall outcome. Pass --terrain_only if you meant "
              "the terrain signal alone.")


def _bandwise_gate(p_state: np.ndarray, labels: np.ndarray, failed: np.ndarray,
                   n_bands: int = 5) -> list[dict]:
    """G2: compare positive- and negative-labelled chunks WITHIN narrow ``V(s_t)`` bands.

    Args:
        p_state: ``[W]`` ``P(fail | s_t)``.
        labels: ``[W]`` in ``{0, 1, 2}``.
        failed: ``[W]`` bool, the realized episode outcome.
        n_bands: Number of equal-count bands over ``p_state``.

    Returns:
        One record per band: its ``P(fail)`` range, the counts, the realized failure rate
        under each label and their difference.  A band with fewer than 50 of either label
        reports ``None`` for the rates rather than a number off a handful of episodes.
    """
    edges = np.quantile(p_state, np.linspace(0.0, 1.0, n_bands + 1))
    edges[-1] = np.inf
    out = []
    for i in range(n_bands):
        sel = (p_state >= edges[i]) & (p_state < edges[i + 1])
        pos = sel & (labels == adv_mod.LABEL_POSITIVE)
        neg = sel & (labels == adv_mod.LABEL_NEGATIVE)
        rec = {"band": i, "p_lo": float(edges[i]),
               "p_hi": float(edges[i + 1]) if np.isfinite(edges[i + 1]) else 1.0,
               "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        if pos.sum() >= 50 and neg.sum() >= 50:
            rec["fail_pos"] = float(failed[pos].mean())
            rec["fail_neg"] = float(failed[neg].mean())
            rec["delta"] = rec["fail_neg"] - rec["fail_pos"]
        else:
            rec["fail_pos"] = rec["fail_neg"] = rec["delta"] = None
        out.append(rec)
    return out


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
    p.add_argument("--variant", default="V_priv", choices=("V_priv", "V_onboard"),
                   help="Which critic to label with.")
    p.add_argument("--w_fail", type=float, default=1.0,
                   help="Weight on the SURVIVAL term of V = w_track*G_hat - w_fail*P(fail), "
                        "with w_track = 1 - w_fail. 1.0 (the default) is the survival-only "
                        "label this script has always produced and is the CONTROL ARM -- it "
                        "takes a separate code path rather than routing through the combined "
                        "function, so it cannot be perturbed by a change to it. Below 1 needs "
                        "a predictions.npz from a fit with a quality head.")
    p.add_argument("--terrain_only", action="store_true",
                   help="the advantage is the TERRAIN reward and nothing else. Requires "
                        "--w_fail 0, --estimator mc, and a critic fitted on a terrain reward "
                        "sidecar; each is CHECKED, because any one missing gives a plausible "
                        "wrong label rather than an error.")
    p.add_argument("--estimator", default="n_step", choices=("n_step", "mc"),
                   help="how the advantage target is formed. 'n_step' bootstraps on the "
                        "critic's own predictions at t+n; 'mc' differences against the "
                        "REALIZED return and outcome -- no bootstrap bias, more variance, and "
                        "no n to tune. Default stays n_step so existing artifacts are "
                        "reproduced unchanged.")
    p.add_argument("--n", type=int, default=8,
                   help="Advantage look-ahead in re-plans (8 = 1.92 s at K=12, 50 Hz).")
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
    p.add_argument("--tail_chunks", type=int, default=1,
                   help="Trailing chunks the non-failure tail averages. MUST match the value "
                        "the critic's own fit used (fit_value.py stamps it as `tail_chunks`); "
                        "a mismatch makes this advantage's G a different quantity from the "
                        "one its baseline was trained to predict.")
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
    key = f"p_{args.variant}"
    if key not in d:
        raise SystemExit(f"{args.predictions} has no {key}; it holds "
                         f"{[k for k in d.files if k.startswith('p_')]}")
    folds = int(d["num_folds"]) if "num_folds" in d.files else 1
    if folds == 1:
        print("[label] WARNING: the critic was fit with --folds 1, so most of these "
              "predictions are IN SAMPLE -- the critic has seen the outcome of the very "
              "episode it is scoring. Under memorisation V(s_t) and V(s_{t+n}) both collapse "
              "onto the realized outcome and the advantage goes to zero, so the label degrades "
              "to noise rather than becoming wrong. Refit with --folds 2 (same wall-clock).")

    p_fail = d[key]
    uid, ridx, fam = d["episode_uid"], d["replan_index"], d["family_code"]
    failed, val_tile = d["failed"].astype(bool), d["val_tile"].astype(bool)

    # The COMBINED advantage when a quality head is present and asked for.
    # `w_fail = 1` takes the survival-only path explicitly rather than passing w_fail=1.0
    # through the combined function: the two are equal by construction and tested to be, but
    # routing the control arm through the code it is controlling for is how a control stops
    # controlling for anything.
    fitted_on = str(d["reward_source"]) if "reward_source" in d.files else "track_err"
    fitted_hash = str(d["reward_hash"]) if "reward_hash" in d.files else "?"
    q_range = (tuple(d["quality_range"].tolist()) if "quality_range" in d.files
               else (0.0, 1.0))
    quality_only = bool(d["quality_only"]) if "quality_only" in d.files else False
    check_reward_consistency(fitted_on, float(args.w_fail), args.estimator,
                             bool(args.terrain_only), fitted_hash, q_range,
                             str(d["quality_kind"]) if "quality_kind" in d.files
                             else "two_hot",
                             quality_only)

    qkey = f"q_{args.variant}"
    if args.estimator == "mc":
        # MONTE CARLO: both halves difference against something measured, so there is no
        # bootstrap and no `n`. The quality half needs the return and its baseline; the
        # survival half needs only the realized outcome, so w_fail=1 works with no quality
        # head at all.
        if float(args.w_fail) >= 1.0:
            g_hat = np.zeros_like(p_fail)
            g = np.zeros_like(p_fail)
            g_valid = np.zeros(p_fail.shape[0], dtype=bool)
        elif qkey not in d.files or "g" not in d.files:
            raise SystemExit(
                f"--estimator mc at --w_fail {args.w_fail} needs the quality head and the "
                f"realized return, but {args.predictions} has no {qkey}/g. Refit with a "
                "quality head, or label at --w_fail 1.")
        else:
            # **`g_mc` when the file has it.** Under fitted value iteration `g` is the
            # target that round FITTED, which past round 0 is `r + gamma^N V_prev` -- so
            # differencing it against the head gives a TD advantage rather than
            # `G_real - V`. `g_mc` is the realized Monte Carlo return the same reducer
            # produced; a file without it either predates the split or was a pure MC fit, in
            # which case `g` IS the realized return.
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
        a, boot = adv_mod.mc_advantage(g, g_hat, p_fail, failed,
                                       w_fail=float(args.w_fail), valid=g_valid)
        boot = np.zeros_like(boot)          # nothing is bootstrapped by construction
        print(f"[label] MC advantage: w_fail {args.w_fail:.2f} / w_track "
              f"{1 - float(args.w_fail):.2f}, "
              f"{int(g_valid.sum())}/{g_valid.size} windows carry a measured return")
    elif float(args.w_fail) >= 1.0:
        a, boot = adv_mod.n_step_advantage(p_fail, uid, ridx, failed, args.n)
    elif qkey not in d.files:
        raise SystemExit(
            f"--w_fail {args.w_fail} asks for the tracking term, but {args.predictions} has "
            f"no {qkey}. It was written by a fit with --quality_bins 0, or before the quality "
            "head existed. Refit with a quality head, or label at --w_fail 1.")
    else:
        g_hat, g_valid = d[qkey], d["g_valid"].astype(bool)
        reward = d["chunk_reward"]
        gamma = float(d["gamma"]) if "gamma" in d.files else 0.95
        # The validity mask is PASSED, not pre-zeroed. Zero-filling the last window of each
        # episode charges it for a chunk that was never measured; dropping it is what
        # `normalized_returns` -- the recursion the quality head was fitted to -- does, and it
        # is what makes this advantage's G the same quantity its own baseline predicts.
        a, boot = adv_mod.combined_n_step_advantage(
            p_fail, g_hat, reward, uid, ridx, failed, args.n,
            gamma=gamma, w_fail=float(args.w_fail), valid=g_valid,
            tail_chunks=int(args.tail_chunks))
        print(f"[label] combined V: w_fail {args.w_fail:.2f} / w_track "
              f"{1 - float(args.w_fail):.2f}, gamma {gamma}, "
              f"{int(g_valid.sum())}/{g_valid.size} windows carry a measured reward")
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
    print(f"[label] n={args.n} re-plans, variant {args.variant}, {folds} fold(s)")
    print(f"[label] advantage: mean {a.mean():+.4f}  sd {a.std():.4f}  "
          f"|A|>0.05 on {float((np.abs(a) > 0.05).mean()):.1%} of windows  "
          f"({float(boot.mean()):.1%} bootstrapped on the outcome)")
    print(f"[label] labels: {counts}")
    for f, r in sorted(per_family.items()):
        tag = "NULL (no spread)" if r["null"] else \
            f"thr {r['threshold']:+.4f}  positive {r['positive_frac']:.1%}"
        print(f"    family {f}: n={r['n']:7d}  IQR {r['iqr']:.4f}  {tag}")

    # The G2 gate BANDS BY P(fail|s_t), so it needs a trained survival head. Under a
    # quality_only fit that head is untrained: every band collapses onto ~0.5, every delta is
    # 0.000, and the gate then prints "the label is reproducing state difficulty" -- a
    # conclusion about the LABEL drawn from a head that was never fitted. Skipped and said so,
    # on the same principle this file's sibling states outright: a number produced by a broken
    # head is worse than no number.
    if quality_only:
        gate = []
        print("[label] G2 gate SKIPPED: it bands by P(fail|s_t) and this critic was fitted "
              "with --quality_only, so that head is untrained. There is no survival "
              "stratification to condition on, and any delta it reported would be an "
              "artefact of an unfitted sigmoid sitting at 0.5.")
    else:
        gate = _bandwise_gate(p_fail, labels, failed)
        print("[label] G2 -- realized failure rate by label, WITHIN P(fail|s_t) bands:")
        print("        (the unbanded comparison is true by construction; only this one is a test)")
        for r in gate:
            if r["delta"] is None:
                print(f"    [{r['p_lo']:.3f}, {r['p_hi']:.3f})  n_pos {r['n_pos']:6d} "
                      f"n_neg {r['n_neg']:6d}   -- too few to report")
            else:
                print(f"    [{r['p_lo']:.3f}, {r['p_hi']:.3f})  n_pos {r['n_pos']:6d} "
                      f"n_neg {r['n_neg']:6d}   fail pos {r['fail_pos']:.3f} / "
                      f"neg {r['fail_neg']:.3f}   delta {r['delta']:+.3f}")
        good = [r["delta"] for r in gate if r["delta"] is not None]
        if good:
            print(f"[label] G2 median band delta {float(np.median(good)):+.3f} "
                  f"({sum(x > 0 for x in good)}/{len(good)} bands positive). A delta at or below "
                  "zero means the label is reproducing state difficulty and conditioning on it "
                  "cannot work.")

    # WRITTEN IN BOTH BRANCHES, and at THIS indentation deliberately. This block used to sit
    # inside the `else` above, so a `--quality_only` fit skipped the G2 gate and silently
    # skipped the WRITE with it: the run printed every statistic, returned 0, and left no
    # advantage.npz -- which a downstream fine-tune then either refused or, worse, satisfied
    # from a stale file. `meta`'s own `g2_skipped` field is the evidence it was always meant to
    # run without the gate. Pinned by test_label_advantage_writes_under_quality_only.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    meta = {"n": int(args.n), "variant": args.variant, "percentile": float(args.percentile),
            "null_spread": float(args.null_spread), "num_folds": folds,
            "w_fail": float(args.w_fail), "progress_bins": int(args.progress_bins),
            "speed_bins": int(args.speed_bins), "rollout": args.rollout,
            "tail_chunks": int(args.tail_chunks),
            "predictions": os.path.abspath(args.predictions),
            "counts": counts, "per_family": {str(k): v for k, v in per_family.items()},
            "g2_bands": gate,
            "g2_skipped": ("quality_only" if quality_only else "")}
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
