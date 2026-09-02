# Copyright 2026 Zachary Olkin. All rights reserved.

"""The offline half of ``critic_advantage_eval.py``: everything computable without a simulator.

Split out because it imports no torch and no IsaacLab -- it reads ``scores.npz`` and nothing
else -- so the analysis can be re-run and iterated on in seconds on any box, which the scoring
pass (12 GB of exteroception through a 27M-parameter encoder) cannot.

**What offline data can and cannot answer, stated once.**  There is exactly one action per
state in a recording, so no offline statistic can say whether a DIFFERENT chunk from the same
state would have done better -- which is the question advantage weighting is ultimately asking.
What offline data CAN settle is everything that has to be true first:

* the critic predicts held-out outcomes at all, and not by memorising episodes (Steps 1-2);
* the advantage varies within an episode, so weighting by it differs from weighting by the
  episode's outcome (the variance decomposition);
* that within-episode variation exceeds the disagreement between two honest estimates of it
  (the reliability floor);
* blame lands near the failure rather than smearing over the episode (credit localisation);
* the implied weights leave a usable dataset (effective sample size).

Any of those failing kills the idea without a simulator being started.  All of them passing
does not establish it, which is what the counterfactual arm is for.
"""

from __future__ import annotations

import json
import os

import numpy as np

from robot_rl.flow_matching import critic_eval as ce
from robot_rl.flow_matching import value_gates as vg

#: GAE decays swept; 0 is the one-chunk TD advantage, 1 the Monte-Carlo one.
LAMBDAS = (0.0, 0.5, 0.8, 0.95, 1.0)

#: AWR temperatures, applied to the standardized advantage.
BETAS = (0.1, 0.3, 1.0, 3.0)

#: Progress buckets used everywhere a metric is reported against time.
N_BUCKETS = 10


def _categorical_r2(y: np.ndarray, *keys: np.ndarray) -> float:
    """``R^2`` of predicting ``y`` by the mean of its cell in a categorical cross.

    The saturated fit for those factors, so it is an upper bound on what ANY model of them
    could explain -- which is what makes "the advantage is mostly a restatement of the episode
    outcome" a falsifiable claim rather than an impression.

    Args:
        y: ``[n]`` values.
        *keys: ``[n]`` integer factors to cross.

    Returns:
        The coefficient of determination.
    """
    key = np.zeros(y.shape[0], dtype=np.int64)
    for k in keys:
        _, inv = np.unique(k, return_inverse=True)
        key = key * (inv.max() + 1) + inv
    _, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    pred = (np.bincount(inv, weights=y) / cnt)[inv]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def _bucket_metric(score: np.ndarray, target: np.ndarray, bucket: np.ndarray, *,
                   binary: bool) -> dict:
    """A discrimination metric per progress bucket.

    Args:
        score: ``[n]`` critic reading, oriented so HIGHER means a better outcome.
        target: ``[n]`` realized outcome -- bool ``failed`` when ``binary``, else the return.
        bucket: ``[n]`` bucket index.
        binary: Whether to report AUC (against ``failed``) or Spearman (against the return).

    Returns:
        ``{"bucket": [...], "n": [...], "value": [...], "base": [...]}``.
    """
    out = {"bucket": [], "n": [], "value": [], "base": []}
    for b in range(N_BUCKETS):
        m = bucket == b
        if m.sum() < 50:
            continue
        out["bucket"].append(b)
        out["n"].append(int(m.sum()))
        if binary:
            # `score` is oriented high-is-good, so failure is the NEGATIVE class and the AUC
            # of predicting it is taken on the negated score.
            out["value"].append(vg.auc(-score[m], target[m].astype(bool)))
            out["base"].append(float(target[m].astype(bool).mean()))
        else:
            out["value"].append(vg.spearman(score[m], target[m]))
            out["base"].append(float(np.mean(target[m])))
    return out


def _head_values(S, variant: str, head: str) -> dict:
    """Assemble one head's value readings, realized return and reward on one scale.

    Args:
        S: The loaded ``scores.npz``.
        variant: ``"V_priv"`` or ``"V_onboard"``.
        head: ``"survival"`` or ``"quality"``.

    Returns:
        A dict with ``v_f0``/``v_f1`` (per-fold readings), ``g`` (realized return),
        ``reward``, ``gamma``, ``normalized`` and ``valid``, every array on the head's own
        scale and oriented so HIGHER IS BETTER.

    Raises:
        KeyError: If the requested head was never fitted for this variant.
    """
    if head == "survival":
        # V = -P(fail). With a terminal-only unit failure cost the return is binary and the
        # discount is 1: nothing accrues between the start and the outcome, so discounting
        # would only shrink an already-constant quantity.
        return {"v_f0": -S[f"p_{variant}_f0"], "v_f1": -S[f"p_{variant}_f1"],
                "g": -S["failed"].astype(np.float64),
                "reward": np.zeros(S["failed"].shape[0], dtype=np.float64),
                "gamma": 1.0, "normalized": False,
                "valid": np.ones(S["failed"].shape[0], dtype=bool)}
    q0, q1 = S[f"q_{variant}_f0"], S[f"q_{variant}_f1"]
    # BOTH folds are required. A variant fitted with a quality head has it in both, so one
    # fold missing it means the two checkpoints are not the pair they claim to be -- and
    # checking only fold 0 would let fold 1's all-NaN column through into every cross-fit
    # reading, where `vg.spearman` and `vg.auc` return finite garbage rather than raising.
    if not (np.isfinite(q0).any() and np.isfinite(q1).any()):
        raise KeyError(f"{variant} has no quality head in this fit "
                       f"(fold0 finite: {bool(np.isfinite(q0).any())}, "
                       f"fold1 finite: {bool(np.isfinite(q1).any())})")
    return {"v_f0": q0, "v_f1": q1, "g": S["g"].astype(np.float64),
            "reward": S["chunk_reward"].astype(np.float64),
            "gamma": float(S["gamma"]), "normalized": True,
            # `isfinite` on BOTH readings as well as the target: a row the critic could not
            # score is not a row with a value, and a NaN reaching a rank statistic is sorted
            # to the END rather than rejected.
            "valid": (S["g_valid"].astype(bool) & np.isfinite(q0) & np.isfinite(q1)
                      & np.isfinite(S["g"]))}


def _cross_fit(v_f0: np.ndarray, v_f1: np.ndarray, held_out_by: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Split two per-fold readings into the honest one and the memorising one.

    Args:
        v_f0: ``[n]`` fold-0 reading.
        v_f1: ``[n]`` fold-1 reading.
        held_out_by: ``[n]`` which fold held this row's episode out, or ``-1`` on a tile
            neither fold ever trained on.

    Returns:
        ``(v_out, v_in)``. On a held-out TILE both folds are honest, so ``v_out`` is their
        mean -- the fit's own convention -- and ``v_in`` is NaN, because no fold trained on
        that row and there is no in-fold reading to name.
    """
    v_out = np.where(held_out_by == 0, v_f0,
                     np.where(held_out_by == 1, v_f1, 0.5 * (v_f0 + v_f1)))
    v_in = np.where(held_out_by == 0, v_f1,
                    np.where(held_out_by == 1, v_f0, np.nan))
    return v_out, v_in


def _estimators(v: np.ndarray, hv: dict, S) -> dict[str, np.ndarray]:
    """Every advantage estimator for one head, from one value reading.

    Args:
        v: ``[n]`` the value reading to bootstrap on (always a CROSS-FIT one).
        hv: The head's value bundle from :func:`_head_values`.
        S: The loaded ``scores.npz``.

    Returns:
        Estimator name -> ``[n]`` advantage.
    """
    uid, ri = S["episode_uid"], S["replan_index"]
    failed = S["failed"].astype(bool)
    g, r, gam = hv["g"], hv["reward"], hv["gamma"]
    est: dict[str, np.ndarray] = {}
    # The bar to beat: the trajectory-level filtered-BC weight, which needs no critic at all.
    #
    # Centred on the VALID rows only. `normalized_returns` leaves `g` at 0 for an episode with
    # no usable reward at all (5.6% of rows on v4_big), and those zeros sit below every real
    # return in [0, 1] -- so a mean taken over them would shift the whole baseline downward
    # and make `A_base` systematically positive for reasons that have nothing to do with any
    # episode's quality. The rows themselves are dropped from every statistic; only their
    # effect on this one scalar had to be removed by hand.
    ok = hv["valid"]
    est["A_base"] = g - (g[ok].mean() if ok.any() else g.mean())
    est["A_MC"] = g - v
    # The SAME validity mask `normalized_returns` was given. Without it the terminal closes on
    # the placeholder zero that `chunk_rewards_from_track_err` leaves on each episode's last
    # row, which biases every lam > 0 GAE in a way confounded with the episode outcome.
    delta, _ = ce.td_residuals(v, r, uid, ri, failed, gamma=gam,
                               normalized=hv["normalized"], valid=ok)
    for lam in LAMBDAS:
        est[f"A_GAE{lam:g}"] = ce.gae_advantage(delta, uid, ri, gamma=gam, lam=lam, valid=ok)
    est["A_TD"] = est["A_GAE0"]
    return est


def _weight_table(a: np.ndarray, S, prog_bucket: np.ndarray, near_fail: np.ndarray) -> list:
    """AWR weight statistics over the beta grid, clipped and unclipped.

    Args:
        a: ``[n]`` advantages.
        S: The loaded ``scores.npz``.
        prog_bucket: ``[n]`` progress bucket.
        near_fail: ``[n]`` bool, within the last three chunks before a realized failure.

    Returns:
        One row per (beta, clipped) with ESS and where the mass lands.
    """
    failed = S["failed"].astype(bool)
    rows = []
    for beta in BETAS:
        for clip in (None, 20.0):
            w = ce.awr_weights(a, beta=beta, w_max=clip)
            tot = float(w.sum())
            by_bucket = [float(w[prog_bucket == b].sum() / tot) if tot > 0 else float("nan")
                         for b in range(N_BUCKETS)]
            # THE ESS IN EPISODES, beside the one in rows. ~22 windows of an episode carry the
            # same terminal label and are adjacent states, so a row-level ESS overstates the
            # independent units an update actually sees -- and the two do not move together:
            # at beta 0.3 the quality head's A_MC keeps 1.5% of rows but 5.8% of episodes,
            # while at beta 1.0 it keeps 33.5% of rows and 44.4% of episodes.
            _, inv_ep = np.unique(S["episode_uid"], return_inverse=True)
            w_ep = np.bincount(inv_ep, weights=w)
            rows.append({
                "beta": beta, "w_max": clip,
                "ess_frac": float(ce.effective_sample_size(w) / w.size),
                "ess_frac_episodes": float(ce.effective_sample_size(w_ep) / w_ep.size),
                "n_episodes": int(w_ep.size),
                "mass_on_success": float(w[~failed].sum() / tot) if tot > 0 else float("nan"),
                "frac_rows_success": float((~failed).mean()),
                "mass_near_failure": float(w[near_fail].sum() / tot) if tot > 0 else float("nan"),
                "frac_rows_near_failure": float(near_fail.mean()),
                "mass_by_progress": by_bucket,
                "max_weight_share": float(w.max() / tot) if tot > 0 else float("nan"),
            })
    return rows


def _demean_by(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """``x`` with each group's own mean removed.

    Args:
        x: ``[n]`` values.
        group: ``[n]`` group id.

    Returns:
        ``[n]`` the within-group deviations.
    """
    _, inv = np.unique(group, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, weights=np.asarray(x, dtype=np.float64)) / cnt
    return np.asarray(x, dtype=np.float64) - means[inv]


def _label_novelty(a: np.ndarray, failed: np.ndarray, family: np.ndarray,
                   percentile: float = 35.0) -> dict:
    """How much a filtered-BC set built from ``a`` would DIFFER from one built from the outcome.

    This is the "is it worth trying" question in its most concrete form.  Filtered BC on the
    episode outcome -- keep every chunk from an episode that succeeded -- needs no critic and
    has been run.  If the advantage's per-family percentile cut selects nearly the same set,
    a fine-tune on it is a re-run of that experiment with extra steps.  What makes the attempt
    worth the cycles is the chunks the two rules DISAGREE about: positives drawn from failed
    episodes, and negatives drawn from successful ones.

    The threshold is the pipeline's own -- :func:`advantage.binarize`'s per-family percentile
    -- because the class balance is fixed by that flag no matter what the critic says, so a
    novelty measured at any other cut would be about a threshold nobody uses.

    Args:
        a: ``[n]`` advantages.
        failed: ``[n]`` bool, the episode's realized outcome.
        family: ``[n]`` family code, since the cut is per family.
        percentile: The cut, matching ``label_advantage.py --percentile``.

    Returns:
        The positive fraction, the two disagreement rates, and ``novel_frac`` -- the share of
        all rows the two rules classify differently, which is the fraction of the fine-tune's
        training signal that is genuinely new.
    """
    pos = np.zeros(a.shape[0], dtype=bool)
    for fam in np.unique(family):
        sel = family == fam
        if not sel.any():
            continue
        pos[sel] = a[sel] >= np.percentile(a[sel], percentile)
    base_pos = ~failed                     # the trajectory-level rule, which needs no critic
    return {"percentile": float(percentile),
            "positive_frac": float(pos.mean()),
            "outcome_positive_frac": float(base_pos.mean()),
            "positives_from_failed_episodes": float((pos & failed).sum() / max(pos.sum(), 1)),
            "negatives_from_survived_episodes": float(((~pos) & ~failed).sum()
                                                      / max((~pos).sum(), 1)),
            "novel_frac": float((pos != base_pos).mean()),
            "agreement": float((pos == base_pos).mean())}


def _credit_offsets(a: np.ndarray, S, valid: np.ndarray | None = None) -> dict:
    """Where in a FAILED episode the most negative advantage lands, relative to the failure.

    ``first_fall_step`` is a control STEP and ``replan_index`` a chunk, so the offset is
    reported in chunks after dividing by the re-plan cadence -- the two are not the same unit
    and an offset quoted in the wrong one is off by a factor of twelve.

    Args:
        a: ``[n]`` advantages.
        S: The loaded ``scores.npz``.

    Returns:
        Summary statistics of the offset, in chunks, plus the fraction landing on the LAST
        chunk (which is what a critic that has learned only "the episode is nearly over"
        produces).
    """
    uid, ri = S["episode_uid"], S["replan_index"]
    failed = S["failed"].astype(bool)
    ok = (np.ones(uid.shape[0], dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool))
    order, starts, ends = ce._episode_groups(uid, ri)
    offs, last_frac, flat = [], [], 0
    for s, e in zip(starts, ends):
        rows = order[s:e]
        # INVALID ROWS ARE DROPPED BEFORE THE ARGMIN, and the episode's end is then its last
        # VALID chunk. On the quality head the last row of every episode carries a placeholder
        # reward, so its advantage is a placeholder too -- and it is systematically extreme,
        # which is what manufactured a "blame lands on the last chunk" reading.
        rows = rows[ok[rows]]
        if rows.size < 3 or not failed[rows[0]]:
            continue
        av = a[rows]
        # A CONSTANT advantage localises nothing, and `argmin` on ties silently returns index
        # 0 -- which is the episode's FIRST chunk, so the offset comes back as minus the
        # episode length and reads as "blame lands at the start". `A_base` is exactly constant
        # within an episode on the survival head, so without this the baseline appears to have
        # a credit-assignment profile when by construction it has none.
        if av.max() - av.min() <= 0.0:
            flat += 1
            continue
        worst = int(ri[rows][int(np.argmin(av))])
        end = int(ri[rows].max())
        offs.append(worst - end)
        last_frac.append(worst == end)
    offs = np.asarray(offs, dtype=np.float64)
    if offs.size == 0:
        return {"n": 0, "n_constant_within_episode": flat}
    return {"n": int(offs.size), "n_constant_within_episode": flat,
            "median_chunks_before_end": float(np.median(offs)),
            "p25": float(np.percentile(offs, 25)), "p75": float(np.percentile(offs, 75)),
            "frac_on_last_chunk": float(np.mean(last_frac)),
            "frac_within_2_chunks": float(np.mean(offs >= -2))}


def stage_metrics(args) -> int:
    """Compute every offline metric and write ``metrics.json`` + ``advantages.npz``.

    Args:
        args: Parsed CLI arguments; only ``out``, ``seed`` and ``n_boot`` are read.

    Returns:
        Process exit status.
    """
    S = np.load(os.path.join(args.out, "scores.npz"), allow_pickle=True)
    if "recovery_ok" in S.files and not bool(S["recovery_ok"]):
        raise SystemExit(
            "this scores.npz records recovery_ok=False: the re-scored readings did not "
            f"reproduce the fit's own predictions ({S['recovery_diag']}). Every metric below "
            "would be about a model that is not the one the fit produced.")
    uid, ri = S["episode_uid"], S["replan_index"]
    failed = S["failed"].astype(bool)
    val_tile = S["val_tile"].astype(bool)
    held = S["held_out_by"]
    fam = S["family_code"]
    tile = S["tile"]
    n = uid.shape[0]

    prog = ce.normalized_progress(uid, ri)
    pb = np.clip((prog * N_BUCKETS).astype(int), 0, N_BUCKETS - 1)
    # Within the last three chunks of a failed episode.
    last_ri = np.zeros(n, dtype=np.int64)
    order, starts, ends = ce._episode_groups(uid, ri)
    for s, e in zip(starts, ends):
        rows = order[s:e]
        last_ri[rows] = ri[rows].max()
    near_fail = failed & (ri >= last_ri - 2)

    report: dict = {"n_windows": int(n), "n_episodes": int(np.unique(uid).size),
                    "n_failed_episodes": int(np.unique(uid[failed]).size),
                    "n_tiles": int(np.unique(tile).size),
                    "n_held_out_tiles": int(np.unique(tile[val_tile]).size),
                    "frac_rows_held_out_tile": float(val_tile.mean()),
                    "gamma_quality": float(S["gamma"]), "heads": {}}
    saved: dict[str, np.ndarray] = {}

    for variant in ("V_priv", "V_onboard"):
        if f"p_{variant}_f0" not in S.files:
            continue
        for head in ("survival", "quality"):
            try:
                hv = _head_values(S, variant, head)
            except KeyError:
                continue
            key = f"{variant}/{head}"
            print(f"[metrics] {key}", flush=True)
            v_out, v_in = _cross_fit(hv["v_f0"], hv["v_f1"], held)
            ok = hv["valid"]
            seen = ok & ~val_tile          # training tiles: an in-fold reading exists
            unseen = ok & val_tile         # held-out tiles: both folds honest

            block: dict = {}

            # ---- Step 1: memorisation -------------------------------------------------
            # Restricted to training tiles, the only rows where an in-fold reading exists.
            mem = {"scope": "training tiles only (a held-out tile has no in-fold reading)",
                   "n": int(seen.sum())}
            for nm, arr in (("out", v_out), ("in", v_in)):
                mem[f"spearman_{nm}"] = vg.spearman(arr[seen], hv["g"][seen])
                mem[f"auc_{nm}"] = vg.auc(-arr[seen], failed[seen])
            mem["spearman_gap"] = mem["spearman_in"] - mem["spearman_out"]
            mem["auc_gap"] = mem["auc_in"] - mem["auc_out"]
            # TILE-CLUSTERED intervals, not row-level ones. 294k windows are ~15k episodes are
            # 76 tiles, and windows within an episode share a label while episodes within a
            # tile share terrain -- so a row-level interval is far too tight (the eval harness
            # measured 14-70% actual coverage against a nominal 95% for exactly this reason).
            # The gap is resampled as ONE statistic so the two folds' shared draw cancels,
            # which a difference of two separately-bootstrapped intervals would not do.
            def _auc_gap(ix, _o=v_out, _i=v_in, _f=failed):
                """AUC(in) - AUC(out) on a resampled row set."""
                return vg.auc(-_i[ix], _f[ix]) - vg.auc(-_o[ix], _f[ix])

            rows_seen = np.flatnonzero(seen)
            pt, lo, hi = ce.clustered_bootstrap_ci(
                rows_seen, tile[rows_seen], statistic=_auc_gap,
                n_boot=int(args.n_boot), seed=int(args.seed))
            mem["auc_gap_ci"] = [pt, lo, hi]
            pt, lo, hi = ce.clustered_bootstrap_ci(
                rows_seen, tile[rows_seen],
                statistic=lambda ix: vg.auc(-v_out[ix], failed[ix]),
                n_boot=int(args.n_boot), seed=int(args.seed))
            mem["auc_out_ci"] = [pt, lo, hi]
            # The same, inside progress buckets, so a critic that has merely learned the
            # clock cannot pass by predicting time.
            mem["by_bucket_out"] = _bucket_metric(v_out[seen], hv["g"][seen], pb[seen],
                                                  binary=False)
            mem["by_bucket_in"] = _bucket_metric(v_in[seen], hv["g"][seen], pb[seen],
                                                 binary=False)
            block["memorization"] = mem

            # ---- Step 2: time-resolved discrimination ---------------------------------
            block["discrimination"] = {
                "auc_by_progress_seen_tiles": _bucket_metric(v_out[seen], failed[seen],
                                                             pb[seen], binary=True),
                "auc_by_progress_unseen_tiles": _bucket_metric(v_out[unseen], failed[unseen],
                                                               pb[unseen], binary=True),
                "spearman_g_by_progress_unseen": _bucket_metric(v_out[unseen], hv["g"][unseen],
                                                                pb[unseen], binary=False),
                "auc_at_first_chunk_unseen": vg.auc(-v_out[unseen & (ri == 0)],
                                                    failed[unseen & (ri == 0)]),
                "n_first_chunk_unseen": int((unseen & (ri == 0)).sum()),
            }

            # ---- the reliability floor -------------------------------------------------
            # Both folds are honest on a held-out tile, so their disagreement is how much two
            # equally valid fits of the same quantity differ. Nothing whose spread is smaller
            # than this can be ordered.
            block["reliability_value_unseen_tiles"] = ce.reliability(
                hv["v_f0"][unseen], hv["v_f1"][unseen])

            # ---- Step 3 + the variance decomposition -----------------------------------
            est_out = _estimators(v_out, hv, S)
            est_f0 = _estimators(hv["v_f0"], hv, S)
            est_f1 = _estimators(hv["v_f1"], hv, S)
            ests: dict = {}
            for name, a in est_out.items():
                a_ok = a[ok]
                vd = ce.variance_decomposition(a_ok, uid[ok])
                rel = ce.reliability(est_f0[name][unseen], est_f1[name][unseen])
                # How much of the advantage is a restatement of things needing no critic.
                r2_out = _categorical_r2(a_ok, failed[ok].astype(np.int64))
                r2_all = _categorical_r2(a_ok, failed[ok].astype(np.int64), fam[ok], pb[ok])
                ents = {
                    "mean": float(a_ok.mean()), "sd": float(a_ok.std()),
                    "frac_positive": float((a_ok > 0).mean()),
                    "var_frac_within_episode": vd["frac_within"],
                    "r2_from_episode_outcome": r2_out,
                    "r2_from_outcome_family_progress": r2_all,
                    "reliability_across_folds_unseen": rel["pearson"],
                    # The SAME coefficient after each episode's own mean is removed. The
                    # overall one is inflated by the easy between-episode signal -- both folds
                    # agree that a failing episode is worse than a clean one, which no
                    # filtered-BC cut needs a critic to know. What a per-chunk label actually
                    # rests on is the WITHIN-episode ordering, and this is whether two honest
                    # fits agree about it.
                    "reliability_within_episode_unseen": ce.reliability(
                        _demean_by(est_f0[name][unseen], uid[unseen]),
                        _demean_by(est_f1[name][unseen], uid[unseen]))["pearson"],
                    "spearman_with_A_base": vg.spearman(a_ok, est_out["A_base"][ok]),
                    "credit": _credit_offsets(a, S, ok),
                    "label_novelty": _label_novelty(a_ok, failed[ok], fam[ok]),
                    "weights": _weight_table(a_ok, {k: S[k][ok] for k in
                                                    ("failed", "episode_uid")},
                                             pb[ok], near_fail[ok]),
                }
                # The within-episode signal fraction: of the variance that is NOT explained by
                # the episode, how much survives the fold disagreement. This is the number the
                # whole method rests on -- it is what a selector would have to order.
                #
                # BOTH SIDES ON THE SAME ROWS, and both WITHIN-EPISODE. The numerator used to
                # be the within-episode variance over every valid row while the denominator
                # was the fold disagreement on held-out tiles only, so the ratio mixed two row
                # sets; and the disagreement carries a per-episode component that the
                # numerator has already removed, which inflates the denominator. Restricting
                # to `unseen` and de-meaning both by episode makes it a like-for-like ratio.
                within_var = vd["within"]
                d_un = _demean_by(est_f0[name][unseen] - est_f1[name][unseen], uid[unseen])
                noise_var = 0.5 * float(np.var(d_un))
                sig_var = float(ce.variance_decomposition(
                    est_f0[name][unseen], uid[unseen])["within"])
                ents["within_episode_var"] = within_var
                ents["within_episode_var_unseen"] = sig_var
                ents["fold_disagreement_var_within_episode"] = noise_var
                ents["within_signal_to_noise"] = (float(sig_var / noise_var)
                                                  if noise_var > 0 else float("inf"))
                ests[name] = ents
            block["estimators"] = ests
            # ---- WHAT THE CRITIC CONTRIBUTES TO A MONTE-CARLO ADVANTAGE ------------------
            #
            # Under `A_MC = G_t - V(s_t)` the critic is a BASELINE, not the signal: the
            # action-dependence comes from the realized return and the critic only has to be
            # a good function of state. A baseline is judged on one thing -- how much of the
            # return's variance it removes -- and it is unbiased for any function of the state
            # alone, which V is by construction. So this is the critic's actual contribution
            # to advantage-weighted BC, and it is a different question from whether it can
            # resolve two actions at one state.
            a_base = est_out["A_base"][ok]
            a_mc = est_out["A_MC"][ok]
            vb, vm = float(a_base.var()), float(a_mc.var())
            block["baseline"] = {
                "sd_A_base": float(np.sqrt(vb)), "sd_A_MC": float(np.sqrt(vm)),
                "variance_reduction": float(1.0 - vm / vb) if vb > 0 else float("nan"),
                "note": "1 - Var(G - V) / Var(G - mean G): the fraction of the return's "
                        "variance the critic removes as an AWR baseline. Unbiased for any "
                        "state-only function, so this is gain with no cost in bias.",
            }
            report["heads"][key] = block
            for name, a in est_out.items():
                saved[f"{variant}__{head}__{name}"] = a.astype(np.float32)
            saved[f"{variant}__{head}__v_out"] = v_out.astype(np.float32)
            saved[f"{variant}__{head}__v_in"] = v_in.astype(np.float32)

    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(_jsonable(report), fh, indent=2)
    np.savez_compressed(os.path.join(args.out, "advantages.npz"),
                        episode_uid=uid, replan_index=ri, family_code=fam, tile=tile,
                        failed=failed, val_tile=val_tile, progress=prog, **saved)
    print(f"[metrics] wrote {os.path.join(args.out, 'metrics.json')}")
    return 0


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
