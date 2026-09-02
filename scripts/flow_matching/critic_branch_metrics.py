#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""The counterfactual half: can the critic order SIBLING chunks, and against what ceiling?

``transfer/sim/branch_probe.py`` produces the only data in this stack that contains a real
counterfactual -- K chunks drawn from ONE state, each rolled to completion, so each carries a
realized outcome the live loop can never observe for more than one of them.  This reads that
file and answers the questions ``--stage report`` does not.

**Four things it adds, each because the existing report cannot support a conclusion without
it.**

1. **A LABEL-NOISE CEILING.**  Common random numbers pair the siblings but do not remove label
   noise: two identical chunks would still land differently under a different continuation
   draw.  With a second, independently-keyed continuation on every sibling (which is what the
   ``second_seed`` change buys), the group's outcomes can be ranked twice from independent
   noise, and the correlation between those two rankings is the highest correlation ANY
   predictor could achieve.  A critic at tau = 0.10 against a ceiling of 0.12 is nearly
   perfect; against a ceiling of 0.90 it is nearly blind.  The same number is not
   interpretable without it.
2. **Spearman with a WITHIN-GROUP permutation null.**  The report's rank statistic is
   documented as Kendall tau-b and computes Goodman-Kruskal gamma, which differ on every group
   with a tied target -- i.e. on every survival group.  The null here permutes predictions
   WITHIN a group, which preserves the group structure and destroys only the association being
   tested.  A global label shuffle is vacuous: it has expectation 0.5 analytically.
3. **The arrival-inside-the-chunk correction.**  ``branches.h5``'s ``failed`` column is True
   whenever the branch terminated inside its own 0.24 s chunk, INCLUDING when it terminated by
   reaching the goal.  Left uncorrected it inflates the failure rate and can make a group read
   as outcome-mixed when every sibling in fact succeeded.
4. **The AWR question, asked directly.**  Selection asks whether ``argmax`` picks a good
   sibling.  Advantage weighting asks something weaker and more forgiving: whether the
   weighted average outcome ``sum_k w_k y_k / sum_k w_k`` beats the unweighted mean.  A critic
   can fail the first and pass the second, and that is exactly the hypothesis under test.

**Each head is judged against its OWN target**, never against episode success in general: the
survival head against the branch's realized fail/succeed, the quality head against the realized
normalized return over its continuation.  Comparing a windowed discounted tracking return to a
binary outcome would score the head on a question it was never asked.

Usage::

    python scripts/flow_matching/critic_branch_metrics.py \\
        --branches logs/value/bprobe_t6_s16/branches.h5 [more.h5 ...] \\
        --out logs/value/critic_eval_v4/branch_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np


def _decode(a: np.ndarray) -> np.ndarray:
    """Bytes-or-str object column to a str array.

    Args:
        a: The raw column.

    Returns:
        ``[n]`` of ``str``.
    """
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load_branches(paths) -> tuple[dict, dict]:
    """Read one or more ``branches.h5`` files, correcting the arrival-as-failure defect.

    Args:
        paths: Files to read.  Group ids are made unique per file, so several cells pool
            without their ``(seed, replan)`` keys colliding.

    Returns:
        ``(columns, meta)``.  ``columns["failed"]`` is the CORRECTED outcome; the raw column
        is kept as ``failed_raw`` so the size of the correction is reportable rather than
        silently applied.

    Raises:
        SystemExit: If a file predates ``terminal_in_branch``, since the correction cannot be
            applied to it and pooling it with corrected files would mix two definitions.
    """
    cols: dict[str, list] = {}
    meta: dict[str, list] = {"files": [], "attrs": []}
    base = 0
    for fi, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            d = {k: f[k][:] for k in f.keys()}
            at = {k: (f.attrs[k].item() if hasattr(f.attrs[k], "item") else f.attrs[k])
                  for k in f.attrs}
        if "terminal_in_branch" not in d:
            raise SystemExit(
                f"{p} predates the `terminal_in_branch` column, so a branch that ARRIVED "
                "inside its own chunk cannot be told from one that failed there. Pooling it "
                "with corrected files would mix two definitions of `failed`; re-run it.")
        bverd = _decode(d["branch_verdict"])
        term = d["terminal_in_branch"].astype(bool)
        arrived = term & (bverd == "success")
        d["failed_raw"] = d["failed"].astype(bool)
        d["failed"] = d["failed_raw"] & ~arrived
        d["arrived_in_branch"] = arrived
        if "failed_b" in d:
            bverd_b = _decode(d.get("cont_verdict_b", np.array([""] * d["cand"].size)))
            d["failed_b_raw"] = d["failed_b"].astype(bool)
            # The second continuation is rolled only where the branch did NOT terminate in its
            # own chunk, so no arrival correction applies; what needs marking is where it was
            # rolled at all.
            d["has_second"] = np.array([v != "" for v in bverd_b])
        # One group per (file, seed, replan) so pooled cells cannot collide.
        d["group"] = (base + d["seed"].astype(np.int64) * 100000
                      + d["replan"].astype(np.int64))
        d["cell"] = np.full(d["cand"].size, fi, dtype=np.int64)
        base += 10 ** 9
        for k, v in d.items():
            cols.setdefault(k, []).append(v)
        meta["files"].append(os.path.abspath(p))
        meta["attrs"].append({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                              for k, v in at.items()})
    # A COLUMN MUST COME FROM EVERY FILE OR IT IS DROPPED. Concatenating one that only some
    # files carry silently produces an array SHORTER than `group`, and numpy then broadcasts
    # or misaligns it against the rest of the table rather than raising -- so a run pooled
    # with an older file would have its second-seed columns quietly attributed to the wrong
    # branches. Dropping is announced, because a silently absent column changes which
    # statistics appear at all.
    n_files = len(meta["files"])
    out, dropped = {}, []
    for k, v in cols.items():
        if len(v) != n_files:
            dropped.append(k)
            continue
        try:
            out[k] = np.concatenate(v)
        except ValueError:
            dropped.append(k)
    if dropped:
        print(f"[branch] dropped {sorted(dropped)}: present in only some of the "
              f"{n_files} file(s), so the columns cannot be aligned to the pooled rows.")
    return out, meta


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with ties averaged, NaN when either side is constant.

    Args:
        a: ``[n]``.
        b: ``[n]``.

    Returns:
        The coefficient, or NaN.
    """
    if a.size < 2:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(x.size, dtype=np.float64)
        s = x[order]
        i = 0
        rr = np.arange(1, x.size + 1, dtype=np.float64)
        while i < s.size:
            j = i
            while j + 1 < s.size and s[j + 1] == s[i]:
                j += 1
            rr[i:j + 1] = rr[i:j + 1].mean()
            i = j + 1
        r[order] = rr
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def grouped_rank(pred: np.ndarray, target: np.ndarray, group: np.ndarray, *,
                 n_perm: int = 2000, seed: int = 0) -> dict:
    """Mean within-group Spearman, with a within-group permutation null.

    Groups whose target is constant carry no information about any ordering and are counted
    but excluded from the mean -- their Spearman is undefined, not zero, and averaging a zero
    in would drag every result toward "no signal" in proportion to how easy the cell was.

    Args:
        pred: ``[n]`` critic score, oriented so HIGHER means a BETTER outcome.
        target: ``[n]`` realized outcome, higher is better.
        group: ``[n]`` group id.
        n_perm: Permutation draws for the null.
        seed: RNG seed.

    Returns:
        ``n_groups``, ``n_informative``, ``rho`` (mean over informative groups), the null's
        ``rho_null_mean`` / ``rho_null_sd``, a two-sided ``p_value``, ``best_pick`` (the rate
        at which the argmax sibling is at or above its group's median target) and its null.
    """
    keys, inv = np.unique(group, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(keys.size)]
    rows, picks = [], []
    for ix in idx:
        if ix.size < 2 or np.all(target[ix] == target[ix][0]):
            continue
        rows.append(_spearman(pred[ix], target[ix]))
        t = target[ix]
        picks.append(float(t[int(np.argmax(pred[ix]))] >= np.median(t)))
    rows = np.asarray([r for r in rows if np.isfinite(r)], dtype=np.float64)
    picks = np.asarray(picks, dtype=np.float64)
    if rows.size == 0:
        return {"n_groups": int(keys.size), "n_informative": 0}

    rng = np.random.default_rng(seed)
    null_rho = np.empty(n_perm, dtype=np.float64)
    null_pick = np.empty(n_perm, dtype=np.float64)
    inf_idx = [ix for ix in idx if ix.size >= 2 and not np.all(target[ix] == target[ix][0])]
    for b in range(n_perm):
        rr, pp = [], []
        for ix in inf_idx:
            pr = rng.permutation(pred[ix])
            v = _spearman(pr, target[ix])
            if np.isfinite(v):
                rr.append(v)
            t = target[ix]
            pp.append(float(t[int(np.argmax(pr))] >= np.median(t)))
        null_rho[b] = np.mean(rr) if rr else np.nan
        null_pick[b] = np.mean(pp) if pp else np.nan
    obs = float(rows.mean())
    # The (b + 1) / (m + 1) correction. Without it a statistic no permutation matched is
    # reported as p = 0.000, which is not a probability the test can produce: the observed
    # arrangement is itself one of the permutations, so the smallest attainable value is
    # 1 / (m + 1). Quoting an exact zero from 2000 draws overstates the evidence.
    p = float((1 + (np.abs(np.nan_to_num(null_rho, nan=0.0)) >= abs(obs)).sum())
              / (n_perm + 1))
    return {"n_groups": int(keys.size), "n_informative": int(rows.size),
            "rho": obs, "rho_sem": float(rows.std(ddof=1) / np.sqrt(rows.size))
            if rows.size > 1 else float("nan"),
            "rho_null_mean": float(np.nanmean(null_rho)),
            "rho_null_sd": float(np.nanstd(null_rho)),
            "p_value": p,
            "best_pick": float(picks.mean()),
            "best_pick_null_mean": float(np.nanmean(null_pick)),
            "best_pick_null_sd": float(np.nanstd(null_pick))}


def label_ceiling(target_a: np.ndarray, target_b: np.ndarray, group: np.ndarray,
                  usable: np.ndarray) -> dict:
    """The highest within-group rank correlation ANY predictor could reach.

    Two independently-keyed continuations of the SAME sibling give two realized outcomes that
    differ only by continuation noise.  Ranking a group by one and correlating against the
    other bounds what any predictor can do, because the target it is scored against still
    carries one draw of that noise.

    **This is simultaneously the MONTE-CARLO ADVANTAGE's own score, and that is not a
    coincidence.**  Within one group every sibling shares the same parent state, so ``V(s_t)``
    is a CONSTANT and ``A_MC = G_t - V(s_t)`` orders the siblings exactly as the realized
    return does.  The MC advantage is therefore optimal by construction at the chunk level --
    it cannot be beaten by any function of the state -- and the only thing that can limit it
    is how noisy one realized return is as a measure of the chunk that preceded it.  That is
    what this measures.  Read against the critic-bootstrapped ``A_TD``, whose action-dependence
    runs entirely through ``V(s_{t+1})`` and is therefore capped by the critic's RESOLUTION,
    the two numbers answer the two halves of "which estimator is worth using".

    **Two ceilings, and they are not the same number.**  Ranking draw A against draw B
    measures the RELIABILITY ``r`` of a single-draw outcome -- how much of its variance is the
    sibling's expected outcome rather than continuation noise.  A predictor that knew the
    expected outcome EXACTLY would not score ``r`` against a single draw, it would score
    ``sqrt(r)``: writing ``y = mu + e`` with independent noise, ``corr(y_A, y_B) = r`` while
    ``corr(mu, y_B) = sqrt(r)``.  So ``rho`` below is the reliability and ``rho_oracle`` is the
    ceiling a critic is actually competing against, which is the HIGHER of the two.  Quoting
    the reliability as the ceiling flatters any critic being measured against it.

    Args:
        target_a: ``[n]`` realized outcome under the first continuation.
        target_b: ``[n]`` realized outcome under the second.
        group: ``[n]`` group id.
        usable: ``[n]`` bool, rows where BOTH continuations exist.

    Returns:
        ``n_groups``, ``rho`` (the single-draw reliability) and its standard error,
        ``rho_oracle`` (``sqrt(rho)``, what a perfect predictor of the expected outcome would
        score against one draw), and ``best_pick`` -- the rate at which the sibling that
        looked best under one draw is at or above the median under the other.
    """
    rows, picks = [], []
    keys, inv = np.unique(group, return_inverse=True)
    for i in range(keys.size):
        ix = np.flatnonzero((inv == i) & usable)
        if ix.size < 2:
            continue
        a, b = target_a[ix], target_b[ix]
        if np.all(a == a[0]) or np.all(b == b[0]):
            continue
        v = _spearman(a, b)
        if np.isfinite(v):
            rows.append(v)
        picks.append(float(b[int(np.argmax(a))] >= np.median(b)))
    rows = np.asarray(rows, dtype=np.float64)
    picks = np.asarray(picks, dtype=np.float64)
    if rows.size == 0:
        return {"n_groups": 0}
    r = float(rows.mean())
    return {"n_groups": int(rows.size), "rho": r,
            "rho_sem": float(rows.std(ddof=1) / np.sqrt(rows.size)) if rows.size > 1
            else float("nan"),
            "rho_oracle": float(np.sqrt(r)) if r > 0 else float("nan"),
            "best_pick": float(picks.mean()) if picks.size else float("nan")}


def by_spread(pred: np.ndarray, target: np.ndarray, group: np.ndarray, *,
              edges=(0.0, 0.05, 0.15, 1e9), n_perm: int = 500, seed: int = 0) -> list:
    """The ranking statistic stratified by how much the siblings' outcomes ACTUALLY differed.

    Groups whose realized outcomes barely differ cannot be ordered by anything, so pooling them
    with the rest dilutes the measurement toward chance in proportion to how easy the cell was.
    The meaningful test is the high-spread stratum, and the low-spread one is a sanity check
    that should sit near zero for any estimator.

    Args:
        pred: ``[n]`` score, higher is better.
        target: ``[n]`` realized outcome, higher is better.
        group: ``[n]`` group id.
        edges: Bin edges on the within-group ``max - min`` of ``target``.
        n_perm: Permutation draws per stratum.
        seed: RNG seed.

    Returns:
        One :func:`grouped_rank` result per stratum, each carrying its bin bounds.
    """
    keys, inv = np.unique(group, return_inverse=True)
    spread = np.zeros(keys.size, dtype=np.float64)
    for i in range(keys.size):
        t = target[inv == i]
        spread[i] = float(t.max() - t.min()) if t.size else 0.0
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (spread >= lo) & (spread < hi)
        m = sel[inv]
        if m.sum() < 4:
            continue
        r = grouped_rank(pred[m], target[m], group[m], n_perm=n_perm, seed=seed)
        r["spread_lo"], r["spread_hi"] = float(lo), (float(hi) if hi < 1e8 else None)
        out.append(r)
    return out


def head_to_head(pred: np.ndarray, target_a: np.ndarray, target_b: np.ndarray,
                 group: np.ndarray, usable: np.ndarray, *, n_perm: int = 2000,
                 seed: int = 0) -> dict:
    """The critic-bootstrapped and Monte-Carlo advantages, on the SAME independent target.

    The two estimators are otherwise scored against different things -- the critic against the
    draw it was never shown, the MC advantage against a second draw -- which makes their
    numbers look comparable while they are not.  Here both are ranked against ``target_b``,
    on the groups where a second continuation exists, so the comparison is like for like.

    * ``critic`` is ``V(s_{t+1})``, whose within-group variation IS the action-dependent term
      of ``A_TD``.
    * ``mc`` is the FIRST draw's realized outcome, which within a group is ``A_MC`` exactly
      (the baseline ``V(s_t)`` is a shared constant and drops out of any ranking).

    Args:
        pred: ``[n]`` critic score at the branch's end state, higher is better.
        target_a: ``[n]`` realized outcome under the first continuation.
        target_b: ``[n]`` realized outcome under the second.
        group: ``[n]`` group id.
        usable: ``[n]`` bool, rows where BOTH continuations exist.
        n_perm: Permutation draws for each null.
        seed: RNG seed.

    Returns:
        ``{"critic": ..., "mc": ...}``, each a :func:`grouped_rank` result on the same groups.
    """
    m = np.asarray(usable, dtype=bool)
    if not m.any():
        return {"critic": {"n_groups": 0, "n_informative": 0},
                "mc": {"n_groups": 0, "n_informative": 0}}
    return {"critic": grouped_rank(pred[m], target_b[m], group[m],
                                   n_perm=n_perm, seed=seed),
            "mc": grouped_rank(target_a[m], target_b[m], group[m],
                               n_perm=n_perm, seed=seed)}


def awr_gain(pred: np.ndarray, target: np.ndarray, group: np.ndarray, *,
             betas=(0.1, 0.3, 1.0, 3.0)) -> list:
    """What advantage WEIGHTING (rather than selection) buys, per temperature.

    The weighted mean outcome inside a group against the unweighted one is exactly what an AWR
    update does to that group's contribution, and unlike ``argmax`` it degrades gracefully: a
    weakly informative score still shifts the mean a little in the right direction.  The oracle
    (``max``) and the floor (``mean``) bracket it.

    Args:
        pred: ``[n]`` critic score, higher is better.
        target: ``[n]`` realized outcome, higher is better.
        group: ``[n]`` group id.
        betas: Temperatures, applied to the score standardized WITHIN its group.

    Returns:
        One row per beta with ``weighted``, ``uniform``, ``oracle`` and ``captured`` -- the
        fraction of the oracle-minus-uniform gap the weighting actually closes.
    """
    keys, inv = np.unique(group, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(keys.size)]
    idx = [ix for ix in idx if ix.size >= 2 and not np.all(target[ix] == target[ix][0])]
    out = []
    for beta in betas:
        w_mean, u_mean, o_mean = [], [], []
        for ix in idx:
            s = pred[ix]
            sd = s.std()
            z = (s - s.mean()) / (sd if sd > 0 else 1.0)
            w = np.exp(np.clip(z / beta, -50, 50))
            t = target[ix]
            w_mean.append(float((w * t).sum() / w.sum()))
            u_mean.append(float(t.mean()))
            o_mean.append(float(t.max()))
        w_m, u_m, o_m = np.mean(w_mean), np.mean(u_mean), np.mean(o_mean)
        out.append({"beta": beta, "n_groups": len(idx), "weighted": float(w_m),
                    "uniform": float(u_m), "oracle": float(o_m),
                    "gain_over_uniform": float(w_m - u_m),
                    "captured": float((w_m - u_m) / (o_m - u_m)) if o_m > u_m
                    else float("nan")})
    return out


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


def analyse(D: dict, meta: dict, *, n_perm: int, seed: int,
            fold_pair: tuple | None = None) -> dict:
    """Every counterfactual statistic, for every critic in the file.

    Args:
        D: The pooled columns from :func:`load_branches`.
        meta: Provenance from the same call.
        n_perm: Permutation draws.
        seed: RNG seed.
        fold_pair: The two critic tags that are CROSS-FIT FOLDS OF ONE VARIANT, for the
            resolution statistic. ``None`` skips it rather than guessing a pair.

    Returns:
        The report dict, written verbatim to JSON by :func:`main`.
    """
    group = D["group"]
    failed = D["failed"]
    g_real = D["g_real"]
    names = sorted({k[2:] for k in D if k.startswith("p_") and k != "p_fail"})

    keys, inv = np.unique(group, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(keys.size)]
    surv_mixed = sum(1 for ix in idx if not np.all(failed[ix] == failed[ix][0]))
    g_ok = np.isfinite(g_real)
    g_spread = [float(g_real[ix][g_ok[ix]].max() - g_real[ix][g_ok[ix]].min())
                for ix in idx if g_ok[ix].sum() >= 2]

    rep: dict = {
        "provenance": meta,
        "n_branches": int(group.size), "n_groups": int(keys.size),
        # THE FIRST THING TO READ. Siblings are drawn from identical conditioning, so they are
        # similar by construction; if their realized outcomes barely differ then no critic
        # could order them and the finding is about the GENERATOR, not the critic.
        "realized_spread": {
            "survival_groups_with_disagreement": surv_mixed,
            "survival_frac_informative": float(surv_mixed / max(keys.size, 1)),
            "quality_spread_median": float(np.median(g_spread)) if g_spread else float("nan"),
            "quality_spread_p90": float(np.percentile(g_spread, 90)) if g_spread
            else float("nan"),
            "failure_rate_corrected": float(failed.mean()),
            "failure_rate_raw": float(D["failed_raw"].mean()),
            "arrived_in_branch": int(D["arrived_in_branch"].sum()),
        },
        "heads": {},
    }

    # ---- the label-noise ceiling ---------------------------------------------------------
    if "has_second" in D:
        # BOTH draws must be finite. Checking only draw B lets a NaN draw-A row through, and
        # `_spearman` ranks NaN as the largest value rather than raising -- so the ceiling
        # would be computed partly from rows that carry no first outcome at all.
        has2 = (D["has_second"] & np.isfinite(g_real)
                & np.isfinite(D.get("g_real_b", np.full(group.size, np.nan))))
        rep["label_ceiling"] = {
            "survival": label_ceiling(-failed.astype(np.float64),
                                      -D["failed_b"].astype(np.float64), group,
                                      D["has_second"]),
            "quality": label_ceiling(g_real, D.get("g_real_b", np.full(group.size, np.nan)),
                                     group, has2),
            "note": "the highest within-group rho any predictor could reach, from two "
                    "independently-keyed continuations of the SAME sibling",
        }

    for nm in names:
        p, q = D[f"p_{nm}"], D.get(f"g_{nm}")
        block = {}
        # Survival head: score is -P(fail), target is -failed, so both are higher-is-better.
        #
        # MASKED ON `isfinite(p)`. A branch that terminated inside its own chunk is never
        # scored -- `branch_probe` only reads the critic when the chunk survived -- so its
        # `p_*` is NaN. Ranking with NaN present does not raise: `argsort` puts NaN LAST, so
        # the tie-averaged rank makes it the HIGHEST-scoring sibling, and the exact rows the
        # critic never saw become the ones it is credited with picking.
        fp = np.isfinite(p)
        block["survival"] = grouped_rank(-p[fp], -failed[fp].astype(np.float64), group[fp],
                                         n_perm=n_perm, seed=seed)
        block["survival_awr"] = awr_gain(-p[fp], -failed[fp].astype(np.float64), group[fp])
        block["survival_n_unscored"] = int((~fp).sum())
        if q is not None and np.isfinite(q).any():
            # Both sides must be finite: `g_real` is NaN wherever the branch terminated in its
            # own chunk, and `q` is NaN wherever the critic was never read.
            m = np.isfinite(q) & np.isfinite(g_real)
            block["quality"] = grouped_rank(q[m], g_real[m], group[m],
                                            n_perm=n_perm, seed=seed)
            block["quality_awr"] = awr_gain(q[m], g_real[m], group[m])
            block["quality_by_spread"] = by_spread(q[m], g_real[m], group[m],
                                                   n_perm=max(n_perm // 4, 100), seed=seed)
        # The comparison that decides which ESTIMATOR is worth using, as opposed to whether
        # this critic is any good: A_TD's action term against A_MC's, on ONE shared target.
        if "has_second" in D:
            gb = D.get("g_real_b", np.full(group.size, np.nan))
            use_q = (D["has_second"] & np.isfinite(g_real) & np.isfinite(gb)
                     & (np.isfinite(q) if q is not None else False))
            if q is not None and use_q.any():
                block["quality_head_to_head"] = head_to_head(
                    q, g_real, gb, group, use_q, n_perm=n_perm, seed=seed)
            use_s = D["has_second"]
            block["survival_head_to_head"] = head_to_head(
                -p, -failed.astype(np.float64), -D["failed_b"].astype(np.float64),
                group, use_s, n_perm=n_perm, seed=seed)
        rep["heads"][nm] = block

    # ---- the critic's own within-group spread against the fold disagreement --------------
    # A selector orders siblings by differences in the critic's output; if those differences
    # are smaller than how much two honest fits of the same critic disagree, no ordering it
    # produces can be trusted, however good its AVERAGE prediction is.
    #
    # The PAIR IS NAMED, never taken as "the first two". The tags are whatever the run's
    # --critic flags said, so a run carrying V_priv fold0, V_priv fold1 and a V_onboard fold
    # would, under any positional rule, eventually compare two DIFFERENT MODELS and report
    # their disagreement as this critic's estimation noise -- inflating the denominator with
    # an information difference and making the resolution look worse than it is.
    if fold_pair is not None:
        missing = [f for f in fold_pair if f"p_{f}" not in D]
        if missing:
            raise SystemExit(
                f"--fold_pair names {missing}, which this file does not carry (it has "
                f"{names}). The pair must be two CROSS-FIT FOLDS OF ONE VARIANT; comparing "
                "two different variants would report an information gap as estimation noise.")
        a, b = fold_pair
        for field, label in (("p_", "p_fail"), ("g_", "G_hat")):
            xa, xb = D.get(field + a), D.get(field + b)
            if xa is None or xb is None or not np.isfinite(xa).any():
                continue
            m = np.isfinite(xa) & np.isfinite(xb)
            # BOTH sides are a median ABSOLUTE PAIRWISE DIFFERENCE, deliberately.
            #
            # The obvious within-group statistic is the range, max - min over the K siblings,
            # and it is not comparable to a fold difference: a range over K draws is roughly
            # twice a typical pairwise gap and grows with K, so pairing it against |f0 - f1|
            # inflates the ratio ~2x at K=4 and makes the number depend on how many candidates
            # the probe happened to draw. Comparing like with like costs one extra line and
            # makes the SNR mean "how far apart two siblings look, against how far apart two
            # honest fits of the same model are about one sibling".
            def _median_pairwise(vals: np.ndarray) -> float:
                """Median |x_i - x_j| over the distinct pairs of one group."""
                d = np.abs(vals[:, None] - vals[None, :])
                iu = np.triu_indices(vals.size, k=1)
                return float(np.median(d[iu]))

            gaps = [_median_pairwise(xa[ix][m[ix]]) for ix in idx if m[ix].sum() >= 2]
            noise = float(np.median(np.abs(xa[m] - xb[m])))
            within = float(np.median(gaps)) if gaps else float("nan")
            rep.setdefault("critic_resolution", {})[label] = {
                "fold_disagreement_median_abs": noise,
                "within_group_median_pairwise": within,
                "n_groups": len(gaps),
                "snr": (float(within / noise) if gaps and noise > 0 else float("nan")),
                "folds": [a, b],
                "note": "both sides are median |pairwise difference|, so the ratio does not "
                        "depend on K; the fold gap UNDERSTATES the critic's true error (the "
                        "two folds share a frozen trunk and an architecture), so this SNR is "
                        "an upper bound on the resolving power.",
            }
    return rep


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit status.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--branches", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fold_pair", nargs=2, default=None, metavar=("F0", "F1"),
                   help="The two --critic TAGS that are cross-fit folds of ONE variant. "
                        "Used only for the resolution statistic, which compares the "
                        "critic's within-group spread against how much two honest fits "
                        "of the SAME model disagree.")
    p.add_argument("--n_perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    D, meta = load_branches(args.branches)
    rep = analyse(D, meta, n_perm=args.n_perm, seed=args.seed,
                  fold_pair=(tuple(args.fold_pair) if args.fold_pair else None))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(_jsonable(rep), fh, indent=2)
    print(json.dumps(_jsonable({k: v for k, v in rep.items() if k != "provenance"}),
                     indent=2)[:4000])
    print(f"\n[branch] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
