# Copyright 2026 Zachary Olkin. All rights reserved.

"""The three things worth asking of a trackability critic, and why aggregate AUC is not one.

A single held-out AUC over every recorded state is the obvious gate and it is close to
uninformative here.  Most states in a rollout are on flat approach where nothing has been
decided yet, so a critic that is right about those and blind at the decisive moment scores
well, and one that is uninformative early and sharp at the riser scores badly.  Worse, a high
aggregate AUC is exactly what a critic that has merely learned WHICH FAMILY it is looking at
would produce -- state difficulty, not the thing the advantage needs.

So:

1. :func:`calibration` -- **testable even where discrimination is impossible.** Bin states by
   predicted ``P(fail)`` and compare against the empirical failure frequency in each bin. A
   critic that says 0.2 and is right one time in five is working, and it is working even in a
   region where no critic could separate individual episodes. This is the check that does not
   presuppose predictability, which is why it comes first.

2. :func:`auc_by_phase` -- **discrimination as a function of PHASE, never in aggregate.** The
   healthy signature is a curve rising from ~0.5 early to high near the decisive moment. Flat
   at 0.5 everywhere is the failure. Flat at 0.5 only at the START is expected and correct:
   nothing has happened yet, so nothing is predictable, and a critic claiming otherwise on the
   first chunk of a stairs episode has learned the tile rather than the state.

3. :func:`lead_time` -- **the one that decides whether the method can work at all.** For each
   failure, how many chunks before the end does ``P(fail)`` rise and stay risen? If it only
   rises once the base is already past recovery, the advantage arrives AFTER the chunk that
   caused the failure and nothing is attributable to any action -- which would be a red flag
   pointing at bias rather than at sampling variance, since it would mean no chunk the
   generator emitted was identifiable as the cause. It also **sets n empirically**: the n-step
   advantage window should span the lead time, and measuring it beats guessing it from where
   falls happen to land.

   Measured against ``first_fall_step`` as well as against the end, because those are
   different questions. Rising before the episode ENDS is weak -- the fall predicate may have
   been crossed long before. Rising before the fall PREDICATE is the claim that matters.

The cheap version of (3) is worth running before any of this: plot ``P(fail)`` over a handful
of stairs episodes, overlay successes and failures, and look at where the two families
separate.  :func:`value_traces` returns exactly that.
"""

from __future__ import annotations

import numpy as np


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC, by the Mann-Whitney rank statistic.

    Exact, ties handled by mid-ranks, and no scikit-learn dependency -- this package is
    imported by scripts that run without one.

    Args:
        scores: ``[n]`` predictions; higher should mean more likely positive.
        positive: ``[n]`` boolean labels.

    Returns:
        The AUC, or NaN when either class is empty (there is nothing to separate, which is a
        different statement from 0.5).
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Mid-ranks for ties, or a critic that outputs a constant would score 1.0 or 0.0 rather
    # than the 0.5 it has earned.
    srt = s[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# TODO(critic-calibration): the critic's PROBABILITIES are not usable as-is on new terrain,
# only its ranking. Measured on held-out tiles, the reliability curve is flatter than the
# diagonal at both ends -- it predicts more extreme values than the outcomes justify, which is
# ordinary over-confidence -- and the ECE moved noticeably between two fits of the SAME
# configuration differing only in seed, so it is not yet a stable property of the model.
#
# This matters more than a ranking metric suggests, and the reason is specific to what the
# critic is FOR. The advantage is a DIFFERENCE of two predictions,
# `P(fail|s_t) - P(fail|s_{t+n})`, so a distortion that varies with the predicted level does
# not cancel: two chunks that move P(fail) by the same true amount get different labels
# depending on whereabouts in an episode they sit. A constant offset would cancel; an
# S-shaped one does not.
#
# The cheap fix is temperature scaling -- one scalar fitted on a held-out split, applied to
# the logits -- which preserves the ranking exactly and so cannot cost anything the AUC
# measures. Fit it on the held-out TILES rather than the held-out episodes, since it is the
# transfer case that is miscalibrated. Do this before any label is trusted.
def calibration(p_fail: np.ndarray, failed: np.ndarray, *, n_bins: int = 10
                ) -> dict[str, np.ndarray | float]:
    """Predicted failure probability against the observed frequency, by bin.

    Args:
        p_fail: ``[n]`` predictions in ``[0, 1]``.
        failed: ``[n]`` boolean outcomes.
        n_bins: Equal-width bins over ``[0, 1]``.

    Returns:
        ``bin_lo``, ``bin_hi``, ``n``, ``predicted`` (mean prediction in the bin),
        ``observed`` (empirical failure rate), and two scalars: ``ece`` (expected calibration
        error, the count-weighted mean ``|predicted - observed|``) and ``mce`` (the worst
        bin). Empty bins carry NaN and contribute nothing.
    """
    p = np.asarray(p_fail, dtype=np.float64)
    y = np.asarray(failed).astype(bool)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    pred = np.full(n_bins, np.nan)
    obs = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            pred[b], obs[b] = float(p[m].mean()), float(y[m].mean())
    ok = cnt > 0
    w = cnt[ok] / max(int(cnt.sum()), 1)
    gap = np.abs(pred[ok] - obs[ok])
    return {
        "bin_lo": edges[:-1], "bin_hi": edges[1:], "n": cnt,
        "predicted": pred, "observed": obs,
        "ece": float((w * gap).sum()) if ok.any() else float("nan"),
        "mce": float(gap.max()) if ok.any() else float("nan"),
    }


def auc_within_stratum(scores: np.ndarray, positive: np.ndarray, stratum: np.ndarray) -> float:
    """AUC computed by comparing episodes ONLY against others in the same stratum, then pooled.

    Plain AUC ranks every positive against every negative in the whole sample, so on a mix of
    easy and hard TILES it rewards a critic for knowing *which tile this is* every bit as much
    as for reading how one rollout is going -- an early-episode critic that has learned
    difficulty from the privileged scan can score well here without separating a single good
    attempt from a bad one on the SAME piece of terrain. Restricting every comparison to a
    common stratum removes exactly that: if the pooled statistic collapses toward 0.5 while
    the unstratified AUC stays high, the unstratified number was difficulty discrimination,
    not the per-rollout signal advantage conditioning needs.

    Pooled rather than averaged per stratum: concordant/discordant PAIR counts are summed
    across strata before dividing, which is what lets a handful of large tiles carry a small
    tile's noisy few-episode AUC rather than have it count equally in a mean of AUCs.

    Args:
        scores: ``[n]`` predictions; higher should mean more likely positive.
        positive: ``[n]`` boolean labels.
        stratum: ``[n]`` group id (e.g. terrain tile) a comparison may not cross.

    Returns:
        The pooled within-stratum AUC, or NaN if no stratum contains both classes.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positive).astype(bool)
    g = np.asarray(stratum)
    concordant, pairs = 0.0, 0.0
    for tile in np.unique(g):
        m = g == tile
        n_pos, n_neg = int(y[m].sum()), int((~y[m]).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        a = auc(s[m], y[m])
        concordant += a * n_pos * n_neg
        pairs += n_pos * n_neg
    return float(concordant / pairs) if pairs > 0 else float("nan")


def auc_by_phase(p_fail: np.ndarray, failed: np.ndarray, phase: np.ndarray, *,
                 n_bins: int = 10) -> dict[str, np.ndarray]:
    """Discrimination as a function of where in the episode the state sits.

    Args:
        p_fail: ``[n]`` predictions.
        failed: ``[n]`` boolean episode outcomes, broadcast to every state of the episode.
        phase: ``[n]`` in ``[0, 1]``; 0 at the episode's first chunk, 1 at its last.
        n_bins: Equal-width phase bins.

    Returns:
        ``bin_lo``, ``bin_hi``, ``n``, ``auc`` per bin. A bin holding one class only gets
        NaN, which is not the same as 0.5 and must not be plotted as if it were.
    """
    ph = np.asarray(phase, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(ph, edges[1:-1], right=False), 0, n_bins - 1)
    out = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b]:
            out[b] = auc(np.asarray(p_fail)[m], np.asarray(failed)[m])
    return {"bin_lo": edges[:-1], "bin_hi": edges[1:], "n": cnt, "auc": out}


# TODO(advantage-n): the two instruments for choosing `n` DISAGREE, and the gap is large.
#
# `lead_time` below reports a few chunks of warning per failure, measured as a sustained
# crossing of `tau`. But plotting P(fail) per episode against time-to-terminal
# (`plot_value_traces.py`) shows the success and failure FAMILIES already well separated
# several seconds -- tens of chunks -- before the end. Both are correct and they answer
# different questions:
#
#   * lead_time asks when an INDIVIDUAL episode's prediction crosses an absolute level, which
#     is sensitive to `tau` and to how confident the critic happens to be;
#   * the traces show when the two POPULATIONS separate, which is what "the outcome became
#     predictable" actually means.
#
# `n` should span the interval over which a chunk's consequence becomes visible to the critic,
# so the population answer is closer to the right one and the current `tau = 0.5` is probably
# conservative. Two ways to settle it without guessing: sweep `tau` and see how much the lead
# moves (a threshold-sensitive number is the wrong statistic), and sweep `n` directly, picking
# the value that maximises the correlation between the resulting advantage and the realised
# outcome on held-out data. The second is the definition that matters and it needs no new
# instrument.
def lead_time(p_fail: np.ndarray, episode: np.ndarray, index: np.ndarray,
              failed: np.ndarray, *, tau: float = 0.5,
              reference: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """How many chunks before the end ``P(fail)`` rises and STAYS risen, per failed episode.

    Defined as a sustained crossing rather than a first touch: the earliest chunk ``t`` such
    that ``p_fail[s] >= tau`` for every ``s >= t``. A single spike three seconds out is not
    the critic knowing something; it is noise, and a first-touch definition would report it
    as a three-second lead.

    Args:
        p_fail: ``[n]`` predictions, one per chunk.
        episode: ``[n]`` episode ids.
        index: ``[n]`` chunk index within the episode.
        failed: ``[n]`` boolean episode outcome, constant within an episode.
        tau: The level that counts as "risen".
        reference: Optional ``[n]`` per-state reference index to measure the lead against
            instead of the episode's last chunk -- pass the chunk containing
            ``first_fall_step`` to ask the sharper question, whether the critic rises before
            the FALL rather than merely before the episode ends.

    Returns:
        ``episode`` (the failed episodes, in order), ``lead`` (chunks of warning; 0 means it
        never rose before the reference, NaN means it never rose at all), and ``n_chunks``.
    """
    p = np.asarray(p_fail, dtype=np.float64)
    ep = np.asarray(episode)
    ix = np.asarray(index)
    bad = np.asarray(failed).astype(bool)
    ref = None if reference is None else np.asarray(reference, dtype=np.float64)

    eps, leads, lens = [], [], []
    for e in np.unique(ep[bad]):
        m = ep == e
        order = np.argsort(ix[m])
        pe = p[m][order]
        end = float(ix[m][order][-1]) if ref is None else float(np.nanmedian(ref[m]))
        if not np.isfinite(end):
            continue
        risen = pe >= float(tau)
        # Earliest index from which it is risen for the rest of the episode.
        sustained = np.flip(np.cumprod(np.flip(risen.astype(np.int64))))
        hit = np.flatnonzero(sustained > 0)
        eps.append(e)
        lens.append(int(pe.size))
        leads.append(float("nan") if hit.size == 0
                     else max(0.0, end - float(ix[m][order][hit[0]])))
    return {"episode": np.asarray(eps), "lead": np.asarray(leads, dtype=np.float64),
            "n_chunks": np.asarray(lens, dtype=np.int64)}


def value_traces(p_fail: np.ndarray, episode: np.ndarray, index: np.ndarray,
                 failed: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Per-episode ``P(fail)`` traces, for the cheap look that comes before any statistic.

    Plotting these for a handful of episodes with successes and failures overlaid, and seeing
    WHERE the two families separate, is worth more than any of the numbers above as a first
    pass -- it is the check PI's Figure 4 is making, that the value drops at the mistake
    rather than at its consequence.

    Args:
        p_fail: ``[n]`` predictions.
        episode: ``[n]`` episode ids.
        index: ``[n]`` chunk index within the episode.
        failed: ``[n]`` boolean episode outcome.

    Returns:
        ``{episode_id: {"index": [k], "p_fail": [k], "failed": bool}}``, each sorted by index.
    """
    ep, ix = np.asarray(episode), np.asarray(index)
    p, bad = np.asarray(p_fail, dtype=np.float64), np.asarray(failed).astype(bool)
    out: dict[int, dict[str, np.ndarray]] = {}
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(ix[m])
        out[int(e)] = {"index": ix[m][order], "p_fail": p[m][order],
                       "failed": bool(bad[m][0])}
    return out


# ----------------------------------------------------------------------------------------
# The QUALITY head.  A regression head needs different questions from a classifier, and the
# first one is whether it did anything at all.
# ----------------------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged.

    Reported instead of (or beside) R^2 because it is the quantity the SELECTOR actually
    needs: best-of-K takes an argmax, so what matters is whether the head ORDERS candidates
    correctly, not whether its scale is right.

    Args:
        a: ``[n]``.
        b: ``[n]``.

    Returns:
        Spearman's rho, or NaN when either side is constant (an undefined correlation, which
        must not be reported as 0.0 -- that would read as "measured, no relationship").
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")

    def _rank(x):
        """Average ranks, so ties do not order by array position.

        Ties are averaged over RUNS OF THE SORTED ARRAY rather than by scanning the whole
        array once per distinct value. The scanning form is ``O(u * n)`` in the number of
        distinct values, which on a continuous score is ``O(n^2)``: at the 294k rows of a
        full rollout table that is ~9e10 element comparisons and the call never returns,
        while at the few thousand rows every existing caller passes it is merely slow. Same
        result, and the tests pin the tie behaviour.
        """
        order = np.argsort(x, kind="mergesort")
        s = x[order]
        r = np.empty(x.size, dtype=np.float64)
        ranks = np.arange(1, x.size + 1, dtype=np.float64)
        # Boundaries of equal-value runs in the sorted array; each run takes its mean rank,
        # which for a contiguous block is the midpoint of its first and last rank.
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        ends = np.r_[starts[1:], s.size]
        means = 0.5 * (ranks[starts] + ranks[ends - 1])
        r[order] = np.repeat(means, ends - starts)
        return r

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def return_gates(pred: np.ndarray, target: np.ndarray, *, n_bins: int = 10,
                 seed: int = 0) -> dict[str, object]:
    """What to ask of the quality head, with the collapse check FIRST.

    **A regression head's characteristic failure is predicting the mean**, and it is a real
    stationary point in this repo rather than a hypothetical: an FM arm once sat at a val loss
    of exactly predict-the-mean for hundreds of steps while its training loss fell.  A head
    that has collapsed there still produces finite, plausible, well-calibrated-looking numbers
    -- its calibration curve is a single point on the diagonal -- and it ranks every candidate
    identically, which presents as "best-of-K does nothing" rather than as a broken critic.

    So ``r2`` is the headline: it is the fraction of variance explained AGAINST THE CONSTANT
    PREDICTOR, so ``<= 0`` means the head has learned nothing a scalar could not do.  ``rho``
    is what the selector needs (it takes an argmax, so ordering is the whole job), and
    ``rho_shuffled`` is its null -- the same statistic on a permuted target, which must land
    at ~0 for the rest to mean anything.

    Args:
        pred: ``[n]`` predicted normalized returns.
        target: ``[n]`` realized normalized returns.
        n_bins: Equal-width calibration bins over ``[0, 1]``.
        seed: For the shuffle control.

    Returns:
        ``r2``, ``rho``, ``rho_shuffled``, ``rmse``, ``pred_std``, ``target_std``,
        ``bin_lo`` / ``bin_hi`` / ``n`` / ``predicted`` / ``observed`` for the calibration
        curve, and ``ece`` (count-weighted mean ``|predicted - observed|``).
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if p.size != t.size:
        raise ValueError(f"pred and target disagree in length: {p.size} vs {t.size}")
    if p.size == 0:
        raise ValueError("no rows to gate: an empty quality set is a setup error, not a score")

    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    pred_b = np.full(n_bins, np.nan)
    obs_b = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=np.int64)
    for bi in range(n_bins):
        m = idx == bi
        cnt[bi] = int(m.sum())
        if cnt[bi]:
            pred_b[bi], obs_b[bi] = float(p[m].mean()), float(t[m].mean())
    ok = cnt > 0
    w = cnt[ok] / max(int(cnt.sum()), 1)
    return {
        "r2": r2,
        "rho": spearman(p, t),
        "rho_shuffled": spearman(p, rng.permutation(t)),
        "rmse": float(np.sqrt(((t - p) ** 2).mean())),
        "pred_std": float(p.std()),
        "target_std": float(t.std()),
        "bin_lo": edges[:-1], "bin_hi": edges[1:], "n": cnt,
        "predicted": pred_b, "observed": obs_b,
        "ece": float((w * np.abs(pred_b[ok] - obs_b[ok])).sum()) if ok.any() else float("nan"),
    }
