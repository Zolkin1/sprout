# Copyright 2026 Zachary Olkin. All rights reserved.

"""The advantage label: what it is, how it is computed, and how it is binarized.

The generator emits one chunk per re-plan.  A critic scores the state it emitted from and
the state it arrived at ``n`` re-plans later; the difference is what that chunk BOUGHT::

    A(s_t, a_t) = P(fail | s_t) - P(fail | s_{t+n})

**The sign is easy to read backwards, so state it twice.**  The n-step advantage is
``V(s_{t+n}) - V(s_t)``, and with ``gamma = 1`` and a terminal-only reward of ``-1`` on
failure, ``V(s) = -P(fail | s)`` -- a Bernoulli failure classifier with a MINUS in front,
because a higher chance of falling is a lower value.  Substituting the minus twice flips the
order of the two terms, which is why ``P(fail|s_t)`` comes first above.  So::

    A > 0   the failure probability FELL over those n re-plans -- the chunk helped
    A < 0   it ROSE -- the chunk hurt

and ``binarize`` below labels high ``A`` positive.  The terminal cases are the quick check
that this is right, and they need no critic at all:

* the episode FAILED within ``n`` re-plans -> ``V(s_{t+n}) = -1``, so ``A = P(fail|s_t) - 1``,
  which is NEGATIVE (``-0.7`` from a state the critic scored at 0.3) -- a chunk followed by a
  fall;
* it REACHED THE GOAL within ``n`` -> ``V(s_{t+n}) = 0``, so ``A = P(fail|s_t)``, which is
  POSITIVE (``+0.3`` from that same state) -- a chunk that carried a risky state to the goal.

Both are the realized outcome, not a prediction, which is the point: near a terminal the
advantage stops depending on the critic at all.

**The binarization is per family, and that is not a detail.**  A global percentile would
label nearly every ``flat`` chunk negative (the family barely ever fails, so its advantages
are a tight cluster of near-zeros that sit below any threshold set by the stair families)
and nearly every ``jump_off_front`` chunk positive.  The threshold has to be a statement
about a chunk relative to OTHER CHUNKS FACING THE SAME PROBLEM.

**A third label, ``null``, is for data with no signal -- not for bad data.**  Where the
within-family spread of advantages is below the noise floor there is nothing to prefer, and
labelling those chunks negative would actively teach the model that competent walking is
bad.  ``null`` is also what the ~30% conditioning dropout writes during training, so
classifier-free guidance over the label is available at inference.
"""

from __future__ import annotations

import numpy as np

#: The three label values, in the order the embedding rows are indexed.
LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1
LABEL_NULL = 2

#: Human-readable names, index-aligned with the constants above.
LABEL_NAMES = ("negative", "positive", "null")


def label_from_name(name: str) -> int:
    """Map ``"negative" / "positive" / "null"`` to its integer label.

    Args:
        name: One of :data:`LABEL_NAMES`.

    Returns:
        The integer label.

    Raises:
        ValueError: On an unrecognised name.  Deliberately not a silent default: the label
            is the whole experiment, and an export that quietly picked ``null`` would look
            exactly like one that picked ``positive``.
    """
    if name not in LABEL_NAMES:
        raise ValueError(f"advantage label must be one of {LABEL_NAMES}, got {name!r}")
    return LABEL_NAMES.index(name)


def n_step_advantage(
    p_fail: np.ndarray,
    episode_uid: np.ndarray,
    replan_index: np.ndarray,
    failed: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``A(s_t, a_t) = P(fail|s_t) - P(fail|s_{t+n})`` over a window table.

    Rows are grouped by ``episode_uid`` and ordered by ``replan_index``.  A row whose
    ``t + n`` lands past the episode's last recorded re-plan is closed with the realized
    terminal value (``1.0`` if the episode failed, ``0.0`` if it reached the goal) rather
    than dropped -- those are the rows nearest the outcome and so the most informative.

    Args:
        p_fail: ``[W]`` critic output ``P(fail | s_t)`` in ``[0, 1]``.
        episode_uid: ``[W]`` episode id per window.
        replan_index: ``[W]`` position within the episode, 0-based.
        failed: ``[W]`` bool, the episode's realized outcome (constant within an episode).
        n: Look-ahead in re-plans.

    Returns:
        ``(advantage, bootstrapped)`` -- ``[W]`` float64 and ``[W]`` bool, the latter True
        where the terminal value was used instead of a critic reading.

    Raises:
        ValueError: If ``n < 1`` or the arrays disagree in length.
    """
    if int(n) < 1:
        raise ValueError(f"n must be at least one re-plan, got {n}")
    lengths = {a.shape[0] for a in (p_fail, episode_uid, replan_index, failed)}
    if len(lengths) != 1:
        raise ValueError(f"arrays disagree in length: {sorted(lengths)}")

    w = p_fail.shape[0]
    adv = np.zeros(w, dtype=np.float64)
    boot = np.zeros(w, dtype=bool)
    # One pass per episode. `order` sorts by (uid, replan_index) so a group is contiguous
    # and internally ordered without a per-episode sort.
    order = np.lexsort((replan_index, episode_uid))
    uid_s = episode_uid[order]
    starts = np.flatnonzero(np.r_[True, uid_s[1:] != uid_s[:-1]])
    ends = np.r_[starts[1:], w] if starts.size else np.array([], dtype=int)
    for s, e in zip(starts, ends):
        rows = order[s:e]
        p = p_fail[rows]
        m = rows.size
        # The future reading, index-shifted; the tail bootstraps on the outcome.
        future = np.empty(m, dtype=np.float64)
        head = max(m - n, 0)
        future[:head] = p[n:]
        future[head:] = 1.0 if bool(failed[rows[0]]) else 0.0
        adv[rows] = p.astype(np.float64) - future
        boot[rows[head:]] = True
    return adv, boot


def combined_n_step_advantage(
    p_fail: np.ndarray,
    g_hat: np.ndarray,
    chunk_reward: np.ndarray,
    episode_uid: np.ndarray,
    replan_index: np.ndarray,
    failed: np.ndarray,
    n: int,
    *,
    gamma: float,
    w_fail: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The n-step advantage of the COMBINED value, survival plus tracking quality.

    ``V(s) = w_track * G_hat(s) - w_fail * P(fail | s)`` with ``w_track = 1 - w_fail``, the
    same definition :meth:`FlowValueModel.combined_value` computes at run time -- so the
    fine-tune's label and the selector's objective are one quantity by construction rather
    than two implementations that happen to agree::

        A = w_track * [ (1-g) * sum_{i<n} g^i R_{t+i}  +  g^n * G_hat(s_{t+n}) - G_hat(s_t) ]
          + w_fail  * [ P(fail|s_t) - P(fail|s_{t+n}) ]

    Two things about the shape of this are easy to get wrong:

    * **The second bracket is the SURVIVAL advantage exactly as it already was**, sign and
      all -- it is already the TD form of a terminal-only reward at ``gamma = 1``. So
      ``w_fail = 1`` reproduces :func:`n_step_advantage` bit for bit and is the control arm,
      not a new configuration.
    * **The ``(1 - gamma)`` factor on the measured rewards is not cosmetic.** ``G_hat`` is a
      NORMALIZED return, carrying that factor by construction, so without it the measured and
      bootstrapped halves are in different units and the sum is dominated by whichever
      happens to be larger.

    The tail bootstraps the same way the survival half does -- on the realized outcome, which
    for the tracking term is ``0.0`` (a failure is absorbing; nothing good follows) or the
    last observed ``G_hat`` (the episode ended by arriving or by the budget, so the motion is
    assumed to continue at the quality it ended on).  Absorbing on a NON-failure would punish
    arriving quickly, which is the confound
    :func:`transfer.deploy.tracking_error.normalized_return` documents.

    Args:
        p_fail: ``[W]`` ``P(fail | s_t)``.
        g_hat: ``[W]`` predicted normalized tracking return at ``s_t``, in ``[0, 1]``.
        chunk_reward: ``[W]`` MEASURED per-chunk tracking reward for the chunk emitted at
            ``t`` -- i.e. already shifted out of ``track_err_prev``, whose row ``i`` holds the
            chunk of row ``i-1``.
        episode_uid: ``[W]`` episode id.
        replan_index: ``[W]`` position within the episode, 0-based.
        failed: ``[W]`` bool, the episode's realized outcome.
        n: Look-ahead in re-plans.
        gamma: Per-CHUNK discount.
        w_fail: Weight on the survival term, in ``[0, 1]``.

    Returns:
        ``(advantage, bootstrapped)`` -- ``[W]`` float64 and ``[W]`` bool.

    Raises:
        ValueError: On a bad ``n``, ``gamma``, ``w_fail``, or ragged inputs.
    """
    if not 0.0 <= float(w_fail) <= 1.0:
        raise ValueError(f"w_fail must be in [0, 1], got {w_fail!r}")
    if not 0.0 < float(gamma) < 1.0:
        raise ValueError(f"gamma must be in (0, 1), got {gamma!r}")

    surv, boot = n_step_advantage(p_fail, episode_uid, replan_index, failed, n)
    w_track = 1.0 - float(w_fail)
    if w_track == 0.0:
        # Bit-for-bit the survival advantage. Returned before touching the tracking arrays so
        # this path cannot be perturbed by a NaN in a column it does not read.
        return float(w_fail) * surv, boot

    lengths = {a.shape[0] for a in (p_fail, g_hat, chunk_reward, episode_uid, replan_index,
                                    failed)}
    if len(lengths) != 1:
        raise ValueError(f"arrays disagree in length: {sorted(lengths)}")

    g = float(gamma)
    w = p_fail.shape[0]
    track = np.zeros(w, dtype=np.float64)
    order = np.lexsort((replan_index, episode_uid))
    uid_s = episode_uid[order]
    starts = np.flatnonzero(np.r_[True, uid_s[1:] != uid_s[:-1]])
    ends = np.r_[starts[1:], w] if starts.size else np.array([], dtype=int)
    disc = g ** np.arange(int(n), dtype=np.float64)
    for s, e in zip(starts, ends):
        rows = order[s:e]
        m = rows.size
        r = chunk_reward[rows].astype(np.float64)
        gh = g_hat[rows].astype(np.float64)
        # The tail value: 0 on a failure (absorbing -- nothing good follows) and the last
        # observed G_hat otherwise (the episode ended by arriving or by the budget, so the
        # motion is assumed to continue at the quality it ended on). Absorbing on a
        # NON-failure would punish arriving quickly.
        tail = 0.0 if bool(failed[rows[0]]) else float(gh[-1])
        for i in range(m):
            # TRUNCATED n-step: a row whose window overruns the episode gets the k chunks
            # that exist and bootstraps at g^k, NOT at g^n. Discounting the tail by the full
            # g^n while padding the missing rewards with zero is inconsistent -- it charges a
            # row for chunks that never happened -- and it is worst on the LAST rows, which
            # are the ones nearest the outcome and so the most informative.
            k = min(int(n), m - i)
            measured = float((r[i:i + k] * disc[:k]).sum())
            future = gh[i + k] if i + k < m else tail
            track[rows[i]] = (1.0 - g) * measured + (g ** k) * future - gh[i]
    return float(w_fail) * surv + w_track * track, boot


def binarize(
    advantage: np.ndarray,
    family_code: np.ndarray,
    *,
    percentile: float = 35.0,
    null_spread: float = 0.02,
    absolute_floor: float | None = None,
) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
    """Turn per-window advantages into ``{negative, positive, null}`` labels, per family.

    A family whose advantage spread is below ``null_spread`` is labelled ``null``
    WHOLESALE.  That is the case the plan singles out: ``flat`` succeeds ~99% of the time,
    so its advantages are a cluster of near-zeros carrying no preference, and cutting them
    at a percentile would manufacture a distinction the data does not contain.

    Args:
        advantage: ``[W]`` advantages.
        family_code: ``[W]`` family code per window.
        percentile: Within-family percentile below which a chunk is negative.  Chunks at or
            above it are positive.
        null_spread: If a family's interquartile range of advantages is below this, the whole
            family is ``null``.
        absolute_floor: When set, the positive threshold becomes
            ``max(family_percentile, absolute_floor)`` -- so a family that is uniformly bad
            cannot label its least-bad third positive.  **Off by default, and a floor of 0.0
            is stronger than it looks**: over a SUCCESSFUL episode the advantages telescope
            to a small total spread over ~25 chunks, so an individual chunk's sign is mostly
            noise and a zero floor labels roughly half of a clean run negative.  The
            within-family percentile is the calibrated statement; this is an opt-in veto for
            a family believed hopeless.

    Returns:
        ``(labels, per_family)`` -- ``[W]`` int64 in ``{0, 1, 2}``, and a per-family record
        of the threshold used, the IQR and the resulting positive fraction (for the manifest,
        so a run states what it labelled rather than leaving it to be re-derived).
    """
    labels = np.full(advantage.shape[0], LABEL_NULL, dtype=np.int64)
    report: dict[int, dict[str, float]] = {}
    for fam in np.unique(family_code):
        sel = family_code == fam
        a = advantage[sel]
        if a.size == 0:
            continue
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        if iqr < float(null_spread):
            report[int(fam)] = {"n": int(a.size), "iqr": iqr, "threshold": float("nan"),
                                "positive_frac": 0.0, "null": True}
            continue
        thr = float(np.percentile(a, percentile))
        if absolute_floor is not None:
            thr = max(thr, float(absolute_floor))
        lab = np.where(a >= thr, LABEL_POSITIVE, LABEL_NEGATIVE)
        labels[sel] = lab
        report[int(fam)] = {"n": int(a.size), "iqr": iqr, "threshold": thr,
                            "positive_frac": float((lab == LABEL_POSITIVE).mean()),
                            "null": False}
    return labels, report
