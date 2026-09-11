# Copyright 2026 Zachary Olkin. All rights reserved.

"""Estimators and statistics for judging a critic BEFORE it is used to reweight a policy.

``advantage.py`` owns the one estimator the fine-tune currently consumes -- the ``n``-step
survival advantage and its per-family binarization.  This module owns the ones it does NOT,
so the question "would a different estimator have been better" is answerable without changing
the pipeline first.

**The two heads answer different questions and live on different scales**, which is why every
function here takes a value array rather than a checkpoint:

* the SURVIVAL head predicts ``P(fail | s)``, and with a terminal-only reward the value is
  ``V_f = -P(fail)`` on ``(-1, 0)``.  Its per-chunk reward is identically zero, so every
  estimator below reduces to a difference of two critic readings plus a terminal bootstrap.
* the QUALITY head predicts the normalized tracking return ``G`` on ``[0, 1]``, built by
  :func:`advantage.normalized_returns` as ``G_t = (1 - gamma) R_t + gamma G_{t+1}``.  That
  recursion is DENSE, so ``V_q`` supports a genuine TD residual and a genuine GAE, and the
  four-estimator menu stops being degenerate.

Mixing the two is not a rescaling.  A weighted sum needs their levels commensurable, and the
survival head's level is horizon-dependent in a way the quality head's is not (see
``fit_value.py``'s module docstring) -- so the arms are kept separate and reported separately.

**Every estimator here is FLAT over a window table**, keyed by ``(episode_uid,
replan_index)``, matching what ``record_rollouts.py`` writes and what
:func:`advantage.n_step_advantage` already assumes.  Rows need not be sorted; each function
sorts a view internally and writes back in the caller's order.
"""

from __future__ import annotations

import numpy as np


def _episode_groups(episode_uid: np.ndarray, replan_index: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous, internally time-ordered row groups, one per episode.

    Args:
        episode_uid: ``[W]`` episode id per row.
        replan_index: ``[W]`` position within the episode.

    Returns:
        ``(order, starts, ends)`` -- ``order`` indexes the caller's rows in
        ``(uid, replan_index)`` order, and ``order[s:e]`` is one episode for each
        ``(s, e)`` in ``zip(starts, ends)``.
    """
    w = episode_uid.shape[0]
    order = np.lexsort((replan_index, episode_uid))
    uid_s = episode_uid[order]
    starts = np.flatnonzero(np.r_[True, uid_s[1:] != uid_s[:-1]])
    ends = np.r_[starts[1:], w] if starts.size else np.array([], dtype=int)
    return order, starts, ends


def td_residuals(value: np.ndarray, reward: np.ndarray, episode_uid: np.ndarray,
                 replan_index: np.ndarray, failed: np.ndarray, *, gamma: float,
                 normalized: bool = True,
                 valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One-chunk TD residuals ``delta_t = (1-g) R_t + g V(s_{t+1}) - V(s_t)``.

    The ``(1 - gamma)`` factor is present exactly when the value being differenced is the
    NORMALIZED return :func:`advantage.normalized_returns` builds, because that is the
    recursion it satisfies.  For the survival head, whose reward is identically zero and
    whose value is ``-P(fail)``, pass ``reward`` zeros and ``normalized=False``; the
    expression collapses to ``g V(s_{t+1}) - V(s_t)`` and, at ``gamma = 1``, to the familiar
    ``P(fail|s_t) - P(fail|s_{t+1})``.

    **The terminal is closed on the REALIZED outcome, never on the critic.** The last chunk
    of an episode has no ``s_{t+1}`` in the table, and bootstrapping it on its own ``V``
    would make the residual identically zero exactly where the outcome is most certain.  The
    tail follows :func:`advantage.normalized_returns`: ``0`` on a failure, which is
    absorbing, and the last observed reward otherwise.

    **``valid`` is not optional in practice, and the reason is a one-row shift.**
    :func:`advantage.chunk_rewards_from_track_err` stores each chunk's reward on the row
    BEFORE it (``reward[rows[:-1]] = per_row[rows[1:]]``), so every episode's LAST row carries
    ``reward = 0.0`` with ``valid = False`` -- a placeholder, not a measurement.
    ``normalized_returns`` drops those rows before it recurses, so a residual computed over
    ALL rows closes the terminal on ``0.0`` instead of on the last real reward.  The error is
    largest exactly where the tail matters most -- successful episodes, whose tail is the last
    reward rather than an absorbing zero -- so it is confounded with the outcome rather than
    being noise, and it propagates into every ``lam > 0`` GAE.  Pass the same mask
    ``normalized_returns`` was given.

    Args:
        value: ``[W]`` critic reading at ``s_t``, on the same scale as the return.
        reward: ``[W]`` per-chunk reward; zeros for the survival head.
        episode_uid: ``[W]``.
        replan_index: ``[W]``.
        failed: ``[W]`` bool, the episode's realized outcome (constant within an episode).
        gamma: Per-chunk discount.
        normalized: Whether ``value`` is on the ``(1 - gamma)``-normalized scale, which also
            selects how the terminal is closed (see above).
        valid: ``[W]`` bool, which rows carry a real reward.  ``None`` means all of them,
            which is right only for the survival head (whose reward is identically zero by
            construction, so no row is a placeholder).

    Returns:
        ``(delta, is_terminal)`` -- ``[W]`` float64 and ``[W]`` bool, the latter True on the
        row whose successor was the realized terminal rather than a critic reading.  Rows
        outside ``valid`` get ``delta = 0`` and are not terminals; they carry no residual
        because they carry no reward.

    Raises:
        ValueError: On a ``gamma`` outside ``(0, 1]`` or arrays that disagree in length.
    """
    if not 0.0 < float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
    lengths = {a.shape[0] for a in (value, reward, episode_uid, replan_index, failed)}
    if len(lengths) != 1:
        raise ValueError(f"arrays disagree in length: {sorted(lengths)}")
    ok = (np.ones(value.shape[0], dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool))
    if ok.shape[0] != value.shape[0]:
        raise ValueError(f"valid has length {ok.shape[0]}, expected {value.shape[0]}")

    g = float(gamma)
    scale = (1.0 - g) if normalized else 1.0
    delta = np.zeros(value.shape[0], dtype=np.float64)
    term = np.zeros(value.shape[0], dtype=bool)
    order, starts, ends = _episode_groups(episode_uid, replan_index)
    for s, e in zip(starts, ends):
        rows = order[s:e]
        rows = rows[ok[rows]]
        if rows.size == 0:
            continue
        v = value[rows].astype(np.float64)
        r = reward[rows].astype(np.float64)
        nxt = np.empty(v.size, dtype=np.float64)
        nxt[:-1] = v[1:]
        # The realized tail, on whichever scale `value` lives. For the survival head the
        # value IS -P(fail), so a failure closes at -1 and an arrival at 0; for the quality
        # head a failure is absorbing at 0 and an arrival continues at the last reward.
        if bool(failed[rows[0]]):
            nxt[-1] = -1.0 if not normalized else 0.0
        else:
            nxt[-1] = 0.0 if not normalized else float(r[-1])
        delta[rows] = scale * r + g * nxt - v
        term[rows[-1]] = True
    return delta, term


def gae_advantage(delta: np.ndarray, episode_uid: np.ndarray, replan_index: np.ndarray, *,
                  gamma: float, lam: float,
                  valid: np.ndarray | None = None) -> np.ndarray:
    """Generalized advantage estimation over the TD residuals, at CHUNK granularity.

    ``A_t = sum_l (gamma * lam)^l delta_{t+l}``, accumulated backwards within each episode and
    truncated at its last chunk -- which is exact rather than an approximation, because
    :func:`td_residuals` already closed that chunk on the realized outcome.

    ``lam = 0`` returns the one-chunk TD advantage and ``lam = 1`` the Monte-Carlo one, so a
    sweep over ``lam`` is a sweep over how much the critic is trusted against how much the
    realized outcome is.  That is the axis worth reporting when the question is whether the
    critic is good enough to bootstrap on at all.

    Args:
        delta: ``[W]`` TD residuals.
        episode_uid: ``[W]``.
        replan_index: ``[W]``.
        gamma: Per-chunk discount.
        lam: GAE decay in ``[0, 1]``.
        valid: ``[W]`` bool; must be the SAME mask :func:`td_residuals` was given, or the
            accumulation runs over rows whose residual is a placeholder zero and the
            ``lam = 1`` identity against ``G - V`` stops holding.

    Returns:
        ``[W]`` float64 advantages, in the caller's row order.

    Raises:
        ValueError: On a ``lam`` outside ``[0, 1]``.
    """
    if not 0.0 <= float(lam) <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam!r}")
    ok = (np.ones(delta.shape[0], dtype=bool) if valid is None
          else np.asarray(valid, dtype=bool))
    gl = float(gamma) * float(lam)
    out = np.zeros(delta.shape[0], dtype=np.float64)
    order, starts, ends = _episode_groups(episode_uid, replan_index)
    for s, e in zip(starts, ends):
        rows = order[s:e]
        rows = rows[ok[rows]]
        if rows.size == 0:
            continue
        d = delta[rows].astype(np.float64)
        acc = 0.0
        vals = np.empty(d.size, dtype=np.float64)
        for i in range(d.size - 1, -1, -1):
            acc = d[i] + gl * acc
            vals[i] = acc
        out[rows] = vals
    return out


def variance_decomposition(x: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """Split ``Var(x)`` into its between-group and within-group parts.

    **This is the measurement that decides whether advantage-weighted BC can differ from
    filtered BC at all.**  If almost all of an advantage's variance is BETWEEN episodes, then
    reweighting by it is reweighting by the episode's outcome, which the trajectory-level
    label already does -- the critic bought nothing.  Only the within-episode part can carry
    "this chunk, in this episode, was better than that one".

    Args:
        x: ``[n]`` values.
        group: ``[n]`` group id per value.

    Returns:
        ``total``, ``between``, ``within`` and ``frac_within``, all population (not sample)
        variances so the two parts sum exactly to the total.

    Raises:
        ValueError: If the arrays disagree in length or are empty.
    """
    if x.shape[0] != group.shape[0]:
        raise ValueError(f"x and group disagree in length: {x.shape[0]} vs {group.shape[0]}")
    if x.size == 0:
        raise ValueError("cannot decompose the variance of an empty array")
    v = np.asarray(x, dtype=np.float64)
    _, inv = np.unique(group, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, weights=v) / cnt
    grand = v.mean()
    between = float((cnt * (means - grand) ** 2).sum() / v.size)
    within = float(((v - means[inv]) ** 2).sum() / v.size)
    total = between + within
    return {"total": total, "between": between, "within": within,
            "frac_within": float(within / total) if total > 0 else float("nan")}


def awr_weights(advantage: np.ndarray, *, beta: float, w_max: float | None = 20.0,
                standardize: bool = True, fit: np.ndarray | None = None) -> np.ndarray:
    """Advantage-weighted-regression weights ``w = clip(exp(A / beta), 0, w_max)``.

    Args:
        advantage: ``[n]`` advantages.
        beta: Temperature; smaller is sharper.
        w_max: Upper clip as a multiple of the MEAN weight, or ``None`` for none.  Reported
            both ways because the clip is what stops one row from owning the update, and how
            much mass it removes is itself a statement about the advantage's tail.
        fit: ``[n]`` bool selecting the rows the clip LEVEL is estimated from, or ``None``
            for all of them.  The level is a statistic of the sample, so computing it over
            rows a run also validates against makes the training weights a function of the
            val split -- measured at 3.1x on individual weights at ``beta = 0.5`` with an
            episode-clustered split.
        standardize: Divide by the advantage's own standard deviation first, so ``beta`` means
            the same thing across estimators whose scales differ by an order of magnitude.
            **The mean is NOT subtracted**: a constant shift is exactly cancelled by the
            normalization ``w / sum(w)`` that every consumer applies, so subtracting it would
            change no weight while making the reported ``exp`` arguments incomparable.

    Returns:
        ``[n]`` non-negative weights.

    Raises:
        ValueError: On a non-positive ``beta``.
    """
    if float(beta) <= 0.0:
        raise ValueError(f"beta must be positive, got {beta!r}")
    a = np.asarray(advantage, dtype=np.float64)
    if standardize:
        sd = float(a.std())
        a = a / sd if sd > 0 else a
    # Shift by the max before exponentiating. This is the standard log-sum-exp guard and it
    # changes no normalized weight, but it is what keeps `exp` finite at small beta -- where
    # an unshifted argument of a few hundred overflows to `inf` and every downstream ESS
    # comes back as NaN rather than as the collapse it actually is.
    z = a / float(beta)
    w = np.exp(z - z.max())
    if w_max is not None:
        # The clip is stated relative to the MEAN weight, which is what "no row is worth more
        # than 20 rows" means; clipping the raw exponential would depend on the shift above.
        #
        # ITERATED TO A FIXED POINT, because clipping LOWERS the mean and so lowers the level
        # the clip is stated against. Applied once against the pre-clip mean the property is
        # not delivered at all where it is most needed: measured post-clip max/mean was 20.0
        # at beta 1.0 but 103 at beta 0.25 and 1061 at beta 0.1 -- i.e. exactly in the sharp
        # arm the clip exists for, a single row could still own the update by 1061 rows'
        # worth while the run reported a clip of 20. The map is monotone decreasing and
        # bounded, so this converges; the cap is a guard, not a schedule.
        m = np.ones(w.shape[0], dtype=bool) if fit is None else np.asarray(fit, dtype=bool)
        if m.shape[0] != w.shape[0]:
            raise ValueError(f"fit has length {m.shape[0]}, expected {w.shape[0]}")
        # Solved by BISECTION rather than by iterating the map. `L -> w_max * mean(min(w, L))`
        # is a contraction but a slow one when a large mass sits at the clip -- 64 passes
        # still left max/mean at 20.000016 -- and the level is a scalar root, so bisecting it
        # is both exact to machine precision and cheap. `g` is positive for small `L` (where
        # it reduces to `(w_max - 1) * L`, and `w_max > 1` in any sane use) and negative
        # beyond `max(w)`, so the bracket is guaranteed.
        wm = float(w_max)
        fit_w = w[m]

        def _g(level: float) -> float:
            return wm * float(np.minimum(fit_w, level).mean()) - level

        lo, hi = 0.0, float(w.max())
        if wm > 1.0 and _g(hi) < 0.0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _g(mid) > 0.0:
                    lo = mid
                else:
                    hi = mid
            w = np.minimum(w, lo)
        else:
            # `w_max <= 1` has no positive fixed point (the level would have to sit at or
            # below the mean it is a multiple of), and a bracket that never goes negative
            # means the clip does not bind at all. Apply it once and stop.
            w = np.minimum(w, wm * float(w[m].mean()))
    return w


def stratified_awr_weights(
    advantage: np.ndarray,
    *,
    beta: float,
    strata: np.ndarray | None = None,
    fit: np.ndarray | None = None,
    w_max: float | None = 20.0,
    center: bool = True,
    balance: bool = True,
) -> np.ndarray:
    """AWR weights whose exponent is centered WITHIN each stratum.

    :func:`awr_weights` deliberately does not subtract a mean, because a GLOBAL shift cancels
    under the ``w / sum(w)`` every consumer applies.  That argument does not survive
    stratification: a per-stratum offset does NOT cancel, so a cell whose advantages sit
    higher takes systematically more weight than a cell whose sit lower -- which is a
    property of the cell, not of any chunk in it.

    That is the same selection effect ``advantage.binarize`` corrects by taking its
    percentile threshold inside the cell rather than across it, and it is measurable: an
    uncorrected Monte-Carlo advantage keeps 0.33 of rows above 1.3 m/s against 0.23 below
    0.9.  Centering here is the weighted arm's version of that correction, so a filtered-BC
    and an AWR run select over the same population and their comparison is about the
    weighting rather than about the speed distribution.

    **The scale stays POOLED on purpose.**  Dividing by each stratum's own standard deviation
    would equalize cells whose advantages genuinely differ in spread, amplifying the noise in
    a flat cell to match the signal in an informative one -- the failure ``null_spread``
    exists to prevent.  One pooled sigma keeps ``beta`` comparable across runs while leaving
    a low-spread cell with low-spread weights.

    Args:
        advantage: ``[n]`` advantages for the rows being weighted.
        beta: Temperature; smaller is sharper.
        strata: ``[n]`` stratum id per row (``advantage.group_ids``), or ``None`` to center
            globally -- which, as above, is the same as not centering at all.
        fit: ``[n]`` bool selecting the rows the mean and sigma are ESTIMATED from, or
            ``None`` for all of them.  Pass the training split: estimating the scale on rows
            the run also validates against makes ``beta`` a mild function of the val set.
            A stratum with no ``fit`` row falls back to the global fit mean rather than to
            zero, which would leave it centered on nothing.
        w_max: Upper clip as a multiple of the mean weight, or ``None``.
        center: Subtract each stratum's own mean.  ``False`` is the CONTROL arm and differs
            in that alone -- the scale stays the within-cell residual sd either way, so the
            two arms run at the same temperature and the comparison is about the centering.
        balance: Rescale each stratum's TOTAL weight to its share of the rows, which is what
            makes AWR's cell composition match filtered BC's.  Centering alone does not:
            ``binarize`` keeps a fixed 30% of EVERY cell, so its keep-set composition equals
            the data's by construction, while centering only removes a cell's MEAN offset
            and leaves its SPREAD free -- and since the scale is pooled, a wider-spread cell
            produces more extreme weights whose upper tail dominates the sum.  Measured on
            the box2 v2 labels at ``beta = 1.0``: max deviation from the row share is 0.29
            points for filtered BC, 3.71 uncentered, 2.54 centered, and ~0 with this on,
            correlating with cell spread at ``+0.627``.  Without it an AWR-vs-filtered-BC
            comparison moves the cell composition as well as the weighting, which is a
            second variable in the one contrast the arm exists to make.  It preserves the
            WITHIN-cell ranking exactly -- only each cell's total is rescaled.

            **It loosens ``w_max`` from a global bound to a per-cell one.**  The rescale is
            applied after the clip, so a cell scaled up carries its clipped rows past the
            global level -- measured 21.19x the mean against a stated 20x, about 6%.  The
            clip's purpose (no single row owning the update) survives that; the exact global
            multiple does not, so a caller reporting it should read the REALISED max rather
            than restate ``w_max``.

    Returns:
        ``[n]`` non-negative weights.

    Raises:
        ValueError: On a non-positive ``beta``, or ragged inputs.
    """
    a = np.asarray(advantage, dtype=np.float64)
    m = np.ones(a.shape[0], dtype=bool) if fit is None else np.asarray(fit, dtype=bool)
    if m.shape[0] != a.shape[0]:
        raise ValueError(f"fit has length {m.shape[0]}, expected {a.shape[0]}")
    if not m.any():
        raise ValueError("`fit` selects no rows, so there is nothing to estimate the "
                         "centering and scale from.")
    # The SCALE is the within-cell residual sd in BOTH arms, so `center` toggles exactly one
    # thing. Deriving it from the centered residual only when centering is on made the
    # uncentered arm divide by the TOTAL sd instead -- larger by the between-cell variance,
    # so the "control" also ran cooler: 1.03x / 1.13x / 1.37x sharper at the same beta as
    # that fraction went 0.05 / 0.20 / 0.50. An A/B that moves the temperature with the
    # treatment cannot attribute its own result.
    resid = a.copy()
    if strata is not None:
        s_all = np.asarray(strata)
        if s_all.shape[0] != a.shape[0]:
            raise ValueError(f"strata has length {s_all.shape[0]}, expected {a.shape[0]}")
        _, inv_all = np.unique(s_all, return_inverse=True)
        k_all = int(inv_all.max()) + 1
        c_all = np.bincount(inv_all[m], minlength=k_all).astype(np.float64)
        t_all = np.bincount(inv_all[m], weights=a[m], minlength=k_all)
        mu_all = np.where(c_all > 0, np.divide(t_all, np.maximum(c_all, 1.0)),
                          float(a[m].mean()))
        resid = a - mu_all[inv_all]
    sd = float(resid[m].std())

    z = a.copy()
    if center and strata is not None:
        s = np.asarray(strata)
        if s.shape[0] != a.shape[0]:
            raise ValueError(f"strata has length {s.shape[0]}, expected {a.shape[0]}")
        _, inv = np.unique(s, return_inverse=True)
        k = int(inv.max()) + 1
        cnt = np.bincount(inv[m], minlength=k).astype(np.float64)
        tot = np.bincount(inv[m], weights=a[m], minlength=k)
        # A cell with no fit row keeps the global fit mean: centering it on 0 would leave it
        # carrying the raw advantage's offset while every other cell had lost it.
        mean = np.where(cnt > 0, np.divide(tot, np.maximum(cnt, 1.0)), float(a[m].mean()))
        z = a - mean[inv]
    if sd > 0.0:
        z = z / sd
    w = awr_weights(z, beta=beta, w_max=w_max, standardize=False, fit=m)
    if balance and strata is not None:
        # Each cell's total weight -> its share of the FIT rows. Applied AFTER the clip, so
        # the clip still bounds a row within its own cell; a per-cell rescale is a positive
        # scalar per cell, so no row's rank inside its cell moves.
        s2 = np.asarray(strata)
        _, inv = np.unique(s2, return_inverse=True)
        k = int(inv.max()) + 1
        want = np.bincount(inv[m], minlength=k).astype(np.float64)
        have = np.bincount(inv, weights=w, minlength=k)
        scale = np.divide(want, have, out=np.ones(k), where=have > 0)
        w = w * scale[inv]
    return w


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's ``ESS = (sum w)^2 / sum w^2``, in rows.

    Args:
        weights: ``[n]`` non-negative weights.

    Returns:
        The effective sample size.  Divide by ``n`` for the fraction that is usually quoted;
        1.0 means uniform weights and a small fraction means the update is being driven by a
        handful of rows.
    """
    w = np.asarray(weights, dtype=np.float64)
    s2 = float((w ** 2).sum())
    return float((w.sum() ** 2) / s2) if s2 > 0 else 0.0


def reliability(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """How much of a quantity's variance is SIGNAL, from two independent estimates of it.

    Two cross-fit folds are two honest estimates of the same value, differing only in which
    episodes each was fitted on.  Under the classical model ``a = t + e_a``, ``b = t + e_b``
    with independent errors of equal variance, ``corr(a, b)`` IS the fraction of the observed
    variance that is the true quantity ``t`` -- the reliability coefficient.

    **This is the noise floor every ranking claim has to clear.**  A within-state spread that
    is mostly disagreement between two estimates of the same number cannot order anything, and
    no amount of averaging over states will make it.

    **The independence assumption is violated here, and it biases in a known direction.**  Two
    cross-fit folds share a frozen generator trunk, an architecture, the held-out tiles and the
    targets, so a component of their error is common.  Common error looks like common signal,
    which INFLATES this coefficient; equivalently, their disagreement misses what they get
    wrong together, which DEFLATES any noise floor built from it.  Both push the same way -- the
    critic looks better than it is -- so a reliability reported here is an upper bound.
    Measured on ``v4_big``'s held-out tiles, the errors of ``V_priv`` and ``V_onboard`` (which
    differ in their INPUTS as well as their split, hence a lower bound on two same-input folds)
    correlate at +0.87 on ``G_hat`` and +0.91 on ``p_fail``, at which a fold-difference noise
    floor understates the true error standard deviation by 2.8x and 3.4x.

    Args:
        a: ``[n]`` first estimate.
        b: ``[n]`` second estimate.

    Returns:
        ``pearson`` (the reliability coefficient), ``spearman``, ``rmse`` between the two,
        ``sd_a``, ``sd_b``, and ``signal_sd`` -- ``sqrt(pearson) * sqrt(sd_a * sd_b)``, the
        implied standard deviation of the true quantity.

    Raises:
        ValueError: If the arrays disagree in length.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"estimates disagree in length: {a.shape[0]} vs {b.shape[0]}")
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    sx, sy = float(x.std()), float(y.std())
    r = float(np.corrcoef(x, y)[0, 1]) if sx > 0 and sy > 0 else float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rs = (float(np.corrcoef(rx, ry)[0, 1])
          if rx.std() > 0 and ry.std() > 0 else float("nan"))
    return {"pearson": r, "spearman": rs,
            "rmse": float(np.sqrt(((x - y) ** 2).mean())),
            "sd_a": sx, "sd_b": sy,
            "signal_sd": float(np.sqrt(max(r, 0.0) * sx * sy))}


def normalized_progress(episode_uid: np.ndarray, replan_index: np.ndarray) -> np.ndarray:
    """Where each row sits in its own episode, on ``[0, 1]``.

    Computed from the row table's OWN last index per episode rather than from the episode
    table's ``n_replans``: the fit drops censored episodes and every window at or after the
    first bad step, so the recorded length and the surviving length are different numbers and
    using the wrong one puts every failed episode's last chunk at less than 1.0.

    Args:
        episode_uid: ``[W]``.
        replan_index: ``[W]``.

    Returns:
        ``[W]`` float64 in ``[0, 1]``; ``0`` for a single-chunk episode.
    """
    _, inv = np.unique(episode_uid, return_inverse=True)
    ri = np.asarray(replan_index, dtype=np.float64)
    last = np.zeros(inv.max() + 1 if inv.size else 0, dtype=np.float64)
    np.maximum.at(last, inv, ri)
    denom = last[inv]
    return np.where(denom > 0, ri / np.maximum(denom, 1.0), 0.0)


def clustered_bootstrap_ci(values: np.ndarray, cluster: np.ndarray, *,
                           statistic=np.mean, n_boot: int = 2000, seed: int = 0,
                           alpha: float = 0.05) -> tuple[float, float, float]:
    """A percentile CI that resamples CLUSTERS, not rows.

    Windows within an episode and episodes within a tile are strongly correlated, so a
    row-level interval is far too tight -- the eval harness measured an actual coverage of
    14-70% against a nominal 95% for exactly this reason.  Resampling whole clusters is what
    makes the interval mean what it says.

    **The values are NOT cast.** A statistic that needs more than one column per row -- an AUC
    gap, say, which needs two predictions and a label -- is expressed by passing ROW INDICES as
    the values and closing over the columns, and a float cast would break the indexing. So the
    array is passed through as given and only resampled.

    Args:
        values: ``[n]`` per-row values, or row indices when ``statistic`` closes over the
            columns it needs.
        cluster: ``[n]`` cluster id per row (a tile, or an episode).
        statistic: Callable over a 1-D array of the same dtype as ``values``.
        n_boot: Bootstrap draws.
        seed: RNG seed.
        alpha: Two-sided level.

    Returns:
        ``(point, lo, hi)``.
    """
    v = np.asarray(values)
    keys, inv = np.unique(cluster, return_inverse=True)
    by = [np.flatnonzero(inv == i) for i in range(keys.size)]
    rng = np.random.default_rng(seed)
    point = float(statistic(v))
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, keys.size, size=keys.size)
        idx = np.concatenate([by[i] for i in pick]) if keys.size else np.array([], int)
        draws[b] = statistic(v[idx]) if idx.size else np.nan
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
