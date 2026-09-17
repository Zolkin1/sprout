# Copyright 2026 Zachary Olkin. All rights reserved.

"""The fixed knot -> node interpolation the adjuster's action is expressed in.

The adjuster adds a residual to the generator's ``F``-node reference window. Emitted at full
width that is ``F * D`` = 2728 dimensions driven by one scalar advantage per decision, and
measured over 840 iterations of run v6 the result was ~97% exploration noise and ~3% learned
mean: with ~3 samples per dimension per iteration and a per-node KL cap, neither the mean nor
the per-dimension stds could move.

So the action becomes ``K`` KNOTS along the node axis, lifted to the full window by a fixed
linear map ``B`` (``r = B u``, applied per channel). At ``K = 13`` that is 572 dimensions,
4.8x fewer, and the interpolation makes the residual smooth along the horizon by
construction rather than by a penalty.

**The basis is justified by measurement, not by taste.** Fitting ``d = authored - generated``
over 512 open-loop validation windows of the deployed generator
(``scripts/flow_matching/knot_basis_fit.py``), the fraction of the correction's energy a
``K = 13`` Catmull-Rom basis captures is:

    base_pos 0.9922   base_ori 0.9907   joint_pos 0.9903
    base_lin_vel 0.9265   base_ang_vel 0.8517   ALL 0.9338

The three blocks that decide where the pelvis and feet go are all >= 0.99. The two weak ones
are the velocity blocks, and that is expected rather than a shortfall: this repo's own
error-structure measurement puts ``ang_vel``'s energy at 0.04/0.11 in {const, const+ramp},
i.e. SUB-white, so the part ``B`` cannot represent there is high-frequency noise and not a
learnable correction. ``K = 21`` would take ALL to 0.9853 for 62% more action dimensions,
spent mostly on that noise.

**Knots span the WHOLE window, endpoints included** -- deliberately NOT the generator's own
command-token positions ``(0, 5, ..., 60)``. Those stop one node short of a 62-node window,
so the last node had to be clamped or extrapolated, and clamping measured 0.0485 of
unexplained RMS there against ~0.021 elsewhere. Spanning ``[0, F - 1]`` makes every node an
interpolation and cut that to 0.0381. Aligning with the command tokens would buy nothing
anyway: the knots index the PREDICTED window, and which node the tracker reads slides forward
every control step of the re-plan interval, so no fixed alignment survives the chunk.
"""

from __future__ import annotations

import numpy as np

#: Knot count the adjuster ships with. See the module docstring for the fit that chose it.
DEFAULT_NUM_KNOTS: int = 13

#: Interpolation kinds :func:`knot_basis` accepts.
KNOT_KINDS: tuple[str, ...] = ("catmull", "linear")


def knot_positions(future_len: int, num_knots: int = DEFAULT_NUM_KNOTS) -> np.ndarray:
    """``num_knots`` knot positions spanning ``[0, future_len - 1]``, endpoints included.

    Positions are real-valued and generally not integers (at ``F = 62, K = 13`` the spacing
    is 5.083 nodes); :func:`knot_basis` handles that directly.

    Args:
        future_len: Node count of the window.
        num_knots: How many knots.

    Returns:
        ``[num_knots]`` ascending positions, first ``0.0`` and last ``future_len - 1``.

    Raises:
        ValueError: If ``num_knots < 2`` (cannot span) or exceeds ``future_len`` (more knots
            than nodes makes the basis rank-deficient and the fit under-determined).
    """
    if num_knots < 2:
        raise ValueError(f"num_knots={num_knots} must be >= 2 to span the window")
    if num_knots > future_len:
        raise ValueError(
            f"num_knots={num_knots} exceeds future_len={future_len}: there would be more "
            "knots than nodes, so the basis is rank-deficient and the residual it can "
            "express is not what the extra knots suggest."
        )
    return np.linspace(0.0, float(future_len - 1), int(num_knots))


def knot_basis(future_len: int, num_knots: int = DEFAULT_NUM_KNOTS,
               kind: str = "catmull") -> np.ndarray:
    """The fixed map ``B`` with ``r = B u``: ``[future_len, num_knots]``.

    Every row sums to 1, so a constant knot vector gives a constant window -- which is what
    makes a zero-initialised head produce exactly the generator's own plan.

    Catmull-Rom duplicates the end knots, giving a zero-second-derivative end condition. Its
    weights include negative lobes (``min B = -0.0741`` at ``F=62, K=13``) so it OVERSHOOTS:
    the worst row has L1 norm **1.2499**, i.e. an interpolated node can reach 1.25x the
    largest knot magnitude. The ``tanh`` is applied AFTER interpolation, so the per-channel
    cap still binds exactly; the only consequence is that non-knot nodes are squashed
    slightly harder than knots. Linear has all-positive weights and L1 exactly 1.0, and is
    the option to take if that ever matters more than smoothness.

    Args:
        future_len: Node count of the window.
        num_knots: How many knots.
        kind: One of :data:`KNOT_KINDS`.

    Returns:
        ``[future_len, num_knots]`` float64.

    Raises:
        ValueError: On an unknown ``kind``, or via :func:`knot_positions`.
    """
    if kind not in KNOT_KINDS:
        raise ValueError(f"kind={kind!r}; expected one of {KNOT_KINDS}")
    knots = knot_positions(future_len, num_knots)
    k = len(knots)
    b = np.zeros((future_len, k), dtype=np.float64)
    for n in range(future_len):
        if n >= knots[-1]:
            b[n, -1] = 1.0
            continue
        j = int(np.searchsorted(knots, n, side="right") - 1)
        t = (n - knots[j]) / (knots[j + 1] - knots[j])
        if kind == "linear":
            b[n, j] = 1.0 - t
            b[n, j + 1] = t
            continue
        i0, i1, i2, i3 = max(j - 1, 0), j, j + 1, min(j + 2, k - 1)
        t2, t3 = t * t, t * t * t
        for idx, w in ((i0, -0.5 * t3 + t2 - 0.5 * t),
                       (i1, 1.5 * t3 - 2.5 * t2 + 1.0),
                       (i2, -1.5 * t3 + 2.0 * t2 + 0.5 * t),
                       (i3, 0.5 * t3 - 0.5 * t2)):
            b[n, idx] += w
    return b


def knot_sample_matrix(future_len: int, num_knots: int = DEFAULT_NUM_KNOTS) -> np.ndarray:
    """The node -> knot sampler ``S``: read a window's own value AT each knot position.

    The other direction from :func:`knot_basis`, and needed because each knot query token is
    embedded from the plan's value there. Linear interpolation, since the knots sit at
    fractional node positions and this is an input feature rather than the residual itself.

    NOT ``knot_basis``'s transpose or pseudo-inverse: ``B.T`` does not have unit row sums (so
    it would scale the feature), and the pseudo-inverse is a least-squares DECONVOLUTION,
    which would hand the token a fitted coefficient rather than the window's local value.

    Args:
        future_len: Node count of the window.
        num_knots: How many knots.

    Returns:
        ``[num_knots, future_len]`` float64, rows summing to 1.
    """
    knots = knot_positions(future_len, num_knots)
    s = np.zeros((len(knots), future_len), dtype=np.float64)
    for k, p in enumerate(knots):
        lo = int(np.floor(p))
        hi = min(lo + 1, future_len - 1)
        frac = float(p - lo)
        if hi == lo:
            s[k, lo] = 1.0
        else:
            s[k, lo] = 1.0 - frac
            s[k, hi] = frac
    return s
