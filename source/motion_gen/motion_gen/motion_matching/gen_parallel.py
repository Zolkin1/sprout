# Copyright 2026 Zachary Olkin. All rights reserved.

"""Fork-based worker pool for generating clip records in parallel.

The per-clip DP solves are INDEPENDENT over a SHARED read-only matcher / candidate graph: building the
matcher is a fixed per-family cost, but the solves are embarrassingly parallel.  This forks a pool
AFTER the matcher is built, so each worker inherits it via copy-on-write (no rebuild, no pickling of
the big feat_raw / graph arrays -- only the small ``units`` go to children and only small record dicts
come back), and pins each worker to ONE BLAS/torch thread (else N processes x M threads oversubscribe
the cores and thrash).

Contract: ``worker_fn(unit) -> list[record]`` and any big read-only state it reads must live in the
CALLER's module (inherited by fork -- pass it via a module global set before calling, NOT as an arg).
Returns the per-unit record lists flattened IN INPUT ORDER.  ``n_workers <= 1`` runs serially.
"""

from __future__ import annotations

import multiprocessing as mp
import os


def _init_worker() -> None:
    """Pin the worker to a single math thread (env + torch) so the fork pool doesn't oversubscribe."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def unit_pool(worker_fn, units, n_workers: int):
    """Map ``worker_fn`` over ``units`` across ``n_workers`` FORKED processes; flatten the per-unit
    record lists in order.  Sets the parent's torch threads to 1 first (before the fork) to avoid the
    torch fork+threadpool deadlock.  ``n_workers <= 1`` (or a single unit) runs serially in-process."""
    units = list(units)
    try:
        import torch
        torch.set_num_threads(1)                     # parent, BEFORE fork -> children inherit it too
    except Exception:
        pass
    if n_workers <= 1 or len(units) <= 1:
        _init_worker()
        return [rec for u in units for rec in worker_fn(u)]
    ctx = mp.get_context("fork")                     # fork => children share the matcher read-only (COW)
    with ctx.Pool(processes=int(n_workers), initializer=_init_worker) as pool:
        per_unit = pool.map(worker_fn, units)
    return [rec for lst in per_unit for rec in lst]


def default_workers(n_units: int, requested: int | None = None) -> int:
    """Worker count: ``requested`` if given (>0), else all cores minus 2 (leave headroom), capped by
    the number of units and floored at 1."""
    if requested is not None and requested > 0:
        w = int(requested)
    else:
        w = max(1, (os.cpu_count() or 2) - 2)
    return max(1, min(w, int(n_units)))
