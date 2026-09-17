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

**COW is not free of the WORK.** What a worker only READS is shared; what it WRITES is its own, and
for a solve whose scratch is sized by the DB (the backward DP's ``[T,N]`` action table) that private
part is gigabytes EACH.  A core-count pool then asks for many times the machine's RAM, and the
answer is the global OOM killer -- which takes whatever else is running with it, since it picks by
size and a browser is the second-biggest thing on a desktop.  So ``default_workers`` takes a
``bytes_per_worker`` and caps the width by ``MemAvailable``; the estimate comes from the object that
does the allocating (``matcher.solve_bytes(T)``) rather than from a constant restated per caller.
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


# A forked worker's own cost beyond the arrays the solve declares: its share of the interpreter and
# the imported extension modules, the allocator arenas it opens, the COW faults it takes on the
# parent's small objects, and the playback/output stage that runs after the search.  Counted per
# worker because that is its dimension.
#
# MEASURED, not assumed: over a 30-clip DP run (6 workers, T=1000, N=5.48M) the peak
# Private_Clean+Private_Dirty per worker was 9.6 GiB against the 7.66 GiB DPMatcher.solve_bytes
# accounts for, so the residual is ~2 GiB.  An earlier 512 MiB guess would have let a 10-worker pool
# past the cap at ~106 GiB on a 125 GiB box -- i.e. straight back into the OOM this guard exists to
# prevent.  Keep solve_bytes a clean statement of the solve's own arrays and carry the slack here.
_WORKER_OVERHEAD_BYTES = 2 << 30


def mem_available_bytes() -> int | None:
    """``MemAvailable`` from ``/proc/meminfo`` in bytes, or ``None`` where it cannot be read.

    ``MemAvailable`` and not ``MemFree``: reclaimable page cache is available to a new worker, and
    on a box that has just read a multi-GB bones cache almost all of the "used" memory is that."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def default_workers(n_units: int, requested: int | None = None, *,
                    bytes_per_worker: int | None = None, reserve_bytes: int = 8 << 30,
                    log=None) -> int:
    """Worker count: ``requested`` if given (>0), else all cores minus 2 (leave headroom), capped by
    the number of units and floored at 1.

    ``bytes_per_worker`` (from :meth:`~motion_gen.motion_matching.vanilla.VanillaMatcher.solve_bytes`)
    additionally caps the width so the pool fits ``MemAvailable`` less ``reserve_bytes``.  That cap
    overrides an EXPLICIT ``requested`` too, loudly (via ``log``) rather than silently: a request the
    machine cannot hold does not become a slower run, it becomes the OOM killer choosing among every
    process on the box.  Reporting the cap is what lets the caller see that a 32-core machine is
    running 5 workers on purpose.  Unreadable ``MemAvailable`` (not Linux) applies no cap."""
    if requested is not None and requested > 0:
        w = int(requested)
    else:
        w = max(1, (os.cpu_count() or 2) - 2)
    if bytes_per_worker is not None and bytes_per_worker > 0:
        avail = mem_available_bytes()
        if avail is not None:
            per = int(bytes_per_worker) + _WORKER_OVERHEAD_BYTES
            fits = max(1, int((avail - reserve_bytes) // per))
            if fits < w and log is not None:
                log(f"capping {w} -> {fits} worker(s): {per / 2**30:.1f} GiB each against "
                    f"{avail / 2**30:.0f} GiB available less a {reserve_bytes / 2**30:.0f} GiB "
                    f"reserve (the pool would not fit in RAM)")
            w = min(w, fits)
    return max(1, min(w, int(n_units)))
