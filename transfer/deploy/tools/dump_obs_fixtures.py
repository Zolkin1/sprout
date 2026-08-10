# Copyright 2026 Zachary Olkin. All rights reserved.

"""Freeze what python's ``_assemble`` produces, over the REAL tracker's term table.

The observation assembly is the last piece of numeric parity in the C++ port and the most
order-dependent. Per term the python casts to float32 FIRST, then clips in float32, then pushes
through the history window, then scales -- and a per-element scale tiles across that window.
Any reordering still yields a correctly sized vector with every element subtly different, which
is exactly the kind of difference no shape check finds.

So this drives the real python code (``GenTrackerPolicy._assemble`` and ``_push_history``, bound
to a stub that supplies only the term table and the history dict) over several consecutive steps,
and records both the inputs and the exact float32 output. The C++ has to reproduce it bit for
bit -- float32 values are stored widened to float64, which is lossless, so the comparison is
``==`` and not a tolerance.

Includes a step with NaN and +/-inf in a clipped term. That is the case that matters most:
``np.clip`` propagates NaN, and the non-finite guard downstream depends on it surviving as far as
``_finite_targets``. Eigen's ``cwiseMin`` would turn it into the clip bound instead.

Also records ``obs_checksum`` per step, so the C++ blake2b (digest_size=8, which is not a
truncation of blake2b-512) is pinned against hashlib.

Needs the metadata fixture; run ``dump_metadata_fixtures.py`` first. Skips cleanly without it.

Run::

    python transfer/deploy/tools/dump_metadata_fixtures.py
    python transfer/deploy/tools/dump_obs_fixtures.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy.diagnostics import obs_checksum  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402


class _AssembleStub:
    """The minimum ``_assemble`` needs: the term table, the history dict and num_obs.

    Borrows the REAL methods rather than reimplementing them -- the point is to record what the
    shipping code does, so a copy here would defeat the exercise.
    """

    _push_history = GenTrackerPolicy._push_history
    _assemble = GenTrackerPolicy._assemble

    def __init__(self, terms: list[dict], num_obs: int) -> None:
        self._trk = SimpleNamespace(get_obs_terms=lambda: terms)
        self._obs_history: dict[str, list[np.ndarray]] = {}
        self._num_obs = num_obs

    def get_num_obs(self) -> int:
        """Width of the assembled vector."""
        return self._num_obs


def _step_values(terms: list[dict], rng: np.random.Generator, step: int,
                 poison: bool) -> dict[str, np.ndarray]:
    """One step's per-term single-sample values.

    Magnitudes deliberately exceed the terms' clips so the clipping path is exercised rather
    than skipped, and vary per step so the history window rolls visibly.
    """
    values: dict[str, np.ndarray] = {}
    for t in terms:
        hist = int(t["history_length"])
        step_dim = t["shape"] // hist if hist > 0 else t["shape"]
        v = rng.standard_normal(step_dim) * (5.0 + step)
        if poison and t["name"] == "actions":
            # The case the guard chain depends on. Also +/-inf, which must clip to the bounds
            # while NaN must NOT.
            v[0] = np.nan
            v[1] = np.inf
            v[2] = -np.inf
        values[t["name"]] = v
    return values


def main() -> int:
    """Write the obs fixtures. Returns a process exit code."""
    default_meta = os.path.join(_TRANSFER, "deploy", "tests", "fixtures", "metadata")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default=default_meta)
    ap.add_argument("--out", default=os.path.join(_TRANSFER, "deploy", "tests", "fixtures", "obs"))
    ap.add_argument("--steps", type=int, default=14, help="more than any term's history length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    expected_path = os.path.join(args.metadata, "tracker_expected.json")
    if not os.path.isfile(expected_path):
        print(f"[obs] no metadata fixture under {args.metadata}; run dump_metadata_fixtures.py "
              "first (this is not an error -- the C++ test skips without it).")
        return 0

    with open(expected_path) as fh:
        meta = json.load(fh)
    terms = meta["obs_terms"]
    num_obs = int(meta["num_obs"])

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    stub = _AssembleStub(terms, num_obs)

    steps = []
    for step in range(args.steps):
        poison = step == args.steps - 1          # last step carries the non-finite values
        values = _step_values(terms, rng, step, poison)
        obs = stub._assemble({k: v.copy() for k, v in values.items()})   # (1, num_obs) float32
        flat = np.ascontiguousarray(obs, dtype=np.float32).ravel()

        entry = {"step": step, "poison": poison, "terms": [], "checksum": str(obs_checksum(flat))}
        for name, v in values.items():
            rel = f"step{step:02d}.in.{name}.bin"
            np.ascontiguousarray(v, dtype=np.float64).tofile(os.path.join(args.out, rel))
            entry["terms"].append({"name": name, "file": rel, "size": int(v.size)})
        rel = f"step{step:02d}.out.obs.bin"
        # float32 widened to float64: lossless, so the C++ comparison stays an equality.
        np.ascontiguousarray(flat, dtype=np.float64).tofile(os.path.join(args.out, rel))
        entry["obs"] = {"file": rel, "size": int(flat.size)}
        steps.append(entry)

    manifest = {
        "format": 1,
        "generator": "transfer/deploy/tools/dump_obs_fixtures.py",
        "numpy": np.__version__,
        "byteorder": sys.byteorder,
        "num_obs": num_obs,
        "steps": steps,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[obs] {len(steps)} steps x {num_obs} values -> {args.out}")
    print(f"[obs] last step carries NaN/+inf/-inf in 'actions'; checksum[0]={steps[0]['checksum']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
