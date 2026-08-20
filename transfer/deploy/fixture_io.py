# Copyright 2026 Zachary Olkin. All rights reserved.

"""The on-disk fixture format: a manifest plus raw little-endian float64 blobs.

ONE writer for a format with a C++ reader on the other side
(``transfer/deploy_cpp/test/fixture.{h,cc}``). It was inline in
``tools/dump_parity_fixtures.py`` until a second producer needed it -- the per-tick replay
stream -- and a second producer of the same format is how the format drifts. The reader
validates ``format`` and ``byteorder`` and nothing else, so a writer that quietly changed
its layout would be caught only by the values coming out wrong.

NOT ``.npz``: reading that from C++ needs a zip reader and an npy parser, which is a lot of
surface for "an array on disk".

Tolerances are NOT defined here. They belong with whatever knows what "agrees" means for a
given comparison, and they travel in the manifest so both languages read one number.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Any

import numpy as np

#: Manifest format version. Bump when the on-disk layout changes, so an old C++ reader
#: refuses a new fixture set instead of misreading it.
MANIFEST_FORMAT = 1


class FixtureWriter:
    """Collects named arrays into blobs plus a manifest.

    Args:
        out_dir: Directory to write ``manifest.json`` and the ``.bin`` blobs into.
        generator: What to record as having produced this set, for the manifest.
    """

    def __init__(self, out_dir: str, generator: str) -> None:
        self.out_dir = out_dir
        self.generator = generator
        self.cases: list[dict[str, Any]] = []
        os.makedirs(out_dir, exist_ok=True)

    def _blob(self, case: str, role: str, name: str, value: np.ndarray) -> dict[str, Any]:
        """Write one array and return its manifest entry.

        Args:
            case: Case name, used in the file name.
            role: ``"in"`` or ``"out"``.
            name: Argument or output name.
            value: The array; cast to float64.

        Returns:
            The manifest entry describing the blob.
        """
        arr = np.ascontiguousarray(value, dtype=np.float64)
        rel = f"{case}.{role}.{name}.bin"
        raw = arr.tobytes(order="C")
        with open(os.path.join(self.out_dir, rel), "wb") as fh:
            fh.write(raw)
        return {
            "name": name,
            "file": rel,
            "shape": list(arr.shape),
            "dtype": "float64",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def add(self, case: str, fn: str, inputs: dict[str, np.ndarray],
            outputs: dict[str, np.ndarray], atol: float, compare: str,
            attrs: dict[str, Any] | None = None) -> None:
        """Record one case's inputs and the oracle's outputs.

        Args:
            case: Unique case name.
            fn: What the C++ side calls the thing under test.
            inputs: Argument name -> array, in the order the function takes them.
            outputs: Output name -> array.
            atol: Absolute tolerance the C++ must match this case to.
            compare: How to compare -- the C++ side's vocabulary (e.g. ``"elementwise"``).
            attrs: Scalars and structure the C++ side needs (block lists, joint counts,
                names). Carried verbatim in the manifest.
        """
        self.cases.append({
            "case": case,
            "fn": fn,
            "atol": atol,
            "compare": compare,
            "attrs": attrs or {},
            "inputs": [self._blob(case, "in", k, v) for k, v in inputs.items()],
            "outputs": [self._blob(case, "out", k, v) for k, v in outputs.items()],
        })

    def write_manifest(self, seed: int, extra: dict[str, Any] | None = None) -> str:
        """Write ``manifest.json`` and return its path.

        Args:
            seed: The RNG seed the set was generated with, for reproducibility.
            extra: Additional top-level keys (the replay stream records the artifacts it
                came from here).

        Returns:
            Path to the manifest.

        Raises:
            RuntimeError: On a big-endian host. The blobs are raw little-endian and the
                reader checks the field, but failing at WRITE time names the real problem.
        """
        if sys.byteorder != "little":
            raise RuntimeError(
                f"fixtures are little-endian raw blobs; this host is {sys.byteorder}. "
                "Writing them here would produce a set the C++ reader refuses.")
        path = os.path.join(self.out_dir, "manifest.json")
        payload = {
            "format": MANIFEST_FORMAT,
            "generator": self.generator,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": f"{platform.system()}-{platform.machine()}",
            "byteorder": sys.byteorder,
            "seed": seed,
            "cases": self.cases,
        }
        payload.update(extra or {})
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        return path
