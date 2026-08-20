# Copyright 2026 Zachary Olkin. All rights reserved.

"""Diff what the C++ runtime computed against what python computed, tick by tick.

Reads the stream python recorded (``deploy.replay_stream``) and the raw blobs the C++
``replay`` tool wrote beside it, and reports a DISTRIBUTION rather than a first failure --
because in teacher-forced mode every tick is an independent test, so "3 of 300 ticks differ,
all of them re-plan ticks" localises a fault and "first divergence at tick 47" does not.

WHAT COUNTS AS AGREEMENT is not one number:

* the OBSERVATION is our own arithmetic with the window held fixed, so it should be exact to
  rounding (``ATOL_EXACT``);
* the ACTION and the generator's plan have passed through onnxruntime, and python runs 1.22
  against the container's 1.24.4 -- different builds of one graph do not agree bit for bit,
  so those are bounded by ``ATOL_NETWORK`` and the real number is whatever this prints.

The ``obs_checksum`` is reported but never decides anything: it is bitwise, so it goes red on
a last-bit difference that the tolerance rightly forgives. It is here to distinguish "the same
vector" from "a vector that agrees to 1e-7", which is a genuinely useful thing to know.

    python transfer/deploy/tools/compare_replay.py --stream /tmp/replay_forced
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_TRANSFER = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TRANSFER)

from deploy.replay_stream import ATOL_EXACT, ATOL_NETWORK, join_checksum, load  # noqa: E402


def _read_cpp(stream: str, name: str, ticks: int, width: int) -> np.ndarray:
    """Read one raw C++ blob and shape it.

    Args:
        stream: Stream directory.
        name: Field name, as ``replay.cc`` wrote it.
        ticks: Expected rows.
        width: Expected values per row.

    Returns:
        ``[ticks, width]``.

    Raises:
        FileNotFoundError: If the C++ side has not been run against this stream.
        ValueError: On a size that disagrees with the expectation -- a partial run, which must
            not be compared as though it were complete.
    """
    path = os.path.join(stream, f"cpp.{name}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} is missing; run deploy_cpp's `replay {stream} --tracker ... --fm ...` first")
    raw = np.fromfile(path, dtype=np.float64)
    if raw.size != ticks * width:
        raise ValueError(f"{path} holds {raw.size} values, expected {ticks}x{width}; the C++ "
                         "run did not finish over this stream")
    return raw.reshape(ticks, width)


def _report(label: str, a: np.ndarray, b: np.ndarray, atol: float,
            replanned: np.ndarray) -> bool:
    """Print one field's divergence distribution.

    Args:
        label: Field name for the report.
        a: The python values, ``[ticks, width]``.
        b: The C++ values.
        atol: Tolerance for "agrees".
        replanned: ``[ticks]`` boolean, for splitting the count by re-plan tick.

    Returns:
        True if every tick agrees.
    """
    err = np.abs(a - b)
    per_tick = err.max(axis=1)
    bad = per_tick > atol
    print(f"  {label:8s} max={per_tick.max():.3e}  p99={np.percentile(per_tick, 99):.3e}  "
          f"median={np.median(per_tick):.3e}  tol={atol:.0e}")
    if bad.any():
        idx = np.flatnonzero(bad)
        on_replan = int(replanned[idx].sum())
        print(f"           OVER TOLERANCE on {bad.sum()}/{len(bad)} ticks "
              f"({on_replan} of them re-plan ticks); first at {idx[0]}, worst at "
              f"{int(np.argmax(per_tick))}")
        # Which columns, so an observation divergence can be attributed to a block.
        worst_col = int(np.unravel_index(np.argmax(err), err.shape)[1])
        print(f"           widest column: {worst_col} of {a.shape[1]}")
    return not bad.any()


def main() -> int:
    """Compare and return 0 when every field agrees within its tolerance."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream", required=True, help="Directory holding both sides' output.")
    args = ap.parse_args()

    py, manifest = load(args.stream)
    ticks = int(manifest["cases"][0]["attrs"]["ticks"])
    replanned = py["replanned"].ravel() > 0.5
    print(f"[compare] {ticks} ticks, {int(replanned.sum())} of them re-plan ticks")
    print(f"[compare] stream from {manifest['artifacts']['tracker_onnx']}")

    ok = True
    print("[compare] our own arithmetic (should be exact):")
    ok &= _report("obs", py["obs"], _read_cpp(args.stream, "obs", ticks, py["obs"].shape[1]),
                  ATOL_EXACT, replanned)
    print("[compare] downstream of a network (bounded by the two ORT builds):")
    ok &= _report("action", py["action"],
                  _read_cpp(args.stream, "action", ticks, py["action"].shape[1]),
                  ATOL_NETWORK, replanned)
    meta = _read_cpp(args.stream, "meta", ticks, 6)
    forced = bool(meta[0][4] > 0.5)
    cpp_window = _read_cpp(args.stream, "window", ticks, py["window"].shape[1])
    if forced:
        # The C++ served ITS OWN plan while observing the recorded one, so the two plans
        # differ by construction -- and they would differ anyway, because the C++ requests
        # each plan a lead ahead of the boundary and conditions it on exteroception from then.
        # Reported for scale, never as a verdict.
        err = np.abs(py["window"] - cpp_window).max(axis=1)
        print(f"  window   max={err.max():.3e}  median={np.median(err):.3e}  "
              "INFORMATIONAL (teacher-forced: this is the C++'s own free-running plan, not a "
              "comparison of like with like -- run without --forced, at lead 0, to compare "
              "plans)")
    else:
        ok &= _report("window", py["window"], cpp_window, ATOL_NETWORK, replanned)

    # history_null: the flag the generator was asked with, per side. A disagreement here is a
    # different function, not a tolerance question, so it is reported before the numbers.
    py_null = py["fm_history_null"].ravel()
    cpp_null = meta[:, 5]
    disagree = np.flatnonzero((py_null != cpp_null) & (py_null >= 0) & (cpp_null >= 0))
    print(f"[compare] history_null: python True on {int((py_null == 1).sum())} ticks, "
          f"C++ True on {int((cpp_null == 1).sum())}; "
          + (f"DISAGREE on {len(disagree)} ticks, first at {disagree[0]}"
             if len(disagree) else "agree everywhere they are both defined"))

    cpp_sum = _read_cpp(args.stream, "checksum", ticks, 2)
    same = sum(join_checksum(cpp_sum[t][0], cpp_sum[t][1])
               == join_checksum(py["obs_checksum_hi"][t][0], py["obs_checksum_lo"][t][0])
               for t in range(ticks))
    print(f"[compare] obs_checksum identical on {same}/{ticks} ticks "
          "(bitwise; informational only)")

    print("[compare] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
