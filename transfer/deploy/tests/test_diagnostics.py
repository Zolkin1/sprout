# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the run-recording helpers in ``deploy.diagnostics``.

These exist to make a recorded run trustworthy, so their own failure modes are quiet ones:
a checksum that is not order-sensitive would silently accept a permuted reconstruction, and
a depth-health rule that disagrees with ``normalize_depth_np`` would report a camera as
healthy on exactly the pixels the policy sees as blank.

Run: ``python transfer/deploy/tests/test_diagnostics.py``
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deploy.depth_obs import normalize_depth_np  # noqa: E402
from deploy.diagnostics import RateTracker, depth_health, obs_checksum  # noqa: E402

CAM = {"min_dist": 0.2, "max_dist": 9.0, "min_clip_val": 0.0, "max_clip_val": 0.0,
       "offset": 0.75, "scale": 0.5}


def test_checksum_is_order_sensitive_and_dtype_stable():
    """A permuted vector must not hash equal; float64 vs float32 input must."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(64).astype(np.float32)
    assert obs_checksum(v) == obs_checksum(v.astype(np.float64)), \
        "the digest must be over the float32 the network sees, not the caller's dtype"
    assert obs_checksum(v) == obs_checksum(v.reshape(8, 8)), "shape must not matter"
    swapped = v.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert obs_checksum(v) != obs_checksum(swapped), \
        "a permuted observation must not check out as identical"
    perturbed = v.copy()
    perturbed[7] += np.float32(1e-3)
    assert obs_checksum(v) != obs_checksum(perturbed)


def test_checksum_is_stable_across_processes():
    """Python's hash() is salted per process; a bag checked on another box needs this."""
    code = (
        "import sys, numpy as np; sys.path.insert(0, %r);"
        "from deploy.diagnostics import obs_checksum;"
        "print(obs_checksum(np.arange(16, dtype=np.float32)))"
        % os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    out = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip() for _ in range(2)]
    assert out[0] == out[1] == str(obs_checksum(np.arange(16, dtype=np.float32)))


def test_depth_health_matches_the_normalisation_validity_rule():
    """invalid_fraction must count exactly the pixels normalisation substitutes."""
    d = np.array([0.1, 0.2, 1.0, 5.0, 9.0, 9.5, np.nan, np.inf, -np.inf])
    h = depth_health(CAM, d)
    # Substituted by normalize_depth_np == not a real measurement.
    norm = normalize_depth_np(d, CAM["offset"], CAM["scale"], CAM["min_dist"],
                              CAM["max_dist"], CAM["min_clip_val"], CAM["max_clip_val"])
    substituted = np.isclose(norm, (CAM["min_clip_val"] - CAM["offset"]) * CAM["scale"]) | \
        np.isclose(norm, (CAM["max_clip_val"] - CAM["offset"]) * CAM["scale"])
    assert h["invalid_fraction"] == substituted.mean(), (h["invalid_fraction"], substituted.mean())
    # Range stats over the valid pixels only: 0.2, 1.0, 5.0, 9.0 (bounds are inclusive).
    assert h["min"] == 0.2 and h["max"] == 9.0
    assert np.isclose(h["mean"], np.mean([0.2, 1.0, 5.0, 9.0]))


def test_depth_health_on_an_all_invalid_frame():
    """The failure this exists to catch: a camera publishing nothing but blanks."""
    h = depth_health(CAM, np.full(780, np.nan))
    assert h["invalid_fraction"] == 1.0
    assert np.isnan(h["min"]) and np.isnan(h["max"]) and np.isnan(h["mean"]), \
        "an all-invalid frame must report NaN range, not a plausible-looking 0"


def test_rate_tracker_measures_hz_and_jitter():
    """Distinguishes a slow stream from a stopped one, and bursty from steady."""
    r = RateTracker(window=10)
    assert r.hz() == 0.0 and r.jitter() == 0.0, "inert before two arrivals"
    for i in range(11):
        r.tick(i * 0.02)
    assert np.isclose(r.hz(), 50.0)
    assert np.isclose(r.jitter(), 0.0, atol=1e-12)

    bursty = RateTracker(window=10)
    t = 0.0
    for i in range(11):
        t += 0.01 if i % 2 else 0.03     # same mean, alternating
        bursty.tick(t)
    assert np.isclose(bursty.hz(), 50.0, atol=1.0)
    assert bursty.jitter() > 0.005, "a bursty stream must not look steady"


def main() -> None:
    """Run every check in this module."""
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("DIAGNOSTICS CHECKS PASSED")


if __name__ == "__main__":
    main()
