# Copyright 2026 Zachary Olkin. All rights reserved.

"""Small torch-free helpers for recording a deployment run.

Lives in ``deploy`` rather than in the ROS node because both consumers want the same
answers and must not compute them two ways: an observation checksum is only useful if the
producer and the offline reconstruction agree on the algorithm, and a depth-health number
is only comparable between MuJoCo and the robot if both derive it identically.

numpy + stdlib only, like the rest of the package.
"""

from __future__ import annotations

import hashlib

import numpy as np



def obs_checksum(obs: np.ndarray) -> int:
    """Order-sensitive 64-bit checksum of an observation vector.

    The point is to make an offline reconstruction PROVABLE. Everything needed to rebuild
    the policy input is recorded, but rebuilding it re-runs the same assembly the
    controller ran, and a bug in that replay looks exactly like a bug in the controller.
    Comparing this checksum settles which one you are looking at.

    Hashes the float32 bytes, because float32 is what the network is fed -- hashing float64
    would fail a reconstruction that differs only in dtype, which is not a difference the
    policy can see.

    blake2b rather than Python's ``hash()`` (randomised per process, so two runs of
    identical data disagree) and rather than a hand-rolled FNV loop (correct, but serial
    Python over ~12 kB every 20 ms). blake2b is a C implementation and takes a few
    microseconds at this size, which is what makes it affordable inside the control tick.

    Args:
        obs: The observation, any shape; flattened in C order.

    Returns:
        A 64-bit digest of the float32 representation. Deterministic across processes and
        machines, so a bag recorded on the robot can be checked on a laptop.
    """
    data = np.ascontiguousarray(obs, dtype=np.float32).tobytes()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def depth_health(spec: dict, frame: np.ndarray) -> dict[str, float]:
    """Per-camera statistics that reveal a silently degraded depth stream.

    Staleness -- the only thing checked today -- catches a camera that stopped. It says
    nothing about one that is still publishing garbage: an all-invalid frame, a wrong
    exposure, half the image saturated. Those read as perfectly healthy to a watchdog and
    are indistinguishable from real terrain to the policy.

    Args:
        spec: One camera's entry from ``tracker.onnx``'s ``depth_camera_spec``. Its
            ``min_dist`` / ``max_dist`` are the same validity band
            ``depth_obs.normalize_depth_np`` applies, so ``invalid_fraction`` counts exactly
            the pixels the policy will see substituted rather than measured.
        frame: Raw depth for that camera, metres along the optical axis, any shape.

    Returns:
        ``{"min", "max", "mean", "invalid_fraction"}``. The range statistics are over VALID
        pixels only, so a frame that is entirely invalid reports NaN range and an
        ``invalid_fraction`` of 1.0 rather than a misleading 0.
    """
    d = np.asarray(frame, dtype=np.float64).ravel()
    # Same rule and the same strict comparisons as normalize_depth_np: too near, too far,
    # or non-finite. errstate because comparing a NaN pixel is the case being classified,
    # not a warning worth printing 50 times a second.
    with np.errstate(invalid="ignore"):
        bad = (~np.isfinite(d)) | (d < float(spec["min_dist"])) | (d > float(spec["max_dist"]))
    good = d[~bad]
    return {
        "min": float(good.min()) if good.size else float("nan"),
        "max": float(good.max()) if good.size else float("nan"),
        "mean": float(good.mean()) if good.size else float("nan"),
        "invalid_fraction": float(bad.mean()) if bad.size else 1.0,
    }


class RateTracker:
    """Measured publish rate and inter-arrival jitter of a stream.

    A camera configured for 50 Hz that is actually delivering 12 Hz is a normal failure and
    is invisible to a staleness timeout tuned for 4 Hz. Rate and jitter are what
    distinguish "slow" from "stopped", and jitter additionally catches a stream that
    averages correctly while arriving in bursts -- which a policy conditioned on it feels
    as intermittently stale data.

    Args:
        window: Inter-arrival samples to average over. 25 is half a second at 50 Hz:
            long enough to be steady, short enough to show a dropout while it is happening.
    """

    def __init__(self, window: int = 25) -> None:
        self._window = int(window)
        self._last: float | None = None
        self._dt: list[float] = []

    def tick(self, now: float) -> None:
        """Record an arrival.

        Args:
            now: Receipt time in seconds. LOCAL receipt, not a header stamp -- a camera's
                clock is not synchronised to this machine, so a header stamp cannot say
                whether the stream is still alive here.
        """
        if self._last is not None:
            self._dt.append(now - self._last)
            if len(self._dt) > self._window:
                self._dt.pop(0)
        self._last = now

    def hz(self) -> float:
        """Measured rate, or 0.0 before two arrivals have been seen."""
        if not self._dt:
            return 0.0
        mean = float(np.mean(self._dt))
        return 1.0 / mean if mean > 0 else 0.0

    def jitter(self) -> float:
        """Standard deviation of the inter-arrival time, in seconds."""
        return float(np.std(self._dt)) if len(self._dt) > 1 else 0.0
