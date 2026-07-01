# Copyright 2026 Zachary Olkin. All rights reserved.

"""Numpy rolling history buffer for the deployment generator inference.

Numpy port of ``transfer/fm/online.BatchedHistory`` (reset-aware, fixed length).
A window becomes valid once an env has ``maxlen`` uninterrupted steps since its
last reset; ``fill`` warm-starts every slot for deployment (no clip warmup).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class BatchedHistory:
    """Fixed-length rolling history of raw world-frame state for ``num_envs`` envs."""

    def __init__(self, num_envs: int, maxlen: int):
        """Create an empty history for ``num_envs`` envs of length ``maxlen``."""
        self.maxlen = maxlen
        self._steps: deque[dict[str, np.ndarray]] = deque(maxlen=maxlen)
        self._steps_since_reset = np.zeros(num_envs, dtype=np.int64)

    def append(self, step: dict[str, np.ndarray]) -> None:
        """Append one step. ``step`` maps each tracked key to an ``(N, ·)`` array."""
        self._steps.append(step)
        self._steps_since_reset += 1

    def invalidate(self, env_mask: np.ndarray) -> None:
        """Restart the validity counter for envs that just reset (``(N,)`` bool)."""
        self._steps_since_reset[env_mask] = 0

    def valid(self) -> np.ndarray:
        """Per-env mask of complete, reset-free history windows (``(N,)`` bool)."""
        if len(self._steps) < self.maxlen:
            return np.zeros_like(self._steps_since_reset, dtype=bool)
        return self._steps_since_reset >= self.maxlen

    def fill(self, step: dict[str, np.ndarray]) -> None:
        """Warm-start: fill every slot with ``step`` and mark all windows valid."""
        self._steps.clear()
        for _ in range(self.maxlen):
            self._steps.append({k: np.array(v, copy=True) for k, v in step.items()})
        self._steps_since_reset[:] = self.maxlen

    def stack(self, key: str) -> np.ndarray:
        """Stack a tracked quantity over the window into ``(N, maxlen, ·)``."""
        return np.stack([s[key] for s in self._steps], axis=1)
