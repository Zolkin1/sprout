# Copyright 2026 Zachary Olkin. All rights reserved.

"""Numpy rolling history buffer for the deployment generator inference.

Numpy port of ``robot_rl.flow_matching.online.BatchedHistory`` (reset-aware, fixed length).
A window becomes valid once an env has ``maxlen`` uninterrupted steps since its last
reset. There is deliberately no way to pre-fill it: every row is a reference-window node
the runtime SERVED, so a fabricated one is off-distribution -- a repeated static snapshot
is a history with no motion between its steps, which training never produces. Until the
window holds ``maxlen`` real rows, :meth:`null_mask` says so and the generator's learned
null-history embedding covers it.
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

    def stack(self, key: str) -> np.ndarray:
        """Stack a tracked quantity over the window into ``(N, maxlen, ·)``."""
        return np.stack([s[key] for s in self._steps], axis=1)

    def null_mask(self) -> np.ndarray:
        """Per-env ``(N,)`` bool: True where the history window is NOT yet complete.

        Hand this to the FM as ``history_null``: training nulled ``null_cond_prob`` of its
        samples, so the learned ``null_history`` embedding covers exactly this case and the
        generator can be sampled from step 0 -- no clip warmup, no fabricated static history.
        """
        if len(self._steps) < self.maxlen:
            return np.ones_like(self._steps_since_reset, dtype=bool)
        return self._steps_since_reset < self.maxlen

    def stack_padded(self, key: str) -> np.ndarray:
        """Like :meth:`stack`, but always ``maxlen`` long -- front-padded by repeating the oldest.

        Only the global deque underfill (the first steps of a run) is padded. Per-env staleness
        is not repaired: those envs are the ones :meth:`null_mask` flags, whose history content
        the model replaces with the learned null embedding. The newest step stays at index -1,
        so the caller's anchor is always the live state.
        """
        if not self._steps:
            raise RuntimeError("stack_padded on an empty history: append at least one step first.")
        steps = list(self._steps)
        pad = [steps[0]] * (self.maxlen - len(steps))
        return np.stack([s[key] for s in (pad + steps)], axis=1)
