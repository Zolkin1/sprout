# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ballast's ``extero`` must reach the model in the rollout buffer's layout.

Pre-training stores exteroception as one FLAT row per window, ``(n_sensors * sensor_dim,)``;
the AWR rollout buffer stores it per sensor, ``(n_sensors, sensor_dim)``. They are the same
numbers in a different shape, and ``PretrainWindowAdapter`` used to pass the flat row straight
through while its own docstring claimed the two "agree on extero".

``ConcatWindowDataset`` caught it -- but only as a shape mismatch, ``(7800,)`` against
``(2, 3900)``. That is the cheap half. **The expensive half is that a WRONG fold has the right
shape**: reshaping channel-major instead of sensor-major, or transposing, yields exactly
``(2, 3900)`` and passes every check in the stack while attributing half of one camera's pixels
to the other. The FM's per-camera CNN would then train on scrambled input with nothing
anywhere reporting it -- the same class as the transposed-camera trap that once made a tracker
read as incapable.

So this file asserts the CONTENT of the fold, not just its shape: row ``s`` must be the ``s``-th
contiguous slice of the flat row, which is how ``FutureStateDataset`` builds it (``hs`` is
``[Nw, n_sensors * sensor_dim]`` and every per-sensor pass indexes it that way).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
rd = pytest.importorskip("robot_rl.flow_matching.rollout_dataset")

PretrainWindowAdapter = rd.PretrainWindowAdapter


class _FakeBase:
    """A stand-in for :class:`FutureStateDataset` with a known flat extero row."""

    def __init__(self, n_sensors: int, sensor_dim: int, n: int = 3) -> None:
        self.extero_sensor_count = n_sensors
        self.extero_sensor_dim = sensor_dim
        self._n = n
        # Values encode their own position, so a mis-fold is visible rather than plausible.
        self._rows = [torch.arange(n_sensors * sensor_dim, dtype=torch.float32) + 1000 * i
                      for i in range(n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        return {"state": torch.zeros(2), "history": torch.zeros(2),
                "extero": self._rows[i], "cmd_tokens": torch.zeros(3, 4)}


def test_the_fold_is_sensor_major_and_matches_the_flat_row() -> None:
    """The property a shape check cannot see."""
    base = _FakeBase(n_sensors=2, sensor_dim=3900)
    ad = PretrainWindowAdapter(base, path="<fake>")
    flat = base[0]["extero"]
    folded = ad[0]["extero"]
    assert tuple(folded.shape) == (2, 3900)
    for s in range(2):
        assert torch.equal(folded[s], flat[s * 3900:(s + 1) * 3900]), (
            f"sensor {s} is not the {s}-th contiguous slice: the fold is not sensor-major, "
            "which silently re-attributes pixels between cameras at the right shape")


def test_the_advertised_shape_matches_what_getitem_returns() -> None:
    """``ConcatWindowDataset`` compares ``extero_shape``; it must describe the real item."""
    ad = PretrainWindowAdapter(_FakeBase(2, 3900), path="<fake>")
    assert ad.extero_shape == tuple(ad[0]["extero"].shape)
    assert ad.extero_shape == (2, 3900)


def test_a_single_sensor_run_is_left_alone() -> None:
    """A scan (or a one-camera ablation) is already the buffer's shape."""
    ad = PretrainWindowAdapter(_FakeBase(1, 176), path="<fake>")
    assert ad[0]["extero"].ndim == 1
    assert ad.extero_shape == (176,)


def test_a_row_that_does_not_divide_is_REFUSED() -> None:
    """Folding on a mismatch would re-attribute pixels; it must raise instead."""
    base = _FakeBase(2, 3900)
    base.extero_sensor_dim = 3901          # declared width no longer divides the row
    with pytest.raises(ValueError, match="re-attribute"):
        PretrainWindowAdapter(base, path="<fake>")
