# Copyright 2026 Zachary Olkin. All rights reserved.

"""The ballast and the buffer must hand ``_collate`` the SAME extero shape -- both flat.

Two different quantities are easy to confuse here, and confusing them costs a fine-tune run:

* a part's ``extero_shape`` is its SOURCE RECORDING's per-window sensor geometry,
  ``(n_sensors, sensor_dim)``. ``RolloutWindowDataset`` reads it straight off
  ``f["extero"].shape[1:]``. It exists for one consumer, ``ConcatWindowDataset``, which uses it
  to refuse parts that describe different rigs.
* an ITEM's ``extero`` is what ``_collate`` stacks, and it is FLAT on both sides, because both
  datasets end in ``online.build_extero_tokens`` -- documented as "flattened to ``(V, T*dim)``
  -- the model reshapes internally".

**The trap, paid for twice.** The adapter first advertised the base's flat width as
``extero_shape``, so the guard compared ``(7800,)`` against the buffer's ``(2, 3900)`` and
refused two datasets that agreed perfectly at the item level. Reshaping the ITEMS to make that
comparison pass then satisfied the guard and broke the collate for real::

    RuntimeError: stack expects each tensor to be equal size,
                  but got [7800] at entry 0 and [2, 3900] at entry 22

-- which is the failure the reshape was supposed to prevent, now caused by it. The guard had to
be told what it was comparing; the items were never wrong.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
rd = pytest.importorskip("robot_rl.flow_matching.rollout_dataset")

PretrainWindowAdapter = rd.PretrainWindowAdapter


class _FakeBase:
    """A stand-in for :class:`FutureStateDataset`: flat extero, declared sensor geometry."""

    def __init__(self, n_sensors: int, sensor_dim: int, n: int = 3) -> None:
        self.extero_sensor_count = n_sensors
        self.extero_sensor_dim = sensor_dim
        self._n = n
        self._rows = [torch.arange(n_sensors * sensor_dim, dtype=torch.float32) + 1000 * i
                      for i in range(n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        return {"state": torch.zeros(2), "history": torch.zeros(2),
                "extero": self._rows[i], "cmd_tokens": torch.zeros(3, 4)}


def test_the_item_stays_flat_so_collate_can_stack_it() -> None:
    """The property the run actually needs."""
    base = _FakeBase(2, 3900)
    ad = PretrainWindowAdapter(base, path="<fake>")
    got = ad[0]["extero"]
    assert got.ndim == 1 and got.shape[0] == 2 * 3900, (
        f"ballast items must stay flat like the buffer's, got {tuple(got.shape)}; reshaping "
        "them is what produced 'stack expects each tensor to be equal size, but got [7800] "
        "at entry 0 and [2, 3900] at entry 22'")
    assert torch.equal(got, base[0]["extero"]), "the row must be passed through untouched"


def test_extero_shape_describes_the_recording_not_the_item() -> None:
    """It is the field ``ConcatWindowDataset`` compares, and the buffer sets it from the FILE."""
    ad = PretrainWindowAdapter(_FakeBase(2, 3900), path="<fake>")
    assert ad.extero_shape == (2, 3900), (
        "extero_shape must be the source recording's (n_sensors, sensor_dim), matching what "
        "RolloutWindowDataset reads off f['extero'].shape[1:] -- not the flat item width")
    assert ad.extero_shape != tuple(ad[0]["extero"].shape), (
        "these two are deliberately different quantities; if they ever coincide this test is "
        "no longer checking anything")


def test_a_ballast_part_joins_a_buffer_part_that_names_the_same_rig() -> None:
    """End to end through the guard: agreeing parts must be accepted."""
    ad = PretrainWindowAdapter(_FakeBase(2, 3900), path="<ballast>")

    class _Buf:                      # the shape RolloutWindowDataset advertises
        path, extero_shape, cmd_dim = "<buffer>", (2, 3900), 4
        # The guard also refuses parts whose OPTIONAL key sets differ; ballast is ungraded, so
        # both sides carry neither here.
        label = weight = None
        def __len__(self): return 5
        def __getitem__(self, i):
            return {"state": torch.zeros(2), "history": torch.zeros(2),
                    "extero": torch.zeros(2 * 3900), "history_null": torch.tensor(False),
                    "cmd_tokens": torch.zeros(3, 4)}

    ds = rd.ConcatWindowDataset([_Buf(), ad])
    shapes = {tuple(ds[i]["extero"].shape) for i in range(len(ds))}
    assert shapes == {(2 * 3900,)}, (
        f"every part must yield the same flat extero for _collate to stack; got {shapes}")


def test_a_scan_run_is_one_sensor() -> None:
    """A privileged scan has a single reading; the geometry still describes it."""
    ad = PretrainWindowAdapter(_FakeBase(1, 176), path="<fake>")
    assert ad.extero_shape == (1, 176)
    assert ad[0]["extero"].ndim == 1
