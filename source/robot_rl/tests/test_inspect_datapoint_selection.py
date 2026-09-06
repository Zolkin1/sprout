# Copyright 2026 Zachary Olkin. All rights reserved.

"""The two pieces of ``inspect_datapoints.py`` that are wrong SILENTLY.

The tool's numeric controls (``force16 == tok_floor``, the window round trip, FK against the
recording's own ankle columns) run every time it is invoked and fail loudly, so they need no
test. These two do not:

* **Row selection.** A datapoint that was actually dropped from training, or one labelled
  held-out when the run trained on it, produces a table that is entirely plausible and
  answers a different question. Both are decided here, from the run's OWN recorded
  ``val_episode_ids`` rather than from a re-derived split -- which is what lets the tool run
  against whichever subset of the training recordings is on the local box.
* **The per-item reduction.** The tool prints per-datapoint numbers next to pooled ones and
  invites subtracting them. They come from two different functions, so this pins that they
  are the same quantity: the tool asserts it at runtime on real data, and this asserts it on
  data whose answer is known.

Simulator-free.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motion_gen.features import _REPO_ROOT  # noqa: E402


def _tool():
    """Import the script as a module (the idiom its sibling diagnostics already use)."""
    path = _REPO_ROOT / "scripts" / "token_planner" / "inspect_datapoints.py"
    spec = importlib.util.spec_from_file_location("inspect_datapoints", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inspect_datapoints"] = mod
    spec.loader.exec_module(mod)
    return mod


FAMILIES = ["unknown", "family_stairs_up", "family_jump_off", "family_distractor"]


def _index(n: int = 400) -> dict[str, np.ndarray]:
    """A synthetic recording index: 4 families, 40 episodes, some distractors."""
    rng = np.random.default_rng(0)
    return {
        "episode_id": np.repeat(np.arange(40, dtype=np.int64), n // 40),
        "clip_family": rng.integers(0, len(FAMILIES), size=n).astype(np.int64),
        "is_distractor": (np.arange(n) % 17 == 0).astype(np.int64),
        "clip_id": np.arange(n, dtype=np.int64),
        "clip_step": (np.arange(n) * 12).astype(np.int64),
        "env_id": (np.arange(n) % 8).astype(np.int64),
    }


def _select(mod, **kw):
    """Call ``select_rows`` with this file's defaults."""
    args = dict(index=_index(), family_names=FAMILIES, val_eps=set(range(0, 40, 4)),
                skip_families=["family_jump_off"], skip_distractors=True, num_points=24,
                split="both", families=[], seed=0)
    args.update(kw)
    return mod.select_rows(**args)


def test_selection_is_deterministic_in_the_seed():
    """The same seed picks the same rows; a different one does not."""
    mod = _tool()
    a = [r.row for r in _select(mod)]
    b = [r.row for r in _select(mod)]
    c = [r.row for r in _select(mod, seed=1)]
    assert a == b
    assert a != c, "a different seed selected an identical set; the shuffle is not keyed"


def test_it_never_selects_a_window_the_run_dropped():
    """No skipped family and no distractor, whichever split is asked for.

    This is the failure mode that reads as success: a ``family_jump_off`` row scores like any
    other and its table looks fine, while the planner was never shown one.
    """
    mod = _tool()
    idx = _index()
    dropped = FAMILIES.index("family_jump_off")
    for split in ("val", "train", "both"):
        for r in _select(mod, split=split, num_points=200):
            assert idx["clip_family"][r.row] != dropped
            assert idx["is_distractor"][r.row] == 0
            assert r.kept


def test_held_out_comes_from_the_recorded_episode_ids():
    """``held_out`` is a lookup in the run's own list, not a re-derived split."""
    mod = _tool()
    val = {3, 7, 11}
    for r in _select(mod, val_eps=val, num_points=200, split="both"):
        assert r.held_out == (r.episode in val)
    assert all(r.held_out for r in _select(mod, val_eps=val, split="val", num_points=50))
    assert not any(r.held_out for r in _select(mod, val_eps=val, split="train",
                                               num_points=50))


def test_the_index_is_a_stable_prefix_in_num_points():
    """Raising ``--num_points`` APPENDS windows; it does not renumber the earlier ones.

    The datapoint index is how a finding gets referred to ("#07 looks wrong"), so it has to
    survive re-running at a different count. Sorting the result -- by family, say, which reads
    better in the dropdown -- breaks this: the first family's block grows and shifts
    everything after it.
    """
    mod = _tool()
    for small, large in ((8, 40), (13, 97)):
        a = [r.row for r in _select(mod, num_points=small)]
        b = [r.row for r in _select(mod, num_points=large)]
        assert a == b[:small], f"{small} is not a prefix of {large}"


def test_the_selection_is_stratified_over_family_and_split():
    """Every eligible (family, split) bucket appears before any bucket is drawn twice.

    Without this the tool answers with whatever the recording holds most of -- and
    ``unknown`` is 31% of the shipped one, so a uniform draw would mostly show flat walking.
    """
    mod = _tool()
    kept = [f for f in FAMILIES if f not in ("family_jump_off", "family_distractor")]
    refs = _select(mod, num_points=2 * 2 * len(kept))
    buckets = {(r.family, r.held_out) for r in refs}
    assert len(buckets) >= 2 * len(kept) - 1, f"only reached {buckets}"


def test_it_refuses_rather_than_returning_an_empty_selection():
    """An impossible request exits with the counts, instead of serving nothing."""
    mod = _tool()
    with pytest.raises(SystemExit) as e:
        _select(mod, families=["family_jump_off"])
    assert "no eligible windows" in str(e.value)
    with pytest.raises(SystemExit):
        _select(mod, families=["family_no_such_thing"])


def test_the_per_item_reduction_is_the_accumulators():
    """Pooling ``group_rmse_per_item`` reproduces ``ReconAccumulator.group_rmse`` exactly.

    The tool prints both tables and the per-datapoint one is meant to be read against the
    pooled one; if the two reductions differ, every comparison between them is off by an
    amount that depends on the spread.
    """
    mod = _tool()
    from robot_rl.token_planner.metrics import ReconAccumulator, channel_group_labels

    nj, nee, blocks = 29, 4, ["base_pose", "base_vel", "joint_pos"]
    groups, labels = channel_group_labels(nj, nee, blocks)
    d = labels.numel()
    g = torch.Generator().manual_seed(0)
    target = torch.randn(11, 64, d, generator=g)
    pred = target + 0.1 * torch.randn(11, 64, d, generator=g)
    std = torch.rand(d, generator=g) + 0.5
    mean = torch.randn(d, generator=g)

    acc = ReconAccumulator(group_names=groups, labels=labels, std=std.clone(),
                           mean=mean.clone(), chunk_len=64, vocab_size=16,
                           num_families=1, family_names=["all"])
    acc.update(target, pred, torch.zeros(11, 4, dtype=torch.long), torch.zeros(11,
                                                                               dtype=torch.long))
    pooled = acc.group_rmse()
    per_item = mod.group_rmse_per_item(target, pred, labels, std, len(groups))
    back = (per_item ** 2).mean(dim=0).sqrt()
    for i, name in enumerate(groups):
        assert abs(float(back[i]) - pooled[name]) < 1e-9, name


def test_the_quaternion_error_is_sign_insensitive():
    """``window_to_world`` may return the negated quaternion; that is the same rotation.

    Comparing raw components would report a spurious 2.0 and fail the round-trip control on a
    tool that is working perfectly.
    """
    mod = _tool()
    q = torch.nn.functional.normalize(torch.randn(5, 7, 4), dim=-1).numpy()
    assert mod._quat_err(q, q) == 0.0
    assert mod._quat_err(q, -q) == 0.0
    assert mod._quat_err(q, np.roll(q, 1, axis=-1)) > 1e-3
