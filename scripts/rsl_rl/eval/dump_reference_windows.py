#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Dump raw reference windows from a recording, as the reference set ``terrain_class.py`` fits.

``terrain_class.py --windows`` wants an ``.npz`` of ``tgt`` / ``clip_family`` / ``clip_id`` /
``family_names``, and nothing in the repo produced one -- the first reference was built by
hand, which is why the measurement it backs could not be reproduced. This is that producer.

**The windows are the CLIP's, not the generator's.** ``tgt`` is the ego-anchored clip future
the generator is trained to predict, packed by the SAME ``FutureStateDataset`` the training
run uses, so a reference window and a runtime plan are literally the same object and no
feature engineering stands between them. Passing a checkpoint is therefore unnecessary and
would be misleading: the reference describes what each KIND of motion looks like, which is a
property of the clip library, not of any model.

Three things here are load-bearing:

* **No window filter is applied.** ``skip_families`` / ``skip_distractors`` exist to keep the
  generator from being TRAINED on windows it cannot serve, but the classifier needs an
  exemplar of every kind -- ``family_jump_off`` and ``family_jump_off_front`` are two of the
  three ``jump`` families in ``FAMILY_TO_COARSE``, and dropping the default one would thin the
  very class the stairs->jump confusion is measured against. The dataset is built directly for
  that reason, bypassing ``build_train_val``.
* **The normalizers are IDENTITY**, so ``ds.state`` comes out raw. ``terrain_class`` documents
  that it wants denormalized nodes, and fitting a normalizer here would make the reference a
  function of whichever subset was sampled.
* **Rows are spread over CLIPS, not taken as a prefix.** A recording walks each clip end to
  end, so the first N rows are the first few clips -- a reference built from them would
  describe a handful of tiles. The classifier's own confusion was measured unstable
  (0.010 to 0.116) below ~12k windows, so both the count and the spread matter.

    python scripts/rsl_rl/eval/dump_reference_windows.py \\
        --data <recording>.h5 --out ref_windows.npz --max_windows 20000
    python scripts/rsl_rl/eval/terrain_class.py \\
        --windows ref_windows.npz --holdout 0.3 --out terrain_ref.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

import _paths  # noqa: F401  (sys.path bootstrap; see eval/_paths.py)

from recorded_data_io import load_recorded_windows  # noqa: E402
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import FutureStateDataset  # noqa: E402
from robot_rl.flow_matching.normalizer import Normalizer  # noqa: E402


def spread_over_clips(clip_id: np.ndarray, budget: int, seed: int = 0) -> np.ndarray:
    """Pick ``budget`` row indices spread evenly over clips and along each clip's timeline.

    Every clip gets ``budget // n_clips`` rows (at least one), taken at evenly spaced points
    within that clip rather than sampled, so a clip contributes its whole arc instead of a
    random cluster. The remainder is handed to a random subset of clips so the total is exact.

    Args:
        clip_id: Per-row clip id.
        budget: How many rows to return (capped at the number available).
        seed: Seed for the remainder draw and the final shuffle.

    Returns:
        Sorted row indices, at most ``budget`` of them.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(clip_id, kind="stable")
    bounds = np.flatnonzero(np.diff(clip_id[order])) + 1
    groups = np.split(order, bounds)
    if budget >= len(clip_id):
        return np.arange(len(clip_id))
    per = max(1, budget // max(1, len(groups)))
    extra = set(rng.choice(len(groups), size=min(len(groups), max(0, budget - per * len(groups))),
                           replace=False).tolist()) if budget > per * len(groups) else set()
    picked: list[np.ndarray] = []
    for gi, g in enumerate(groups):
        take = min(len(g), per + (1 if gi in extra else 0))
        picked.append(g[np.linspace(0, len(g) - 1, take).astype(int)])
    out = np.unique(np.concatenate(picked))
    return out[:budget] if len(out) > budget else out


def identity_normalizer(dim: int) -> Normalizer:
    """A pass-through :class:`Normalizer` of width ``dim`` (mean 0, std 1, mask all False)."""
    return Normalizer(torch.zeros(dim), torch.ones(dim), torch.zeros(dim, dtype=torch.bool))


def main() -> int:
    """Read the recording, pack the windows, write the reference ``.npz``."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Windowed recording .h5.")
    p.add_argument("--out", required=True, help="Reference .npz to write.")
    p.add_argument("--max_windows", type=int, default=20000,
                   help="Row budget. The classifier's own confusion was unstable below ~12k.")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import h5py
    with h5py.File(a.data, "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        clip_all = np.asarray(f["clip_id"][:n]).reshape(-1)
    rows = spread_over_clips(clip_all, a.max_windows, a.seed)
    print(f"[dump-ref] {a.data}: {n} windows over {len(np.unique(clip_all))} clips "
          f"-> {len(rows)} rows over {len(np.unique(clip_all[rows]))} clips")

    data, meta = load_recorded_windows(a.data, rows=rows)
    names = list(getattr(meta, "family_names", []) or [])
    if not names:
        raise SystemExit(
            f"{a.data} carries no family_names, so every window would label as 'unknown' and "
            "the reference would have exactly one class. Use a recording made against a "
            "library with a terrain manifest."
        )

    cfg = FlowMatchingConfig()
    # Width is read off the packed target, so the identity normalizers are built after a
    # one-row probe rather than from a config field that could disagree with the recording.
    probe = FutureStateDataset(
        {k: v[:1] for k, v in data.items() if isinstance(v, np.ndarray)}, meta, cfg,
        np.arange(1), state_normalizer=identity_normalizer(1),
        cmd_normalizer=identity_normalizer(1), extero_normalizer=identity_normalizer(1),
    )
    d = int(probe.state.shape[-1])
    ds = FutureStateDataset(
        data, meta, cfg, np.arange(len(rows)),
        state_normalizer=identity_normalizer(d),
        cmd_normalizer=identity_normalizer(int(probe.cmd_tokens.shape[-1]))
        if probe.cmd_tokens is not None else None,
        extero_normalizer=identity_normalizer(int(probe.extero.shape[-1])),
    )
    tgt = ds.state.numpy().astype(np.float32)                       # (N, F, D), raw
    fam = np.asarray(data["clip_family"]).reshape(-1).astype(np.int64)
    cid = np.asarray(data["clip_id"]).reshape(-1).astype(np.int64)
    if not (len(tgt) == len(fam) == len(cid)):
        raise SystemExit(f"row misalignment: tgt {len(tgt)}, family {len(fam)}, clip {len(cid)}")

    counts = {names[c]: int((fam == c).sum()) for c in sorted(set(fam.tolist()))}
    print(f"[dump-ref] windows {tgt.shape} raw; per family: {counts}")
    np.savez_compressed(a.out, tgt=tgt, clip_family=fam, clip_id=cid,
                        family_names=json.dumps(names))
    print(f"[dump-ref] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
