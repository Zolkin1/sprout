#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""How many FM training windows contain a terrain transition, broken down per motion family.

A window "contains a transition" when the clip's anchor mask changes pinned-ness inside the
horizon the FM predicts -- the reference goes from ego-anchored flat walking to
terrain-registered, or back. That is ground truth from the clip library rather than a height
heuristic, and it is the label the ``--transition_fraction`` sampler already uses, so this
script **imports that exact function** (``flow_matching.dataset.transition_window_mask``)
instead of restating the arithmetic. A restatement diverges in at least six places -- the mask
BITS tested, the 255 sentinel, ``clip_len`` padding, the strictly-inside convention, the
degenerate-span guard, and ``cfg.future_len`` vs the recorded ``ref_future_len``.

Reads only the five index columns from the recording, never the window payload, so it runs in
seconds against a 9 GB file instead of pulling it into RAM.

Reports two populations, because they differ and both are quoted:

* **ALL** windows in the file;
* **TRAINED SUBSET** -- what a specific checkpoint actually saw, after its own
  ``skip_distractors`` / ``skip_families`` filters (passed via ``--checkpoint``).

Example::

    python scripts/rsl_rl/transition_census.py \
        --data logs/rsl_rl/g1/<run>/recorded_data/recorded_20260809_152805.h5 \
        --checkpoint logs/flow_matching/<fm_run>/best.pt --out plots/transition_census
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields

import matplotlib

matplotlib.use("Agg")  # before pyplot: this runs headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (os.path.join(_REPO, "source", "robot_rl"), os.path.join(_REPO, "source", "motion_gen"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recorded_data_io import RecordingMeta  # noqa: E402

#: The only columns needed. `clip_id`/`clip_step` drive the label, `clip_family` the
#: breakdown, `is_distractor` the trained-subset filter, `episode_id` nothing yet but it is
#: free and makes the npz self-contained.
_COLUMNS = ("clip_id", "clip_step", "clip_family", "is_distractor", "episode_id")


def read_meta(path: str) -> RecordingMeta:
    """RecordingMeta from an ``.h5``'s root attrs, without touching any window array."""
    import h5py

    with h5py.File(path, "r") as f:
        raw = {k: json.loads(v) for k, v in f.attrs.items()
               if k not in ("num_windows", "num_windows_written")}
    names = {fld.name for fld in fields(RecordingMeta)}
    return RecordingMeta(**{k: v for k, v in raw.items() if k in names})


def read_index_columns(path: str) -> tuple[dict[str, np.ndarray], int]:
    """The index columns and the written window count.

    Deliberately NOT ``recorded_data_io.load_recorded_windows``: that loads every dataset,
    including the ~9 GB depth block, for five int columns worth of information.
    """
    import h5py

    with h5py.File(path, "r") as f:
        written = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        if not written:
            raise SystemExit(f"{path} reports no written windows.")
        data = {k: f[k][:written] for k in _COLUMNS if k in f}
    missing = {"clip_id", "clip_step"} - set(data)
    if missing:
        raise SystemExit(
            f"{path} is missing {sorted(missing)}, so windows cannot be joined to the clip "
            "library and no transition label exists. Re-record with the current "
            "record_data.py (it sets tag_clip_index=True)."
        )
    return data, written


def load_pinned(meta: RecordingMeta, cfg) -> tuple[np.ndarray, np.ndarray]:
    """``(pinned [N, T] bool, clip_len [N])`` for the library this recording indexes into.

    Uses the sentinel-resolved per-step mask and the pinned-ness predicate, i.e. the same two
    helpers the label uses -- reading ``anchor_mask_t`` raw would count the 255 "unset" rows and
    the height/lateral bits as boundaries that the trainer never saw.
    """
    from motion_gen.clips.clip_io import load_clip_library
    from motion_gen.world_lift import anchor_mask_is_pinned

    path = cfg.clip_library_path or meta.clip_library_path
    lib = load_clip_library(path, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    return (anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy(),
            lib.clip_len_or_full().numpy())


def first_boundary_offset(data: dict, pinned: np.ndarray, clip_len: np.ndarray,
                         future_len: int, label: np.ndarray) -> np.ndarray:
    """Node offset of the first boundary in each window; ``-1`` where there is none.

    Answers "where in the horizon does the transition land", which the boolean label cannot.
    Mirrors the label's span exactly -- strictly inside ``[clip_step + 1, end)`` with
    ``end = min(clip_step + future_len, clip_len)`` -- and then **asserts** that its own
    yes/no agrees with the imported label on every window. That assert is the point: it keeps
    this auxiliary arithmetic from drifting away from the function that defines the number.

    Raises:
        AssertionError: If the offsets disagree with ``label`` anywhere.
    """
    cid = data["clip_id"].astype(np.int64)
    cstep = data["clip_step"].astype(np.int64)
    end = np.minimum(cstep + int(future_len), clip_len[cid])
    out = np.full(cid.shape, -1, dtype=np.int64)
    change = np.zeros_like(pinned, dtype=bool)
    change[:, 1:] = pinned[:, 1:] != pinned[:, :-1]
    for i in range(cid.shape[0]):
        lo, hi = min(cstep[i] + 1, end[i]), end[i]
        if hi - cstep[i] < 2 or hi <= lo:
            continue
        idx = np.nonzero(change[cid[i], lo:hi])[0]
        if idx.size:
            out[i] = int(lo + idx[0] - cstep[i])
    got = out >= 0
    bad = int((got != label).sum())
    assert bad == 0, (
        f"first_boundary_offset disagrees with transition_window_mask on {bad} windows; the "
        "auxiliary span arithmetic has drifted from the label.")
    return out


def trained_subset_mask(data: dict, meta: RecordingMeta, cfg, num_windows: int) -> np.ndarray:
    """Keep-mask for the windows a checkpoint's own filters left in, as the dataset applies them."""
    from robot_rl.flow_matching.dataset import _distractor_keep_mask, _family_keep_mask

    keep = _distractor_keep_mask(data, cfg, num_windows)
    fam = _family_keep_mask(data, meta, cfg, num_windows)
    if keep is None and fam is None:
        return np.ones(num_windows, dtype=bool)
    if keep is None:
        return fam
    return keep if fam is None else (keep & fam)


def build_config(checkpoint: str | None, future_len: int | None, clip_library_path: str | None):
    """The config the label is computed under, and a note on where ``future_len`` came from.

    ``future_len`` decides the horizon the boundary must fall inside, and the label uses the
    TRAINING config's value -- not the recording's ``ref_future_len``, which can differ (this
    repo has recorded at 51 while training at 62). Taking it from the checkpoint is what makes
    the number comparable to a specific run.
    """
    from robot_rl.flow_matching.config import FlowMatchingConfig

    if checkpoint:
        import torch

        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = FlowMatchingConfig.from_dict(ck["config"])
        source = f"checkpoint {os.path.basename(checkpoint)}"
    else:
        cfg = FlowMatchingConfig()
        source = "FlowMatchingConfig default"
    if future_len is not None:
        cfg.future_len = int(future_len)
        source = f"--future_len {future_len} (override)"
    if clip_library_path:
        cfg.clip_library_path = clip_library_path
    # Read-only measurement: leaving this set makes build_train_val's sampler path run and
    # print during what should be a pure count.
    cfg.transition_fraction = None
    return cfg, source


def census_one(path: str, cfg, cfg_source: str) -> dict:
    """Label every window in one recording and aggregate per family."""
    from robot_rl.flow_matching.dataset import transition_window_mask

    meta = read_meta(path)
    data, written = read_index_columns(path)
    label = transition_window_mask(data, meta, cfg, written)
    pinned, clip_len = load_pinned(meta, cfg)
    offset = first_boundary_offset(data, pinned, clip_len, cfg.future_len, label)
    trained = trained_subset_mask(data, meta, cfg, written)

    names = list(getattr(meta, "family_names", []) or [])
    fam_code = data.get("clip_family")
    rows = []
    groups: list[tuple[str, np.ndarray]] = []
    if fam_code is None:
        groups.append(("(no clip_family column)", np.ones(written, bool)))
    else:
        for code in np.unique(fam_code):
            name = names[int(code)] if int(code) < len(names) else f"code {int(code)}"
            groups.append((name, fam_code == code))
    for name, sel in groups:
        n = int(sel.sum())
        rows.append({
            "family": name, "windows": n,
            "transition": int(label[sel].sum()),
            "share": float(label[sel].mean()) if n else 0.0,
            "trained": int(trained[sel].sum()),
            "trained_transition": int((label & trained)[sel].sum()),
            "trained_share": (float(label[sel & trained].mean())
                              if int((sel & trained).sum()) else 0.0),
        })
    rows.sort(key=lambda r: -r["windows"])
    total = {
        "family": "ALL", "windows": written, "transition": int(label.sum()),
        "share": float(label.mean()), "trained": int(trained.sum()),
        "trained_transition": int((label & trained).sum()),
        "trained_share": float(label[trained].mean()) if int(trained.sum()) else 0.0,
    }
    return {
        "path": path, "rows": rows, "total": total, "future_len": int(cfg.future_len),
        "cfg_source": cfg_source, "offset": offset, "label": label, "trained": trained,
        "clip_library_path": cfg.clip_library_path or meta.clip_library_path,
        "ref_future_len": int(getattr(meta, "ref_future_len", -1)),
        "skip_distractors": bool(getattr(cfg, "skip_distractors", False)),
        "skip_families": list(getattr(cfg, "skip_families", None) or []),
    }


def report(res: dict) -> str:
    """One recording's census as a text table."""
    out = ["", "=" * 92, f"TRANSITION CENSUS  {os.path.basename(res['path'])}", "=" * 92,
           f"  library     {res['clip_library_path']}",
           f"  future_len  {res['future_len']}  (from {res['cfg_source']}; recording's "
           f"ref_future_len = {res['ref_future_len']})"]
    if res["skip_distractors"] or res["skip_families"]:
        out.append(f"  checkpoint filters: skip_distractors={res['skip_distractors']} "
                   f"skip_families={res['skip_families']}")
    out += ["",
            f"  {'family':<26}{'windows':>10}{'transition':>12}{'share':>9}"
            f"{'trained':>10}{'trans':>8}{'share':>9}",
            f"  {'-' * 26}{'-' * 10}{'-' * 12}{'-' * 9}{'-' * 10}{'-' * 8}{'-' * 9}"]
    for r in res["rows"] + [res["total"]]:
        if r["family"] == "ALL":
            out.append(f"  {'-' * 84}")
        out.append(f"  {r['family']:<26}{r['windows']:>10}{r['transition']:>12}"
                   f"{100 * r['share']:>8.1f}%{r['trained']:>10}{r['trained_transition']:>8}"
                   f"{100 * r['trained_share']:>8.1f}%")
    off = res["offset"][res["offset"] >= 0]
    if off.size:
        out += ["", f"  boundary node within the {res['future_len']}-node horizon: "
                    f"p10 {np.percentile(off, 10):.0f}  p50 {np.percentile(off, 50):.0f}  "
                    f"p90 {np.percentile(off, 90):.0f}   (0 = at the window's first node)"]
    out.append("=" * 92)
    return "\n".join(out)


def plot(results: list[dict], out_dir: str) -> list[str]:
    """Per-family transition share, and where the boundary lands in the horizon."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.9 * n), squeeze=False)
    for i, res in enumerate(results):
        rows = [r for r in res["rows"] if r["windows"] > 0]
        ax = axes[i][0]
        pos = np.arange(len(rows))
        ax.barh(pos - 0.2, [100 * r["share"] for r in rows], height=0.38,
                color="#0077BB", label="all windows")
        ax.barh(pos + 0.2, [100 * r["trained_share"] for r in rows], height=0.38,
                color="#EE7733", label="trained subset")
        ax.axvline(100 * res["total"]["share"], color="#CC3311", ls="--", lw=1.2,
                   label=f"all: {100 * res['total']['share']:.1f}%")
        ax.set_yticks(pos)
        ax.set_yticklabels([f"{r['family']}\n({r['windows']})" for r in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("windows containing a transition [%]")
        ax.set_title(f"{os.path.basename(res['path'])}  (F={res['future_len']})", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="x", alpha=0.3)

        ax2 = axes[i][1]
        off = res["offset"][res["offset"] >= 0]
        if off.size:
            ax2.hist(off, bins=min(res["future_len"], 62), color="#009988")
            ax2.set_xlabel(f"first boundary node in the horizon (0..{res['future_len'] - 1})")
            ax2.set_ylabel("windows")
            ax2.set_title(f"{off.size} transition windows", fontsize=9)
            ax2.grid(alpha=0.3)
        else:
            ax2.set_title("no transition windows")
    fig.tight_layout()
    p = os.path.join(out_dir, "transition_census.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)
    npz = os.path.join(out_dir, "transition_census.npz")
    np.savez_compressed(npz, **{
        f"{os.path.basename(r['path'])}::{k}": v
        for r in results for k, v in (("label", r["label"]), ("offset", r["offset"]),
                                      ("trained", r["trained"]))})
    written.append(npz)
    return written


def main() -> int:
    """Parse args, census each recording, print the tables and write the figures."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", required=True,
                   help="Recording .h5 (repeatable; each is reported separately).")
    p.add_argument("--checkpoint", default=None,
                   help="FM checkpoint whose config supplies future_len and the "
                        "skip_distractors/skip_families filters for the trained subset.")
    p.add_argument("--future_len", type=int, default=None,
                   help="Override the horizon the boundary must fall inside.")
    p.add_argument("--clip_library_path", default=None,
                   help="Override the library the recording names (it must still hash to what "
                        "the recording was made against).")
    p.add_argument("--out", default="plots/transition_census", help="Output directory.")
    p.add_argument("--no_plot", action="store_true", help="Text tables only.")
    args = p.parse_args()

    cfg, source = build_config(args.checkpoint, args.future_len, args.clip_library_path)
    results = []
    for path in args.data:
        res = census_one(path, cfg, source)
        print(report(res))
        results.append(res)
    if not args.no_plot:
        for w in plot(results, args.out):
            print(f"[transition_census] wrote {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
