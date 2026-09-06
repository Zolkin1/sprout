# Copyright 2026 Zachary Olkin. All rights reserved.

"""Concatenate several ``record_rollouts.py`` outputs into one stem.

``fit_value.py`` takes ONE rollout stem, and an initial-condition sweep produces one file per
cell -- the box2 grid pins the commanded speed per run, so six runs cover the speed axis.  This
joins them.

**``episode_uid`` is renumbered, and that is the whole reason this is a script and not a
``cat``.**  The uid is assigned per RUN, starting from 0, so two files both contain episode 0.
``fit_value`` splits BY EPISODE precisely so that near-duplicate windows from one episode
cannot straddle the split -- and a colliding uid defeats exactly that: two unrelated episodes
merge into one group, their windows are dealt to both sides, and the held-out metrics measure
memorisation while looking clean.  This repo has already paid for that once, on the FM's own
episode split, where 86.5% of held-out clips also appeared in training.  So each file's uids
are offset past the running maximum, the same rule ``train.py`` applies to ``episode_id``.

**Contract fields must AGREE, not merely coexist.**  Window shapes (history length, state and
command dims, the exteroception and proprioception layouts) are the tensor widths a critic is
built for, and the camera/scan specs say what those numbers MEAN.  A mismatch is refused by
name rather than broadcast or truncated: concatenating a 2-camera row onto a 1-camera one would
produce a table whose width is right and whose content is two different sensors.

What is deliberately NOT checked: the commanded speed, the terrain library, the tracker and the
generator.  Merging across those is the POINT -- an IC sweep varies them on purpose -- and the
per-run values are recorded under ``merged_sources`` so a reader can still tell the cells apart
(the ``terrain_id`` column already carries the tile, and on the box2 terrain the tile IS the
approach distance).

Usage::

    python scripts/rsl_rl/merge_rollouts.py \\
        --out logs/box2/rollouts/train_all/rollout_box2_train \\
        logs/box2/rollouts/train_v0.50/rollout_<trk> \\
        logs/box2/rollouts/train_v0.83/rollout_<trk> ...
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _pkg in ("motion_gen", "robot_rl"):
    _p = str(_REPO_ROOT / "source" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import rollout_outcomes as ro  # noqa: E402

#: Attributes that must be IDENTICAL across the merged files: they describe the tensor layout a
#: critic is built for, and what the numbers in it mean.  Anything absent on every side is
#: skipped; present-on-some is a mismatch, because "missing" and "different" are both reasons
#: not to concatenate.
_CONTRACT_ATTRS: tuple[str, ...] = (
    "num_history", "state_dim", "future_len", "cmd_dim", "extero_shape", "proprio_hw",
    "scan_dim", "track_hw", "depth_camera_spec", "height_scan_spec", "schema_version",
)


def _attr_key(v) -> str:
    """A comparable, printable form for an attribute of any shape.

    Args:
        v: The attribute value.

    Returns:
        Its JSON form, with numpy types coerced.
    """
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (np.integer, np.floating)):
        v = v.item()
    return json.dumps(v, sort_keys=True, default=str)


def assert_contracts_agree(attrs: list[dict], stems: list[str]) -> None:
    """Refuse a merge whose files describe different tensors.

    Args:
        attrs: One attribute dict per file, in ``stems`` order.
        stems: The stems, for the message.

    Raises:
        SystemExit: Naming the attribute and the two values on the first disagreement.
    """
    for name in _CONTRACT_ATTRS:
        seen: dict[str, str] = {}
        for stem, a in zip(stems, attrs):
            if name not in a:
                continue
            seen.setdefault(_attr_key(a[name]), stem)
        if len(seen) > 1:
            lines = "\n".join(f"    {s}: {k}" for k, s in seen.items())
            raise SystemExit(
                f"{name!r} differs across the files being merged, so they do not describe the "
                f"same tensor:\n{lines}\n"
                "Concatenating these would give a table of the right WIDTH holding two "
                "different things. Merge only runs that share an observation contract.")
        present = [stem for stem, a in zip(stems, attrs) if name in a]
        if present and len(present) != len(stems):
            missing = [s for s in stems if s not in present]
            raise SystemExit(
                f"{name!r} is present on {present} and absent on {missing}. 'Missing' and "
                "'different' are both reasons not to merge; re-record the odd one out.")


def merge(stems: list[str], out_stem: str) -> None:
    """Concatenate the window and episode tables of every stem into ``out_stem``.

    Args:
        stems: Input rollout stems (without ``.h5``).
        out_stem: Output stem; ``<out_stem>.h5`` and ``<out_stem>.episodes.h5`` are written.

    Raises:
        SystemExit: On a contract mismatch, or if a window's episode has no outcome row.
    """
    win, wattrs, eps, eattrs = [], [], [], []
    for s in stems:
        for suffix in (".h5", ".episodes.h5"):
            if not os.path.isfile(s + suffix):
                raise SystemExit(f"{s + suffix} not found -- is {s!r} a rollout stem?")
        W, wa = ro.read_rollout_windows(s + ".h5")
        E, ea = ro.read_rollout_episodes(s + ".episodes.h5")
        win.append(W); wattrs.append(wa); eps.append(E); eattrs.append(ea)
        print(f"[merge] {s}: {len(E['episode_uid'])} episodes, "
              f"{len(W['episode_uid'])} windows")
    assert_contracts_agree(wattrs, stems)

    # Renumber per file, past the running max. See the module docstring for why a collision
    # here is not cosmetic.
    offset = 0
    for W, E in zip(win, eps):
        n = int(max(E["episode_uid"].max(initial=-1), W["episode_uid"].max(initial=-1))) + 1
        E["episode_uid"] = E["episode_uid"] + offset
        W["episode_uid"] = W["episode_uid"] + offset
        offset += n

    out_w = {k: np.concatenate([W[k] for W in win], axis=0) for k in win[0]}
    out_e = {k: np.concatenate([E[k] for E in eps], axis=0) for k in eps[0]}
    n_eps = len(out_e["episode_uid"])
    if len(np.unique(out_e["episode_uid"])) != n_eps:
        raise SystemExit("episode_uid is not unique after renumbering -- the offset is wrong; "
                         "refusing rather than writing a table fit_value would mis-split.")
    orphans = np.setdiff1d(np.unique(out_w["episode_uid"]), out_e["episode_uid"])
    if orphans.size:
        raise SystemExit(f"{orphans.size} window episode_uid(s) have no outcome row "
                         f"(e.g. {orphans[:5].tolist()}); the join would drop them silently.")

    os.makedirs(os.path.dirname(os.path.abspath(out_stem)) or ".", exist_ok=True)
    prov = {"merged_sources": [
        {"stem": s,
         "n_episodes": int(len(E["episode_uid"])),
         "n_windows": int(len(W["episode_uid"])),
         # Everything a reader needs to tell the cells apart after the join.
         "attrs": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in ea.items()
                   if k not in ("depth_camera_spec", "height_scan_spec")}}
        for s, W, E, ea in zip(stems, win, eps, eattrs)]}

    for path, cols, base in ((out_stem + ".h5", out_w, wattrs[0]),
                             (out_stem + ".episodes.h5", out_e, eattrs[0])):
        with h5py.File(path, "w") as f:
            for k, v in cols.items():
                f.create_dataset(k, data=v, compression="lzf" if v.nbytes > (1 << 20) else None)
            for k, v in base.items():
                f.attrs[k] = v if isinstance(v, (str, bytes)) else json.dumps(
                    v.tolist() if isinstance(v, np.ndarray) else v, default=str)
            f.attrs["merge_provenance"] = json.dumps(prov, default=str)
        print(f"[merge] wrote {path}: {len(next(iter(cols.values())))} rows")
    print(f"[merge] {n_eps} episodes, {len(out_w['episode_uid'])} windows, "
          f"uids 0..{int(out_e['episode_uid'].max())}")


def main() -> int:
    """Parse arguments and merge."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stems", nargs="+", help="rollout stems (without .h5)")
    p.add_argument("--out", required=True, help="output stem")
    a = p.parse_args()
    merge(list(a.stems), a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
