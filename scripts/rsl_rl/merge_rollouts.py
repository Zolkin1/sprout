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


def subsample_episodes(W: dict, E: dict, keep: int, rng: np.random.Generator) -> None:
    """Cut a stem down to ``keep`` episodes, IN PLACE, keeping each one whole.

    **Sampled by EPISODE, never by window.** ``fit_value`` splits by episode because every
    window of an episode carries the same terminal label, so dropping windows from the middle
    of an episode would thin the label without removing it -- the effective sample size is the
    episode count either way, and a partial episode just costs coverage of its own trajectory.

    Args:
        W: The stem's window columns, mutated.
        E: The stem's episode columns, mutated.
        keep: Episodes to retain; a value at or above the stem's count is a no-op.
        rng: Draw source, so a subsampled merge is reproducible.
    """
    uids = E["episode_uid"]
    if keep >= uids.size:
        return
    chosen = np.sort(rng.choice(uids, size=keep, replace=False))
    emask = np.isin(uids, chosen)
    wmask = np.isin(W["episode_uid"], chosen)
    for d, m in ((E, emask), (W, wmask)):
        for k in list(d):
            d[k] = d[k][m]


def merge(stems: list[str], out_stem: str, max_episodes_per_stem: int = 0,
          seed: int = 0) -> None:
    """Concatenate the window and episode tables of every stem into ``out_stem``.

    **Streamed, one stem at a time**, which is what makes the full grid mergeable at all.
    Holding every input plus the output in RAM peaked at 21.8 GB over 9000 episodes -- about
    73 GB extrapolated to 30k, against 75 GB available -- while the FIT itself needs only 8.1
    GB at 9000 (``_load`` reads just the columns the critic uses and skips ``fm_plan``, 11.3
    kB/window on its own). So the memory ceiling was the merge, not the training, and capping
    the dataset to fit it was throwing away 70% of the episodes for no reason. Appending into
    resizable datasets caps the peak at the largest single stem.

    Args:
        stems: Input rollout stems (without ``.h5``).
        out_stem: Output stem; ``<out_stem>.h5`` and ``<out_stem>.episodes.h5`` are written.
        max_episodes_per_stem: Keep at most this many episodes from each stem; 0 keeps all.
            Balanced BY STEM rather than over the pool, so one cell cannot dominate. Rarely
            needed now that the merge streams -- it is for deliberately thinning a set, not
            for fitting one into memory.
        seed: Draw seed for the subsample.

    Raises:
        SystemExit: On a contract mismatch, a uid collision, or a window whose episode has no
            outcome row.
    """
    for s in stems:
        for suffix in (".h5", ".episodes.h5"):
            if not os.path.isfile(s + suffix):
                raise SystemExit(f"{s + suffix} not found -- is {s!r} a rollout stem?")
    os.makedirs(os.path.dirname(os.path.abspath(out_stem)) or ".", exist_ok=True)

    wattrs: list[dict] = []
    eattrs: list[dict] = []
    prov: list[dict] = []
    offset, n_win_total, n_eps_total = 0, 0, 0
    seen_uids: set[int] = set()
    wpath, epath = out_stem + ".h5", out_stem + ".episodes.h5"

    with h5py.File(wpath, "w") as fw, h5py.File(epath, "w") as fe:
        for i, stem in enumerate(stems):
            W, wa = ro.read_rollout_windows(stem + ".h5")
            E, ea = ro.read_rollout_episodes(stem + ".episodes.h5")
            n_all = len(E["episode_uid"])
            if max_episodes_per_stem:
                subsample_episodes(W, E, int(max_episodes_per_stem),
                                   np.random.default_rng([seed, i]))
            wattrs.append(wa)
            eattrs.append(ea)
            # Checked as each stem arrives, so a mismatch is caught before anything is
            # written rather than after the whole set has been read.
            assert_contracts_agree(wattrs, stems[:i + 1])

            # An orphan can only exist WITHIN a stem: offsetting shifts both tables equally.
            orphans = np.setdiff1d(np.unique(W["episode_uid"]), E["episode_uid"])
            if orphans.size:
                raise SystemExit(
                    f"{stem}: {orphans.size} window episode_uid(s) have no outcome row "
                    f"(e.g. {orphans[:5].tolist()}); the join would drop them silently.")

            n = int(max(E["episode_uid"].max(initial=-1),
                        W["episode_uid"].max(initial=-1))) + 1
            E["episode_uid"] = E["episode_uid"] + offset
            W["episode_uid"] = W["episode_uid"] + offset
            offset += n
            clash = seen_uids & set(E["episode_uid"].tolist())
            if clash:
                raise SystemExit(
                    f"{stem}: uid(s) {sorted(clash)[:5]} already used. The offset is wrong, "
                    "and fit_value would deal one episode's windows to both sides of its "
                    "by-episode split. Refusing.")
            seen_uids |= set(E["episode_uid"].tolist())

            for f, cols in ((fw, W), (fe, E)):
                for k, v in cols.items():
                    if k not in f:
                        # CHUNK WHOLE ROWS. `chunks=True` lets h5py pick, and its heuristic
                        # balances the axes rather than knowing that a row is the unit every
                        # reader asks for -- on a (326270, 64, 44) window table it chose
                        # (10196, 2, 3), so ONE row's `fm_plan` spans ceil(64/2)*ceil(44/3) =
                        # 480 chunks. Measured across the four window datasets that is
                        # 172 MB decompressed to serve a 27 KB window, against 16.8 MB from
                        # the recorder's own `_chunk` -- and at 86k windows an epoch of the
                        # fine-tune's random-order reads became 14.9 TB of lzf, which is a
                        # 50-minute epoch on data the FM trains through in a fraction of that.
                        # `_chunk` is the recorder's rule and is reused rather than restated.
                        f.create_dataset(
                            k, data=v, maxshape=(None,) + v.shape[1:],
                            chunks=ro._chunk((1 << 24,) + v.shape[1:], v.dtype.str),
                            compression="lzf" if v.nbytes > (1 << 20) else None)
                    else:
                        d = f[k]
                        d.resize(d.shape[0] + v.shape[0], axis=0)
                        d[-v.shape[0]:] = v
            n_win_total += len(W["episode_uid"])
            n_eps_total += len(E["episode_uid"])
            note = "" if len(E["episode_uid"]) == n_all else f"  (subsampled from {n_all})"
            print(f"[merge] {stem}: {len(E['episode_uid'])} episodes, "
                  f"{len(W['episode_uid'])} windows{note}", flush=True)
            prov.append({"stem": stem, "n_episodes": int(len(E["episode_uid"])),
                         "n_windows": int(len(W["episode_uid"])),
                         "attrs": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in ea.items()
                                   if k not in ("depth_camera_spec", "height_scan_spec")}})
            del W, E                                    # release before the next stem

        # BOTH files keep their source attrs. The episode table's are not decoration:
        # `fit_value._tracking_spec_from` reads the strict spec off them, and `family_names`
        # is what makes the family_code column legible. Dropping them left the merged set
        # without a tracking spec -- which the hand-built test fixtures could not see,
        # because they only carry schema_version.
        for f, base, is_windows in ((fw, wattrs[0], True), (fe, eattrs[0], False)):
            for k, v in base.items():
                f.attrs[k] = v if isinstance(v, (str, bytes)) else json.dumps(
                    v.tolist() if isinstance(v, np.ndarray) else v, default=str)
            # **The window COUNT is the merged one, not the first file's.**
            # `read_rollout_windows` slices `[:num_windows_written]`, so inheriting the
            # source's value silently truncates the merged table to the first file's length:
            # the file holds every row, every reader sees a fraction of them, and a critic
            # trained on it reports nothing wrong. Caught by test_merge_rollouts.
            if is_windows:
                for key in ("num_windows", "num_windows_written"):
                    f.attrs[key] = n_win_total
            f.attrs["merge_provenance"] = json.dumps({"merged_sources": prov}, default=str)
        fe.attrs["schema_version"] = ro.SCHEMA_VERSION

    print(f"[merge] wrote {wpath}: {n_win_total} rows")
    print(f"[merge] wrote {epath}: {n_eps_total} rows")
    print(f"[merge] {n_eps_total} episodes, {n_win_total} windows, uids 0..{offset - 1}")


def main() -> int:
    """Parse arguments and merge."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stems", nargs="+", help="rollout stems (without .h5)")
    p.add_argument("--out", required=True, help="output stem")
    p.add_argument("--max_episodes_per_stem", type=int, default=0,
                   help="keep at most this many episodes from EACH stem (0 = all). The "
                        "exteroception column is ~15.6 kB a window, so a six-run grid is "
                        "~28 GB and the concatenation peaks at about twice that.")
    p.add_argument("--seed", type=int, default=0, help="subsample draw seed")
    a = p.parse_args()
    merge(list(a.stems), a.out, a.max_episodes_per_stem, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
