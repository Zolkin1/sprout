#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Pick BALLAST windows: a fixed slice of the ORIGINAL pre-training set, carried at weight 1.

AWR reweights the generator toward one course. Ballast is what stops that quietly costing the
distribution it was pre-trained on: a fixed number of windows from the pre-training recording,
trained on at weight 1 alongside the fine-tune buffer, not graded, not fitted, and taking no
part in the advantage.

**The exclusion is STRUCTURAL, not a flag.** The ballast lives in its own file and only
``finetune_generator.py`` is ever pointed at it -- ``fit_value.py`` and ``label_advantage.py``
see the buffer alone. The alternative, merging ballast windows into the buffer, would need them
renumbered into a uid range no iteration can ever reach, because every join downstream is
``(episode_uid, replan_index)``; one bad offset there makes a ballast window indistinguishable
from a graded one, and nothing checks it. Separate files cannot be got wrong that way.

**BALLAST READS THE PRE-TRAINING RECORDING ITSELF, not a re-recording of it.** The two files
look different and are not: ``recorded_*.h5`` stores the raw ingredients (``ref_future``,
``history_state``, ``depth``) and ``FutureStateDataset`` assembles them into model space, while
a fine-tune ``rollout_*.h5`` stores windows already assembled (``fm_plan``, ``history``,
``extero``). Both end as the same four tensors in the same loss -- ``state``, ``history``,
``extero``, ``cmd_tokens`` -- so the pre-training loader is simply the right reader for the
pre-training file, and either kind can serve as ballast. :func:`build_ballast_dataset`
dispatches on which one it was handed.

**The normalizers are the FINE-TUNE'S, never freshly fitted.** ``FutureStateDataset`` will fit
its own if none are passed -- which is correct when it IS the training set and silently wrong
here: the ballast would land in a different space from the buffer and quietly pull the model
toward a rescaled version of itself, with no error and no shape change to show for it.

**The flat share is declared, because a terrain course contains none.** The fine-tune buffer is
entirely terrain windows, so drift on FLAT ground is invisible to it -- the buffer has no flat
rows for the critic to grade or the loss to see. ``ballast_flat_frac`` is the share drawn from
flat windows specifically, and the rest come from the terrain families, so the ballast is a
stated mixture rather than whatever the pre-training recording happened to be made of.

**Ballast needs the RECORDING'S OWN CLIP LIBRARY on disk, and the buffer path does not.**
The fine-tune's buffer is rollout windows, already in model space; ballast goes through
``FutureStateDataset``, which resolves the library the recording names (for the anchor masks
and the frame metadata). So a checkout that can run a fine-tune today can still fail at the
ballast step with "clip library '...' (from the recording) does not exist" -- a loud error,
but a surprising one, because nothing else in the loop wants that file.

**A stratum that cannot fill its share is an ERROR.** Silently returning fewer windows would
make the declared mixture a fiction while every log line still read normally, and the whole
point of the number is that it is declared.
"""

from __future__ import annotations

import json
import os
import sys

import h5py
import numpy as np

#: ``scripts/rsl_rl`` on the path, because THIS module is the one with the dependency.
#:
#: ``build_ballast_dataset`` reads a PRE-TRAINING recording, and the reader for those
#: (``recorded_data_io``) is a sibling of ``record_data.py`` over in ``scripts/rsl_rl`` --
#: while this file sits in ``scripts/flow_matching``. Its importer, ``finetune_generator.py``,
#: puts ``source/robot_rl`` and ``source/motion_gen`` on the path and nothing else, so the
#: deferred import below resolved only by luck of who had already imported what.
#:
#: It went unnoticed because THE BALLAST PATH IS THE ONLY CALLER: of the four fine-tune
#: configs only ``multi`` sets ``ballast_chunks > 0``, and the import is deferred to the
#: bottom of ``build_ballast_dataset``. So the other three run clean, and multi got through
#: collect, grade, the buffer, both critic rounds and the AWR weights -- 67 minutes -- before
#: dying on a missing module at stage 8 of 8.
#:
#: Set here rather than in the caller so any future importer of this module inherits it.
_RSL_RL = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       os.pardir, "rsl_rl"))
if _RSL_RL not in sys.path:
    sys.path.insert(0, _RSL_RL)

#: Family names that count as FLAT. Matched after stripping a trailing ``.h5``, the way
#: ``record_rollouts`` normalises them, and by exact name rather than substring: a substring
#: test on "flat" also matches nothing today but would match any future ``flat_*`` family
#: silently, which is the class of bug ``isaac_tile_course``'s FAMILY_KLASS table exists for.
#:
#: **``unknown`` is in the set on purpose, and it is the one that carries the flat windows.**
#: A flat clip sits on a synthesised patch with no terrain-manifest entry, so it has no family
#: to record and lands in code 0. Measured on the v6g pre-training recording
#: (``recorded_20260912_090151.h5``, 200k windows): code 0 is 30.5% of them against the
#: library's known ~30% flat share, and every other code names a real terrain family.
#:
#: That is an inference from a count, not a guarantee -- ``unknown`` means "no family was
#: recorded", which is flat on every library built this way and would be something else on a
#: library that failed to stamp its families at all. :func:`describe` prints the realised
#: flat/terrain split into the run banner for exactly that reason, so a recording where the
#: assumption does not hold shows up as a share nobody asked for rather than as nothing.
FLAT_FAMILY_NAMES: frozenset[str] = frozenset({"flat", "family_flat", "unknown"})


def _family_names(f: h5py.File, path: str = "") -> list[str]:
    """The recording's own ``family_names`` table, or an empty list.

    **It is not always on the window file.** ``record_rollouts`` writes ``family_code`` into
    the WINDOW file but stamps ``family_names`` onto the ``.episodes.h5`` sibling, so a
    rollout recording carries the codes and its neighbour carries what they mean. Looking only
    at the window file made every ``ballast_flat_frac > 0`` run refuse with a message blaming
    the recording's age, which is not the reason.

    Args:
        f: The open window file.
        path: That file's path, so the ``.episodes.h5`` sibling can be found.

    Returns:
        ``family_names[code]``.
    """
    raw = f.attrs.get("family_names")
    if raw is None and "meta" in f.attrs:
        try:
            raw = json.loads(str(f.attrs["meta"])).get("family_names")
        except Exception:
            raw = None
    if raw is None and path:
        # The sibling. `prov` is written JSON-encoded, so the value needs decoding either way.
        sib = path[:-3] + ".episodes.h5" if path.endswith(".h5") else ""
        if sib and os.path.isfile(sib):
            with h5py.File(sib, "r") as g:
                raw = g.attrs.get("family_names")
    if raw is None:
        return []
    if isinstance(raw, (bytes, str)):
        try:
            raw = json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception:
            return []
    return [str(v) for v in np.asarray(raw).ravel().tolist()]


def _n_windows(f: h5py.File) -> int:
    """How many windows a recording holds, by attr or by array length.

    Both writers stamp ``num_windows_written``; the array fallback covers a file written by
    something that did not, where a count of 0 would otherwise refuse every draw as "more
    than the file holds".

    Args:
        f: The open window file.

    Returns:
        The window count.
    """
    n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
    if n:
        return n
    for k in ("ref_future", "fm_plan", "episode_id"):
        if k in f:
            return int(f[k].shape[0])
    return 0


def is_pretraining_recording(path: str) -> bool:
    """Whether ``path`` is a pre-training recording rather than a fine-tune rollout file.

    Discriminated on the ARRAYS rather than the name: a pre-training recording carries
    ``ref_future`` (the raw reference window, assembled at load) and a rollout file carries
    ``fm_plan`` (the window already in model space).

    Args:
        path: The ``.h5``.

    Returns:
        True for a pre-training recording.

    Raises:
        SystemExit: If it is neither, which means it is not a window recording at all.
    """
    with h5py.File(path, "r") as f:
        if "ref_future" in f:
            return True
        if "fm_plan" in f:
            return False
    raise SystemExit(
        f"{path} carries neither `ref_future` (a pre-training recording) nor `fm_plan` (a "
        "fine-tune rollout file), so it holds no windows this can draw ballast from.")


def _family_columns(path: str) -> tuple[np.ndarray, list[str]]:
    """``(per-window family code, names[code])`` from either recording schema.

    The two schemas name this differently and keep the names in different places, which is
    the whole reason a declared flat share needed a dispatch: a rollout file has
    ``family_code`` with its names on the ``.episodes.h5`` SIBLING, and a pre-training
    recording has ``clip_family`` with ``family_names`` in its own metadata.

    Args:
        path: The window ``.h5``.

    Returns:
        The codes and their names.
    """
    # h5py only: SELECTION must not need the recording's full metadata. Both writers store
    # their meta fields as JSON-encoded attrs, so the names are readable either way and the
    # only real difference is the column's name -- `clip_family` on a pre-training recording,
    # `family_code` on a rollout file.
    col = "clip_family" if is_pretraining_recording(path) else "family_code"
    with h5py.File(path, "r") as f:
        n = _n_windows(f)
        names = _family_names(f, path)
        if col not in f:
            return np.zeros(0, dtype=np.int64), names
        return f[col][:n], names


def flat_mask(path: str) -> np.ndarray:
    """``[W]`` bool: which windows of a recording are FLAT-family.

    Args:
        path: The window ``.h5``.

    Returns:
        The mask over every written window.

    Raises:
        SystemExit: If the recording carries no family table, so flat cannot be told from
            terrain. A declared flat share would then be unverifiable.
    """
    code, names = _family_columns(path)
    if not names or code.size == 0:
        raise SystemExit(
            f"{path} carries no family table, so FLAT windows cannot be told from terrain "
            "ones. On a fine-tune rollout file the names live on the `.episodes.h5` SIBLING, "
            "so check it exists beside this one; on a pre-training recording they are in its "
            "own metadata. Failing either, the recording is flat-only (which legitimately has "
            "no families) or predates the stamp -- set ballast_flat_frac to 0 to draw without "
            "a stratum.")
    norm = [v[:-3] if v.endswith(".h5") else v for v in names]
    flat_codes = {i for i, v in enumerate(norm) if v in FLAT_FAMILY_NAMES}
    return np.isin(code, np.array(sorted(flat_codes), dtype=code.dtype))


def select_ballast_rows(path: str, n_chunks: int, flat_frac: float = 0.0,
                        seed: int = 0) -> np.ndarray:
    """Row indices into ``path`` for the ballast, honouring the declared flat share.

    Args:
        path: The pre-training window ``.h5``.
        n_chunks: How many windows to carry. ``0`` disables the ballast.
        flat_frac: Share drawn from FLAT windows; the rest from terrain ones. ``0`` draws
            uniformly and never reads the family table.
        seed: Draw seed.

    Returns:
        Sorted row indices, exactly ``n_chunks`` of them.

    Raises:
        SystemExit: If the file is unreadable, if ``n_chunks`` exceeds what it holds, or if a
            stratum cannot fill its declared share.
    """
    if n_chunks <= 0:
        return np.zeros(0, dtype=np.int64)
    if not (0.0 <= flat_frac <= 1.0):
        raise SystemExit(f"ballast_flat_frac must be in [0, 1], got {flat_frac}")
    try:
        with h5py.File(path, "r") as f:
            n = _n_windows(f)
    except OSError as e:
        raise SystemExit(f"cannot read the ballast recording {path}: {e}") from None
    if n_chunks > n:
        raise SystemExit(
            f"asked for {n_chunks} ballast windows but {path} holds {n}. The ballast is a "
            "fixed slice of the pre-training set, so a short one is a different experiment "
            "from the declared one.")

    rng = np.random.default_rng(seed)
    if flat_frac <= 0.0:
        return np.sort(rng.choice(n, size=n_chunks, replace=False)).astype(np.int64)

    is_flat = flat_mask(path)
    want_flat = int(round(n_chunks * float(flat_frac)))
    want_terr = n_chunks - want_flat
    picked = []
    for name, want, pool in (("flat", want_flat, np.flatnonzero(is_flat)),
                             ("terrain", want_terr, np.flatnonzero(~is_flat))):
        if want == 0:
            continue
        if pool.size < want:
            raise SystemExit(
                f"the ballast asks for {want} {name} window(s) but {path} holds only "
                f"{pool.size}. Returning fewer would make the declared "
                f"ballast_flat_frac={flat_frac} a fiction while every log line read normally.")
        picked.append(rng.choice(pool, size=want, replace=False))
    return np.sort(np.concatenate(picked)).astype(np.int64)


def describe(path: str, rows: np.ndarray) -> str:
    """A one-line summary of what the ballast slice contains, for the run banner.

    Args:
        path: The window file.
        rows: The selected rows.

    Returns:
        The summary.
    """
    if rows.size == 0:
        return "ballast: none"
    try:
        flat = int(flat_mask(path)[rows].sum())
        share = f", {flat} flat / {rows.size - flat} terrain"
    except SystemExit:
        share = ""
    return f"ballast: {rows.size} window(s) from {path}{share}, weight 1, ungraded"


def build_ballast_dataset(path: str, rows, ctx, cfg, *, null_prob: float, seed: int,
                          label: bool, weight: bool):
    """The ballast slice as a dataset in the fine-tune's item shape.

    Dispatches on which kind of recording it was handed. Either is legitimate: the
    pre-training set is what "ballast" literally means, and a rollout recording of the parent
    policy is the same thing collected later.

    **The normalizers come from ``ctx``**, i.e. from the checkpoint being fine-tuned.
    ``FutureStateDataset`` fits its own when none are passed, which is right when it IS the
    training set and silently wrong here -- the ballast would sit in a different space from
    the buffer, pulling the model toward a rescaled version of itself with no error to show.

    Args:
        path: The ballast recording.
        rows: Row indices from :func:`select_ballast_rows`.
        ctx: The loaded generator context, for its normalizers.
        cfg: The checkpoint's ``FlowMatchingConfig``, for the window layout.
        null_prob: History-dropout probability, matched to the buffer's.
        seed: Draw seed.
        label: Emit an ``advantage`` key (the null label), mirroring the buffer.
        weight: Emit a ``weight`` key of 1.0, mirroring the buffer.

    Returns:
        A dataset whose items carry the fine-tune's keys.
    """
    import numpy as _np
    from robot_rl.flow_matching.advantage import LABEL_NULL
    from robot_rl.flow_matching.rollout_dataset import (
        PretrainWindowAdapter,
        RolloutWindowDataset,
    )

    n = int(_np.asarray(rows).size)
    lab = _np.full(n, LABEL_NULL, dtype=_np.int64) if label else None
    wgt = _np.ones(n, dtype=_np.float64) if weight else None

    if not is_pretraining_recording(path):
        # A rollout file is already in model space; its label/weight arrays are per FILE row.
        with h5py.File(path, "r") as f:
            n_file = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        return RolloutWindowDataset(
            path, _np.asarray(rows), ctx=ctx, seed=seed, null_prob=null_prob,
            label=(_np.full(n_file, LABEL_NULL, dtype=_np.int64) if label else None),
            weight=(_np.ones(n_file, dtype=_np.float64) if weight else None))

    from recorded_data_io import load_recorded_windows
    from robot_rl.flow_matching.dataset import FutureStateDataset

    # ROWS AT LOAD, not after: the pre-training recording is the whole training set (200k
    # windows with depth) and the ballast is a couple of thousand, so loading all of it to
    # keep 1% would cost gigabytes for nothing.
    data, meta = load_recorded_windows(path, rows=_np.asarray(rows))
    # THE SAME CAMERA SUBSET THE CHECKPOINT WAS TRAINED UNDER. `train.py` calls this once on
    # the training set, and its docstring says that is deliberate: every consumer derives the
    # sensor count from `data`'s SHAPE, so slicing here leaves the rest alone. The ballast path
    # loads its own recording and so needs its own call -- without it a NEAR-ONLY generator
    # (extero_sensor_count=1) is handed the recording's TWO-camera rows, because
    # `PretrainWindowAdapter` takes `extero_shape` from the recording's attributes rather than
    # from the model. Measured on the v6g pair: the model wants 1x3900 and the recording holds
    # (N, 2, 780).
    #
    # `cfg` is the CHECKPOINT's own config, so this follows the parent automatically: a
    # two-camera parent has `extero_keep_cameras=None` and the call is a no-op.
    from robot_rl.flow_matching.dataset import apply_camera_subset  # noqa: PLC0415
    data, meta = apply_camera_subset(
        data, meta, getattr(cfg, "extero_keep_cameras", None),
        getattr(cfg, "extero_source", "depth"))
    # NORMALIZERS CLONED TO THE CPU, exactly as `rollout_dataset._CtxView` does and for the
    # same two reasons -- this path had been passing `ctx`'s own straight through.
    #
    #   * A dataset builds CPU tensors, so a CUDA normalizer gives
    #     "Expected all tensors to be on the same device" inside `Normalizer.normalize`.
    #     It is invisible on a CPU-loaded context, which is how it survived an offline check.
    #   * `Normalizer.to` moves IN PLACE, so normalizing without the deepcopy would leave the
    #     loaded generator holding CPU statistics against a CUDA model -- which fails later,
    #     at the next sample, far from here and looking like a model fault.
    import copy  # noqa: PLC0415
    state_norm = copy.deepcopy(ctx.state_norm).to("cpu")
    extero_norm = copy.deepcopy(ctx.extero_norm).to("cpu")
    cmd_norm = None if ctx.cmd_norm is None else copy.deepcopy(ctx.cmd_norm).to("cpu")
    base = FutureStateDataset(
        data, meta, cfg, _np.arange(n),
        state_normalizer=state_norm, cmd_normalizer=cmd_norm,
        extero_normalizer=extero_norm, add_noise=False)
    return PretrainWindowAdapter(base, path=path, label=lab, weight=wgt)
