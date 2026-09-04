# Copyright 2026 Zachary Olkin. All rights reserved.

"""Chunk data for the tokenizer, taken from the flow-matching dataset unchanged.

The tokenizer trains on exactly the array the generator regresses:
:attr:`~robot_rl.flow_matching.dataset.FutureStateDataset.state`, the normalized ``(H, D)``
ego-anchored clip future. So this module does not build windows -- it calls
:func:`~robot_rl.flow_matching.dataset.build_train_val` and keeps the ``state`` half. Three
things follow, and each of them is the reason for doing it this way:

* the **split is identical** to the generator's (by ``episode_id``, from ``cfg.seed`` and
  ``cfg.val_fraction``), so a later AR-vs-FM comparison is on the same held-out data;
* the **normalizer is identical** -- the same rot6d standardization exemption, the same
  declared per-block scales -- so a reconstruction error in normalized units means the same
  thing in both stages;
* there is **one implementation of window construction**, so it cannot drift. Reproducing
  ``_rows_to_window`` here would be a second answer to a question that already has one.

.. admonition:: The exteroception is loaded, and never read

    ``FutureStateDataset`` requires an exteroception array (it is a conditioning-model
    dataset), but a chunk-local autoencoder conditions on NOTHING -- it sees one chunk of
    nodes and reconstructs it. So :func:`build_chunk_data` selects the *cheapest* array
    present purely to satisfy that requirement: ``height_scan`` is 176 floats per window
    where the depth pair is 7800, which on the shipped 200k-window recording is 141 MB
    against 6.2 GB. Nothing downstream of the build touches it.

    That this cannot move the targets is asserted, not asserted-by-comment:
    ``test_extero_source_does_not_move_the_state_targets`` builds the same synthetic
    recording under both sources and requires the ``state`` tensors to be bit-identical.

.. admonition:: The cache exists so the ablations are cheap

    Building from the recording reads and decompresses tens of GB and takes minutes; the
    tokenizer's own sweeps (``D``, ``channels``, the FSQ level table) change none of it.
    :func:`save_chunk_cache` / :func:`load_chunk_cache` therefore persist the built chunks,
    keyed on everything that could change them -- the input file hashes, the horizon, the
    block selection, the split seed and fraction, the filter flags. A key mismatch REFUSES
    rather than rebuilding silently, because a cache quietly serving the wrong split is the
    one failure that would make every number in the report wrong and nothing would say so.
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

@functools.lru_cache(maxsize=1)
def _recorded_data_io() -> ModuleType:
    """The recording reader, which lives beside the scripts that write recordings.

    ``recorded_data_io`` is not part of either installed package -- it sits in
    ``scripts/rsl_rl/`` next to ``record_data.py``, and every consumer so far has been a
    script whose own directory is already ``sys.path[0]``. This module is the first LIBRARY
    consumer, so it has to put that directory on the path itself.

    The root comes from :data:`motion_gen.features._REPO_ROOT`, which is the repo's one
    sanctioned root marker; assembling it from ``os.pardir`` hops off ``__file__`` has
    silently broken here before when a module moved between packages, which is why
    CLAUDE.md forbids it. Same manoeuvre as ``scripts/rsl_rl/eval/_paths.py``.

    Returns:
        The imported ``recorded_data_io`` module.
    """
    from motion_gen.features import _REPO_ROOT

    scripts_dir = str(_REPO_ROOT / "scripts" / "rsl_rl")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import recorded_data_io  # noqa: PLC0415

    return recorded_data_io


#: Cache format version. Bump on any change to what :func:`build_chunk_data` returns; a
#: cache written by an older version is refused rather than reinterpreted.
CACHE_VERSION: int = 1

#: Exteroception arrays in increasing cost per window, for the "cheapest present" choice.
_EXTERO_BY_COST: tuple[str, ...] = ("height_scan", "depth")


@dataclass
class ChunkData:
    """Built chunks plus everything needed to interpret and reproduce them.

    Attributes:
        train_nodes: ``(Nw_train, H, D)`` normalized chunks.
        val_nodes: ``(Nw_val, H, D)`` normalized chunks; empty when the split held nothing
            out.
        train_family / val_family: ``(Nw,)`` int64 family codes, indices into
            :attr:`family_names`. These are what "split by terrain type" means in the
            metrics -- the recording's own per-window family tag, joined by window index.
        state_normalizer_state: the state :class:`~robot_rl.flow_matching.normalizer.Normalizer`
            as a ``state_dict``, so reconstruction error can be reported in PHYSICAL units.
        family_names: Family name per code, from the recording.
        reference_blocks / num_joints / num_ee_frames / chunk_len / node_dim / dt: the layout
            the chunks are in.
        val_episode_ids: The held-out episode ids, so the split is auditable and a later
            stage can assert it matched.
        provenance: Input paths, their hashes, and the resolved build settings.
    """

    train_nodes: torch.Tensor
    val_nodes: torch.Tensor
    train_family: torch.Tensor
    val_family: torch.Tensor
    state_normalizer_state: dict[str, torch.Tensor]
    family_names: list[str]
    reference_blocks: list[str]
    num_joints: int
    num_ee_frames: int
    chunk_len: int
    node_dim: int
    dt: float
    val_episode_ids: list[int]
    provenance: dict[str, Any] = field(default_factory=dict)


def cheapest_extero_source(data: dict[str, np.ndarray]) -> str:
    """Pick the narrowest exteroception array the recording carries.

    Args:
        data: Recorded arrays.

    Returns:
        An entry of :data:`_EXTERO_BY_COST`.

    Raises:
        ValueError: If the recording carries neither, which means it is not an FM recording
            and the state layout cannot be trusted either.
    """
    for name in _EXTERO_BY_COST:
        if data.get(name) is not None:
            return name
    raise ValueError(
        "recording carries no exteroception array "
        f"({', '.join(_EXTERO_BY_COST)}); it is not a flow-matching recording."
    )


def load_recordings(
    paths: Sequence[str], keys: Sequence[str] | None = None
) -> tuple[dict[str, np.ndarray], Any]:
    """Load and concatenate windowed recordings, offsetting ``episode_id`` per file.

    A faithful sibling of the combine block in ``scripts/flow_matching/train.py`` (which
    lives inside that script's ``main`` and is therefore not importable). The parts that are
    load-bearing rather than convenient:

    * **the episode offset.** The train/val split is by ``episode_id``, so two files whose
      ids both start at 0 would put the same id in both splits and leak the held-out data.
    * **the identical-key requirement.** ``np.concatenate`` on a missing key is a
      ``KeyError`` at best and a silently short array at worst.

    The FM's sensor-geometry agreement check is deliberately NOT reproduced: it guards the
    exteroception, which nothing here reads.

    Args:
        paths: One or more windowed ``.h5`` recordings.
        keys: Array keys to load, or ``None`` for all of them.

    Returns:
        ``(data, meta)`` with ``meta`` taken from the FIRST file, matching the FM.

    Raises:
        ValueError: On no paths, or on files whose array key sets disagree.
    """
    load_recorded_windows = _recorded_data_io().load_recorded_windows

    if not paths:
        raise ValueError("load_recordings needs at least one path.")
    data: dict[str, np.ndarray] = {}
    meta = None
    ep_offset = 0
    for i, path in enumerate(paths):
        d, m = load_recorded_windows(path, keys=list(keys) if keys else None)
        d = {k: v for k, v in d.items() if v is not None}
        # Offset FIRST, then store or concatenate. The offset is 0 for the first file, so
        # doing it after the store happened to work -- but the write went to a dict that was
        # about to be discarded, which is the kind of dead line that becomes a bug the moment
        # the first file's offset stops being zero.
        if d.get("episode_id") is not None:
            d["episode_id"] = np.asarray(d["episode_id"]) + ep_offset
            ep_offset = int(d["episode_id"].max()) + 1
        if i == 0:
            meta = m
            data = d
            continue
        if set(d) != set(data):
            raise ValueError(
                f"{path} carries arrays {sorted(d)} but the first file carries "
                f"{sorted(data)}; concatenating them would misalign the windows."
            )
        for k in list(data):
            data[k] = np.concatenate([data[k], d[k]], axis=0)
    return data, meta


class ChunkDataset(Dataset):
    """One normalized chunk per item, with its family code.

    Args:
        nodes: ``(Nw, H, D)`` normalized chunks.
        family: ``(Nw,)`` int64 family codes.
    """

    def __init__(self, nodes: torch.Tensor, family: torch.Tensor):
        if nodes.shape[0] != family.shape[0]:
            raise ValueError(
                f"{nodes.shape[0]} chunk(s) but {family.shape[0]} family code(s)."
            )
        self.nodes = nodes
        self.family = family

    def __len__(self) -> int:
        return int(self.nodes.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Return ``{"nodes": (H, D), "family": ()}``."""
        return {"nodes": self.nodes[i], "family": self.family[i]}


def build_chunk_data(
    paths: Sequence[str],
    chunk_len: int,
    *,
    seed: int = 0,
    val_fraction: float = 0.1,
    stride: int = 1,
    skip_distractors: bool = True,
    skip_families: Sequence[str] | None = None,
    reference_blocks: Sequence[str] | None = None,
) -> ChunkData:
    """Build train/val chunks by running the flow-matching dataset and keeping ``state``.

    Args:
        paths: Windowed ``.h5`` recordings, concatenated in order.
        chunk_len: Nodes per chunk ``H``; becomes the FM config's ``future_len``, so the
            recording must carry at least this many ``ref_future`` nodes.
        seed: Split seed. Must match the generator's to get the generator's split.
        val_fraction: Held-out fraction of EPISODES.
        stride: Global window subsample, applied after the split.
        skip_distractors: Drop distractor windows, as the shipped generator does.
        skip_families: Families to drop; ``None`` takes the FM's own default (which is
            ``family_jump_off``, unobservable to a forward-facing camera).
        reference_blocks: Node block selection, or ``None`` to take the recording's.

    Returns:
        The built :class:`ChunkData`.

    Raises:
        ValueError: If the recording's per-window family tag is missing while a family
            filter is requested, or if the built dataset has no windows.
    """
    from robot_rl.flow_matching.config import DEFAULT_SKIP_FAMILIES, FlowMatchingConfig
    from robot_rl.flow_matching.dataset import (
        _distractor_keep_mask,
        _family_keep_mask,
        build_train_val,
    )

    rdio = _recorded_data_io()
    paths = [str(p) for p in paths]
    # Only the arrays the state target and the split need, plus the family tag and the
    # cheapest exteroception. Loading `depth`/`depth_lag` here would read 6.2 GB per copy
    # for an array the autoencoder never sees.
    want = {
        "ref_future", "history_state", "history_ref", "episode_id",
        "is_distractor", "clip_family", "clip_id", "clip_step", "env_id", "cmd_future",
    }
    # Which exteroception array to satisfy the FM dataset with -- ONE statement of the
    # rule, so a recording carrying neither hits its own error here rather than resolving
    # to "depth" and failing later with an unrelated message.
    head, _ = load_recordings(paths[:1], keys=list(_EXTERO_BY_COST))
    extero = cheapest_extero_source(head)
    del head
    keys = sorted((want | {extero}) & set(rdio.WINDOW_ARRAY_KEYS))
    data, meta = load_recordings(paths, keys=keys)

    cfg = FlowMatchingConfig(
        future_len=int(chunk_len),
        num_history_states=1,          # the smallest legal value; the history is not used
        stride=int(stride),
        val_fraction=float(val_fraction),
        seed=int(seed),
        skip_distractors=bool(skip_distractors),
        skip_families=(list(DEFAULT_SKIP_FAMILIES) if skip_families is None
                       else [str(s) for s in skip_families]),
        reference_blocks=(None if reference_blocks is None else list(reference_blocks)),
        extero_source=extero,
        # No exteroception normalization work, no camera-latency draw, no clip-library join:
        # three separate passes over the widest arrays in the build, all of them producing
        # quantities that only a conditioning model reads.
        extero_norm_mode="metric",
        depth_latency_probs=None,
        pinned_axis_loss_weight=1.0,
        pinned_axis_metrics=False,
        cmd_array=None,
        obs_noise=False,
    )
    train_ds, val_ds = build_train_val(data, meta, cfg)
    if len(train_ds) == 0:
        raise ValueError("the build produced no training chunks.")

    fam_names = [str(s) for s in (getattr(meta, "family_names", None) or ["unknown"])]
    # ``window_sel`` indexes the FILTERED window array, not the recording. The dataset builds
    # its own local ``data`` with the distractor and family masks already applied
    # (``_build_train_val_windowed``), so joining a label straight off the recording is off by
    # however many windows the filters removed -- which is exactly the shape of a silent
    # mislabelling: every count is plausible and the tensor is the right length. It was caught
    # only because ``family_jump_off`` then appeared among the labels of a run that drops it.
    #
    # So the masks are reproduced here with the dataset's OWN mask functions, in the same
    # composition order, and then the RESULT is checked against the filters' postconditions
    # below. The check is the part that matters: a third filter added to the dataset would
    # desync this join again, and an assertion on the labels notices where restating the
    # composition cannot.
    n_all = int(np.asarray(data["history_state"]).shape[0])
    keep = _distractor_keep_mask(data, cfg, n_all)
    fam_keep = _family_keep_mask(data, meta, cfg, n_all)
    keep = fam_keep if keep is None else (keep if fam_keep is None else (keep & fam_keep))
    fam_all = data.get("clip_family")
    if fam_all is not None:
        fam_all = np.asarray(fam_all)
        fam_all = fam_all if keep is None else fam_all[keep]
    dis_all = data.get("is_distractor")
    if dis_all is not None:
        dis_all = np.asarray(dis_all)
        dis_all = dis_all if keep is None else dis_all[keep]
    dropped = {i for i, nm in enumerate(fam_names) if nm in set(cfg.skip_families)}

    def _family_of(ds) -> torch.Tensor:
        """Per-window family codes for a split, joined through the filtered index space.

        Args:
            ds: A built :class:`FutureStateDataset`, or ``None``.

        Returns:
            ``(len(ds),)`` int64 family codes.

        Raises:
            RuntimeError: If the join produced a window whose family was supposed to be
                dropped, or a distractor under ``skip_distractors`` -- i.e. if the index
                spaces have come apart again.
        """
        if ds is None:
            return torch.zeros(0, dtype=torch.long)
        n = len(ds)
        sel = getattr(ds, "window_sel", None)
        if fam_all is None or sel is None:
            # A recording without the tag is legal (a flat-only library has no families);
            # everything then reports under one bucket rather than silently mislabelled.
            return torch.zeros(n, dtype=torch.long)
        sel = np.asarray(sel)
        codes = fam_all[sel]
        bad = sorted({int(c) for c in codes} & dropped)
        if bad:
            raise RuntimeError(
                f"family join desynced: {[fam_names[i] for i in bad]} appear among the "
                f"labels of a build that drops {sorted(cfg.skip_families)}. The window "
                "index space moved -- check the filter composition against "
                "dataset._build_train_val_windowed."
            )
        if cfg.skip_distractors and dis_all is not None and int(dis_all[sel].max(initial=0)):
            raise RuntimeError(
                "family join desynced: a distractor window appears among the labels of a "
                "build with skip_distractors=True."
            )
        return torch.as_tensor(codes, dtype=torch.long)

    return ChunkData(
        train_nodes=train_ds.state.contiguous(),
        val_nodes=(val_ds.state.contiguous() if val_ds is not None
                   else torch.zeros(0, int(chunk_len), train_ds.state_dim)),
        train_family=_family_of(train_ds),
        val_family=_family_of(val_ds),
        state_normalizer_state=train_ds.state_normalizer.state_dict(),
        family_names=fam_names,
        reference_blocks=list(train_ds.blocks),
        num_joints=int(train_ds.num_joints),
        num_ee_frames=int(train_ds.num_extra_frames),
        chunk_len=int(chunk_len),
        node_dim=int(train_ds.state_dim),
        dt=float(getattr(meta, "dt", 0.02)),
        val_episode_ids=[int(v) for v in (getattr(train_ds, "val_episode_ids", None) or [])],
        provenance={
            "data": ",".join(paths),
            "data_sha256": ",".join(rdio.sha256_file(p) for p in paths),
            "extero_source_loaded": extero,
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "stride": int(stride),
            "skip_distractors": bool(skip_distractors),
            "skip_families": list(cfg.skip_families),
            "recording_dt": float(getattr(meta, "dt", 0.02)),
            "recording_replan_period": int(getattr(meta, "fm_snap_every", 0) or 0),
            "recording_ref_future_len": int(getattr(meta, "ref_future_len", 0) or 0),
        },
    )


def cache_key(provenance: dict[str, Any], chunk_len: int,
              reference_blocks: Sequence[str]) -> dict[str, Any]:
    """The fields a cached build must agree on to be reusable.

    Deliberately NOT the whole provenance: ``extero_source_loaded`` is in there and cannot
    affect the chunks (pinned by a test), so keying on it would invalidate a perfectly good
    cache. Everything listed here can change the chunks or the split.

    Args:
        provenance: A :attr:`ChunkData.provenance`.
        chunk_len: Nodes per chunk.
        reference_blocks: The resolved block selection.

    Returns:
        The key dict, saved beside the tensors and compared on load.
    """
    return {
        "version": CACHE_VERSION,
        "data_sha256": provenance.get("data_sha256", ""),
        "chunk_len": int(chunk_len),
        "reference_blocks": list(reference_blocks),
        "seed": int(provenance.get("seed", 0)),
        "val_fraction": float(provenance.get("val_fraction", 0.0)),
        "stride": int(provenance.get("stride", 1)),
        "skip_distractors": bool(provenance.get("skip_distractors", True)),
        "skip_families": list(provenance.get("skip_families", [])),
    }


def save_chunk_cache(path: str, chunks: ChunkData) -> None:
    """Write built chunks to ``path`` for reuse by the ablations.

    Args:
        path: Destination ``.pt``.
        chunks: The build to persist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "key": cache_key(chunks.provenance, chunks.chunk_len, chunks.reference_blocks),
        "chunks": {k: v for k, v in vars(chunks).items()},
    }
    torch.save(payload, path)


def load_chunk_cache(path: str, key: dict[str, Any]) -> ChunkData:
    """Read a cached build, refusing one that does not match ``key``.

    Args:
        path: A file written by :func:`save_chunk_cache`.
        key: The key the caller requires, from :func:`cache_key`.

    Returns:
        The cached :class:`ChunkData`.

    Raises:
        ValueError: On a key mismatch, naming every field that differs. Rebuilding silently
            would be worse: the whole point of the cache is that a run's numbers are about
            the data the run says they are about.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    have = dict(payload.get("key", {}))
    diff = {k: (have.get(k), v) for k, v in key.items() if have.get(k) != v}
    if diff:
        lines = "\n".join(f"    {k}: cached {a!r} != requested {b!r}"
                          for k, (a, b) in sorted(diff.items()))
        raise ValueError(
            f"chunk cache {path} does not match this build:\n{lines}\n"
            "Delete it or point --cache somewhere else -- it is not rebuilt in place, "
            "because a cache serving the wrong split would make every number in the run "
            "wrong with nothing to say so."
        )
    return ChunkData(**payload["chunks"])
