# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read/write dense motion-clip libraries as HDF5.

A clip library is a single ``.h5`` file holding, for ``N`` equal-length
clips of ``T`` control steps, the dense per-step arrays produced by
:class:`OfflineClipGenerator` (clip-world reference outputs + contact/skill
labels).  The per-step stance anchor is baked into the outputs at generation
time, so a clip is a single clip-world frame -- the runtime only applies a
per-env placement.  Layout metadata (dt, output names, contact frames, skill
list, fingerprints, generator config) lives in JSON-encoded root attributes.

The writer is streaming -- datasets are pre-sized to ``N`` clips and each
generated batch is written into its slice -- so generating a large library
never holds more than one batch in memory at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import h5py
import numpy as np
import torch

# Per-step dense datasets, with their on-disk dtype.
_DENSE_DTYPES: dict[str, str] = {
    "y_des": "float32",
    "dy_des": "float32",
    "contact": "uint8",
    "skill_id": "uint8",
    "stride_idx": "int32",
    "target_vel": "float32",
}

# Optional per-step DEBUG datasets (only present when the generator ran with
# ``debug_interp=True``): the skill-transition cross-fade state per step, for the
# ``plot_clip_frames`` util.  Created lazily on the first batch that carries them.
_DEBUG_DTYPES: dict[str, str] = {
    "blend_active": "uint8",    # [N, T] 1 while mid-cross-fade
    "blend_alpha": "float32",   # [N, T] blend weight 0 (pure old) -> 1 (pure new)
    "blend_old": "float32",     # [N, T, P] pre-blend OLD nominal (NaN off-blend)
    "blend_new": "float32",     # [N, T, P] pre-blend NEW nominal (NaN off-blend)
}

# Per-clip (1-D, shape [N]) terrain tags.  ``terrain_id`` indexes the embedded
# ``terrain_manifest`` root attr (-1 for flat clips, which have no manifest
# entry); ``terrain_type_code`` indexes ``_TERRAIN_TYPE_NAMES``.
_PERCLIP_DTYPES: dict[str, str] = {
    "terrain_id": "int32",
    "terrain_type_code": "uint8",
}

# Canonical terrain-type ordering (code = index).  ``flat`` is the original
# no-terrain clip; ``terrain`` is terrain-only (e.g. a staircase); ``transition``
# is a flat lead-in that approaches and climbs the terrain.
_TERRAIN_TYPE_NAMES: list[str] = ["flat", "terrain", "transition"]
_TERRAIN_TYPE_CODE: dict[str, int] = {n: i for i, n in enumerate(_TERRAIN_TYPE_NAMES)}


class ClipLibraryWriter:
    """Streaming HDF5 writer for a dense motion-clip library.

    Args:
        path: Output ``.h5`` path.
        num_clips: Total number of clips the file will hold.
        meta: Layout metadata from :meth:`OfflineClipGenerator.generate_batch`
            (``dt``, ``num_steps``, ``pos_output_names``, ``vel_output_names``,
            ``contact_frames``, ``ref_frame``, ``skill_list``).
        compression: h5py compression filter (``"lzf"`` is fast; ``"gzip"``
            compresses smooth splines better but is slower).  ``None`` off.
    """

    def __init__(self, path: str, num_clips: int, meta: dict, compression: str | None = "lzf") -> None:
        self.path = str(path)
        self.num_clips = int(num_clips)
        self.meta = dict(meta)
        T = int(meta["num_steps"])
        P = len(meta["pos_output_names"])
        V = len(meta["vel_output_names"])
        C = len(meta["contact_frames"])
        shapes = {
            "y_des": (num_clips, T, P),
            "dy_des": (num_clips, T, V),
            "contact": (num_clips, T, C),
            "skill_id": (num_clips, T),
            "stride_idx": (num_clips, T),
            "target_vel": (num_clips, T, 3),
        }
        self._compression = compression
        self._f = h5py.File(self.path, "w")
        for key, shape in shapes.items():
            # Chunk per-clip so reads of a single clip touch one chunk.
            chunk = (1,) + shape[1:]
            self._f.create_dataset(
                key, shape=shape, dtype=_DENSE_DTYPES[key],
                chunks=chunk, compression=compression,
            )
        # Per-clip terrain tags (1-D).  Default flat: id -1, code 0.
        self._f.create_dataset("terrain_id", shape=(num_clips,), dtype="int32")
        self._f["terrain_id"][...] = -1
        self._f.create_dataset("terrain_type_code", shape=(num_clips,), dtype="uint8")
        self._f.attrs["terrain_type_names"] = json.dumps(_TERRAIN_TYPE_NAMES)
        # Store layout metadata as JSON attrs.
        for k, v in self.meta.items():
            self._f.attrs[k] = json.dumps(v)
        self._f.attrs["num_clips"] = num_clips
        self._written = 0

    def write_batch(self, start_idx: int, batch: dict) -> None:
        """Write one generated batch into clips ``[start_idx, start_idx + n)``."""
        n = batch["y_des"].shape[0]
        end = start_idx + n
        if end > self.num_clips:
            raise ValueError(f"write_batch overruns library: {end} > {self.num_clips}")
        for key in _DENSE_DTYPES:
            if key not in batch:
                continue  # optional dataset (e.g. older generator without it)
            arr = batch[key].detach().to("cpu").numpy()
            self._f[key][start_idx:end] = arr.astype(_DENSE_DTYPES[key], copy=False)
        # Per-clip terrain tags: ``terrain_id`` is a per-clip tensor; the batch
        # carries a single ``terrain_type`` string (all clips share a terrain).
        if "terrain_id" in batch:
            tid = batch["terrain_id"].detach().to("cpu").numpy()
            self._f["terrain_id"][start_idx:end] = tid.astype("int32", copy=False)
        ttype = batch.get("terrain_type", "flat")
        if ttype not in _TERRAIN_TYPE_CODE:
            raise ValueError(f"unknown terrain_type '{ttype}'; expected one of {_TERRAIN_TYPE_NAMES}")
        self._f["terrain_type_code"][start_idx:end] = _TERRAIN_TYPE_CODE[ttype]
        # Optional debug datasets (only when the generator logged them); create
        # lazily on the first batch that carries them.
        for key, dt in _DEBUG_DTYPES.items():
            if key not in batch:
                continue
            arr = batch[key].detach().to("cpu").numpy()
            if key not in self._f:
                shape = (self.num_clips,) + arr.shape[1:]   # per-step [.,T] or [.,T,P]
                self._f.create_dataset(
                    key, shape=shape, dtype=dt,
                    chunks=(1,) + arr.shape[1:], compression=self._compression,
                )
            self._f[key][start_idx:end] = arr.astype(dt, copy=False)
        self._written = max(self._written, end)

    def set_attr(self, name: str, value) -> None:
        """Store an extra JSON-encoded root attribute (e.g. fingerprints, config)."""
        self._f.attrs[name] = json.dumps(value)

    def close(self) -> None:
        self._f.attrs["num_clips_written"] = self._written
        self._f.close()

    def __enter__(self) -> "ClipLibraryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class ClipLibraryData:
    """In-memory dense clip library (tensors on the requested device)."""

    y_des: torch.Tensor          # [N, T, P] clip-world position outputs
    dy_des: torch.Tensor         # [N, T, V] clip-world velocity outputs
    contact: torch.Tensor        # [N, T, C] uint8
    skill_id: torch.Tensor       # [N, T] uint8
    stride_idx: torch.Tensor     # [N, T] int32
    target_vel: torch.Tensor     # [N, T, 3] ramped (vx, vy, wz) command; empty if absent
    terrain_id: torch.Tensor     # [N] int32 manifest id (-1 flat); empty if absent
    terrain_type_code: torch.Tensor  # [N] uint8 index into terrain_type_names; empty if absent
    meta: dict
    attrs: dict

    @property
    def num_clips(self) -> int:
        return self.y_des.shape[0]

    @property
    def num_steps(self) -> int:
        return self.y_des.shape[1]


def load_clip_library(
    path: str, device: str | torch.device = "cpu", keys: list[str] | None = None,
) -> ClipLibraryData:
    """Load a dense clip library HDF5 into tensors on ``device``.

    Args:
        path: Library ``.h5`` path.
        device: Torch device for the loaded tensors (keep CPU for large
            libraries; the runtime gathers assigned clips to GPU per reset).
        keys: Optional subset of dense datasets to load (others are empty
            placeholders).  ``None`` loads all.

    Returns:
        A :class:`ClipLibraryData`.
    """
    load = set(keys) if keys is not None else set(_DENSE_DTYPES)
    with h5py.File(str(path), "r") as f:
        tensors: dict[str, torch.Tensor] = {}
        for key in _DENSE_DTYPES:
            # Guard `key in f` so libraries written before a field was added
            # (e.g. target_vel) still load, with an empty placeholder.
            if key in load and key in f:
                tensors[key] = torch.from_numpy(f[key][...]).to(device)
            else:
                tensors[key] = torch.empty(0, device=device)
        # Per-clip terrain tags (always loaded; empty for pre-terrain libraries).
        for key in _PERCLIP_DTYPES:
            if key in f:
                tensors[key] = torch.from_numpy(f[key][...]).to(device)
            else:
                tensors[key] = torch.empty(0, device=device)
        attrs = {}
        meta = {}
        meta_keys = {
            "dt", "clip_length_s", "num_steps", "pos_output_names",
            "vel_output_names", "contact_frames", "ref_frame", "skill_list",
        }
        for k, v in f.attrs.items():
            if isinstance(v, (str, bytes)):
                try:
                    decoded = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    decoded = v
            else:
                decoded = v.item() if isinstance(v, np.generic) else v
            (meta if k in meta_keys else attrs)[k] = decoded
    return ClipLibraryData(meta=meta, attrs=attrs, **tensors)


def group_clips_by_terrain(lib: ClipLibraryData) -> dict[int, torch.Tensor]:
    """Group clip indices by their ``terrain_id`` (``-1`` = flat / no terrain).

    Used to assign each env a clip from the terrain that env stands on: a reset
    (or the standalone verify script) looks up the env's terrain id and samples
    uniformly from that terrain's clip list.

    Args:
        lib: A loaded clip library (must carry per-clip ``terrain_id``).

    Returns:
        ``{terrain_id: clip_indices}`` where each value is a 1-D long tensor of
        clip indices (on ``lib.terrain_id``'s device), for every distinct
        ``terrain_id`` present (including ``-1``).

    Raises:
        ValueError: If the library carries no per-clip ``terrain_id`` (a
            pre-terrain library); such libraries cannot be terrain-assigned.
    """
    if lib.terrain_id.numel() == 0:
        raise ValueError(
            "clip library has no per-clip terrain_id (pre-terrain library); "
            "cannot group clips by terrain. Regenerate with the terrain-aware "
            "generator (scripts/rsl_rl/generate_terrain_clips.py)."
        )
    tid = lib.terrain_id.to(torch.long)
    return {int(t): (tid == int(t)).nonzero(as_tuple=False).flatten() for t in torch.unique(tid)}
