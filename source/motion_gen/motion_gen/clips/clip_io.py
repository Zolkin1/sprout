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
from collections.abc import Sequence
from dataclasses import dataclass

import h5py
import numpy as np
import torch

from motion_gen.world_lift import _ANCHOR_MASK_LEGACY

# Sentinel in the per-step ``anchor_mask_t`` dataset meaning "no per-step value here -> use the
# clip's per-clip ``anchor_mask``".  Valid masks are 6-bit (0..63), so 255 is unambiguous.
_ANCHOR_MASK_T_UNSET = 255

# ``terrain_id`` of a clip with no terrain tile (flat ground).  Mirrors
# ``robot_rl...terrains.clip_terrain_loader._FLAT_TERRAIN_ID``, which cannot be imported here
# (motion_gen stays IsaacLab-free); the value is baked into every library on disk.
_FLAT_TERRAIN_ID = -1

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

# Per-clip (1-D, shape [N]) tags.  ``terrain_id`` indexes the embedded
# ``terrain_manifest`` root attr (-1 for flat clips, which have no manifest
# entry); ``terrain_type_code`` indexes ``_TERRAIN_TYPE_NAMES``.  ``anchor_mask``
# is the per-clip root-DOF FREE/FIXED bitmask (see :mod:`motion_gen.world_lift`);
# it tells the RL which root DOFs re-anchor to the robot vs stay clip/terrain-fixed.
_PERCLIP_DTYPES: dict[str, str] = {
    "terrain_id": "int32",
    "terrain_type_code": "uint8",
    "anchor_mask": "uint8",
    # Real length (valid frames) before padding.  Clips are stored padded to a common ``T``
    # (repeat-last-frame); ``clip_len`` marks where the actual motion ends so the RL can end
    # the episode at the clip's own last frame instead of playing the held padding tail.
    "clip_len": "int32",
}

# Canonical terrain-type ordering (code = index; APPEND only -- codes are stored
# per-clip so existing values must stay put).  ``flat`` is the original no-terrain
# clip; ``terrain`` is terrain-only (e.g. a staircase); ``transition`` is a flat
# lead-in that approaches and climbs the terrain; ``slope`` is a uniform tilted plane
# (a flat clip rigidly rotated to the slope pitch, see ``motion_matching.slope_rotate``);
# ``box`` is flat ground carrying a standalone overlay volume the clip jumps onto /
# off / over (the manifest entry carries an ``overlays`` list; see the terrain blocks'
# ``OverlayVolumeCfg``).
_TERRAIN_TYPE_NAMES: list[str] = ["flat", "terrain", "transition", "slope", "box"]
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
        self._T = T
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
        # Per-clip root-DOF anchor mask (1-D). Default legacy (x,y,z,yaw FREE) so a
        # writer that sets nothing reproduces today's re-anchoring behavior.
        self._f.create_dataset("anchor_mask", shape=(num_clips,), dtype="uint8")
        self._f["anchor_mask"][...] = _ANCHOR_MASK_LEGACY
        # Per-clip real length (valid frames before padding); default = full T so a writer that sets
        # nothing => the whole clip is real (no early clip-end, i.e. today's behavior).
        self._f.create_dataset("clip_len", shape=(num_clips,), dtype="int32")
        self._f["clip_len"][...] = T
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
        # Per-clip anchor mask (optional; leave the legacy default if the batch omits it).
        if "anchor_mask" in batch:
            am = batch["anchor_mask"].detach().to("cpu").numpy()
            self._f["anchor_mask"][start_idx:end] = am.astype("uint8", copy=False)
        # Per-clip real length (optional; leave the full-T default if the batch omits it), clamped to T.
        if "clip_len" in batch:
            cl = batch["clip_len"].detach().to("cpu").numpy()
            self._f["clip_len"][start_idx:end] = np.clip(cl, 1, self._T).astype("int32", copy=False)
        # Optional per-STEP anchor mask [N, T] -- box clips vary anchoring by segment (jump-on
        # world-pinned, approach / on-top walk / recovery ego), so a single per-clip mask can't
        # express them.  Lazily created; absent -> the per-clip ``anchor_mask`` applies to every
        # step (see :meth:`ClipLibraryData.anchor_mask_per_step`).
        if "anchor_mask_t" in batch:
            amt = batch["anchor_mask_t"].detach().to("cpu").numpy()
            if "anchor_mask_t" not in self._f:
                ds = self._f.create_dataset(
                    "anchor_mask_t", shape=(self.num_clips,) + amt.shape[1:], dtype="uint8",
                    chunks=(1,) + amt.shape[1:], compression=self._compression)
                # Sentinel 255 = "unset" so clips NOT carrying a per-step mask fall back to their
                # per-clip ``anchor_mask`` (0 is a VALID mask -- fully pinned -- so we can't 0-fill).
                ds[...] = _ANCHOR_MASK_T_UNSET
            self._f["anchor_mask_t"][start_idx:end] = amt.astype("uint8", copy=False)
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
    anchor_mask: torch.Tensor    # [N] uint8 root-DOF FREE/FIXED bitmask; empty (=> legacy) if absent
    clip_len: torch.Tensor       # [N] int32 real length before padding; empty (=> full T) if absent
    anchor_mask_t: torch.Tensor  # [N, T] uint8 per-STEP anchor mask (box clips); empty if absent
    meta: dict
    attrs: dict

    @property
    def num_clips(self) -> int:
        return self.y_des.shape[0]

    @property
    def num_steps(self) -> int:
        return self.y_des.shape[1]

    def anchor_mask_or_legacy(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Per-clip anchor bitmask ``[N]`` uint8; a full legacy-filled tensor if absent.

        A pre-mask library carries no ``anchor_mask`` dataset (loaded as an empty
        tensor); such clips re-anchor exactly as before, so they map to
        ``_ANCHOR_MASK_LEGACY``.  This is the single place that interpretation lives.
        """
        dev = device if device is not None else self.anchor_mask.device
        if self.anchor_mask.numel() == 0:
            return torch.full((self.num_clips,), _ANCHOR_MASK_LEGACY, dtype=torch.uint8, device=dev)
        return self.anchor_mask.to(dev)

    def clip_len_or_full(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Per-clip real length ``[N]`` (long); a full ``num_steps`` tensor if the dataset is absent.

        A library written before ``clip_len`` existed loads it empty; those clips are treated as
        full-length (no early clip-end), i.e. exactly the previous behavior.  Single interpretation
        point so consumers always get a valid per-clip length.
        """
        dev = device if device is not None else self.clip_len.device
        if self.clip_len.numel() == 0:
            return torch.full((self.num_clips,), self.num_steps, dtype=torch.long, device=dev)
        return self.clip_len.to(dev).long()

    def anchor_mask_per_step(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Per-clip-per-step anchor mask ``[N, T]`` uint8.

        Uses the per-step ``anchor_mask_t`` dataset if present (box clips that vary
        anchoring by segment); otherwise broadcasts the per-clip mask
        (:meth:`anchor_mask_or_legacy`) across all ``T`` steps.  Even when the dataset is
        present, any ``255`` sentinel (clips that carried no per-step mask) falls back to
        that clip's per-clip mask -- so a MIXED library (some box, some flat) is correct.
        Single interpretation point so the RL always sees a per-step mask.
        """
        dev = device if device is not None else self.anchor_mask.device
        per_clip = self.anchor_mask_or_legacy(dev)                     # [N]
        if self.anchor_mask_t.numel() == 0:
            return per_clip[:, None].expand(self.num_clips, self.num_steps).contiguous()
        amt = self.anchor_mask_t.to(dev)                               # [N, T]
        unset = amt == _ANCHOR_MASK_T_UNSET
        return torch.where(unset, per_clip[:, None].expand_as(amt), amt).contiguous()


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
        # Optional per-step anchor mask [N, T] (empty for libraries without it).
        tensors["anchor_mask_t"] = (torch.from_numpy(f["anchor_mask_t"][...]).to(device)
                                    if "anchor_mask_t" in f else torch.empty(0, device=device))
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


def write_clip_subset(src: str, dst: str, clip_indices: Sequence[int],
                      compression: str | None = None) -> list[dict]:
    """Copy a clip library keeping only ``clip_indices`` (and only their manifest tiles).

    The point of a subset library is REPRODUCIBILITY: with a handful of clips in the file
    the terrain importer builds exactly their tiles, the terrain reset has nothing else to
    sample, and one env per tile therefore replays one known clip every run. That is what
    ``scripts/rsl_rl/compare_sim2sim.py`` needs (one clip) and what
    ``scripts/rsl_rl/fm_debug.py`` needs (one per terrain family).

    Every dataset is sliced by the same index list, so per-clip order in the output follows
    ``clip_indices``. Root attributes are copied verbatim except three that would otherwise
    describe the SOURCE library:

    * ``num_clips`` / ``num_clips_written`` -- rewritten to the kept count;
    * ``terrain_manifest`` -- filtered to the tiles the kept clips actually stand on;
    * ``generator_config.counts.flat`` -- rewritten to the number of kept FLAT clips. The
      importer sizes its flat-tile count from that field in preference to the clip count
      (``clip_terrain_loader.default_num_flat_tiles``), so a copied ``flat: 60`` would build
      60 flat tiles for a 3-clip library and the terrain reset then aborts with "terrain
      id(s) [-1] have an env tile but NO clips".

    Args:
        src: Source library ``.h5``.
        dst: Output ``.h5`` (overwritten).
        clip_indices: Clip indices to keep, in output order. Must be non-empty and unique.
        compression: h5py filter for the copied datasets. ``None`` (the default) keeps the
            historical behaviour of writing them uncompressed; ``"lzf"`` matches what the
            generators write and is worth ~1.8x on these arrays, which matters once a subset
            is shipped as part of an artifact rather than used as a scratch file.

    Returns:
        The filtered terrain manifest (one entry per kept non-flat tile).

    Raises:
        ValueError: If ``clip_indices`` is empty or contains duplicates.
    """
    idx = [int(i) for i in clip_indices]
    if not idx:
        raise ValueError("write_clip_subset needs at least one clip index.")
    if len(set(idx)) != len(idx):
        raise ValueError(f"write_clip_subset got duplicate clip indices: {idx}")

    with h5py.File(str(src), "r") as fi, h5py.File(str(dst), "w") as fo:
        tid = np.asarray(fi["terrain_id"][:])[idx]
        for key in fi.keys():
            d = fi[key]
            # h5py fancy indexing needs the selection sorted+unique, so gather per clip and
            # restack in the requested order. The libraries here are per-clip-chunked, so
            # this is a handful of reads, not a full-file scan.
            arr = np.stack([np.asarray(d[i]) for i in idx])
            fo.create_dataset(key, data=arr, dtype=d.dtype, compression=compression,
                              chunks=((1,) + arr.shape[1:]) if compression else None)
        for key, value in fi.attrs.items():
            fo.attrs[key] = value
        keep_tiles = {int(t) for t in tid if int(t) != _FLAT_TERRAIN_ID}
        manifest = [m for m in json.loads(fi.attrs["terrain_manifest"])
                    if int(m.get("terrain_id", -2)) in keep_tiles]
        fo.attrs["terrain_manifest"] = json.dumps(manifest)
        fo.attrs["num_clips"] = len(idx)
        fo.attrs["num_clips_written"] = len(idx)

        raw = fi.attrs.get("generator_config")
        gcfg = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(gcfg, dict):
            gcfg = {}
        gcfg.setdefault("counts", {})
        gcfg["counts"]["flat"] = int(sum(1 for t in tid if int(t) == _FLAT_TERRAIN_ID))
        gcfg["subset_of"] = {"path": str(src), "clip_indices": idx}
        fo.attrs["generator_config"] = json.dumps(gcfg)
    return manifest


def set_flat_tile_count(path: str, tiles: int) -> int:
    """Set a library's flat-TILE count, overriding what :func:`write_clip_subset` left.

    ``generator_config.counts.flat`` is a TILE count -- the importer's
    ``default_num_flat_tiles`` reads it as one, and its docstring is explicit that flat
    tiles are not one-per-clip: every flat tile hosts many flat clips, because the reset
    samples any flat clip onto any flat tile (they all share ``terrain_id == -1``).

    :func:`write_clip_subset` cannot know the intended tile count, so it writes the kept
    flat CLIP count and leaves the caller to correct it.  Getting that wrong is quiet and
    expensive in both directions: too few tiles and the terrain reset aborts with
    "terrain id(s) [-1] have an env tile but NO clips"; too many and the importer builds
    one large flat tile per clip, so a 3000-clip flat library asks for a 3000-tile mesh.

    Args:
        path: Library ``.h5`` to edit in place.
        tiles: Flat tile count to record.  Must be >= 1.

    Returns:
        The value written.

    Raises:
        ValueError: If ``tiles`` is less than 1.
    """
    if int(tiles) < 1:
        raise ValueError(f"flat tile count must be >= 1, got {tiles}")
    with h5py.File(str(path), "r+") as f:
        raw = f.attrs.get("generator_config")
        gcfg = json.loads(raw) if isinstance(raw, str) else (raw or {})
        if not isinstance(gcfg, dict):
            gcfg = {}
        gcfg.setdefault("counts", {})
        gcfg["counts"]["flat"] = int(tiles)
        f.attrs["generator_config"] = json.dumps(gcfg)
    return int(tiles)


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
            "generator (scripts/rsl_rl/generate_mb_library.py)."
        )
    tid = lib.terrain_id.to(torch.long)
    return {int(t): (tid == int(t)).nonzero(as_tuple=False).flatten() for t in torch.unique(tid)}
