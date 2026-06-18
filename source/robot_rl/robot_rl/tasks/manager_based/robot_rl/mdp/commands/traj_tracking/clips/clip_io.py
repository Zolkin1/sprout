# Copyright 2026 Zachary Olkin. All rights reserved.

"""Read/write dense motion-clip libraries as HDF5.

A clip library is a single ``.h5`` file holding, for ``N`` equal-length
clips of ``T`` control steps, the dense per-step arrays produced by
:class:`OfflineClipGenerator` (anchor-relative reference outputs + the
world anchor schedule + contact/skill labels).  Layout metadata (dt,
output names, contact frames, skill list, fingerprints, generator config)
lives in JSON-encoded root attributes.

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
    "anchor_pos": "float32",
    "anchor_quat": "float32",
    "contact": "uint8",
    "skill_id": "uint8",
    "stride_idx": "int32",
    "target_vel": "float32",
}


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
            "anchor_pos": (num_clips, T, 3),
            "anchor_quat": (num_clips, T, 4),
            "contact": (num_clips, T, C),
            "skill_id": (num_clips, T),
            "stride_idx": (num_clips, T),
            "target_vel": (num_clips, T, 3),
        }
        self._f = h5py.File(self.path, "w")
        for key, shape in shapes.items():
            # Chunk per-clip so reads of a single clip touch one chunk.
            chunk = (1,) + shape[1:]
            self._f.create_dataset(
                key, shape=shape, dtype=_DENSE_DTYPES[key],
                chunks=chunk, compression=compression,
            )
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

    y_des: torch.Tensor          # [N, T, P]
    dy_des: torch.Tensor         # [N, T, V]
    anchor_pos: torch.Tensor     # [N, T, 3]
    anchor_quat: torch.Tensor    # [N, T, 4]
    contact: torch.Tensor        # [N, T, C] uint8
    skill_id: torch.Tensor       # [N, T] uint8
    stride_idx: torch.Tensor     # [N, T] int32
    target_vel: torch.Tensor     # [N, T, 3] ramped (vx, vy, wz) command; empty if absent
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
