# Copyright 2026 Zachary Olkin. All rights reserved.

"""Export a debug clip library to the binary pack the C++ deploy runtime plays.

``transfer/deploy/clip_reference.py`` reads the ``.h5`` with h5py. The C++ runtime cannot:
the obelisk image ships no HDF5, and the library's datasets are ``lzf``-compressed
(``clip_io.py``), which the plain HDF5 C library cannot decode without h5py's filter plugin.
Adding either to the container would put a new third-party dependency on the control path to
read what is, in the end, three arrays and two name lists.

So the clip crosses the language boundary the way everything else already does -- as the
``manifest.json`` + raw-little-endian-float64-blob pack that :mod:`deploy.fixture_io` writes
and ``g1_deploy::BlobPack`` reads. One writer, one reader, one format.

**The channel NAMES travel, not resolved indices.** ``ClipReference`` looks its channels up by
name (``"{base}:pos_x"``, ``"joint:{n}"``, ``"{body}@root:ori_w"``), and the C++ port does the
same against the tracker's own joint and frame names -- so a pack built for a different robot
fails loudly instead of reading the right columns of the wrong motion. The two libraries on
disk genuinely differ in width and in ``ref_frame``, which is exactly the mistake positional
indices would not survive.

One CASE per pack entry, named as the library's JSON sidecar names it, so the C++ selects
``"walk_backward"`` rather than an index that moves whenever the library is rebuilt.

Usage::

    python transfer/deploy/tools/dump_clip_pack.py \\
        transfer/obelisk/g1_control/resource/debug_clips/g1_debug_v1.h5

    # ... writes <lib>_pack/ beside it. --out picks another directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deploy.fixture_io import FixtureWriter  # noqa: E402

#: What the C++ side asks :cpp:func:`BlobPack::ForFunction` for.
CLIP_FN = "clip_reference"

#: Datasets a case carries. ``target_vel`` is optional in the format and mandatory here: a
#: clip served with no command would drive the tracker's ``command_window`` term to zeros,
#: which is a different motion, not a missing field.
_DATASETS = ("y_des", "dy_des", "target_vel")


def _cases(h5_path: str) -> dict[str, dict]:
    """The library's case map, from its JSON sidecar.

    Args:
        h5_path: The clip library ``.h5``.

    Returns:
        Case name -> the sidecar's entry (which carries at least ``clip_index``).

    Raises:
        SystemExit: If the sidecar is missing -- the library was not built by
            ``make_debug_clips.py`` and therefore has no case names to export under.
    """
    sidecar = h5_path.rsplit(".h5", 1)[0] + ".json"
    if not os.path.exists(sidecar):
        sys.exit(
            f"[ERROR] no case sidecar at {sidecar}. A pack is addressed BY CASE NAME (the C++ "
            "controller names 'stand', not an index that moves whenever the library is "
            "rebuilt), so a library without one cannot be exported. Build it with "
            "scripts/rsl_rl/make_debug_clips.py."
        )
    with open(sidecar) as fh:
        return json.load(fh)["cases"]


def build_pack(h5_path: str, out_dir: str) -> dict:
    """Write the pack and return a summary.

    Args:
        h5_path: Source clip library ``.h5``.
        out_dir: Destination directory (created; existing blobs are overwritten).

    Returns:
        ``{case: {"clip_index": int, "steps": int, "real_steps": int}}``.
    """
    cases = _cases(h5_path)
    writer = FixtureWriter(out_dir, generator="transfer/deploy/tools/dump_clip_pack.py")
    summary: dict[str, dict] = {}

    with h5py.File(h5_path, "r") as f:
        pos_names = json.loads(f.attrs["pos_output_names"])
        vel_names = json.loads(f.attrs["vel_output_names"])
        dt = float(json.loads(f.attrs["dt"]))
        num_steps = int(json.loads(f.attrs["num_steps"]))
        ref_frame = json.loads(f.attrs["ref_frame"]) if "ref_frame" in f.attrs else "pelvis_link"
        for name, entry in sorted(cases.items(), key=lambda kv: kv[1]["clip_index"]):
            i = int(entry["clip_index"])
            arrays = {}
            for key in _DATASETS:
                if key not in f:
                    sys.exit(f"[ERROR] {h5_path} has no '{key}' dataset; it is not a clip library.")
                arrays[key] = np.asarray(f[key][i], dtype=np.float64)
            real = (int(np.asarray(f["clip_len"][i])) if "clip_len" in f
                    else int(arrays["y_des"].shape[0]))
            real = max(1, min(real, arrays["y_des"].shape[0]))
            attrs = {
                "pos_output_names": list(pos_names),
                "vel_output_names": list(vel_names),
                "ref_frame": ref_frame,
                "dt": dt,
                "num_steps": int(arrays["y_des"].shape[0]),
                "real_steps": real,
                "clip_index": i,
                "source_clip": int(entry.get("source_clip", -1)),
                "note": str(entry.get("note", "")),
            }
            # The motion span, when the library carries one: the C++ zeroes the command over
            # the holds, and a metric run scores the fall against the MOTION rather than
            # against playback saturating on its last frame.
            for span in ("lead_in_steps", "motion_start_step", "motion_end_step", "tail_steps"):
                if span in entry:
                    attrs[span] = int(entry[span])
            # atol/compare are the format's, not this pack's: nothing here is compared against
            # an oracle. 0.0 is the honest value -- a nonzero one would imply a tolerance
            # somebody had chosen.
            writer.add(case=name, fn=CLIP_FN, inputs=arrays, outputs={},
                       atol=0.0, compare="none", attrs=attrs)
            summary[name] = {"clip_index": i, "steps": attrs["num_steps"], "real_steps": real}

    writer.write_manifest(seed=0, extra={
        "clip_source": {
            "library": os.path.basename(h5_path),
            "num_steps": num_steps,
            "dt": dt,
            # Whatever the library says about where IT came from, carried through rather than
            # restated, so a pack can be traced back to a generation run.
            "fingerprint": _fingerprint(h5_path),
        },
    })
    return summary


def _fingerprint(h5_path: str) -> dict:
    """The library's own provenance attrs, or ``{}`` if it carries none."""
    out = {}
    with h5py.File(h5_path, "r") as f:
        for key in ("clip_fingerprint", "source_library_fingerprint", "debug_clip_source"):
            if key in f.attrs:
                raw = f.attrs[key]
                try:
                    out[key] = json.loads(raw)
                except (TypeError, ValueError):
                    out[key] = str(raw)
    return out


def main() -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip", help="Clip library .h5 (needs its make_debug_clips.py JSON sidecar).")
    p.add_argument("--out", default=None,
                   help="Destination directory (default: <lib>_pack/ beside the .h5).")
    args = p.parse_args()

    out = args.out or (args.clip.rsplit(".h5", 1)[0] + "_pack")
    summary = build_pack(args.clip, out)
    total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"[OK] wrote {len(summary)} case(s) to {out} ({total / 1e6:.1f} MB)")
    for name, s in summary.items():
        print(f"     {name:<18} clip {s['clip_index']}  {s['real_steps']}/{s['steps']} real steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
