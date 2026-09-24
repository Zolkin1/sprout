#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Browse and export the DEPTH IMAGES inside a flow-matching recording.

A windowed recording (``record_data.py --num_windows``) stores what the generator was
conditioned on, and that includes every camera's normalised depth image -- 200k windows x 2
cameras in the one the shipped generator trained on.  Nothing could look at them:
``transfer/sim/verify_depth_cameras.py --save_dir`` renders a LIVE MuJoCo scene, and the viser
viz package draws height-scan hits rather than images.  So picking a frame for a figure meant
re-running a simulator to approximate one that was already on disk.

Two stages, because 200k images is far too many to look through and one is too few to choose
between:

* a CONTACT SHEET -- a labelled grid of many windows at once, optionally restricted to one
  terrain family, stretched per image so structure is what you scan for;
* an EXPORT -- the windows you picked, at paper size, on a SHARED scale with a colour bar in
  metres, so two exported images are comparable.

Every camera number is read from the recording's own ``depth_camera_spec`` and interpreted
through ``transfer/deploy/depth_obs.py`` -- the same parser, denormalisation and validity test
the deploy runtime uses.  Nothing here restates a camera constant, so an image this tool draws
is the image the generator was given.

**Validity is by RANGE, not by sentinel equality** (:func:`deploy.depth_obs.invalid_depth_mask`).
Recordings are made with the training noise model, whose 3x3 blur leaves a geometric miss NEAR
the sentinel rather than on it; an equality test would call those pixels valid and place a fake
surface ~0.1 m from the lens.  They are drawn as "no reading" instead.

Simulator-free: h5py, numpy and matplotlib only, so it runs under ``env_transfer`` as well as
``env_isaaclab``.

Examples::

    # what is in the file -- families, counts, cameras, grid
    python scripts/rsl_rl/browse_depth_recording.py --recording <rec>.h5 --list

    # 48 stairs-up windows as one sheet to scan
    python scripts/rsl_rl/browse_depth_recording.py --recording <rec>.h5 \\
        --family family_stairs_up --count 48 --sheet /tmp/stairs_up.png

    # the ones you picked off the sheet, at paper size
    python scripts/rsl_rl/browse_depth_recording.py --recording <rec>.h5 \\
        --windows 51043,51120 --out_dir figs/depth
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

# Same hop as scripts/rsl_rl/export_sensor_configs.py, which also reaches the deploy contract
# helpers from here.  Derived from __file__ rather than counted in ``..`` at the call site.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "transfer"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deploy.depth_obs import (  # noqa: E402
    camera_key,
    denormalize_depth_np,
    invalid_depth_mask,
    parse_depth_spec,
)

#: Colour drawn where a pixel carries no range reading.  Named explicitly rather than left
#: transparent, so the figure reads the same on any background: "no reading" is a different
#: KIND of value from a depth, and a reader must not take it for a near or a far surface.
#: White sits outside viridis/magma entirely, so it cannot be mistaken for either end.
#:
#: Expect a lot of it, scattered: 6.4% (near) / 8.9% (far) of pixels carry no reading in the
#: shipped recordings. That speckle is the sensor model, not a bug in this tool.
_INVALID_COLOR: str = "white"


def open_recording(path: str):
    """Open a windowed recording and refuse anything this tool cannot read.

    Args:
        path: Path to the ``.h5`` written by ``record_data.py --num_windows``.

    Returns:
        The open :class:`h5py.File`; the caller closes it.

    Raises:
        SystemExit: If the file is not a windowed recording, or carries no depth.
    """
    f = h5py.File(path, "r")
    fmt = json.loads(f.attrs.get("format", '""'))
    if fmt != "windowed_h5":
        f.close()
        raise SystemExit(
            f"{path}: format={fmt!r}, not 'windowed_h5'. This reads the windowed recordings "
            "record_data.py --num_windows writes; the legacy per-step .npz path has no "
            "depth block to browse.")
    if "depth" not in f:
        f.close()
        raise SystemExit(
            f"{path} has no 'depth' dataset -- it was recorded from a height-scan task "
            f"(datasets present: {sorted(f.keys())}).")
    return f


def recording_spec(f, path: str) -> list[dict]:
    """The recording's own camera contract, parsed by the deploy parser.

    Args:
        f: Open recording.
        path: Its path, for error messages.

    Returns:
        One validated contract dict per camera, in observation order.

    Raises:
        SystemExit: If the recording carries no contract to read.
    """
    raw = f.attrs.get("depth_camera_spec")
    if raw is None:
        raise SystemExit(
            f"{path} stores depth but no 'depth_camera_spec'. Without the contract the "
            "normalisation, the grid and the validity threshold are all unknown, and a "
            "guessed one would render a plausible wrong picture.")
    return parse_depth_spec(json.loads(raw), source=os.path.basename(path))


def family_table(f) -> tuple[list[str], np.ndarray]:
    """Family names and the per-window family code.

    Args:
        f: Open recording.

    Returns:
        ``(names, codes)`` where ``codes[i]`` indexes ``names``.
    """
    names = json.loads(f.attrs["family_names"]) if "family_names" in f.attrs else []
    codes = f["clip_family"][:] if "clip_family" in f else np.zeros(0, dtype=np.uint8)
    return list(names), np.asarray(codes)


def describe(f, spec: list[dict], path: str) -> None:
    """Print what the recording holds: cameras, grid, families, distractors.

    Args:
        f: Open recording.
        spec: The parsed camera contract.
        path: The recording's path.
    """
    n = int(f["depth"].shape[0])
    print(f"{path}\n  {n} windows, recorded {json.loads(f.attrs['recorded_at'])}")
    print(f"  run: {json.loads(f.attrs['run_name'])}")
    print(f"  clips: {json.loads(f.attrs['clip_library_path'])}")
    print("  cameras:")
    for cam in spec:
        print(f"    {camera_key(cam):12s} {cam['width']}x{cam['height']} "
              f"mount={cam['mount_body']} range {cam['min_dist']:.2f}..{cam['max_dist']:.2f} m")
    names, codes = family_table(f)
    if names:
        print("  families (use --family):")
        for i, name in enumerate(names):
            count = int((codes == i).sum())
            if count:
                print(f"    {name:26s} {count:7d} windows")
    if "is_distractor" in f:
        print(f"    {'(of which distractors)':26s} {int(f['is_distractor'][:].sum()):7d}")


def select_windows(f, families: list[str] | None, exclude_distractors: bool,
                   count: int, seed: int) -> np.ndarray:
    """Choose which windows to draw.

    Sampled WITHOUT replacement across the whole matching set rather than taken as the first
    ``count``: consecutive windows come from the same episode and look nearly identical, so a
    head slice would show one moment many times over.

    Args:
        f: Open recording.
        families: Family names to keep, or ``None`` for all.
        exclude_distractors: Drop windows flagged as distractors.
        count: How many to sample.
        seed: RNG seed, so a sheet is reproducible and an index means the same thing twice.

    Returns:
        Sorted window indices.

    Raises:
        SystemExit: On an unknown family name, or when nothing matches.
    """
    n = int(f["depth"].shape[0])
    keep = np.ones(n, dtype=bool)
    names, codes = family_table(f)
    if families:
        unknown = [name for name in families if name not in names]
        if unknown:
            raise SystemExit(
                f"unknown --family {unknown}; this recording has {names}")
        wanted = {names.index(name) for name in families}
        keep &= np.isin(codes, list(wanted))
    if exclude_distractors and "is_distractor" in f:
        keep &= f["is_distractor"][:] == 0
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise SystemExit("no windows match that filter")
    if idx.size > count:
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=count, replace=False))
    return idx


def read_images(f, spec: list[dict], windows: np.ndarray) -> np.ndarray:
    """Read the selected windows' depth block and fold it to images.

    Args:
        f: Open recording.
        spec: The parsed camera contract, which supplies each grid.
        windows: Sorted window indices.

    Returns:
        ``[len(windows), num_cameras]`` object array of ``(height, width)`` normalised images.

    Raises:
        SystemExit: If a camera's row width disagrees with its declared grid.
    """
    block = f["depth"][windows, :, :]                      # [W, C, D]
    out = np.empty((len(windows), len(spec)), dtype=object)
    for c, cam in enumerate(spec):
        height, width = int(cam["height"]), int(cam["width"])
        if block.shape[2] != height * width:
            raise SystemExit(
                f"camera {camera_key(cam)}: contract says {width}x{height} = "
                f"{width * height} values, the recording stores {block.shape[2]}.")
        for w in range(len(windows)):
            # Row-major (height, width) -- the fold transfer/sim/verify_depth_cameras.py and
            # the live obs term both use. Transposing it keeps the shape and ruins the image.
            out[w, c] = block[w, c].reshape(height, width)
    return out


def to_metres(image: np.ndarray, cam: dict) -> np.ma.MaskedArray:
    """Denormalise one image to metres, masking the pixels with no reading.

    Args:
        image: One normalised ``(height, width)`` image.
        cam: That camera's contract entry.

    Returns:
        Masked array of metres; masked where the range test says "no reading".
    """
    metres = denormalize_depth_np(image, float(cam["offset"]), float(cam["scale"]))
    return np.ma.masked_array(metres, mask=invalid_depth_mask(cam, image))


def _cmap(name: str):
    """A colour map with "no reading" painted :data:`_INVALID_COLOR`.

    Args:
        name: Any matplotlib colour map name.

    Returns:
        The configured colour map.
    """
    import matplotlib as mpl

    cmap = mpl.colormaps[name].copy()
    cmap.set_bad(_INVALID_COLOR)
    return cmap


def write_sheet(images, windows, spec, cams, f, path: str, cmap_name: str) -> None:
    """Write one labelled contact sheet of many windows, stretched per image.

    Per-image stretch on purpose: a camera looking at ground a metre away uses a few percent
    of its 8 m range, so a shared scale renders most tiles as one flat colour and hides the
    very structure you are scanning for.  The EXPORT is the one that shares a scale.

    Args:
        images: ``[W, C]`` object array of normalised images.
        windows: The window indices, for the tile labels.
        spec: The full camera contract.
        cams: Indices into ``spec`` of the cameras to draw.
        f: Open recording, for the family labels.
        path: Output PNG.
        cmap_name: Colour map.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, codes = family_table(f)
    tiles = [(w, c) for w in range(len(windows)) for c in cams]
    ncol = int(np.ceil(np.sqrt(len(tiles) / 0.7)))
    nrow = int(np.ceil(len(tiles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.7 * ncol, 1.9 * nrow), squeeze=False)
    cmap = _cmap(cmap_name)
    for ax, (w, c) in zip(axes.ravel(), tiles):
        cam = spec[c]
        ax.imshow(to_metres(images[w, c], cam), cmap=cmap, interpolation="nearest")
        idx = int(windows[w])
        code = int(codes[idx]) if idx < codes.size else -1
        fam = names[code].replace("family_", "") if 0 <= code < len(names) else ""
        ax.set_title(f"{idx}  {camera_key(cam)}\n{fam}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(tiles):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}  ({len(tiles)} tiles, per-image stretch)")
    print("  pick the window numbers you want, then re-run with --windows a,b,c --out_dir ...")


def write_exports(images, windows, spec, cams, out_dir: str, cmap_name: str,
                  vmin: float | None, vmax: float | None) -> None:
    """Write one paper-sized figure per window, on a scale shared across the export set.

    The shared scale is what makes two exported images comparable, and the colour bar states
    it in metres so a reader is not left guessing at a stretch.  It is taken from the data
    actually exported (the valid pixels' 1st and 99th percentile per camera) rather than the
    camera's full declared range, which would waste most of the bar on distances nothing in
    the frame reaches.

    Args:
        images: ``[W, C]`` object array of normalised images.
        windows: The window indices.
        spec: The full camera contract.
        cams: Indices into ``spec`` of the cameras to draw.
        out_dir: Directory for the PNGs.
        cmap_name: Colour map.
        vmin: Shared scale floor (m), or ``None`` to derive it.
        vmax: Shared scale ceiling (m), or ``None`` to derive it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_style import apply_paper_style, figsize, save_figure

    apply_paper_style()
    # The repo's own single-column size rather than a size picked here: apply_paper_style sets
    # 20-pt labels for journal width, and at a smaller figure the colour-bar label swamps a
    # 30x26 image.
    fig_size = figsize(cols=1)
    os.makedirs(out_dir, exist_ok=True)
    cmap = _cmap(cmap_name)

    limits = {}
    for c in cams:
        pool = np.ma.concatenate([to_metres(images[w, c], spec[c]).ravel()
                                  for w in range(len(windows))])
        valid = pool.compressed()
        lo = vmin if vmin is not None else float(np.percentile(valid, 1))
        hi = vmax if vmax is not None else float(np.percentile(valid, 99))
        limits[c] = (lo, hi)
        print(f"  {camera_key(spec[c]):12s} shared scale {lo:.2f}..{hi:.2f} m "
              f"({100.0 * pool.mask.mean():.1f}% of pixels carry no reading)")

    for w in range(len(windows)):
        for c in cams:
            cam = spec[c]
            lo, hi = limits[c]
            fig, ax = plt.subplots(figsize=fig_size)
            im = ax.imshow(to_metres(images[w, c], cam), cmap=cmap, vmin=lo, vmax=hi,
                           interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            bar.set_label("depth (m)")
            stem = os.path.join(out_dir, f"depth_{int(windows[w]):06d}_{camera_key(cam)}")
            save_figure(fig, stem, svg=False)
            plt.close(fig)
            print(f"  wrote {stem}.png")


def main() -> None:
    """Parse arguments and run the requested stage."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", required=True, help="Windowed recording .h5.")
    ap.add_argument("--list", action="store_true",
                    help="Print the cameras, families and counts, then exit.")
    ap.add_argument("--camera", default="both", choices=["near", "far", "both"],
                    help="Which camera(s) to draw. Matched against the contract's own names.")
    ap.add_argument("--family", default=None,
                    help="Comma-separated family names to keep (see --list).")
    ap.add_argument("--exclude_distractors", action="store_true",
                    help="Drop windows flagged is_distractor.")
    ap.add_argument("--count", type=int, default=48, help="Windows on the contact sheet.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed; a sheet is reproducible.")
    ap.add_argument("--sheet", default=None, help="Write the contact sheet here.")
    ap.add_argument("--windows", default=None,
                    help="Comma-separated window indices to EXPORT (skips sampling).")
    ap.add_argument("--out_dir", default=None, help="Directory for the exported figures.")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colour map.")
    ap.add_argument("--vmin", type=float, default=None, help="Shared scale floor, metres.")
    ap.add_argument("--vmax", type=float, default=None, help="Shared scale ceiling, metres.")
    args = ap.parse_args()

    f = open_recording(args.recording)
    try:
        spec = recording_spec(f, args.recording)
        if args.list:
            describe(f, spec, args.recording)
            return

        cams = [i for i, cam in enumerate(spec)
                if args.camera == "both" or args.camera in camera_key(cam)]
        if not cams:
            raise SystemExit(
                f"--camera {args.camera} matches none of "
                f"{[camera_key(c) for c in spec]}")

        if args.windows:
            windows = np.array(sorted({int(v) for v in args.windows.split(",")}), dtype=np.int64)
            n = int(f["depth"].shape[0])
            if windows.min() < 0 or windows.max() >= n:
                raise SystemExit(f"--windows out of range; this recording has {n}")
        else:
            windows = select_windows(
                f, args.family.split(",") if args.family else None,
                args.exclude_distractors, args.count, args.seed)

        images = read_images(f, spec, windows)

        if args.out_dir:
            write_exports(images, windows, spec, cams, args.out_dir,
                          args.cmap, args.vmin, args.vmax)
        else:
            sheet = args.sheet or "depth_sheet.png"
            write_sheet(images, windows, spec, cams, f, sheet, args.cmap)
    finally:
        f.close()


if __name__ == "__main__":
    main()
