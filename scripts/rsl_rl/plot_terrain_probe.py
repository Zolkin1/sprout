#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Render ``fm_terrain_probe.py``'s result: what each camera SAW, and what was PLANNED.

Two panels, and the left one is the more important of the two. The probe's whole risk is
answering "does a wall read as terrain?" from conditions the cameras never registered, so the
stimulus has to be shown next to the response rather than assumed behind it.

* **left -- what the cameras saw.** Per-camera mean absolute change in the normalised depth
  channel against the empty control. The scale is interpretable: sentinel-to-surface is ~4.7,
  a ray that already hit ground being shortened is ~0.1-1, and 0 means the condition did not
  exist as far as that lens is concerned.
* **right -- what the generator planned**, as the excess of jump+stairs over the same control.
  A bar here is meaningless unless its row on the left is non-zero, so rows the cameras
  missed are drawn hollow.

IsaacLab-free -- reads only the probe's ``.json``.

    python scripts/rsl_rl/plot_terrain_probe.py --probe fm_terrain_probe.json --out probe.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_style import apply_paper_style  # noqa: E402


def main() -> int:
    """Draw the stimulus/response pair and print the same numbers as a table."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe", required=True, help="fm_terrain_probe.json")
    p.add_argument("--out", default=None, help="Figure path (default: beside the json).")
    a = p.parse_args()

    with open(a.probe) as fh:
        d = json.load(fh)
    conds = [c for c in d["conditions"] if c != d["conditions"][0]]
    ctrl = d["conditions"][0]
    cams = list(d["stimulus"][conds[0]]["per_camera"].keys())

    stim = {c: [d["stimulus"][c]["per_camera"][k]["mean_abs_delta"] for k in cams] for c in conds}
    resp = {c: d["rows"][c]["excess_vs_control"] for c in conds}
    # A floor, not an epsilon. 1e-6 passes conditions whose depth moved by a rounding error
    # -- wall_d4.0 and wall_d6.0 both did -- and calling those "seen" is exactly the mistake
    # this column exists to stop. 1e-3 of normalised depth is 2 mm over the whole frame,
    # already generous against a sentinel-to-surface change of ~4.7.
    SEEN_FLOOR = 1e-3
    saw = {c: sum(stim[c]) > SEEN_FLOOR for c in conds}

    print(f"\n=== terrain probe: {os.path.basename(a.probe)} "
          f"({d['repeats']} envs/condition, control '{ctrl}') ===")
    hdr = "".join(f"{k[:10]:>12s}" for k in cams)
    print(f"  {'condition':<22s}{hdr}{'saw it?':>9s}{'plan excess':>13s}")
    for c in conds:
        print(f"  {c:<22s}" + "".join(f"{v:12.4f}" for v in stim[c])
              + f"{('yes' if saw[c] else 'NO'):>9s}{resp[c]:+13.3f}")
    missed = [c for c in conds if not saw[c]]
    if missed:
        print(f"\n  {len(missed)} condition(s) registered on NO camera: {missed}")
        print("  Their plan-excess numbers say nothing about geometry -- the geometry never "
              "reached the model.")

    apply_paper_style()
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 0.42 * len(conds) + 2.2), sharey=True)
    ypos = np.arange(len(conds))[::-1]
    left = np.zeros(len(conds))
    for k in cams:
        v = np.array([d["stimulus"][c]["per_camera"][k]["mean_abs_delta"] for c in conds])
        ax0.barh(ypos, v, left=left, height=0.66, label=k)
        left += v
    ax0.set_yticks(ypos, conds)
    ax0.set_xlabel(r"mean $|\Delta|$ normalised depth vs control")
    ax0.set_title("what the cameras saw")
    ax0.legend(title="camera", loc="lower right")
    ax0.grid(axis="x", alpha=0.3)

    vals = np.array([resp[c] for c in conds])
    ax1.barh(ypos, vals, height=0.66,
             color=["#3b7dd8" if saw[c] else "none" for c in conds],
             edgecolor=["#3b7dd8" if saw[c] else "#b0b0b0" for c in conds],
             hatch=[None if saw[c] else "//" for c in conds])
    ax1.axvline(0.0, color="k", lw=1.0)
    ax1.set_xlabel(f"P(jump or stairs) minus P(control), control = {ctrl}")
    ax1.set_title("what the generator planned\n(hollow = camera never saw it)")
    ax1.grid(axis="x", alpha=0.3)

    out = a.out or os.path.splitext(a.probe)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
