#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Render ``fm_terrain_probe.py``'s result: what each camera SAW, and how far the plan MOVED.

The response is a paired difference, ``plan(condition) - plan(empty)``, not a classification.
Every env in a repeat shares the clip, the step, the ground, the command and the sampling
noise, so the only thing separating a condition from the control is its decor and the
difference is attributable by construction. That removes the classifier -- and with it the
classifier's own error -- from a measurement it was only ever a proxy for.

* **left -- what the cameras saw.** Per-camera mean absolute change in the normalised depth
  channel. 0 means the condition did not exist as far as that lens is concerned, and its bar
  on the right is then meaningless (drawn hollow).
* **right -- how far the plan moved**, as the predicted base path's peak displacement in
  metres. Given that nothing in this stack avoids obstacles, a WALL ideally moves this very
  little while a BOX or STAIRS should move it a lot; the interesting quantity is therefore
  the wall bars read against the two real-terrain ones.

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
    resp = {c: d["delta"][c]["base_pos_max_m"] for c in conds}
    jnt = {c: d["delta"][c]["joint_pos_rms"] for c in conds}
    # A floor, not an epsilon. At 1e-6 a condition whose depth moved by a rounding error
    # passes as "seen", which is the mistake this column exists to stop.
    SEEN_FLOOR = 1e-3
    saw = {c: sum(stim[c]) > SEEN_FLOOR for c in conds}

    print(f"\n=== terrain probe: {os.path.basename(a.probe)} "
          f"({d['repeats']} envs/condition, paired against '{ctrl}') ===")
    hdr = "".join(f"{k[:9]:>11s}" for k in cams)
    print(f"  {'condition':<22s}{hdr}{'saw it?':>9s}{'plan dxy':>10s}{'joint':>9s}{'per img':>9s}")
    print(f"  {'':<22s}{'':>{11 * len(cams)}s}{'':>9s}{'(m)':>10s}{'(rad)':>9s}{'':>9s}")
    for c in conds:
        tot = sum(stim[c])
        ratio = resp[c] / tot if tot > SEEN_FLOOR else float("nan")
        print(f"  {c:<22s}" + "".join(f"{v:11.4f}" for v in stim[c])
              + f"{('yes' if saw[c] else 'NO'):>9s}{resp[c]:10.4f}{jnt[c]:9.4f}{ratio:9.2f}")
    missed = [c for c in conds if not saw[c]]
    if missed:
        print(f"\n  {len(missed)} condition(s) registered on NO camera: {missed}")
        print("  Their plan differences are not about geometry -- it never reached the model.")

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
    # The two real-terrain conditions are the yardstick: the generator SHOULD react to those.
    real = [c for c in conds if c.startswith(("box_", "stairs_"))]
    ax1.barh(ypos, vals, height=0.66,
             color=["#c2543d" if c in real else ("#3b7dd8" if saw[c] else "none") for c in conds],
             edgecolor=["#c2543d" if c in real else ("#3b7dd8" if saw[c] else "#b0b0b0")
                        for c in conds],
             hatch=[None if saw[c] else "//" for c in conds])
    for c in real:
        ax1.axvline(resp[c], color="#c2543d", lw=1.0, ls="--", alpha=0.7)
    ax1.set_xlabel("peak displacement of the predicted base path (m)")
    ax1.set_title("how far the plan moved\n(red = real terrain, the yardstick; hollow = unseen)")
    ax1.grid(axis="x", alpha=0.3)

    out = a.out or os.path.splitext(a.probe)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
