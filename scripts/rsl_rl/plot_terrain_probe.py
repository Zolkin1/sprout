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

``--compare`` draws a SECOND probe beside the first, which is how a retrain is read: the
response to a wall is not interpretable on its own (a generator that ignores everything
scores a flat zero and looks calm), so what the number means is how it MOVED against the
same conditions on the previous checkpoint. Both runs must have been probed with the same
library, or the conditions do not name the same scenes; only conditions present in both are
drawn.

IsaacLab-free -- reads only the probe's ``.json``.

    python scripts/rsl_rl/plot_terrain_probe.py --probe fm_terrain_probe.json --out probe.png
    python scripts/rsl_rl/plot_terrain_probe.py --probe after.json --compare before.json \
        --labels "after clutter,before"
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
    p.add_argument("--compare", default=None,
                   help="A second fm_terrain_probe.json, drawn beside the first. Use the "
                        "OLDER checkpoint here: --probe is the one being judged.")
    p.add_argument("--labels", default="",
                   help="'primary,compare' legend labels. Defaults to each checkpoint's run "
                        "directory, which is the name that identifies it.")
    p.add_argument("--out", default=None, help="Figure path (default: beside the json).")
    a = p.parse_args()

    with open(a.probe) as fh:
        d = json.load(fh)
    d2 = None
    if a.compare:
        with open(a.compare) as fh:
            d2 = json.load(fh)
    conds = [c for c in d["conditions"] if c != d["conditions"][0]]
    ctrl = d["conditions"][0]
    if d2 is not None:
        # Shared conditions only. A condition in one run and not the other would otherwise
        # shift the bars and put two different scenes at the same y.
        shared = [c for c in conds if c in d2["delta"] and c in d2["stimulus"]]
        dropped = [c for c in conds if c not in shared]
        if dropped:
            print(f"[plot] not in --compare, dropped: {', '.join(dropped)}")
        if not shared:
            raise SystemExit("--compare shares no conditions with --probe; they were almost "
                             "certainly probed with different libraries.")
        conds = shared
    def _run_name(src: dict, fallback: str) -> str:
        """The checkpoint's run directory, which is the name that identifies it."""
        ck = str(src.get("fm_checkpoint", "")).rstrip("/")
        return os.path.basename(os.path.dirname(ck)) or fallback

    given = [t.strip() for t in a.labels.split(",") if t.strip()]
    labels = [given[0] if len(given) > 0 else _run_name(d, "probe"),
              given[1] if len(given) > 1 else (_run_name(d2, "compare") if d2 else "")]
    cams = list(d["stimulus"][conds[0]]["per_camera"].keys())

    stim = {c: [d["stimulus"][c]["per_camera"][k]["mean_abs_delta"] for k in cams] for c in conds}
    resp = {c: d["delta"][c]["base_pos_max_m"] for c in conds}
    jnt = {c: d["delta"][c]["joint_pos_rms"] for c in conds}
    resp2 = {c: d2["delta"][c]["base_pos_max_m"] for c in conds} if d2 else None
    # A floor, not an epsilon. At 1e-6 a condition whose depth moved by a rounding error
    # passes as "seen", which is the mistake this column exists to stop.
    SEEN_FLOOR = 1e-3
    saw = {c: sum(stim[c]) > SEEN_FLOOR for c in conds}

    print(f"\n=== terrain probe: {os.path.basename(a.probe)} "
          f"({d['repeats']} envs/condition, paired against '{ctrl}') ===")
    hdr = "".join(f"{k[:9]:>11s}" for k in cams)
    extra = f"{'was':>10s}{'change':>9s}" if resp2 else ""
    print(f"  {'condition':<22s}{hdr}{'saw it?':>9s}{'plan dxy':>10s}{extra}"
          f"{'joint':>9s}{'per img':>9s}")
    print(f"  {'':<22s}{'':>{11 * len(cams)}s}{'':>9s}{'(m)':>10s}"
          f"{('%10s%9s' % ('(m)', '')) if resp2 else ''}{'(rad)':>9s}{'':>9s}")
    for c in conds:
        tot = sum(stim[c])
        ratio = resp[c] / tot if tot > SEEN_FLOOR else float("nan")
        # Absolute, not a ratio: the control conditions sit near zero, where a ratio is
        # dominated by whichever run happened to be closer to it.
        was = f"{resp2[c]:10.4f}{resp[c] - resp2[c]:+9.4f}" if resp2 else ""
        print(f"  {c:<22s}" + "".join(f"{v:11.4f}" for v in stim[c])
              + f"{('yes' if saw[c] else 'NO'):>9s}{resp[c]:10.4f}{was}"
              + f"{jnt[c]:9.4f}{ratio:9.2f}")
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
    h, off = (0.32, 0.17) if resp2 else (0.66, 0.0)
    if resp2:
        # The older run in grey and BEHIND, so the judged run keeps the colour semantics
        # (red = real terrain, hollow = the camera never saw it) rather than both fighting
        # for them.
        ax1.barh(ypos - off, [resp2[c] for c in conds], height=h,
                 color="#b9b9b9", edgecolor="#8a8a8a", label=labels[1])
    ax1.barh(ypos + off, vals, height=h,
             color=["#c2543d" if c in real else ("#3b7dd8" if saw[c] else "none") for c in conds],
             edgecolor=["#c2543d" if c in real else ("#3b7dd8" if saw[c] else "#b0b0b0")
                        for c in conds],
             hatch=[None if saw[c] else "//" for c in conds],
             label=labels[0] if resp2 else None)
    for c in real:
        ax1.axvline(resp[c], color="#c2543d", lw=1.0, ls="--", alpha=0.7)
    if resp2:
        ax1.legend(loc="lower right")
    ax1.set_xlabel("peak displacement of the predicted base path (m)")
    ax1.set_title("how far the plan moved\n(red = real terrain, the yardstick; hollow = unseen)"
                  + (f"\n{labels[0]} vs {labels[1]}" if resp2 else ""))
    ax1.grid(axis="x", alpha=0.3)

    out = a.out or os.path.splitext(a.probe)[0] + ".png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
