#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Step through a rollout's generator proposals with the arrow keys, one figure per re-plan.

Opens a matplotlib window and walks the RE-PLANS -- the steps where the generator actually
inferred, not every control step, since the steps in between only serve a plan already made.
Each frame redraws the x-z panels (each foot, then the base) with the terrain profile, all N
draws, the served one, the previous solve's draws, the clip over the same nodes, the history the
generator was conditioned on, and its depth images.

Give it TWO bundles and they stack as two rows moving together: the same re-plan index under a
teacher-forced history and under the closed loop, on shared axes. That is the comparison that
separates a weak model from a diverging loop, and side by side it reads at a glance.

Keys::

    right / left     next / previous RE-PLAN          (the main navigation)
    up / down        next / previous single step
    ] / [            next / previous CLIP (each case is its own env and tile)
    1..9             jump straight to that clip
    page up / down   jump 10 re-plans
    home / end       first / last re-plan
    s                save the current figure as a PNG
    q                quit

Examples::

    # one bundle
    python scripts/rsl_rl/fm_feet_browser.py --out logs/fm_debug/stepper16
    # closed loop vs teacher forcing, stacked and stepping together
    python scripts/rsl_rl/fm_feet_browser.py --out logs/fm_debug/stepper16 \
        --out logs/fm_debug/stepper16_tf

Needs an interactive matplotlib backend (a display). With none available it says so rather than
opening a window that never draws; use the viser button in ``fm_debug --stage view`` instead, or
``--save_all`` to write every re-plan's figure headlessly.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
for _p in (os.path.join(_REPO, "source", "robot_rl"), os.path.join(_REPO, "source", "motion_gen"),
           _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _pick_backend(headless: bool) -> bool:
    """Select an interactive backend if one is usable; returns whether it is interactive.

    Chosen BEFORE ``fm_feet_plot`` is imported, because that module claims Agg when no
    interactive backend has been set.
    """
    if headless or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg")
        return False
    for name in ("TkAgg", "QtAgg", "GTK4Agg"):
        try:
            matplotlib.use(name)
            return True
        except Exception:
            continue
    matplotlib.use("Agg")
    return False


def main() -> int:
    """Parse args, build the sources, and run the key loop."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", action="append", required=True,
                   help="fm_debug run directory (repeatable; each becomes a stacked row).")
    p.add_argument("--mode", default="fm", help="Which bundle in each run dir.")
    p.add_argument("--env", type=int, default=0, help="Env index within the bundles.")
    p.add_argument("--step", type=int, default=None, help="Start at this rollout step.")
    p.add_argument("--save_dir", default=None,
                   help="Where 's' writes PNGs (default: the first run directory).")
    p.add_argument("--save_all", action="store_true",
                   help="Headlessly write every re-plan's figure and exit.")
    p.add_argument("--headless", action="store_true", help="Force the Agg backend.")
    args = p.parse_args()

    interactive = _pick_backend(args.headless or args.save_all)
    import matplotlib.pyplot as plt
    import numpy as np

    from fm_debug import load_bundle
    from fm_debug_scene import _depth_image
    from fm_feet_data import FeetPlotSource
    import fm_feet_plot

    sources = []
    for out in args.out:
        data, meta = load_bundle(out, args.mode)
        src = FeetPlotSource(data, meta, mode=args.mode, depth_image_fn=_depth_image)
        if not src.has_draws:
            raise SystemExit(
                f"{out}/{args.mode}.npz carries no 'fm_draws'. Capture with "
                "`fm_debug --stage run --num_draws N` (N > 1); a single-draw bundle has no "
                "proposals to compare.")
        src.label = f"{os.path.basename(os.path.normpath(out))}\nhistory={src.history_source}"
        sources.append(src)
    save_dir = args.save_dir or args.out[0]
    os.makedirs(save_dir, exist_ok=True)

    # Navigate on the FIRST bundle's re-plans and apply the same step to the rest: the runs are
    # separate rollouts, so their re-plan steps only coincide because the cadence is fixed --
    # asserted here rather than assumed, since a mismatch would silently compare different
    # moments in the two runs.
    marks = sources[0].replans(args.env)
    for src in sources[1:]:
        other = src.replans(args.env)
        n = min(marks.size, other.size)
        if not np.array_equal(marks[:n], other[:n]):
            print("[browser] WARNING: the bundles re-plan at different steps; rows are aligned "
                  "by STEP, so one row may be mid-plan.")
            break
    # Rows must share a step axis, so the shortest bundle sets the budget. Say so when it
    # actually truncates something: a 130-step control bundle browsed next to a 441-step one
    # would otherwise silently hide two thirds of the longer rollout, which reads as "the
    # plotter stops early" rather than "this bundle is short".
    lengths = [s.num_steps() for s in sources]
    n_steps = min(lengths)
    if len(set(lengths)) > 1:
        print(f"[browser] bundles differ in length {dict(zip(args.out, lengths))}; the shortest "
              f"({n_steps} steps) sets the range, so the longer ones are truncated. Browse them "
              "separately to see the whole rollout.")
    marks = marks[marks < n_steps]
    print(f"[browser] {n_steps} steps, re-plans {marks[0]}..{marks[-1]} ({marks.size} of them)")
    # State is the STEP, not an index into `marks`: navigating by index made the single-step
    # keys dead (a +-1 step snapped straight back to the same re-plan). Re-plans are then just
    # the coarse jumps, and mid-plan steps stay reachable -- worth having, since between
    # re-plans the conditioning is held and only the robot moves, which is itself a thing to
    # look at.
    state = {"step": int(marks[0])}
    if args.step is not None:
        state["step"] = int(np.clip(int(args.step), 0, n_steps - 1))
    fig = plt.figure(figsize=(4.6 * 3 + 2.6, 3.9 * len(sources)))

    # Each case is one env on its own tile, so switching env switches BOTH the clip and the
    # terrain under it -- which is what makes "the same finding on another clip" one keypress.
    cases = list(sources[0].meta.get("cases", []))
    env_tid = np.asarray(sources[0].meta.get("env_terrain_id") or []).reshape(-1)
    envs = sorted({int(e) for e in range(env_tid.size)}) or [args.env]
    case_of_env = {}
    for e in envs:
        hit = next((c for c in cases if int(c.get("terrain_id", -999)) == int(env_tid[e])), None)
        case_of_env[e] = (f"{hit['name']} (clip {hit['clip_index']}, tile {hit['terrain_id']})"
                          if hit else f"env {e}")
    state_env = {"i": envs.index(args.env) if args.env in envs else 0}

    def rows_for(step: int) -> list[dict]:
        """Assemble one input dict per bundle at ``step``, for the selected env."""
        env = envs[state_env["i"]]
        rows = []
        for src in sources:
            row = src.inputs(step, env)
            if row is not None:
                row["row_label"] = src.label
                rows.append(row)
        return rows

    def redraw() -> None:
        """Rebuild the figure in place for the current step."""
        step = state["step"]
        rows = rows_for(step)
        if not rows:
            return
        at = int(np.searchsorted(marks, step, side="right")) - 1
        offset = step - int(marks[at]) if at >= 0 else 0
        where = (f"re-plan {at + 1}/{marks.size}" if offset == 0
                 else f"re-plan {at + 1}/{marks.size} + {offset} steps (conditioning held)")
        fm_feet_plot.panels_figure(
            rows, fig=fig,
            suptitle=f"{where}  |  step {step} (t={step * sources[0].dt:.2f} s)  |  "
                     f"clip {state_env['i'] + 1}/{len(envs)} -- {rows[0].get('clip_label', '')}"
                     f"  |  "
                     f"left/right = re-plan, up/down = step, [ ] = clip, s = save, q = quit")
        fig.canvas.draw_idle()

    def save() -> str:
        """Write the current figure next to the bundle."""
        env = envs[state_env["i"]]
        path = os.path.join(save_dir, f"feet_xz_env{env}_step{state['step']:04d}.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"[browser] wrote {path}")
        return path

    def _replan_hop(step: int, delta: int) -> int:
        """The re-plan ``delta`` places from the one at or before ``step``."""
        at = int(np.searchsorted(marks, step, side="right")) - 1
        # ``at`` is the re-plan at or BEFORE step. Going forward from mid-plan, at+delta is
        # already the upcoming one; going BACKWARD, at+delta would skip over the re-plan this
        # step belongs to, so the first press should land on it.
        if delta < 0 and step > int(marks[max(at, 0)]):
            delta += 1
        return int(marks[int(np.clip(at + delta, 0, marks.size - 1))])

    def on_key(event) -> None:
        """left/right walk re-plans, up/down walk single steps."""
        if event.key in ("q", "escape"):
            plt.close(fig)
            return
        if event.key == "s":
            save()
            return
        step = state["step"]
        # Clip switching: '[' / ']' cycle, or a digit picks one directly.
        if event.key in ("[", "]"):
            state_env["i"] = (state_env["i"] + (1 if event.key == "]" else -1)) % len(envs)
            redraw()
            return
        if event.key and event.key.isdigit() and 1 <= int(event.key) <= len(envs):
            state_env["i"] = int(event.key) - 1
            redraw()
            return
        if event.key == "home":
            step = int(marks[0])
        elif event.key == "end":
            step = int(marks[-1])
        elif event.key in ("right", "left"):
            step = _replan_hop(step, 1 if event.key == "right" else -1)
        elif event.key in ("pageup", "pagedown"):
            step = _replan_hop(step, 10 if event.key == "pageup" else -10)
        elif event.key in ("up", "down"):
            step = step + (1 if event.key == "up" else -1)
        else:
            return
        state["step"] = int(np.clip(step, 0, n_steps - 1))
        redraw()

    if args.save_all or not interactive:
        if not interactive and not args.save_all:
            print("[browser] no interactive backend available (no DISPLAY?). Writing every "
                  "re-plan instead; use fm_debug --stage view for the interactive path.")
        for e in range(len(envs)):
            state_env["i"] = e
            for m in marks:
                state["step"] = int(m)
                redraw()
                save()
        return 0

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    print(f"[browser] {marks.size} re-plans over {len(sources)} bundle(s); "
          f"arrows navigate, s saves, q quits.")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
