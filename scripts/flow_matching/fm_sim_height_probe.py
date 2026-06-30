# Copyright 2026 Zachary Olkin. All rights reserved.

"""Isolate the FM root/feet height drift in a sim rollout: re-anchor vs FM-intrinsic.

The in-sim FM (``GeneratorTrackingCommand``) re-anchors each prediction onto the
**robot root** (xy + z + yaw), whereas the offline ``fm_test_bench.py forecast_step``
re-anchors onto the **clip root** as a proxy. When the FM reference root/feet z sink
below the clip (and the feet clip through the ground), the cause is one (or both) of:

  A. **Re-anchor / robot sink** -- the robot root is below the clip, and the z-snap
     copies that low root onto the reference, dragging the whole body (incl. feet)
     down. This is an artifact of anchoring onto the *real robot* instead of the clip.
  B. **FM-intrinsic** -- the FM predicts the root/feet low *relative to its own
     anchor*; even anchored at the correct clip height the reference would still sink.

Because every predicted frame is lifted through the single root anchor ``p_ref``
(``world_z = p_ref.z + rel_z``, yaw rotation preserves z), the decomposition is exact
arithmetic on the logged world-frame z series (env ``e``):

    D_root(t) = fm_root_z - clip_root_z = (fm_root_z - robot_root_z) + (robot_root_z - clip_root_z)
              =        B_root(t)         +              A(t)
    A(t)      = robot_root_z - clip_root_z          # re-anchor / robot-sink contribution
    B_x(t)    = fm_x_z - robot_root_z               # FM prediction relative to its anchor (x in {root, foot})

The **clip-anchored counterfactual** (what the reference would be if anchored on the
clip root, like the bench) shifts every frame up by the root sink ``A``:

    clip_anchored_x_z(t) = fm_x_z(t) - A(t)         # x in {root, foot}

So the three curves the user asked for, per frame z:
  * **auto-regressive as-is**  = ``fm_x_z``            (robot-anchored, what the policy tracked)
  * **re-anchored on the clip**= ``fm_x_z - A``        (clip-anchored counterfactual)
  * **the motion clip**        = ``clip_x_z``

Reads the raw arrays dumped by ``play.py --dump_play_data`` (``play_data.npz`` in the
task plots dir). Numpy + matplotlib only -- no torch / isaaclab.

Example:
    python scripts/rsl_rl/play.py --task G1-tracker-clf-fm-ar-play \
        --clip clips/g1_parity_v1.h5 --plots fm_predictions --dump_play_data <...>
    python scripts/flow_matching/fm_sim_height_probe.py \
        --data logs/rsl_rl/g1/<run>/plots/G1-tracker-clf-fm-ar-play/play_data.npz
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load(path: str) -> dict:
    """Load ``play_data.npz`` into a plain dict (object arrays unwrapped)."""
    raw = np.load(path, allow_pickle=True)
    out: dict = {}
    for k in raw.files:
        v = raw[k]
        if v.dtype == object and v.ndim == 0:
            out[k] = v.item()
        elif v.dtype == object:
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _col(pos_names: list[str], name: str) -> int:
    """Index of channel ``name`` in the ordered pos-output names; -1 if absent."""
    return pos_names.index(name) if name in pos_names else -1


def _series(arr: np.ndarray | None, env: int, col: int) -> np.ndarray | None:
    """Extract the ``[T]`` time series for env ``env``, channel ``col`` from ``[T,N,P]``."""
    if arr is None or col < 0:
        return None
    return np.asarray(arr)[:, env, col].astype(np.float64)


def _new_inference_mask(pred_idx: np.ndarray, novel_start: int) -> np.ndarray:
    """Boolean ``[T]`` mask of steps where the FM re-inferred (window index reset)."""
    idx = np.asarray(pred_idx).astype(np.int64)
    fresh = np.zeros_like(idx, dtype=bool)
    fresh[0] = True
    fresh[1:] = idx[1:] <= idx[:-1]   # idx resets to N then increments between re-infers
    return fresh


def _height_panels(
    ax_overlay, ax_decomp, t, clip_z, robot_z, fm_z, A, active, label: str, is_foot: bool
):
    """Draw the z-overlay (left) + drift decomposition (right) for one frame.

    ``A`` (the root-sink, shared by every frame because the re-anchor shifts the whole
    body by the root) is passed in. The decomposition is exact for every frame::

        D = fm_z - clip_z   (total: FM ref vs clip)
        B = D - A           (FM-intrinsic: clip-anchored ref vs clip, sink removed)
        D = A + B           (identity)

    where the clip-anchored counterfactual is ``fm_z - A`` (frame z if anchored on the
    clip root instead of the robot root).
    """
    clip_anchored = fm_z - A   # shift the frame up by the root sink -> clip-anchored
    D = fm_z - clip_z          # total drift (FM ref below clip)
    B = D - A                  # FM-intrinsic = clip_anchored - clip_z (exact: D = A + B)

    # Mask the warmup / clip-driven region (FM not authoritative) for readability.
    def m(y):
        y = y.copy()
        y[~active] = np.nan
        return y

    ax_overlay.plot(t, clip_z, color="C2", lw=2.0, label="clip (motion ref)")
    ax_overlay.plot(t, robot_z, color="0.4", lw=1.4, ls=":", label="robot (measured)")
    ax_overlay.plot(t, m(fm_z), color="C3", lw=2.0, label="FM ref (robot-anchored, as-is)")
    ax_overlay.plot(t, m(clip_anchored), color="C0", lw=1.8, ls="--",
                    label="FM ref (clip-anchored counterfactual)")
    if is_foot:
        ax_overlay.axhline(0.0, color="k", lw=1.0, alpha=0.6, label="ground (z=0)")
    ax_overlay.set_ylabel(f"{label} z [m]")
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.legend(fontsize=7, loc="best")

    ax_decomp.plot(t, m(D), color="C3", lw=2.0, label="total drift (FM - clip)")
    ax_decomp.plot(t, m(A), color="C1", lw=1.8, label="A: re-anchor / robot sink (robot root - clip root)")
    ax_decomp.plot(t, m(B), color="C0", lw=1.8, ls="--",
                   label="B: FM-intrinsic (clip-anchored ref - clip)")
    ax_decomp.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_decomp.set_ylabel(f"{label} z drift [m]")
    ax_decomp.grid(True, alpha=0.3)
    ax_decomp.legend(fontsize=7, loc="best")

    # Numeric summary over the FM-active region.
    sel = active & np.isfinite(D)
    return {
        "mean_total": float(np.nanmean(D[sel])) if sel.any() else float("nan"),
        "mean_reanchor_A": float(np.nanmean(A[sel])) if sel.any() else float("nan"),
        "mean_fm_intrinsic_B": float(np.nanmean(B[sel])) if sel.any() else float("nan"),
        "min_fm_z": float(np.nanmin(fm_z[sel])) if sel.any() else float("nan"),
        "min_clip_anchored_z": float(np.nanmin(clip_anchored[sel])) if sel.any() else float("nan"),
        "min_clip_z": float(np.nanmin(clip_z[sel])) if sel.any() else float("nan"),
    }


def run(args: argparse.Namespace) -> None:
    """Load the dumped play data and write the height-decomposition figure."""
    d = _load(args.data)
    pos_names = list(d["pos_names"])
    dt = float(d.get("dt", 0.02))
    t_off = float(d.get("t_offset", 0.0))
    e = args.env

    y_act = d.get("y_act")
    y_clip = d.get("y_des_clip")
    y_fm = d.get("y_des_fm", d.get("y_des"))
    if y_act is None or y_clip is None or y_fm is None:
        raise SystemExit(
            "play_data.npz is missing y_act / y_des_clip / y_des_fm. Re-run play with "
            "--dump_play_data on the FM-ar task (GeneratorTrackingCommand)."
        )
    T = y_fm.shape[0]
    t = t_off + np.arange(T) * dt

    fm_active = d.get("fm_active")
    active = (np.asarray(fm_active)[:, e].astype(bool) if fm_active is not None
              else np.ones(T, dtype=bool))

    # Root z anchor series (the single anchor that lifts every frame).
    root = "pelvis_link"
    rc = _col(pos_names, f"{root}:pos_z")
    if rc < 0:
        raise SystemExit(f"no '{root}:pos_z' channel in pos_names: {pos_names[:8]}...")
    clip_root = _series(y_clip, e, rc)
    robot_root = _series(y_act, e, rc)
    fm_root = _series(y_fm, e, rc)
    A = robot_root - clip_root   # re-anchor / robot-sink contribution (shared by all frames)

    # Frames to analyse: root + the requested feet/ee.
    frames = [(root, f"{root}:pos_z", False)]
    for f in args.frames.split(","):
        f = f.strip()
        if f and _col(pos_names, f"{f}:pos_z") >= 0:
            frames.append((f, f"{f}:pos_z", True))
        elif f:
            print(f"[WARN] frame '{f}' has no '{f}:pos_z' channel; skipping.")

    nrows = len(frames)
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 3.2 * nrows), squeeze=False)
    summary: dict[str, dict] = {}
    for r, (fname, chan, is_foot) in enumerate(frames):
        c = _col(pos_names, chan)
        clip_z = _series(y_clip, e, c)
        robot_z = _series(y_act, e, c)
        fm_z = _series(y_fm, e, c)
        summary[fname] = _height_panels(
            axes[r][0], axes[r][1], t, clip_z, robot_z, fm_z, A, active, fname, is_foot
        )
    axes[-1][0].set_xlabel("time [s]"); axes[-1][1].set_xlabel("time [s]")
    fig.suptitle(
        f"FM height drift: re-anchor (robot sink) vs FM-intrinsic  [env {e}]\n"
        f"{os.path.basename(os.path.dirname(args.data))}/{os.path.basename(args.data)}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = args.out or os.path.join(os.path.dirname(args.data), f"fm_height_probe_env{e}.png")
    fig.savefig(out, dpi=130)
    print(f"[INFO] wrote {out}")

    # Console summary: how much of the sink is the re-anchor (A) vs the FM (B).
    print("\n=== FM height-drift decomposition (mean over FM-active steps, metres) ===")
    print(f"{'frame':<26}{'total(FM-clip)':>16}{'A re-anchor':>14}{'B FM-intrinsic':>16}"
          f"{'min FM z':>12}{'min clip-anch':>15}")
    for fname, s in summary.items():
        print(f"{fname:<26}{s['mean_total']:>16.4f}{s['mean_reanchor_A']:>14.4f}"
              f"{s['mean_fm_intrinsic_B']:>16.4f}{s['min_fm_z']:>12.4f}{s['min_clip_anchored_z']:>15.4f}")
    print("\nReading: total = A + B. A dominates -> the height drift is the robot-root "
          "re-anchor (robot sinks, z-snap copies it). B dominates -> the FM itself predicts "
          "low relative to its anchor. For feet, 'min clip-anch' < 0 means the FM still "
          "predicts feet in the ground even at the correct clip root height (FM-intrinsic).")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="play_data.npz from play.py --dump_play_data.")
    p.add_argument("--env", type=int, default=0, help="Env index to analyse.")
    p.add_argument("--frames", default="left_ankle_roll_link,right_ankle_roll_link",
                   help="Comma-separated ee frame names to add (root is always included).")
    p.add_argument("--out", default=None, help="Output PNG (default: alongside --data).")
    run(p.parse_args())


if __name__ == "__main__":
    main()
