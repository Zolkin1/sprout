# Copyright 2026 Zachary Olkin. All rights reserved.

"""Measure a generator's window error PER TABLE ENTRY and PER OFFSET.

``REFERENCE_WINDOW_NOISE`` stands in for the generator's prediction error, in two places that
model the same thing (the policy's ``reference_window`` observation noise and the FM's own
autoregressive history noise). But nothing measured that error: ``channel_metrics``' ``_rmse``
collapses the offset axis AND the channel axis into one scalar per block, so the table's
per-axis magnitudes were never checked against what they claim to model. The one number per
block is applied to every observed offset, while the error grows 3-10x across the horizon --
so this reports the curve, not an average, and prints the half-width each entry would need.

Run it after every FM retrain: the table is only as good as the generator it was calibrated
against, and a checkpoint's error profile shifts with capacity, horizon and data.

Example::

    python scripts/flow_matching/fm_error_by_offset.py \\
        --checkpoint logs/flow_matching/<run>/best.pt \\
        --data logs/rsl_rl/g1/<run>/recorded_data/<rec>.h5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields

import h5py
import numpy as np
import torch

from recorded_data_io import RecordingMeta
from robot_rl.flow_matching.config import FlowMatchingConfig
from robot_rl.flow_matching.dataset import build_train_val
from robot_rl.flow_matching.flow import sample_trajectory
from robot_rl.flow_matching.online import load_fm_context
from robot_rl.flow_matching.window import (
    REFERENCE_WINDOW_NOISE,
    resolve_reference_blocks,
    window_block_slices,
)

#: Uniform noise is drawn on ``[-hw, hw]``, whose standard deviation is ``hw / sqrt(3)``. The
#: table stores half-widths, so every comparison against a measured RMS goes through this.
_UNIFORM_STD_PER_HALF_WIDTH = 1.0 / np.sqrt(3.0)


def load_capped(path: str, cap: int, allow_per_step_history: bool) -> tuple[dict, RecordingMeta]:
    """Load at most ``cap`` windows from a recording, so a 10 GB file need not fit in RAM.

    Args:
        path: Windowed ``.h5`` recording.
        cap: Maximum windows to read.
        allow_per_step_history: Stamp ``history_anchor="plan"`` on a recording that predates
            the plan-frame history, so the dataset's guard admits it. Only correct when the
            checkpoint being measured was itself TRAINED on this recording -- then a per-step
            history is the input that model expects. The guard exists to stop a per-step
            recording training a NEW generator, which is not what this script does. The
            target (``ref_future``) is untouched by history anchoring, and the target is what
            the error is measured against.

    Returns:
        ``(data, meta)`` as :func:`load_recorded_windows` would, trimmed to ``cap``.
    """
    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        n = int(f.attrs.get("num_windows_written", f.attrs.get("num_windows", 0)))
        n = min(n, cap)
        for key in f.keys():
            data[key] = f[key][:n]
        raw = {k: json.loads(v) for k, v in f.attrs.items()
               if k not in ("num_windows", "num_windows_written")}
    names = {fl.name for fl in fields(RecordingMeta)}
    meta = RecordingMeta(**{k: v for k, v in raw.items() if k in names})
    if allow_per_step_history and meta.history_anchor != "plan":
        print(f"[warn] history_anchor={meta.history_anchor or '<absent>'!r} -> treating as "
              "'plan' for this READ-ONLY measurement (--allow_per_step_history).")
        meta.history_anchor = "plan"
    return data, meta


def probe_slices(num_joints: int, num_extra_frames: int, blocks) -> dict[str, tuple[slice, bool]]:
    """Map each reachable ``REFERENCE_WINDOW_NOISE`` key to its node slice.

    Args:
        num_joints: Robot joint count.
        num_extra_frames: Extra body frames packed per node.
        blocks: The window's resolved block selection.

    Returns:
        ``{table_key: (slice, per_axis)}``. ``per_axis`` marks the position triples, which the
        table stores as ``(x, y, z)`` and which are therefore reported per channel; everything
        else is one table number and is pooled over its channels.
    """
    bs = window_block_slices(num_joints, num_extra_frames, blocks)
    out: dict[str, tuple[slice, bool]] = {}
    if "base_pose" in bs:
        start = bs["base_pose"].start
        out["base_pos"] = (slice(start, start + 3), True)
        out["base_ori"] = (slice(start + 3, start + 9), False)
    if "base_vel" in bs:
        start = bs["base_vel"].start
        out["base_lin_vel"] = (slice(start, start + 3), False)
        out["base_ang_vel"] = (slice(start + 3, start + 6), False)
    for key in ("joint_pos", "joint_vel"):
        if key in bs:
            out[key] = (bs[key], False)
    if "ee_pos" in bs and num_extra_frames > 0:
        out["ee_pos"] = (bs["ee_pos"], True)
    return out


def main() -> None:
    """Sample a checkpoint over the val split and report error by entry and by offset."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="FM .pt checkpoint (uses EMA weights).")
    ap.add_argument("--data", required=True, help="The windowed .h5 it was trained on.")
    ap.add_argument("--cap", type=int, default=20000, help="Max windows to read from the file.")
    ap.add_argument("--batches", type=int, default=12, help="Val batches to sample.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=5, help="Offsets to print (every Nth).")
    ap.add_argument("--allow_per_step_history", action="store_true",
                    help="Admit a pre-plan-frame recording; see load_capped.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = load_fm_context(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    num_joints = int(ckpt["num_joints"])
    print(f"[cfg] F={cfg.future_len} N={cfg.num_history_states} D={cfg.state_dim} "
          f"steps={cfg.num_sampling_steps} blocks={cfg.reference_blocks}")

    data, meta = load_capped(args.data, args.cap, args.allow_per_step_history)
    _, val = build_train_val(data, meta, cfg)
    if val is None or len(val) == 0:
        raise SystemExit("No val split -- raise --cap or check val_fraction.")
    loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False)

    blocks = resolve_reference_blocks(cfg.reference_blocks)
    probes = probe_slices(num_joints, len(meta.state_frame_names or []), blocks)
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in probes}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            pred = sample_trajectory(
                ctx.model, batch["history"].to(device), batch["height"].to(device),
                batch["cmd_tokens"].to(device) if "cmd_tokens" in batch else None,
                cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
            )
            p = ctx.state_norm.denormalize(pred)
            t = ctx.state_norm.denormalize(batch["state"].to(device))
            for key, (slc, _) in probes.items():
                acc[key].append((p[..., slc] - t[..., slc]).cpu())

    missing = sorted(set(REFERENCE_WINDOW_NOISE) - set(probes))
    for key, (_, per_axis) in probes.items():
        e = torch.cat(acc[key])                                   # (W, F, C)
        if per_axis:
            rms = e.pow(2).mean(0).sqrt().numpy()                 # (F, C) per axis
        else:
            rms = e.pow(2).mean(dim=(0, 2)).sqrt().numpy()[:, None]  # (F, 1) pooled
        current = np.atleast_1d(np.asarray(REFERENCE_WINDOW_NOISE[key], dtype=float))
        implied = current * _UNIFORM_STD_PER_HALF_WIDTH
        mean_err = rms.mean(0)
        print(f"\n=== {key} ===   ({e.shape[0]} val windows, F={e.shape[1]})")
        print("  offset " + "".join(f"{c:>9s}" for c in (("x", "y", "z") if per_axis else ("all",))))
        for k in list(range(0, e.shape[1], args.stride)) + [e.shape[1] - 1]:
            print(f"   {k:4d}  " + "".join(f"{v:9.4f}" for v in rms[k]))
        print(f"  mean over offsets   : {np.round(mean_err, 4)}")
        print(f"  growth 0 -> {e.shape[1]-1:2d}      : {np.round(rms[-1] / np.maximum(rms[0], 1e-12), 1)}x")
        print(f"  table half-width    : {current}  -> implied std {np.round(implied, 4)}")
        print(f"  table/err (mean)    : {np.round(implied / np.maximum(mean_err, 1e-12), 2)}"
              "   (<1 = table UNDER-noises)")
        print(f"  hw matching the mean: {np.round(mean_err / _UNIFORM_STD_PER_HALF_WIDTH, 3)}")
    if missing:
        print(f"\n[note] not measurable from this checkpoint's blocks: {missing}")


if __name__ == "__main__":
    main()
