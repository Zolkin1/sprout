# Copyright 2026 Zachary Olkin. All rights reserved.

"""Does the generator reproduce the target it was TRAINED on, for a window and its augments?

The registration augmentation was measured inert by ``fm_registration_probe.py``, which
displaces the robot in Isaac and asks whether the predicted terrain segment stays put. This
asks a different and more direct question: given exactly the conditioning the training set
contains -- one source window and its N displaced counterparts, each with its OWN depth --
does the model produce each one's OWN target?

The two outcomes point opposite ways, which is why it is worth running:

* fits the augmented targets but still slides in the Isaac probe -> the data teaches something
  the probe does not ask, and the PROBE is the suspect;
* misses the augmented targets too -> the model never learned the mapping, and the lever is
  capacity / offset magnitude / an explicit terrain-relative parameterisation.

Everything is packed through :class:`FutureStateDataset` with the CHECKPOINT's normalizers, so
the conditioning is byte-for-byte what training fed the model -- not a re-derivation that could
differ in a way that would itself explain a miss.

    python scripts/flow_matching/fm_augment_probe.py \\
        --checkpoint logs/flow_matching/<run>/best.pt \\
        --data <original>.h5 --augmented <augmented>.h5 --out probe.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO, "source", "robot_rl"))
sys.path.insert(0, os.path.join(_REPO, "source", "motion_gen"))
sys.path.insert(0, os.path.join(_REPO, "scripts", "rsl_rl"))

from recorded_data_io import load_recorded_windows  # noqa: E402
from robot_rl.flow_matching.config import FlowMatchingConfig  # noqa: E402
from robot_rl.flow_matching.dataset import FutureStateDataset  # noqa: E402
from robot_rl.flow_matching.flow import sample_trajectory  # noqa: E402
from robot_rl.flow_matching.online import load_fm_context  # noqa: E402


def pick_source(data: dict, lib_path: str, cfg: FlowMatchingConfig, want_transition: bool,
                seed: int) -> int:
    """Index of a source window to probe, preferring one whose horizon crosses terrain.

    Args:
        data: Original recording arrays.
        lib_path: Clip library, for the anchor mask.
        cfg: Config (``future_len`` bounds the horizon).
        want_transition: Prefer a window containing a pinned<->flat boundary.
        seed: RNG seed for the choice.

    Returns:
        Row index into the original recording.
    """
    rng = np.random.default_rng(seed)
    if not want_transition:
        return int(rng.integers(0, data["history_state"].shape[0]))
    from motion_gen.clips.clip_io import load_clip_library
    from motion_gen.world_lift import anchor_mask_is_pinned

    lib = load_clip_library(lib_path, device="cpu",
                            keys=["anchor_mask", "anchor_mask_t", "clip_len"])
    pin = anchor_mask_is_pinned(lib.anchor_mask_per_step()).numpy()
    clen = lib.clip_len_or_full().numpy()
    cid, cst = np.asarray(data["clip_id"]), np.asarray(data["clip_step"])
    F = int(cfg.future_len)
    order = rng.permutation(cid.shape[0])
    for i in order[:200000]:
        seg = pin[cid[i], cst[i]:min(cst[i] + F, clen[cid[i]])]
        if seg.size > 1 and bool((seg[1:] != seg[:-1]).any()):   # a boundary inside the horizon
            return int(i)
    return int(order[0])


@torch.no_grad()
def main() -> None:
    """Run one source window and its augments through the model and plot the comparison."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="Original recording.")
    ap.add_argument("--augmented", required=True, help="Its augmented counterpart.")
    ap.add_argument("--out", default="fm_augment_probe.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--any_window", action="store_true",
                    help="Do not insist on a window whose horizon crosses terrain.")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    orig, meta = load_recorded_windows(args.data)
    aug, _ = load_recorded_windows(args.augmented)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    cfg.transition_fraction = None
    # Context on CPU: its normalizers are handed to the dataset, which packs on CPU. Only the
    # MODEL moves to the GPU -- loading the whole context there puts the normalizer buffers on
    # cuda and the dataset's own tensors on cpu, which is a device-mismatch at pack time.
    ctx = load_fm_context(args.checkpoint, "cpu")
    model = ctx.model.to(dev)

    i = pick_source(orig, meta.clip_library_path, cfg, not args.any_window, args.seed)
    ep, cs = int(orig["episode_id"][i]), int(orig["clip_step"][i])
    # The augments of a source window carry its episode_id and clip_step verbatim.
    a_rows = np.flatnonzero((np.asarray(aug["episode_id"]) == ep)
                            & (np.asarray(aug["clip_step"]) == cs))
    print(f"[probe] source row {i}  episode {ep}  clip_step {cs}  -> {len(a_rows)} augments")

    # One small dataset over [source, *augments]. Normalizers come from the CHECKPOINT, so the
    # packing is exactly what training saw rather than refit on four rows.
    keys = [k for k, v in orig.items() if isinstance(v, np.ndarray)
            and v.shape[:1] == (orig["history_state"].shape[0],)]
    combo = {k: np.concatenate([orig[k][i:i + 1], aug[k][a_rows]]) for k in keys}
    ds = FutureStateDataset(combo, meta, cfg, np.arange(1 + len(a_rows)),
                            state_normalizer=ctx.state_norm,
                            cmd_normalizer=getattr(ctx, "cmd_norm", None),
                            extero_normalizer=getattr(ctx, "extero_norm", None))
    pos = ds.block_slices["base_pose"]
    pos = slice(pos.start, pos.start + 3)

    preds, targets, depths = [], [], []
    for k in range(len(ds)):
        item = ds[k]
        torch.manual_seed(args.seed)          # same noise draw for every row: the conditioning
        p = sample_trajectory(                # is the only thing allowed to differ
            model, item["history"][None].to(dev), item["extero"][None].to(dev),
            item["cmd_tokens"][None].to(dev) if "cmd_tokens" in item else None,
            cfg.future_len, cfg.state_dim, cfg.num_sampling_steps,
        )
        preds.append(ctx.state_norm.denormalize(p.cpu())[0, :, pos].numpy())
        targets.append(ctx.state_norm.denormalize(item["state"][None])[0, :, pos].numpy())
        depths.append(combo["depth"][k])
    preds, targets = np.stack(preds), np.stack(targets)

    err = np.linalg.norm(preds - targets, axis=2)             # (rows, F)
    print(f"\n{'row':>10s} {'|pred-target| p50':>18s} {'final node':>12s}")
    for k in range(len(preds)):
        lbl = "source" if k == 0 else f"aug {k}"
        print(f"{lbl:>10s} {np.median(err[k]):18.4f} {err[k, -1]:12.4f}")
    # Does the model track its OWN target, or does it emit the SOURCE's answer regardless?
    cross = np.linalg.norm(preds[1:] - targets[0][None], axis=2)
    print(f"\naugment predictions vs the SOURCE target : p50 {np.median(cross):.4f} m")
    print(f"augment predictions vs their OWN target  : p50 {np.median(err[1:]):.4f} m")
    print("  (own << source  =>  the model USES the displaced depth;"
          "  own ~ source  =>  it ignores it)")

    _plot(args.out, depths, preds, targets)
    np.savez(os.path.splitext(args.out)[0] + ".npz", preds=preds, targets=targets,
             depth=np.stack(depths), err=err)
    print(f"[probe] wrote {args.out}")


def _plot(out: str, depths, preds, targets) -> None:
    """Depth images plus predicted-vs-target paths, one row per window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(preds)
    H, W = 26, 30
    to_m = lambda v: v / 0.5 + 0.75
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.1 * n))
    for k in range(n):
        lbl = "SOURCE" if k == 0 else f"AUGMENT {k}"
        for c, cam in enumerate(("near", "far")):
            ax = axes[k][c]
            ax.imshow(to_m(depths[k][c].reshape(H, W)), cmap="viridis", aspect="auto")
            ax.set_title(f"{lbl}  depth {cam}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = axes[k][2]
        ax.plot(targets[k][:, 0], targets[k][:, 2], color="#0077BB", lw=3, label="target")
        ax.plot(preds[k][:, 0], preds[k][:, 2], color="#EE7733", lw=3, label="prediction")
        ax.set_title(f"{lbl}  side view", fontsize=9)
        ax.set_xlabel("downrange x (m)", fontsize=8); ax.set_ylabel("vertical z (m)", fontsize=8)
        if k == 0:
            ax.legend(fontsize=8)
        ax = axes[k][3]
        ax.plot(np.linalg.norm(preds[k] - targets[k], axis=1), color="#CC3311", lw=2.5)
        ax.plot(np.linalg.norm(preds[k] - targets[0], axis=1), color="#BBBBBB", lw=2,
                ls="--", label="vs SOURCE target")
        ax.set_title("|pred - target| per node (m)", fontsize=9)
        ax.set_xlabel("node", fontsize=8)
        if k == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)


if __name__ == "__main__":
    main()
