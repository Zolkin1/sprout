# Copyright 2026 Zachary Olkin. All rights reserved.

"""How much does visual clutter move the generator's plan, on a model that never saw any?

``augment_registration.py --visual_only --clutter`` re-renders a recording changing ONLY the
pixels: the reference, the poses and the commands come back bit-identical. So a checkpoint
forwarded over both halves is a controlled experiment on the conditioning -- any difference in
what it predicts was caused by the scenery and nothing else. No training is involved, which is
the point: this is the BEFORE number a clutter-trained checkpoint gets read against, and it
can be taken the day the renders finish.

The clutter is vetted to sit clear of everywhere the robot was recorded, so on almost every
window the correct response is *no change at all*. A plan that moves anyway is keying on
scenery it should be ignoring -- the shortcut this line of work exists to break.

Two splits, and the first is a control rather than a result:

* **depth unchanged** -- the clutter was out of frame. The two runs are then the same
  conditioning, so the plans must agree to the sampler's own determinism. A non-zero number
  here is a bug in the harness (unpaired noise, a normalizer refit), not a finding, and it is
  checked before anything else is believed.
* **depth changed** -- the response. Reported against how much of the frame moved, so it is a
  sensitivity rather than a bare displacement: a big plan change to a big image change is not
  the same failure as a big plan change to a small one.

Noise is drawn ONCE and handed to both members of a pair, so the conditioning is the only
thing that differs. Without that the sampler's own variance swamps the effect being measured.

    python scripts/flow_matching/fm_clutter_sensitivity.py \\
        --checkpoint logs/flow_matching/<run>/best.pt \\
        --clean <recorded>.h5 --clutter <recorded>.clutter_v1.h5 --num_windows 2000
"""

from __future__ import annotations

import argparse
import json
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


def window_keys(d: dict) -> np.ndarray:
    """``(episode, clip, step)`` per window, the identity a re-render cannot change.

    The renderer regroups windows by env, so output row ``i`` is NOT input row ``i`` and an
    index-wise comparison silently pairs unrelated windows.

    Args:
        d: Recording arrays.

    Returns:
        ``(W,)`` int64 keys.
    """
    e = np.asarray(d["episode_id"]).astype(np.int64)
    c = np.asarray(d["clip_id"]).astype(np.int64)
    s = np.asarray(d["clip_step"]).astype(np.int64)
    return (e << 40) | (c << 20) | s


@torch.no_grad()
def main() -> int:
    """Forward one checkpoint over a clean/cluttered pair and report the paired difference.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Flow-matching .pt.")
    ap.add_argument("--clean", required=True, help="The recording as recorded.")
    ap.add_argument("--clutter", required=True,
                    help="Its --visual_only --clutter re-render. Same windows, new pixels.")
    ap.add_argument("--num_windows", type=int, default=2000,
                    help="Windows to forward, spread evenly over the recording.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="Result .json (default: beside the checkpoint).")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clean, meta = load_recorded_windows(a.clean)
    dirty, _ = load_recorded_windows(a.clutter)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = FlowMatchingConfig.from_dict(ckpt["config"])
    cfg.transition_fraction = None
    # A checkpoint carries the ABSOLUTE mjcf path of the box it was trained on, and training
    # runs on the server. The 5-channel depth row needs that model for its camera FK, so a
    # checkpoint moved between machines cannot be packed at all until this is re-pointed.
    # Derived from the package root, never a count of ".." hops from __file__.
    if not os.path.exists(str(cfg.fk_model_path)):
        from motion_gen.features import DEFAULT_MJCF
        print(f"[sensitivity] fk_model_path {cfg.fk_model_path} is not on this box; "
              f"using {DEFAULT_MJCF}")
        cfg.fk_model_path = DEFAULT_MJCF
    ctx = load_fm_context(a.checkpoint, "cpu")
    model = ctx.model.to(dev).eval()

    kc, kd = window_keys(clean), window_keys(dirty)
    order = np.argsort(kd)
    sel = np.linspace(0, len(kc) - 1, min(a.num_windows, len(kc))).astype(np.int64)
    pos_in_d = order[np.searchsorted(kd, kc[sel], sorter=order)]
    if not np.array_equal(kd[pos_in_d], kc[sel]):
        raise SystemExit("the two recordings do not hold the same windows; --clutter must be "
                         "a --visual_only re-render of --clean, not a different recording.")

    keys = [k for k, v in clean.items() if isinstance(v, np.ndarray)
            and v.shape[:1] == (clean["history_state"].shape[0],)]
    sub_c = {k: clean[k][sel] for k in keys}
    sub_d = {k: dirty[k][pos_in_d] for k in keys}
    # The reference is what the generator is asked to produce. If the re-render moved it, the
    # pair is not a controlled experiment and every number below is uninterpretable.
    if not np.array_equal(sub_c["ref_future"], sub_d["ref_future"]):
        raise SystemExit("ref_future differs between the two files: the re-render was not "
                         "--visual_only, so a plan difference cannot be attributed to pixels.")

    def pack(sub):
        """Pack one half with the CHECKPOINT's normalizers, exactly as training did."""
        return FutureStateDataset(sub, meta, cfg, np.arange(len(sel)),
                                  state_normalizer=ctx.state_norm,
                                  cmd_normalizer=getattr(ctx, "cmd_norm", None),
                                  extero_normalizer=getattr(ctx, "extero_norm", None))

    ds_c, ds_d = pack(sub_c), pack(sub_d)
    blk = ds_c.block_slices["base_pose"]
    xyz = slice(blk.start, blk.start + 3)

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    moved, frac_px = [], []
    for lo in range(0, len(sel), a.batch):
        hi = min(lo + a.batch, len(sel))
        items_c = [ds_c[k] for k in range(lo, hi)]
        items_d = [ds_d[k] for k in range(lo, hi)]

        def stack(items, key):
            return torch.stack([it[key] for it in items]).to(dev)

        # ONE noise draw, both members. sample_trajectory takes `noise` precisely so a paired
        # comparison is not fighting the sampler's own variance.
        noise = torch.randn(hi - lo, cfg.future_len, cfg.state_dim, generator=gen).to(dev)
        outs = []
        for items in (items_c, items_d):
            p = sample_trajectory(
                model, stack(items, "history"), stack(items, "extero"),
                stack(items, "cmd_tokens") if "cmd_tokens" in items[0] else None,
                cfg.future_len, cfg.state_dim, cfg.num_sampling_steps, noise=noise,
            )
            outs.append(ctx.state_norm.denormalize(p.cpu())[:, :, xyz].numpy())
        moved.append(np.linalg.norm(outs[1] - outs[0], axis=2).mean(axis=1))
        d0 = sub_c["depth"][lo:hi, 1, :]; d1 = sub_d["depth"][lo:hi, 1, :]
        frac_px.append((np.abs(d1 - d0) > 1e-4).mean(axis=1))

    moved = np.concatenate(moved); frac_px = np.concatenate(frac_px)
    unseen, seen = frac_px <= 0.0, frac_px > 0.0

    print(f"\n=== clutter sensitivity: {os.path.basename(a.checkpoint)} ===")
    print(f"  {len(moved)} windows; clutter visible in {100 * seen.mean():.1f}% of them\n")
    if not unseen.any():
        print("  NO CONTROL GROUP: every window's depth changed, so nothing here separates "
              "the clutter's effect from the renderer's.\n  This is what running against "
              "DR-ON files looks like -- the depth DR is re-drawn per render, independently "
              "of clutter. Re-render the pair with --no_depth_dr; the DR-on files are for "
              "TRAINING, not for this measurement.")
        return 1
    qs = [50, 90, 99, 100]
    c = np.percentile(moved[unseen], qs)
    r = np.percentile(moved[seen], qs)
    print(f"  {'':28s}{'p50':>9s}{'p90':>9s}{'p99':>9s}{'max':>9s}")
    print(f"  {'CONTROL (clutter unseen)':28s}" + "".join(f"{v:9.4f}" for v in c)
          + f"   n={int(unseen.sum())}")
    print(f"  {'RESPONSE (clutter seen)':28s}" + "".join(f"{v:9.4f}" for v in r)
          + f"   n={int(seen.sum())}")
    # The HEADLINE is an exceedance, not a ratio: the control is ~0 over most of its range,
    # so a percentile ratio divides by near-zero and reports millions.
    #
    # The threshold is the control's p99.9 and NOT its max. The max is one window, so the
    # statistic inherits that window's luck -- measured across two clutter draws of the same
    # checkpoint the max moved 0.0066 -> 0.0241 and dragged the headline 47% -> 13%, while
    # p99.9 held at 0.0050 / 0.0045 and the two agreed to five points. A threshold that
    # unstable cannot compare two checkpoints, which is the entire job here.
    thr = float(np.percentile(moved[unseen], 99.9))
    big = frac_px > 0.02
    print(f"\n  Windows whose plan moved further than the control's p99.9 ({thr:.4f} m):")
    print(f"    of all clutter-visible windows      : {100 * (moved[seen] > thr).mean():5.1f}% "
          f"({int((moved[seen] > thr).sum())} of {int(seen.sum())})")
    if big.any():
        print(f"    of windows with >2% of frame changed: "
              f"{100 * (moved[big] > thr).mean():5.1f}% "
              f"({int((moved[big] > thr).sum())} of {int(big.sum())})   <-- the headline")
    print(f"\n    sensitivity (plan m per unit of frame changed): "
          f"p50 {np.median(moved[seen] / frac_px[seen]):.4f}")
    res = dict(checkpoint=a.checkpoint, clean=a.clean, clutter=a.clutter,
               n_windows=int(len(moved)), frac_seen=float(seen.mean()),
               control_max_m=float(moved[unseen].max()) if unseen.any() else None,
               control_p99_m=float(np.percentile(moved[unseen], 99)) if unseen.any() else None,
               control_p999_m=float(np.percentile(moved[unseen], 99.9)) if unseen.any() else None,
               exceed_control_frac=float(
                   (moved[frac_px > 0.02] > np.percentile(moved[unseen], 99.9)).mean())
               if (unseen.any() and (frac_px > 0.02).any()) else None,
               response_p50_m=float(np.median(moved[seen])) if seen.any() else None,
               response_p90_m=float(np.percentile(moved[seen], 90)) if seen.any() else None,
               moved_m=moved.tolist(), frac_pixels=frac_px.tolist())
    out = a.out or os.path.join(os.path.dirname(a.checkpoint), "fm_clutter_sensitivity.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[sensitivity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
