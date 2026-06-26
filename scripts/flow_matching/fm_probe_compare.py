#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Compare MuJoCo vs IsaacLab FM-probe dumps (feeding parity + FM-map parity).

Loads the two ``.npz`` dumps written by ``transfer/sim/fm_probe_mujoco.py`` and
``scripts/flow_matching/fm_probe_isaac.py`` (identical clip-derived probe states,
no rollout drift) and reports:

  Part 1  measured FM-input parity: per-probe max-abs diff of the frame-origin
          twists the FM history consumes (root + each ee frame) and the assembled
          history token. Nonzero here => an input/convention mismatch (feeding bug).
  Part 2  FM-map parity: with the same fixed noise, ONNX-vs-torch on the IsaacLab
          token (export faithfulness) and ONNX(isaac token) vs ONNX(mujoco token)
          (how much the token diff alone moves the prediction). Plus the dumped
          prediction diff (combined).

Usage::

    python scripts/flow_matching/fm_probe_compare.py \
        --mujoco /tmp/fm_probe_mujoco.npz --isaac /tmp/fm_probe_isaac.npz \
        --fm_onnx logs/rsl_rl/g1/<run>/exported/fm.onnx \
        --ckpt logs/flow_matching/<run>/ckpt_<n>.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _mx(a: np.ndarray, b: np.ndarray) -> float:
    """Max abs difference between two arrays (broadcast)."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).max())


def main() -> int:
    """Load both dumps and print the Part 1 / Part 2 parity report."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mujoco", required=True)
    p.add_argument("--isaac", required=True)
    p.add_argument("--fm_onnx", default=None, help="fm.onnx for the Part 2 FM-map re-run (optional).")
    p.add_argument("--ckpt", default=None, help="FM .pt for the torch faithfulness check (optional).")
    p.add_argument("--tol", type=float, default=1e-3, help="Flag threshold for the Part 1 table.")
    args = p.parse_args()

    m = np.load(args.mujoco, allow_pickle=True)
    iz = np.load(args.isaac, allow_pickle=True)
    steps = m["steps"]
    assert np.array_equal(steps, iz["steps"]), "probe steps differ between dumps"
    if _mx(m["noise"], iz["noise"]) > 0:
        print("[WARN] dumps used different noise; Part 2 prediction diffs will include noise.")

    # ---- Part 1: measured FM-input parity (per probe) ----
    print("\n=== Part 1: measured FM-input parity (MuJoCo vs IsaacLab) ===")
    qkeys = ["cur_root_lin", "cur_root_ang", "cur_ee_pos", "cur_ee_quat", "cur_ee_lin", "cur_ee_ang", "cond"]
    hdr = "step  " + "".join(f"{k.replace('cur_',''):>11s}" for k in qkeys)
    print(hdr)
    worst = {k: 0.0 for k in qkeys}
    for i, s in enumerate(steps):
        cells = []
        for k in qkeys:
            e = _mx(m[k][i], iz[k][i])
            worst[k] = max(worst[k], e)
            cells.append(f"{e:>11.1e}")
        print(f"{int(s):<5d} " + "".join(cells))
    flagged = [k for k, e in worst.items() if e > args.tol]
    print(f"worst " + "".join(f"{worst[k]:>11.1e}" for k in qkeys))
    print(f"--> {'FEEDING MISMATCH in ' + ', '.join(flagged) if flagged else 'feeding matches (all <= tol)'}"
          f"  [tol={args.tol:.0e}]")

    # ---- Part 2: FM-map parity ----
    print("\n=== Part 2: FM-map parity (shared noise) ===")
    print(f"dumped pred  mujoco-vs-isaac : {_mx(m['pred'], iz['pred']):.2e}  (combined token + FM-impl)")

    if args.fm_onnx:
        sys.path.insert(0, os.path.join(ROOT, "transfer"))
        from deploy.fm_session import load_fm_context as load_onnx  # noqa: E402
        octx = load_onnx(args.fm_onnx)
        noise = m["noise"].astype(np.float32)
        N, cmd_steps = octx.N, list(octx.cmd_steps)

        def _known(f: int) -> np.ndarray:
            k = np.zeros((1, f), dtype=np.float32)
            k[:, :N] = 1.0
            return k

        def onnx_pred(cond: np.ndarray, fut: np.ndarray) -> np.ndarray:
            feed = {"cond": cond[None].astype(np.float32), "known": _known(cond.shape[0]),
                    "cmd_tokens": fut[cmd_steps][None].astype(np.float32), "noise": noise}
            return octx.sess.run(["trajectory"], feed)[0][0]

        tw = None
        if args.ckpt:
            sys.path.insert(0, os.path.join(ROOT, "scripts", "flow_matching"))
            import torch  # noqa: E402
            from export_fm_onnx import _FmSamplerOnnx  # noqa: E402
            from robot_rl.flow_matching.online import load_fm_context as load_torch  # noqa: E402
            wrap = _FmSamplerOnnx(load_torch(args.ckpt, device="cpu")).eval()

            def torch_pred(cond: np.ndarray, fut: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return wrap(torch.tensor(cond[None], dtype=torch.float32),
                                torch.tensor(_known(cond.shape[0]), dtype=torch.float32),
                                torch.tensor(fut[cmd_steps][None], dtype=torch.float32),
                                torch.tensor(noise, dtype=torch.float32)).numpy()[0]
            tw = torch_pred

        faith, tok_impact = 0.0, 0.0
        for i in range(len(steps)):
            fut_i = iz["future_cmd"][i]
            o_iz = onnx_pred(iz["cond"][i], fut_i)
            o_mj = onnx_pred(m["cond"][i], m["future_cmd"][i])
            tok_impact = max(tok_impact, _mx(o_iz, o_mj))
            if tw is not None:
                faith = max(faith, _mx(o_iz, tw(iz["cond"][i], fut_i)))
        if tw is not None:
            print(f"onnx-vs-torch (isaac cond)  : {faith:.2e}  (FM export faithful if ~1e-4)")
        print(f"onnx(isaac)-vs-onnx(mujoco)  : {tok_impact:.2e}  (prediction impact of the cond diff)")
    else:
        print("(pass --fm_onnx [--ckpt] to re-run the FM map on both tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
