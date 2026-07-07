# Copyright 2026 Zachary Olkin. All rights reserved.

"""Export BOTH sim2sim deploy ONNX artifacts (RL tracker + FM generator) in one step.

The MuJoCo generator+tracker runner (``transfer/sim/gen_tracker_runner.py``) loads two
ONNX files from an RL run's ``exported/`` directory:

* ``tracker.onnx`` -- the RL policy, re-exported from ``<run>/exported/policy.pt`` (the
  TorchScript ``play.py`` produces) with ``policy_parameters.yaml`` embedded.
* ``fm.onnx`` -- the flow-matching generator, exported from an FM ``.pt`` checkpoint.

This wraps the two existing exporters (``export_tracker_onnx`` + ``export_fm_onnx``) so a
single command produces both, writing them where the runner expects. It is torch-only
(no Isaac Sim). Run ``play.py`` first to (verify and) produce ``<run>/exported/policy.pt``
from the RL checkpoint you want to deploy, then::

    PYTHONPATH=$PWD/source/robot_rl:$PYTHONPATH python scripts/rsl_rl/export_deploy_onnx.py \\
        --run_dir logs/rsl_rl/g1/<run> \\
        --fm_checkpoint logs/flow_matching/<fm_run>/best.pt

Then deploy::

    cd transfer/sim && python gen_tracker_runner.py --mode joystick \\
        --scene terrain_scene --load_run <run>
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Import the two single-artifact exporters from their respective script dirs.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))                 # scripts/rsl_rl
_FM_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, "flow_matching"))
for _d in (_THIS_DIR, _FM_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
from export_tracker_onnx import export_tracker_onnx  # noqa: E402
from export_fm_onnx import export_fm_onnx  # noqa: E402


def _warn_if_stale_policy(run_dir: str, exported_dir: str) -> None:
    """Warn if ``policy.pt`` predates the newest ``model_*.pt`` (a stale jit export).

    ``tracker.onnx`` re-exports ``policy.pt`` verbatim, so if a newer RL checkpoint was
    trained after the last ``play.py`` export, the tracker ONNX would reflect the OLDER
    policy. Point this out so the user re-runs ``play.py`` on the newest checkpoint.
    """
    pt = os.path.join(exported_dir, "policy.pt")
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not ckpts or not os.path.exists(pt):
        return
    newest = max(ckpts, key=os.path.getmtime)
    if os.path.getmtime(newest) > os.path.getmtime(pt) + 1.0:
        print(f"[WARN] {os.path.basename(pt)} is OLDER than {os.path.basename(newest)}: the "
              "tracker ONNX will reflect the older checkpoint. Re-run play.py on the newest "
              "checkpoint first to refresh policy.pt if you want the newer policy.")


def main() -> None:
    """CLI entry point: export tracker.onnx + fm.onnx into ``<run_dir>/exported``."""
    p = argparse.ArgumentParser(
        description="Export tracker.onnx + fm.onnx into <run>/exported for sim2sim deploy."
    )
    p.add_argument("--run_dir", required=True,
                   help="RL run directory; uses <run_dir>/exported for policy.pt and outputs.")
    p.add_argument("--fm_checkpoint", required=True,
                   help="Flow-matching .pt checkpoint to export to fm.onnx.")
    p.add_argument("--fm_out", default=None,
                   help="Override the fm.onnx output path (default: <run_dir>/exported/fm.onnx).")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    exported_dir = os.path.join(args.run_dir, "exported")
    policy_pt = os.path.join(exported_dir, "policy.pt")
    params_yaml = os.path.join(exported_dir, "policy_parameters.yaml")
    if not (os.path.exists(policy_pt) and os.path.exists(params_yaml)):
        run_name = os.path.basename(os.path.normpath(args.run_dir))
        raise FileNotFoundError(
            f"Missing {policy_pt} and/or policy_parameters.yaml. Run play.py on the RL "
            "checkpoint first to export the TorchScript policy + parameters, e.g.:\n"
            f"  python scripts/rsl_rl/play.py --task <tracker task> --load_run {run_name} "
            "--fm_checkpoint <fm.pt> --clip_device cpu"
        )
    if not os.path.exists(args.fm_checkpoint):
        raise FileNotFoundError(f"FM checkpoint not found: {args.fm_checkpoint}")

    _warn_if_stale_policy(args.run_dir, exported_dir)

    fm_out = args.fm_out or os.path.join(exported_dir, "fm.onnx")

    print("[1/2] RL tracker policy -> tracker.onnx")
    export_tracker_onnx(exported_dir, args.opset)
    print("[2/2] Flow-matching generator -> fm.onnx")
    export_fm_onnx(args.fm_checkpoint, fm_out, args.opset)

    run_name = os.path.basename(os.path.normpath(args.run_dir))
    print(f"\n[INFO] Deploy artifacts ready:")
    print(f"[INFO]   {os.path.join(exported_dir, 'tracker.onnx')}   (from policy.pt)")
    print(f"[INFO]   {fm_out}   (from {args.fm_checkpoint})")
    print("[INFO] Sim2sim: cd transfer/sim && python gen_tracker_runner.py "
          f"--mode joystick --scene terrain_scene --load_run {run_name}")


if __name__ == "__main__":
    main()
