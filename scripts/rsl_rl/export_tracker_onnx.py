# Copyright 2026 Zachary Olkin. All rights reserved.

"""Export a trained tracker policy to a self-contained ONNX with embedded metadata.

The exported tracker ``.pt`` (TorchScript from ``play.py``) already bakes in the
observation normalizer (raw ``obs`` -> ``actions``), so this re-exports it to ONNX
verbatim and embeds the deployment parameters (the contents of
``policy_parameters.yaml``) into the ONNX ``metadata_props`` under key
``policy_parameters``. The deployment runtime (``transfer/deploy``) then needs
only the single ``.onnx`` -- no separate YAML.

Usage:
    python scripts/rsl_rl/export_tracker_onnx.py --run_dir logs/rsl_rl/g1/<run>
    # reads <run>/exported/{policy.pt, policy_parameters.yaml}
    # writes <run>/exported/tracker.onnx
"""

from __future__ import annotations

import argparse
import json
import os

import onnx
import torch
import yaml


def export_tracker_onnx(exported_dir: str, opset: int = 18) -> dict:
    """Export ``policy.pt`` -> ``tracker.onnx`` and embed ``policy_parameters``.

    Args:
        exported_dir: Directory holding ``policy.pt`` + ``policy_parameters.yaml``.
        opset: ONNX opset version.

    Returns:
        The embedded parameter dict.
    """
    pt_path = os.path.join(exported_dir, "policy.pt")
    yaml_path = os.path.join(exported_dir, "policy_parameters.yaml")
    out_path = os.path.join(exported_dir, "tracker.onnx")
    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    policy = torch.jit.load(pt_path, map_location="cpu").eval()
    num_obs = int(params["num_obs"])
    dummy = torch.zeros(1, num_obs)
    torch.onnx.export(
        policy, dummy, out_path,
        export_params=True, opset_version=opset, dynamo=False,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "N"}, "actions": {0: "N"}},
    )

    model = onnx.load(out_path)
    entry = onnx.StringStringEntryProto(key="policy_parameters", value=json.dumps(params))
    model.metadata_props.append(entry)
    onnx.save(model, out_path)
    print(f"[INFO] Wrote {out_path} (num_obs={num_obs}, num_actions={params['num_actions']})")
    return params


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Export a tracker policy.pt to ONNX with metadata.")
    p.add_argument("--run_dir", required=True, help="Run directory (uses <run_dir>/exported).")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()
    export_tracker_onnx(os.path.join(args.run_dir, "exported"), args.opset)


if __name__ == "__main__":
    main()
