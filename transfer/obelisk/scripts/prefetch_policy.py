#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Warm the on-robot policy cache while the machine still has a route out.

The controller fetches ``tracker.onnx`` + ``fm.onnx`` OFFLINE-FIRST inside its lifecycle
``on_configure``: a cache hit never touches the network, and a miss is bounded and
retried. That is the right behaviour on a robot LAN with no DNS, but it means a cold cache
turns into a bring-up failure at exactly the wrong moment. Run this on the onboard machine
while it still has internet, before you go near the robot.

    python3 transfer/obelisk/scripts/prefetch_policy.py --hf_policy_folder <name>
    python3 transfer/obelisk/scripts/prefetch_policy.py --hf_policy_folder <name> --refresh

``--refresh`` re-downloads over a cache hit, which is what you want after re-pushing a
policy to the SAME folder -- otherwise the robot keeps running the old one, silently.

The fetched pair is loaded before this exits, so a tracker/generator that disagree about
the depth cameras are caught here rather than on the robot.

That load runs under the CONTROLLER's OWN rules, read from the config fragment every launch
file includes -- currently just ``allow_robot_asset_mismatch``, which lives there and
nowhere else. Otherwise this script can be STRICTER than the node it is warming a cache
for, which it was: the config allowed one known-benign ``mjcf_sha256`` delta (an actuator
gain line the generator never reads) while this refused it, and a validator refusing what
the robot will happily arm reads as a broken policy rather than as a two-place switch.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_OBELISK_ROOT = os.path.dirname(_HERE)
_TRANSFER = os.path.dirname(_OBELISK_ROOT)
sys.path.insert(0, _TRANSFER)

from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.hardware import DEFAULT_HF_REPO_ID, fetch_artifacts  # noqa: E402

#: Where the controller looks; must equal ``g1_control.controller.HF_CACHE_DIR``. Both are
#: built from the same anchor (``<repo>/transfer``) in the same way, which is the only reason
#: they can be trusted to agree -- importing the constant from here would drag in rclpy and
#: the obelisk ROS stack, which this script deliberately does not need.
DEFAULT_CACHE = os.path.join(_OBELISK_ROOT, "g1_control", "resource", "hf_cache")

#: The shared low-level fragment. It is a FRAGMENT and cannot be launched on its own, which is
#: exactly why it is the right file to read: ``g1.yml``, ``g1_cpp.yml`` and ``g1_debug_clip.yml``
#: all include it, so one answer here covers every launch path.
DEFAULT_CONTROLLER_CONFIG = os.path.join(_OBELISK_ROOT, "g1_control", "config", "common",
                                         "low_level.yml")


def controller_params(config_path: str) -> dict:
    """Read the low-level controller node's ROS params out of its config fragment.

    Parsed as plain YAML rather than through obelisk's loader: this script deliberately
    imports no rclpy, and nothing here needs the include/keyed-merge semantics -- the params
    this reads are declared in the fragment itself.

    Args:
        config_path: Config fragment declaring the ``low_level`` control node.

    Returns:
        That node's ``params`` mapping, or ``{}`` if the file, the ``control`` list or the
        entry is absent. An unreadable config therefore falls back to the STRICT defaults,
        which is the safe direction for a validator to fail in.
    """
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    except OSError:
        return {}
    for entry in cfg.get("control") or []:
        if isinstance(entry, dict) and entry.get("name") == "low_level":
            return entry.get("params") or {}
    return {}


def main() -> None:
    """Download and validate the policy pair into the controller's cache."""
    parser = argparse.ArgumentParser(description="Pre-warm the on-robot policy cache.")
    parser.add_argument("--hf_repo_id", default=DEFAULT_HF_REPO_ID,
                        help=f"HuggingFace repo id; must match the controller config. "
                             f"Defaults to {DEFAULT_HF_REPO_ID}.")
    parser.add_argument("--hf_policy_folder", required=True,
                        help="Folder within the repo, as set in the controller config.")
    parser.add_argument("--cache_root", default=DEFAULT_CACHE,
                        help="Cache directory (defaults to the one the controller reads).")
    parser.add_argument("--controller_config", default=DEFAULT_CONTROLLER_CONFIG,
                        help="Config fragment to take the controller's own load-time rules "
                             "from (currently allow_robot_asset_mismatch). Defaults to the "
                             "fragment every launch file includes.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download even on a cache hit (after re-pushing a policy).")
    args = parser.parse_args()

    # This script's whole job is to go online, so the offline-first default is lifted here
    # rather than in the library -- the controller must keep it.
    os.environ["HF_HUB_OFFLINE"] = "0"

    print(f"[INFO] Fetching {args.hf_repo_id}/{args.hf_policy_folder} -> {args.cache_root}")
    tracker, fm = fetch_artifacts(args.hf_repo_id, args.hf_policy_folder,
                                  os.path.abspath(args.cache_root), refresh=args.refresh)
    for path in (tracker, fm):
        print(f"[INFO]   {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    params = controller_params(args.controller_config)
    allow_mismatch = bool(params.get("allow_robot_asset_mismatch", False))
    if allow_mismatch:
        print(f"[INFO] allow_robot_asset_mismatch: true, from {args.controller_config} -- a "
              "fingerprint mismatch WARNS below instead of refusing, exactly as it will on the "
              "robot. Read the fields it names: only an actuator-gain-only mjcf_sha256 delta "
              "is benign.")
    policy = GenTrackerPolicy(tracker, fm, allow_robot_asset_mismatch=allow_mismatch)
    policy.load()
    spec = policy.get_depth_camera_spec()
    extero = (", ".join(f"{c.get('sensor') or c['obs_term']} {int(c['width'])}x{int(c['height'])}"
                        for c in spec) if spec
              else f"height scan ({policy.ctx.extero_sensor_dim} rays)" if policy.needs_height_scan()
              else "none")
    print(f"[INFO] Loaded OK: obs={policy.get_num_obs()} act={policy.get_num_actions()} "
          f"dt={policy.get_dt():.4f}s")
    print(f"[INFO] Exteroception the robot must supply: {extero}")
    print(f"[INFO] Joints ({len(policy.get_joint_names())}): {policy.get_joint_names()}")
    print("[INFO] Cache warm. The controller will now start without touching the network.")


if __name__ == "__main__":
    main()
