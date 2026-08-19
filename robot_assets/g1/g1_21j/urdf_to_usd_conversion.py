"""Convert G1 21-joint URDF to USD using the Isaac Lab v3 URDF importer.

Usage:
    python urdf_to_usd_conversion.py

Notes:
    - The v3 importer auto-generates output at {usd_dir}/{robot_name}/{robot_name}.usda
      For g1_21j_merged.urdf this will be: g1_21j/g1_21j_merged/g1_21j_merged.usda
    - merge_fixed_joints is handled as a URDF XML pre-processing step
    - self_collision is a direct config parameter
    - For floating-base robots (fix_base=False), ArticulationRootAPI placement is handled
      by the importer's asset transformer pipeline
    - After running, update the usd_path in assets/robots/g1_21j.py to point to the new output
"""

import argparse
import os

# Isaac Sim app must be launched before importing converter
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert G1 URDF to USD.")
# parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode.")
parser.add_argument(
    "--urdf", default="g1_21j_box_merged_29j.urdf",
    help="URDF to convert, relative to this script's directory (or an absolute path). "
         "The importer names the output directory after the file, so a new URDF lands in "
         "its own tree and never overwrites the live asset.",
)
parser.add_argument(
    "--usd_dir", default=None,
    help="Where the output tree goes. Defaults to the URDF's OWN directory, which is also "
         "where the converter's cache lives -- see the note below on why that matters.",
)
parser.add_argument(
    "--force", action="store_true",
    help="Reconvert even if the cache thinks nothing changed.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
urdf_path = args.urdf if os.path.isabs(args.urdf) else os.path.join(SCRIPT_DIR, args.urdf)

# The cache -- config.yaml and .asset_hash -- lives in USD_DIR, not in the per-robot output
# subdirectory, so every URDF converted into one usd_dir shares a single hash. Converting a
# second robot there compares it against the FIRST one's hash and can decide nothing changed.
# Defaulting usd_dir to the URDF's own directory gives each robot its own cache. (The
# committed cache in robot_assets/g1/g1_21j/ still names a .claude/worktrees/teacher-refine
# path, which is what that failure looks like when it happens.)
usd_dir = os.path.abspath(args.usd_dir) if args.usd_dir else os.path.dirname(urdf_path)

cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    usd_dir=usd_dir,
    force_usd_conversion=args.force,
    fix_base=False,
    merge_fixed_joints=False,
    make_instanceable=False,
    self_collision=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=400.0,
            damping=40.0,
        ),
    ),
)

converter = UrdfConverter(cfg)
print(f"USD saved to: {converter.usd_path}")

# Post-process: remove the invalid root_joint from all USD payload files.
# The importer creates a FixedJoint "root_joint" with body0 pointing to the root
# Xform (not a rigid body), which causes NaN in PhysX for floating-base robots.
if not cfg.fix_base:
    import pathlib
    import re

    pattern = re.compile(
        r'\n\s*(?:def PhysicsFixedJoint|over)\s+"root_joint"[^{]*\{[^}]*\}',
    )
    for usda in pathlib.Path(converter.usd_path).parent.rglob("payloads/**/*.usda"):
        text = usda.read_text()
        new_text = pattern.sub("", text)
        if new_text != text:
            usda.write_text(new_text)
            print(f"Removed root_joint from {usda}")

simulation_app.close()
