# Copyright 2026 Zachary Olkin. All rights reserved.

"""View an arbitrary USD/USDA/USDC/USDZ asset in Isaac Sim, with no robot and no task.

Intended for eyeballing scene scans (e.g. a Scaniverse ``.usdz``) before deciding whether
they are worth turning into terrain. The asset is REFERENCED into a fresh Z-up stage rather
than opened as the root layer, so the up-axis conversion below is explicit and the source
file is never modified.

Scanner exports are almost always Y-up (the USDZ convention), while Isaac Lab stages are
Z-up, so ``--up auto`` reads ``upAxis`` off the source stage and rotates +90 deg about X
when it says Y. That maps (x, y, z)_Yup -> (x, -z, y)_Zup.

Seeing the MESH rather than the photo texture, in the GUI (no relaunch needed):

* ``Shift+W`` toggles wireframe. Same as the viewport's ``Render`` menu -> ``Wireframe``.
* ``Render`` menu -> ``Disable Materials (White Mode)`` drops the scan's baked-in photo
  texture and shades the geometry plain white, which is the one that actually shows surface
  shape -- on a photogrammetry scan the texture hides how lumpy the mesh really is.
* The two compose, and ``--shading`` sets either or both at startup.

``--viz kit`` is REQUIRED to get a window: this Isaac Lab defaults to headless and only the
``kit`` visualizer implies a GUI. Without it the app starts, builds the stage and shows
nothing, which is indistinguishable from a failed load. ``--check`` is the deliberate
no-window path.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a USD asset in Isaac Sim.")
parser.add_argument("usd", type=str, help="Path to the .usd/.usda/.usdc/.usdz file.")
parser.add_argument(
    "--up",
    type=str,
    default="auto",
    choices=["auto", "Y", "Z"],
    help="Up-axis of the SOURCE asset. 'auto' reads it from the file's own upAxis metadata.",
)
parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale applied to the asset.")
parser.add_argument("--z_offset", type=float, default=0.0, help="Shift the asset along world z, in metres.")
parser.add_argument("--light", type=float, default=1000.0, help="Dome light intensity; 0 disables it.")
parser.add_argument(
    "--shading",
    type=str,
    default="textured",
    choices=["textured", "white", "wireframe", "wireframe_white"],
    help=(
        "Startup shading. 'white' drops the scan's photo texture so you see surface SHAPE;"
        " 'wireframe' overlays the triangles. Toggle either at any time in the GUI -- see"
        " the module docstring."
    ),
)
parser.add_argument(
    "--screenshot",
    type=str,
    default=None,
    help="Render to this PNG and exit, instead of opening an interactive window. Needs --viz kit.",
)
parser.add_argument(
    "--check",
    action="store_true",
    help="Build the stage, report the world bounds and exit. Use this to validate a file without a window.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import os
import sys

import carb.settings
import isaacsim.core.utils.stage as stage_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


_SETTLE_FRAMES = 60
"""Frames to pump before touching shading, and again before a capture, so RTX has converged."""


def _log(msg: str) -> None:
    """Print to stderr, because Kit captures stdout and swallows plain prints."""
    print(f"[view_usd] {msg}", file=sys.stderr, flush=True)


def _apply_shading(mode: str) -> None:
    """Force the viewport's shading mode, and keep the Render menu's checkmarks agreeing with it.

    The two RTX settings are what actually change the picture; the two ``/exts/`` settings are
    only what the Render menu reads to draw its ticks. Setting the RTX pair alone renders
    correctly but leaves the menu claiming "Default", so a later click there is a no-op that
    looks broken.
    """
    white = mode in ("white", "wireframe_white")
    wire = mode in ("wireframe", "wireframe_white")

    settings = carb.settings.get_settings()
    settings.set("/rtx/debugMaterialType", 0 if white else -1)
    settings.set("/rtx/wireframe/mode", (2 if white else 1) if wire else 0)
    settings.set("/exts/omni.kit.viewport.menubar.render/materialMode", "disabled" if white else "default")
    settings.set("/exts/omni.kit.viewport.menubar.render/shadingMode", "wireframe" if wire else "default")
    _log(f"shading: {mode} (rtx/wireframe/mode={settings.get('/rtx/wireframe/mode')}, "
         f"rtx/debugMaterialType={settings.get('/rtx/debugMaterialType')})")


def _screenshot(path: str) -> None:
    """Render a while, then write the viewport to ``path`` as a PNG."""
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    for _ in range(_SETTLE_FRAMES):
        simulation_app.update()
    capture_viewport_to_file(viewport, path)
    for _ in range(_SETTLE_FRAMES):
        simulation_app.update()
    _log(f"wrote {path}")


def main() -> None:
    """Reference the asset into a fresh Z-up stage and idle until the window is closed."""
    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd))
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"No such USD file: {usd_path}")

    # read the source up-axis WITHOUT composing it into our stage
    source_up = args_cli.up
    if source_up == "auto":
        source_up = str(UsdGeom.GetStageUpAxis(Usd.Stage.Open(usd_path)))
    _log(usd_path)
    _log(f"source up-axis: {source_up} -> stage up-axis: Z")

    stage_utils.create_new_stage()
    stage = stage_utils.get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # USD applies xformOpOrder LAST-LISTED-FIRST, so this standard translate/rotate/scale
    # listing scales first, then fixes the up-axis, then offsets in WORLD z. Listing the
    # translate last instead rotates and scales the offset itself -- silently, since the
    # stage still loads and only the placement is wrong.
    xform = UsdGeom.Xform.Define(stage, Sdf.Path("/World/Scan"))
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(args_cli.z_offset)))
    if source_up == "Y":
        xform.AddRotateXOp().Set(90.0)
    if args_cli.scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3f(args_cli.scale, args_cli.scale, args_cli.scale))
    xform.GetPrim().GetReferences().AddReference(usd_path)

    if args_cli.light > 0.0:
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(float(args_cli.light))

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    extent = bounds.ComputeWorldBound(xform.GetPrim()).ComputeAlignedRange()
    _log(f"world bbox min {extent.GetMin()} max {extent.GetMax()} size {extent.GetSize()}")

    if args_cli.check:
        _log("--check: the stage was built, exiting without a window")
        return

    # let the viewport and its menus build before forcing shading, or the Render menu's own
    # startup pass reasserts its defaults over ours
    for _ in range(_SETTLE_FRAMES):
        simulation_app.update()
    _apply_shading(args_cli.shading)

    if args_cli.screenshot is not None:
        _screenshot(os.path.abspath(os.path.expanduser(args_cli.screenshot)))
        return

    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
