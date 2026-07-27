# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate the Obelisk sensor configs (and the ZED's targets) from a deploy artifact.

The camera geometry a policy trained on is embedded in ``tracker.onnx``
(``policy_parameters.depth_camera_spec`` / ``height_scan_spec``) by ``mdp.depth_camera_specs``.
That contract is already what the MuJoCo sim2sim caster and the hardware controller read;
this writes the THIRD consumer's copy -- the Obelisk MuJoCo sim's ray-caster config -- from
the same source, so training, both sims and the real camera cannot drift apart.  Transcribing
30x26 and an intrinsic matrix by hand into a YAML is exactly how they drift.

Emitted per exteroception source:

* depth -> one ``scan_config`` file per camera (``pattern.type: depth_camera``), plus a
  ``zed_targets`` file holding what the Jetson node must be configured to produce;
* height scan -> one ``scan_config`` file (``pattern.type: height_scan``).

Usage::

    python scripts/rsl_rl/export_sensor_configs.py --run_dir logs/rsl_rl/g1/<run>
    python scripts/rsl_rl/export_sensor_configs.py --run_dir <run> --stdout   # preview

Obelisk takes these either inline as a ``scan_config:`` block on the sensor entry, or by
path via ``config_path:``.  Files are written because ``include:`` merges whole top-level
sections and cannot splice a fragment into a nested ``sensor_settings`` list -- so a
generated file referenced by ``config_path:`` is what keeps the generated part generated.
Isaac-free and torch-free.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(_REPO_ROOT, "transfer"))

from deploy.depth_obs import parse_depth_spec  # noqa: E402
from deploy.tracker_session import TrackerSession  # noqa: E402

#: MuJoCo geom groups the cameras cast against, matching
#: ``transfer/sim/depth_camera.MujocoDepthCameras``: the robot's per-link VISUAL meshes so
#: it occludes itself as in training, plus the terrain.  Group 0 (collision primitives) is
#: excluded there because the collision foot is a box, not the visual foot.
DEPTH_GEOM_GROUPS = [1, 2]

#: The height scan only ever wants the ground, never the robot.
SCAN_GEOM_GROUPS = [2]


def _fov_deg(pixels: int, focal: float, principal: float) -> float:
    """Full field of view of a pinhole axis, honouring an off-centre principal point.

    Args:
        pixels: Resolution along the axis.
        focal: Focal length in pixels (``fx`` / ``fy``).
        principal: Principal point along the axis (``cx`` / ``cy``).

    Returns:
        The total FOV in degrees -- the two half-angles summed, which differ when the
        principal point is not centred.
    """
    return math.degrees(math.atan(principal / focal) + math.atan((pixels - principal) / focal))


def depth_scan_config(cam: dict) -> dict:
    """The Obelisk ``scan_config`` reproducing one camera from the contract.

    Args:
        cam: One parsed camera contract entry.

    Returns:
        A ``scan_config`` mapping ready to serialise.
    """
    x, y, z, w = (float(v) for v in cam["offset_quat_xyzw"])
    return {
        # A site at the mount body's ORIGIN, with the whole camera transform carried in
        # pattern.offset below. The alternative -- baking the mount pose into the site --
        # would put the extrinsics in the MJCF, where they could drift from the contract
        # and would need a model edit every time the camera moves. This way the MJCF only
        # has to say where the body is, which it already knows.
        "site_name": f"{cam['mount_body']}_origin_site",
        "geom_groups": list(DEPTH_GEOM_GROUPS),
        "frame": "world",
        "pattern": {
            "type": "depth_camera",
            "width": int(cam["width"]),
            "height": int(cam["height"]),
            "intrinsic_matrix": [
                float(cam["fx"]), 0.0, float(cam["cx"]),
                0.0, float(cam["fy"]), float(cam["cy"]),
                0.0, 0.0, 1.0,
            ],
            # Obelisk quaternions are wxyz; the contract stores xyzw.
            "offset": {"pos": [float(v) for v in cam["offset_pos"]], "rot": [w, x, y, z]},
        },
    }


def zed_targets(cam: dict) -> dict:
    """What the Jetson ZED node must be configured to emit for this camera.

    The ZED publishes a downsampled, FOV-clipped depth image; these are the numbers that
    image has to match for the policy to see what it trained on.  Field NAMES here are
    ours, not ``zed_depth_publisher``'s -- map them onto its params when that node is
    vendored in, rather than guessing its schema now.

    Args:
        cam: One parsed camera contract entry.

    Returns:
        The target resolution, field of view, range and rate.
    """
    # Emitted at full precision deliberately: these are a geometry contract, and rounding
    # a field of view for looks is the same drift this generator exists to prevent. The
    # consumer can round to whatever its camera actually accepts.
    return {
        "width": int(cam["width"]),
        "height": int(cam["height"]),
        "hfov_deg": _fov_deg(int(cam["width"]), float(cam["fx"]), float(cam["cx"])),
        "vfov_deg": _fov_deg(int(cam["height"]), float(cam["fy"]), float(cam["cy"])),
        "min_range_m": float(cam["min_dist"]),
        "max_range_m": float(cam["max_dist"]),
        # A miss must arrive as the same sentinel a missed ray produces in sim; anything
        # non-finite or out of range normalises to it, so NaN is a safe encoding too.
        "invalid_range_m": float(cam["max_clip_val"]),
        "rate_hz": 1.0 / float(cam["update_period"]) if cam.get("update_period") else None,
    }


def height_scan_config(spec: dict) -> dict:
    """The Obelisk ``scan_config`` reproducing the height scan from the contract.

    Args:
        spec: The tracker's ``height_scan_spec``.

    Returns:
        A ``scan_config`` mapping ready to serialise.
    """
    return {
        "site_name": "height_scan_site",
        "geom_groups": list(SCAN_GEOM_GROUPS),
        "frame": "world",
        "pattern": {
            "type": "height_scan",
            # The RL grid pattern is x-major within a row; obelisk calls that "xy".
            "ordering": "xy",
            "resolution": float(spec["resolution"]),
            "size": [float(v) for v in spec["size"]],
            "offset": {"pos": [float(v) for v in spec["offset_pos"]]},
        },
    }


def _dump(obj, indent: int = 0) -> str:
    """Serialise a config mapping as YAML, without taking a PyYAML dependency.

    The structures here are small and closed (mappings, lists of numbers, scalars), so a
    direct writer keeps the output diff-friendly and ordered as written.

    Args:
        obj: The mapping to serialise.
        indent: Current indent depth, in levels of two spaces.

    Returns:
        The YAML text.
    """
    pad = "  " * indent
    out = []
    for key, value in obj.items():
        if isinstance(value, dict):
            out.append(f"{pad}{key}:")
            out.append(_dump(value, indent + 1))
        elif isinstance(value, list):
            out.append(f"{pad}{key}: [{', '.join(_scalar(v) for v in value)}]")
        else:
            out.append(f"{pad}{key}: {_scalar(value)}")
    return "\n".join(out)


def _scalar(v) -> str:
    """Render one scalar the way YAML wants it.

    Args:
        v: A string, number, bool or None.

    Returns:
        The YAML representation.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return repr(float(v)) if isinstance(v, float) else str(v)


def _header(source: str, what: str) -> str:
    """The do-not-edit banner every generated file carries.

    Args:
        source: The artifact the contract came from.
        what: One line naming what the file configures.

    Returns:
        The comment block.
    """
    return (
        "# GENERATED by scripts/rsl_rl/export_sensor_configs.py -- DO NOT EDIT.\n"
        f"# {what}\n"
        f"# Source of truth: {source}\n"
        "# Regenerate after any change to the camera configuration, and re-export the\n"
        "# policy: the geometry here MUST match what the networks were trained on.\n"
    )


def _describe_path(path: str) -> str:
    """A readable provenance string: repo-relative inside the repo, absolute outside it.

    Args:
        path: Absolute path to the source artifact.

    Returns:
        The path as it should appear in the generated file's header.
    """
    rel = os.path.relpath(path, _REPO_ROOT)
    return path if rel.startswith(os.pardir) else rel


def generate(exported_dir: str) -> dict[str, str]:
    """Build every sensor config this artifact implies.

    Args:
        exported_dir: Directory holding ``tracker.onnx``.

    Returns:
        Mapping of output filename to file contents.

    Raises:
        FileNotFoundError: If ``tracker.onnx`` is absent.
        ValueError: If the artifact declares no exteroception at all.
    """
    tracker = os.path.join(exported_dir, "tracker.onnx")
    if not os.path.exists(tracker):
        raise FileNotFoundError(f"No tracker.onnx in {exported_dir}; run export_deploy_onnx.py.")
    session = TrackerSession(tracker)
    src = _describe_path(tracker)

    cams = parse_depth_spec(session.get_depth_camera_spec(), "tracker.onnx")
    if cams:
        files = {}
        for cam in cams:
            name = str(cam["obs_term"])
            files[f"{name}.yml"] = (
                _header(src, f"Obelisk ray-caster config for the '{name}' depth camera.")
                + _dump(depth_scan_config(cam)) + "\n"
            )
        files["zed_targets.yml"] = (
            _header(src, "What the Jetson ZED node must emit, per camera.")
            + "\n".join(f"{c['obs_term']}:\n" + _dump(zed_targets(c), 1) for c in cams) + "\n"
        )
        return files

    scan = session.get_height_scan_spec()
    if scan:
        return {"height_scan.yml": (
            _header(src, "Obelisk ray-caster config for the height scan.")
            # The deploy subtracts sensor_offset to turn a hit into clearance; whether
            # ObkScan already reports clearance is a sim-side question, so it is verified
            # against the MuJoCo caster rather than assumed here.
            + f"# NOTE: the policy expects CLEARANCE with sensor_offset={scan['sensor_offset']} "
              "removed;\n#       confirm ObkScan's convention against transfer/sim before "
              "trusting this.\n"
            + _dump(height_scan_config(scan)) + "\n"
        )}

    raise ValueError(
        f"{src} declares neither depth cameras nor a height scan, so there is no sensor "
        "to configure. A blind policy needs no exteroception config."
    )


def main() -> None:
    """Write (or preview) the Obelisk sensor configs for a run's exported artifacts."""
    parser = argparse.ArgumentParser(
        description="Generate Obelisk sensor configs from a tracker.onnx camera contract.")
    parser.add_argument("--run_dir", required=True, help="Run directory; reads <run_dir>/exported.")
    parser.add_argument("--out_dir", default=os.path.join(
        "transfer", "obelisk", "g1_control", "config", "scan"),
        help="Where to write the generated configs (repo-relative).")
    parser.add_argument("--stdout", action="store_true", help="Print instead of writing.")
    args = parser.parse_args()

    files = generate(os.path.join(os.path.abspath(args.run_dir), "exported"))
    if args.stdout:
        for name, text in files.items():
            print(f"########## {name} ##########\n{text}")
        return

    out_dir = os.path.join(_REPO_ROOT, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    for name, text in files.items():
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(text)
        print(f"[INFO] Wrote {os.path.relpath(path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
