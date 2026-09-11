# Copyright 2026 Zachary Olkin. All rights reserved.

"""Generate the Obelisk sensor configs (and the ZED's targets) from a deploy artifact.

The camera geometry a policy trained on is embedded in ``tracker.onnx``
(``policy_parameters.depth_camera_spec`` / ``height_scan_spec``) by ``mdp.depth_camera_specs``.
That contract is already what the MuJoCo sim2sim caster and the hardware controller read;
this writes the THIRD consumer's copy -- the Obelisk MuJoCo sim's ray-caster config -- from
the same source, so training, both sims and the real camera cannot drift apart.  Transcribing
30x26 and an intrinsic matrix by hand into a YAML is exactly how they drift.

These are generated from the TRACKER's contract, and the tracker's cameras are no longer the
whole rig: the policy reads the near camera alone while the generator reads both. A pair like
that is REFUSED here (see :func:`generate`) rather than emitted as a config set covering half
the deployment -- the ROS/C++ runtime has no subset wiring, so the generator's extra camera
would read as zeros with nothing raising. A refusal is the deliverable; support is not.

Two consequences of that, each of which is a way the refusal used to be reachable around:

* the generator must be EXPORTED BESIDE the tracker, because it is the only artifact naming
  the whole rig. With no ``fm.onnx`` there is no second contract to compare against, so the
  subset check simply did not run and half a rig went out unremarked; an absent generator is
  now itself the refusal.
* every write PRUNES the generated files an earlier run left in the output directory. The
  controller reads that directory by fixed name, so a stale ``depth_far.yml`` beside a fresh
  ``depth_near.yml`` is a complete-looking set describing two different policies' cameras.

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
        # Ray drawing OFF, emitted rather than left to obelisk's default. Both generated files
        # carried `viz: false` by hand, and regenerating them silently dropped it -- which is
        # the trap a generated config that still needs hand-editing always is. There are
        # 30x26 rays per camera and two cameras, so this is not a free setting. Flip it here
        # (or in a local edit you expect to lose) when you want to see the cones.
        "viz": False,
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


#: First line of every file this script writes.  The prune in :func:`main` deletes only
#: files carrying it, so a hand-written config sharing the output directory is never touched.
_BANNER = "# GENERATED by scripts/rsl_rl/export_sensor_configs.py -- DO NOT EDIT.\n"


def _header(source: str, what: str) -> str:
    """The do-not-edit banner every generated file carries.

    Args:
        source: The artifact the contract came from.
        what: One line naming what the file configures.

    Returns:
        The comment block.
    """
    return (
        _BANNER
        + f"# {what}\n"
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
        ValueError: If the artifact declares no exteroception at all; if a depth tracker has
            no generator beside it to establish that its cameras are the whole rig; or if the
            generator that is there reads a camera the tracker does not (a SUBSET pair) or
            samples a shared lens on a different grid (a SPLIT sampling) -- none of which the
            hardware path can express, so all are refused rather than half-emitted.
    """
    tracker = os.path.join(exported_dir, "tracker.onnx")
    if not os.path.exists(tracker):
        raise FileNotFoundError(f"No tracker.onnx in {exported_dir}; run play.py to export it.")
    session = TrackerSession(tracker)
    src = _describe_path(tracker)

    cams = parse_depth_spec(session.get_depth_camera_spec(), "tracker.onnx")
    # These configs describe ONE stream per camera -- a ray-caster grid for the MuJoCo world
    # node and a ZED target for the Jetson -- and they are generated from the TRACKER's
    # contract. Three ways that stops describing the deployment, all refused below rather
    # than half-emitted, because a missing stream is a camera the robot never publishes and
    # the controller then waits on forever:
    #
    #   MISSING -- no generator beside the tracker at all, so nothing can say whether the
    #              tracker's cameras are the whole rig. "Cannot say" used to be read as
    #              "nothing to check": the check below was gated on the fm.onnx existing, so
    #              its absence skipped it and emitted the tracker's half of the rig in
    #              silence -- the SUBSET failure, reached through a missing file.
    #   SUBSET  -- the generator reads a camera the policy does not. That is now the standard
    #              training shape (the policy reads the near camera, the generator both), and
    #              the ROS/C++ side is not wired for it: g1_deploy sizes the generator's
    #              extero row from fm_meta but fills it by looping the TRACKER's cameras, so
    #              the missing camera's plane would stay 0.0 with nothing raising.
    #   SPLIT   -- the generator samples the same lens on its OWN grid, which needs a second
    #              stream there is nowhere here to declare. (sim2sim casts both because it
    #              can; hardware has never supported it.)
    #
    # Exports are named for their FM checkpoint, so a run dir can hold several. Refusing an
    # ambiguous one (resolve_fm_onnx raises) is right here: these are HARDWARE configs, and
    # describing whichever generator happened to be newest is how they would go out wrong.
    from deploy.artifacts import resolve_fm_onnx
    fm_onnx = resolve_fm_onnx(exported_dir, None, required=False)
    if cams and fm_onnx is None:
        raise ValueError(
            f"no generator in {exported_dir}, so nothing here can say whether tracker.onnx's "
            f"camera(s) {[str(c.get('sensor', c['obs_term'])) for c in cams]} are the whole "
            "rig. They are not, in the shipped configuration: the policy is conditioned on "
            "the NEAR camera alone and the generator reads both, so these configs would "
            "declare half the deployment's cameras -- and a camera no config declares is one "
            "the robot never publishes, which the controller then waits on forever. Export "
            "the generator into this directory (play.py --fm_checkpoint, or "
            "scripts/flow_matching/export_fm_onnx.py) and re-run. There is nothing to work "
            "around: these files describe the PAIR, and half of it is not here."
        )
    if cams:
        from deploy.depth_obs import index_cameras_by_key
        from deploy.fm_session import load_fm_context
        from deploy.scan_obs import parse_extero_spec
        fm_ctx = load_fm_context(fm_onnx)
        # Parsed by the checkpoint's own SOURCE: fm_meta.extero_spec holds cameras for one
        # generator and a ray grid for another, and reading a scan as depth fails on a
        # missing fx. A scan generator has no camera set to compare against the tracker's,
        # so there is nothing to refuse here (it is refused on the deploy path instead).
        fm_cams = (parse_extero_spec(fm_ctx.extero_spec, fm_ctx.extero_source, "fm.onnx")
                   if fm_ctx.extero_source == "depth" else [])
        if fm_cams:
            # By NAME, never by index: with the two artifacts listing different cameras,
            # position i no longer names one lens, and comparing the resolution LISTS would
            # report a split that is really a subset.
            trk_by_key = index_cameras_by_key(cams, "tracker.onnx")
            fm_by_key = index_cameras_by_key(fm_cams, "fm.onnx")
            missing = [k for k in fm_by_key if k not in trk_by_key]
            if missing:
                raise ValueError(
                    f"fm.onnx conditions on depth camera(s) {missing} that tracker.onnx does "
                    f"not read (it reads {sorted(trk_by_key)}). These configs are generated "
                    "from the TRACKER's contract, so emitting them would give the robot a "
                    "sensor set covering only part of what the deployed pair needs -- and the "
                    "obelisk/ZED and C++ runtime have no wiring for a subset pair, so the "
                    "generator's extra camera would silently read as zeros rather than fail. "
                    "This is a refusal, not a limitation to work around: wire the subset pair "
                    "through transfer/obelisk and transfer/deploy_cpp first. MuJoCo sim2sim "
                    "handles it already and needs none of these files."
                )
            split = {
                k: ((int(trk_by_key[k]["width"]), int(trk_by_key[k]["height"])),
                    (int(c["width"]), int(c["height"])))
                for k, c in fm_by_key.items()
                if (int(trk_by_key[k]["width"]), int(trk_by_key[k]["height"]))
                != (int(c["width"]), int(c["height"]))
            }
            if split:
                raise ValueError(
                    "tracker.onnx and fm.onnx sample the same lens on different grids "
                    + ", ".join(f"{k}: {t} vs {f}" for k, (t, f) in split.items())
                    + ". These configs carry one stream per camera, so there is nowhere to "
                    "declare the generator's -- the obelisk/ZED path has no split-sampling "
                    "wiring yet. Re-train the generator at the policy's resolution before "
                    "deploying to hardware; MuJoCo sim2sim handles the split and needs none "
                    "of these files."
                )
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


def prune_stale(out_dir: str, keep: set[str]) -> list[str]:
    """Delete configs a PREVIOUS run of this script wrote that this one did not.

    The output directory is a fixed path the controller reads by name, so a file left behind
    is not inert: with the tracker's cameras no longer the whole rig, a run that emits only
    ``depth_near.yml`` beside an older run's ``depth_far.yml`` describes the near camera of
    one policy and the far camera of ANOTHER -- a pair of files that looks complete, is
    internally plausible, and aims a camera using a mount pose the deployed networks never
    trained on.  Only files carrying :data:`_BANNER` are removed, so a hand-written config in
    the same directory is never touched.

    Args:
        out_dir: Directory the configs were written to.
        keep: Filenames this run produced.

    Returns:
        The filenames removed, in sorted order.
    """
    removed = []
    for name in sorted(os.listdir(out_dir)):
        if name in keep or not name.endswith((".yml", ".yaml")):
            continue
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            if f.readline() != _BANNER:
                continue
        os.remove(path)
        removed.append(name)
    return removed


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
    for name in prune_stale(out_dir, set(files)):
        print(f"[INFO] Removed stale {os.path.relpath(os.path.join(out_dir, name), _REPO_ROOT)}"
              " -- generated by an earlier run and not by this artifact.")


if __name__ == "__main__":
    main()
