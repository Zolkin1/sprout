# Copyright 2026 Zachary Olkin. All rights reserved.

"""Visualize a flow-matching recording in viser (no Isaac Sim required).

Loads a recording produced by ``scripts/flow_matching/record_viz.py`` (the
newest one in ``logs/flow_matching/viz/`` by default) and serves an
interactive 3D scene in the browser: terrain, every recorded robot, and
per-robot prediction overlays (ghost robots fading with the future horizon,
the predicted-root curve, and height-scan points), with play/pause, frame
stepping (click the frame slider, then use the arrow keys), and a frame
number readout.

Examples:
    # View the newest recording:
    python scripts/flow_matching/visualize.py

    # View a specific recording on another port:
    python scripts/flow_matching/visualize.py --data logs/flow_matching/viz/viz_<ts>.npz --port 8081
"""

from __future__ import annotations

import argparse
import os

from robot_rl.flow_matching.viz import FlowMatchingViewer, find_newest_recording, load_viz_recording

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def resolve_urdf_path(cli_urdf: str | None, recorded_urdf: str) -> str:
    """Pick the URDF to render: CLI override, else the recorded path.

    A relative recorded path that does not exist from the cwd is retried
    relative to the repo root (recordings store repo-relative paths).
    """
    path = cli_urdf or recorded_urdf
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_REPO_ROOT, path)
        if os.path.exists(candidate):
            return candidate
    if not os.path.exists(path):
        raise FileNotFoundError(f"URDF not found: {path} (override with --urdf)")
    return path


def main() -> None:
    """Load the recording and serve the viser viewer until interrupted."""
    parser = argparse.ArgumentParser(description="Visualize a flow-matching recording in viser.")
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to a viz_*.npz recording (default: newest in logs/flow_matching/viz).",
    )
    parser.add_argument("--urdf", type=str, default=None, help="Override the recorded URDF path.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port (default 8080).")
    parser.add_argument("--max_ghosts", type=int, default=10, help="Ghost slot-pool size per robot (default 10).")
    parser.add_argument("--ghosts", type=int, default=6, help="Initially shown ghost count (default 6).")
    parser.add_argument("--point_size", type=float, default=0.02, help="Height-scan point size in m (default 0.02).")
    args = parser.parse_args()

    data_path = args.data or find_newest_recording()
    data, meta = load_viz_recording(data_path)
    urdf_path = resolve_urdf_path(args.urdf, meta.urdf_path)
    print(
        f"[INFO] {data_path}\n"
        f"[INFO] task={meta.task_name} recorded_at={meta.recorded_at} dry_run={meta.dry_run}\n"
        f"[INFO] T={meta.num_steps} N={meta.num_envs} H={meta.history_len} F={meta.future_len} "
        f"nj={meta.num_joints} dt={meta.dt:.4f}s scan_rays={meta.num_scan_rays} terrain={meta.terrain_source}\n"
        f"[INFO] urdf={urdf_path}"
    )

    viewer = FlowMatchingViewer(
        data, meta, urdf_path,
        max_ghosts=args.max_ghosts, initial_ghosts=min(args.ghosts, args.max_ghosts),
        point_size=args.point_size, port=args.port,
    )
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
