#!/usr/bin/env python3
"""
Generate a real-time video of the robot's XY trajectory being drawn.

The script loads odometry data (pos_w_x, pos_w_y, time), linearly interpolates
to 60 Hz, and renders an MP4 where the path is traced at real-time speed.

Usage:
    python plot_trajectory_video.py [--data path_to_csv] [--output video.mp4]

If no path is provided, uses the most recent CSV from odom_logs.
"""

import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from plot_odom import find_most_recent_odom_csv, load_odom_data, slice_data

FPS = 60


def interpolate_to_fps(
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fps: int = FPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate position data to a uniform frame rate.

    Args:
        time: Original timestamp array (seconds).
        x: X position array.
        y: Y position array.
        fps: Target frame rate in Hz.

    Returns:
        Tuple of (time_interp, x_interp, y_interp) at the target frame rate.
    """
    t_interp = np.arange(time[0], time[-1], 1.0 / fps)
    x_interp = np.interp(t_interp, time, x)
    y_interp = np.interp(t_interp, time, y)
    return t_interp, x_interp, y_interp


def generate_trajectory_video(
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_path: str,
    target_x: np.ndarray | None = None,
    target_y: np.ndarray | None = None,
) -> None:
    """Generate and save an MP4 video of the trajectory being traced.

    Args:
        time: Interpolated timestamp array (seconds, uniform at FPS).
        x: Interpolated X positions.
        y: Interpolated Y positions.
        output_path: File path for the output MP4.
        target_x: Optional target X positions (full trajectory, drawn statically).
        target_y: Optional target Y positions.
    """
    # LaTeX font configuration
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts}",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # Rotate 90 deg right: plot Y on horizontal axis, X on vertical axis
    fig, ax = plt.subplots(figsize=(6, 10))

    # Fixed axis limits with padding (Y -> horizontal, X -> vertical)
    x_margin = (x.max() - x.min()) * 0.1 + 0.5
    y_margin = (y.max() - y.min()) * 0.1 + 0.5
    ax.set_xlim(y.min() - y_margin, y.max() + y_margin)
    ax.set_ylim(x.max() + x_margin, x.min() - x_margin)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$y$ Position (m)")
    ax.set_ylabel(r"$x$ Position (m)")
    ax.set_title(r"$(x,y)$ Trajectory")
    ax.grid(True, alpha=0.3)

    # Static elements
    if target_x is not None and target_y is not None:
        ax.plot(target_y, target_x, "r--", linewidth=1.5, alpha=0.6, label="Target")
    ax.plot(y[0], x[0], "go", markersize=10, zorder=5, label="Start")

    # Animated elements
    (trail_line,) = ax.plot([], [], "b-", linewidth=2)
    (current_dot,) = ax.plot([], [], "bo", markersize=8, zorder=5)

    n_frames = len(time)

    def init():
        """Initialize the animation."""
        trail_line.set_data([], [])
        current_dot.set_data([], [])
        return trail_line, current_dot

    def update(frame: int):
        """Update the animation for a given frame."""
        trail_line.set_data(y[: frame + 1], x[: frame + 1])
        current_dot.set_data([y[frame]], [x[frame]])
        return trail_line, current_dot

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000.0 / FPS,
        blit=True,
    )

    print(f"Rendering {n_frames} frames ({time[-1] - time[0]:.2f}s of data at {FPS} fps)...")
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved trajectory video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a real-time video of the XY trajectory",
    )
    parser.add_argument(
        "--data",
        help="Path to odom data directory containing odom_data.csv (if not provided, uses most recent from odom_logs)",
    )
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--end-time", type=float, default=None, help="End time (seconds)")
    parser.add_argument("--output", default=None, help="Output MP4 path (default: saves next to data dir)")
    parser.add_argument("--target", action="store_true", help="Overlay the target trajectory if available")
    args = parser.parse_args()

    # Determine CSV file
    if args.data:
        csv_path = os.path.join(args.data, "odom_data.csv")
        if not os.path.exists(csv_path):
            print(f"Error: odom_data.csv not found in {args.data}")
            sys.exit(1)
    else:
        csv_path = find_most_recent_odom_csv()
        if csv_path is None:
            sys.exit(1)

    # Load and slice data
    print(f"Loading data from: {csv_path}")
    data = load_odom_data(csv_path)
    print(f"Loaded {len(data['time'])} data points spanning {data['time'][-1] - data['time'][0]:.2f} seconds")
    data = slice_data(data, args.start_time, args.end_time)

    # Interpolate to 60 Hz
    t_interp, x_interp, y_interp = interpolate_to_fps(data["time"], data["pos_w_x"], data["pos_w_y"])
    print(f"Interpolated to {len(t_interp)} frames at {FPS} fps")

    # Target trajectory (optional)
    target_x, target_y = None, None
    if args.target and "target_pos_w_x" in data and "target_pos_w_y" in data:
        target_x = data["target_pos_w_x"]
        target_y = data["target_pos_w_y"]

    # Determine output path
    output_path = args.output or os.path.join(os.path.dirname(csv_path), "trajectory.mp4")

    generate_trajectory_video(t_interp, x_interp, y_interp, output_path, target_x, target_y)


if __name__ == "__main__":
    main()
