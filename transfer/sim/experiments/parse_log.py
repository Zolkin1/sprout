import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from velocity_commands import smooth_ramp
def parse_g1_control_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        "time", 
        "pos_x", "pos_y", "pos_z", 
        "ori_x", "ori_y", "ori_z", "ori_w"
    ]
    return df

def smooth_signal(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def estimate_velocity(df, smooth_window=10):
    df["vel_x"] = np.gradient(df["pos_x"], df["time"])
    df["vel_y"] = np.gradient(df["pos_y"], df["time"])
    df["vel_z"] = np.gradient(df["pos_z"], df["time"])

    df["vel_x"] = smooth_signal(df["vel_x"], smooth_window)
    df["vel_y"] = smooth_signal(df["vel_y"], smooth_window)
    df["vel_z"] = smooth_signal(df["vel_z"], smooth_window)

    return df

def compute_body_frame_vx(df, t_start=None, t_end=None):
    if t_start is not None:
        df = df[df["time"] >= t_start]
    if t_end is not None:
        df = df[df["time"] <= t_end]
    df = df.reset_index(drop=True)

    v_world = np.stack([df["vel_x"], df["vel_y"], df["vel_z"]], axis=1)
    quats = df[["ori_x", "ori_y", "ori_z", "ori_w"]].values
    rot = R.from_quat(quats)
    yaw = rot.as_euler('xyz', degrees=False)[:, 2]

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rot_mats = np.stack([
        np.stack([cos_yaw, -sin_yaw], axis=1),
        np.stack([sin_yaw,  cos_yaw], axis=1)
    ], axis=1)

    v_xy_world = v_world[:, :2][:, :, None]
    v_body = np.matmul(rot_mats, v_xy_world).squeeze(-1)
    vx_body = v_body[:, 0]

    return df["time"], vx_body

def plot_multiple_policies(file_paths, labels=None, time_ranges=None, smooth_window=10):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for all text
        "font.family": "serif",  # Use serif font (default LaTeX style)
        "font.serif": ["Computer Modern Roman"],  # LaTeX default font
    })
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

    if labels is None:
        labels = [f"Policy {i+1}" for i in range(len(file_paths))]
    if time_ranges is None:
        time_ranges = [(None, None)] * len(file_paths)

    for file_path, label, (t_start, t_end) in zip(file_paths, labels, time_ranges):
        df = parse_g1_control_csv(file_path)
        df = estimate_velocity(df, smooth_window)
        time, vx_body = compute_body_frame_vx(df, t_start=t_start, t_end=t_end)
        time = time - time.iloc[0]  # reset to start from 0
        plt.plot(time, vx_body, label=label, linewidth=1.5)


    # Add reference velocity ramp
    # Segments
    
#     t_pad = np.arange(0, 2, 0.01)
#     vx_pad = np.zeros_like(t_pad)

    vx_max = 0.75
    t_ref = np.linspace(0, 8, 500)
    ramp_time = 2.0
    slope = vx_max / ramp_time
    vx_ref = np.minimum(slope * t_ref, vx_max)
    #add a few more entries so that the first two seconds is 0
    #at the end just set to 0
    vx_ref[-1] = 0.0
    plt.plot(t_ref+3.5, vx_ref, "k--", label="Reference Ramp", linewidth=2)



    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel(r"$v_x$ (m/s)", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

# === Example Usage ===
if __name__ == "__main__":
    file_paths = [
        "experiments/g1_control_hzd_minimum.csv",
        "experiments/g1_control_baseline.csv"
    ]
    labels = [
        "HZD CLF Policy",
        "Baseline"
    ]
    time_ranges = [
        (3, 25),   # for first policy
        (124, 148)      # for second policy
    ]
    smooth_window = 10

    plot_multiple_policies(
        file_paths=file_paths,
        labels=labels,
        time_ranges=time_ranges,
        smooth_window=smooth_window
    )
