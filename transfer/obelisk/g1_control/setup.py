# Copyright 2026 Zachary Olkin. All rights reserved.

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "g1_control"


def config_files(base_dir: str = "config"):
    """Install the config tree, preserving its subdirectory layout.

    ``include:`` paths resolve relative to the including file's directory, so the installed
    layout has to mirror the source one -- a flat glob would put ``common/low_level.yml``
    next to the file that includes it and break every include.

    Args:
        base_dir: Config directory to install, relative to this package.

    Returns:
        ``data_files`` entries, one per source subdirectory.
    """
    entries = []
    for path in glob(os.path.join(base_dir, "**", "*.y*ml"), recursive=True):
        install_dir = os.path.join("share", package_name, os.path.dirname(path))
        entries.append((install_dir, [path]))
    return entries


data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
# Policies are NOT shipped in the package any more: the robot pulls tracker.onnx + fm.onnx
# from HuggingFace at configure time (deploy.hardware.fetch_artifacts, warmed by
# scripts/prefetch_policy.py), so a checkpoint swap needs no colcon rebuild and the repo
# carries no binaries.
data_files += config_files()
# The RViz layout the `viz:` block names. Installed to share/g1_control/rviz/, which is where
# obelisk's bringup looks it up (get_package_share_directory(rviz_pkg) / "rviz" / rviz_config).
data_files += [("share/" + package_name + "/rviz", glob(os.path.join("rviz", "*.rviz")))]

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zolkin",
    maintainer_email="zach.olkin@gmail.com",
    description="G1 low-level RL deployment: the generator+tracker controller and its estimator.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "g1_controller = g1_control.controller:main",
            "g1_estimator = g1_control.estimator:main",
            "fsm_state_echo = g1_control.fsm_state_echo_node:main",
            "scripted_vel_cmd = g1_control.scripted_vel_cmd_node:main",
            "fm_plan_viz = g1_control.fm_plan_viz_node:main",
            "depth_cloud_viz = g1_control.depth_cloud_viz_node:main",
            "depth_delay_relay = g1_control.depth_delay_relay_node:main",
        ],
    },
)
