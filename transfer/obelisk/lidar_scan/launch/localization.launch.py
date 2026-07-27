# Copyright 2026 Zachary Olkin. All rights reserved.

"""Bring up localisation: FAST-LIO mapping, plus the map->odom link.

Two layers, because FAST-LIO provides only one of them:

1. **FAST-LIO** (``relocalization_bringup``) consumes /livox/lidar + /livox/imu and
   publishes /Odometry, /cloud_registered{,_body} and the ``odom -> imu`` TF. This is the
   estimator's only source of base position and world linear velocity, which the
   flow-matching generator's history tokens are built from -- so on this stack losing
   localisation corrupts the reference the tracker is chasing, it does not merely degrade
   navigation.

2. **map -> odom**, which FAST-LIO never publishes. Either ``scan_lock`` (ICP against a
   prior PCD, the default) or, with ``reloc:=false``, a static identity transform. Without
   one of the two, any ``map``-frame lookup fails outright.

The same launch serves sim and hardware: ``g1_world.yml`` publishes simulated /livox/lidar
and /livox/imu on the real topics, so FAST-LIO runs unmodified against MuJoCo.

    ros2 launch lidar_scan localization.launch.py
    ros2 launch lidar_scan localization.launch.py reloc:=false        # no prior map
    ros2 launch lidar_scan localization.launch.py map:=lab.yaml       # pick the prior map
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

#: Relocalisation configs live with the rest of the control config, not in this package, so
#: a per-experiment prior map is a config change rather than a package rebuild.
CONFIG_DIR = os.path.join(
    os.environ.get("ROBOT_RL_ROOT", ""), "g1_control", "config", "relocalization")


def _scan_lock_pcd(config_file: str) -> str:
    """The PCD a scan_lock config points at, so a preview cannot show the wrong map.

    Args:
        config_file: File name within the relocalisation config directory.

    Returns:
        The configured ``pcd_file_name``, or the empty string if it cannot be read.
    """
    path = os.path.join(CONFIG_DIR, config_file)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data["/**"]["ros__parameters"]["scan_lock"]["pcd_file_name"]
    except (OSError, KeyError, TypeError):
        return ""


def _setup(context, *args, **kwargs):
    """Build the launch actions once the arguments are resolved.

    Args:
        context: Launch context supplying the argument values.

    Returns:
        The list of actions to launch.
    """
    reloc = LaunchConfiguration("reloc").perform(context).lower() == "true"
    map_file = LaunchConfiguration("map").perform(context)
    mapping_file = LaunchConfiguration("mapping_config").perform(context)

    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("relocalization_bringup"), "launch", "mapping.launch.py")),
        launch_arguments={"config_path": CONFIG_DIR, "config_file": mapping_file}.items(),
    )]

    if reloc:
        pcd = _scan_lock_pcd(map_file)
        print(f"[localization] scan_lock against {pcd or '<unreadable pcd_file_name>'}")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("scan_lock"), "launch", "scan_lock.launch.py")),
            launch_arguments={"config_path": CONFIG_DIR, "config_file": map_file}.items(),
        ))
    else:
        # Identity map->odom. Only honest when there is no prior map: it does not localise,
        # it just makes map-frame lookups resolve so the rest of the graph runs.
        print("[localization] reloc:=false -- static identity map->odom (NOT localised).")
        actions.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="static_map_odom",
            arguments=["--x", "0", "--y", "0", "--z", "0",
                       "--yaw", "0", "--pitch", "0", "--roll", "0",
                       "--frame-id", "map", "--child-frame-id", "odom"],
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    """Declare the arguments and defer construction until they resolve."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "reloc", default_value="true",
            description="ICP against a prior map (true) or a static identity map->odom (false)."),
        DeclareLaunchArgument(
            "map", default_value="scan_lock.yaml",
            description="scan_lock config in g1_control/config/relocalization -- the prior map."),
        DeclareLaunchArgument(
            "mapping_config", default_value="mid360_relocalization.yaml",
            description="FAST-LIO config in g1_control/config/relocalization."),
        OpaqueFunction(function=_setup),
    ])
