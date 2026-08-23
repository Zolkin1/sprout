# Copyright 2026 Zachary Olkin. All rights reserved.
"""Launch BOTH ZED depth nodes, one per physical camera.

The stock ``zed_depth.launch.py`` starts ONE node from ``default_params.yaml``, which asks
for HD720 -- a mode no GMSL ZED X can open -- and carries no camera selector, so on a
two-camera box two instances would race for camera 0.  This launches the pair from the
per-camera configs instead, each pinned to a serial and published on the topics the
controller subscribes to (``/depth/{near,far}/image_processed``; ``g1_world.yml`` refuses
to arm when the topic count disagrees with ``tracker.onnx``).

Node names are remapped, so the configs are keyed ``/**`` -- a params file keyed by node
name applies to nothing once the node is renamed, silently leaving every default in place.

Usage, inside the container (or ``run_node.sh pair``)::

    ros2 launch /opt/zed_launch/zed_depth_pair.launch.py
    ros2 launch /opt/zed_launch/zed_depth_pair.launch.py config_dir:=/some/other/dir
    ros2 launch /opt/zed_launch/zed_depth_pair.launch.py respawn:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

#: Camera role -> node name. The role is also the config basename and the topic namespace,
#: so the three cannot drift apart.
ROLES = ("near", "far")

#: Seconds before a died node is restarted.  Generous on purpose: ``sl::Camera::close()``
#: takes 25-30 s to release a ZED X, and a camera that is still being torn down will refuse
#: to reopen.  A short delay here would turn one nvargus blip into a restart loop that never
#: catches -- and nvargus-daemon HAS crashed repeatedly on this hardware, taking the pair
#: down with it, which is the whole reason respawn is on by default.
RESPAWN_DELAY_S = 10.0


def generate_launch_description() -> LaunchDescription:
    """Build the launch description for the near/far ZED depth pair.

    Returns:
        The description launching one ``zed_depth_node`` per camera role.
    """
    config_dir = LaunchConfiguration("config_dir")
    respawn = LaunchConfiguration("respawn")

    actions: list = [
        DeclareLaunchArgument(
            "config_dir",
            default_value="/opt/zed_config",
            description="Directory holding near.yaml and far.yaml (bind-mounted, so the "
                        "FoV/range numbers can change with the policy without a rebuild).",
        ),
        DeclareLaunchArgument(
            "respawn",
            default_value="true",
            description="Restart a node if it dies. Off for debugging, where a crash loop "
                        "hides the first traceback behind the second.",
        ),
    ]

    for role in ROLES:
        params = PathJoinSubstitution([config_dir, f"{role}.yaml"])
        # Two otherwise-identical Nodes, differing only in `respawn`, because respawn takes a
        # bool and not a substitution -- a LaunchConfiguration there is truthy whatever its
        # value, so `respawn:=false` would silently keep respawning.
        common = dict(
            package="zed_depth_publisher",
            executable="zed_depth_node",
            name=f"zed_{role}",
            output="screen",
            parameters=[params],
            emulate_tty=True,
        )
        actions.append(Node(condition=IfCondition(respawn), respawn=True,
                            respawn_delay=RESPAWN_DELAY_S, **common))
        actions.append(Node(condition=UnlessCondition(respawn), respawn=False, **common))

    return LaunchDescription(actions)
