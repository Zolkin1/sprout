import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("lidar_scan")

    # RViz args (only defined here)
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')

    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )

    # Make this a FILE path, not a folder
    cur_path = os.path.split(os.path.realpath(__file__))[0] + '/../'
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=os.path.join(cur_path, 'rviz', 'fastlio.rviz'),
        description='RViz config file'
    )

    # Include the non-RViz launch (brings in use_sim_time/config_path/config_file + mid360 + fastlio)
    fastlio_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'fastlio_launch.py')
        ),
        # Optional: forward args (nice for clarity, but not strictly required)
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_path':  LaunchConfiguration('config_path'),
            'config_file':  LaunchConfiguration('config_file'),
        }.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use),
        output='screen',
    )

    return LaunchDescription([
        # forwardable args (so you can still do: ros2 launch ... use_sim_time:=true etc.)
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('config_path', default_value=os.path.join(pkg_share, 'config')),
        DeclareLaunchArgument('config_file', default_value='fastlio_config.yaml'),

        declare_rviz_cmd,
        declare_rviz_config_file_cmd,

        fastlio_launch,
        rviz_node,
    ])