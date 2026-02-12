import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

################### user configure parameters ###################
xfer_format   = 0
multi_topic   = 0
data_src      = 0
publish_freq  = 10.0
output_type   = 0
frame_id      = 'livox_frame'
lvx_file_path = '/home/livox/livox_test.lvx'
cmdline_bd_code = 'livox0000000001'

pkg_share = get_package_share_directory("lidar_scan")
user_config_path = os.path.join(pkg_share, 'config/mid360_config.json')
cur_path = os.path.split(os.path.realpath(__file__))[0] + '/../'
rviz_config_path = os.path.join(cur_path, 'rviz', 'mid360.rviz')

################### end ###################

livox_ros2_params = [
    {"xfer_format": xfer_format},
    {"multi_topic": multi_topic},
    {"data_src": data_src},
    {"publish_freq": publish_freq},
    {"output_data_type": output_type},
    {"frame_id": frame_id},
    {"lvx_file_path": lvx_file_path},
    {"user_config_path": user_config_path},
    {"cmdline_input_bd_code": cmdline_bd_code},
]

def generate_launch_description():
    livox_driver = Node(
        package='livox_ros_driver2',
        executable='livox_ros_driver2_node',
        name='livox_lidar_publisher',
        output='screen',
        parameters=livox_ros2_params,
    )

    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_path],
    )

    return LaunchDescription([
        livox_driver,
        rviz2,
    ])