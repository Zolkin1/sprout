export ROS_DOMAIN_ID=2

source ~/livox_driver_ws/install/setup.bash
source ~/lidar_odom_ws/install/setup.bash

ros2 launch livox_ros_driver2 msg_MID360_launch.py & ros2 launch fast_lio_vel mapping.launch.py config:=mid360.yaml rviz:=False