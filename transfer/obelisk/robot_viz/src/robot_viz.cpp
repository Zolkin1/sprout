// Copyright 2026 Zachary Olkin. All rights reserved.

#include "obelisk_ros_utils.h"
#include "robot_viz/robot_viz.h"

int main(int argc, char* argv[]) {
    obelisk::utils::SpinObelisk<obelisk::viz::RobotViz<obelisk_estimator_msgs::msg::EstimatedState>,
                                rclcpp::executors::MultiThreadedExecutor>(argc, argv, "robot_viz");
}