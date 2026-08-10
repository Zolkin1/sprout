// Copyright 2026 Zachary Olkin. All rights reserved.

// Entry point for the C++ G1 controller. The class lives in g1_controller.h/.cc rather than
// being header-only: robot_viz is a template and has to be, this is not, and a ~600-line header
// would recompile everything on every edit.

#include <obelisk_ros_utils.h>

#include "g1_control/g1_controller.h"

int main(int argc, char* argv[]) {
  // MultiThreadedExecutor because the estimator runs at 1 kHz against a 50 Hz control tick and
  // the depth cameras have their own callback group. The generator does NOT run on an executor
  // thread -- it has its own, inside PolicyRuntime, because a 17 ms callback would occupy a
  // thread the estimator needs and because its priority and affinity have to be set separately.
  obelisk::utils::SpinObelisk<g1_control::G1Controller, rclcpp::executors::MultiThreadedExecutor>(
      argc, argv, "g1_controller");
}
