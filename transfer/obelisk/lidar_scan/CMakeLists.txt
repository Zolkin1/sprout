cmake_minimum_required(VERSION 3.8)
project(lidar_scan)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------
# Dependencies
# --------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(obelisk_sensor_msgs REQUIRED)

# --------------------
# Install
# --------------------
# Launch/config/rviz ONLY -- this package builds no node. The lidar driver comes from
# livox_ros_driver2 and the localisation from LIO-Localization (both built into the image);
# this is the thin wrapper that points them at OUR configs. The former src/lidar_scan.cpp
# stub was never implemented and has been removed.

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(
  DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# No `install(DIRECTORY include/)`: this package ships no headers, and the directory was
# never committed (git does not track empty ones), so ament_cmake_symlink_install aborted
# the build with "can't find .../lidar_scan/include/" on any fresh checkout -- taking
# g1_model, robot_sim and robot_viz down with it. Left over from the deleted C++ stub.

# --------------------
# Testing / lint
# --------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()