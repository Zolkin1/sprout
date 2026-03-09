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
# Include directories
# --------------------
include_directories(include)

# --------------------
# Executable
# --------------------
# add_executable(lidar_scan_node src/lidar_scan.cpp)
# ament_target_dependencies(lidar_scan_node
#   rclcpp
#   sensor_msgs
#   std_msgs
#   obelisk_sensor_msgs
# )

# --------------------
# Install
# --------------------
# install(TARGETS lidar_scan_node
#   DESTINATION lib/${PROJECT_NAME}
# )

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

install(
  DIRECTORY include/
  DESTINATION include
)

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