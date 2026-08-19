# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The C++ G1 controller. A NEW package rather than a conversion of g1_control: that one is
# ament_python and also ships the estimator, the FSM echo node and the whole config/ tree, so
# converting it would drag all of that along. Two packages also means the python and C++
# controllers can coexist, which is what makes a regression bisectable by flipping one config
# line -- see config/g1_cpp.yml.

cmake_minimum_required(VERSION 3.8)
project(g1_control_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(fastcdr REQUIRED)  # Required for ament transitive dependencies
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(g1_msgs REQUIRED)
# The obelisk message packages are deliberately NOT find_package'd here. They live in the
# obelisk workspace, and Obelisk::Core already carries them (its own CMake does
# ament_target_dependencies(ObkCore INTERFACE ... obelisk_control_msgs ...)), which is how
# robot_sim and robot_viz reach them. Asking for them directly means finding them before
# FetchContent has put the obelisk workspace on the prefix path -- and failing for a reason
# that has nothing to do with what is missing.

include(FetchContent)
# Obelisk by SOURCE_DIR, the pattern robot_sim and robot_viz already use: the image has the
# source at $OBELISK_ROOT, so there is nothing to download.
FetchContent_Declare(Obelisk SOURCE_DIR $ENV{OBELISK_ROOT}/obelisk/cpp)
# The deploy runtime, from the sibling ROS-FREE project. Same mechanism, and keeping it out of
# the ament graph is what stops rclcpp reaching the numeric code.
FetchContent_Declare(G1Deploy SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../deploy_cpp)
FetchContent_MakeAvailable(Obelisk G1Deploy)

add_executable(g1_controller src/main.cc src/g1_controller.cc)
target_include_directories(g1_controller PRIVATE include)

# All-KEYWORD linking, and no ament_target_dependencies. Two reasons, both of which cost a
# build to discover:
#
#   * ament_target_dependencies uses the PLAIN target_link_libraries signature, and CMake
#     forbids mixing plain and keyword forms on one target. Obelisk::Core has to be linked with
#     a keyword (it is an imported/alias target), so everything here must be.
#   * for an interface package found with find_package -- as opposed to one generated in THIS
#     project -- the exported targets are ${<pkg>_TARGETS}. rosidl_get_typesupport_target is
#     for the generating project and fails with "g1_msgs is not a CMake target" here, which
#     reads like a missing dependency rather than the wrong API.
#
# No other package in this workspace links g1_msgs from C++ yet, so this is the pattern.
# G1Deploy::Fk is optional in deploy_cpp (the sim2sim runner and the parity tests do not need
# pinocchio) but REQUIRED here: the shipped generator reads 5 channels per pixel, and four of
# them are the camera's position in the root's yaw-aligned frame. Failing at CONFIGURE time
# with the install line beats linking a controller that refuses every policy at bring-up.
if(NOT TARGET G1Deploy::Fk)
  message(FATAL_ERROR
    "pinocchio was not found, so the camera FK is not built. The controller needs it for any "
    "generator that conditions on more than depth. Install it (the container gets it from "
    "docker/requirements.txt) or set CMAKE_PREFIX_PATH so find_package(pinocchio) resolves.")
endif()

target_link_libraries(g1_controller PRIVATE
  Obelisk::Core
  G1Deploy::Ort
  G1Deploy::Fk
  rclcpp::rclcpp
  rclcpp_lifecycle::rclcpp_lifecycle
  ${sensor_msgs_TARGETS}
  ${g1_msgs_TARGETS})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS g1_controller DESTINATION lib/${PROJECT_NAME})
ament_package()
