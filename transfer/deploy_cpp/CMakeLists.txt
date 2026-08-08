# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The C++ deploy runtime: the port of transfer/deploy (python), which stays the MuJoCo
# sim2sim driver and the parity oracle. This half is what runs on the robot.
#
# Deliberately a PLAIN CMake project, not an ament package. rclcpp is not on this target's
# include path, so "ROS-free" is a property the compiler enforces rather than a convention
# somebody has to remember -- the same guarantee source/motion_gen has against isaaclab. It
# also means these tests build and run with no colcon and no ROS, which matters because the
# oracle they are checked against runs that way too.
#
# The ROS node (transfer/obelisk/g1_control_cpp) consumes this with
#   FetchContent_Declare(G1Deploy SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../deploy_cpp)
# which is the same pattern robot_sim/robot_viz already use for Obelisk itself.
#
# Standalone build and test (no ROS, no docker):
#   cmake -S transfer/deploy_cpp -B build/deploy_cpp && cmake --build build/deploy_cpp -j
#   ctest --test-dir build/deploy_cpp --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(G1Deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED)

# ONNX metadata is JSON. ordered_json is the part that matters: the default object type sorts
# keys, and the observation term order IS the observation layout (see policy_params.h).
# Preferred from the system when present so a clean build does not re-fetch it.
find_package(nlohmann_json 3.9 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(g1_deploy STATIC
  src/quat_math.cc
  src/window.cc
  src/depth_obs.cc
  src/node_history.cc
  src/policy_params.cc
  src/plan_scheduler.cc
)
add_library(G1Deploy::Core ALIAS g1_deploy)

target_include_directories(g1_deploy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(g1_deploy PUBLIC Eigen3::Eigen)
# PRIVATE: json appears only in policy_params.cc, never in a public header, so consumers do
# not inherit the dependency.
target_link_libraries(g1_deploy PRIVATE nlohmann_json::nlohmann_json)
target_compile_options(g1_deploy PRIVATE -Wall -Wextra -Wpedantic)

# -ffp-contract=off is a PARITY requirement, not a style preference. GCC defaults to
# `fast`, which contracts a*b+c into an FMA -- more accurate than numpy, and therefore
# DIFFERENT from numpy. On its own that turns a 1e-14 tolerance into a permanently fuzzy
# one. PUBLIC so a consumer cannot link this and then reintroduce the difference in its own
# translation units.
target_compile_options(g1_deploy PUBLIC -ffp-contract=off)

# -ffast-math / -Ofast fold std::isfinite() to a compile-time true, which silently removes
# the non-finite guard chain that keeps a NaN off the motors. types.h #errors on
# __FAST_MATH__; this catches -Ofast at configure time, before anyone waits for a compile.
foreach(flag IN LISTS CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELWITHDEBINFO)
  if(flag MATCHES "-ffast-math|-Ofast")
    message(FATAL_ERROR "g1_deploy must not be built with -ffast-math/-Ofast: it deletes "
                        "the NaN guards. Offending flags: ${flag}")
  endif()
endforeach()

# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
# Optional so a deploy image can build the library without a test framework present. On a
# dev box both are there and `ctest` just works.
include(CTest)
if(BUILD_TESTING)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    foreach(name quat_math_parity window_parity depth_obs_parity node_history policy_params plan_scheduler)
      add_executable(${name}_test test/${name}_test.cc test/fixture.cc)
      target_include_directories(${name}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
      target_link_libraries(${name}_test
        PRIVATE g1_deploy nlohmann_json::nlohmann_json GTest::gtest GTest::gtest_main)
      target_compile_options(${name}_test PRIVATE -Wall -Wextra -Wpedantic)
      # Where the committed fixtures live. Overridable at runtime with G1_PARITY_FIXTURES so
      # a freshly dumped set can be checked without rebuilding.
      target_compile_definitions(${name}_test PRIVATE
        G1_DEPLOY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/parity"
        G1_DEPLOY_METADATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/metadata")
      add_test(NAME ${name} COMMAND ${name}_test)
    endforeach()
  else()
    message(STATUS "GTest not found -- skipping g1_deploy tests")
  endif()
endif()
