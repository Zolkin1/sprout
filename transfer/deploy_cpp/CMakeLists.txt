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
find_package(Threads REQUIRED)  # the generator runs on its own thread (fm_worker)

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
  src/gen_tracker_core.cc
  src/fm_worker.cc
  src/obs_assembler.cc
)
add_library(G1Deploy::Core ALIAS g1_deploy)

target_include_directories(g1_deploy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(g1_deploy PUBLIC Eigen3::Eigen Threads::Threads)
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
# onnxruntime (optional)
# --------------------------------------------------------------------------------------
# The image supplies it at /opt/onnxruntime (docker/Dockerfile.ort builds it; the prefix is
# identical on both architectures, so nothing here branches on arch). OPTIONAL on purpose:
# the pure-math half of this library is the part held to the python oracle, and it must stay
# buildable and testable on a dev box with no ORT and no GPU -- which is also how the oracle
# itself runs.
set(ONNXRUNTIME_ROOT "/opt/onnxruntime" CACHE PATH "onnxruntime install prefix")
find_path(ORT_INCLUDE_DIR onnxruntime/onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT}/include
          NO_DEFAULT_PATH)
find_library(ORT_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)

if(ORT_INCLUDE_DIR AND ORT_LIBRARY)
  message(STATUS "onnxruntime found at ${ONNXRUNTIME_ROOT} -- building the session layer")
  add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ORT_LIBRARY}"
    # SYSTEM include: third-party headers are not clean under -Wall -Wextra -Wpedantic, and
    # those warnings are worth keeping on our own code. Imported targets get -isystem by
    # default; this is belt and braces in case NO_SYSTEM_FROM_IMPORTED is ever set.
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}")

  # The CUDA provider is a SEPARATE .so that libonnxruntime dlopens at session creation. It is
  # never linked, but its absence is the difference between a GPU run and a silent CPU one.
  if(NOT EXISTS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so)
    message(WARNING "onnxruntime at ${ONNXRUNTIME_ROOT} has no CUDA provider library; "
                    "this is a CPU-only build and the policy pair cannot meet its budget on it.")
  endif()

  add_library(g1_deploy_ort STATIC
    src/ort_session.cc
    src/tracker_session.cc
    src/fm_session.cc
  )
  add_library(G1Deploy::Ort ALIAS g1_deploy_ort)
  target_link_libraries(g1_deploy_ort
    PUBLIC g1_deploy onnxruntime::onnxruntime
    PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(g1_deploy_ort PRIVATE -Wall -Wextra -Wpedantic)
  set(G1_DEPLOY_HAVE_ORT TRUE)
else()
  message(STATUS "onnxruntime not found under ${ONNXRUNTIME_ROOT} -- session layer skipped "
                 "(the math core and its parity tests still build)")
  set(G1_DEPLOY_HAVE_ORT FALSE)
endif()

# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
# Optional so a deploy image can build the library without a test framework present. On a
# dev box both are there and `ctest` just works.
include(CTest)
if(BUILD_TESTING)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    foreach(name quat_math_parity window_parity depth_obs_parity node_history policy_params plan_scheduler gen_tracker_core fm_worker obs_assembler)
      add_executable(${name}_test test/${name}_test.cc test/fixture.cc)
      target_include_directories(${name}_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
      target_link_libraries(${name}_test
        PRIVATE g1_deploy nlohmann_json::nlohmann_json GTest::gtest GTest::gtest_main)
      target_compile_options(${name}_test PRIVATE -Wall -Wextra -Wpedantic)
      # Where the committed fixtures live. Overridable at runtime with G1_PARITY_FIXTURES so
      # a freshly dumped set can be checked without rebuilding.
      target_compile_definitions(${name}_test PRIVATE
        G1_DEPLOY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/parity"
        G1_DEPLOY_METADATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/metadata"
        G1_DEPLOY_OBS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/obs")
      add_test(NAME ${name} COMMAND ${name}_test)
    endforeach()

    # The session tests need onnxruntime AND a real policy pair. They skip without the
    # artifacts, so they are safe to register wherever ORT exists.
    if(G1_DEPLOY_HAVE_ORT)
      add_executable(ort_sessions_test test/ort_sessions_test.cc)
      target_link_libraries(ort_sessions_test
        PRIVATE g1_deploy_ort GTest::gtest GTest::gtest_main)
      target_compile_options(ort_sessions_test PRIVATE -Wall -Wextra -Wpedantic)
      target_compile_definitions(ort_sessions_test PRIVATE
        G1_DEPLOY_POLICY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../obelisk/g1_control/resource/hf_cache/zolkin_robot_rl/2026-08-04_21-55-17_yaw_k12_v1")
      add_test(NAME ort_sessions COMMAND ort_sessions_test)
    endif()
  else()
    message(STATUS "GTest not found -- skipping g1_deploy tests")
  endif()
endif()
