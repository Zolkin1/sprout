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
  src/blob_pack.cc
  src/clip_reference.cc
  src/fm_meta.cc
  src/robot_asset.cc
  src/quat_math.cc
  src/window.cc
  src/depth_obs.cc
  src/node_history.cc
  src/policy_params.cc
  src/plan_scheduler.cc
  src/gen_tracker_core.cc
  src/fm_worker.cc
  src/obs_assembler.cc
  src/imu_hold.cc
  src/pose_history.cc
)
add_library(G1Deploy::Core ALIAS g1_deploy)

target_include_directories(g1_deploy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(g1_deploy PUBLIC Eigen3::Eigen Threads::Threads)
# PRIVATE: json appears only in policy_params.cc and blob_pack.cc, never in a public header,
# so consumers do not inherit the dependency.
target_link_libraries(g1_deploy PRIVATE nlohmann_json::nlohmann_json)
# Where the committed parity fixtures live. PRIVATE and optional: blob_pack.cc bakes it in as
# the default for DefaultFixtureDir() when it is defined, and throws a diagnosable error when
# it is not -- which is the deploy image, where the source tree is absent and every pack is
# named explicitly anyway.
target_compile_definitions(g1_deploy PRIVATE
  G1_DEPLOY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/parity")
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
    src/reference_source.cc
    src/ort_session.cc
    src/tracker_session.cc
    src/fm_session.cc
    src/policy_runtime.cc
  )
  add_library(G1Deploy::Ort ALIAS g1_deploy_ort)
  target_link_libraries(g1_deploy_ort
    PUBLIC g1_deploy onnxruntime::onnxruntime
    PRIVATE nlohmann_json::nlohmann_json)
  target_compile_options(g1_deploy_ort PRIVATE -Wall -Wextra -Wpedantic)
  # The pre-ROS gate: does onnxruntime run THIS pair on THIS GPU inside the budget? It reads
  # the shapes off the graph rather than restating them, so it also answers "is this artifact
  # what I think it is" before a controller, a world and a joystick are in the picture.
  #
  # It shipped with the first C++ commit and was wired into NOTHING until now, which is why it
  # answered that question for nobody. It links onnxruntime directly and none of this library:
  # the point is to isolate the runtime, so a failure here cannot be our arithmetic.
  add_executable(ort_smoke tools/ort_smoke.cc)
  target_link_libraries(ort_smoke PRIVATE onnxruntime::onnxruntime)
  target_compile_options(ort_smoke PRIVATE -Wall -Wextra -Wpedantic)

  # The cross-language replay: this runtime over a stream the python one recorded. Links the
  # FIXTURE READER from test/ deliberately -- the stream IS a fixture set, and a second reader
  # for one format is how the format drifts.
  add_executable(replay tools/replay.cc)
  target_include_directories(replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(replay PRIVATE g1_deploy_ort nlohmann_json::nlohmann_json)
  target_compile_options(replay PRIVATE -Wall -Wextra -Wpedantic)
  # fixture.cc's DefaultDir() bakes this in. The tool never calls it -- the stream directory is
  # argv[1] -- but the translation unit still has to compile.
  target_compile_definitions(replay PRIVATE
    G1_DEPLOY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/parity")

  set(G1_DEPLOY_HAVE_ORT TRUE)
else()
  # WARNING, not STATUS. What is skipped is four source files -- ort_session, tracker_session,
  # fm_session, policy_runtime -- which means a SYNTAX ERROR IN THEM SURVIVES A CLEAN BUILD
  # here and is only found in the container. That is not hypothetical: an extraction that ate
  # the head of GaussianNoise::Fill was committed and passed every local check, because
  # nothing on the dev box ever compiled the file.
  message(WARNING "onnxruntime not found under ${ONNXRUNTIME_ROOT}: the session layer is NOT "
                  "COMPILED (ort_session, tracker_session, fm_session, policy_runtime). The "
                  "math core and its parity tests still build, but a build that is green here "
                  "says nothing about those four files -- check them in the container.")
  set(G1_DEPLOY_HAVE_ORT FALSE)
endif()

# --------------------------------------------------------------------------------------
# Camera forward kinematics (optional: pinocchio)
# --------------------------------------------------------------------------------------
# Optional for the same reason the ORT layer is: the math core must build on a box that has
# neither. A 1-channel generator never needs this, and a multi-channel one is refused at load
# with a message naming what is missing rather than failing later with a wrong cloud.
#
# It lives HERE rather than in the ROS node so it builds and tests without ROS -- the camera
# pose is the one input where a mistake is invisible (a plausible cloud in the wrong place),
# so it needs to be checkable against the python oracle on any dev box.
# coal (the collision library, ex-hpp-fcl) FIRST, and QUIET. pinocchio 4.0 -- what the ROS
# image ships -- has pinocchio_parsers link pinocchio_collision, whose interface names
# coal::coal, but pinocchioConfig does not find_dependency it. Without this the link fails at
# CMake GENERATE time with "Target coal::coal not found", pointing at the CONSUMER's
# target_link_libraries line rather than at anything to do with collision. A no-op under 3.9
# (the pip/cmeel build), where the parsers target does not reach collision at all.
find_package(coal QUIET)
if(TARGET coal::coal)
  # ...and PROMOTED, because an imported target is directory-scoped by default. This project
  # is pulled into the ROS controller by FetchContent, so without this the target exists here
  # and is invisible in the consumer's own target_link_libraries -- which is where the
  # generator expression is evaluated, and where the "not found" error is then reported.
  # CMake 3.24's `find_package(... GLOBAL)` would say this in one word; the image has 3.22.
  set_target_properties(coal::coal PROPERTIES IMPORTED_GLOBAL TRUE)
endif()
find_package(pinocchio QUIET)
if(pinocchio_FOUND)
  add_library(g1_deploy_fk STATIC src/camera_fk.cc)
  target_include_directories(g1_deploy_fk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # pinocchio_parsers, NOT the umbrella pinocchio::pinocchio. The umbrella drags in
  # pinocchio::pinocchio_collision, whose interface names coal::coal (the collision library,
  # ex-hpp-fcl) -- a target that is not defined unless the consumer also find_package(coal)s,
  # so linking it fails at GENERATE time inside the ROS image with an error about a target
  # nobody asked for. Nothing here collides: this is FK of an open chain. The parsers target
  # carries the URDF reader and the core, which is the whole dependency, and it exists under
  # both the ROS package (4.0) and the pip/cmeel one (3.9).
  target_link_libraries(g1_deploy_fk PUBLIC g1_deploy pinocchio::pinocchio_parsers)
  target_compile_options(g1_deploy_fk PRIVATE -Wall -Wextra -Wpedantic)
  add_library(G1Deploy::Fk ALIAS g1_deploy_fk)
  message(STATUS "pinocchio ${pinocchio_VERSION} -- camera FK enabled")
else()
  message(STATUS "pinocchio not found -- camera FK skipped (a >1-channel generator needs it)")
endif()

# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
# Optional so a deploy image can build the library without a test framework present. On a
# dev box both are there and `ctest` just works.
include(CTest)
if(BUILD_TESTING)
  # Preferred from the system, FETCHED when absent -- the same treatment nlohmann_json gets
  # above, and for the same reason: these are PARITY tests held to exact float equality
  # against the python oracle, so "the tests silently did not build" is the one outcome that
  # must not depend on what a particular box happens to have installed.
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
      GIT_SHALLOW TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GTest_FOUND TRUE)
  endif()
  if(GTest_FOUND AND TARGET g1_deploy_fk)
    add_executable(camera_fk_parity_test test/camera_fk_parity_test.cc)
    target_include_directories(camera_fk_parity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
    target_link_libraries(camera_fk_parity_test
      PRIVATE g1_deploy_fk nlohmann_json::nlohmann_json GTest::gtest GTest::gtest_main)
    target_compile_definitions(camera_fk_parity_test PRIVATE
      G1_DEPLOY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/parity")
    add_test(NAME camera_fk_parity COMMAND camera_fk_parity_test)
  endif()
  if(GTest_FOUND)
    foreach(name quat_math_parity window_parity clip_reference_parity depth_obs_parity fm_meta node_history policy_params plan_scheduler gen_tracker_core fm_worker obs_assembler imu_hold pose_history robot_asset)
      add_executable(${name}_test test/${name}_test.cc)
      # test/ for the fixture reader, and src/ for the INTERNAL headers a test may need --
      # robot_asset_json.h lives there rather than in include/ because it takes an nlohmann
      # type, and nlohmann is PRIVATE to this library (a public header would leak it into the
      # ROS node that consumes us).
      target_include_directories(${name}_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_SOURCE_DIR}/src)
      target_link_libraries(${name}_test
        PRIVATE g1_deploy nlohmann_json::nlohmann_json GTest::gtest GTest::gtest_main)
      target_compile_options(${name}_test PRIVATE -Wall -Wextra -Wpedantic)
      # Where the committed fixtures live. Overridable at runtime with G1_PARITY_FIXTURES so
      # a freshly dumped set can be checked without rebuilding.
      target_compile_definitions(${name}_test PRIVATE
        G1_DEPLOY_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/parity"
        # transfer/, so a clip pack recorded relative to it resolves. Relative because the
        # pack is a RUNTIME artifact that ships with the controller, not a fixture -- an
        # absolute path in the manifest would be one machine's.
        G1_DEPLOY_TRANSFER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/.."
        G1_DEPLOY_METADATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/metadata"
        G1_DEPLOY_OBS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../deploy/tests/fixtures/obs")
      add_test(NAME ${name} COMMAND ${name}_test)
    endforeach()

    # The session tests need onnxruntime AND a real policy pair. They skip without the
    # artifacts, so they are safe to register wherever ORT exists.
    if(G1_DEPLOY_HAVE_ORT)
      foreach(ortname ort_sessions policy_runtime)
        add_executable(${ortname}_test test/${ortname}_test.cc)
        target_link_libraries(${ortname}_test
          PRIVATE g1_deploy_ort GTest::gtest GTest::gtest_main)
        target_compile_options(${ortname}_test PRIVATE -Wall -Wextra -Wpedantic)
        # NOTE the `policies/` element. The controller loads
        # hf_cache/<repo>/policies/<run> (hardware.HF_POLICY_PREFIX), and this default was
        # missing it -- so these two tests SILENTLY SKIPPED on every dev box, which is exactly
        # the outcome test_policy_cache_layout.py exists to prevent across the language
        # boundary and could not see here. Overridable at runtime with G1_POLICY_DIR.
        target_compile_definitions(${ortname}_test PRIVATE
          G1_DEPLOY_POLICY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../obelisk/g1_control/resource/hf_cache/zolkin_robot_rl/policies/2026-08-20_16-36-10_mode11_decor_plan_v1"
          # The clip pack the runtime plays when no generator is loaded. A RUNTIME artifact
          # that ships with the controller, not a fixture -- the parity tests read the same one.
          G1_DEPLOY_CLIP_PACK="${CMAKE_CURRENT_SOURCE_DIR}/../obelisk/g1_control/resource/debug_clips/g1_debug_v1_pack")
        add_test(NAME ${ortname} COMMAND ${ortname}_test)
      endforeach()
    endif()
  else()
    message(STATUS "GTest not found -- skipping g1_deploy tests")
  endif()
endif()
