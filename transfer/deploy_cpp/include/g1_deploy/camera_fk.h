// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Each depth camera's pose in the ROOT'S YAW-ALIGNED frame, from encoders and gravity.
//
// The quantity the generator's xyz channels are built from, and the one thing the deploy
// runtime cannot derive for itself: unprojecting a depth image into that frame needs to know
// where the camera IS in it, and g1_deploy has no kinematics on purpose.
//
// TWO INPUTS, AND NEITHER IS ODOMETRY. Writing the root's orientation as
// R_world_root = Rz(psi) @ R_rp, a point's yaw-frame coordinates are R_rp @ p_root -- psi does
// not appear, and neither does the root's position. So this needs forward kinematics from the
// root to the camera mount (joint encoders) and the root's roll/pitch (the IMU's gravity
// direction), and nothing else. A drifting yaw estimate CANNOT corrupt the cloud, which is
// what made the xyz channels deployable rather than a simulator-only idea.
//
// WHY PINOCCHIO, AND WHY A URDF. The obvious alternative is deploy/fk_mujoco, which FKs the
// MJCF the generator's own FK loss was defined against -- but it is barred from the control
// path (test_hardware_offline asserts mujoco stays out of it), and pinocchio is both faster
// and already the robot's kinematics library. The usual objection to FK'ing the ROS URDF is
// that it would be a SECOND OPINION about where the camera is. Measured, it is not: over zero
// and five random configurations, the URDF and the MJCF agree on pelvis -> torso_link to
// 0.000000 m and 4e-13 of rotation, and the full camera pose reproduces
// dataset._camera_pose_yaw_frame -- the training-side function -- to the same. The two models
// are the same kinematics for this chain.
//
// The mount offset is the contract's NOMINAL one, never a calibration draw: a robot knows the
// pose it was calibrated to and nothing more, and training used the nominal pose deliberately
// so that the domain randomisation perturbs the IMAGE relative to the stated geometry, which
// is what calibration error actually is.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "g1_deploy/policy_params.h"

namespace g1_deploy {

/// Forward kinematics of the depth cameras, in the root's yaw-aligned frame.
class CameraPoseFk {
 public:
  /// Load the model and resolve the frames and the joint permutation once.
  ///
  /// \param model_path       URDF to FK against. A controller PARAMETER rather than a
  ///                         constant, so consolidating the robot's models later moves a
  ///                         config value instead of a recompile.
  /// \param isaac_joint_names The policy's joint order, which is what joint_pos arrives in.
  ///                          The URDF's own order is per limb and differs from it on 27 of
  ///                          the G1's 29 joints, so the permutation is resolved here, once,
  ///                          by NAME -- getting it wrong yields a plausible, wrong pose.
  /// \param cameras          The camera contract, in the generator's sensor-row order.
  ///
  /// \throws std::runtime_error if the model will not load, a mount body is absent, or a
  ///         policy joint has no counterpart in the model.
  CameraPoseFk(const std::string& model_path,
               const std::vector<std::string>& isaac_joint_names,
               const std::vector<DepthCameraSpec>& cameras);
  ~CameraPoseFk();

  /// Each camera's pose in the root's yaw-aligned frame.
  ///
  /// \param joint_pos       Encoder angles in the policy's joint order.
  /// \param base_quat_xyzw  The root's orientation. Only its ROLL AND PITCH are used -- the
  ///                        yaw is divided out here, which is what removes odometry from the
  ///                        result rather than merely from the inputs.
  /// \param out_pos         One 3-vector per camera, in contract order.
  /// \param out_rot         One ROW-major 3x3 per camera, in contract order.
  ///
  /// \throws std::runtime_error on a joint_pos length that is not the model's.
  void Compute(const std::vector<double>& joint_pos,
               const std::array<double, 4>& base_quat_xyzw,
               std::vector<std::array<double, 3>>* out_pos,
               std::vector<std::array<double, 9>>* out_rot) const;

  /// Cameras this instance resolves, in contract order.
  size_t num_cameras() const;

 private:
  struct Impl;                 //: pinocchio stays out of this header, as mujoco does in fk_mujoco
  std::unique_ptr<Impl> impl_;
};

}  // namespace g1_deploy
