// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/camera_fk.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <cmath>
#include <stdexcept>

namespace g1_deploy {

struct CameraPoseFk::Impl {
  pinocchio::Model model;
  mutable pinocchio::Data data;
  std::vector<int> q_index;                 //: policy joint slot -> the model's q slot
  pinocchio::FrameIndex root_frame = 0;
  std::vector<pinocchio::FrameIndex> mounts;
  std::vector<pinocchio::SE3> mount_to_cam;  //: the contract's NOMINAL offset, per camera

  Impl() : data(model) {}
};

namespace {

/// A rotation matrix from a scalar-LAST quaternion.
///
/// Scalar-last because everything crossing this codebase's boundaries is: the ONNX metadata,
/// the camera contract's offset_quat_xyzw, and the estimator's base orientation. Pinocchio's
/// own Eigen::Quaterniond is scalar-FIRST in its (w, x, y, z) constructor, which is exactly
/// the sort of silent 180-degree error this conversion exists to make explicit.
Eigen::Matrix3d MatrixFromQuatXyzw(const std::array<double, 4>& q) {
  return Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
}

}  // namespace

CameraPoseFk::CameraPoseFk(const std::string& model_path,
                           const std::vector<std::string>& isaac_joint_names,
                           const std::vector<DepthCameraSpec>& cameras)
    : impl_(new Impl()) {
  try {
    pinocchio::urdf::buildModel(model_path, impl_->model);
  } catch (const std::exception& e) {
    throw std::runtime_error("camera FK cannot load '" + model_path + "': " + e.what());
  }
  impl_->data = pinocchio::Data(impl_->model);

  // FIXED base: this computes a RELATIVE transform (root -> camera), so a floating joint would
  // add seven coordinates nobody has and a world frame nobody wants.
  impl_->q_index.reserve(isaac_joint_names.size());
  for (const std::string& name : isaac_joint_names) {
    if (!impl_->model.existJointName(name)) {
      throw std::runtime_error(
          "camera FK: the policy names joint '" + name + "' but '" + model_path +
          "' has no such joint; these must be the same robot.");
    }
    impl_->q_index.push_back(impl_->model.joints[impl_->model.getJointId(name)].idx_q());
  }

  // The root frame the pose is expressed relative to. Named by the MODEL rather than assumed
  // to be the world: a fixed-base URDF's root link is wherever the file puts it.
  const std::string root = impl_->model.frames[1].name;
  impl_->root_frame = impl_->model.getFrameId(root);

  for (const DepthCameraSpec& cam : cameras) {
    if (!impl_->model.existFrame(cam.mount_body)) {
      throw std::runtime_error(
          "camera FK: camera '" + cam.sensor + "' is mounted on '" + cam.mount_body +
          "', which '" + model_path + "' does not define.");
    }
    impl_->mounts.push_back(impl_->model.getFrameId(cam.mount_body));
    impl_->mount_to_cam.emplace_back(MatrixFromQuatXyzw(cam.offset_quat_xyzw),
                                     Eigen::Vector3d(cam.offset_pos[0], cam.offset_pos[1],
                                                     cam.offset_pos[2]));
  }
}

CameraPoseFk::~CameraPoseFk() = default;

size_t CameraPoseFk::num_cameras() const { return impl_->mounts.size(); }

void CameraPoseFk::Compute(const std::vector<double>& joint_pos,
                           const std::array<double, 4>& base_quat_xyzw,
                           std::vector<std::array<double, 3>>* out_pos,
                           std::vector<std::array<double, 9>>* out_rot) const {
  if (joint_pos.size() != impl_->q_index.size()) {
    throw std::runtime_error(
        "camera FK: " + std::to_string(joint_pos.size()) + " joint angles for " +
        std::to_string(impl_->q_index.size()) + " named joints.");
  }
  Eigen::VectorXd q = Eigen::VectorXd::Zero(impl_->model.nq);
  for (size_t i = 0; i < joint_pos.size(); ++i) q[impl_->q_index[i]] = joint_pos[i];
  pinocchio::framesForwardKinematics(impl_->model, impl_->data, q);

  // Divide the YAW out of the root's orientation, leaving the tilt. R_rp is what carries the
  // camera from the root frame into the yaw-aligned one, and building it this way -- rather
  // than from a yaw estimate -- is what makes the result independent of one.
  const Eigen::Matrix3d r_world_root = MatrixFromQuatXyzw(base_quat_xyzw);
  const double yaw = std::atan2(2.0 * (base_quat_xyzw[3] * base_quat_xyzw[2] +
                                       base_quat_xyzw[0] * base_quat_xyzw[1]),
                                1.0 - 2.0 * (base_quat_xyzw[1] * base_quat_xyzw[1] +
                                             base_quat_xyzw[2] * base_quat_xyzw[2]));
  const Eigen::Matrix3d r_rp =
      Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * r_world_root;

  const pinocchio::SE3 root_inv = impl_->data.oMf[impl_->root_frame].inverse();
  out_pos->resize(impl_->mounts.size());
  out_rot->resize(impl_->mounts.size());
  for (size_t c = 0; c < impl_->mounts.size(); ++c) {
    const pinocchio::SE3 cam_in_root =
        root_inv * impl_->data.oMf[impl_->mounts[c]] * impl_->mount_to_cam[c];
    const Eigen::Vector3d p = r_rp * cam_in_root.translation();
    const Eigen::Matrix3d r = r_rp * cam_in_root.rotation();
    (*out_pos)[c] = {p.x(), p.y(), p.z()};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) (*out_rot)[c][static_cast<size_t>(i * 3 + j)] = r(i, j);
    }
  }
}

}  // namespace g1_deploy
