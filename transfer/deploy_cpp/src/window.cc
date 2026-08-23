// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/window.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "g1_deploy/quat_math.h"

namespace g1_deploy {
namespace {

/// Normalize a 3-vector, guarding a zero row exactly the way the numpy does.
///
/// The 1e-8 floor is not decoration: the generator's raw 9-vector can produce a degenerate
/// second column, and dividing by its norm would put NaN into the observation.
Vec3 Unit(const Vec3& v) { return v / std::max(v.norm(), 1e-8); }

int BlockWidth(ReferenceBlock block, int num_joints, int num_ee_frames) {
  switch (block) {
    case ReferenceBlock::kBasePose:
      return 9;  // position(3) + rot6d(6)
    case ReferenceBlock::kBaseVel:
      return 6;
    case ReferenceBlock::kJointPos:
    case ReferenceBlock::kJointVel:
      return num_joints;
    case ReferenceBlock::kEePos:
      return 3 * num_ee_frames;
  }
  throw std::runtime_error("unreachable: unknown reference block");
}

}  // namespace

namespace {

//: The two strings robot_rl.flow_matching.window.BASE_POSE_CONVENTIONS can stamp. Restated
//: here rather than parsed out of a shared file because there is no shared file across the
//: language boundary -- the artifact is. Both are named so the refusal below can say which of
//: the two it got and what that means, instead of echoing a hex-looking string back.
constexpr const char* kConventionRobotNow =
    "pose_in_anchor_yaw_frame_pos_ref_minus_actual_ori_rot6d";
constexpr const char* kConventionPlan =
    "pose_in_plan_yaw_frame_pos_ref_minus_plan_ori_rot6d";

}  // namespace

WindowAnchor AnchorFromConvention(const std::string& convention) {
  if (convention == kConventionPlan) return WindowAnchor::kPlan;
  if (convention == kConventionRobotNow) {
    throw std::runtime_error(
        std::string("tracker.onnx reference_window base_pose_convention is '") + convention +
        "' (window_anchor=\"robot_now\"), which this runtime refuses BY DESIGN rather than "
        "cannot serve. That arm measures the reference from the robot's base position every "
        "control step, so serving it needs odometry -- and this deploy path has none: the "
        "controller reads encoders and the IMU, and a base position it does not trust would "
        "move the whole reference window. Train or export with window_anchor=\"plan\", whose "
        "prediction IS the observation. For a robot_now A/B use the python runtime "
        "(transfer/sim/gen_tracker_runner.py), where MuJoCo supplies a true base pose.");
  }
  throw std::runtime_error(
      std::string("tracker.onnx reference_window base_pose_convention is '") + convention +
      "', which this runtime does not recognise. Known: '" + kConventionPlan +
      "' (served) and '" + kConventionRobotNow +
      "' (refused: needs odometry). Tracker and runtime disagree about what the observation "
      "MEANS, which no shape check can see.");
}

const char* ConventionForAnchor(WindowAnchor anchor) {
  return anchor == WindowAnchor::kPlan ? kConventionPlan : kConventionRobotNow;
}

std::string BlockName(ReferenceBlock block) {
  switch (block) {
    case ReferenceBlock::kBasePose:
      return "base_pose";
    case ReferenceBlock::kBaseVel:
      return "base_vel";
    case ReferenceBlock::kJointPos:
      return "joint_pos";
    case ReferenceBlock::kJointVel:
      return "joint_vel";
    case ReferenceBlock::kEePos:
      return "ee_pos";
  }
  throw std::runtime_error("unreachable: unknown reference block");
}

ReferenceBlock BlockFromName(const std::string& name) {
  for (ReferenceBlock b : kReferenceWindowBlocks) {
    if (BlockName(b) == name) return b;
  }
  throw std::runtime_error(
      "unknown reference-window block '" + name +
      "'; expected one of base_pose, base_vel, joint_pos, joint_vel, ee_pos");
}

std::vector<ReferenceBlock> ResolveReferenceBlocks(const std::vector<ReferenceBlock>& blocks) {
  if (blocks.empty()) throw std::runtime_error("a reference window needs at least one block");
  std::vector<ReferenceBlock> out;
  for (ReferenceBlock canonical : kReferenceWindowBlocks) {
    if (std::find(blocks.begin(), blocks.end(), canonical) != blocks.end()) {
      out.push_back(canonical);
    }
  }
  return out;
}

std::vector<BlockSlice> WindowBlockSlices(int num_joints, int num_ee_frames,
                                          const std::vector<ReferenceBlock>& blocks) {
  std::vector<BlockSlice> out;
  int offset = 0;
  for (ReferenceBlock b : ResolveReferenceBlocks(blocks)) {
    const int width = BlockWidth(b, num_joints, num_ee_frames);
    out.push_back({b, offset, width});
    offset += width;
  }
  return out;
}

int WindowDim(int num_joints, int num_ee_frames, const std::vector<ReferenceBlock>& blocks) {
  int total = 0;
  for (const BlockSlice& s : WindowBlockSlices(num_joints, num_ee_frames, blocks)) {
    total += s.width;
  }
  return total;
}

void MatrixToRot6d(const Mat3& matrix, double* out6) {
  for (int r = 0; r < 3; ++r) out6[r] = matrix(r, 0);
  for (int r = 0; r < 3; ++r) out6[3 + r] = matrix(r, 1);
}

Mat3 Rot6dToMatrix(const double* d6) {
  const Vec3 a(d6[0], d6[1], d6[2]);
  const Vec3 b(d6[3], d6[4], d6[5]);
  const Vec3 b1 = Unit(a);
  // Gram-Schmidt: remove b1's component from b. The dot is summed in the same order numpy's
  // (b1 * b).sum(-1) walks the axis.
  const double dot = b1.x() * b.x() + b1.y() * b.y() + b1.z() * b.z();
  const Vec3 b2 = Unit(b - dot * b1);
  const Vec3 b3 = Cross(b1, b2);
  Mat3 m;
  m.col(0) = b1;
  m.col(1) = b2;
  m.col(2) = b3;
  return m;
}

void QuatToRot6d(const QuatXyzw& q, double* out6) { MatrixToRot6d(MatrixFromQuat(q), out6); }

QuatXyzw Rot6dToQuat(const double* d6) { return QuatFromMatrix(Rot6dToMatrix(d6)); }

void WorldToWindow(const Vec3& root_pos_w, const QuatXyzw& root_quat_w,
                   const std::vector<WorldNode>& nodes, int num_joints, int num_ee_frames,
                   const std::vector<ReferenceBlock>& blocks, double* out) {
  const auto slices = WindowBlockSlices(num_joints, num_ee_frames, blocks);
  const int node_dim = WindowDim(num_joints, num_ee_frames, blocks);
  // Yaw-only anchor, matching the torch original: heading, never roll or pitch.
  const QuatXyzw act_q = YawQuat(root_quat_w);
  const Vec3& act_p = root_pos_w;

  for (size_t n = 0; n < nodes.size(); ++n) {
    const WorldNode& node = nodes[n];
    double* row = out + n * static_cast<size_t>(node_dim);
    for (const BlockSlice& s : slices) {
      double* dst = row + s.offset;
      switch (s.block) {
        case ReferenceBlock::kBasePose: {
          const Vec3 pos = QuatApplyInverse(act_q, node.ref_pos_w - act_p);
          dst[0] = pos.x();
          dst[1] = pos.y();
          dst[2] = pos.z();
          // R_yawT R_ref
          QuatToRot6d(QuatMul(QuatConjugate(act_q), node.ref_quat_w), dst + 3);
          break;
        }
        case ReferenceBlock::kBaseVel: {
          const Vec3 lin = QuatApplyInverse(act_q, node.ref_lin_w);
          const Vec3 ang = QuatApplyInverse(act_q, node.ref_ang_w);
          dst[0] = lin.x(); dst[1] = lin.y(); dst[2] = lin.z();
          dst[3] = ang.x(); dst[4] = ang.y(); dst[5] = ang.z();
          break;
        }
        case ReferenceBlock::kJointPos:
          std::copy(node.joint_pos.begin(), node.joint_pos.end(), dst);
          break;
        case ReferenceBlock::kJointVel:
          std::copy(node.joint_vel.begin(), node.joint_vel.end(), dst);
          break;
        case ReferenceBlock::kEePos: {
          // EE positions relative to THAT node's own reference base -- not the anchor, which
          // is what makes this block anchor-invariant and so a plain copy in ReanchorWindow.
          for (int e = 0; e < num_ee_frames; ++e) {
            const Vec3 rel =
                QuatApplyInverse(node.ref_quat_w, node.ee_pos_w[e] - node.ref_pos_w);
            dst[3 * e + 0] = rel.x();
            dst[3 * e + 1] = rel.y();
            dst[3 * e + 2] = rel.z();
          }
          break;
        }
      }
    }
  }
}

void ReanchorWindow(const double* in, int num_nodes, const Vec3& old_pos_w,
                    const QuatXyzw& old_quat_w, const Vec3& new_pos_w,
                    const QuatXyzw& new_quat_w, int num_joints, int num_ee_frames,
                    const std::vector<ReferenceBlock>& blocks, double* out) {
  const auto slices = WindowBlockSlices(num_joints, num_ee_frames, blocks);
  const int node_dim = WindowDim(num_joints, num_ee_frames, blocks);
  const QuatXyzw old_q = YawQuat(old_quat_w);
  const QuatXyzw new_q = YawQuat(new_quat_w);
  // R_newT R_old. Both anchors are yaw-only, so this is a pure heading rotation.
  const QuatXyzw q_delta = QuatMul(QuatConjugate(new_q), old_q);

  for (int n = 0; n < num_nodes; ++n) {
    const double* src = in + static_cast<size_t>(n) * node_dim;
    double* dst = out + static_cast<size_t>(n) * node_dim;
    for (const BlockSlice& s : slices) {
      const double* sb = src + s.offset;
      double* db = dst + s.offset;
      switch (s.block) {
        case ReferenceBlock::kBasePose: {
          // Position round-trips through the world; the orientation half does not need to.
          // R_yaw(new)T R_ref = (R_yaw(new)T R_yaw(old))(R_yaw(old)T R_ref), so each stored
          // COLUMN is just rotated by the anchor delta -- exactly linear, no Gram-Schmidt.
          const Vec3 ref_p = old_pos_w + QuatApply(old_q, Vec3(sb[0], sb[1], sb[2]));
          const Vec3 pos = QuatApplyInverse(new_q, ref_p - new_pos_w);
          const Vec3 c0 = QuatApply(q_delta, Vec3(sb[3], sb[4], sb[5]));
          const Vec3 c1 = QuatApply(q_delta, Vec3(sb[6], sb[7], sb[8]));
          db[0] = pos.x(); db[1] = pos.y(); db[2] = pos.z();
          db[3] = c0.x();  db[4] = c0.y();  db[5] = c0.z();
          db[6] = c1.x();  db[7] = c1.y();  db[8] = c1.z();
          break;
        }
        case ReferenceBlock::kBaseVel: {
          const Vec3 lin = QuatApplyInverse(new_q, QuatApply(old_q, Vec3(sb[0], sb[1], sb[2])));
          const Vec3 ang = QuatApplyInverse(new_q, QuatApply(old_q, Vec3(sb[3], sb[4], sb[5])));
          db[0] = lin.x(); db[1] = lin.y(); db[2] = lin.z();
          db[3] = ang.x(); db[4] = ang.y(); db[5] = ang.z();
          break;
        }
        default:
          // joint_pos / joint_vel / ee_pos are frame-invariant. `in` and `out` may alias, so
          // this must stay a forward copy of a non-overlapping range, which it is.
          if (db != sb) std::copy(sb, sb + s.width, db);
          break;
      }
    }
  }
}

}  // namespace g1_deploy
