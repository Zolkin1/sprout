// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Port of transfer/deploy/window_np.py: the reference-window representation.
//
// One node packs, in this order:
//
//     [ base_pose(9) | base_vel(6) | joint_pos(nj) | joint_vel(nj) | ee_pos(3 * E) ]
//
// base_pose is [ R_yawT (p_ref - p_act), rot6d(R_yawT R_ref) ] -- the reference pose in the
// anchor's YAW (heading) frame, both halves the same way round. The anchor supplies heading
// only, never roll or pitch, so a leaning robot does not rotate the frame its reference is
// observed in.
//
// A teacher can drop blocks it does not need, so the selection is data, not a constant: the
// live artifact carries base_pose + base_vel + joint_pos only (D = 44 at nj = 29). Widths
// come from metadata for the same reason -- hard-coding 29 or 44 here would make the C++
// refuse a policy the python accepts.
//
// Two maps, matching the python:
//
//   * WorldToWindow  -- the FORWARD map, needed for the clip reference path.
//   * ReanchorWindow -- re-express nodes against a DIFFERENT anchor, needed to serve a
//                       prediction more than one control step after it was made.
//
// The full inverse (window_to_world) is deliberately not ported, as in the python: it cannot
// be completed for a block subset at all, since the return leg needs every block.

#include <string>
#include <vector>

#include "g1_deploy/types.h"

namespace g1_deploy {

/// The blocks a window node can pack, in canonical packing order.
enum class ReferenceBlock { kBasePose, kBaseVel, kJointPos, kJointVel, kEePos };

/// Canonical order. Mirrors REFERENCE_WINDOW_BLOCK_NAMES.
inline constexpr ReferenceBlock kReferenceWindowBlocks[] = {
    ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel, ReferenceBlock::kJointPos,
    ReferenceBlock::kJointVel, ReferenceBlock::kEePos};

/// Wire name of a block, as it appears in the ONNX metadata and the fixtures.
std::string BlockName(ReferenceBlock block);

/// Parse a wire name. Throws std::runtime_error on an unknown name.
ReferenceBlock BlockFromName(const std::string& name);

/// One block's placement inside a node.
struct BlockSlice {
  ReferenceBlock block;
  int offset;
  int width;
};

/// Validate a selection and put it in canonical packing order.
///
/// An empty selection or an unknown name throws: the layout is the contract between the
/// generator and the tracker, and a silently-reordered one reads velocities as positions.
std::vector<ReferenceBlock> ResolveReferenceBlocks(const std::vector<ReferenceBlock>& blocks);

/// Per-node layout in packing order, with each block's offset and width.
std::vector<BlockSlice> WindowBlockSlices(int num_joints, int num_ee_frames,
                                          const std::vector<ReferenceBlock>& blocks);

/// Width D of one node for the given selection.
int WindowDim(int num_joints, int num_ee_frames, const std::vector<ReferenceBlock>& blocks);

/// First two COLUMNS of a rotation matrix, concatenated: [..., 6].
void MatrixToRot6d(const Mat3& matrix, double* out6);

/// Rotation matrix from the 6D representation, via Gram-Schmidt.
///
/// Robust to non-orthonormal input, which is the point: the generator emits a raw 9-vector
/// and only approximately satisfies the constraint. The recovered vectors are COLUMNS.
Mat3 Rot6dToMatrix(const double* d6);

/// 6D rotation representation of a quaternion.
void QuatToRot6d(const QuatXyzw& q, double* out6);

/// Quaternion from the 6D rotation representation.
QuatXyzw Rot6dToQuat(const double* d6);

/// Reference state of one node in the world frame, the input to WorldToWindow.
struct WorldNode {
  Vec3 ref_pos_w;
  QuatXyzw ref_quat_w;
  Vec3 ref_lin_w;
  Vec3 ref_ang_w;
  std::vector<double> joint_pos;   //: nj, robot joint order
  std::vector<double> joint_vel;   //: nj
  std::vector<Vec3> ee_pos_w;      //: E
};

/// Build window nodes from world-frame reference state and a measured anchor.
///
/// Writes `nodes.size()` rows of D doubles into `out`, row-major.
///
/// Args:
///   root_pos_w / root_quat_w: the anchor (the robot's measured base), world frame. Only its
///     YAW is used.
void WorldToWindow(const Vec3& root_pos_w, const QuatXyzw& root_quat_w,
                   const std::vector<WorldNode>& nodes, int num_joints, int num_ee_frames,
                   const std::vector<ReferenceBlock>& blocks, double* out);

/// Re-express window nodes against a DIFFERENT anchor, block by block.
///
/// A window is relative to one measured robot pose. Serving the same nodes a few control
/// steps later means expressing them against where the robot is NOW, and only the blocks
/// that reference the anchor need touching: base_pose round-trips through the world,
/// base_vel is a pure rotation, and joint_pos / joint_vel / ee_pos are frame-invariant and
/// copied.
///
/// Done per block rather than through the world and back, which is what makes it work for a
/// block SELECTION at all: the world round trip needs every block on the return leg, and a
/// window that drops joint_vel or ee_pos cannot supply them.
///
/// Only the RELATIVE transform between the anchors matters -- a common translation cancels
/// and a common rotation is equivariant. `in` and `out` may alias.
void ReanchorWindow(const double* in, int num_nodes, const Vec3& old_pos_w,
                    const QuatXyzw& old_quat_w, const Vec3& new_pos_w,
                    const QuatXyzw& new_quat_w, int num_joints, int num_ee_frames,
                    const std::vector<ReferenceBlock>& blocks, double* out);

}  // namespace g1_deploy
