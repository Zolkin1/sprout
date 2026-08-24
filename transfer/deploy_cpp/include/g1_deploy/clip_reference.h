// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Port of transfer/deploy/clip_reference.py: a baked motion clip served as tracker
// references, with the generator out of the loop entirely.
//
// This is what lets a bring-up ask "is anything downstream of the observation working at
// all?" through the SAME runtime the generator path uses. It used to be answerable only
// through the deprecated python controller, so a clip case swapped the controller LANGUAGE
// at the same time as it removed the generator and isolated neither.
//
// THE CLIP ARRIVES AS A PACK, NOT AS HDF5 (see tools/dump_clip_pack.py): the obelisk image
// ships no HDF5 and the library is lzf-compressed, which the plain HDF5 C library cannot
// decode. Channels are resolved BY NAME against the tracker's own joint and frame names, so
// a pack built for a different robot fails loudly rather than reading the right columns of
// the wrong motion.
//
// ODOMETRY: none is read, and that is a property of the CADENCE rather than of this file.
// Under window_anchor="plan" the caller latches the window anchor and re-places the clip on
// the same step, and the robot's world pose then cancels exactly out of every block --
// base_pose reduces to Rz(-yaw_latch) (p_clip(s+o) - p_clip(s_latch)). PolicyRuntime enforces
// the cadence agreement; ClipReferenceSource does the latching. See
// GenTrackerPolicy.assert_clip_reference_compatible for the same rule stated on the python
// side, and deploy/tests/test_clip_window_anchor.py for the measurement.

#include <string>
#include <vector>

#include "g1_deploy/types.h"
#include "g1_deploy/window.h"

namespace g1_deploy {

/// Serve a baked motion clip as world-frame tracker references.
class ClipReference {
 public:
  /// How to open one case of a pack.
  struct Config {
    std::string pack_dir;                          //: written by dump_clip_pack.py
    std::string case_name;                         //: e.g. "walk_backward"
    std::vector<std::string> joint_names;          //: the tracker's order; jp/jv come back in it
    std::vector<std::string> ee_frame_names;       //: the tracker's ee target order
    std::vector<std::string> root_frame_bodies;    //: refused non-empty; see Load
    //: Empty (the default) takes the clip's OWN ref_frame out of the pack.
    std::string base_frame_name;
    double start_time = 0.0;                       //: seconds into the clip to begin
    /// Re-anchor cadence, in control steps. MUST equal the tracker's fm_update_every under
    /// window_anchor="plan"; PolicyRuntime checks it, and Serve() is what makes the two fire
    /// on the same tick.
    int ego_anchor_every = 1;
  };

  /// Read one case.
  ///
  /// \throws std::runtime_error On a missing pack, an unknown case (naming the ones present),
  ///   a channel the pack does not carry (naming it), a dt/width disagreement, or a non-empty
  ///   root_frame_bodies -- a tracking_mode="body" tracker needs per-body root-frame channels
  ///   that this path does not serve, and ObsAssembler would otherwise fail much later.
  static ClipReference Load(const Config& cfg);

  double dt() const { return dt_; }
  int num_steps() const { return num_steps_; }
  /// Steps of real motion; the rest of the array is repeat-last padding.
  int real_steps() const { return real_steps_; }
  /// Step playback begins at, from Config::start_time.
  int start_step() const { return start_step_; }
  /// First and one-past-last step of the MOTION, excluding the holds a debug case is
  /// bracketed with. Equal to (0, real_steps) when the pack declares no span.
  int motion_start() const { return motion_start_; }
  int motion_end() const { return motion_end_; }
  /// Whether `step` is inside a bracketing hold, where the command must be zero.
  bool InHold(int step) const { return step < motion_start_ || step >= motion_end_; }

  /// Set the clip's world placement (yaw-only ground frame).
  void SetPlacement(const Vec3& anchor_pos, const QuatXyzw& anchor_quat_xyzw);

  /// Re-place the clip so its `step` base lands on the robot's root (planar + heading + z).
  ///
  /// Mirrors IsaacLab MotionSequenceCommand::_reanchor_ego_envs / world_lift.ego_placement.
  void ReanchorEgo(int step, const Vec3& robot_pos_w, const QuatXyzw& robot_quat_xyzw);

  /// World-frame reference at `step`, placement-lifted. Saturates on the last REAL step.
  WorldNode FrameState(int step) const;

  /// Commanded base velocity (vx, vy, wz) at `step`. Zero inside a bracketing hold: playback
  /// saturates on its last played frame rather than stopping, and the padding repeats that
  /// frame's COMMAND -- so a case ending mid-walk would otherwise hold a walk command against
  /// a reference that has stopped moving, and put the robot down for reasons that are the
  /// harness's rather than the policy's.
  Vec3 CommandAt(int step) const;

  /// One node per offset, offset 0 (the current step) first.
  void ReferenceAt(int step, const std::vector<int>& offsets,
                   std::vector<WorldNode>* out) const;

  /// Each offset's own command, concatenated: [offset0(3), offset1(3), ...].
  ///
  /// The clip authored a command per step, so command_window is NOT the current command
  /// tiled the way it is on the generator path.
  void CommandWindow(int step, const std::vector<int>& offsets, std::vector<double>* out) const;

  /// One playback tick: re-place the clip, latch the anchor, and serve the window.
  ///
  /// THE CADENCE COUPLING LIVES HERE, in one method, on purpose. The clip's placement and the
  /// window's anchor are both solved from `robot_pos`/`robot_quat` on the same tick, and that
  /// is the entire reason the served window is a pure function of the clip: the robot's world
  /// pose cancels out of every block, so nothing downstream reads odometry. Split across two
  /// call sites it drifts apart silently -- which is exactly what happened on the python side,
  /// where a clip re-anchored every step was served against an anchor latched every 12 and up
  /// to 11 steps of egomotion leaked into base_pose (deploy/tests/test_clip_window_anchor.py
  /// measures both arms).
  ///
  /// \param tick Playback ticks elapsed. NOT the clip step: it starts at 0 whatever
  ///   `start_time` was, so the cadence phase moves with the start. It must FREEZE when
  ///   playback freezes, or the latch slides by the length of the stall.
  /// \param window_out `offsets.size() * WindowDim(...)` doubles, row-major.
  /// \param command_out Each offset's own command; may be null.
  void Serve(int tick, const Vec3& robot_pos, const QuatXyzw& robot_quat,
             const std::vector<int>& offsets, int num_joints, int num_ee_frames,
             const std::vector<ReferenceBlock>& blocks, double* window_out,
             std::vector<double>* command_out);

  /// The clip step a playback tick serves.
  int StepForTick(int tick) const { return start_step_ + tick; }

  /// The anchor the last Serve() latched.
  const Vec3& anchor_pos() const { return latched_pos_; }
  const QuatXyzw& anchor_quat() const { return latched_quat_; }

 private:
  /// Column indices of one named channel group, resolved against the pack's name lists.
  struct Channels {
    std::vector<int> base_pos, base_ori, base_lin, base_ang, jp, jv;
    std::vector<std::vector<int>> ee_pos;
  };

  Vec3 LiftPos(const Vec3& p) const;
  QuatXyzw LiftQuat(const QuatXyzw& q) const;
  Vec3 LiftVec(const Vec3& v) const;

  std::vector<double> y_;        //: [T, P] clip-world position outputs
  std::vector<double> dy_;       //: [T, V]
  std::vector<double> tv_;       //: [T, 3]
  int pos_width_ = 0, vel_width_ = 0;
  int num_steps_ = 0, real_steps_ = 0, start_step_ = 0;
  int motion_start_ = 0, motion_end_ = 0;
  double dt_ = 0.0;
  Channels ch_;
  int ego_anchor_every_ = 1;
  Vec3 anchor_pos_ = Vec3::Zero();
  QuatXyzw inv_yaw_ = QuatXyzw(0.0, 0.0, 0.0, 1.0);
  bool latched_ = false;
  Vec3 latched_pos_ = Vec3::Zero();
  QuatXyzw latched_quat_ = QuatXyzw(0.0, 0.0, 0.0, 1.0);
  std::vector<WorldNode> scratch_;   //: reused per Serve, so a 50 Hz tick allocates nothing
};

}  // namespace g1_deploy
