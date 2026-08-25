// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// tracker.onnx: the policy network plus the contract embedded alongside it.
//
// Port of the session half of transfer/deploy/tracker_session.py. The metadata parsing lives
// in policy_params.h and is tested without a model; this is the part that needs one.

#include <string>
#include <vector>

#include "g1_deploy/ort_session.h"
#include "g1_deploy/policy_params.h"
#include "g1_deploy/window.h"  // ReferenceBlock, for the block-agreement check

namespace g1_deploy {

class TrackerSession {
 public:
  TrackerSession(const std::string& onnx_path, const std::vector<Provider>& providers,
                 int intra_threads);

  /// Run the policy. `obs` is num_obs floats; `action_out` receives num_actions RAW actions
  /// in the POLICY's joint order.
  void Infer(const float* obs, float* action_out);

  /// Run the policy and return joint POSITION TARGETS in `joint_names_out` order.
  ///
  /// target = raw_action * action_scale + default_joint_angles, then permuted. The permutation
  /// is resolved once and cached: the policy's joint order is interleaved
  /// (left_hip_pitch, right_hip_pitch, waist_yaw, ...) while the robot's is per limb, and 27
  /// of 29 positions differ on the G1. Subtracting one from the other gives a large, smooth,
  /// entirely fictitious error rather than an obvious failure.
  void GetAction(const float* obs, const std::vector<std::string>& joint_names_out,
                 double* targets_out);

  const PolicyParams& params() const { return params_; }
  OrtModel& model() { return model_; }
  /// The raw action from the last Infer, in the POLICY's joint order.
  const std::vector<float>& last_action() const { return last_action_; }
  /// Wall time of the last inference, milliseconds.
  ///
  /// Not atomic, unlike FmSession's: the tracker only ever runs on the control thread, so
  /// writer and reader are the same thread.
  double last_infer_ms() const { return last_infer_ms_; }

 private:
  OrtModel model_;
  PolicyParams params_;
  std::vector<float> obs_buf_;
  std::vector<float> last_action_;
  std::string in_name_;
  std::string out_name_;
  //: joint_names_out -> index into the policy's order. Rebuilt only when the names change.
  std::vector<int> out_permutation_;
  std::vector<std::string> cached_out_names_;
  std::vector<double> targets_isaac_;
  double last_infer_ms_ = 0.0;
};

/// Refuse a generator and a tracker that do not describe the same window.
///
/// The generator's output IS the tracker's reference_window observation, with no translation
/// layer between them, so the two have to be the same vector. A disagreement is invisible
/// where it matters: drop a block on one side only and the widths differ (caught late, as a
/// shape error deep in the observation assembly); swap two SAME-WIDTH blocks and nothing
/// differs in shape at all -- the policy just reads velocities as positions.
void AssertWindowBlocksAgree(const std::vector<std::pair<std::string, int>>& tracker_blocks,
                             const std::vector<ReferenceBlock>& fm_blocks);

}  // namespace g1_deploy
