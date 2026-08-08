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

  /// Run the policy. `obs` is num_obs floats; `action_out` receives num_actions.
  void Infer(const float* obs, float* action_out);

  const PolicyParams& params() const { return params_; }
  OrtModel& model() { return model_; }
  /// The raw action from the last Infer, in the POLICY's joint order.
  const std::vector<float>& last_action() const { return last_action_; }

 private:
  OrtModel model_;
  PolicyParams params_;
  std::vector<float> obs_buf_;
  std::vector<float> last_action_;
  std::string in_name_;
  std::string out_name_;
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
