// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Parser for tracker.onnx's embedded `policy_parameters` JSON -- the contract between the
// training run and this runtime. Port of the metadata half of
// transfer/deploy/tracker_session.py.
//
// Deliberately takes the JSON as a STRING rather than reaching into onnxruntime itself, so
// the parsing is testable against a recorded artifact with no ORT, no GPU and no model load.
// The ORT-facing half (pull the string out of the session) is a two-line caller.
//
// ============================ the trap this file exists to avoid ==========================
// The observation terms are walked in the JSON's INSERTION order, and that order IS the
// layout of the observation vector the policy was trained on. nlohmann::json's default object
// type is a std::map, which SORTS keys. For the live artifact that is the difference between
//
//   base_ang_vel, projected_gravity, joint_pos, joint_vel, actions, depth_near, depth_far,
//   reference_window, command_window                                    <- correct
//   actions, base_ang_vel, command_window, depth_far, depth_near, ...   <- sorted
//
// Both produce a correctly sized 3101-element vector. Every shape check downstream passes.
// The policy just reads its inputs permuted, forever. Hence nlohmann::ordered_json, and hence
// a parity test against the real artifact rather than a synthetic one.

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace g1_deploy {

/// One policy observation term, resolved.
struct ObsTerm {
  std::string name;
  int width = 0;               //: FLAT width; multi-dimensional terms are flattened
  int history_length = 0;
  //: Per-element scale. Size 1 means "broadcast to the whole term"; otherwise size == width.
  //: Not a scalar, because it genuinely is not: reference_window carries one entry per
  //: element while the proprioceptive terms carry a single number.
  std::vector<double> scale;
  //: Training clip MAGNITUDE, absent when the term was unclipped. Carried per term so the
  //: deploy saturates exactly as training did instead of consulting a hardcoded table.
  std::optional<double> clip;
};

/// The reference-window contract: how a node is packed and where it is sampled.
struct ReferenceWindowSpec {
  std::vector<int> offsets;
  std::vector<std::pair<std::string, int>> blocks;  //: (name, width), in packing order
  std::vector<std::string> joint_order;
  std::vector<std::string> ee_frame_names;
  //: REQUIRED. Which way round base_pose's orientation half points. A tracker trained under
  //: the other convention loads clean -- same width, same everything -- and steers wrong.
  std::string base_pose_convention;
};

/// The command-window contract, when the policy observes one.
struct CommandWindowSpec {
  std::vector<int> offsets;
  int cmd_dim = 0;
};

/// Everything this runtime reads out of tracker.onnx's metadata.
struct PolicyParams {
  std::vector<ObsTerm> obs_terms;  //: IN ORDER. See the header comment.
  std::vector<std::string> joint_names_isaac;
  std::vector<std::string> joint_out_names;
  int num_obs = 0;
  int num_actions = 0;
  double dt = 0.0;
  std::vector<double> kp;
  std::vector<double> kd;
  //: The generator re-plan period, in control steps. Absent for a tracker-only artifact;
  //: GenTrackerPolicy raises rather than defaulting when an fm.onnx is loaded without it,
  //: because a wrong cadence is a silent tracking failure.
  std::optional<int> fm_update_every;
  std::optional<ReferenceWindowSpec> reference_window;
  std::optional<CommandWindowSpec> command_window;

  /// Sum of the term widths. Must equal num_obs; a mismatch means the layout and the graph
  /// disagree, which no downstream check would catch in time.
  int TotalObsWidth() const;
};

/// Parse the raw `policy_parameters` JSON string.
///
/// Throws std::runtime_error on malformed JSON, a missing required field, or a term whose
/// declared scale is neither a scalar nor exactly `width` long.
PolicyParams ParsePolicyParameters(const std::string& json);

}  // namespace g1_deploy
