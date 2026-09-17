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
#include <array>
#include <string>
#include <utility>
#include <vector>

#include "g1_deploy/robot_asset.h"

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

/// One camera's contract: where it is, what it sees, and how its depth is normalised.
///
/// Derived by mdp.depth_camera_specs from the sensor cfgs and obs terms, and embedded in BOTH
/// artifacts so nothing off-sim restates a camera number. `obs_term` is what ties a reading to
/// its slot in the observation.
struct DepthCameraSpec {
  std::string sensor;
  std::string obs_term;
  std::string mount_body;
  int width = 0;
  int height = 0;
  double offset = 0.75;
  double scale = 0.5;
  double min_dist = 0.2;
  double max_dist = 9.0;
  double min_clip_val = 0.0;
  double max_clip_val = 0.0;
  /// Pinhole intrinsics, in pixels. Needed to turn a reading into a DIRECTION, which is what
  /// the generator's xyz channels are built from. Absent on a contract exported before the
  /// channels existed, which is why they default rather than throw.
  double fx = 1.0;
  double fy = 1.0;
  double cx = 0.0;
  double cy = 0.0;
  /// The camera's NOMINAL pose on its mount body: where a calibrated robot believes it is.
  /// Never a per-episode jitter draw -- the training term used the nominal pose deliberately,
  /// so that the domain randomisation perturbs the IMAGE relative to the stated geometry,
  /// which is what calibration error actually is.
  std::array<double, 3> offset_pos{{0.0, 0.0, 0.0}};
  std::array<double, 4> offset_quat_xyzw{{0.0, 0.0, 0.0, 1.0}};

  int pixels() const { return width * height; }
};

/// The PHYSICAL camera a contract entry describes, as a name two artifacts can match on.
///
/// Mirrors `transfer/deploy/depth_obs.camera_key`, which is the oracle. `sensor` is the scene
/// name and is the identity; `obs_term` is the fallback for a contract carrying no sensor
/// name; a leading `fm_` is stripped because `install_fm_depth_group` names the GENERATOR's
/// own sampling of a lens `fm_depth_near` -- a second grid on the SAME mount, not a second
/// camera.
///
/// Matching on raw `obs_term` instead is why a correctly exported subset pair was refused on
/// every tick: the tracker says `depth_near`, the generator says `fm_depth_near`, and the two
/// name one lens. The python runtime accepted the same pair, so the pair validated at export
/// and then failed only in the C++ controller.
std::string CameraKey(const DepthCameraSpec& cam);


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
  //: Per-joint, in joint_names_isaac order. Resolved from the metadata's PATTERN table.
  std::vector<double> action_scale;
  //: Per-joint offset the action is added to, in joint_names_isaac order.
  std::vector<double> default_joint_angles;
  //: The generator re-plan period, in control steps. Absent for a tracker-only artifact;
  //: GenTrackerPolicy raises rather than defaulting when an fm.onnx is loaded without it,
  //: because a wrong cadence is a silent tracking failure.
  std::optional<int> fm_update_every;
  std::optional<ReferenceWindowSpec> reference_window;
  std::optional<CommandWindowSpec> command_window;
  //: In CONTRACT order -- the order a caller must supply readings in.
  std::vector<DepthCameraSpec> depth_cameras;
  //: True if the policy observes a height scan instead of (or as well as) depth.
  bool needs_height_scan = false;
  //: WHICH ROBOT this policy was trained on. Nothing else in the exported set says: joint
  //: names and their order are unchanged by a payload, a hand swap or a gear ratio, so
  //: `joint_names_isaac` -- the only field that came close -- matches across all of them.
  //: Compared against the generator's at load.
  RobotAsset robot_asset;

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
