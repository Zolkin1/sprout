// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// WHICH ROBOT an artifact was made with, and the refusal of a pair that disagrees.
//
// Port of the comparison half of motion_gen/robot_asset.py, as consumed by
// transfer/deploy/gen_tracker.py::_assert_robot_assets_agree.
//
// A deploy box has no "current robot" to check an artifact against -- it has whatever hardware
// it is bolted to. What it CAN check is that the two halves of the pair agree: there is one
// physical robot, so a tracker and a generator running together cannot have been trained on
// two. The same argument, and the same shape, as AssertDepthSpecsAgree for the camera contract.
//
// This is not a theoretical hole. Joint names and their ORDER survive a payload change, a hand
// swap and a gear-ratio change -- `joint_names_isaac`, the only field that came close, matches
// across all of them -- so a mismatched pair loads clean and the only symptom is a robot that
// tracks slightly wrong. The perceptive G1 changed 39 body names, added 2.64 kg and rescaled
// hip-pitch actions by 1.56x while leaving all 29 joints exactly as they were.

#include <optional>
#include <string>
#include <vector>

namespace g1_deploy {

/// One artifact's robot fingerprint. Every field optional: the artifact carries what its
/// producer could see, and the comparison uses only what both sides have.
struct RobotAsset {
  //: False when the artifact carried no fingerprint at all, which is not the same as one whose
  //: fields all happen to be empty. Absent means "cannot compare"; empty would mean "compared
  //: and agreed", and agreeing by silence is exactly what this is meant to prevent.
  bool present = false;
  std::string short_hash;   //: the JSON's `short`, for messages
  std::vector<std::string> joint_names;
  std::vector<std::string> body_names;
  std::optional<double> total_mass_kg;
  std::string urdf_sha256;
  std::string mjcf_sha256;
  std::string usd_sha256;
};

/// Refuse a tracker/generator pair trained on DIFFERENT robots.
///
/// Silent when either side carries no fingerprint. Every artifact exported before the field
/// existed has none, and refusing those would break every pair on disk; this guards against a
/// NEW mismatch, it is not a migration gate.
///
/// Args:
///   tracker: `tracker.onnx`'s `policy_parameters.robot_asset`.
///   fm: `fm.onnx`'s `fm_meta.robot_asset`.
///   tracker_name / fm_name: paths, for the message.
///   allow_mismatch: WARN on stderr and load the pair anyway, instead of throwing. Off by
///     default and it must stay that way -- a wrong pairing is invisible to every shape check,
///     and this is what would catch a tracker exported against one robot running a generator
///     trained on another. Turn it on only for a delta someone has read and understood; the
///     warning names every differing field precisely so that is possible.
///
/// Throws:
///   std::runtime_error if both carry a fingerprint, a shared field differs, and
///   `allow_mismatch` is false.
void AssertRobotAssetsAgree(const RobotAsset& tracker, const RobotAsset& fm,
                            const std::string& tracker_name, const std::string& fm_name,
                            bool allow_mismatch = false);

}  // namespace g1_deploy
