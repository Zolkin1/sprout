// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// JSON -> DepthCameraSpec, in ONE place.
//
// Same reasoning as robot_asset_json.h, and the same shape: src/ rather than include/ because
// nlohmann_json is PRIVATE to g1_deploy, and a TEMPLATE because the two callers use different
// json types.
//
// The two callers read the SAME block from DIFFERENT artifacts, which is the whole point:
// `mdp.depth_camera_specs` is the single producer and stamps it into `tracker.onnx`
// (`policy_parameters.depth_camera_spec` -- the cameras the POLICY reads) and into `fm.onnx`
// (`fm_meta.extero_spec` -- the cameras the GENERATOR reads). Since the policy went near-only
// those are no longer the same set: the generator's is the whole RIG and the tracker's is a
// SUBSET of it. Two parsers is how a camera number would come to differ between the policy and
// the generator looking through the same lens.

#include <cstddef>

#include "g1_deploy/policy_params.h"

namespace g1_deploy {

/// Read one camera's contract block.
///
/// Throws (via nlohmann's `at`) when a required field is missing: a camera whose width,
/// normalisation or range had to be guessed is not a camera this runtime will serve.
template <typename Json>
DepthCameraSpec ParseDepthCameraSpecJson(const Json& cam) {
  // ONE parser for BOTH artifacts. `mdp.depth_camera_specs` is the single producer of this
  // block and stamps the same shape into `tracker.onnx` (`policy_parameters.depth_camera_spec`)
  // and `fm.onnx` (`fm_meta.extero_spec`) -- so the generator's rig and the tracker's subset
  // are the same struct, read the same way. Two parsers is how they would drift, and the
  // failure would be a camera number disagreeing between the policy and the generator that
  // observes the same lens.
  DepthCameraSpec d;
  if (cam.contains("sensor")) d.sensor = cam.at("sensor").template get<std::string>();
  d.obs_term = cam.at("obs_term").template get<std::string>();
  if (cam.contains("mount_body")) d.mount_body = cam.at("mount_body").template get<std::string>();
  d.width = cam.at("width").template get<int>();
  d.height = cam.at("height").template get<int>();
  d.offset = cam.at("offset").template get<double>();
  d.scale = cam.at("scale").template get<double>();
  d.min_dist = cam.at("min_dist").template get<double>();
  d.max_dist = cam.at("max_dist").template get<double>();
  d.min_clip_val = cam.at("min_clip_val").template get<double>();
  d.max_clip_val = cam.at("max_clip_val").template get<double>();
  // Optional: a contract exported before the xyz channels carries no intrinsics, and a
  // 1-channel generator never asks for them. A multi-channel one does, and is refused
  // at the point of use rather than here, where the reason would be less obvious.
  if (cam.contains("fx")) d.fx = cam.at("fx").template get<double>();
  if (cam.contains("fy")) d.fy = cam.at("fy").template get<double>();
  if (cam.contains("cx")) d.cx = cam.at("cx").template get<double>();
  if (cam.contains("cy")) d.cy = cam.at("cy").template get<double>();
  if (cam.contains("offset_pos")) {
    for (size_t i = 0; i < 3 && i < cam.at("offset_pos").size(); ++i) {
      d.offset_pos[i] = cam.at("offset_pos")[i].template get<double>();
    }
  }
  if (cam.contains("offset_quat_xyzw")) {
    for (size_t i = 0; i < 4 && i < cam.at("offset_quat_xyzw").size(); ++i) {
      d.offset_quat_xyzw[i] = cam.at("offset_quat_xyzw")[i].template get<double>();
    }
  }
  return d;
}

}  // namespace g1_deploy
