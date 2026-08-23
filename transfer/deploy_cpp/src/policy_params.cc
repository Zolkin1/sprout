// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/policy_params.h"

#include "g1_deploy/depth_obs.h"
#include "robot_asset_json.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <regex>
#include <stdexcept>

namespace g1_deploy {
namespace {

// ordered_json, NOT json: the default object type sorts keys, and the observation term order
// IS the observation layout. See the header.
using OrderedJson = nlohmann::ordered_json;

/// Flat width of a term's declared shape. A shape may be a bare number or a list (the
/// per-camera image groups export [1, H, W]); either way the observation manager concatenates
/// the flattened term, so the product is what matters.
int FlatDim(const OrderedJson& shape) {
  if (shape.is_number()) return shape.get<int>();
  if (!shape.is_array()) throw std::runtime_error("term shape is neither a number nor a list");
  int n = 1;
  for (const auto& d : shape) n *= d.get<int>();
  return n;
}

std::vector<double> ParseScale(const OrderedJson& term, int width, const std::string& name) {
  if (!term.contains("scale")) return {1.0};
  const auto& scale = term.at("scale");
  if (scale.is_number()) return {scale.get<double>()};
  if (scale.is_array()) {
    std::vector<double> out;
    out.reserve(scale.size());
    for (const auto& v : scale) out.push_back(v.get<double>());
    if (out.size() != static_cast<size_t>(width) && out.size() != 1) {
      throw std::runtime_error("observation term '" + name + "' has a scale of " +
                               std::to_string(out.size()) + " entries but a width of " +
                               std::to_string(width));
    }
    return out;
  }
  throw std::runtime_error("observation term '" + name + "' has a non-numeric scale");
}

std::vector<std::string> StringList(const OrderedJson& node) {
  std::vector<std::string> out;
  for (const auto& v : node) out.push_back(v.get<std::string>());
  return out;
}

std::vector<double> DoubleList(const OrderedJson& node) {
  std::vector<double> out;
  for (const auto& v : node) out.push_back(v.get<double>());
  return out;
}

std::vector<int> IntList(const OrderedJson& node) {
  std::vector<int> out;
  for (const auto& v : node) out.push_back(v.get<int>());
  return out;
}

}  // namespace

int PolicyParams::TotalObsWidth() const {
  int total = 0;
  for (const ObsTerm& t : obs_terms) total += t.width;
  return total;
}

PolicyParams ParsePolicyParameters(const std::string& json) {
  OrderedJson root;
  try {
    root = OrderedJson::parse(json);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("policy_parameters is not valid JSON: ") + e.what());
  }

  PolicyParams p;
  if (!root.contains("observation_terms") || !root["observation_terms"].contains("policy")) {
    throw std::runtime_error("policy_parameters has no observation_terms.policy");
  }
  for (const auto& [name, term] : root["observation_terms"]["policy"].items()) {
    ObsTerm t;
    t.name = name;
    t.width = FlatDim(term.at("shape"));
    t.history_length = term.contains("history_length") ? term.at("history_length").get<int>() : 0;
    t.scale = ParseScale(term, t.width, name);
    // A clip arrives as [lo, hi]; the runtime saturates symmetrically at the larger
    // magnitude, matching the python. `null` means the term was not clipped in training.
    if (term.contains("clip") && !term.at("clip").is_null()) {
      const auto& c = term.at("clip");
      t.clip = std::max(std::abs(c.at(0).get<double>()), std::abs(c.at(1).get<double>()));
    }
    p.obs_terms.push_back(std::move(t));
  }

  p.joint_names_isaac = StringList(root.at("joint_names_isaac"));
  p.num_obs = root.at("num_obs").get<int>();
  p.num_actions = root.at("num_actions").get<int>();
  p.dt = root.at("dt").get<double>();
  p.kp = DoubleList(root.at("kp"));
  p.kd = DoubleList(root.at("kd"));
  if (root.contains("default_joint_angles")) {
    p.default_joint_angles = DoubleList(root.at("default_joint_angles"));
  }

  // action_scale is a PATTERN table, not a per-joint list. Resolution rules that matter:
  //   * std::regex_match, not regex_search -- the python uses re.fullmatch, so a pattern has
  //     to match the WHOLE joint name. `.*hip.*` and `hip` behave very differently otherwise.
  //   * FIRST match in the table's own order wins, which is why the metadata is parsed with
  //     ordered_json: with a sorted map a broad pattern could outrank the specific one that
  //     was written after it.
  //   * an unmatched joint throws. A default of 0 would silently freeze that joint.
  if (root.contains("action_scale")) {
    const auto& patterns = root.at("action_scale");
    p.action_scale.reserve(p.joint_names_isaac.size());
    for (const std::string& joint : p.joint_names_isaac) {
      bool matched = false;
      for (const auto& [pattern, value] : patterns.items()) {
        if (std::regex_match(joint, std::regex(pattern))) {
          p.action_scale.push_back(value.get<double>());
          matched = true;
          break;
        }
      }
      if (!matched) {
        throw std::runtime_error("no action_scale pattern matches joint '" + joint + "'");
      }
    }
  }

  if (root.contains("fm_update_every") && !root["fm_update_every"].is_null()) {
    p.fm_update_every = root["fm_update_every"].get<int>();
  }
  // The legacy key is accepted because exports predating the rename still carry it, exactly
  // as tracker_session.py does.
  if (root.contains("joint_out_names") && !root["joint_out_names"].is_null()) {
    p.joint_out_names = StringList(root["joint_out_names"]);
  } else if (root.contains("ordered_pos_output_names")) {
    p.joint_out_names = StringList(root["ordered_pos_output_names"]);
  }

  if (root.contains("reference_window_spec") && !root["reference_window_spec"].is_null()) {
    const auto& rw = root["reference_window_spec"];
    ReferenceWindowSpec spec;
    spec.offsets = IntList(rw.at("offsets"));
    for (const auto& b : rw.at("blocks")) {
      spec.blocks.emplace_back(b.at(0).get<std::string>(), b.at(1).get<int>());
    }
    if (rw.contains("joint_order")) spec.joint_order = StringList(rw.at("joint_order"));
    if (rw.contains("ee_frame_names") && !rw["ee_frame_names"].is_null()) {
      spec.ee_frame_names = StringList(rw["ee_frame_names"]);
    }
    // REQUIRED, not defaulted. An artifact predating the key was exported under the legacy
    // mixed convention, and defaulting would wave every one of them through as if it agreed.
    if (!rw.contains("base_pose_convention") || rw["base_pose_convention"].is_null()) {
      throw std::runtime_error(
          "tracker.onnx reference_window_spec carries no base_pose_convention. It predates "
          "the key, so it was exported under the legacy mixed convention (position "
          "ref-minus-actual, orientation actual-minus-ref). Re-export from the training run "
          "if the policy was trained under this runtime's convention; retrain if it was not.");
    }
    spec.base_pose_convention = rw["base_pose_convention"].get<std::string>();
    p.reference_window = std::move(spec);
  }

  if (root.contains("command_window_spec") && !root["command_window_spec"].is_null()) {
    const auto& cw = root["command_window_spec"];
    CommandWindowSpec spec;
    spec.offsets = IntList(cw.at("offsets"));
    spec.cmd_dim = cw.at("cmd_dim").get<int>();
    p.command_window = std::move(spec);
  }

  // The camera contract. Order is the CONTRACT order a caller supplies readings in, so it is
  // read from the array as written -- not sorted, and not keyed by name.
  if (root.contains("depth_camera_spec") && root["depth_camera_spec"].is_array()) {
    for (const auto& cam : root["depth_camera_spec"]) {
      DepthCameraSpec d;
      if (cam.contains("sensor")) d.sensor = cam.at("sensor").get<std::string>();
      d.obs_term = cam.at("obs_term").get<std::string>();
      if (cam.contains("mount_body")) d.mount_body = cam.at("mount_body").get<std::string>();
      d.width = cam.at("width").get<int>();
      d.height = cam.at("height").get<int>();
      d.offset = cam.at("offset").get<double>();
      d.scale = cam.at("scale").get<double>();
      d.min_dist = cam.at("min_dist").get<double>();
      d.max_dist = cam.at("max_dist").get<double>();
      d.min_clip_val = cam.at("min_clip_val").get<double>();
      d.max_clip_val = cam.at("max_clip_val").get<double>();
      // Optional: a contract exported before the xyz channels carries no intrinsics, and a
      // 1-channel generator never asks for them. A multi-channel one does, and is refused
      // at the point of use rather than here, where the reason would be less obvious.
      if (cam.contains("fx")) d.fx = cam.at("fx").get<double>();
      if (cam.contains("fy")) d.fy = cam.at("fy").get<double>();
      if (cam.contains("cx")) d.cx = cam.at("cx").get<double>();
      if (cam.contains("cy")) d.cy = cam.at("cy").get<double>();
      if (cam.contains("offset_pos")) {
        for (size_t i = 0; i < 3 && i < cam.at("offset_pos").size(); ++i) {
          d.offset_pos[i] = cam.at("offset_pos")[i].get<double>();
        }
      }
      if (cam.contains("offset_quat_xyzw")) {
        for (size_t i = 0; i < 4 && i < cam.at("offset_quat_xyzw").size(); ++i) {
          d.offset_quat_xyzw[i] = cam.at("offset_quat_xyzw")[i].get<double>();
        }
      }
      p.depth_cameras.push_back(std::move(d));
    }
  }
  // Whether a height scan is needed comes from the OBS TERMS, not from the presence of a
  // height_scan_spec: an artifact can carry the spec and not observe it.
  for (const ObsTerm& t : p.obs_terms) {
    if (t.name == "height_scan") p.needs_height_scan = true;
  }

  if (p.TotalObsWidth() != p.num_obs) {
    throw std::runtime_error("observation terms sum to " + std::to_string(p.TotalObsWidth()) +
                             " but the graph declares num_obs=" + std::to_string(p.num_obs));
  }
  // Same treatment as the fm side: optional, tolerated when absent. The comparison against
  // the generator's happens in PolicyRuntime::Load, where both artifacts are in hand.
  p.robot_asset = ParseRobotAsset(root);
  return p;
}

CameraGeom GeomFromSpec(const DepthCameraSpec& spec) {
  CameraGeom g;
  g.width = spec.width;
  g.height = spec.height;
  g.fx = spec.fx;
  g.fy = spec.fy;
  g.cx = spec.cx;
  g.cy = spec.cy;
  g.norm.offset = spec.offset;
  g.norm.scale = spec.scale;
  g.norm.min_dist = spec.min_dist;
  g.norm.max_dist = spec.max_dist;
  g.norm.min_clip_val = spec.min_clip_val;
  g.norm.max_clip_val = spec.max_clip_val;
  return g;
}

}  // namespace g1_deploy
