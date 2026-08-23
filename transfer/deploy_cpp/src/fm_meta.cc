// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/fm_meta.h"

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "g1_deploy/depth_obs.h"
#include "robot_asset_json.h"

namespace g1_deploy {

// ordered_json, not json: the default object type SORTS keys, and key order is meaningful
// elsewhere in this metadata (see policy_params.h). Kept the same here so one parser cannot
// disagree with another about what the document says.
using OrderedJson = nlohmann::ordered_json;

namespace {

std::vector<int> IntList(const OrderedJson& node) {
  std::vector<int> out;
  for (const auto& v : node) out.push_back(v.get<int>());
  return out;
}
}  // namespace

FmMeta ParseFmMeta(const std::string& json) {
  OrderedJson root;
  try {
    root = OrderedJson::parse(json);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("fm_meta is not valid JSON: ") + e.what());
  }
  FmMeta m;
  // Optional and tolerated when absent: a graph exported before the key existed has none, and
  // AssertRobotAssetsAgree stays silent on a missing side rather than refusing every pair on
  // disk. See robot_asset.h for why the fingerprint is needed at all.
  m.robot_asset = ParseRobotAsset(root);
  m.n = root.at("N").get<int>();
  m.f = root.at("F").get<int>();
  m.d = root.at("D").get<int>();
  m.num_joints = root.at("nj").get<int>();
  m.cmd_dim = root.at("cmd_dim").get<int>();
  m.cmd_steps = IntList(root.at("cmd_steps"));
  m.num_cmd_tokens = root.at("num_cmd_tokens").get<int>();
  // Exteroception shape, under EITHER key spelling. The exporter renamed these when the
  // per-camera CNN encoder landed: one flat `extero_tokens` x `height_dim` row became
  // `extero_sensor_count` sensors of `extero_sensor_dim` each. Only the NAMES moved -- the
  // graph input is the same flat (V, count*dim) row of raw readings in camera order, because
  // the encoder (MLP or CNN) and the attention mask both live INSIDE the graph. Verified by
  // exporting the same tracker's CNN generator and diffing the two artifacts: identical input
  // shapes, identical extero_spec, identical window_blocks; only these two keys differ.
  //
  // A graph carrying ONLY the old spellings is REFUSED, matching deploy/fm_session.py exactly
  // -- same condition, same remedy. This runtime used to read either, on the grounds that
  // deployed artifacts are a mix and an old fm.onnx must keep loading. That tolerance is
  // withdrawn on purpose: the two loaders disagreeing about which artifacts are valid is
  // worse than either answer, because it makes "does this policy load?" depend on which
  // language asked. Re-exporting is one command, and the checkpoint is untouched.
  //
  // A CNN checkpoint also carries `extero_tokens_per_sensor` (how many tokens the encoder
  // emits into the transformer). That is internal to the graph; this runtime never sees it.
  std::vector<std::string> stale;
  for (const auto& [old_key, new_key] :
       {std::pair<const char*, const char*>{"height_dim", "extero_sensor_dim"},
        std::pair<const char*, const char*>{"extero_tokens", "extero_sensor_count"}}) {
    if (!root.contains(new_key) && root.contains(old_key)) stale.emplace_back(old_key);
  }
  if (!stale.empty()) {
    std::string names;
    for (size_t i = 0; i < stale.size(); ++i) names += (i ? ", " : "") + stale[i];
    throw std::runtime_error(
        "fm.onnx: fm_meta carries the pre-rename exteroception keys [" + names +
        "] and not their replacements (extero_sensor_dim / extero_sensor_count). Re-export "
        "with scripts/flow_matching/export_fm_onnx.py -- the checkpoint itself still loads.");
  }
  m.height_dim = root.at("extero_sensor_dim").get<int>();
  if (root.contains("extero_sensor_count")) {
    m.extero_tokens = root.at("extero_sensor_count").get<int>();
  }
  if (root.contains("extero_source")) {
    m.extero_source = root.at("extero_source").get<std::string>();
  }
  // REQUIRED. The generator's output IS the tracker's reference_window observation -- there
  // is no translation layer between them -- so the two must name the same blocks in the same
  // order. Dropping joint_vel on one side only is caught by the widths; swapping two
  // same-width blocks is caught by nothing else at all.
  if (!root.contains("window_blocks")) {
    throw std::runtime_error(
        "fm.onnx carries no window_blocks in its fm_meta, so its output layout cannot be "
        "checked against the tracker's reference_window. Re-export with export_fm_onnx.py.");
  }
  for (const auto& b : root.at("window_blocks")) {
    // Accept both ["name", width] pairs and bare names, matching the python's tolerance.
    m.window_blocks.push_back(BlockFromName(b.is_array() ? b.at(0).get<std::string>()
                                                         : b.get<std::string>()));
  }
  if (root.contains("extero_channels") && !root.at("extero_channels").is_null()) {
    m.extero_channels = root.at("extero_channels").get<int>();
  }
  if (root.contains("extero_tokens_per_sensor") && !root.at("extero_tokens_per_sensor").is_null()) {
    m.extero_tokens_per_sensor = root.at("extero_tokens_per_sensor").get<int>();
  }
  if (root.contains("extero_image_hw") && !root.at("extero_image_hw").is_null()) {
    for (const auto& hw : root.at("extero_image_hw")) {
      m.extero_image_hw.emplace_back(hw.at(0).get<int>(), hw.at(1).get<int>());
    }
  }
  if (m.extero_channels < 1 || m.height_dim % m.extero_channels != 0) {
    throw std::runtime_error(
        "fm_meta says extero_channels=" + std::to_string(m.extero_channels) +
        " but the sensor row is " + std::to_string(m.height_dim) +
        " wide; the row is channels * H * W, so it must divide.");
  }
  // The clip band. A DEPTH graph that does not state one is REFUSED rather than defaulted,
  // because `absent` no longer names a single number: the training-side per-source default
  // moved to +-5.575 while this runtime's ExteroClip() still says +-2.0, and each is right
  // for the artifacts of its own era. A graph in between would run 3.5 of a band tighter
  // than it trained, saturating range at 4.75 m instead of 11.9 -- silently, since both
  // values are legitimate. Nothing in the graph says which era it came from, so it is stated
  // or it is refused; unlike a checkpoint, fixing a graph is one re-export. The height scan
  // is unaffected: its default never moved, so absent there still names 1.0.
  if (root.contains("extero_clip") && !root.at("extero_clip").is_null()) {
    m.extero_clip = root.at("extero_clip").get<double>();
  } else if (m.extero_source == "depth") {
    throw std::runtime_error(
        "fm.onnx states no extero_clip for a DEPTH generator. That is ambiguous rather than "
        "absent -- the training and deploy per-source defaults are +-5.575 and +-2.0, so the "
        "graph would run a different band than it trained on. Re-export with "
        "scripts/flow_matching/export_fm_onnx.py from a checkpoint whose config states it.");
  } else {
    m.extero_clip = ExteroClip(m.extero_source);
  }
  if (static_cast<int>(m.cmd_steps.size()) != m.num_cmd_tokens) {
    throw std::runtime_error("fm_meta cmd_steps has " + std::to_string(m.cmd_steps.size()) +
                             " entries but num_cmd_tokens is " +
                             std::to_string(m.num_cmd_tokens));
  }
  return m;
}

}  // namespace g1_deploy
