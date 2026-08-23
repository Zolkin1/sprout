// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// JSON -> RobotAsset, in ONE place.
//
// Deliberately in src/ and not include/: nlohmann_json is PRIVATE to g1_deploy, so putting a
// json type in a public header would leak the dependency into the ROS node that consumes this
// library. Both parsers that need it (fm_meta.cc for fm.onnx, policy_params.cc for
// tracker.onnx) live in src/, so an internal header is enough.
//
// A TEMPLATE because the two callers use different json types -- policy_params keeps
// `ordered_json` (the observation term order IS the observation layout, so key order matters)
// while nothing stops fm_meta doing the same. Templating avoids picking one and avoids the
// second copy that would otherwise appear.

#include <string>

#include "g1_deploy/robot_asset.h"

namespace g1_deploy {

/// Read a `robot_asset` object, or return an ABSENT fingerprint.
///
/// Every field is optional and silently skipped when missing or null. That is not laxity: the
/// producer writes what it could see (a mass read off the PhysX view can fail, a file hash
/// needs the file), and AssertRobotAssetsAgree compares only fields BOTH sides carry. What is
/// NOT tolerated downstream is two artifacts sharing no comparable field at all.
///
/// Args:
///   parent: The enclosing object (`fm_meta` or `policy_parameters`).
///   key: The member to read, normally "robot_asset".
template <typename Json>
RobotAsset ParseRobotAsset(const Json& parent, const std::string& key = "robot_asset") {
  RobotAsset out;
  if (!parent.contains(key) || parent[key].is_null() || !parent[key].is_object()) return out;
  const auto& node = parent[key];
  out.present = true;
  auto str = [&node](const char* k, std::string* dst) {
    if (node.contains(k) && node[k].is_string()) *dst = node[k].template get<std::string>();
  };
  auto list = [&node](const char* k, std::vector<std::string>* dst) {
    if (!node.contains(k) || !node[k].is_array()) return;
    for (const auto& v : node[k]) {
      if (v.is_string()) dst->push_back(v.template get<std::string>());
    }
  };
  str("short", &out.short_hash);
  str("urdf_sha256", &out.urdf_sha256);
  str("mjcf_sha256", &out.mjcf_sha256);
  str("usd_sha256", &out.usd_sha256);
  list("joint_names", &out.joint_names);
  list("body_names", &out.body_names);
  if (node.contains("total_mass_kg") && node["total_mass_kg"].is_number()) {
    out.total_mass_kg = node["total_mass_kg"].template get<double>();
  }
  return out;
}

}  // namespace g1_deploy
