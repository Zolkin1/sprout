// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/robot_asset.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace g1_deploy {
namespace {

/// Names in `a` that are not in `b`, for the mismatch message.
std::string OnlyIn(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  std::vector<std::string> only;
  for (const std::string& s : a) {
    if (std::find(b.begin(), b.end(), s) == b.end()) only.push_back(s);
  }
  std::sort(only.begin(), only.end());
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < only.size(); ++i) out << (i ? ", " : "") << only[i];
  out << "]";
  return out.str();
}

/// First 16 characters, matching the python's `str(t)[:16]` so the two messages line up.
std::string Head(const std::string& s) { return s.substr(0, 16); }

}  // namespace

void AssertRobotAssetsAgree(const RobotAsset& tracker, const RobotAsset& fm,
                            const std::string& tracker_name, const std::string& fm_name) {
  if (!tracker.present || !fm.present) return;

  std::vector<std::string> diffs;
  // Field order matches _ROBOT_ASSET_FIELDS in transfer/deploy/gen_tracker.py, so a mismatch
  // reads the same whichever runtime reported it.
  if (!tracker.joint_names.empty() && !fm.joint_names.empty() &&
      tracker.joint_names != fm.joint_names) {
    std::ostringstream o;
    o << "  joint_names: " << tracker.joint_names.size() << " vs " << fm.joint_names.size();
    diffs.push_back(o.str());
  }
  if (tracker.total_mass_kg && fm.total_mass_kg && *tracker.total_mass_kg != *fm.total_mass_kg) {
    std::ostringstream o;
    o << "  total_mass_kg: " << *tracker.total_mass_kg << " vs " << *fm.total_mass_kg;
    diffs.push_back(o.str());
  }
  if (!tracker.body_names.empty() && !fm.body_names.empty() &&
      tracker.body_names != fm.body_names) {
    diffs.push_back("  body_names: tracker-only " + OnlyIn(tracker.body_names, fm.body_names) +
                    ", fm-only " + OnlyIn(fm.body_names, tracker.body_names));
  }
  const std::pair<const char*, std::pair<const std::string*, const std::string*>> hashes[] = {
      {"urdf_sha256", {&tracker.urdf_sha256, &fm.urdf_sha256}},
      {"mjcf_sha256", {&tracker.mjcf_sha256, &fm.mjcf_sha256}},
      {"usd_sha256", {&tracker.usd_sha256, &fm.usd_sha256}},
  };
  for (const auto& [name, pair] : hashes) {
    if (!pair.first->empty() && !pair.second->empty() && *pair.first != *pair.second) {
      diffs.push_back(std::string("  ") + name + ": " + Head(*pair.first) + " vs " +
                      Head(*pair.second));
    }
  }
  if (diffs.empty()) return;

  std::ostringstream msg;
  msg << "ROBOT ASSET MISMATCH between " << tracker_name << " ("
      << (tracker.short_hash.empty() ? "?" : tracker.short_hash) << ") and " << fm_name << " ("
      << (fm.short_hash.empty() ? "?" : fm.short_hash)
      << "): they were trained on different robots, and there is only one robot to run them "
         "on.";
  for (const std::string& d : diffs) msg << "\n" << d;
  msg << "\nRe-export both from one run, or point the generator at the one trained against "
         "this tracker.";
  throw std::runtime_error(msg.str());
}

}  // namespace g1_deploy
