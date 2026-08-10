// Copyright 2026 Zachary Olkin. All rights reserved.

// The C++ metadata parser against the REAL tracker.onnx's metadata and what python resolves
// it to (transfer/deploy/tools/dump_metadata_fixtures.py).
//
// Skips when the fixture is absent, so the suite never depends on a policy download -- the
// same idiom test_hardware_offline.py uses.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <string>

#include "g1_deploy/policy_params.h"

namespace g1_deploy {
namespace {

std::string FixtureDir() {
  if (const char* env = std::getenv("G1_METADATA_FIXTURES")) return env;
  return G1_DEPLOY_METADATA_DIR;
}

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

class PolicyParamsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!ReadFile(FixtureDir() + "/tracker_policy_parameters.json", &raw_)) {
      GTEST_SKIP() << "no metadata fixture; run transfer/deploy/tools/dump_metadata_fixtures.py";
    }
    std::string expected_raw;
    ASSERT_TRUE(ReadFile(FixtureDir() + "/tracker_expected.json", &expected_raw))
        << "the raw metadata is present but the expected resolution is not; regenerate both";
    expected_ = nlohmann::json::parse(expected_raw);
  }

  std::string raw_;
  nlohmann::json expected_;
};

TEST_F(PolicyParamsFixture, ObservationTermOrderMatchesPython) {
  // THE test. A sorted-key JSON type produces the same total width and the same term set,
  // and only this ordering check would ever notice.
  const PolicyParams p = ParsePolicyParameters(raw_);
  ASSERT_EQ(p.obs_terms.size(), expected_["obs_terms"].size());
  for (size_t i = 0; i < p.obs_terms.size(); ++i) {
    EXPECT_EQ(p.obs_terms[i].name, expected_["obs_terms"][i]["name"].get<std::string>())
        << "term " << i << " is out of order -- the observation layout is permuted";
  }
}

TEST_F(PolicyParamsFixture, TermWidthsScalesClipsAndHistoryMatchPython) {
  const PolicyParams p = ParsePolicyParameters(raw_);
  for (size_t i = 0; i < p.obs_terms.size(); ++i) {
    const ObsTerm& t = p.obs_terms[i];
    const auto& e = expected_["obs_terms"][i];
    EXPECT_EQ(t.width, e["shape"].get<int>()) << t.name;
    EXPECT_EQ(t.history_length, e["history_length"].get<int>()) << t.name;

    if (e["clip"].is_null()) {
      EXPECT_FALSE(t.clip.has_value()) << t.name << " should be unclipped";
    } else {
      ASSERT_TRUE(t.clip.has_value()) << t.name << " should be clipped";
      EXPECT_DOUBLE_EQ(*t.clip, e["clip"].get<double>()) << t.name;
    }

    // scale is scalar for most terms and a per-element vector for reference_window. Both
    // shapes have to survive the round trip; collapsing the vector to its first element
    // would be invisible here (they are all 1.0) but wrong the moment training scales a
    // block differently.
    if (e["scale"].is_array()) {
      ASSERT_EQ(t.scale.size(), e["scale"].size()) << t.name << ": per-element scale lost";
      for (size_t k = 0; k < t.scale.size(); ++k) {
        EXPECT_DOUBLE_EQ(t.scale[k], e["scale"][k].get<double>()) << t.name << "[" << k << "]";
      }
    } else {
      ASSERT_EQ(t.scale.size(), 1u) << t.name << ": scalar scale became a vector";
      EXPECT_DOUBLE_EQ(t.scale[0], e["scale"].get<double>()) << t.name;
    }
  }
}

TEST_F(PolicyParamsFixture, ScalarsAndJointNamesMatchPython) {
  const PolicyParams p = ParsePolicyParameters(raw_);
  EXPECT_EQ(p.num_obs, expected_["num_obs"].get<int>());
  EXPECT_EQ(p.num_actions, expected_["num_actions"].get<int>());
  EXPECT_DOUBLE_EQ(p.dt, expected_["dt"].get<double>());
  EXPECT_EQ(p.TotalObsWidth(), p.num_obs);

  ASSERT_EQ(p.joint_names_isaac.size(), expected_["joint_names_isaac"].size());
  for (size_t i = 0; i < p.joint_names_isaac.size(); ++i) {
    // Joint ORDER, not just membership. The policy's order is interleaved while the robot's
    // is per limb, and 27 of 29 positions differ -- mixing them gives a large, smooth,
    // entirely fictitious error rather than an obvious failure.
    EXPECT_EQ(p.joint_names_isaac[i], expected_["joint_names_isaac"][i].get<std::string>())
        << "joint " << i;
  }
  ASSERT_EQ(p.kp.size(), expected_["kp"].size());
  ASSERT_EQ(p.kd.size(), expected_["kd"].size());
}

TEST_F(PolicyParamsFixture, ReplanCadenceComesFromMetadata) {
  const PolicyParams p = ParsePolicyParameters(raw_);
  if (expected_["fm_update_every"].is_null()) {
    EXPECT_FALSE(p.fm_update_every.has_value());
  } else {
    ASSERT_TRUE(p.fm_update_every.has_value())
        << "the re-plan cadence comes from tracker.onnx metadata and nowhere else";
    EXPECT_EQ(*p.fm_update_every, expected_["fm_update_every"].get<int>());
  }
}

TEST_F(PolicyParamsFixture, ActionScalePatternsResolveExactlyAsPythonDoes) {
  // action_scale is a PATTERN table; the python resolves it with re.fullmatch, taking the
  // FIRST match in the table's own order. std::regex_search instead of regex_match, or a
  // sorted map instead of ordered_json, would give a different per-joint scale -- and a wrong
  // action scale is a robot that moves the right way by the wrong amount.
  const PolicyParams p = ParsePolicyParameters(raw_);
  const auto& want = expected_["action_scale"];
  ASSERT_EQ(p.action_scale.size(), want.size())
      << "every joint must resolve to a scale; an unmatched one has to throw, not default";
  for (size_t i = 0; i < p.action_scale.size(); ++i) {
    EXPECT_DOUBLE_EQ(p.action_scale[i], want[i].get<double>())
        << "joint " << i << " (" << p.joint_names_isaac[i] << ")";
  }
  EXPECT_EQ(static_cast<int>(p.default_joint_angles.size()), p.num_actions);
}

TEST_F(PolicyParamsFixture, ReferenceWindowSpecMatchesPython) {
  const PolicyParams p = ParsePolicyParameters(raw_);
  ASSERT_TRUE(p.reference_window.has_value());
  const auto& e = expected_["reference_window_spec"];
  const auto& rw = *p.reference_window;

  ASSERT_EQ(rw.offsets.size(), e["offsets"].size());
  for (size_t i = 0; i < rw.offsets.size(); ++i) {
    EXPECT_EQ(rw.offsets[i], e["offsets"][i].get<int>()) << "offset " << i;
  }
  ASSERT_EQ(rw.blocks.size(), e["blocks"].size());
  for (size_t i = 0; i < rw.blocks.size(); ++i) {
    // Block order is the node layout. Same-width blocks swapped (joint_pos for joint_vel)
    // changes nothing about any shape and makes the policy read velocities as positions.
    EXPECT_EQ(rw.blocks[i].first, e["blocks"][i][0].get<std::string>()) << "block " << i;
    EXPECT_EQ(rw.blocks[i].second, e["blocks"][i][1].get<int>()) << "block " << i;
  }
  EXPECT_EQ(rw.base_pose_convention, e["base_pose_convention"].get<std::string>());
}

// Behaviour that does not need the artifact.

TEST(PolicyParams, RejectsAMissingBasePoseConvention) {
  // The one field with no default: an artifact predating it was exported under the legacy
  // mixed convention, and defaulting would wave it through as if it agreed.
  const std::string json = R"({
    "observation_terms": {"policy": {"a": {"shape": 2}}},
    "joint_names_isaac": ["j0"], "num_obs": 2, "num_actions": 1, "dt": 0.02,
    "kp": [1.0], "kd": [0.1],
    "reference_window_spec": {"offsets": [0], "blocks": [["base_pose", 9]]}
  })";
  EXPECT_THROW(ParsePolicyParameters(json), std::runtime_error);
}

TEST(PolicyParams, RejectsATermSumThatDisagreesWithNumObs) {
  const std::string json = R"({
    "observation_terms": {"policy": {"a": {"shape": 2}, "b": {"shape": 3}}},
    "joint_names_isaac": ["j0"], "num_obs": 99, "num_actions": 1, "dt": 0.02,
    "kp": [1.0], "kd": [0.1]
  })";
  EXPECT_THROW(ParsePolicyParameters(json), std::runtime_error);
}

TEST(PolicyParams, PreservesInsertionOrderForDeliberatelyAntiAlphabeticalKeys) {
  // A synthetic guard that fails loudly if someone swaps ordered_json for json: these keys
  // are in reverse alphabetical order, so a sorted map inverts them.
  const std::string json = R"({
    "observation_terms": {"policy": {
      "zebra": {"shape": 1}, "yak": {"shape": 1}, "aardvark": {"shape": 1}}},
    "joint_names_isaac": ["j0"], "num_obs": 3, "num_actions": 1, "dt": 0.02,
    "kp": [1.0], "kd": [0.1]
  })";
  const PolicyParams p = ParsePolicyParameters(json);
  ASSERT_EQ(p.obs_terms.size(), 3u);
  EXPECT_EQ(p.obs_terms[0].name, "zebra");
  EXPECT_EQ(p.obs_terms[1].name, "yak");
  EXPECT_EQ(p.obs_terms[2].name, "aardvark");
}

TEST(PolicyParams, FlattensMultiDimensionalTermShapes) {
  // The per-camera image groups export [1, H, W]; the observation manager concatenates the
  // flattened term, so the product is the width.
  const std::string json = R"({
    "observation_terms": {"policy": {"img": {"shape": [1, 4, 5]}}},
    "joint_names_isaac": ["j0"], "num_obs": 20, "num_actions": 1, "dt": 0.02,
    "kp": [1.0], "kd": [0.1]
  })";
  const PolicyParams p = ParsePolicyParameters(json);
  EXPECT_EQ(p.obs_terms[0].width, 20);
}

}  // namespace
}  // namespace g1_deploy
