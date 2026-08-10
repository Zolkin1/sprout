// Copyright 2026 Zachary Olkin. All rights reserved.

// The observation assembly against python's, over the REAL tracker's term table.
//
// Held to bit equality: the fixtures store the python's float32 output widened to float64,
// which is lossless, so this compares with == and not a tolerance. There is nothing here that
// could legitimately differ -- the arithmetic is a cast, a clip, a memcpy and a multiply.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "g1_deploy/obs_assembler.h"
#include "g1_deploy/policy_params.h"

namespace g1_deploy {
namespace {

std::string ObsDir() {
  if (const char* env = std::getenv("G1_OBS_FIXTURES")) return env;
  return G1_DEPLOY_OBS_DIR;
}

std::string MetadataDir() {
  if (const char* env = std::getenv("G1_METADATA_FIXTURES")) return env;
  return G1_DEPLOY_METADATA_DIR;
}

bool ReadText(const std::string& path, std::string* out) {
  std::ifstream in(path);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return true;
}

std::vector<double> ReadBlob(const std::string& path, size_t count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<double> v(count);
  in.read(reinterpret_cast<char*>(v.data()),
          static_cast<std::streamsize>(count * sizeof(double)));
  if (!in) throw std::runtime_error("short read on " + path);
  return v;
}

class ObsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string raw;
    if (!ReadText(ObsDir() + "/manifest.json", &raw)) {
      GTEST_SKIP() << "no obs fixture; run transfer/deploy/tools/dump_obs_fixtures.py";
    }
    manifest_ = nlohmann::json::parse(raw);

    std::string params_raw;
    ASSERT_TRUE(ReadText(MetadataDir() + "/tracker_policy_parameters.json", &params_raw))
        << "the obs fixture is present but the metadata one is not; regenerate both";
    params_ = ParsePolicyParameters(params_raw);
  }

  nlohmann::json manifest_;
  PolicyParams params_;
};

TEST_F(ObsFixture, AssemblyIsBitIdenticalToPythonAcrossHistoryRolling) {
  ObsAssembler assembler(params_.obs_terms, params_.num_obs);
  const int num_obs = manifest_["num_obs"].get<int>();
  ASSERT_EQ(num_obs, params_.num_obs);

  std::vector<float> out(static_cast<size_t>(num_obs), 0.0f);
  int steps_checked = 0;
  for (const auto& step : manifest_["steps"]) {
    std::map<std::string, std::vector<double>> values;
    for (const auto& t : step["terms"]) {
      values[t["name"].get<std::string>()] =
          ReadBlob(ObsDir() + "/" + t["file"].get<std::string>(), t["size"].get<size_t>());
    }
    assembler.Assemble(values, out.data());

    const std::vector<double> want =
        ReadBlob(ObsDir() + "/" + step["obs"]["file"].get<std::string>(),
                 step["obs"]["size"].get<size_t>());
    ASSERT_EQ(want.size(), out.size());

    int mismatches = 0;
    int first_bad = -1;
    for (size_t i = 0; i < out.size(); ++i) {
      const double got = static_cast<double>(out[i]);
      // NaN is expected in the poisoned step and must be equal-by-NaN-ness, not by ==.
      const bool both_nan = std::isnan(got) && std::isnan(want[i]);
      if (!both_nan && got != want[i]) {
        ++mismatches;
        if (first_bad < 0) first_bad = static_cast<int>(i);
      }
    }
    EXPECT_EQ(mismatches, 0) << "step " << step["step"].get<int>() << ": " << mismatches
                             << " of " << out.size() << " differ; first at index " << first_bad;
    ++steps_checked;
  }
  // More steps than any term's history length, so the rolling path is genuinely exercised and
  // not just the first-push fill.
  EXPECT_GE(steps_checked, 12) << "too few steps to exercise the history rolling";
  std::cout << "    [obs] " << steps_checked << " steps x " << num_obs
            << " values, bit-identical to python\n";
}

TEST_F(ObsFixture, NanSurvivesTheClipAndReachesTheObservation) {
  // The single most dangerous line in the port. np.clip propagates NaN and the non-finite
  // guard depends on it: a NaN has to reach _finite_targets, which refuses to publish and
  // eventually disarms. Eigen's cwiseMin(NaN, c) returns c, which would delete the guard.
  ObsAssembler assembler(params_.obs_terms, params_.num_obs);
  std::vector<float> out(static_cast<size_t>(params_.num_obs), 0.0f);

  bool found_poisoned = false;
  for (const auto& step : manifest_["steps"]) {
    std::map<std::string, std::vector<double>> values;
    for (const auto& t : step["terms"]) {
      values[t["name"].get<std::string>()] =
          ReadBlob(ObsDir() + "/" + t["file"].get<std::string>(), t["size"].get<size_t>());
    }
    assembler.Assemble(values, out.data());
    if (!step["poison"].get<bool>()) continue;

    found_poisoned = true;
    int nans = 0;
    for (float v : out) {
      if (std::isnan(v)) ++nans;
    }
    EXPECT_GT(nans, 0) << "the NaN was swallowed by the clip -- the non-finite guard is dead";
    for (float v : out) EXPECT_FALSE(std::isinf(v)) << "+/-inf must clip to the bound";
  }
  ASSERT_TRUE(found_poisoned) << "the fixture has no non-finite step; regenerate it";
}

TEST_F(ObsFixture, ChecksumMatchesPythonHashlib) {
  // blake2b with digest_size=8, which is NOT a truncation of blake2b-512, so this pins the
  // vendored implementation against hashlib on real observation bytes.
  ObsAssembler assembler(params_.obs_terms, params_.num_obs);
  std::vector<float> out(static_cast<size_t>(params_.num_obs), 0.0f);
  for (const auto& step : manifest_["steps"]) {
    std::map<std::string, std::vector<double>> values;
    for (const auto& t : step["terms"]) {
      values[t["name"].get<std::string>()] =
          ReadBlob(ObsDir() + "/" + t["file"].get<std::string>(), t["size"].get<size_t>());
    }
    assembler.Assemble(values, out.data());
    const uint64_t want = std::stoull(step["checksum"].get<std::string>());
    EXPECT_EQ(ObsChecksum(out.data(), out.size()), want)
        << "checksum differs at step " << step["step"].get<int>();
  }
}

// Behaviour that does not need the artifact.

TEST(ObsAssembler, ClipPreservesNanButBoundsInfinities) {
  EXPECT_TRUE(std::isnan(ClipPreservingNan(std::nanf(""), 2.0f)));
  EXPECT_EQ(ClipPreservingNan(std::numeric_limits<float>::infinity(), 2.0f), 2.0f);
  EXPECT_EQ(ClipPreservingNan(-std::numeric_limits<float>::infinity(), 2.0f), -2.0f);
  EXPECT_EQ(ClipPreservingNan(0.5f, 2.0f), 0.5f);
  EXPECT_EQ(ClipPreservingNan(-3.0f, 2.0f), -2.0f);
}

TEST(ObsAssembler, FirstPushFillsTheWindowRatherThanZeroPadding) {
  // Isaac Lab's CircularBuffer semantics, which the policy trained against. Zero-padding would
  // hand it a discontinuity at every reset that training never produced.
  ObsTerm t;
  t.name = "x";
  t.width = 6;            // 3 history steps of 2
  t.history_length = 3;
  t.scale = {1.0};
  ObsAssembler a({t}, 6);
  std::vector<float> out(6, -1.0f);
  a.Assemble({{"x", {1.0, 2.0}}}, out.data());
  const std::vector<float> want = {1, 2, 1, 2, 1, 2};
  EXPECT_EQ(out, want);
}

TEST(ObsAssembler, WindowRollsOldestFirstNewestLast) {
  ObsTerm t;
  t.name = "x";
  t.width = 6;
  t.history_length = 3;
  t.scale = {1.0};
  ObsAssembler a({t}, 6);
  std::vector<float> out(6, 0.0f);
  a.Assemble({{"x", {1.0, 1.0}}}, out.data());
  a.Assemble({{"x", {2.0, 2.0}}}, out.data());
  a.Assemble({{"x", {3.0, 3.0}}}, out.data());
  const std::vector<float> want = {1, 1, 2, 2, 3, 3};
  EXPECT_EQ(out, want) << "the flattened window must read [v_{t-2}, v_{t-1}, v_t]";
  a.Assemble({{"x", {4.0, 4.0}}}, out.data());
  const std::vector<float> rolled = {2, 2, 3, 3, 4, 4};
  EXPECT_EQ(out, rolled);
}

TEST(ObsAssembler, APerElementScaleTilesAcrossTheHistory) {
  // The scale describes ONE sample, so it repeats over the window. Applying it un-tiled would
  // scale the wrong channels in every slot but the first.
  ObsTerm t;
  t.name = "x";
  t.width = 6;
  t.history_length = 3;
  t.scale = {10.0, 100.0};
  ObsAssembler a({t}, 6);
  std::vector<float> out(6, 0.0f);
  a.Assemble({{"x", {1.0, 1.0}}}, out.data());
  const std::vector<float> want = {10, 100, 10, 100, 10, 100};
  EXPECT_EQ(out, want);
}

TEST(ObsAssembler, ScaleIsAppliedAfterStackingNotBefore) {
  // The history buffer holds the clipped but UNSCALED sample. Scaling before the push would
  // compound the scale on every slot that survives a roll.
  ObsTerm t;
  t.name = "x";
  t.width = 4;
  t.history_length = 2;
  t.scale = {2.0};
  ObsAssembler a({t}, 4);
  std::vector<float> out(4, 0.0f);
  a.Assemble({{"x", {1.0, 1.0}}}, out.data());
  a.Assemble({{"x", {1.0, 1.0}}}, out.data());
  const std::vector<float> want = {2, 2, 2, 2};
  EXPECT_EQ(out, want) << "a surviving slot was scaled twice";
}

TEST(ObsAssembler, ResetClearsTheWindows) {
  ObsTerm t;
  t.name = "x";
  t.width = 4;
  t.history_length = 2;
  t.scale = {1.0};
  ObsAssembler a({t}, 4);
  std::vector<float> out(4, 0.0f);
  a.Assemble({{"x", {1.0, 1.0}}}, out.data());
  a.Assemble({{"x", {2.0, 2.0}}}, out.data());
  a.Reset();
  a.Assemble({{"x", {5.0, 5.0}}}, out.data());
  const std::vector<float> want = {5, 5, 5, 5};
  EXPECT_EQ(out, want) << "a window survived the reset and fed pre-reset motion to the policy";
}

TEST(ObsAssembler, RefusesAMissingTermAndAWrongWidth) {
  ObsTerm t;
  t.name = "x";
  t.width = 2;
  t.scale = {1.0};
  ObsAssembler a({t}, 2);
  std::vector<float> out(2, 0.0f);
  EXPECT_THROW(a.Assemble({{"y", {1.0, 2.0}}}, out.data()), std::runtime_error);
  EXPECT_THROW(a.Assemble({{"x", {1.0}}}, out.data()), std::runtime_error);
}

TEST(ObsAssembler, BodyTermsSayWhyTheyAreUnsupported) {
  // tracking_mode='body' needs root-frame body targets, which only a clip reference supplies.
  ObsTerm t;
  t.name = "body_pos_ref";
  t.width = 3;
  t.scale = {1.0};
  ObsAssembler a({t}, 3);
  std::vector<float> out(3, 0.0f);
  try {
    a.Assemble({}, out.data());
    FAIL() << "expected a throw";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("tracking_mode"), std::string::npos);
  }
}

}  // namespace
}  // namespace g1_deploy
