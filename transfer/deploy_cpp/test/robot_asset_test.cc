// Copyright 2026 Zachary Olkin. All rights reserved.

// The tracker/generator robot check, and the JSON that feeds it.
//
// Mirrors transfer/deploy/tests/test_robot_asset_pair.py. The property under test is narrow and
// easy to get backwards: compare only fields BOTH sides carry, stay silent when either side has
// no fingerprint at all, and refuse otherwise. Getting "silent" wrong in the permissive
// direction makes the guard dead code -- which is exactly what it was on the python side for as
// long as fm.onnx carried no robot_asset.

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <string>

#include "g1_deploy/robot_asset.h"
#include "robot_asset_json.h"

namespace g1_deploy {
namespace {

using Json = nlohmann::ordered_json;

RobotAsset Make(const std::string& short_hash, std::vector<std::string> joints,
                std::vector<std::string> bodies, const std::string& urdf) {
  RobotAsset a;
  a.present = true;
  a.short_hash = short_hash;
  a.joint_names = std::move(joints);
  a.body_names = std::move(bodies);
  a.urdf_sha256 = urdf;
  return a;
}

TEST(RobotAsset, AgreeingFingerprintsPass) {
  const RobotAsset a = Make("abc", {"j1", "j2"}, {"b1"}, "deadbeef");
  EXPECT_NO_THROW(AssertRobotAssetsAgree(a, a, "tracker.onnx", "fm.onnx"));
}

TEST(RobotAsset, AMissingSideIsSilentRatherThanRefused) {
  // Every artifact exported before the field existed has none. Refusing those would break
  // every pair on disk, so this is a guard against a NEW mismatch, not a migration gate.
  const RobotAsset present = Make("abc", {"j1"}, {"b1"}, "deadbeef");
  const RobotAsset absent;
  ASSERT_FALSE(absent.present);
  EXPECT_NO_THROW(AssertRobotAssetsAgree(present, absent, "t", "f"));
  EXPECT_NO_THROW(AssertRobotAssetsAgree(absent, present, "t", "f"));
  EXPECT_NO_THROW(AssertRobotAssetsAgree(absent, absent, "t", "f"));
}

TEST(RobotAsset, ADifferentUrdfHashIsRefusedAndBothShortsAreNamed) {
  const RobotAsset t = Make("aaaa", {"j1"}, {"b1"}, "1111111111111111ffff");
  const RobotAsset f = Make("bbbb", {"j1"}, {"b1"}, "2222222222222222ffff");
  try {
    AssertRobotAssetsAgree(t, f, "tracker.onnx", "gen_fm.onnx");
    FAIL() << "a different urdf hash was accepted";
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("ROBOT ASSET MISMATCH"), std::string::npos) << m;
    EXPECT_NE(m.find("aaaa"), std::string::npos) << m;
    EXPECT_NE(m.find("bbbb"), std::string::npos) << m;
    EXPECT_NE(m.find("urdf_sha256"), std::string::npos) << m;
    // Truncated to 16 characters, matching the python's str(t)[:16] so the two runtimes'
    // messages line up rather than merely both being right.
    EXPECT_NE(m.find("1111111111111111"), std::string::npos) << m;
    EXPECT_EQ(m.find("1111111111111111ffff"), std::string::npos) << m;
  }
}

TEST(RobotAsset, DifferentBodiesAreReportedAsTheSetDifference) {
  // The perceptive G1 changed 39 body names while leaving all 29 joints exactly as they were,
  // so "which bodies" is the useful half of the message.
  const RobotAsset t = Make("aaaa", {"j1"}, {"palm", "shared"}, "same");
  const RobotAsset f = Make("bbbb", {"j1"}, {"rubber_hand", "shared"}, "same");
  try {
    AssertRobotAssetsAgree(t, f, "t", "f");
    FAIL() << "different body sets were accepted";
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("tracker-only [palm]"), std::string::npos) << m;
    EXPECT_NE(m.find("fm-only [rubber_hand]"), std::string::npos) << m;
    EXPECT_EQ(m.find("shared"), std::string::npos) << "the shared body should not be listed";
  }
}

TEST(RobotAsset, AFieldOnlyONESideCarriesIsNotAMismatch) {
  // "Cannot compare" must not read as "disagrees" -- a producer writes what it could see, and
  // a mass read off the PhysX view is allowed to fail.
  RobotAsset t = Make("aaaa", {"j1"}, {"b1"}, "same");
  RobotAsset f = Make("bbbb", {"j1"}, {"b1"}, "same");
  t.total_mass_kg = 42.0;                       // f has none
  f.mjcf_sha256 = "abc";                        // t has none
  EXPECT_NO_THROW(AssertRobotAssetsAgree(t, f, "t", "f"));
  f.total_mass_kg = 43.0;                       // now both carry it, and they differ
  EXPECT_THROW(AssertRobotAssetsAgree(t, f, "t", "f"), std::runtime_error);
}

TEST(RobotAsset, JsonWithNoKeyParsesAsAbsentNotAsEmpty) {
  // The distinction the whole thing turns on: an absent fingerprint cannot be compared, while
  // an all-empty one would compare equal to anything and wave every pair through.
  const Json root = Json::parse(R"({"N": 4})");
  EXPECT_FALSE(ParseRobotAsset(root).present);
  EXPECT_FALSE(ParseRobotAsset(Json::parse(R"({"robot_asset": null})")).present);
  // A non-object is also not a fingerprint, and must not throw either.
  EXPECT_FALSE(ParseRobotAsset(Json::parse(R"({"robot_asset": "abc"})")).present);
}

TEST(RobotAsset, JsonReadsEveryComparableFieldAndSkipsTheRest) {
  const Json root = Json::parse(R"({"robot_asset": {
      "short": "1de4e4908cfe", "version": 1,
      "covers": ["joint_names"],
      "joint_names": ["a", "b"], "body_names": ["x"],
      "total_mass_kg": 41.5,
      "urdf_sha256": "u", "mjcf_sha256": "m", "usd_sha256": "s",
      "something_new": {"nested": 1}}})");
  const RobotAsset a = ParseRobotAsset(root);
  ASSERT_TRUE(a.present);
  EXPECT_EQ(a.short_hash, "1de4e4908cfe");
  EXPECT_EQ(a.joint_names, (std::vector<std::string>{"a", "b"}));
  EXPECT_EQ(a.body_names, (std::vector<std::string>{"x"}));
  ASSERT_TRUE(a.total_mass_kg.has_value());
  EXPECT_DOUBLE_EQ(*a.total_mass_kg, 41.5);
  EXPECT_EQ(a.urdf_sha256, "u");
  EXPECT_EQ(a.mjcf_sha256, "m");
  EXPECT_EQ(a.usd_sha256, "s");
}

TEST(RobotAsset, ANullOrWronglyTypedFieldIsSkippedRatherThanFatal) {
  // A guard must never be the reason a load dies; the python producer says the same.
  const Json root = Json::parse(R"({"robot_asset": {
      "short": null, "joint_names": ["a", 7, "b"], "total_mass_kg": "heavy"}})");
  const RobotAsset a = ParseRobotAsset(root);
  ASSERT_TRUE(a.present);
  EXPECT_TRUE(a.short_hash.empty());
  EXPECT_EQ(a.joint_names, (std::vector<std::string>{"a", "b"}));
  EXPECT_FALSE(a.total_mass_kg.has_value());
}

}  // namespace
}  // namespace g1_deploy
