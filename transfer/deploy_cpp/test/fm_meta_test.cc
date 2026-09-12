// Copyright 2026 Zachary Olkin. All rights reserved.

// ParseFmMeta: the contract fm.onnx carries about its own exteroception.
//
// Reached only through the ORT session layer until now, which needs onnxruntime and so is
// skipped on any box without it -- leaving the parse itself, and every refusal in it,
// untested wherever the runtime is not fully installed. It is a pure string -> struct
// function, so it needs neither.

#include <gtest/gtest.h>

#include <string>

#include "g1_deploy/fm_meta.h"

namespace g1_deploy {
namespace {

/// A minimal but VALID fm_meta, with `extra` spliced in.
///
/// Everything ParseFmMeta requires and nothing it does not, so a test that adds one key is
/// testing that key rather than the surrounding shape.
std::string Meta(const std::string& extra) {
  return std::string(R"({"N":4,"F":62,"D":44,"nj":29,"cmd_dim":3,)") +
         R"("cmd_steps":[0,5],"num_cmd_tokens":2,)" +
         R"("window_blocks":[["base_pose",9],["base_vel",6],["joint_pos",29]],)" + extra + "}";
}

TEST(FmMeta, DepthWithoutAClipBandIsRefused) {
  // The band is ambiguous, not absent: the training-side per-source default moved to
  // +-5.575 while this runtime's table still says +-2.0, and each is right for artifacts of
  // its own era. A graph in between would clamp 3.5 of a band tighter than it trained --
  // saturating range at 4.75 m rather than 11.9 -- with both values legitimate, so nothing
  // downstream could notice. Refused, because nothing in the graph says which era it is.
  EXPECT_THROW(ParseFmMeta(Meta(R"("extero_source":"depth","extero_sensor_count":2,)"
                                R"("extero_sensor_dim":3900,"extero_channels":5)")),
               std::runtime_error);
}

TEST(FmMeta, DepthWithAClipBandUsesIt) {
  const FmMeta m = ParseFmMeta(Meta(
      R"("extero_source":"depth","extero_sensor_count":2,"extero_sensor_dim":3900,)"
      R"("extero_channels":5,"extero_clip":5.575,"extero_tokens_per_sensor":56,)"
      R"("extero_image_hw":[[26,30],[26,30]])"));
  EXPECT_EQ(m.extero_source, "depth");
  EXPECT_EQ(m.extero_tokens, 2);
  EXPECT_EQ(m.height_dim, 3900);
  EXPECT_EQ(m.extero_channels, 5);
  EXPECT_DOUBLE_EQ(m.extero_clip, 5.575);
  EXPECT_EQ(m.extero_tokens_per_sensor, 56);
  ASSERT_EQ(m.extero_image_hw.size(), 2u);
  EXPECT_EQ(m.extero_image_hw[0].first, 26);
  EXPECT_EQ(m.extero_image_hw[0].second, 30);
  // The row is channels * H * W, so the pixel count is what an image shape is checked
  // against -- 3900 / 5 = 780 = 26 * 30. Comparing the ROW width would reject the very
  // geometry the field exists to describe.
  EXPECT_EQ(m.extero_pixels(), 780);
  EXPECT_EQ(m.extero_pixels(), m.extero_image_hw[0].first * m.extero_image_hw[0].second);
}

TEST(FmMeta, AHeightScanWithoutAClipBandKeepsItsDefault) {
  // Unaffected by the refusal above: the scan's per-source default never moved, so an absent
  // band there still names exactly one number.
  const FmMeta m = ParseFmMeta(Meta(
      R"("extero_source":"height_scan","extero_sensor_count":1,"extero_sensor_dim":176)"));
  EXPECT_EQ(m.extero_channels, 1);
  EXPECT_DOUBLE_EQ(m.extero_clip, 1.0);
  EXPECT_EQ(m.extero_pixels(), 176);
}

TEST(FmMeta, ChannelsMustDivideTheRowWidth) {
  // channels * H * W IS the row width, so a count that does not divide it describes a
  // different model -- and would otherwise surface as a fold error deep inside the graph.
  EXPECT_THROW(ParseFmMeta(Meta(R"("extero_source":"depth","extero_sensor_count":2,)"
                                R"("extero_sensor_dim":3900,"extero_channels":7,)"
                                R"("extero_clip":5.575)")),
               std::runtime_error);
}

TEST(FmMeta, APreRenameGraphIsRefusedJustAsThePythonRefusesIt) {
  // The two loaders must agree about which artifacts are valid. When they did not, "does
  // this policy load?" depended on which language asked -- the python refusing a graph the
  // C++ would happily run, on a robot where only one of them is in the loop. Re-exporting is
  // one command and leaves the checkpoint alone, so refusing costs little and removes the
  // divergence entirely.
  EXPECT_THROW(ParseFmMeta(Meta(
                   R"("extero_source":"height_scan","extero_tokens":1,"height_dim":176)")),
               std::runtime_error);
  // Only the OLD key alone is refused: a graph carrying both keeps the new one, which is
  // what an artifact exported across the rename looks like.
  const FmMeta m = ParseFmMeta(Meta(
      R"("extero_source":"height_scan","extero_tokens":1,"height_dim":999,)"
      R"("extero_sensor_count":1,"extero_sensor_dim":176)"));
  EXPECT_EQ(m.height_dim, 176);
}

}  // namespace

// --- THE RIG -----------------------------------------------------------------------------
//
// Since `efe93b4e` the POLICY reads the near camera alone while the GENERATOR reads both, so
// `fm_meta.extero_spec` is the only artifact naming the whole rig and the tracker's
// `depth_camera_spec` is a SUBSET of it. Sizing the generator's row off the tracker's list
// left the missing camera's plane at 0.0 with nothing raising -- an absent cloud that looks
// exactly like a plausible one.

/// One camera's contract block, with the fields the shared parser requires.
std::string Cam(const std::string& obs_term, int width, int height) {
  return R"({"obs_term":")" + obs_term + R"(","sensor":"cam_)" + obs_term +
         R"(","mount_body":"torso_link","width":)" + std::to_string(width) +
         R"(,"height":)" + std::to_string(height) +
         R"(,"offset":0.0,"scale":1.0,"min_dist":0.1,"max_dist":10.0,)" +
         R"("min_clip_val":-1.0,"max_clip_val":1.0,"fx":12.0,"fy":18.0,"cx":15.0,"cy":13.0})";
}

TEST(FmMeta, TheGeneratorCarriesTheWholeRig) {
  const FmMeta m = ParseFmMeta(Meta(
      R"("extero_source":"depth","extero_sensor_count":2,"extero_sensor_dim":3900,)"
      R"("extero_channels":5,"extero_clip":5.575,"extero_spec":[)" +
      Cam("depth_near", 30, 26) + "," + Cam("depth_far", 30, 26) + "]"));
  ASSERT_EQ(m.cameras.size(), 2u);
  EXPECT_EQ(m.cameras[0].obs_term, "depth_near");
  EXPECT_EQ(m.cameras[1].obs_term, "depth_far");
  // Parsed with the SAME parser the tracker's block uses, so the geometry survives too -- a
  // camera whose intrinsics were dropped here unprojects to the wrong place, silently.
  EXPECT_DOUBLE_EQ(m.cameras[0].fx, 12.0);
  EXPECT_EQ(m.cameras[0].width, 30);
}

TEST(FmMeta, ARigThatDisagreesWithTheDeclaredCountIsRefused) {
  // The row is SIZED from extero_sensor_count and FILLED from the list. Disagreeing means one
  // camera's worth of the row is never written and stays zero.
  EXPECT_THROW(ParseFmMeta(Meta(
                   R"("extero_source":"depth","extero_sensor_count":2,)"
                   R"("extero_sensor_dim":3900,"extero_channels":5,"extero_clip":5.575,)"
                   R"("extero_spec":[)" + Cam("depth_near", 30, 26) + "]")),
               std::runtime_error);
}

TEST(FmMeta, AScanGeneratorNamesNoCamerasAndThatIsNotAnError) {
  // extero_spec holds a camera list for one generator and a ray grid for another. A scan
  // generator has no cameras and needs none; the depth path refuses an empty rig at the point
  // of USE, where the reason is specific, rather than here where it would also catch this.
  const FmMeta m = ParseFmMeta(Meta(
      R"("extero_source":"height_scan","extero_sensor_count":1,"extero_sensor_dim":176)"));
  EXPECT_TRUE(m.cameras.empty());
}

}  // namespace g1_deploy
