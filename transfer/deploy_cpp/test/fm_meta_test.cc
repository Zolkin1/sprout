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
}  // namespace g1_deploy
