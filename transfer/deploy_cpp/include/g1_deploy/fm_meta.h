// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// What fm.onnx says about itself: shapes, window layout, and the exteroception contract.
//
// Split out of fm_session.h so it does not require onnxruntime. The parse is a pure
// string -> struct function and every REFUSAL in it (an ambiguous clip band, a channel count
// that cannot divide the row) is a correctness guard -- guards that could only be tested
// where the full runtime happens to be installed, which is not where they are most likely to
// be wrong. It now builds and tests in the math core, alongside the depth math it describes.

#include <string>

#include "g1_deploy/policy_params.h"
#include <utility>
#include <vector>

#include "g1_deploy/robot_asset.h"
#include "g1_deploy/window.h"

namespace g1_deploy {

/// The generator's contract, from fm.onnx's `fm_meta`.
struct FmMeta {
  //: WHICH ROBOT the recording behind this generator was made with. Compared against the
  //: tracker's at load; see AssertRobotAssetsAgree. Absent on any graph exported before the
  //: key existed, which is tolerated rather than refused.
  RobotAsset robot_asset;
  int n = 0;  //: N, history length
  int f = 0;  //: F, horizon
  int d = 0;  //: D, node width
  int num_joints = 0;
  int cmd_dim = 0;
  std::vector<int> cmd_steps;  //: which window steps the command tokens are sampled at
  int num_cmd_tokens = 0;
  int height_dim = 0;
  int extero_tokens = 1;
  std::string extero_source = "height_scan";
  std::vector<ReferenceBlock> window_blocks;
  bool has_history_null = false;  //: inferred from the GRAPH, not the metadata

  /// Planes per pixel the ROW carries: 1 = depth alone, 4 = + xyz in the root's yaw-aligned
  /// frame, 5 = + validity. `height_dim` COUNTS these (it is the row width), so anything
  /// comparing that width against an image shape must divide by this first.
  int extero_channels = 1;
  /// Per-sensor (height, width) the graph folds a row with. Empty for the flat encoder.
  /// `extero_dim()` cannot distinguish 30x26 from 26x30, and under a convolution a
  /// transposed camera is scrambled input rather than a reordered vector.
  std::vector<std::pair<int, int>> extero_image_hw;
  /// Tokens the encoder emits per sensor. Internal to the graph; carried for diagnostics.
  int extero_tokens_per_sensor = 1;
  /// The clip band this generator TRAINED under, from the graph. Always positive after a
  /// successful parse -- see ParseFmMeta for why a depth graph that omits it is refused
  /// rather than defaulted.
  double extero_clip = 0.0;

  /// THE WHOLE RIG: every camera the GENERATOR conditions on, in its own contract order.
  ///
  /// Parsed from `fm_meta.extero_spec`, and this is the only artifact that names the whole rig.
  /// Since `efe93b4e` the POLICY reads the near camera alone while the generator reads both, so
  /// `PolicyParams::depth_cameras` is a SUBSET of this and cannot be used to size the
  /// generator's row -- doing that left the missing camera's plane at 0.0 with nothing raising,
  /// which is a plausible-looking cloud that is simply absent. Empty for a scan generator.
  std::vector<DepthCameraSpec> cameras;

  int extero_dim() const { return extero_tokens * height_dim; }
  /// Pixels per sensor: the row width with the channels divided out.
  int extero_pixels() const { return height_dim / (extero_channels > 0 ? extero_channels : 1); }
};

/// Parse `fm_meta`. `has_history_null` is left false; the loader sets it from the graph.
FmMeta ParseFmMeta(const std::string& json);

}  // namespace g1_deploy
