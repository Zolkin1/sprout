// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Port of transfer/deploy/depth_obs.py: depth normalization and exteroception sanitising.
//
// This is the one part of the port that is held to EXACT float equality rather than a
// tolerance. There are no reductions and no transcendentals here -- just comparisons and one
// affine -- so a single ulp of disagreement would mean a real difference in behaviour, not
// arithmetic noise. The python is already held to exact equality against its torch original
// (transfer/deploy/tests/test_depth_obs.py), and this inherits that.

#include <string>
#include <vector>

#include "g1_deploy/types.h"

namespace g1_deploy {

/// Symmetric clip band per exteroception source.
///
/// The height scan is metres of ground clearance (+-1 m); depth is ALREADY normalised by
/// (d - offset) * scale, and its observation terms clip at +-2. Clamping depth at the scan's
/// 1.0 would silently discard real readings.
double ExteroClip(const std::string& source);

/// One camera's normalization contract, as carried in the ONNX metadata.
struct DepthNorm {
  double offset = 0.75;
  double scale = 0.5;
  double min_dist = 0.2;
  double max_dist = 9.0;
  double min_clip_val = 0.0;
  double max_clip_val = 0.0;
};

/// Clip invalid depth, then apply (d - offset) * scale. Input is metres.
///
/// The ORDER is load-bearing and matches the torch original: the too-near test runs FIRST,
/// so -inf is caught there and becomes min_clip_val rather than max_clip_val. Comparisons
/// are strict, so a reading exactly at min_dist / max_dist stays valid. Both substitutions
/// happen in metres, before the affine.
inline double NormalizeDepth(double d, const DepthNorm& n) {
  double out = d;
  // NaN fails this comparison (as it does in numpy), so it survives to the next line and is
  // caught by the non-finite test -- which is why NaN lands on max_clip_val, not min.
  if (out < n.min_dist) out = n.min_clip_val;
  if (out > n.max_dist || !std::isfinite(out)) out = n.max_clip_val;
  return (out - n.offset) * n.scale;
}

/// Normalize a whole image in place-compatible fashion (out may alias in).
void NormalizeDepthImage(const double* in, size_t count, const DepthNorm& n, double* out);

/// Inverse of NormalizeDepth's affine half, for diagnostics.
inline double DenormalizeDepth(double v, double offset, double scale) {
  return v / scale + offset;
}

/// The normalised value a camera reports for "no reading".
///
/// Every invalid class -- too near, beyond range, non-finite, dropped -- lands on this one
/// value, so a consumer identifies them by comparing against it.
inline double InvalidDepthValue(const DepthNorm& n) {
  return (n.max_clip_val - n.offset) * n.scale;
}

/// Make an exteroception reading finite and clamp it to its source's band.
///
/// The fm.onnx graph folds in the exteroception normalization, so ONE non-finite entry would
/// poison the whole conditioning.
///
/// NOTE: the clamp is written out rather than using Eigen's cwiseMin/cwiseMax. Eigen's min
/// is (a<b)?a:b, so cwiseMin(NaN, clip) returns CLIP -- it would convert a NaN into a
/// legitimate-looking value. Here that is harmless because nan_to_num runs first, but the
/// same shape of bug in the observation assembly would silently delete the non-finite guard
/// chain, so the habit is established here on purpose.
void SanitizeExtero(const double* in, size_t count, const std::string& source, double* out);

}  // namespace g1_deploy
