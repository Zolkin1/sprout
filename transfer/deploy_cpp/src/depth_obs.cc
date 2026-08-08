// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/depth_obs.h"

#include <cmath>
#include <stdexcept>

namespace g1_deploy {

double ExteroClip(const std::string& source) {
  if (source == "height_scan") return 1.0;
  if (source == "depth") return 2.0;
  throw std::runtime_error("unknown exteroception source '" + source +
                           "'; expected 'height_scan' or 'depth'");
}

void NormalizeDepthImage(const double* in, size_t count, const DepthNorm& n, double* out) {
  for (size_t i = 0; i < count; ++i) out[i] = NormalizeDepth(in[i], n);
}

void SanitizeExtero(const double* in, size_t count, const std::string& source, double* out) {
  const double clip = ExteroClip(source);
  for (size_t i = 0; i < count; ++i) {
    double v = in[i];
    // np.nan_to_num(nan=0.0, posinf=clip, neginf=-clip), then a clip to the same band.
    if (std::isnan(v)) {
      v = 0.0;
    } else if (std::isinf(v)) {
      v = v > 0.0 ? clip : -clip;
    }
    if (v < -clip) v = -clip;
    if (v > clip) v = clip;
    out[i] = v;
  }
}

}  // namespace g1_deploy
