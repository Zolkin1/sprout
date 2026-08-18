// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/depth_obs.h"

#include <cmath>
#include <algorithm>
#include <memory>
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
  SanitizeExteroBand(in, count, ExteroClip(source), out);
}

void SanitizeExteroBand(const double* in, size_t count, double clip, double* out) {
  for (size_t i = 0; i < count; ++i) {
    double v = in[i];
    // np.nan_to_num(nan=0.0, posinf=clip, NEGINF=+clip), then a clip to the same band.
    //
    // BOTH infinities go to +clip, which is not a typo and is the one line here that is not
    // symmetric. A missed downward ray means "no ground within range" -- ground very far
    // BELOW -- so the drop-off end is where it belongs. Sending -inf to -clip instead
    // encodes a HOLE AS A WALL, at a value real terrain also produces, so nothing downstream
    // can tell the two apart. This port had it as `v > 0 ? clip : -clip` and the committed
    // parity fixtures were stale enough not to contain a -inf; regenerating them caught it.
    if (std::isnan(v)) {
      v = 0.0;
    } else if (std::isinf(v)) {
      v = clip;
    }
    if (v < -clip) v = -clip;
    if (v > clip) v = clip;
    out[i] = v;
  }
}

std::vector<double> PinholeRayDirections(const CameraGeom& cam) {
  std::vector<double> dirs(cam.pixels() * 3);
  size_t k = 0;
  // v is the OUTER loop and u the inner: row-major, v = 0 the top row. numpy's meshgrid with
  // indexing="xy" produces exactly this order, and it is the order the encoder folds.
  for (int v = 0; v < cam.height; ++v) {
    for (int u = 0; u < cam.width; ++u, ++k) {
      const double x = 1.0;
      const double y = -(static_cast<double>(u) + 0.5 - cam.cx) / cam.fx;
      const double z = -(static_cast<double>(v) + 0.5 - cam.cy) / cam.fy;
      const double inv = 1.0 / std::sqrt(x * x + y * y + z * z);
      dirs[k * 3 + 0] = x * inv;
      dirs[k * 3 + 1] = y * inv;
      dirs[k * 3 + 2] = z * inv;
    }
  }
  return dirs;
}

void InvalidDepthMask(const double* image, size_t count, const DepthNorm& n, bool* out) {
  for (size_t i = 0; i < count; ++i) {
    const double depth_m = DenormalizeDepth(image[i], n.offset, n.scale);
    // NaN fails `<`, so the non-finite test is what catches it -- the same reason the python
    // writes `~isfinite | (depth_m < min_dist)` rather than relying on the comparison alone.
    out[i] = !std::isfinite(depth_m) || depth_m < n.min_dist;
  }
}

void UnprojectToYawFrame(const CameraGeom& cam, const double* image, const double* cam_pos,
                         const double* cam_rot, const double* dirs, double* out) {
  const size_t k = cam.pixels();
  // A plain heap array, not std::vector<bool>: that specialisation packs bits and has no
  // data(), so it cannot be written through a bool*.
  std::unique_ptr<bool[]> invalid(new bool[k]);
  InvalidDepthMask(image, k, cam.norm, invalid.get());
  for (size_t i = 0; i < k; ++i) {
    // z-depth -> ray LENGTH for a real reading (dirs[0] is the cosine to the optical axis);
    // an invalid pixel takes max_dist as a ray length DIRECTLY, so misses form a spherical
    // shell at the sensor's reach instead of a plane whose corners sit far beyond it.
    const double depth_m = DenormalizeDepth(image[i], cam.norm.offset, cam.norm.scale);
    const double ray_len = invalid[i] ? cam.norm.max_dist : depth_m / dirs[i * 3 + 0];
    const double px = dirs[i * 3 + 0] * ray_len;
    const double py = dirs[i * 3 + 1] * ray_len;
    const double pz = dirs[i * 3 + 2] * ray_len;
    // Point-major (all x, then all y, then all z), matching the python's (3, K).
    out[0 * k + i] = cam_rot[0] * px + cam_rot[1] * py + cam_rot[2] * pz + cam_pos[0];
    out[1 * k + i] = cam_rot[3] * px + cam_rot[4] * py + cam_rot[5] * pz + cam_pos[1];
    out[2 * k + i] = cam_rot[6] * px + cam_rot[7] * py + cam_rot[8] * pz + cam_pos[2];
  }
}

void DepthXyzRows(const std::vector<CameraGeom>& cams,
                  const std::vector<const double*>& images,
                  const std::vector<const double*>& cam_pos,
                  const std::vector<const double*>& cam_rot, int channels, double clip,
                  double* out) {
  if (channels != 4 && channels != 5) {
    throw std::invalid_argument(
        "DepthXyzRows emits 4 ([d, x, y, z]) or 5 (plus validity) channels; 1 needs no widening");
  }
  if (images.size() != cams.size() || cam_pos.size() != cams.size() ||
      cam_rot.size() != cams.size()) {
    throw std::invalid_argument("one image and one pose per camera, in contract order");
  }
  size_t at = 0;
  for (size_t c = 0; c < cams.size(); ++c) {
    const size_t k = cams[c].pixels();
    const std::vector<double> dirs = PinholeRayDirections(cams[c]);
    // Clamp the IMAGE first (see the header): everything below reads `clamped`, so channel 0,
    // the unprojection and the validity plane all describe one and the same reading.
    std::vector<double> clamped(k);
    SanitizeExteroBand(images[c], k, clip, clamped.data());
    // Channel 0 is that image, untouched from here on: the cloud's wall then sits at exactly
    // the range the depth channel claims.
    std::copy(clamped.begin(), clamped.end(), out + at);
    UnprojectToYawFrame(cams[c], clamped.data(), cam_pos[c], cam_rot[c], dirs.data(),
                        out + at + k);
    if (channels == 5) {
      std::unique_ptr<bool[]> invalid(new bool[k]);
      InvalidDepthMask(clamped.data(), k, cams[c].norm, invalid.get());
      double* v = out + at + 4 * k;
      for (size_t i = 0; i < k; ++i) v[i] = invalid[i] ? 0.0 : 1.0;
    }
    at += static_cast<size_t>(channels) * k;
  }
}

}  // namespace g1_deploy
