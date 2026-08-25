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

/// As above, with the band stated rather than looked up by source.
///
/// This is the overload every caller holding a loaded graph should use: the band belongs to
/// the CHECKPOINT, and ExteroClip()'s per-source table is only correct for artifacts of the
/// era it was written in (depth moved 2.0 -> 5.575 on the training side). FmMeta::extero_clip
/// is that number, refused at parse rather than defaulted, so passing it here is what keeps
/// the runtime clamping the way the model was trained.
void SanitizeExteroBand(const double* in, size_t count, double clip, double* out);

/// One camera's pinhole geometry, alongside its normalization contract.
struct CameraGeom {
  int width = 0;
  int height = 0;
  double fx = 1.0;
  double fy = 1.0;
  double cx = 0.0;
  double cy = 0.0;
  DepthNorm norm{};

  size_t pixels() const { return static_cast<size_t>(width) * static_cast<size_t>(height); }
};

/// Fill a CameraGeom from a parsed contract entry.
///
/// Declared here and defined in policy_params.cc so depth_obs stays free of the metadata
/// layer. The point is that no consumer ever restates a camera number: the contract is the
/// single producer, and this is the one place it becomes geometry.
struct DepthCameraSpec;
CameraGeom GeomFromSpec(const DepthCameraSpec& spec);

/// Unit ray directions in the camera frame, matching IsaacLab's pinhole pattern.
///
/// Pixel CENTRES through the inverse intrinsic matrix, then into the robotics camera frame
/// (x = optical axis, y = left, z = up). Row-major flat order `k = v * width + u`, with
/// `v = 0` the TOP row and `u = 0` the LEFT column. That ordering is a CONTRACT, not a
/// convention: the encoder folds these rows into an image, so transposing it scrambles every
/// convolution kernel while every shape stays correct.
///
/// Depends only on the contract, so it is constant for the life of a run -- compute it once
/// and pass it to UnprojectToYawFrame rather than per frame.
///
/// \return `width * height` rows of 3, flattened.
std::vector<double> PinholeRayDirections(const CameraGeom& cam);

/// Which pixels of a NORMALISED image carry no usable range reading.
///
/// The test is `denormalised depth < min_dist` (or non-finite), NOT equality with the
/// sentinel NormalizeDepth writes, and the difference is load-bearing. The sentinels are
/// exact only in a CLEAN image; in training they are not, because the depth noiser runs
/// AFTER normalisation and its 3x3 blur mixes a missed pixel with its neighbours, leaving it
/// NEAR the sentinel rather than on it. An equality test would call those pixels valid and
/// unproject them to ~0.1 m -- a fake surface on the robot itself.
void InvalidDepthMask(const double* image, size_t count, const DepthNorm& n, bool* out);

/// A normalised depth image -> its 3-D points in the root's yaw-aligned frame.
///
/// Port of transfer/deploy/depth_obs.py's unproject_to_yaw_frame. The frame is the
/// generator's own anchor frame: origin at the root link, axes yaw-aligned, so its z IS
/// gravity and STATIC TERRAIN KEEPS ITS COORDINATES WHEN THE ROBOT TILTS. A pelvis pitch
/// rewrites every pixel but moves the camera by exactly as much, and composing the two puts
/// the terrain back where it was.
///
/// Nothing here needs a world pose. The caller supplies the camera pose already in that
/// frame, and the heading cancels out of it -- so a deployment needs forward kinematics to
/// the camera and the IMU's roll/pitch, never odometry, and a drifting yaw estimate cannot
/// corrupt the result.
///
/// Two range subtleties, both consequences of feeding the model's own input back through the
/// pinhole model rather than reading the sensor:
///   - `image` must be the SAME array the depth channel carries, post-clip. A valid pixel
///     therefore saturates at the clip band's equivalent range and the cloud has a wall there.
///   - an invalid pixel is placed at `max_dist` of RAY LENGTH -- a spherical shell at the
///     sensor's reach -- rather than of z-depth, which would put the image corners far past
///     it. Left alone they would denormalise to 0 m and pile a fake surface on the robot's
///     own origin, the worst place for a convolution to find one.
///
/// \param cam       This camera's geometry and normalization contract.
/// \param image     Normalised depth, `cam.pixels()` values.
/// \param cam_pos   Camera origin in the yaw frame, 3 values.
/// \param cam_rot   Camera-to-yaw-frame rotation, 9 values, ROW-major.
/// \param dirs      Precomputed PinholeRayDirections for this camera.
/// \param out       `3 * cam.pixels()` values, point-major (all x, then all y, then all z).
void UnprojectToYawFrame(const CameraGeom& cam, const double* image, const double* cam_pos,
                         const double* cam_rot, const double* dirs, double* out);

/// Widen per-camera depth images to `[d, x, y, z]` (optionally `+ v`) rows, CHANNEL-MAJOR.
///
/// The layout the encoder folds: all of channel 0's pixels, then all of channel 1's, each
/// row-major over (height, width). Interleaving folds into a scrambled image with every
/// shape still correct.
///
/// Channel 0 is the image UNCHANGED, so the cloud has its wall at exactly the range the depth
/// channel claims; the xyz planes are derived from that same post-clip array.
///
/// \param cams      Per-camera geometry, in the generator's sensor-row order.
/// \param images    One normalised image per camera, each `pixels()` long.
/// \param cam_pos   One 3-vector per camera, in the yaw frame.
/// \param cam_rot   One row-major 3x3 per camera, in the yaw frame.
/// The band is applied to the DEPTH IMAGE here, before the unprojection -- never to the
/// assembled row afterwards. That ordering is the dataset's, and it is the only one that
/// means anything: `clip` is stated in NORMALISED DEPTH units, so clamping METRES of position
/// with it is a category error that merely looks reasonable because the two are similar
/// numbers. Sanitising the widened row instead clamped 21% of the far camera's position
/// values and moved points by up to 6 m, undoing one layer later exactly the range the band
/// was widened to keep. Doing it inside this function means a caller cannot get the order
/// wrong: what comes out is already clamped where clamping belongs.
///
/// \param channels  4 for `[d, x, y, z]`, 5 to append the validity plane.
/// \param clip      The band this generator trained under (FmMeta::extero_clip).
/// \param out       `cams.size() * channels * pixels()` values.
void DepthXyzRows(const std::vector<CameraGeom>& cams,
                  const std::vector<const double*>& images,
                  const std::vector<const double*>& cam_pos,
                  const std::vector<const double*>& cam_rot, int channels, double clip,
                  double* out);

}  // namespace g1_deploy
