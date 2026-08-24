// Copyright 2026 Zachary Olkin. All rights reserved.

// Parity: g1_deploy depth normalization vs transfer/deploy/depth_obs.py.
//
// Held to EXACT equality (the manifest carries atol = 0). There is nothing here that could
// legitimately differ by an ulp, so anything that does is a real behavioural difference.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixture.h"
#include "g1_deploy/depth_obs.h"

namespace g1_deploy {
namespace {

using test::Blob;
using test::Case;
using test::Fixtures;

const Fixtures& Loaded() {
  static const Fixtures fixtures = Fixtures::Load(DefaultFixtureDir());
  return fixtures;
}

std::vector<const Case*> Cases(const std::string& fn) {
  auto cases = Loaded().ForFunction(fn);
  EXPECT_FALSE(cases.empty()) << "no fixture case for '" << fn << "'; regenerate the fixtures";
  return cases;
}

/// The camera contract the fixture was generated with. Read from the manifest, never
/// defaulted -- a port that silently used its own defaults would agree with the oracle on
/// the default case and diverge on every real one.
DepthNorm NormOf(const Case& c) {
  DepthNorm n;
  n.offset = c.Double("offset");
  n.scale = c.Double("scale");
  n.min_dist = c.Double("min_dist");
  n.max_dist = c.Double("max_dist");
  n.min_clip_val = c.Double("min_clip_val");
  n.max_clip_val = c.Double("max_clip_val");
  return n;
}

TEST(DepthParity, NormalizeDepthIsExact) {
  for (const Case* c : Cases("normalize_depth")) {
    const Blob& in = c->In("d");
    const Blob& want = c->Out("result");
    ASSERT_EQ(in.data.size(), want.data.size());
    const DepthNorm n = NormOf(*c);

    int mismatches = 0;
    int first_bad = -1;
    for (size_t i = 0; i < in.data.size(); ++i) {
      const double got = NormalizeDepth(in.data[i], n);
      // Bit equality, not a tolerance. Both sides are finite here by construction (every
      // invalid class is substituted before the affine), so == is the right comparison and
      // a NaN on either side is itself a failure.
      if (!(got == want.data[i])) {
        ++mismatches;
        if (first_bad < 0) first_bad = static_cast<int>(i);
      }
    }
    EXPECT_EQ(mismatches, 0) << c->name << ": " << mismatches << " of " << in.data.size()
                             << " pixels differ; first at index " << first_bad << " (input "
                             << in.data[first_bad < 0 ? 0 : first_bad] << ")";
    std::cout << "    [margin] " << c->name << " exact over " << in.data.size() << " values\n";
  }
}

TEST(DepthParity, SanitizeExteroIsExact) {
  for (const Case* c : Cases("sanitize_extero")) {
    const Blob& in = c->In("values");
    const Blob& want = c->Out("result");
    const std::string source = c->strings.at("source");
    std::vector<double> got(in.data.size());
    SanitizeExtero(in.data.data(), in.data.size(), source, got.data());

    int mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      if (!(got[i] == want.data[i])) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0) << c->name << " (" << source << "): " << mismatches << " differ";
    std::cout << "    [margin] " << c->name << " exact over " << got.size() << " values\n";
  }
}

// The ordering properties, stated directly. The fixtures cover these too, but a failure
// there reports "index 517 differs" while these say which rule broke.

TEST(Depth, NegativeInfinityBecomesMinClipNotMax) {
  // The too-near test runs FIRST, so -inf is caught there. Swap the two lines and this is
  // the only input that notices -- and it is one a real camera produces.
  DepthNorm n;
  n.min_clip_val = 1.0;
  n.max_clip_val = 8.0;
  EXPECT_EQ(NormalizeDepth(-std::numeric_limits<double>::infinity(), n),
            (n.min_clip_val - n.offset) * n.scale);
}

TEST(Depth, PositiveInfinityAndNanBecomeMaxClip) {
  DepthNorm n;
  n.min_clip_val = 1.0;
  n.max_clip_val = 8.0;
  const double expect = (n.max_clip_val - n.offset) * n.scale;
  EXPECT_EQ(NormalizeDepth(std::numeric_limits<double>::infinity(), n), expect);
  // NaN fails BOTH comparisons, so it reaches the non-finite test and lands here. If the
  // port used a clamp that "handled" NaN, it would land somewhere else silently.
  EXPECT_EQ(NormalizeDepth(std::numeric_limits<double>::quiet_NaN(), n), expect);
}

TEST(Depth, BoundsAreStrictSoReadingsAtTheLimitStayValid) {
  DepthNorm n;
  n.min_clip_val = 1.0;
  n.max_clip_val = 8.0;
  EXPECT_EQ(NormalizeDepth(n.min_dist, n), (n.min_dist - n.offset) * n.scale);
  EXPECT_EQ(NormalizeDepth(n.max_dist, n), (n.max_dist - n.offset) * n.scale);
  // One ulp outside is invalid, which is what makes the comparison strictness observable.
  EXPECT_EQ(NormalizeDepth(std::nextafter(n.min_dist, -1e9), n),
            (n.min_clip_val - n.offset) * n.scale);
  EXPECT_EQ(NormalizeDepth(std::nextafter(n.max_dist, 1e9), n),
            (n.max_clip_val - n.offset) * n.scale);
}

TEST(Depth, EveryInvalidClassLandsOnInvalidDepthValue) {
  // Consumers identify "no reading" by comparing against this one value, so all of the
  // invalid classes except too-near have to agree with it exactly.
  DepthNorm n;
  n.max_clip_val = 8.0;
  const double invalid = InvalidDepthValue(n);
  EXPECT_EQ(NormalizeDepth(std::numeric_limits<double>::infinity(), n), invalid);
  EXPECT_EQ(NormalizeDepth(std::numeric_limits<double>::quiet_NaN(), n), invalid);
  EXPECT_EQ(NormalizeDepth(n.max_dist + 1.0, n), invalid);
}

TEST(Depth, SanitizeExteroClipsPerSourceBand) {
  // Depth is already normalised and its obs terms clip at +-2; the height scan is metres of
  // clearance at +-1. Using the scan's band for depth would silently discard real readings.
  EXPECT_EQ(ExteroClip("height_scan"), 1.0);
  EXPECT_EQ(ExteroClip("depth"), 2.0);
  EXPECT_THROW(ExteroClip("lidar"), std::runtime_error);

  const double in[] = {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity(), 5.0, -5.0};
  double out[5];
  SanitizeExtero(in, 5, "depth", out);
  EXPECT_EQ(out[0], 0.0);   // NaN -> 0, not -> clip
  EXPECT_EQ(out[1], 2.0);
  // -inf -> +clip, NOT -clip: a missed ray is "no ground within range", i.e. ground far
  // below, so it belongs at the drop-off end. -clip would encode a hole as a wall.
  EXPECT_EQ(out[2], 2.0);
  EXPECT_EQ(out[3], 2.0);
  EXPECT_EQ(out[4], -2.0);
}

/// The camera geometry a fixture case was generated with, alongside its normalization.
CameraGeom GeomOf(const Case& c) {
  CameraGeom g;
  g.width = c.ints.at("width");
  g.height = c.ints.at("height");
  g.fx = c.Double("fx");
  g.fy = c.Double("fy");
  g.cx = c.Double("cx");
  g.cy = c.Double("cy");
  g.norm = NormOf(c);
  return g;
}

TEST(DepthParity, PinholeRayDirectionsMatchPython) {
  for (const Case* c : Cases("pinhole_ray_directions")) {
    const CameraGeom g = GeomOf(*c);
    const Blob& want = c->Out("dirs");
    const std::vector<double> got = PinholeRayDirections(g);
    ASSERT_EQ(got.size(), want.data.size());

    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      worst = std::max(worst, std::abs(got[i] - want.data[i]));
    }
    EXPECT_LE(worst, c->atol) << c->name << ": worst |d| " << worst;
    // The ordering is the thing this can silently get wrong: a transposed loop gives the
    // same SET of directions and the same norms, so only position-by-position agreement
    // catches it. Pixel 0 must be the TOP-LEFT, i.e. +y (left) and +z (up) both positive.
    EXPECT_GT(got[1], 0.0) << "pixel 0 is not the left column -- u/v loops transposed?";
    EXPECT_GT(got[2], 0.0) << "pixel 0 is not the top row -- u/v loops transposed?";
    std::cout << "    [margin] " << c->name << " worst " << worst << " vs atol " << c->atol
              << "\n";
  }
}

TEST(DepthParity, InvalidDepthMaskIsExact) {
  for (const Case* c : Cases("invalid_depth_mask")) {
    const CameraGeom g = GeomOf(*c);
    const Blob& img = c->In("image");
    const Blob& want = c->Out("mask");
    ASSERT_EQ(img.data.size(), want.data.size());

    std::vector<char> got(img.data.size());
    InvalidDepthMask(img.data.data(), img.data.size(), g.norm,
                     reinterpret_cast<bool*>(got.data()));
    int mismatches = 0;
    int n_invalid = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      const double as_double = got[i] ? 1.0 : 0.0;
      if (as_double != want.data[i]) ++mismatches;
      n_invalid += got[i] ? 1 : 0;
    }
    EXPECT_EQ(mismatches, 0) << c->name << ": " << mismatches << " pixels classified differently";
    // The fixture is built to contain both, so a mask that is all-true or all-false would
    // agree with nothing real and must not pass by being uniform.
    EXPECT_GT(n_invalid, 0) << "no invalid pixels -- the fixture should carry misses";
    EXPECT_LT(n_invalid, static_cast<int>(got.size())) << "every pixel invalid";
    std::cout << "    [margin] " << c->name << " exact over " << got.size() << " pixels ("
              << n_invalid << " invalid)\n";
  }
}

TEST(DepthParity, UnprojectToYawFrameMatchesPython) {
  for (const Case* c : Cases("unproject_to_yaw_frame")) {
    const CameraGeom g = GeomOf(*c);
    const Blob& img = c->In("image");
    const Blob& pos = c->In("cam_pos");
    const Blob& rot = c->In("cam_rot");
    const Blob& want = c->Out("points");
    const std::vector<double> dirs = PinholeRayDirections(g);
    const size_t k = g.pixels();
    ASSERT_EQ(static_cast<size_t>(img.stride()), k);

    double worst = 0.0;
    std::vector<double> got(3 * k);
    for (int r = 0; r < img.rows(); ++r) {
      UnprojectToYawFrame(g, img.row(r), pos.row(r), rot.row(r), dirs.data(), got.data());
      for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, std::abs(got[i] - want.row(r)[i]));
      }
    }
    EXPECT_LE(worst, c->atol) << c->name << ": worst |d| " << worst << " m over " << img.rows()
                              << " poses";
    std::cout << "    [margin] " << c->name << " worst " << worst << " m vs atol " << c->atol
              << " over " << img.rows() << " poses\n";
  }
}

TEST(Depth, XyzRowsAreChannelMajorAndKeepTheDepthPlane) {
  // Layout, not numerics -- the geometry is covered by the parity cases above. This is what
  // an interleaved row would break while every shape stayed correct, scrambling the image
  // the encoder folds and every convolution kernel with it.
  CameraGeom g;
  g.width = 5;
  g.height = 4;
  g.fx = 2.5;
  g.fy = 2.2;
  g.cx = 2.5;
  g.cy = 2.0;
  g.norm.min_dist = 0.1;
  g.norm.max_dist = 7.9;
  const size_t k = g.pixels();

  std::vector<double> img(k);
  for (size_t i = 0; i < k; ++i) {
    // A spread of real readings with a few misses mixed in, normalised through the contract.
    img[i] = NormalizeDepth(i % 4 == 0 ? 0.01 : 0.5 + 0.1 * static_cast<double>(i), g.norm);
  }
  const double pos[3] = {0.15, 0.0, 0.30};
  const double rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<CameraGeom> cams{g, g};
  const std::vector<const double*> images{img.data(), img.data()};
  const std::vector<const double*> positions{pos, pos};
  const std::vector<const double*> rotations{rot, rot};

  std::vector<double> out(2 * 5 * k);
  DepthXyzRows(cams, images, positions, rotations, 5, /*clip=*/5.575, out.data());

  for (size_t c = 0; c < 2; ++c) {
    const double* row = out.data() + c * 5 * k;
    for (size_t i = 0; i < k; ++i) {
      EXPECT_EQ(row[i], img[i]) << "channel 0 must be the image itself, camera " << c;
    }
    std::vector<char> invalid(k);
    InvalidDepthMask(img.data(), k, g.norm, reinterpret_cast<bool*>(invalid.data()));
    int n_invalid = 0;
    for (size_t i = 0; i < k; ++i) {
      EXPECT_EQ(row[4 * k + i], invalid[i] ? 0.0 : 1.0) << "validity plane, pixel " << i;
      n_invalid += invalid[i] ? 1 : 0;
    }
    EXPECT_GT(n_invalid, 0);
    EXPECT_LT(n_invalid, static_cast<int>(k));
  }
  EXPECT_THROW(DepthXyzRows(cams, images, positions, rotations, 1, 5.575, out.data()),
               std::invalid_argument);
  EXPECT_THROW(DepthXyzRows(cams, {img.data()}, positions, rotations, 5, 5.575, out.data()),
               std::invalid_argument);

  // The band bounds the DEPTH plane and not the position planes: xyz are metres, the band is
  // normalised depth units, and conflating them clamps the far camera's real reach.
  CameraGeom far = g;
  far.fx = 21.42222;
  far.fy = 33.86616;
  far.norm.max_dist = 11.9;
  std::vector<double> reach(k);
  for (size_t i = 0; i < k; ++i) {
    reach[i] = NormalizeDepth(0.5 + 11.0 * static_cast<double>(i) / static_cast<double>(k),
                              far.norm);
  }
  std::vector<double> one(5 * k);
  DepthXyzRows({far}, {reach.data()}, {pos}, {rot}, 5, 5.575, one.data());
  double worst_xyz = 0.0;
  for (size_t i = k; i < 4 * k; ++i) worst_xyz = std::max(worst_xyz, std::abs(one[i]));
  EXPECT_GT(worst_xyz, 5.575) << "xyz must not be limited by a normalised-depth band";
  for (size_t i = 0; i < k; ++i) {
    EXPECT_LE(std::abs(one[i]), 5.575 + 1e-12) << "the depth plane IS bounded by the band";
  }
}

}  // namespace
}  // namespace g1_deploy
