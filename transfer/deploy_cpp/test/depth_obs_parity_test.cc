// Copyright 2026 Zachary Olkin. All rights reserved.

// Parity: g1_deploy depth normalization vs transfer/deploy/depth_obs.py.
//
// Held to EXACT equality (the manifest carries atol = 0). There is nothing here that could
// legitimately differ by an ulp, so anything that does is a real behavioural difference.

#include <gtest/gtest.h>

#include <cmath>
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
  static const Fixtures fixtures = Fixtures::Load(Fixtures::DefaultDir());
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
  EXPECT_EQ(out[2], -2.0);
  EXPECT_EQ(out[3], 2.0);
  EXPECT_EQ(out[4], -2.0);
}

}  // namespace
}  // namespace g1_deploy
