// Copyright 2026 Zachary Olkin. All rights reserved.

// Parity: g1_deploy::QuatMath (C++) vs transfer/deploy/quat_math_np.py (the numpy oracle),
// through the recorded fixtures. Tolerances come from the manifest, not from this file.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>

#include "fixture.h"
#include "g1_deploy/quat_math.h"

namespace g1_deploy {
namespace {

using test::Case;
using test::Fixtures;

const Fixtures& Loaded() {
  static const Fixtures fixtures = Fixtures::Load(DefaultFixtureDir());
  return fixtures;
}

/// Worst elementwise error between a computed row and the oracle's.
double RowError(const double* got, const double* want, int n) {
  double worst = 0.0;
  for (int i = 0; i < n; ++i) worst = std::max(worst, std::abs(got[i] - want[i]));
  return worst;
}

/// Worst error treating the two rows as QUATERNIONS, i.e. modulo an overall sign.
///
/// q and -q are the same rotation. QuatFromMatrix picks among four candidates by argmax, so
/// near a pivot tie this and numpy can legitimately return negations of each other; asking
/// for componentwise agreement there is asking the wrong question.
double RowErrorModuloSign(const double* got, const double* want, int n) {
  double same = 0.0;
  double flipped = 0.0;
  for (int i = 0; i < n; ++i) {
    same = std::max(same, std::abs(got[i] - want[i]));
    flipped = std::max(flipped, std::abs(got[i] + want[i]));
  }
  return std::min(same, flipped);
}

/// Run `compute` over every row of a case and assert against the manifest's tolerance.
///
/// `compute` writes `out_dim` doubles for row `i`. Reports the worst row so a failure names
/// a specific input rather than just a number.
template <typename Fn>
void CheckCase(const Case& c, const std::string& out_name, int out_dim, Fn compute) {
  const auto& want = c.Out(out_name);
  ASSERT_EQ(want.stride(), out_dim) << c.name << ": fixture output width disagrees with the port";
  ASSERT_GT(want.rows(), 0) << c.name << ": empty fixture";

  double worst = 0.0;
  int worst_row = -1;
  std::vector<double> got(out_dim);
  for (int i = 0; i < want.rows(); ++i) {
    compute(i, got.data());
    const double err = (c.compare == "quat_rotation")
                           ? RowErrorModuloSign(got.data(), want.row(i), out_dim)
                           : RowError(got.data(), want.row(i), out_dim);
    if (err > worst) {
      worst = err;
      worst_row = i;
    }
  }
  EXPECT_LE(worst, c.atol) << c.name << " (" << c.fn << ", compare=" << c.compare
                           << "): worst row " << worst_row << " differs by " << worst
                           << ", tolerance " << c.atol;

  // Report the margin, not just the verdict. A parity test that passes says nothing about
  // how close it came, and "passed" with 1e-19 of headroom on a 1e-14 budget is a different
  // fact from "passed" at 9e-15. It is also the cheapest guard against a vacuous pass: an
  // all-zero error column means the comparison never really ran.
  std::cout << "    [margin] " << c.name << "/" << out_name << " worst=" << worst
            << " atol=" << c.atol << " rows=" << want.rows() << "\n";
}

QuatXyzw QuatAt(const test::Blob& b, int i) {
  const double* r = b.row(i);
  return QuatXyzw(r[0], r[1], r[2], r[3]);
}

Vec3 Vec3At(const test::Blob& b, int i) {
  const double* r = b.row(i);
  return Vec3(r[0], r[1], r[2]);
}

Mat3 Mat3At(const test::Blob& b, int i) {
  const double* r = b.row(i);
  Mat3 m;
  m << r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8];
  return m;
}

void Store(const QuatXyzw& q, double* out) {
  out[0] = q.x();
  out[1] = q.y();
  out[2] = q.z();
  out[3] = q.w();
}

void Store(const Vec3& v, double* out) {
  out[0] = v.x();
  out[1] = v.y();
  out[2] = v.z();
}

/// Every fixture case for `fn`, asserting there is at least one.
///
/// Without this assert a renamed fixture would turn its test into a silent no-op, which is
/// the failure mode a parity suite can least afford.
std::vector<const Case*> Cases(const std::string& fn) {
  auto cases = Loaded().ForFunction(fn);
  EXPECT_FALSE(cases.empty()) << "no fixture case for '" << fn << "'; regenerate the fixtures";
  return cases;
}

TEST(QuatMathParity, QuatMul) {
  for (const Case* c : Cases("quat_mul")) {
    CheckCase(*c, "result", 4, [&](int i, double* out) {
      Store(QuatMul(QuatAt(c->In("a"), i), QuatAt(c->In("b"), i)), out);
    });
  }
}

TEST(QuatMathParity, QuatConjugate) {
  for (const Case* c : Cases("quat_conjugate")) {
    CheckCase(*c, "result", 4,
              [&](int i, double* out) { Store(QuatConjugate(QuatAt(c->In("q"), i)), out); });
  }
}

TEST(QuatMathParity, QuatInv) {
  for (const Case* c : Cases("quat_inv")) {
    CheckCase(*c, "result", 4,
              [&](int i, double* out) { Store(QuatInv(QuatAt(c->In("q"), i)), out); });
  }
}

TEST(QuatMathParity, QuatApply) {
  for (const Case* c : Cases("quat_apply")) {
    CheckCase(*c, "result", 3, [&](int i, double* out) {
      Store(QuatApply(QuatAt(c->In("q"), i), Vec3At(c->In("v"), i)), out);
    });
  }
}

TEST(QuatMathParity, QuatApplyInverse) {
  for (const Case* c : Cases("quat_apply_inverse")) {
    CheckCase(*c, "result", 3, [&](int i, double* out) {
      Store(QuatApplyInverse(QuatAt(c->In("q"), i), Vec3At(c->In("v"), i)), out);
    });
  }
}

TEST(QuatMathParity, YawQuat) {
  for (const Case* c : Cases("yaw_quat")) {
    CheckCase(*c, "result", 4,
              [&](int i, double* out) { Store(YawQuat(QuatAt(c->In("q"), i)), out); });
  }
}

TEST(QuatMathParity, QuatUnique) {
  for (const Case* c : Cases("quat_unique")) {
    CheckCase(*c, "result", 4,
              [&](int i, double* out) { Store(QuatUnique(QuatAt(c->In("q"), i)), out); });
  }
}

TEST(QuatMathParity, MatrixFromQuat) {
  for (const Case* c : Cases("matrix_from_quat")) {
    CheckCase(*c, "result", 9, [&](int i, double* out) {
      const Mat3 m = MatrixFromQuat(QuatAt(c->In("q"), i));
      for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) out[r * 3 + col] = m(r, col);
      }
    });
  }
}

TEST(QuatMathParity, QuatFromMatrix) {
  for (const Case* c : Cases("quat_from_matrix")) {
    CheckCase(*c, "result", 4, [&](int i, double* out) {
      Store(QuatFromMatrix(Mat3At(c->In("matrix"), i)), out);
    });
  }
}

TEST(QuatMathParity, AxisAngleFromQuat) {
  for (const Case* c : Cases("axis_angle_from_quat")) {
    CheckCase(*c, "result", 3,
              [&](int i, double* out) { Store(AxisAngleFromQuat(QuatAt(c->In("q"), i)), out); });
  }
}

TEST(QuatMathParity, QuatFromAxisAngle) {
  for (const Case* c : Cases("quat_from_axis_angle")) {
    CheckCase(*c, "result", 4, [&](int i, double* out) {
      Store(QuatFromAxisAngle(Vec3At(c->In("delta"), i)), out);
    });
  }
}

TEST(QuatMathParity, QuatBoxMinus) {
  for (const Case* c : Cases("quat_box_minus")) {
    CheckCase(*c, "result", 3, [&](int i, double* out) {
      Store(QuatBoxMinus(QuatAt(c->In("a"), i), QuatAt(c->In("b"), i)), out);
    });
  }
}

TEST(QuatMathParity, QuatBoxPlus) {
  for (const Case* c : Cases("quat_box_plus")) {
    CheckCase(*c, "result", 4, [&](int i, double* out) {
      Store(QuatBoxPlus(QuatAt(c->In("q"), i), Vec3At(c->In("delta"), i)), out);
    });
  }
}

TEST(QuatMathParity, CombineFrameTransforms) {
  for (const Case* c : Cases("combine_frame_transforms")) {
    CheckCase(*c, "pos", 3, [&](int i, double* out) {
      Vec3 p;
      QuatXyzw q;
      CombineFrameTransforms(Vec3At(c->In("p01"), i), QuatAt(c->In("q01"), i),
                             Vec3At(c->In("p12"), i), QuatAt(c->In("q12"), i), &p, &q);
      Store(p, out);
    });
    CheckCase(*c, "quat", 4, [&](int i, double* out) {
      Vec3 p;
      QuatXyzw q;
      CombineFrameTransforms(Vec3At(c->In("p01"), i), QuatAt(c->In("q01"), i),
                             Vec3At(c->In("p12"), i), QuatAt(c->In("q12"), i), &p, &q);
      Store(q, out);
    });
  }
}

TEST(QuatMathParity, SubtractFrameTransforms) {
  for (const Case* c : Cases("subtract_frame_transforms")) {
    CheckCase(*c, "pos", 3, [&](int i, double* out) {
      Vec3 p;
      QuatXyzw q;
      SubtractFrameTransforms(Vec3At(c->In("p01"), i), QuatAt(c->In("q01"), i),
                              Vec3At(c->In("p02"), i), QuatAt(c->In("q02"), i), &p, &q);
      Store(p, out);
    });
    CheckCase(*c, "quat", 4, [&](int i, double* out) {
      Vec3 p;
      QuatXyzw q;
      SubtractFrameTransforms(Vec3At(c->In("p01"), i), QuatAt(c->In("q01"), i),
                              Vec3At(c->In("p02"), i), QuatAt(c->In("q02"), i), &p, &q);
      Store(q, out);
    });
  }
}

// Guards that do not depend on the oracle: properties the port must hold on its own.

TEST(QuatMath, BoxPlusInvertsBoxMinus) {
  const QuatXyzw a = QuatUnique(QuatFromAxisAngle(Vec3(0.3, -0.7, 1.1)));
  const QuatXyzw b = QuatUnique(QuatFromAxisAngle(Vec3(-0.2, 0.4, 0.9)));
  const QuatXyzw round_trip = QuatBoxPlus(b, QuatBoxMinus(a, b));
  EXPECT_LT(RowErrorModuloSign(round_trip.data(), a.data(), 4), 1e-12);
}

TEST(QuatMath, ZeroDeltaIsIdentityNotNan) {
  // The eps floor exists so a stationary reference -- which happens every step of a stand --
  // does not divide by zero here.
  const QuatXyzw q = QuatFromAxisAngle(Vec3::Zero());
  EXPECT_TRUE(std::isfinite(q.norm()));
  EXPECT_NEAR(q.w(), 1.0, 1e-12);
}

TEST(QuatMath, YawQuatDropsRollAndPitch) {
  // A pure roll must project to identity: the anchor supplies heading only, so a leaning
  // robot must not rotate the frame its reference is observed in.
  const QuatXyzw roll = QuatFromAxisAngle(Vec3(0.6, 0.0, 0.0));
  const QuatXyzw yaw = YawQuat(roll);
  EXPECT_NEAR(yaw.x(), 0.0, 1e-15);
  EXPECT_NEAR(yaw.y(), 0.0, 1e-15);
  EXPECT_NEAR(std::abs(yaw.w()), 1.0, 1e-12);
}

}  // namespace
}  // namespace g1_deploy
