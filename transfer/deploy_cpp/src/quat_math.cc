// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/quat_math.h"

#include <algorithm>
#include <cmath>

namespace g1_deploy {

QuatXyzw QuatMul(const QuatXyzw& a, const QuatXyzw& b) {
  const double ax = a.x(), ay = a.y(), az = a.z(), aw = a.w();
  const double bx = b.x(), by = b.y(), bz = b.z(), bw = b.w();
  return QuatXyzw(aw * bx + ax * bw + ay * bz - az * by,
                  aw * by - ax * bz + ay * bw + az * bx,
                  aw * bz + ax * by - ay * bx + az * bw,
                  aw * bw - ax * bx - ay * by - az * bz);
}

QuatXyzw QuatConjugate(const QuatXyzw& q) { return QuatXyzw(-q.x(), -q.y(), -q.z(), q.w()); }

QuatXyzw QuatInv(const QuatXyzw& q) {
  // Squared norm summed in the same order numpy's (q * q).sum(-1) walks the axis.
  const double sq = q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w();
  return QuatXyzw(QuatConjugate(q) / sq);
}

Vec3 QuatApply(const QuatXyzw& q, const Vec3& v) {
  const Vec3 qvec(q.x(), q.y(), q.z());
  const double w = q.w();
  const Vec3 t = 2.0 * Cross(qvec, v);
  return v + w * t + Cross(qvec, t);
}

Vec3 QuatApplyInverse(const QuatXyzw& q, const Vec3& v) { return QuatApply(QuatConjugate(q), v); }

QuatXyzw YawQuat(const QuatXyzw& q) {
  const double qx = q.x(), qy = q.y(), qz = q.z(), qw = q.w();
  const double yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  const double half = 0.5 * yaw;
  return QuatXyzw(0.0, 0.0, std::sin(half), std::cos(half));
}

Mat3 MatrixFromQuat(const QuatXyzw& q) {
  const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
  const double xx = x * x, yy = y * y, zz = z * z;
  const double xy = x * y, xz = x * z, yz = y * z;
  const double wx = w * x, wy = w * y, wz = w * z;
  Mat3 m;
  m << 1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),
       2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
       2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy);
  return m;
}

QuatXyzw QuatFromMatrix(const Mat3& matrix) {
  const double m00 = matrix(0, 0), m01 = matrix(0, 1), m02 = matrix(0, 2);
  const double m10 = matrix(1, 0), m11 = matrix(1, 1), m12 = matrix(1, 2);
  const double m20 = matrix(2, 0), m21 = matrix(2, 1), m22 = matrix(2, 2);

  // The clip at 0 is load-bearing: for a matrix that is only approximately a rotation (the
  // generator emits a raw 9-vector) a trace term can go slightly negative, and sqrt of that
  // is NaN rather than a small number.
  double q_abs[4];
  q_abs[0] = std::sqrt(std::max(1.0 + m00 + m11 + m22, 0.0));
  q_abs[1] = std::sqrt(std::max(1.0 + m00 - m11 - m22, 0.0));
  q_abs[2] = std::sqrt(std::max(1.0 - m00 + m11 - m22, 0.0));
  q_abs[3] = std::sqrt(std::max(1.0 - m00 - m11 + m22, 0.0));

  const QuatXyzw candidates[4] = {
      QuatXyzw(m21 - m12, m02 - m20, m10 - m01, q_abs[0] * q_abs[0]),
      QuatXyzw(q_abs[1] * q_abs[1], m10 + m01, m02 + m20, m21 - m12),
      QuatXyzw(m10 + m01, q_abs[2] * q_abs[2], m12 + m21, m02 - m20),
      QuatXyzw(m02 + m20, m12 + m21, q_abs[3] * q_abs[3], m10 - m01),
  };

  // np.argmax returns the FIRST maximum on a tie. std::max_element does too, but spell the
  // comparison out: on a pivot tie the choice of candidate is exactly what makes this
  // function's output differ in sign from the numpy's, and the tie-break must match.
  int pivot = 0;
  for (int i = 1; i < 4; ++i) {
    if (q_abs[i] > q_abs[pivot]) pivot = i;
  }

  const double denom = std::max(2.0 * q_abs[pivot], 4.0 * kQuatEps);
  QuatXyzw quat = candidates[pivot] / denom;
  if (quat.w() < 0.0) quat = -quat;
  return QuatXyzw(quat / std::max(quat.norm(), kQuatEps));
}

void CombineFrameTransforms(const Vec3& p01, const QuatXyzw& q01, const Vec3& p12,
                            const QuatXyzw& q12, Vec3* p02, QuatXyzw* q02) {
  *q02 = QuatMul(q01, q12);
  *p02 = p01 + QuatApply(q01, p12);
}

void SubtractFrameTransforms(const Vec3& p01, const QuatXyzw& q01, const Vec3& p02,
                             const QuatXyzw& q02, Vec3* p12, QuatXyzw* q12) {
  const QuatXyzw q01_inv = QuatConjugate(q01);
  *q12 = QuatMul(q01_inv, q02);
  *p12 = QuatApply(q01_inv, p02 - p01);
}

Vec3 AxisAngleFromQuat(const QuatXyzw& quat, double eps) {
  // Canonicalise w >= 0 by multiplying by +/-1, the same way the numpy does it, so the
  // rotation vector is the short way round.
  const QuatXyzw q = quat * (1.0 - 2.0 * (quat.w() < 0.0 ? 1.0 : 0.0));
  const Vec3 xyz(q.x(), q.y(), q.z());
  const double mag = xyz.norm();
  const double half_angle = std::atan2(mag, q.w());
  const double angle = 2.0 * half_angle;
  // Taylor branch near zero: sin(a/2)/a -> 1/2 - a^2/48. Without it this is 0/0 at identity,
  // which is not a rare input here -- a stationary reference hits it every step.
  const double sin_half_over_angle =
      (std::abs(angle) > eps) ? (std::sin(half_angle) / angle) : (0.5 - angle * angle / 48.0);
  return xyz / sin_half_over_angle;
}

Vec3 QuatBoxMinus(const QuatXyzw& q1, const QuatXyzw& q2) {
  return AxisAngleFromQuat(QuatMul(q1, QuatConjugate(q2)));
}

QuatXyzw QuatUnique(const QuatXyzw& q) { return q.w() < 0.0 ? QuatXyzw(-q) : q; }

QuatXyzw QuatFromAxisAngle(const Vec3& delta, double eps) {
  const double norm = std::max(delta.norm(), eps);
  const double half = 0.5 * norm;
  const Vec3 xyz = (delta / norm) * std::sin(half);
  const QuatXyzw q(xyz.x(), xyz.y(), xyz.z(), std::cos(half));
  return QuatXyzw(q / q.norm());
}

QuatXyzw QuatBoxPlus(const QuatXyzw& q, const Vec3& delta, double eps) {
  return QuatUnique(QuatMul(QuatFromAxisAngle(delta, eps), q));
}

}  // namespace g1_deploy
