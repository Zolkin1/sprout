// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Port of transfer/deploy/quat_math_np.py. All quaternions are (x, y, z, w).
//
// WHY THIS IS HAND-WRITTEN RATHER THAN Eigen::Quaterniond:
//
//   * Eigen's operator* and _transformVector are algebraically identical to the numpy but
//     group the arithmetic differently, so they land a few ulp away -- which is fine until
//     you are trying to hold a port to 1e-14 and cannot tell a grouping difference from a
//     real bug.
//   * Eigen's constructor takes (w, x, y, z) while every array in this system is xyzw. That
//     mismatch does not fail, it rotates things, which is the worst way to be wrong.
//   * There is no Eigen equivalent of QuatFromMatrix's branchless four-candidate argmax with
//     w >= 0 canonicalisation, and that specific formulation is what the training side used.
//
// So these are scalar functions over Eigen vectors, transcribed expression for expression
// from the numpy. Broadcasting is NOT reimplemented: every broadcast in the python is over a
// node axis of at most 62, so call sites loop. That is faster, exactly ordered, and a third
// of the code.

#include "g1_deploy/types.h"

namespace g1_deploy {

//: Matches _EPS in quat_math_np.py. Used as the denominator floor in QuatFromMatrix.
inline constexpr double kQuatEps = 1.0e-7;

//: Default small-angle threshold for the exp/log maps, matching the python `eps` default.
inline constexpr double kAxisAngleEps = 1.0e-6;

/// Cross product, written out rather than Eigen's .cross().
///
/// Not because Eigen is wrong -- it agrees -- but because this file's job is to be checkable
/// against the numpy by reading, and np.cross's term order should be visible. It also keeps
/// the library off Eigen/Geometry, which nothing else here needs.
inline Vec3 Cross(const Vec3& a, const Vec3& b) {
  return Vec3(a.y() * b.z() - a.z() * b.y(),
              a.z() * b.x() - a.x() * b.z(),
              a.x() * b.y() - a.y() * b.x());
}

/// Hamilton product a (x) b.
QuatXyzw QuatMul(const QuatXyzw& a, const QuatXyzw& b);

/// Conjugate [-x, -y, -z, w].
QuatXyzw QuatConjugate(const QuatXyzw& q);

/// Inverse: conjugate / squared norm. Exact for unit quaternions.
QuatXyzw QuatInv(const QuatXyzw& q);

/// Rotate v by q.
Vec3 QuatApply(const QuatXyzw& q, const Vec3& v);

/// Rotate v by the inverse of q.
Vec3 QuatApplyInverse(const QuatXyzw& q, const Vec3& v);

/// Heading-only quaternion: drop roll and pitch, keep the z rotation.
///
/// The anchor supplies heading only, so a leaning robot does not rotate the frame its
/// reference is observed in.
QuatXyzw YawQuat(const QuatXyzw& q);

/// Rotation matrix from a quaternion.
Mat3 MatrixFromQuat(const QuatXyzw& q);

/// Quaternion from a rotation matrix.
///
/// Branchless four-candidate formulation; selects the candidate with the largest pivot and
/// canonicalises w >= 0. Near a pivot tie this and the numpy can select DIFFERENT candidates
/// of the four -- they are the same rotation to ~1e-8 but differ in sign pattern, which is
/// why the parity fixtures compare this one as a rotation rather than componentwise.
QuatXyzw QuatFromMatrix(const Mat3& matrix);

/// Compose T01 (x) T12 -> (p02, q02).
void CombineFrameTransforms(const Vec3& p01, const QuatXyzw& q01, const Vec3& p12,
                            const QuatXyzw& q12, Vec3* p02, QuatXyzw* q02);

/// Inverse of CombineFrameTransforms: T01^-1 (x) T02 -> (p12, q12).
void SubtractFrameTransforms(const Vec3& p01, const QuatXyzw& q01, const Vec3& p02,
                             const QuatXyzw& q02, Vec3* p12, QuatXyzw* q12);

/// Axis-angle (so(3)) rotation vector from a quaternion.
Vec3 AxisAngleFromQuat(const QuatXyzw& quat, double eps = kAxisAngleEps);

/// Tangent-space difference q1 (-) q2 -> rotation vector.
Vec3 QuatBoxMinus(const QuatXyzw& q1, const QuatXyzw& q2);

/// Standard form with a non-negative real part.
///
/// q and -q are the same rotation; picking one makes results comparable componentwise.
QuatXyzw QuatUnique(const QuatXyzw& q);

/// Exponential map: rotation vector -> quaternion. Inverse of AxisAngleFromQuat.
///
/// eps floors the norm so a zero perturbation gives identity rather than dividing by zero.
QuatXyzw QuatFromAxisAngle(const Vec3& delta, double eps = kAxisAngleEps);

/// Apply an axis-angle increment: Exp(delta) * q, sign-normalised.
///
/// Returns a quaternion with a non-negative real part -- possibly the negation of what a
/// caller expects, which is the SAME rotation. Compare rotations, not components.
QuatXyzw QuatBoxPlus(const QuatXyzw& q, const Vec3& delta, double eps = kAxisAngleEps);

}  // namespace g1_deploy
