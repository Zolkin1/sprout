// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Shared numeric types for the C++ deploy runtime.
//
// This library is the C++ port of transfer/deploy (python), which stays the parity oracle
// and the MuJoCo sim2sim driver. Everything here is transcribed from the numpy expression
// for expression, and pinned against it by transfer/deploy/tests/fixtures/parity.
//
// Two conventions carried over from the python, both of which are silent-failure material
// if broken:
//
//   * EVERY quaternion is (x, y, z, w). Eigen::Quaterniond stores xyzw internally but its
//     constructor takes (w, x, y, z) and its operators group the arithmetic differently, so
//     it is deliberately NOT used -- see quat_math.h.
//   * Every 2-D array that mirrors a numpy array is ROW-MAJOR, so .row(i) is contiguous,
//     a flatten is a memcpy, and .data() is a legal ONNX tensor pointer.

#include <Eigen/Core>

// -ffast-math would fold std::isfinite() to a compile-time true, silently deleting the
// non-finite guard chain that stops a NaN reaching the motors. There is no runtime symptom
// until there is. Fail the build instead.
#ifdef __FAST_MATH__
#error "g1_deploy must not be built with -ffast-math / -Ofast: it removes the NaN guards."
#endif

namespace g1_deploy {

//: Quaternion, (x, y, z, w).
using QuatXyzw = Eigen::Vector4d;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

//: Row-major dynamic matrices, matching numpy's default memory order.
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

}  // namespace g1_deploy
