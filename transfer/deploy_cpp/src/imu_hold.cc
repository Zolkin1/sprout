// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/imu_hold.h"

#include <cmath>

namespace g1_deploy {

ImuHold::Result ImuHold::Update(const QuatXyzw& quat, const Vec3& ang_vel) {
  ++samples_;
  const double n = quat.norm();
  if (!std::isfinite(n) || n < min_norm_) {
    ++bad_samples_;
    // Both, from the same test. See the header: the gyro is corrupted by the same dropout and
    // is undetectable on its own.
    return Result{last_quat_, last_ang_vel_, true};
  }
  last_quat_ = quat / n;
  last_ang_vel_ = ang_vel;
  any_good_ = true;
  return Result{last_quat_, last_ang_vel_, false};
}

}  // namespace g1_deploy
