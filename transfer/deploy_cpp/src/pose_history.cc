// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/pose_history.h"

#include <cmath>

#include "g1_deploy/imu_hold.h"

namespace g1_deploy {

void PoseHistory::Push(double stamp, const std::vector<double>& joint_pos,
                       const QuatXyzw& quat) {
  if (!IsUsableQuat(quat)) return;
  size_t slot = 0;
  if (count_ == ring_.size()) {
    slot = head_;
    head_ = (head_ + 1) % ring_.size();
  } else {
    slot = (head_ + count_) % ring_.size();
    ++count_;
  }
  Sample& out = ring_[slot];
  out.stamp = stamp;
  out.joint_pos.assign(joint_pos.begin(), joint_pos.end());
  out.quat = quat / quat.norm();
}

const PoseHistory::Sample* PoseHistory::Nearest(double t, bool* clamped) const {
  if (count_ == 0) {
    *clamped = true;
    return nullptr;
  }
  const size_t n = ring_.size();
  const Sample* best = nullptr;
  double best_dt = 0.0;
  // Linear rather than a binary search: the ring wraps, the span is a few hundred entries, and
  // this runs twice per 20 ms tick. A wraparound bug here would be a pose from the wrong
  // moment, which is invisible in the output -- so the boring version is the right one.
  for (size_t i = 0; i < count_; ++i) {
    const Sample& cand = ring_[(head_ + i) % n];
    const double dt = std::fabs(cand.stamp - t);
    if (best == nullptr || dt < best_dt) {
      best = &cand;
      best_dt = dt;
    }
  }
  const double oldest = ring_[head_].stamp;
  const double newest = ring_[(head_ + count_ - 1) % n].stamp;
  *clamped = (t < oldest) || (t > newest);
  return best;
}

}  // namespace g1_deploy
