// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Hold-last-valid for the base IMU, for a robot whose estimator intermittently publishes an
// ALL-ZERO message (measured at roughly 1 sample in 10 on one G1). A hardware fault, patched
// in software because the alternative is not running.
//
// The detector is the base quaternion's NORM. That choice is forced rather than preferred:
// a unit quaternion has norm 1 and a zeroed one has norm 0, so the corruption is trivially
// visible there -- while the SAME dropout leaves the gyro reading (0, 0, 0), which is a
// perfectly plausible measurement (a robot standing still) that no test on the gyro alone can
// reject. So the quaternion gates both, and everything the dropout corrupts is held together.
//
// Why holding is safe: the estimator runs at 1 kHz against a 50 Hz control tick, so one held
// sample is ~1 ms of staleness in a value the policy reads every 20 ms. Even if only the
// quaternion were ever bad -- and a good gyro were therefore held needlessly -- that is far
// cheaper than telling a falling robot's policy that it is not rotating.
//
// What this deliberately does NOT do: bound the number of consecutive holds. A sustained
// dropout keeps the messages ARRIVING, so an age watchdog cannot see it, and this class would
// serve a frozen attitude indefinitely behind a counter nobody is watching. That is survivable
// for the spotty fault it was written for and wrong for a disconnected IMU; the caller owns the
// escalation policy, and `bad_samples()` is what it should escalate on.

#include <cstdint>

#include "g1_deploy/types.h"

namespace g1_deploy {

/// Last-valid hold for base orientation + angular velocity.
class ImuHold {
 public:
  /// One update's worth of output.
  struct Result {
    QuatXyzw quat;    //: normalized, and equal to the last good one when `held`
    Vec3 ang_vel;
    bool held;        //: this sample was rejected and the previous one reused
  };

  /// Args:
  ///   min_norm: a quaternion whose norm is below this (or non-finite) is treated as NO
  ///     READING. 0.5 is the shipped value: the test must reject garbage, not police
  ///     normalisation, and Update renormalises the good path anyway.
  explicit ImuHold(double min_norm = 0.5) : min_norm_(min_norm) {}

  /// Accept `quat`/`ang_vel` if the quaternion is usable, else reuse the last good pair.
  ///
  /// Before any good sample has ever arrived, a rejected sample yields the identity
  /// orientation and a zero gyro -- a fabrication, and the reason `any_good()` exists for a
  /// caller that would rather not act at all than act on it.
  Result Update(const QuatXyzw& quat, const Vec3& ang_vel);

  /// Whether a usable sample has ever been seen. False means any `Result` so far was invented.
  bool any_good() const { return any_good_; }

  int64_t samples() const { return samples_; }
  int64_t bad_samples() const { return bad_samples_; }

  /// Zero the counters, keeping the held values.
  ///
  /// The values are kept on purpose: they are the most recent real measurement of where the
  /// robot is, and re-arming does not make them less true. Clearing them to identity would
  /// hand the next tick a fabricated upright attitude if that tick's sample is a bad one.
  void ResetCounters() {
    samples_ = 0;
    bad_samples_ = 0;
  }

 private:
  double min_norm_;
  QuatXyzw last_quat_{0.0, 0.0, 0.0, 1.0};
  Vec3 last_ang_vel_{Vec3::Zero()};
  bool any_good_ = false;
  int64_t samples_ = 0;
  int64_t bad_samples_ = 0;
};

}  // namespace g1_deploy
