// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// A short history of base poses, so the camera pose can be matched to the image's CAPTURE
// time instead of to the tick that consumes it.
//
// Why this exists: the generator's xyz channels unproject each depth image into the root's
// yaw-aligned frame, and that is only correct when pose and pixels describe the same instant.
// A torso pitch rewrites every pixel and moves the camera by the same amount, and the
// unprojection composes it back out -- which is the whole reason static terrain stays still
// when the robot tilts. Pair an image with a pose from later and it composes out the wrong
// rotation, so the terrain swings by whatever the robot did in between. On the ZED path that
// is ~50 ms; at 1 rad/s of pitch, 2.9 deg, which at 3 m of range is ~15 cm -- a stair riser.
// Training has no latency at all, so nothing in the policy is prepared for it.
//
// Sized by TIME, not by the delay expected. A frame that arrives late for any reason must
// still find its sample; the alternative is falling back to a pose from the wrong moment,
// which produces an entirely plausible cloud in the wrong place. `Nearest` says when it had
// to clamp so the caller can count it rather than absorb it.

#include <cstddef>
#include <vector>

#include "g1_deploy/types.h"

namespace g1_deploy {

/// Fixed-capacity ring of recent base poses, newest last.
class PoseHistory {
 public:
  /// One sample: when it was measured, and what camera FK needs from it.
  struct Sample {
    double stamp = -1.0;              //: the estimator message's own header stamp, seconds
    std::vector<double> joint_pos;    //: ROBOT joint order, as the message carried it
    QuatXyzw quat{0.0, 0.0, 0.0, 1.0};   //: normalized base orientation, xyzw
  };

  /// Args:
  ///   capacity: samples retained. Derive it from a DURATION and an upper bound on the
  ///     producer's rate, so the span this can match within is the stated one rather than
  ///     whatever a hardcoded count happens to buy at today's rate.
  explicit PoseHistory(size_t capacity) : ring_(capacity ? capacity : 1) {}

  /// Append one sample, evicting the oldest when full.
  ///
  /// A quaternion that fails `IsUsableQuat` is DROPPED rather than stored: it is the same
  /// all-zero estimator fault `ImuHold` guards against on the control thread, and a lookup
  /// landing on one would place the cloud with an attitude that was never measured. Dropping
  /// it means the nearest neighbour is the nearest sample that was actually measured.
  ///
  /// Does not allocate once the ring has filled: `joint_pos` is assigned over a vector that
  /// is already the right size, which matters because this runs on a 1 kHz callback.
  void Push(double stamp, const std::vector<double>& joint_pos, const QuatXyzw& quat);

  /// The sample nearest `t`, or nullptr when the history is empty.
  ///
  /// `*clamped` is set when `t` lies outside the span held, meaning the answer is the closest
  /// END rather than a match. That is the caller's cue to count it: the result is still the
  /// best available, but it is not the pose that image was taken with.
  const Sample* Nearest(double t, bool* clamped) const;

  /// Forget everything, keeping the capacity.
  ///
  /// Used on arming, where the joint-name permutation is being resolved: a sample from before
  /// it may carry a different joint set, which FK would happily permute into a plausible and
  /// wrong posture.
  void Clear() {
    head_ = 0;
    count_ = 0;
  }

  size_t size() const { return count_; }
  size_t capacity() const { return ring_.size(); }
  bool empty() const { return count_ == 0; }

 private:
  std::vector<Sample> ring_;
  size_t head_ = 0;    //: index of the OLDEST retained sample
  size_t count_ = 0;
};

}  // namespace g1_deploy
