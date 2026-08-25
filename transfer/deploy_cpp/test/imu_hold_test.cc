// Copyright 2026 Zachary Olkin. All rights reserved.

// The hold-last-valid IMU guard, for the robot that intermittently publishes an all-zero
// estimator message.
//
// The invariant worth pinning is the GYRO one. The quaternion half of this guard existed for
// months while the angular velocity went straight through unchecked -- because a zeroed gyro
// reads as a plausible "standing still" and nothing about it looks wrong. That omission is the
// exact shape of regression these tests exist to catch, so they assert on the gyro at least as
// hard as on the orientation.

#include <gtest/gtest.h>

#include "g1_deploy/imu_hold.h"

namespace g1_deploy {
namespace {

//: A tilted, definitely-not-identity orientation, and a definitely-nonzero rotation rate.
//: Normalised here so the expectations below can be exact: Update renormalises what it
//: accepts, so comparing against a 7-digit literal would only measure that literal's error.
const QuatXyzw kGood = QuatXyzw(0.0, 0.3826834, 0.0, 0.9238795).normalized();  // 45 deg / y
const Vec3 kSpin(0.7, -1.3, 0.2);

TEST(ImuHold, PassesAGoodSampleThroughNormalised) {
  ImuHold hold;
  const ImuHold::Result r = hold.Update(kGood, kSpin);
  EXPECT_FALSE(r.held);
  EXPECT_NEAR(r.quat.norm(), 1.0, 1e-12);
  EXPECT_NEAR((r.quat - kGood).norm(), 0.0, 1e-15);
  EXPECT_NEAR((r.ang_vel - kSpin).norm(), 0.0, 1e-12);
  EXPECT_TRUE(hold.any_good());
  EXPECT_EQ(hold.samples(), 1);
  EXPECT_EQ(hold.bad_samples(), 0);
}

TEST(ImuHold, AnAllZeroSampleHoldsTheGyroAndNotOnlyTheOrientation) {
  ImuHold hold;
  hold.Update(kGood, kSpin);
  // The real fault: the WHOLE message is zeros, so the gyro is zero too -- and a zero gyro is
  // indistinguishable from a robot standing still, which is why it has to ride on the
  // quaternion's test rather than on one of its own.
  const ImuHold::Result r = hold.Update(QuatXyzw::Zero(), Vec3::Zero());
  EXPECT_TRUE(r.held);
  EXPECT_NEAR((r.quat - kGood).norm(), 0.0, 1e-15) << "orientation was not held";
  EXPECT_NEAR((r.ang_vel - kSpin).norm(), 0.0, 1e-12)
      << "the ANGULAR VELOCITY was not held: the policy would be told it is not rotating";
  EXPECT_EQ(hold.bad_samples(), 1);
  EXPECT_EQ(hold.samples(), 2);
}

TEST(ImuHold, RejectsOnNormAloneAndNotOnTheGyrosValue) {
  ImuHold hold;
  hold.Update(kGood, kSpin);
  // A legitimately stationary robot: good quaternion, genuinely zero rotation. This must be
  // ACCEPTED -- a guard that rejected it would refuse every standing sample.
  const ImuHold::Result r = hold.Update(kGood, Vec3::Zero());
  EXPECT_FALSE(r.held);
  EXPECT_NEAR(r.ang_vel.norm(), 0.0, 1e-12);
  EXPECT_EQ(hold.bad_samples(), 0);
}

TEST(ImuHold, HoldsAcrossARunAndResumesOnRecovery) {
  ImuHold hold;
  hold.Update(kGood, kSpin);
  for (int i = 0; i < 5; ++i) {
    const ImuHold::Result r = hold.Update(QuatXyzw::Zero(), Vec3::Zero());
    EXPECT_TRUE(r.held);
    EXPECT_NEAR((r.ang_vel - kSpin).norm(), 0.0, 1e-12);
  }
  EXPECT_EQ(hold.bad_samples(), 5);
  const QuatXyzw fresh(0.0, 0.0, 0.7071068, 0.7071068);
  const Vec3 fresh_spin(-0.1, 0.0, 2.0);
  const ImuHold::Result r = hold.Update(fresh, fresh_spin);
  EXPECT_FALSE(r.held);
  EXPECT_NEAR((r.ang_vel - fresh_spin).norm(), 0.0, 1e-12);
}

TEST(ImuHold, NonFiniteIsRejectedToo) {
  ImuHold hold;
  hold.Update(kGood, kSpin);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(hold.Update(QuatXyzw(nan, nan, nan, nan), kSpin).held);
  EXPECT_TRUE(hold.Update(QuatXyzw(0.0, 0.0, 0.0, 1.0) *
                          std::numeric_limits<double>::infinity(), kSpin).held);
}

TEST(ImuHold, BeforeAnyGoodSampleTheOutputIsAdmittedlyInvented) {
  // Nothing has been measured yet, so the identity returned here is a fabrication. any_good()
  // is how a caller distinguishes it from a real reading; without checking it, a bad FIRST
  // sample silently presents a perfectly upright robot.
  ImuHold hold;
  const ImuHold::Result r = hold.Update(QuatXyzw::Zero(), Vec3::Zero());
  EXPECT_TRUE(r.held);
  EXPECT_FALSE(hold.any_good());
  EXPECT_NEAR((r.quat - QuatXyzw(0.0, 0.0, 0.0, 1.0)).norm(), 0.0, 1e-12);
  EXPECT_NEAR(r.ang_vel.norm(), 0.0, 1e-12);
}

TEST(ImuHold, ResetCountersKeepsTheHeldValues) {
  ImuHold hold;
  hold.Update(kGood, kSpin);
  hold.Update(QuatXyzw::Zero(), Vec3::Zero());
  hold.ResetCounters();
  EXPECT_EQ(hold.samples(), 0);
  EXPECT_EQ(hold.bad_samples(), 0);
  EXPECT_TRUE(hold.any_good()) << "re-arming does not un-measure the robot";
  const ImuHold::Result r = hold.Update(QuatXyzw::Zero(), Vec3::Zero());
  EXPECT_NEAR((r.quat - kGood).norm(), 0.0, 1e-15);
  EXPECT_NEAR((r.ang_vel - kSpin).norm(), 0.0, 1e-12);
}

TEST(ImuHold, TheThresholdIsConfigurable) {
  ImuHold strict(0.99);
  // Norm 0.8: garbage under a strict threshold, fine under the default.
  const QuatXyzw shrunk = kGood * 0.8;
  EXPECT_TRUE(strict.Update(shrunk, kSpin).held);
  ImuHold loose(0.5);
  EXPECT_FALSE(loose.Update(shrunk, kSpin).held);
}

}  // namespace
}  // namespace g1_deploy
