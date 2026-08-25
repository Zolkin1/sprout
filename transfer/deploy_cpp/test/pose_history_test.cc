// Copyright 2026 Zachary Olkin. All rights reserved.

// The pose history that matches the camera pose to the depth image's capture time.
//
// Every failure mode here produces a plausible cloud in the WRONG PLACE -- a pose from the
// wrong moment looks exactly like a pose from the right one, so none of this is visible in
// the output. That is why the ring wraparound, the clamp reporting and the degenerate-sample
// drop are each pinned rather than trusted.

#include <gtest/gtest.h>

#include <vector>

#include "g1_deploy/pose_history.h"

namespace g1_deploy {
namespace {

const QuatXyzw kUpright(0.0, 0.0, 0.0, 1.0);

/// A sample whose joint_pos encodes `k`, so a lookup's answer identifies which one it found.
void PushMarked(PoseHistory* h, double stamp, double k, const QuatXyzw& q = kUpright) {
  h->Push(stamp, std::vector<double>{k, k, k}, q);
}

TEST(PoseHistory, FindsTheNearestSampleInTime) {
  PoseHistory h(16);
  for (int i = 0; i < 10; ++i) PushMarked(&h, 100.0 + 0.001 * i, static_cast<double>(i));
  bool clamped = true;
  // 100.0064 sits between samples 6 and 7, nearer 6.
  const PoseHistory::Sample* s = h.Nearest(100.0064, &clamped);
  ASSERT_NE(s, nullptr);
  EXPECT_FALSE(clamped);
  EXPECT_DOUBLE_EQ(s->joint_pos[0], 6.0);
  EXPECT_DOUBLE_EQ(h.Nearest(100.0066, &clamped)->joint_pos[0], 7.0);
}

TEST(PoseHistory, KeepsWorkingAfterTheRingWraps) {
  // The case a hand-rolled ring gets wrong: more pushes than capacity, so head_ is not 0 and
  // the newest entries sit BEFORE the oldest in memory.
  PoseHistory h(8);
  for (int i = 0; i < 30; ++i) PushMarked(&h, 200.0 + 0.001 * i, static_cast<double>(i));
  EXPECT_EQ(h.size(), 8u);
  bool clamped = true;
  // Only 22..29 survive. Ask for 27 and 22 (the two ends) and one in the middle.
  EXPECT_DOUBLE_EQ(h.Nearest(200.027, &clamped)->joint_pos[0], 27.0);
  EXPECT_FALSE(clamped);
  EXPECT_DOUBLE_EQ(h.Nearest(200.022, &clamped)->joint_pos[0], 22.0);
  EXPECT_FALSE(clamped);
  EXPECT_DOUBLE_EQ(h.Nearest(200.0256, &clamped)->joint_pos[0], 26.0);
}

TEST(PoseHistory, AnExactTieResolvesToTheOlderSample) {
  // Not a preference, just a guarantee: an exactly-midway stamp must land somewhere
  // deterministic. `Nearest` keeps the first strictly-better candidate and scans oldest-first,
  // so ties go to the older. Half a sample period either way is physically irrelevant; a
  // lookup that answered differently run to run would not be.
  PoseHistory h(8);
  PushMarked(&h, 900.000, 0.0);
  PushMarked(&h, 900.002, 1.0);
  bool clamped = true;
  EXPECT_DOUBLE_EQ(h.Nearest(900.001, &clamped)->joint_pos[0], 0.0);
  EXPECT_FALSE(clamped);
}

TEST(PoseHistory, SaysWhenItRanOffEitherEnd) {
  PoseHistory h(8);
  for (int i = 0; i < 30; ++i) PushMarked(&h, 300.0 + 0.001 * i, static_cast<double>(i));
  bool clamped = false;
  // Older than anything retained: the answer is the oldest end, and that must be REPORTED --
  // silently returning it is how a stale frame gets a pose it was never taken with.
  const PoseHistory::Sample* old = h.Nearest(300.000, &clamped);
  EXPECT_TRUE(clamped);
  EXPECT_DOUBLE_EQ(old->joint_pos[0], 22.0);
  // Newer than anything retained (a stamp from the future -- clock skew).
  clamped = false;
  EXPECT_DOUBLE_EQ(h.Nearest(999.0, &clamped)->joint_pos[0], 29.0);
  EXPECT_TRUE(clamped);
}

TEST(PoseHistory, AnEmptyHistoryIsClampedAndNull) {
  PoseHistory h(4);
  bool clamped = false;
  EXPECT_EQ(h.Nearest(1.0, &clamped), nullptr);
  EXPECT_TRUE(clamped) << "an empty history has no match, and must not claim one";
}

TEST(PoseHistory, DegenerateQuaternionsNeverEnter) {
  // The all-zero estimator fault. If such a sample were stored, a lookup could land on it and
  // unproject with an attitude that was never measured -- so it is dropped, and the nearest
  // neighbour becomes the nearest sample that was real.
  PoseHistory h(8);
  PushMarked(&h, 400.000, 0.0);
  PushMarked(&h, 400.001, 1.0, QuatXyzw::Zero());
  PushMarked(&h, 400.002, 2.0);
  EXPECT_EQ(h.size(), 2u);
  bool clamped = false;
  // Asking exactly at the dropped sample's time resolves to a real neighbour, not to it.
  const PoseHistory::Sample* s = h.Nearest(400.001, &clamped);
  ASSERT_NE(s, nullptr);
  EXPECT_NE(s->joint_pos[0], 1.0);
  EXPECT_GT(s->quat.norm(), 0.5);
}

TEST(PoseHistory, StoresANormalisedOrientation) {
  PoseHistory h(4);
  h.Push(500.0, {0.0}, QuatXyzw(0.0, 0.6, 0.0, 0.6));   // norm 0.8485, usable but not unit
  bool clamped = false;
  EXPECT_NEAR(h.Nearest(500.0, &clamped)->quat.norm(), 1.0, 1e-15);
}

TEST(PoseHistory, ClearForgetsTheSamplesButNotTheCapacity) {
  PoseHistory h(8);
  for (int i = 0; i < 5; ++i) PushMarked(&h, 600.0 + 0.001 * i, static_cast<double>(i));
  h.Clear();
  EXPECT_TRUE(h.empty());
  EXPECT_EQ(h.capacity(), 8u);
  bool clamped = false;
  EXPECT_EQ(h.Nearest(600.0, &clamped), nullptr);
  PushMarked(&h, 700.0, 42.0);
  EXPECT_DOUBLE_EQ(h.Nearest(700.0, &clamped)->joint_pos[0], 42.0);
  EXPECT_FALSE(clamped);
}

TEST(PoseHistory, ACapacityOfOneStillWorks) {
  // Degenerate sizing must not divide by zero or wrap wrongly.
  PoseHistory h(1);
  PushMarked(&h, 800.0, 1.0);
  PushMarked(&h, 800.001, 2.0);
  EXPECT_EQ(h.size(), 1u);
  bool clamped = false;
  EXPECT_DOUBLE_EQ(h.Nearest(800.001, &clamped)->joint_pos[0], 2.0);
}

}  // namespace
}  // namespace g1_deploy
