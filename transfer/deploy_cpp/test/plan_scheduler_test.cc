// Copyright 2026 Zachary Olkin. All rights reserved.

// The receding-horizon schedule. No ONNX, no threads, no ROS -- the decision logic alone, so
// every property here is deterministic.
//
// The configuration used throughout is the LIVE artifact's: K=12, F=62, max_offset=50, which
// has exactly zero horizon slack (62 == 50 + 12). Testing at the real numbers matters here:
// a scheduler that works at F=100 and fails at F=62 would look correct in a synthetic test.

#include <gtest/gtest.h>

#include <vector>

#include "g1_deploy/plan_scheduler.h"

namespace g1_deploy {
namespace {

constexpr int kK = 12;
constexpr int kF = 62;
constexpr int kMaxOffset = 50;

ScheduleConfig LiveConfig(int lead = 3) {
  ScheduleConfig cfg;
  cfg.replan_period = kK;
  cfg.horizon = kF;
  cfg.max_offset = kMaxOffset;
  // Defaults to 3, which is NOT the shipped lead (one tick) but the more general case: a
  // multi-tick lead is still legal, and every property here is lead-independent. The shipped
  // configuration gets its own test below.
  cfg.lead_ticks = lead;
  return cfg;
}

/// Drive `ticks` control steps, always delivering the requested plan on time.
///
/// Mirrors what the FM worker does when it meets its deadline: a request issued at tick
/// `t` for target `t + lead` is offered from tick `t + lead` onwards.
struct Harness {
  explicit Harness(const ScheduleConfig& cfg) : sched(cfg) {}

  void Tick(bool deliver = true) {
    sched.BeginTick();
    if (sched.ShouldRequest()) pending.push_back(sched.RequestTargetStep());
    if (deliver && !pending.empty()) {
      // Offer the oldest outstanding plan; TryAdopt refuses it until its boundary.
      if (sched.TryAdopt(pending.front())) pending.erase(pending.begin());
    }
    outcomes.push_back(sched.ResolveTick());
  }

  PlanScheduler sched;
  std::vector<int64_t> pending;
  std::vector<TickOutcome> outcomes;
};

TEST(PlanScheduler, RejectsAHorizonTooShortForTheCadence) {
  ScheduleConfig cfg = LiveConfig();
  cfg.replan_period = 13;  // 50 + 13 = 63 > 62
  EXPECT_THROW(PlanScheduler{cfg}, std::runtime_error);
  cfg.replan_period = 12;  // exactly fits
  EXPECT_NO_THROW(PlanScheduler{cfg});
}

TEST(PlanScheduler, RejectsALeadThatOverlapsThePreviousPlan) {
  ScheduleConfig cfg = LiveConfig();
  cfg.lead_ticks = kK;
  EXPECT_THROW(PlanScheduler{cfg}, std::runtime_error);
  cfg.lead_ticks = kK - 1;
  EXPECT_NO_THROW(PlanScheduler{cfg});
}

TEST(PlanScheduler, FirstTickRequestsImmediatelyRatherThanWaitingForABoundary) {
  // Arming must not need a warm-up: the generator is sampled at once, with the learned
  // null-history embedding covering the empty window.
  Harness h(LiveConfig());
  h.sched.BeginTick();
  EXPECT_EQ(h.sched.ctrl_step(), 0);
  EXPECT_TRUE(h.sched.ShouldRequest());
  EXPECT_EQ(h.sched.RequestTargetStep(), 0) << "the first plan is wanted NOW, not at a boundary";
}

TEST(PlanScheduler, CtrlStepStartsAtMinusOneSoTheFirstTickIsZero) {
  // As an unsigned type the initial -1 would be UINT64_MAX, and UINT64_MAX % 12 == 3, which
  // shifts the entire cadence phase silently.
  PlanScheduler s(LiveConfig());
  EXPECT_EQ(s.ctrl_step(), -1);
  s.BeginTick();
  EXPECT_EQ(s.ctrl_step(), 0);
  EXPECT_EQ(s.ctrl_step() % kK, 0);
}

TEST(PlanScheduler, RequestsLeadTicksBeforeEachBoundaryAndAdoptsOnIt) {
  const int lead = 3;
  Harness h(LiveConfig(lead));
  std::vector<int64_t> adopt_steps;
  for (int i = 0; i < 5 * kK; ++i) {
    h.Tick();
    if (h.outcomes.back() == TickOutcome::kAdoptedOnTime) adopt_steps.push_back(h.sched.ctrl_step());
  }
  // Adoption lands on the cadence boundaries: tick 0 (the forced first plan) then 12, 24, ...
  ASSERT_GE(adopt_steps.size(), 5u);
  EXPECT_EQ(adopt_steps[0], 0);
  for (size_t i = 1; i < adopt_steps.size(); ++i) {
    EXPECT_EQ(adopt_steps[i] % kK, 0) << "adoption " << i << " is off-cadence";
  }
  EXPECT_EQ(h.sched.late_events(), 0u) << "an on-time generator must never be late";
  EXPECT_EQ(h.sched.offsets_clamped(), 0u);
}

TEST(PlanScheduler, CursorWalksZeroToKMinusOneBetweenAdoptions) {
  Harness h(LiveConfig());
  std::vector<int> cursors;
  for (int i = 0; i < 3 * kK; ++i) {
    h.Tick();
    cursors.push_back(h.sched.cursor());
  }
  // After the forced first plan at tick 0, the cursor cycles 0..K-1.
  for (int i = 0; i < 3 * kK; ++i) {
    EXPECT_EQ(cursors[static_cast<size_t>(i)], i % kK) << "tick " << i;
  }
}

TEST(PlanScheduler, ServedIndexNeverLeavesTheHorizonWhenOnTime) {
  // The whole point of the F >= max_offset + K rule. At the live numbers the last tick of a
  // segment reads node 11 + 50 = 61 = F-1 -- exactly the final node, and never past it.
  Harness h(LiveConfig());
  for (int i = 0; i < 4 * kK; ++i) {
    h.Tick();
    for (int off : {0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}) {
      const int idx = h.sched.NodeIndex(off);
      EXPECT_GE(idx, 0);
      EXPECT_LT(idx, kF) << "tick " << i << " offset " << off;
    }
  }
  EXPECT_EQ(h.sched.offsets_clamped(), 0u) << "an on-time schedule must never saturate";
}

TEST(PlanScheduler, APlanArrivingEarlyWaitsForItsBoundary) {
  // Adopting early would serve a window built for a later instant -- the lag this design
  // exists to avoid, in the opposite direction.
  PlanScheduler s(LiveConfig());
  s.BeginTick();                       // tick 0
  EXPECT_TRUE(s.TryAdopt(0));          // the forced first plan
  s.ResolveTick();
  s.BeginTick();                       // tick 1
  EXPECT_FALSE(s.TryAdopt(12)) << "a plan for tick 12 must not be adopted at tick 1";
  EXPECT_EQ(s.cursor(), 0);
  s.ResolveTick();
  EXPECT_EQ(s.cursor(), 1) << "the cursor advances into the live plan meanwhile";
}

TEST(PlanScheduler, LateGeneratorAdvancesTheCursorAndSaturatesTheFarOffsets) {
  Harness h(LiveConfig());
  // Run one clean segment, then stop delivering.
  for (int i = 0; i < kK; ++i) h.Tick();
  ASSERT_EQ(h.sched.late_events(), 0u);

  for (int i = 0; i < 6; ++i) h.Tick(/*deliver=*/false);

  EXPECT_TRUE(h.sched.late());
  EXPECT_EQ(h.sched.late_events(), 1u) << "one continuous outage is ONE event";
  EXPECT_GT(h.sched.late_ticks(), 0u);
  EXPECT_GT(h.sched.cursor(), kK - 1) << "time keeps advancing; the cursor does not freeze";

  // The far offsets saturate rather than running off the end; the near ones keep moving.
  const int near_idx = h.sched.NodeIndex(0);
  const int far_idx = h.sched.NodeIndex(kMaxOffset);
  EXPECT_EQ(far_idx, kF - 1) << "the far end must repeat the last node";
  EXPECT_LT(near_idx, kF - 1) << "the near end must still be advancing";
  EXPECT_GT(h.sched.offsets_clamped(), 0u) << "saturation must be counted, not silent";
}

/// The SHIPPED configuration: one tick of lead, and a generator that misses it ~6% of runs.
///
/// `fm_lead_sec = 0.02` buys 40 ms less exteroception staleness in exchange for the occasional
/// late plan, so the degradation that trade actually costs is pinned here rather than left to
/// be discovered on a robot. The claim being pinned is narrow and worth stating exactly: at
/// one tick late the ONLY offset that repeats is the furthest, because F == max_offset + K.
TEST(PlanScheduler, OneTickOfLeadRepeatsOnlyTheFurthestOffsetWhenAPlanSlips) {
  Harness h(LiveConfig(/*lead=*/1));
  for (int i = 0; i < kK; ++i) h.Tick();
  ASSERT_EQ(h.sched.late_events(), 0u);
  ASSERT_EQ(h.sched.offsets_clamped(), 0u);

  // The generator overruns its 20 ms budget, so no plan exists for this boundary.
  h.Tick(/*deliver=*/false);
  ASSERT_EQ(h.outcomes.back(), TickOutcome::kLate);
  EXPECT_EQ(h.sched.cursor(), kK) << "the cursor advances; the reference must not stall";

  // Walk out from the tracked node. Everything up to and including offset max-1 resolves to a
  // real node of the live plan -- checked through the CLAMP COUNTER, since offset max-1 lands
  // exactly on the last node and so is indistinguishable from a clamp by its return value.
  EXPECT_EQ(h.sched.NodeIndex(0), kK) << "the node being tracked is a genuine prediction";
  EXPECT_EQ(h.sched.offsets_clamped(), 0u);
  EXPECT_EQ(h.sched.NodeIndex(kMaxOffset - 1), kF - 1);
  EXPECT_EQ(h.sched.offsets_clamped(), 0u) << "offset max-1 lands ON the last node, unclamped";
  EXPECT_EQ(h.sched.NodeIndex(kMaxOffset), kF - 1) << "the 1 s lookahead repeats";
  EXPECT_EQ(h.sched.offsets_clamped(), 1u) << "exactly one offset, and it is counted";

  // One tick later the plan lands and the cadence self-heals to a time-correct cursor.
  h.Tick();
  EXPECT_EQ(h.outcomes.back(), TickOutcome::kAdoptedLate);
  EXPECT_EQ(h.sched.cursor(), 1);
  EXPECT_FALSE(h.sched.late());
  EXPECT_EQ(h.sched.late_events(), 1u);
}

TEST(PlanScheduler, ALatePlanIsAdoptedOnArrivalAtATimeCorrectCursor) {
  Harness h(LiveConfig());
  for (int i = 0; i < kK; ++i) h.Tick();
  // Miss the boundary at tick 12 by two ticks.
  h.Tick(/*deliver=*/false);
  h.Tick(/*deliver=*/false);
  ASSERT_TRUE(h.sched.late());

  h.Tick();  // the plan for tick 12 finally lands, at tick 14
  EXPECT_EQ(h.outcomes.back(), TickOutcome::kAdoptedLate);
  EXPECT_FALSE(h.sched.late()) << "adopting a plan ends the outage";
  EXPECT_EQ(h.sched.late_events(), 1u) << "the event stays counted after recovery";
  // Its node j is for step target + j, so serving at 14 a plan for 12 means cursor 2 --
  // adopting at cursor 0 would re-introduce exactly the lag this design avoids.
  EXPECT_EQ(h.sched.cursor(), 2);
}

TEST(PlanScheduler, APlanTooStaleToServeIsDroppedRatherThanSaturatingImmediately) {
  PlanScheduler s(LiveConfig());
  s.BeginTick();
  ASSERT_TRUE(s.TryAdopt(0));
  s.ResolveTick();
  // Advance well past a full period without delivery.
  for (int i = 0; i < 20; ++i) {
    s.BeginTick();
    s.ResolveTick();
  }
  // A plan for step 0 offered now would need a cursor of 21, past F - max_offset - 1 = 11.
  EXPECT_FALSE(s.TryAdopt(0));
}

TEST(PlanScheduler, ResetClearsTheScheduleButNotTheHistoryOfEvents) {
  Harness h(LiveConfig());
  for (int i = 0; i < kK; ++i) h.Tick();
  for (int i = 0; i < 4; ++i) h.Tick(/*deliver=*/false);
  ASSERT_GT(h.sched.late_events(), 0u);

  h.sched.Reset();
  EXPECT_EQ(h.sched.ctrl_step(), -1);
  EXPECT_FALSE(h.sched.has_plan());
  EXPECT_FALSE(h.sched.late());
  EXPECT_EQ(h.sched.cursor(), 0);
  // After a reset the next tick must force a fresh plan rather than serve a stale one.
  h.sched.BeginTick();
  EXPECT_TRUE(h.sched.ShouldRequest());
}

TEST(PlanScheduler, PlanIdCountsAdoptionsAndSurvivesReset) {
  Harness h(LiveConfig());
  EXPECT_EQ(h.sched.plan_id(), 0u) << "no plan has been adopted yet";

  for (int i = 0; i < kK; ++i) h.Tick();
  const uint64_t after_first = h.sched.plan_id();
  EXPECT_GT(after_first, 0u);

  // A plan offered before its boundary is refused, and a refusal must not consume an id --
  // otherwise the id would count OFFERS, and a bag could not join a tick to the plan that
  // actually drove it.
  h.sched.BeginTick();
  EXPECT_FALSE(h.sched.TryAdopt(h.sched.ctrl_step() + 5));
  EXPECT_EQ(h.sched.plan_id(), after_first);

  // Likewise a plan so stale it would saturate from its first tick.
  EXPECT_FALSE(h.sched.TryAdopt(h.sched.ctrl_step() - (kF - kMaxOffset)));
  EXPECT_EQ(h.sched.plan_id(), after_first);

  // Exactly one increment per successful adoption.
  EXPECT_TRUE(h.sched.TryAdopt(h.sched.ctrl_step()));
  EXPECT_EQ(h.sched.plan_id(), after_first + 1);

  // Monotonic across a re-arm: ids stay unique for the life of the process, so a bag can tell
  // a re-arm from a re-plan. This is the property plan_scheduler.h's Reset() doc promises.
  const uint64_t before_reset = h.sched.plan_id();
  h.sched.Reset();
  EXPECT_EQ(h.sched.plan_id(), before_reset);
  h.sched.BeginTick();
  EXPECT_TRUE(h.sched.TryAdopt(h.sched.ctrl_step()));
  EXPECT_EQ(h.sched.plan_id(), before_reset + 1);
}

TEST(PlanScheduler, LatenessIsAboutTheScheduleNotTheHorizon) {
  // With a longer-horizon generator, being late and saturating stop coinciding. Lateness must
  // still be reported at cursor >= K, because that is when the reference stopped being
  // refreshed -- even though the window is still being served from real nodes.
  ScheduleConfig cfg = LiveConfig();
  cfg.horizon = 80;  // the future retrain: max_offset + K + slack
  Harness h(cfg);
  for (int i = 0; i < kK; ++i) h.Tick();
  for (int i = 0; i < 5; ++i) h.Tick(/*deliver=*/false);

  EXPECT_TRUE(h.sched.late()) << "late is cursor >= K, independent of the horizon";
  EXPECT_EQ(h.sched.offsets_clamped(), 0u)
      << "with slack in the horizon a late plan must NOT saturate";
  EXPECT_LT(h.sched.NodeIndex(kMaxOffset), cfg.horizon - 1);
}

}  // namespace
}  // namespace g1_deploy
