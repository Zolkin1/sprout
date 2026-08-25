// Copyright 2026 Zachary Olkin. All rights reserved.

// The receding-horizon algorithm, ported property-for-property from
// transfer/deploy/tests/test_replan_cadence.py.
//
// Same technique as that file's _FakePolicy: the generator is a deterministic stand-in and
// everything on the serving path -- cursor, cadence, anchor, stash, re-anchor -- is real. A
// plan's node n of plan p is p + n/1000, so both the plan and the node are recoverable from
// any served value, which is what lets a test tell "node 3 of plan 2" from "node 0 of plan 3".
//
// Where the C++ deliberately differs from the python, the test says so and pins the NEW
// behaviour: the generator is asked early and answers asynchronously, and a late plan
// saturates the far offsets instead of raising.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "g1_deploy/gen_tracker_core.h"
#include "g1_deploy/quat_math.h"

namespace g1_deploy {
namespace {

constexpr int kNJ = 29;
constexpr int kEE = 0;
constexpr int kF = 62;
constexpr int kN = 4;
constexpr int kMaxOffset = 50;

const std::vector<ReferenceBlock> kSlim = {ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel,
                                           ReferenceBlock::kJointPos};
const std::vector<int> kOffsets = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

/// A generator stand-in: plan p, node n -> p + n/1000, delivered after `delay` ticks.
class FakeSampler : public FmSampler {
 public:
  FakeSampler(int node_dim, int horizon, int delay_ticks)
      : node_dim_(node_dim), horizon_(horizon), delay_(delay_ticks) {}

  void Submit(const FmRequest& req) override {
    ++plan_count_;
    requests.push_back(req);
    pending_.push_back({req.target_ctrl_step, plan_count_, delay_});
  }

  bool TryCollect(double* pred_out, int64_t* target) override {
    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
      if (it->ticks_left > 0) {
        --it->ticks_left;
        continue;
      }
      for (int n = 0; n < horizon_; ++n) {
        const double v = static_cast<double>(it->plan_id) + n / 1000.0;
        for (int d = 0; d < node_dim_; ++d) {
          pred_out[static_cast<size_t>(n) * node_dim_ + d] = v;
        }
      }
      *target = it->target;
      pending_.erase(it);
      return true;
    }
    return false;
  }

  /// Set the delay applied to plans submitted from now on.
  void set_delay(int ticks) { delay_ = ticks; }
  int plan_count() const { return plan_count_; }

  std::vector<FmRequest> requests;

 private:
  struct Pending {
    int64_t target;
    int plan_id;
    int ticks_left;
  };
  int node_dim_;
  int horizon_;
  int delay_;
  int plan_count_ = 0;
  std::vector<Pending> pending_;
};

CoreConfig MakeConfig(int k, int lead, WindowAnchor anchor = WindowAnchor::kPlan) {
  CoreConfig cfg;
  cfg.window_anchor = anchor;
  cfg.num_joints = kNJ;
  cfg.num_ee_frames = kEE;
  cfg.blocks = kSlim;
  cfg.offsets = kOffsets;
  cfg.history_length = kN;
  cfg.cmd_dim = 3;
  cfg.schedule.replan_period = k;
  cfg.schedule.horizon = kF;
  cfg.schedule.max_offset = kMaxOffset;
  cfg.schedule.lead_ticks = lead;
  return cfg;
}

CoreState MakeState(const Vec3& pos = Vec3::Zero(),
                    const QuatXyzw& quat = QuatXyzw(0.0, 0.0, 0.0, 1.0)) {
  CoreState s;
  s.root_pos = pos;
  s.root_quat = quat;
  s.command = {0.5, 0.0, 0.0};
  return s;
}

/// Recover (plan, node) from a served value. joint_pos is copied verbatim by the re-anchor,
/// so its first element still carries the tag even after the frame correction.
void DecodeTag(double v, int* plan, int* node) {
  *plan = static_cast<int>(std::floor(v + 1e-9));
  *node = static_cast<int>(std::llround((v - *plan) * 1000.0));
}

struct Harness {
  explicit Harness(int k, int lead = 3, int delay = 0,
                   WindowAnchor anchor = WindowAnchor::kPlan)
      : cfg(MakeConfig(k, lead, anchor)),
        sampler(std::make_shared<FakeSampler>(
            WindowDim(kNJ, kEE, kSlim), kF, delay)),
        core(cfg, sampler),
        window(static_cast<size_t>(kOffsets.size()) *
               static_cast<size_t>(WindowDim(kNJ, kEE, kSlim))) {}

  TickOutcome Step(const CoreState& s) { return core.Step(s, window.data()); }

  /// The joint_pos column of served row `r` -- untouched by the re-anchor, so it still
  /// carries the plan/node tag.
  double Tag(int r) const {
    const int dim = WindowDim(kNJ, kEE, kSlim);
    const auto slices = WindowBlockSlices(kNJ, kEE, kSlim);
    int joint_off = 0;
    for (const auto& s : slices) {
      if (s.block == ReferenceBlock::kJointPos) joint_off = s.offset;
    }
    return window[static_cast<size_t>(r) * dim + joint_off];
  }

  CoreConfig cfg;
  std::shared_ptr<FakeSampler> sampler;
  GenTrackerCore core;
  std::vector<double> window;
};

// ---------------------------------------------------------------------------------------
// Cadence and cursor
// ---------------------------------------------------------------------------------------

TEST(GenTrackerCore, ReplansExactlyEveryKSteps) {
  for (int k : {2, 4, 12}) {
    Harness h(k, /*lead=*/1);
    std::vector<int64_t> adoptions;
    for (int i = 0; i < 6 * k; ++i) {
      if (h.Step(MakeState()) == TickOutcome::kAdoptedOnTime) {
        adoptions.push_back(h.core.scheduler().ctrl_step());
      }
    }
    ASSERT_GE(adoptions.size(), 5u) << "k=" << k;
    for (int64_t s : adoptions) EXPECT_EQ(s % k, 0) << "k=" << k << " adoption at " << s;
  }
}

TEST(GenTrackerCore, ServedNodesComeFromTheCursorNotTheWindowStart) {
  // At K>1 the served window is cursor + offsets. Reading from 0 would serve a plan that is
  // up to K-1 steps stale in time while looking perfectly well-formed.
  Harness h(12, /*lead=*/1);
  for (int i = 0; i < 5; ++i) h.Step(MakeState());
  // 5 ticks in: the first plan was adopted at tick 0, so the cursor is 4.
  ASSERT_EQ(h.core.scheduler().cursor(), 4);
  for (size_t r = 0; r < kOffsets.size(); ++r) {
    int plan = 0, node = 0;
    DecodeTag(h.Tag(static_cast<int>(r)), &plan, &node);
    EXPECT_EQ(node, 4 + kOffsets[r]) << "offset " << kOffsets[r] << " served the wrong node";
  }
}

TEST(GenTrackerCore, HistoryRowIsTheNodeServedLastStep) {
  // The generator's context is its OWN outputs, one per step, raw and in the plan's frame.
  Harness h(12, /*lead=*/1);
  h.Step(MakeState());
  ASSERT_TRUE(h.core.has_plan());
  int plan = 0, node = 0;
  DecodeTag(h.core.prev_served_node()[0], &plan, &node);
  EXPECT_EQ(node, 0) << "the first serve is node 0 of the fresh plan";

  h.Step(MakeState());
  DecodeTag(h.core.prev_served_node()[0], &plan, &node);
  EXPECT_EQ(node, 1) << "and the stash follows the cursor";
}

TEST(GenTrackerCore, TheStashIsRawNotReAnchored) {
  // The row pushed into the history must be the node in its PLAN's frame. Re-anchoring it
  // would feed odometry into the generator's own conditioning, where it compounds silently.
  Harness h(12, /*lead=*/1);
  h.Step(MakeState());  // adopt at the origin
  const std::vector<double> after_first = h.core.prev_served_node();

  // Move the robot a long way and step again. The stash must track the cursor, NOT the pose.
  Harness moved(12, /*lead=*/1);
  moved.Step(MakeState());
  moved.Step(MakeState(Vec3(5.0, -3.0, 0.0), QuatFromAxisAngle(Vec3(0, 0, 1.1))));
  int plan = 0, node = 0;
  DecodeTag(moved.core.prev_served_node()[0], &plan, &node);
  EXPECT_EQ(node, 1);
  // Every element of the stashed node is the plain tag: no frame correction has touched it.
  for (double v : moved.core.prev_served_node()) EXPECT_DOUBLE_EQ(v, plan + node / 1000.0);
  EXPECT_EQ(after_first.size(), moved.core.prev_served_node().size());
}

// ---------------------------------------------------------------------------------------
// Anchoring
// ---------------------------------------------------------------------------------------

TEST(GenTrackerCore, AStaleAnchorIsCorrectedWhenTheRobotHasMoved) {
  // The core of K>1 UNDER kRobotNow: serving must not shear the window by the distance
  // travelled since the plan was adopted. joint_pos is anchor-invariant and must be untouched;
  // base_pose is not and must change.
  //
  // kRobotNow is asked for explicitly because it is no longer what a CoreConfig defaults to,
  // and PolicyRuntime::Load refuses to build it at all -- that arm needs a measured base
  // position, i.e. odometry, which this deploy path does not have. The mechanism is kept and
  // kept under test because it is ReanchorWindow's only integration test, and because the
  // refusal is a policy decision that could be revisited without the maths having rotted.
  Harness h(12, /*lead=*/1, /*delay=*/0, WindowAnchor::kRobotNow);
  h.Step(MakeState());  // adopt at the origin, identity orientation
  const auto slices = WindowBlockSlices(kNJ, kEE, kSlim);
  const int dim = WindowDim(kNJ, kEE, kSlim);
  const std::vector<double> at_origin(h.window.begin(), h.window.begin() + dim);

  h.Step(MakeState(Vec3(1.0, 2.0, 0.0), QuatFromAxisAngle(Vec3(0.0, 0.0, 0.6))));

  int base_off = 0, joint_off = 0;
  for (const auto& s : slices) {
    if (s.block == ReferenceBlock::kBasePose) base_off = s.offset;
    if (s.block == ReferenceBlock::kJointPos) joint_off = s.offset;
  }
  // base_pose must have been corrected...
  double base_delta = 0.0;
  for (int i = 0; i < 9; ++i) {
    base_delta = std::max(base_delta, std::abs(h.window[base_off + i] - at_origin[base_off + i]));
  }
  EXPECT_GT(base_delta, 0.5) << "a moved robot must change base_pose -- the anchor is stale";
  // ...while joint_pos, which does not reference the anchor, must only reflect the cursor.
  int plan = 0, node = 0;
  DecodeTag(h.window[joint_off], &plan, &node);
  EXPECT_EQ(node, 1);
}

TEST(GenTrackerCore, ThePlanArmServesThePredictionUntouchedEvenAfterTheRobotMoves) {
  // The whole point of window_anchor="plan": the generator already predicts in the frame of
  // the root at the re-plan, which IS the frame the observation is defined against, so the
  // prediction is the observation and is served exactly as it was made.
  //
  // The second Step moves the robot 5 m and turns it, i.e. the state that would make a
  // re-anchor rewrite base_pose completely. Every element must still be the plain plan/node
  // tag: nothing derived from state.root_pos may reach the window. That is the negative
  // control, and it is the one that matters -- rtr's own commit records that a test double
  // which returned the measured root made this arm look inert and pass.
  Harness h(12, /*lead=*/1);
  h.Step(MakeState());  // adopt at the origin
  h.Step(MakeState(Vec3(5.0, -3.0, 0.7), QuatFromAxisAngle(Vec3(0.0, 0.0, 1.1))));

  const int dim = WindowDim(kNJ, kEE, kSlim);
  for (size_t r = 0; r < kOffsets.size(); ++r) {
    // Plan 1, node = cursor(1) + offset. FakeSampler tags every element identically.
    const double want = 1.0 + (1 + kOffsets[r]) / 1000.0;
    for (int d = 0; d < dim; ++d) {
      EXPECT_DOUBLE_EQ(h.window[r * dim + d], want)
          << "row " << r << " element " << d
          << ": the plan arm must serve the prediction byte-identically, and must not read "
             "the measured base pose at all";
    }
  }
}

TEST(GenTrackerCore, TheTwoArmsDisagreeOnceTheRobotHasMoved) {
  // Same plan, same states, one field different -- and the served windows differ. This is
  // what makes a mislabelled artifact dangerous: the two arms have identical widths and
  // identical blocks, so nothing downstream can tell them apart, which is why the convention
  // is checked at load instead of inferred.
  Harness plan_arm(12, /*lead=*/1);
  Harness now_arm(12, /*lead=*/1, /*delay=*/0, WindowAnchor::kRobotNow);
  const CoreState adopt = MakeState();
  const CoreState moved = MakeState(Vec3(1.0, 2.0, 0.0), QuatFromAxisAngle(Vec3(0.0, 0.0, 0.6)));
  for (Harness* h : {&plan_arm, &now_arm}) {
    h->Step(adopt);
    h->Step(moved);
  }
  const auto slices = WindowBlockSlices(kNJ, kEE, kSlim);
  int base_off = 0;
  for (const auto& s : slices) {
    if (s.block == ReferenceBlock::kBasePose) base_off = s.offset;
  }
  double delta = 0.0;
  for (int i = 0; i < 9; ++i) {
    delta = std::max(delta, std::abs(plan_arm.window[base_off + i] - now_arm.window[base_off + i]));
  }
  EXPECT_GT(delta, 0.5) << "the two arms produced the same base_pose for a robot that moved; "
                           "one of them is not doing what it says";
}

TEST(GenTrackerCore, ReanchorOnTheAdoptionTickIsAnIdentity) {
  // The anchor is captured from the measured base at adoption, so on that very tick the
  // re-anchor has nothing to correct and must not perturb the window at all.
  Harness h(12, /*lead=*/1);
  const CoreState s = MakeState(Vec3(2.0, -1.0, 0.4), QuatFromAxisAngle(Vec3(0.0, 0.0, 0.9)));
  h.Step(s);
  const int dim = WindowDim(kNJ, kEE, kSlim);
  for (size_t r = 0; r < kOffsets.size(); ++r) {
    const double want = 1.0 + kOffsets[r] / 1000.0;  // plan 1, node = offset
    for (int d = 0; d < dim; ++d) {
      EXPECT_NEAR(h.window[r * dim + d], want, 2e-15)
          << "row " << r << " element " << d << " was perturbed on the adoption tick";
    }
  }
}

// ---------------------------------------------------------------------------------------
// The async schedule: what differs from the python, on purpose
// ---------------------------------------------------------------------------------------

TEST(GenTrackerCore, TheProjectedHistoryIsWhatTheSynchronousRuntimeWouldHaveBuilt) {
  // The assertion the early request rests on. The request issued lead ticks before a
  // boundary must carry the history rows for the ticks up TO that boundary -- which are the
  // nodes this runtime will serve meanwhile, and are already determined by the live plan.
  const int lead = 3;
  Harness h(12, lead);
  for (int i = 0; i < 12; ++i) h.Step(MakeState());

  // The request issued at tick 9 targets tick 12. Its newest history row must be the node
  // served at tick 11 -- cursor 11 of plan 1 -- not the node served at tick 9.
  const FmRequest* req = nullptr;
  for (const auto& r : h.sampler->requests) {
    if (r.target_ctrl_step == 12) req = &r;
  }
  ASSERT_NE(req, nullptr) << "no request targeted the boundary at tick 12";
  EXPECT_EQ(req->request_ctrl_step, 12 - lead);

  const int dim = WindowDim(kNJ, kEE, kSlim);
  int plan = 0, node = 0;
  DecodeTag(req->history_nodes[static_cast<size_t>(kN - 1) * dim], &plan, &node);
  EXPECT_EQ(node, 11) << "the newest projected row must be the node served at tick 11";
  DecodeTag(req->history_nodes[static_cast<size_t>(kN - 2) * dim], &plan, &node);
  EXPECT_EQ(node, 10);
}

TEST(GenTrackerCore, TheFirstPlanIsRequestedWithHistoryNullSet) {
  // Arming runs the generator from step 0 with the learned null-history embedding rather
  // than a warm-up or a fabricated static window -- a repeated snapshot is a history that
  // never occurs in training, since real history always has motion between its steps.
  Harness h(12, /*lead=*/3);
  h.Step(MakeState());
  ASSERT_FALSE(h.sampler->requests.empty());
  EXPECT_TRUE(h.sampler->requests.front().history_null);
  EXPECT_EQ(h.sampler->requests.front().target_ctrl_step, 0)
      << "the first plan is wanted immediately, not at the next boundary";
}

TEST(GenTrackerCore, HistoryNullHoldsUntilTheWindowIsFullOfRealServedNodes) {
  // The generator is NEVER handed a zero placeholder dressed up as real history. Training
  // never produced that input, so it is off-distribution, and it would arrive with
  // `history_null` OFF -- silently, for exactly the steps the learned embedding exists to
  // cover.
  //
  //   tick : history_null : oldest row is the zero placeholder
  //    0..3 :      1      :   1        placeholder present -> embedding covers it
  //    4+   :      0      :   0        N genuinely served nodes, and only then
  //
  // The window goes valid at tick N, not N-1: the first append happens before anything has
  // been served, so it is a placeholder and must not count toward validity. Verified to match
  // transfer/deploy/gen_tracker.py, which got the same fix -- the two have to agree or sim2sim
  // and hardware condition the generator differently for the first N steps after arming.
  const int lead = 0;
  Harness h(1, lead);  // K=1 so a request fires every tick
  for (int t = 0; t < 6; ++t) h.Step(MakeState());

  ASSERT_GE(h.sampler->requests.size(), 6u);
  const std::vector<bool> want_null = {true, true, true, true, false, false};
  for (size_t t = 0; t < want_null.size(); ++t) {
    EXPECT_EQ(h.sampler->requests[t].history_null, want_null[t]) << "tick " << t;
  }

  // And the row that made it valid is a real served node, not the placeholder.
  const int dim = WindowDim(kNJ, kEE, kSlim);
  const auto& oldest = h.sampler->requests[4].history_nodes;
  double mag = 0.0;
  for (int d = 0; d < dim; ++d) mag = std::max(mag, std::abs(oldest[static_cast<size_t>(d)]));
  EXPECT_GT(mag, 0.0) << "the oldest row must be a served node once the window is valid";
}

TEST(GenTrackerCore, ALateGeneratorSaturatesTheFarOffsetsInsteadOfRaising) {
  // The python raises when the cursor runs past the horizon, because there a clamp could
  // only mean a misconfigured horizon. Here lateness is a real, expected, flagged condition,
  // so the far end repeats its last node while the near end keeps advancing.
  Harness h(12, /*lead=*/1);
  for (int i = 0; i < 12; ++i) h.Step(MakeState());
  h.sampler->set_delay(100);  // the generator stops answering, from the NEXT request on

  // The plan for tick 12 was already submitted at tick 11, so it still lands. The outage
  // only bites at the following boundary (tick 24), and the cursor must pass K before
  // lateness is declared -- so run well past it rather than assuming.
  for (int i = 0; i < 20; ++i) h.Step(MakeState());
  EXPECT_TRUE(h.core.scheduler().late());
  EXPECT_GT(h.core.scheduler().offsets_clamped(), 0u);

  int plan = 0, near_node = 0, far_node = 0;
  DecodeTag(h.Tag(0), &plan, &near_node);
  DecodeTag(h.Tag(static_cast<int>(kOffsets.size()) - 1), &plan, &far_node);
  EXPECT_EQ(far_node, kF - 1) << "the far offset must repeat the last node";
  EXPECT_LT(near_node, kF - 1) << "the near offset must still be advancing";
}

TEST(GenTrackerCore, ResetClearsTheAnchorAndTheStash) {
  // A stash from before a reset describes another episode's pose; keeping it would push a
  // stale row straight into the generator's conditioning on the first step after arming.
  Harness h(12, /*lead=*/1);
  for (int i = 0; i < 5; ++i) {
    h.Step(MakeState(Vec3(1.0, 1.0, 0.0), QuatFromAxisAngle(Vec3(0.0, 0.0, 0.4))));
  }
  ASSERT_TRUE(h.core.has_plan());
  ASSERT_NE(h.core.prev_served_node()[0], 0.0);

  h.core.Reset();
  EXPECT_FALSE(h.core.has_plan());
  EXPECT_EQ(h.core.anchor_pos(), Vec3::Zero());
  for (double v : h.core.prev_served_node()) EXPECT_DOUBLE_EQ(v, 0.0);
  EXPECT_EQ(h.core.scheduler().ctrl_step(), -1);
}

TEST(GenTrackerCore, KEqualsOneServesNodeZeroEveryTick) {
  // At K=1 the anchor IS the current pose every tick, so the re-anchor is an identity and
  // the cursor never leaves 0.
  Harness h(1, /*lead=*/0);
  for (int i = 0; i < 6; ++i) {
    h.Step(MakeState(Vec3(0.1 * i, 0.0, 0.0)));
    EXPECT_EQ(h.core.scheduler().cursor(), 0);
    int plan = 0, node = 0;
    DecodeTag(h.Tag(0), &plan, &node);
    EXPECT_EQ(node, 0) << "tick " << i;
    EXPECT_EQ(plan, i + 1) << "a fresh plan every tick at K=1";
  }
}

}  // namespace
}  // namespace g1_deploy
