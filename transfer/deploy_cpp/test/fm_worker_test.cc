// Copyright 2026 Zachary Olkin. All rights reserved.

// The generator's thread. The highest-value assertion here is the LAST one: with the worker
// meeting its deadline, the threaded run must produce output bit-identical to the same core
// driven synchronously. That separates "is the schedule right" from "is the threading right",
// so a failure in either is attributable.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "g1_deploy/fm_worker.h"
#include "g1_deploy/gen_tracker_core.h"

namespace g1_deploy {
namespace {

constexpr int kNJ = 29;
constexpr int kF = 62;
constexpr int kN = 4;
constexpr int kK = 12;
constexpr int kMaxOffset = 50;

const std::vector<ReferenceBlock> kSlim = {ReferenceBlock::kBasePose, ReferenceBlock::kBaseVel,
                                          ReferenceBlock::kJointPos};
const std::vector<int> kOffsets = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};

int NodeDim() { return WindowDim(kNJ, 0, kSlim); }

/// Deterministic stand-in that optionally sleeps, to model inference latency.
///
/// The plan encodes (plan_id, node) as plan + node/1000, and ALSO folds in the request's
/// history so a test can prove the threaded path conditioned on the same thing the
/// synchronous one did.
class SleepySampler : public FmSampler {
 public:
  SleepySampler(int node_dim, int horizon, double sleep_ms)
      : node_dim_(node_dim), horizon_(horizon), sleep_ms_(sleep_ms) {}

  void Submit(const FmRequest& req) override {
    if (sleep_ms_ > 0.0) {
      std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(sleep_ms_));
    }
    ++plan_count_;
    // Fold in the newest history row so a wrong conditioning changes the OUTPUT, not just an
    // invisible input. Without this the parity test below could pass on a worker that
    // conditioned on stale rows.
    const double hist_tag =
        req.history_nodes.empty()
            ? 0.0
            : req.history_nodes[static_cast<size_t>(kN - 1) * static_cast<size_t>(node_dim_)];
    for (int n = 0; n < horizon_; ++n) {
      const double v = plan_count_ + n / 1000.0 + hist_tag / 1e6;
      for (int d = 0; d < node_dim_; ++d) {
        scratch_[static_cast<size_t>(n) * node_dim_ + d] = v;
      }
    }
    target_ = req.target_ctrl_step;
    ready_ = true;
  }

  bool TryCollect(double* pred_out, int64_t* target) override {
    if (!ready_) return false;
    std::copy(scratch_.begin(), scratch_.end(), pred_out);
    *target = target_;
    ready_ = false;
    return true;
  }

  void Init() { scratch_.assign(static_cast<size_t>(horizon_) * node_dim_, 0.0); }
  void set_sleep_ms(double ms) { sleep_ms_ = ms; }

 private:
  int node_dim_;
  int horizon_;
  double sleep_ms_;
  int plan_count_ = 0;
  int64_t target_ = 0;
  bool ready_ = false;
  std::vector<double> scratch_;
};

std::shared_ptr<SleepySampler> MakeSampler(double sleep_ms) {
  auto s = std::make_shared<SleepySampler>(NodeDim(), kF, sleep_ms);
  s->Init();
  return s;
}

CoreConfig MakeConfig(int lead) {
  CoreConfig cfg;
  cfg.num_joints = kNJ;
  cfg.num_ee_frames = 0;
  cfg.blocks = kSlim;
  cfg.offsets = kOffsets;
  cfg.history_length = kN;
  cfg.cmd_dim = 3;
  cfg.schedule.replan_period = kK;
  cfg.schedule.horizon = kF;
  cfg.schedule.max_offset = kMaxOffset;
  cfg.schedule.lead_ticks = lead;
  return cfg;
}

CoreState MakeState(int tick) {
  CoreState s;
  s.root_pos = Vec3(0.01 * tick, 0.0, 0.0);  // walking forward, so the re-anchor does work
  s.root_quat = QuatXyzw(0.0, 0.0, 0.0, 1.0);
  s.command = {0.5, 0.0, 0.0};
  return s;
}

TEST(PlanMailbox, PublishAndConsumeRoundTrip) {
  PlanMailbox box(3, 4);
  double out[4] = {0, 0, 0, 0};
  int64_t target = 0;
  EXPECT_FALSE(box.TryConsume(out, &target)) << "an empty mailbox has nothing to give";

  double* slot = box.AcquireSlot();
  for (int i = 0; i < 4; ++i) slot[i] = i + 1.0;
  box.Publish(42);
  ASSERT_TRUE(box.TryConsume(out, &target));
  EXPECT_EQ(target, 42);
  for (int i = 0; i < 4; ++i) EXPECT_DOUBLE_EQ(out[i], i + 1.0);
  EXPECT_FALSE(box.TryConsume(out, &target)) << "the same plan must not be handed out twice";
}

TEST(PlanMailbox, SupersededPlansAreCountedNotSilent) {
  PlanMailbox box(3, 2);
  for (int p = 1; p <= 3; ++p) {
    double* slot = box.AcquireSlot();
    slot[0] = p;
    slot[1] = p;
    box.Publish(p);
  }
  double out[2];
  int64_t target = 0;
  ASSERT_TRUE(box.TryConsume(out, &target));
  EXPECT_EQ(target, 3) << "the consumer must get the NEWEST plan";
  EXPECT_EQ(box.dropped(), 2u) << "the two superseded plans must be counted";
}

TEST(PlanMailbox, TheProducerNeverReusesTheSlotJustPublished) {
  // Otherwise a publish could overwrite the bytes a consumer is about to copy.
  PlanMailbox box(3, 1);
  double* a = box.AcquireSlot();
  box.Publish(1);
  double* b = box.AcquireSlot();
  EXPECT_NE(a, b);
  box.Publish(2);
  double* c = box.AcquireSlot();
  EXPECT_NE(b, c);
}

TEST(FmWorker, DeliversAPlanAndStopsCleanly) {
  auto inner = MakeSampler(1.0);
  FmWorker worker(inner, static_cast<size_t>(kF) * NodeDim());
  worker.Start();

  FmRequest req;
  req.target_ctrl_step = 5;
  req.history_nodes.assign(static_cast<size_t>(kN) * NodeDim(), 0.0);
  req.history_cmd.assign(static_cast<size_t>(kN) * 3, 0.0);
  worker.Submit(req);

  std::vector<double> plan(static_cast<size_t>(kF) * NodeDim(), 0.0);
  int64_t target = 0;
  ASSERT_TRUE(worker.WaitForPlan(2.0, plan.data(), &target));
  EXPECT_EQ(target, 5);
  EXPECT_EQ(worker.plans_produced(), 1u);
  worker.Stop();
  worker.Stop();  // idempotent
}

TEST(FmWorker, ARequestWhileBusyIsSkippedNotQueued) {
  auto inner = MakeSampler(120.0);  // slow enough that the second request lands mid-flight
  FmWorker worker(inner, static_cast<size_t>(kF) * NodeDim());
  worker.Start();

  FmRequest req;
  req.target_ctrl_step = 1;
  req.history_nodes.assign(static_cast<size_t>(kN) * NodeDim(), 0.0);
  req.history_cmd.assign(static_cast<size_t>(kN) * 3, 0.0);
  worker.Submit(req);
  // Give the worker a moment to pick it up, then pile on.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (int i = 0; i < 5; ++i) {
    req.target_ctrl_step = 2 + i;
    worker.Submit(req);
  }
  EXPECT_GT(worker.requests_skipped(), 0u)
      << "queueing would let the generator fall arbitrarily behind";

  std::vector<double> plan(static_cast<size_t>(kF) * NodeDim(), 0.0);
  int64_t target = 0;
  EXPECT_TRUE(worker.WaitForPlan(2.0, plan.data(), &target));
  EXPECT_EQ(target, 1) << "the in-flight request is the one that completes";
  worker.Stop();
}

TEST(FmWorker, TimesOutRatherThanBlockingForever) {
  auto inner = MakeSampler(0.0);
  FmWorker worker(inner, static_cast<size_t>(kF) * NodeDim());
  worker.Start();
  std::vector<double> plan(static_cast<size_t>(kF) * NodeDim(), 0.0);
  int64_t target = 0;
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_FALSE(worker.WaitForPlan(0.05, plan.data(), &target)) << "nothing was submitted";
  const double waited =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  EXPECT_GE(waited, 0.04);
  EXPECT_LT(waited, 1.0) << "the arming wait must be bounded, not indefinite";
  worker.Stop();
}

// ---------------------------------------------------------------------------------------
// The one that matters: threading must not change the answer.
// ---------------------------------------------------------------------------------------

/// Wraps a worker so the FIRST collect blocks, modelling what the ROS node does at arming.
///
/// Without this the async run has no plan on tick 0 while the synchronous run does, and the
/// difference does not stay local: the sampler folds the newest history row into its output,
/// that output becomes the next history row, so a single differing tick propagates through the
/// autoregressive conditioning forever. Arming is exactly why FmWorker::WaitForPlan exists --
/// there is no live plan and deliberately no PD warm-up to hide behind.
class BlockingFirstPlan : public FmSampler {
 public:
  BlockingFirstPlan(std::shared_ptr<FmWorker> worker, double timeout_s)
      : worker_(std::move(worker)), timeout_s_(timeout_s) {}

  void Submit(const FmRequest& req) override { worker_->Submit(req); }

  bool TryCollect(double* pred_out, int64_t* target) override {
    if (!first_done_) {
      const bool got = worker_->WaitForPlan(timeout_s_, pred_out, target);
      if (got) first_done_ = true;
      return got;
    }
    return worker_->TryCollect(pred_out, target);
  }

 private:
  std::shared_ptr<FmWorker> worker_;
  double timeout_s_;
  bool first_done_ = false;
};

/// Drive a core for `ticks` steps and return every served window, concatenated.
std::vector<double> RunCore(GenTrackerCore* core, int ticks, bool pump_worker) {
  const size_t row = static_cast<size_t>(kOffsets.size()) * NodeDim();
  std::vector<double> all;
  std::vector<double> window(row, 0.0);
  for (int t = 0; t < ticks; ++t) {
    core->Step(MakeState(t), window.data());
    if (pump_worker) {
      // Let the worker finish before the tick that would adopt its plan. On the robot the
      // lead window does this; here it makes the comparison deterministic rather than
      // dependent on how fast the test machine happens to be.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    all.insert(all.end(), window.begin(), window.end());
  }
  return all;
}

TEST(FmWorker, ThreadedOutputIsIdenticalToSynchronousWhenDeadlinesAreMet) {
  const int lead = 3;
  const int ticks = 4 * kK;

  // Synchronous: the sampler runs inline, so a plan is ready the instant it is asked for.
  auto sync_inner = MakeSampler(0.0);
  GenTrackerCore sync_core(MakeConfig(lead), sync_inner);
  const std::vector<double> sync_out = RunCore(&sync_core, ticks, /*pump_worker=*/false);

  // Threaded: the same sampler, on a thread, with a latency well inside the lead window, and
  // the arming wait the node performs.
  auto async_inner = MakeSampler(0.5);
  auto worker = std::make_shared<FmWorker>(async_inner, static_cast<size_t>(kF) * NodeDim());
  worker->Start();
  GenTrackerCore async_core(MakeConfig(lead),
                            std::make_shared<BlockingFirstPlan>(worker, 1.0));
  const std::vector<double> async_out = RunCore(&async_core, ticks, /*pump_worker=*/true);
  worker->Stop();

  ASSERT_EQ(sync_out.size(), async_out.size());
  size_t diffs = 0;
  for (size_t i = 0; i < sync_out.size(); ++i) {
    if (sync_out[i] != async_out[i]) ++diffs;
  }
  EXPECT_EQ(diffs, 0u) << diffs << " of " << sync_out.size()
                       << " served values differ between the synchronous and threaded runs";
  EXPECT_EQ(async_core.scheduler().late_events(), 0u);
  EXPECT_EQ(async_core.scheduler().offsets_clamped(), 0u);
  EXPECT_EQ(worker->plans_dropped(), 0u);
}

TEST(FmWorker, ASlowGeneratorMakesTheScheduleLateButNeverUnsafe) {
  // 400 ms per plan against a 240 ms cadence: the worker cannot keep up. The tick must keep
  // running, the far offsets must saturate rather than read out of bounds, and it must all be
  // counted.
  auto inner = MakeSampler(400.0);
  auto worker = std::make_shared<FmWorker>(inner, static_cast<size_t>(kF) * NodeDim());
  worker->Start();
  GenTrackerCore core(MakeConfig(3), std::make_shared<BlockingFirstPlan>(worker, 2.0));

  const size_t row = static_cast<size_t>(kOffsets.size()) * NodeDim();
  std::vector<double> window(row, 0.0);
  // Paced at the real control period. Without it the loop runs in microseconds, no plan ever
  // completes, and the core simply reports "no plan" -- which is a different state from LATE
  // and would make this test pass for the wrong reason.
  for (int t = 0; t < 40; ++t) {
    core.Step(MakeState(t), window.data());
    for (double v : window) ASSERT_TRUE(std::isfinite(v)) << "tick " << t;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  worker->Stop();

  EXPECT_GT(core.scheduler().late_events(), 0u) << "a generator this slow must report late";
  EXPECT_GT(core.scheduler().offsets_clamped(), 0u);
}

}  // namespace
}  // namespace g1_deploy
