// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// The generator on its own thread.
//
// FmWorker IS an FmSampler and WRAPS an FmSampler, so GenTrackerCore does not change at all
// between the synchronous tests and the threaded robot: the same core, the same schedule, the
// same FmSession underneath. Only where the inference runs differs.
//
// ============================ why a thread at all =========================================
// Measured on a Jetson Thor with clocks pinned, real artifacts, GPU-resident: the generator's
// p99 is 19.0 ms against a 20 ms control period. That "fits" with ~5% headroom on an IDLE
// GPU -- which the ZED depth pipeline shares. Off the control thread the same work has the
// whole lead window, so contention can cost something without missing a deadline. On this
// hardware the thread is load-bearing, not an optimisation.
//
// ============================ the handoff =================================================
// One producer (the worker), one consumer (the control tick), and the consumer COPIES the
// plan out rather than retaining the buffer -- FmSampler::TryCollect's contract. That is what
// keeps this small: a slot only has to survive one copy, so a short rotation plus an atomic
// "which slot is being read" is enough, with no lock in the tick and no allocation after
// construction.
//
// A mutex was considered and rejected for the READ path: the worker runs at a deliberately
// lower priority on a different core, so blocking the 50 Hz tick behind it is a textbook
// priority inversion. The worker still takes a lock to wait for work, but only ever on its
// own side.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "g1_deploy/gen_tracker_core.h"

namespace g1_deploy {

/// Single-producer/single-consumer handoff of the newest plan.
///
/// Three slots: one being written, one published, one possibly being read. The producer never
/// writes the slot the consumer is reading, so a publish can never tear a copy in progress.
class PlanMailbox {
 public:
  PlanMailbox(int slots, size_t plan_elems);

  /// Producer: a slot to write into. Never the slot the consumer is reading.
  double* AcquireSlot();

  /// Producer: make the slot just written the newest.
  void Publish(int64_t target_ctrl_step);

  /// Consumer: copy the newest unread plan out. Returns false if there is nothing new.
  bool TryConsume(double* out, int64_t* target_ctrl_step);

  /// Plans published but superseded before the consumer read them.
  uint32_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

 private:
  int slots_;
  size_t plan_elems_;
  std::vector<double> storage_;      //: slots_ * plan_elems_
  std::vector<int64_t> targets_;
  int write_slot_ = 0;
  std::atomic<int> newest_{-1};      //: slot index of the newest published plan, -1 if none
  std::atomic<int> reading_{-1};     //: slot the consumer is copying from
  std::atomic<uint64_t> published_{0};
  uint64_t consumed_ = 0;
  std::atomic<uint32_t> dropped_{0};
};

/// Runs an inner FmSampler on a dedicated thread.
class FmWorker : public FmSampler {
 public:
  /// Args:
  ///   inner: the synchronous sampler to run (an FmSession on the robot).
  ///   plan_elems: F * D, the size of one plan.
  FmWorker(std::shared_ptr<FmSampler> inner, size_t plan_elems);
  ~FmWorker() override;

  /// Start the thread. Call from on_activate, after any affinity or priority has been set on
  /// the calling thread -- the worker inherits nothing, but the ORT session it creates does.
  void Start();

  /// Stop and join. Safe to call twice; the destructor calls it.
  void Stop();

  /// Hand over a request. Never blocks and never allocates.
  ///
  /// If the worker is still busy with the previous request this is DROPPED and counted rather
  /// than queued: a queue would let the generator fall arbitrarily far behind while still
  /// producing plans for boundaries that had already passed.
  void Submit(const FmRequest& req) override;

  /// Collect a finished plan, if there is one. Never blocks.
  bool TryCollect(double* pred_out, int64_t* target_ctrl_step) override;

  /// Block until a plan arrives or `timeout_s` elapses. Returns false on timeout.
  ///
  /// The ONE blocking path, for the first tick after arming: there is no live plan at all
  /// then, and there is deliberately no PD warm-up to hide behind. Must be unreachable once a
  /// plan exists.
  bool WaitForPlan(double timeout_s, double* pred_out, int64_t* target_ctrl_step);

  bool busy() const { return busy_.load(std::memory_order_acquire); }
  uint32_t requests_skipped() const { return skipped_.load(std::memory_order_relaxed); }
  uint32_t plans_dropped() const { return mailbox_.dropped(); }
  uint32_t plans_produced() const { return produced_.load(std::memory_order_relaxed); }

 private:
  void Loop();

  std::shared_ptr<FmSampler> inner_;
  size_t plan_elems_;
  PlanMailbox mailbox_;

  std::thread thread_;
  std::mutex mu_;                 //: guards pending_ and has_pending_, worker side only
  std::condition_variable cv_;
  FmRequest pending_;
  bool has_pending_ = false;
  bool stop_ = false;

  std::atomic<bool> busy_{false};
  std::atomic<bool> running_{false};
  std::atomic<uint32_t> skipped_{0};
  std::atomic<uint32_t> produced_{0};
  std::vector<double> scratch_;   //: worker-private, the inner sampler's output
};

}  // namespace g1_deploy
