// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/fm_worker.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace g1_deploy {

PlanMailbox::PlanMailbox(int slots, size_t plan_elems)
    : slots_(slots), plan_elems_(plan_elems) {
  if (slots < 3) throw std::runtime_error("PlanMailbox needs at least 3 slots");
  storage_.assign(static_cast<size_t>(slots) * plan_elems, 0.0);
  targets_.assign(static_cast<size_t>(slots), 0);
}

double* PlanMailbox::AcquireSlot() {
  // Never the slot the consumer is copying from, and never the newest (which the consumer may
  // be about to read). With three slots one always remains.
  const int reading = reading_.load(std::memory_order_acquire);
  const int newest = newest_.load(std::memory_order_acquire);
  for (int i = 0; i < slots_; ++i) {
    const int candidate = (write_slot_ + i) % slots_;
    if (candidate != reading && candidate != newest) {
      write_slot_ = candidate;
      return storage_.data() + static_cast<size_t>(write_slot_) * plan_elems_;
    }
  }
  throw std::runtime_error("PlanMailbox has no free slot; this is impossible with 3 slots "
                           "and one reader");
}

void PlanMailbox::Publish(int64_t target_ctrl_step) {
  targets_[static_cast<size_t>(write_slot_)] = target_ctrl_step;
  // Release so the slot's contents are visible before the index that points at them.
  newest_.store(write_slot_, std::memory_order_release);
  published_.fetch_add(1, std::memory_order_release);
}

bool PlanMailbox::TryConsume(double* out, int64_t* target_ctrl_step) {
  const uint64_t published = published_.load(std::memory_order_acquire);
  if (published == consumed_) return false;
  if (published - consumed_ > 1) {
    // The producer published more than once since the last read: the intermediate plans were
    // superseded. Counted rather than silent -- it means the generator produced a plan whose
    // boundary the tracker never saw.
    dropped_.fetch_add(static_cast<uint32_t>(published - consumed_ - 1),
                       std::memory_order_relaxed);
  }
  const int slot = newest_.load(std::memory_order_acquire);
  reading_.store(slot, std::memory_order_release);
  const double* src = storage_.data() + static_cast<size_t>(slot) * plan_elems_;
  std::copy(src, src + plan_elems_, out);
  *target_ctrl_step = targets_[static_cast<size_t>(slot)];
  reading_.store(-1, std::memory_order_release);
  consumed_ = published;
  return true;
}

FmWorker::FmWorker(std::shared_ptr<FmSampler> inner, size_t plan_elems)
    : inner_(std::move(inner)), plan_elems_(plan_elems), mailbox_(3, plan_elems) {
  if (!inner_) throw std::runtime_error("FmWorker needs an inner FmSampler");
  scratch_.assign(plan_elems, 0.0);
}

FmWorker::~FmWorker() { Stop(); }

void FmWorker::Start() {
  if (running_.exchange(true)) return;
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = false;
  }
  thread_ = std::thread([this] { Loop(); });
}

void FmWorker::Stop() {
  if (!running_.exchange(false)) return;
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void FmWorker::Submit(const FmRequest& req) {
  // Dropped, not queued. A queue would let the generator fall arbitrarily behind while still
  // producing plans for boundaries that have already gone by -- which the scheduler would then
  // have to reject one at a time, turning one slow tick into a permanent lag.
  if (busy_.load(std::memory_order_acquire)) {
    skipped_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    pending_ = req;  // copies the conditioning; the worker must not read caller-owned buffers
    has_pending_ = true;
  }
  busy_.store(true, std::memory_order_release);
  cv_.notify_one();
}

bool FmWorker::TryCollect(double* pred_out, int64_t* target_ctrl_step) {
  return mailbox_.TryConsume(pred_out, target_ctrl_step);
}

bool FmWorker::WaitForPlan(double timeout_s, double* pred_out, int64_t* target_ctrl_step) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    if (mailbox_.TryConsume(pred_out, target_ctrl_step)) return true;
    // Poll rather than wait on a second condition variable: this runs once, at arming, and a
    // 200 us granularity is irrelevant against a 200 ms timeout. A cv here would add a
    // notification path that exists solely for one tick of one run.
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return mailbox_.TryConsume(pred_out, target_ctrl_step);
}

void FmWorker::Loop() {
  for (;;) {
    FmRequest req;
    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] { return has_pending_ || stop_; });
      if (stop_ && !has_pending_) return;
      req = std::move(pending_);
      has_pending_ = false;
    }

    // The inner sampler is synchronous: Submit runs the graph, TryCollect hands it back. Same
    // code path as the single-threaded tests and the same one measured by the benchmark.
    int64_t target = 0;
    inner_->Submit(req);
    if (inner_->TryCollect(scratch_.data(), &target)) {
      double* slot = mailbox_.AcquireSlot();
      std::copy(scratch_.begin(), scratch_.end(), slot);
      mailbox_.Publish(target);
      produced_.fetch_add(1, std::memory_order_relaxed);
    }
    // Cleared last: the control thread uses this to decide whether to issue a request, and
    // clearing it before the plan is published would let it ask again while this one was
    // still in flight.
    busy_.store(false, std::memory_order_release);
  }
}

}  // namespace g1_deploy
