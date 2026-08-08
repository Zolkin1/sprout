// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/plan_scheduler.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace g1_deploy {

PlanScheduler::PlanScheduler(const ScheduleConfig& cfg) : cfg_(cfg) {
  if (cfg_.replan_period < 1) throw std::runtime_error("replan_period (K) must be >= 1");
  if (cfg_.lead_ticks < 0) throw std::runtime_error("lead_ticks must be >= 0");
  // The cursor sits up to K-1 nodes past the re-plan when the furthest sample is read, so
  // serving offset `o` at a period of K needs F >= o + K. Nothing clamps in normal
  // operation: a short horizon would freeze the far end of the window silently, which reads
  // as a tracking bug rather than a horizon one.
  const int need = cfg_.max_offset + cfg_.replan_period;
  if (cfg_.horizon < need) {
    throw std::runtime_error(
        "the generator predicts F=" + std::to_string(cfg_.horizon) + " nodes, but serving "
        "offset " + std::to_string(cfg_.max_offset) + " at a re-plan period of " +
        std::to_string(cfg_.replan_period) + " needs F >= " + std::to_string(need) +
        " (the cursor is up to " + std::to_string(cfg_.replan_period - 1) +
        " nodes past the re-plan when the furthest sample is read). Lower fm_update_every, "
        "or re-record with --ref_future_len " + std::to_string(need) +
        " and retrain with --future_len " + std::to_string(need) + ".");
  }
  // A request issued before the previous plan was adopted would overwrite a plan still in
  // flight and make the conditioning projection ill-defined.
  if (cfg_.lead_ticks >= cfg_.replan_period) {
    throw std::runtime_error("lead_ticks (" + std::to_string(cfg_.lead_ticks) +
                             ") must be < the re-plan period K (" +
                             std::to_string(cfg_.replan_period) + ")");
  }
}

void PlanScheduler::Reset() {
  ctrl_step_ = -1;
  has_plan_ = false;
  cursor_ = 0;
  plan_target_ = 0;
  late_ = false;
  late_ticks_ = 0;
  late_events_ = 0;
  offsets_clamped_ = 0;
}

void PlanScheduler::BeginTick() {
  ++ctrl_step_;
  adopted_this_tick_ = false;
}

bool PlanScheduler::ShouldRequest() const {
  // No live plan: ask immediately rather than waiting for the next boundary. This is the
  // first armed tick, and it is why arming does not need a warm-up -- the generator is
  // sampled with the learned null-history embedding instead.
  if (!has_plan_) return true;
  return ((ctrl_step_ + cfg_.lead_ticks) % cfg_.replan_period) == 0;
}

int64_t PlanScheduler::RequestTargetStep() const {
  if (!has_plan_) return ctrl_step_;  // wanted now, not at the next boundary
  return ctrl_step_ + cfg_.lead_ticks;
}

bool PlanScheduler::TryAdopt(int64_t target_ctrl_step) {
  // A plan that arrives before its boundary waits. Adopting it early would serve a window
  // built for a later instant, which is the lag this whole design exists to avoid.
  if (target_ctrl_step > ctrl_step_) return false;

  // The cursor that keeps the plan time-correct is how late it is: its node `j` corresponds
  // to step target + j, so serving at `ctrl_step_` means node `ctrl_step_ - target`. On time
  // that is 0. This is bounded by K-1 in practice, because a plan more than a full period
  // late is superseded by the next one.
  const int64_t delta = ctrl_step_ - target_ctrl_step;
  const int max_cursor = cfg_.horizon - cfg_.max_offset - 1;
  if (delta > max_cursor) {
    // Too stale to serve at all: adopting it would saturate from its very first tick. Drop
    // it and keep the previous plan (or none), and let the caller's lateness reporting stand.
    return false;
  }

  cursor_ = static_cast<int>(delta);
  plan_target_ = target_ctrl_step;
  has_plan_ = true;
  adopted_this_tick_ = true;
  adopted_late_ = (delta > 0);
  late_ = false;  // recovered; the EVENT stays counted
  return true;
}

TickOutcome PlanScheduler::ResolveTick() {
  // Adoption already placed the cursor for this tick, so do not also advance it.
  if (adopted_this_tick_) {
    return adopted_late_ ? TickOutcome::kAdoptedLate : TickOutcome::kAdoptedOnTime;
  }
  if (!has_plan_) return TickOutcome::kNoPlan;

  ++cursor_;
  // Late is a property of the SCHEDULE, not of the horizon: by now a plan for the current
  // boundary should have been adopted and the cursor reset. Saturation of the far offsets is
  // a separate, horizon-dependent consequence -- with the current artifact the two coincide
  // exactly (F == max_offset + K), and with a longer-horizon generator they will not.
  if (cursor_ >= cfg_.replan_period) {
    if (!late_) {
      late_ = true;
      ++late_events_;
    }
    ++late_ticks_;
    return TickOutcome::kLate;
  }
  return TickOutcome::kAdvanced;
}

int PlanScheduler::NodeIndex(int offset) {
  const int raw = cursor_ + offset;
  const int last = cfg_.horizon - 1;
  if (raw > last) {
    ++offsets_clamped_;
    return last;
  }
  return raw;
}

}  // namespace g1_deploy
