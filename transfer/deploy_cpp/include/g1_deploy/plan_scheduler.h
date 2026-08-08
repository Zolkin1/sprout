// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// The receding-horizon schedule: when to ask the generator for a plan, when to adopt one,
// which node to serve, and what to do when a plan does not arrive in time.
//
// Deliberately holds NO tensors, no ONNX session and no threads -- only the decision logic,
// so every property below is testable without a GPU, a model or a race. The FM worker and
// the observation assembly consume it.
//
// ============================ the schedule ================================================
// The cadence is `ctrl_step % K` on an ABSOLUTE counter, mirroring Isaac's
// `episode_length_buf % K`. The generator is asked for a plan `lead_ticks` BEFORE the
// boundary that plan is for, so it has a whole lead window to do ~16 ms of work instead of
// having to fit inside one 20 ms tick.
//
// That is only sound because the conditioning can be built for the FUTURE boundary at
// request time: the history rows are the nodes this runtime will SERVE between now and then,
// and those are already determined by the live plan (see NodeHistory::StackPaddedNodesProjected).
// So the generator sees exactly what a synchronous implementation would have handed it at the
// boundary, and its output is a plan FOR the boundary -- not for the request tick. Hence
// adoption at cursor 0 on time, and the reference is not time-lagged.
//
// What IS sampled early, by lead_ticks, is the exteroception and the velocity command.
//
// ============================ lateness ====================================================
// If the plan has not arrived by its boundary the cursor keeps ADVANCING -- time does not
// stop -- and served node indices saturate at F-1, so the far end of the window repeats its
// last node while the near offsets keep moving. A late plan is adopted on arrival at the
// cursor that keeps it time-correct, which is how late it is.
//
// Note the current artifact has F == max_offset + K exactly (62 == 50 + 12), so saturation
// begins on precisely the tick a plan becomes late. That coincidence is NOT relied on: being
// late is a property of the SCHEDULE (cursor >= K) and saturation is a property of the
// HORIZON (cursor + offset > F-1). A generator retrained with more buffer separates them, and
// the tests pin both independently.

#include <cstdint>
#include <vector>

namespace g1_deploy {

/// Why the cursor moved this tick. Recorded so a bag can distinguish the cases offline.
enum class TickOutcome {
  kNoPlan,           //: nothing adopted yet; nothing can be served
  kAdoptedOnTime,    //: a plan arrived for this boundary
  kAdoptedLate,      //: a plan arrived after its boundary and was adopted on arrival
  kAdvanced,         //: walked one node further into the live plan
  kLate,             //: no plan for this boundary; cursor advanced, far offsets may saturate
};

/// Configuration, all of it derived from artifact metadata rather than chosen here.
struct ScheduleConfig {
  int replan_period = 1;   //: K, from tracker.onnx `fm_update_every`
  int horizon = 1;         //: F, the generator's node count
  int max_offset = 0;      //: the largest reference-window offset the tracker observes
  int lead_ticks = 1;      //: how far before its boundary a plan is requested
};

/// Tracks the cadence, the cursor and lateness. One instance per policy.
class PlanScheduler {
 public:
  /// Throws std::runtime_error if the configuration cannot work:
  ///   * F < max_offset + K -- the cursor would leave the horizon during normal operation,
  ///     which would silently freeze the far end of the window and read as a tracking bug.
  ///   * lead_ticks >= K -- a request would be issued before the previous plan was adopted.
  explicit PlanScheduler(const ScheduleConfig& cfg);

  /// Reset to the pre-armed state. Does not clear plan_id, so ids stay unique across resets
  /// and a bag can tell a re-arm from a re-plan.
  void Reset();

  /// Advance the tick counter. Call once at the top of each control tick, before anything
  /// else here.
  void BeginTick();

  /// True if the generator should be asked for a plan on this tick.
  ///
  /// Fires `lead_ticks` before a cadence boundary, and also whenever there is no live plan at
  /// all (after a reset), which is what lets the first armed tick get a plan without waiting
  /// for the next boundary.
  bool ShouldRequest() const;

  /// The control step the plan requested on this tick is FOR.
  int64_t RequestTargetStep() const;

  /// Offer a delivered plan. Adopted only once its target step has arrived, so a plan that
  /// lands early waits for its boundary rather than taking effect at the wrong time.
  ///
  /// Returns true if it was adopted.
  bool TryAdopt(int64_t target_ctrl_step);

  /// Resolve the tick after any adoption: advance the cursor, or record lateness.
  ///
  /// Call once per tick, after TryAdopt.
  TickOutcome ResolveTick();

  /// Node index for a reference-window offset, saturating at the last node.
  ///
  /// Saturation is the lateness contingency: the far end of the window repeats its final node
  /// rather than running off the end. Counted, so it is visible rather than silent.
  int NodeIndex(int offset);

  bool has_plan() const { return has_plan_; }
  int cursor() const { return cursor_; }
  int64_t ctrl_step() const { return ctrl_step_; }
  bool late() const { return late_; }
  uint32_t late_ticks() const { return late_ticks_; }
  uint32_t late_events() const { return late_events_; }
  uint32_t offsets_clamped() const { return offsets_clamped_; }
  const ScheduleConfig& config() const { return cfg_; }

 private:
  ScheduleConfig cfg_;
  // int64_t, NOT uint64_t: this starts at -1 so that the first BeginTick() makes it 0 and
  // `0 % K == 0` forces the first plan. As an unsigned type -1 is UINT64_MAX, and
  // UINT64_MAX % 12 == 3, which would silently shift the entire cadence phase.
  int64_t ctrl_step_ = -1;
  bool has_plan_ = false;
  bool adopted_this_tick_ = false;
  bool adopted_late_ = false;
  int cursor_ = 0;
  int64_t plan_target_ = 0;
  bool late_ = false;
  uint32_t late_ticks_ = 0;
  uint32_t late_events_ = 0;
  uint32_t offsets_clamped_ = 0;
};

}  // namespace g1_deploy
