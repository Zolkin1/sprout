// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// WHERE the tracker's reference_window comes from -- the generator, or a baked clip.
//
// The runtime used to be generator-shaped throughout: PolicyRuntime::Load refused an empty
// fm.onnx outright, Step called GenTrackerCore unconditionally, and BuildValues tiled the
// current command across the offsets because the generator has no per-step command to serve.
// Clip playback differs in all three, so it went in behind one seam rather than as three
// branches -- a source is chosen once, at load, and the tick does not know which it has.
//
// That matters for what clip playback is FOR. A clip case removes the generator from the
// loop, so a failure is unambiguously the stack below it; but until this existed the only way
// to run one was the deprecated python controller, which changed the controller LANGUAGE at
// the same time and isolated neither. Now both go through this runtime and only the reference
// differs.

#include <memory>
#include <string>
#include <vector>

#include "g1_deploy/clip_reference.h"
#include "g1_deploy/fm_session.h"
#include "g1_deploy/fm_worker.h"
#include "g1_deploy/gen_tracker_core.h"
#include "g1_deploy/plan_scheduler.h"
#include "g1_deploy/policy_params.h"
#include "g1_deploy/types.h"

namespace g1_deploy {

struct HardwareReadings;
struct TickDiagnostics;

/// One tick's worth of reference, however it was produced.
class ReferenceSource {
 public:
  virtual ~ReferenceSource() = default;

  /// Clear per-episode state. Called on arming.
  virtual void Reset() = 0;

  /// Advance one control step and write `offsets.size() * D` doubles into `window_out`.
  virtual TickOutcome Step(const HardwareReadings& readings, const CoreState& state,
                           double* window_out) = 0;

  /// Each offset's own command, or null to hold the current one across all of them.
  ///
  /// The generator has no per-step command so it holds -- which is what the deployed joystick
  /// does anyway. A clip authored one per step, and serving the held command instead would
  /// drive command_window with a number the motion never asked for.
  virtual const std::vector<double>* CommandWindow() const { return nullptr; }

  /// Fill the half of the diagnostics that is this source's.
  virtual void FillDiagnostics(TickDiagnostics* diag) const = 0;
};

/// The flow-matching generator: schedule, worker, conditioning and served window.
class FmReferenceSource : public ReferenceSource {
 public:
  /// \param params The TRACKER's params -- the camera contract the conditioning is built from.
  FmReferenceSource(std::shared_ptr<FmSession> fm, std::shared_ptr<FmWorker> worker,
                    std::shared_ptr<FmSampler> sampler, std::unique_ptr<GenTrackerCore> core,
                    const PolicyParams& params, int lead_ticks);

  void Reset() override;
  TickOutcome Step(const HardwareReadings& readings, const CoreState& state,
                   double* window_out) override;
  void FillDiagnostics(TickDiagnostics* diag) const override;

  const FmMeta& meta() const { return fm_->meta(); }
  const std::vector<double>& plan_window() const { return core_->predicted_window(); }
  const FmRequest& plan_request() const { return core_->adopted_request(); }
  const FmConditioning& conditioning() const { return last_cond_; }
  int replan_period() const { return core_->scheduler().config().replan_period; }
  int lead_ticks() const { return lead_ticks_; }
  /// Re-arm the blocking first collect after a reset.
  void set_blocker(std::shared_ptr<FmSampler> blocker) { blocker_ = std::move(blocker); }

 private:
  /// Everything the generator is conditioned on this tick, from the raw readings.
  void BuildConditioning(const HardwareReadings& readings);

  std::shared_ptr<FmSession> fm_;
  std::shared_ptr<FmWorker> worker_;
  std::shared_ptr<FmSampler> sampler_;
  std::shared_ptr<FmSampler> blocker_;
  std::unique_ptr<GenTrackerCore> core_;
  const PolicyParams& params_;
  int lead_ticks_ = 1;
  TickOutcome last_outcome_ = TickOutcome::kNoPlan;
  FmConditioning last_cond_;
};

/// A baked motion clip, with the generator not loaded at all.
class ClipReferenceSource : public ReferenceSource {
 public:
  ClipReferenceSource(ClipReference clip, std::vector<int> offsets, int num_joints,
                      int num_ee_frames, std::vector<ReferenceBlock> blocks,
                      double duration_sec);

  void Reset() override;
  TickOutcome Step(const HardwareReadings& readings, const CoreState& state,
                   double* window_out) override;
  const std::vector<double>* CommandWindow() const override { return &command_window_; }
  void FillDiagnostics(TickDiagnostics* diag) const override;

  const ClipReference& clip() const { return clip_; }
  /// Whether playback has reached its last played frame and is holding it.
  bool exhausted() const { return exhausted_; }

 private:
  ClipReference clip_;
  std::vector<int> offsets_;
  int num_joints_ = 0;
  int num_ee_frames_ = 0;
  std::vector<ReferenceBlock> blocks_;
  int stop_tick_ = 0;          //: last tick that PLAYS; beyond it the reference holds
  int tick_ = -1;
  bool exhausted_ = false;
  std::vector<double> command_window_;
};

}  // namespace g1_deploy
