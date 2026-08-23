// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// The receding-horizon algorithm, without the network.
//
// Port of the serving path in transfer/deploy/gen_tracker.py (_update_generator, _run_fm's
// bookkeeping, _serve_nodes) with the generator behind an injectable seam. That seam is the
// same trick test_replan_cadence.py's _FakePolicy uses: stub the network, keep cursor,
// cadence, anchor, stash and re-anchor real, so the algorithm is testable with no ONNX, no
// GPU and no threads.
//
// The step order IS the algorithm, and it mirrors the python:
//
//   1. push the node served LAST step into the history (raw, in its plan's own frame);
//   2. request a plan if the cadence says so -- conditioned on the history PROJECTED to the
//      boundary that plan is for, which is what makes the early request exact;
//   3. adopt a delivered plan, or advance the cursor into the live one;
//   4. serve the window at the tracker's offsets, re-anchored onto the measured base NOW;
//   5. stash the node served THIS step, raw, for the next step's push.
//
// Step 5 keeps the RAW node rather than the re-anchored one on purpose: the history is the
// generator's own past output in its own frame, which is how the dataset builds it, so there
// is nothing to transform and no odometry entering the generator's conditioning.

#include <cstdint>
#include <memory>
#include <vector>

#include "g1_deploy/node_history.h"
#include "g1_deploy/plan_scheduler.h"
#include "g1_deploy/types.h"
#include "g1_deploy/window.h"

namespace g1_deploy {

/// The measured robot state the generator path reads.
struct CoreState {
  Vec3 root_pos = Vec3::Zero();
  QuatXyzw root_quat{0.0, 0.0, 0.0, 1.0};
  std::vector<double> command;  //: cmd_dim wide
};

/// Everything a generator needs to produce one plan, captured on the control thread.
struct FmRequest {
  int64_t target_ctrl_step = 0;  //: the boundary this plan is FOR
  int64_t request_ctrl_step = 0;
  //: N x D, oldest first -- the history as it WILL be at target_ctrl_step. Not as it is now.
  std::vector<double> history_nodes;
  //: N x cmd_dim. The rows past the request hold the command at request time; the future
  //: command is not knowable, which is the same assumption the FM's own command horizon makes.
  std::vector<double> history_cmd;
  //: history_null as it will be at the boundary, so the learned null embedding is applied on
  //: exactly the steps the synchronous runtime would have applied it.
  bool history_null = true;
};

/// The generator, behind a seam. Real implementations run fm.onnx; tests substitute a
/// deterministic stand-in.
class FmSampler {
 public:
  virtual ~FmSampler() = default;

  /// Begin producing a plan for `req`. May complete immediately or later.
  virtual void Submit(const FmRequest& req) = 0;

  /// If a plan is ready, write its F x D nodes (row-major) into `pred_out`, set
  /// `target_ctrl_step`, and return true. Called once per tick.
  virtual bool TryCollect(double* pred_out, int64_t* target_ctrl_step) = 0;
};

/// Sizes and layout, all from artifact metadata.
struct CoreConfig {
  int num_joints = 0;
  int num_ee_frames = 0;
  std::vector<ReferenceBlock> blocks;
  std::vector<int> offsets;  //: the tracker's reference-window sample offsets
  int history_length = 0;    //: N
  int cmd_dim = 3;
  ScheduleConfig schedule;
  //: Which serve arm the ARTIFACT asks for -- a property of the tracker, not of this build,
  //: so one binary plays whatever it is handed (subject to AnchorFromConvention refusing the
  //: odometry-dependent one). Defaulted to kPlan, the only arm PolicyRuntime::Load can
  //: actually produce, so a config assembled without setting it serves the prediction as made
  //: rather than silently re-anchoring on a base position this runtime does not trust.
  WindowAnchor window_anchor = WindowAnchor::kPlan;
};

/// The algorithm. Single-threaded by construction: the FM seam is what crosses a thread in
/// the real runtime, and it is the only thing that does.
class GenTrackerCore {
 public:
  GenTrackerCore(const CoreConfig& cfg, std::shared_ptr<FmSampler> sampler);

  /// Clear the plan, cursor, anchor and stash.
  ///
  /// A stash from before a reset describes another episode's pose; keeping it would push a
  /// stale row into the generator's conditioning on the first step after arming.
  void Reset();

  /// Run one control step and write the served window into `window_out`
  /// (offsets.size() x D, row-major), re-anchored onto the measured base.
  ///
  /// Returns what the scheduler did, so the caller can log lateness.
  TickOutcome Step(const CoreState& state, double* window_out);

  int node_dim() const { return node_dim_; }
  int window_rows() const { return static_cast<int>(cfg_.offsets.size()); }
  const PlanScheduler& scheduler() const { return sched_; }
  bool has_plan() const { return sched_.has_plan(); }
  /// The raw node served last step, in its plan's own frame. Empty before the first serve.
  const std::vector<double>& prev_served_node() const { return prev_served_node_; }
  const Vec3& anchor_pos() const { return anchor_pos_; }
  const QuatXyzw& anchor_quat() const { return anchor_quat_; }

  /// The LIVE plan, F x D row-major, in its own anchor frame. Port of
  /// GenTrackerPolicy.predicted_window().
  ///
  /// FOR DIAGNOSTICS. The serving path never reads it whole -- it indexes the nodes the
  /// tracker's offsets ask for and re-anchors those. Meaningful only while has_plan(), and
  /// only between Step() calls: Step() swaps this buffer with the incoming one.
  const std::vector<double>& predicted_window() const { return pred_; }

  /// The request the LIVE plan was produced from: its history, command history and null mask.
  ///
  /// The generator's conditioning is its OWN past output, so when it goes wrong it does so
  /// silently and compounds -- this is the only way to see what it was actually asked.
  /// `target_ctrl_step` is 0 and the vectors are empty before the first adoption.
  const FmRequest& adopted_request() const { return adopted_request_; }

 private:
  void BuildRequest(FmRequest* req) const;

  CoreConfig cfg_;
  std::shared_ptr<FmSampler> sampler_;
  PlanScheduler sched_;
  NodeHistory history_;
  int node_dim_ = 0;

  std::vector<double> pred_;      //: F x D, the live plan
  //: F x D. A plan that has been produced but whose boundary has not arrived -- the model of
  //: the real runtime's mailbox. Held, not discarded: with a lead the generator finishes
  //: early by design.
  std::vector<double> incoming_;
  bool has_pending_ = false;
  int64_t pending_target_ = 0;
  std::vector<double> prev_served_node_;  //: D, raw
  FmRequest last_request_;                //: the newest request submitted, in flight or not
  FmRequest adopted_request_;             //: the one that produced pred_
  //: Whether anything has actually been served. Until it has, the history rows are
  //: placeholders and the window must be reported as null.
  bool has_served_ = false;
  Vec3 anchor_pos_ = Vec3::Zero();
  QuatXyzw anchor_quat_{0.0, 0.0, 0.0, 1.0};
};

}  // namespace g1_deploy
