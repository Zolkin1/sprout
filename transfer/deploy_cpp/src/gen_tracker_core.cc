// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/gen_tracker_core.h"

#include <algorithm>
#include <stdexcept>

namespace g1_deploy {

GenTrackerCore::GenTrackerCore(const CoreConfig& cfg, std::shared_ptr<FmSampler> sampler)
    : cfg_(cfg),
      sampler_(std::move(sampler)),
      sched_(cfg.schedule),
      history_(cfg.history_length, WindowDim(cfg.num_joints, cfg.num_ee_frames, cfg.blocks),
               cfg.cmd_dim),
      node_dim_(WindowDim(cfg.num_joints, cfg.num_ee_frames, cfg.blocks)) {
  if (cfg_.offsets.empty()) throw std::runtime_error("the tracker observes no window offsets");
  if (!sampler_) throw std::runtime_error("GenTrackerCore needs an FmSampler");
  const size_t plan_elems = static_cast<size_t>(cfg_.schedule.horizon) * node_dim_;
  pred_.assign(plan_elems, 0.0);
  incoming_.assign(plan_elems, 0.0);
  prev_served_node_.assign(node_dim_, 0.0);
}

void GenTrackerCore::Reset() {
  sched_.Reset();
  history_ = NodeHistory(cfg_.history_length, node_dim_, cfg_.cmd_dim);
  std::fill(pred_.begin(), pred_.end(), 0.0);
  std::fill(prev_served_node_.begin(), prev_served_node_.end(), 0.0);
  has_pending_ = false;
  has_served_ = false;
  anchor_pos_ = Vec3::Zero();
  anchor_quat_ = QuatXyzw(0.0, 0.0, 0.0, 1.0);
  // Cleared for the same reason the stash is: these describe another episode. PlanScheduler's
  // plan_id deliberately does NOT reset, so a bag can still tell a re-arm from a re-plan.
  last_request_ = FmRequest{};
  adopted_request_ = FmRequest{};
}

void GenTrackerCore::BuildRequest(FmRequest* req) const {
  req->request_ctrl_step = sched_.ctrl_step();
  req->target_ctrl_step = sched_.RequestTargetStep();

  // The rows between now and the boundary are the nodes this runtime WILL serve on those
  // ticks, and they are already determined: they are the live plan's nodes at the cursor and
  // after it. Handing them over now is what makes the early request exact rather than an
  // approximation -- the generator sees precisely the history a synchronous implementation
  // would have built at the boundary.
  //
  // With no live plan there is nothing to project from, and there is nothing to project TO
  // either: the request targets this tick, so the lookahead is zero.
  const int lookahead = static_cast<int>(req->target_ctrl_step - req->request_ctrl_step);
  std::vector<const double*> future_nodes;
  std::vector<const double*> future_cmds;
  if (lookahead > 0 && sched_.has_plan()) {
    for (int i = 0; i < lookahead; ++i) {
      // cursor() + 1 + i, not cursor() + i: this runs BEFORE ResolveTick, so the cursor is
      // still the previous tick's. The node served on THIS tick is cursor + 1, and the rows
      // wanted are the ones for ticks request+1 .. target. Off by one here would condition
      // the generator on a history shifted a step into the past -- which is exactly the kind
      // of error that trains fine, deploys fine, and tracks slightly wrong forever.
      const int idx = std::min(sched_.cursor() + 1 + i, cfg_.schedule.horizon - 1);
      future_nodes.push_back(pred_.data() + static_cast<size_t>(idx) * node_dim_);
      // The future command is not knowable; hold the one at request time. The FM's own
      // future-command horizon already makes this assumption.
      future_cmds.push_back(nullptr);
    }
  }

  req->history_nodes.assign(static_cast<size_t>(cfg_.history_length) * node_dim_, 0.0);
  req->history_cmd.assign(static_cast<size_t>(cfg_.history_length) * cfg_.cmd_dim, 0.0);
  history_.StackPaddedNodesProjected(future_nodes, req->history_nodes.data());

  // The command rows need real pointers; reuse the newest command for every projected row.
  std::vector<const double*> cmd_ptrs;
  std::vector<double> held(static_cast<size_t>(cfg_.cmd_dim), 0.0);
  if (!future_cmds.empty()) {
    // The newest stored command is the last row of the unprojected stack.
    std::vector<double> stack(static_cast<size_t>(cfg_.history_length) * cfg_.cmd_dim, 0.0);
    history_.StackPaddedCmds(stack.data());
    const double* newest = stack.data() + static_cast<size_t>(cfg_.history_length - 1) * cfg_.cmd_dim;
    held.assign(newest, newest + cfg_.cmd_dim);
    cmd_ptrs.assign(future_cmds.size(), held.data());
  }
  history_.StackPaddedCmdsProjected(cmd_ptrs, req->history_cmd.data());
  req->history_null = history_.NullMaskAfter(std::max(lookahead, 0));
}

TickOutcome GenTrackerCore::Step(const CoreState& state, double* window_out) {
  sched_.BeginTick();

  // 1. Push the node served LAST step.
  //
  //    Before the first serve there is nothing real to push, so the row is a zero placeholder
  //    -- and the window is then kept INVALID so the learned null-history embedding covers it.
  //    A zero row counted as real history is off-distribution: training never presented one,
  //    and it would reach the generator with `history_null` OFF, silently, for exactly the
  //    steps that embedding exists to handle. Without the invalidate the window went valid at
  //    step N-1 with its oldest row still the placeholder.
  history_.Append(prev_served_node_.data(), state.command.data());
  if (!has_served_) history_.Invalidate();

  // 2. Ask for a plan, conditioned on the history projected to the boundary it is for.
  if (sched_.ShouldRequest()) {
    FmRequest req;
    BuildRequest(&req);
    sampler_->Submit(req);
    // Kept so the plan that eventually lands can be published with the conditioning it was
    // built from. One slot is enough: only one request is ever in flight, since FmWorker drops
    // rather than queues while busy and PlanScheduler asserts lead_ticks < K.
    last_request_ = std::move(req);
  }

  // 3. Collect a finished plan, then adopt it once its boundary has arrived.
  //
  //    These are two steps, not one. With a lead the generator finishes BEFORE the boundary
  //    its plan is for, and TryAdopt rightly refuses it until then -- so the plan has to be
  //    HELD. Collecting and discarding would silently skip the boundary and leave the cursor
  //    running on the previous plan until it saturated, which looks exactly like a slow
  //    generator rather than a dropped one. The real runtime's mailbox holds it for the same
  //    reason; this is that mailbox.
  if (!has_pending_ && sampler_->TryCollect(incoming_.data(), &pending_target_)) {
    has_pending_ = true;
  }
  if (has_pending_) {
    if (sched_.TryAdopt(pending_target_)) {
      has_pending_ = false;
      pred_.swap(incoming_);
      // The anchor is the measured base AT ADOPTION, not at the request. The generator's
      // conditioning history ends at this boundary, so the window it produced is expressed
      // relative to the robot HERE -- capturing the request-time pose would shear it by the
      // egomotion of the lead window.
      // TODO(fm-async): revisit after hardware testing. The exteroception in that request was
      // sampled lead_ticks earlier, so if the model's effective "now" tracks the image rather
      // than the history, the anchor should move back with it.
      anchor_pos_ = state.root_pos;
      anchor_quat_ = state.root_quat;
      // Promote the stashed request to "the one that produced the LIVE plan", but only if it
      // is actually that one. Compared on target step rather than assumed: after a dropped or
      // stale plan the newest request is for a LATER boundary than the plan being adopted, and
      // publishing that as this plan's conditioning would be a confident lie in a field whose
      // whole purpose is showing what the generator was asked.
      //
      // On a mismatch it is CLEARED rather than left alone. Keeping the previous value would
      // be the same lie one plan further back, and silent -- the FmPlan would carry a new
      // plan_id, a new trajectory and a new anchor beside a history from two cadence periods
      // ago, with nothing marking the mismatch. Empty is legible: the consumer sees no
      // conditioning and knows not to trust one.
      adopted_request_ =
          (last_request_.target_ctrl_step == pending_target_) ? last_request_ : FmRequest{};
    } else if (pending_target_ +
                   (cfg_.schedule.horizon - cfg_.schedule.max_offset - 1) < sched_.ctrl_step()) {
      // Refused because it is too STALE, not because it is early: adopting it would saturate
      // from its first tick. Drop it so the slot is free for the next one -- holding it would
      // block every future plan behind one that can never be used.
      has_pending_ = false;
    }
  }
  const TickOutcome outcome = sched_.ResolveTick();

  // 4. Serve the window at the tracker's offsets. WHICH pose the base_pose block is measured
  //    from is the artifact's choice, and the two arms differ only in meaning -- same widths,
  //    same blocks -- so this branch is the whole difference between them.
  //
  //    kPlan: the generator already predicts in the frame of the root at the re-plan, which
  //    IS what the observation is defined against, so the prediction is served exactly as it
  //    was made. `state.root_pos` is never read here. That is what makes this deploy path
  //    odometry-free, and it is the only arm PolicyRuntime::Load will build (see
  //    AnchorFromConvention).
  //
  //    kRobotNow: the observation is defined against the robot's base RIGHT NOW, so the
  //    prediction -- anchored where the robot was at adoption -- has to be re-expressed, or
  //    the policy gets a window wrong by exactly the distance travelled since. Reachable only
  //    from a hand-built CoreConfig; a robot_now artifact is refused at load, because doing
  //    this on hardware means trusting a measured base position.
  const int rows = window_rows();
  if (!sched_.has_plan()) {
    std::fill(window_out, window_out + static_cast<size_t>(rows) * node_dim_, 0.0);
    return outcome;
  }
  for (int r = 0; r < rows; ++r) {
    const int idx = sched_.NodeIndex(cfg_.offsets[static_cast<size_t>(r)]);
    const double* src = pred_.data() + static_cast<size_t>(idx) * node_dim_;
    double* dst = window_out + static_cast<size_t>(r) * node_dim_;
    if (cfg_.window_anchor == WindowAnchor::kPlan) {
      std::copy(src, src + node_dim_, dst);
    } else {
      ReanchorWindow(src, 1, anchor_pos_, anchor_quat_, state.root_pos, state.root_quat,
                     cfg_.num_joints, cfg_.num_ee_frames, cfg_.blocks, dst);
    }
  }

  // 5. Stash the node served THIS step -- RAW, in its plan's own frame, never re-anchored.
  //    That keeps odometry out of the generator's conditioning; the re-anchored copy above is
  //    what feeds the POLICY, and only the policy.
  //    Computed directly rather than through NodeIndex(0): that counts saturations, and the
  //    stash is not a served OFFSET, so routing it through there would inflate the clamp
  //    statistic with an event that never happened.
  const int cursor_idx = std::min(sched_.cursor(), cfg_.schedule.horizon - 1);
  const double* served = pred_.data() + static_cast<size_t>(cursor_idx) * node_dim_;
  prev_served_node_.assign(served, served + node_dim_);
  has_served_ = true;
  return outcome;
}

}  // namespace g1_deploy
