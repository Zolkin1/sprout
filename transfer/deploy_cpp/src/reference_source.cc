// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/reference_source.h"

#include <algorithm>
#include <stdexcept>

#include "g1_deploy/depth_obs.h"
#include "g1_deploy/policy_runtime.h"

namespace g1_deploy {

// ---------------------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------------------

FmReferenceSource::FmReferenceSource(std::shared_ptr<FmSession> fm,
                                     std::shared_ptr<FmWorker> worker,
                                     std::shared_ptr<FmSampler> sampler,
                                     std::unique_ptr<GenTrackerCore> core,
                                     const PolicyParams& params, int lead_ticks)
    : fm_(std::move(fm)),
      worker_(std::move(worker)),
      sampler_(std::move(sampler)),
      core_(std::move(core)),
      params_(params),
      lead_ticks_(lead_ticks) {}

void FmReferenceSource::Reset() {
  core_->Reset();
  // Re-arming means the next tick again has no plan, so the blocking first collect has to
  // come back with it.
  if (blocker_) blocker_->Rearm();
}

void FmReferenceSource::BuildConditioning(const HardwareReadings& readings) {
  FmConditioning cond;
  cond.extero.assign(static_cast<size_t>(fm_->meta().extero_dim()), 0.0);
  const int channels = fm_->meta().extero_channels;
  if (channels > 1) {
    // The generator reads more than depth, so each pixel carries its position in the root's
    // yaw-aligned frame. That needs the camera pose, which this runtime cannot derive --
    // it has no kinematics and, deliberately, no odometry. Refused rather than defaulted:
    // a wrong pose is a cloud in the wrong place, which looks entirely plausible.
    if (readings.fm_camera_pos.size() != readings.depth_raw.size() ||
        readings.fm_camera_rot.size() != readings.depth_raw.size()) {
      throw std::runtime_error(
          "fm.onnx conditions on " + std::to_string(channels) +
          " channels per pixel, so it needs each camera's pose in the root's yaw-aligned "
          "frame (fm_camera_pos / fm_camera_rot, one per camera in contract order). Build "
          "it from FK to the mount plus the IMU's roll/pitch -- never odometry.");
    }
    // THE GENERATOR'S OWN CAMERAS, not the tracker's.
    //
    // Since `efe93b4e` the policy reads the near camera alone while the generator reads both,
    // so `params_.depth_cameras` is a SUBSET of the rig. Looping it here sized the row from
    // the generator's metadata but FILLED it from the tracker's list, so the far camera's
    // plane stayed 0.0 -- an absent cloud that looks exactly like a plausible one, with
    // nothing raising. Readings arrive in RIG order (see HardwareReadings::depth_raw), which
    // is this list, so the index is direct.
    const std::vector<DepthCameraSpec>& rig = fm_->meta().cameras;
    // Absent rather than wrong: a contract exported before extero_spec existed parses fine and
    // is refused HERE, where the reason is specific, rather than at parse time, where it would
    // also refuse the scan generators and the metadata-only readers that never need a camera.
    if (rig.empty()) {
      throw std::runtime_error(
          "fm.onnx conditions on depth but its fm_meta carries no extero_spec, so nothing here "
          "knows which cameras the generator reads or where they are mounted. The tracker's own "
          "list cannot stand in -- the policy reads the near camera alone and this is the whole "
          "rig. Re-export with scripts/flow_matching/export_fm_onnx.py.");
    }
    if (readings.depth_raw.size() != rig.size()) {
      throw std::runtime_error(
          "the generator conditions on " + std::to_string(rig.size()) +
          " camera(s) (fm_meta.extero_spec) but " + std::to_string(readings.depth_raw.size()) +
          " reading(s) were supplied. Readings are in the GENERATOR's contract order -- the "
          "whole rig -- and the tracker takes its subset of them by name; supplying only the "
          "tracker's cameras is what left a camera reading as zeros.");
    }
    std::vector<CameraGeom> geoms;
    std::vector<std::vector<double>> normalised;
    std::vector<const double*> images, positions, rotations;
    for (size_t c = 0; c < readings.depth_raw.size(); ++c) {
      const CameraGeom g = GeomFromSpec(rig[c]);
      geoms.push_back(g);
      normalised.emplace_back(readings.depth_raw[c].size());
      NormalizeDepthImage(readings.depth_raw[c].data(), readings.depth_raw[c].size(), g.norm,
                          normalised.back().data());
      positions.push_back(readings.fm_camera_pos[c].data());
      rotations.push_back(readings.fm_camera_rot[c].data());
    }
    for (const auto& img : normalised) images.push_back(img.data());
    // The band goes IN, so the image is clamped before the unprojection and the position
    // channels are not clamped at all -- see DepthXyzRows.
    DepthXyzRows(geoms, images, positions, rotations, channels, fm_->meta().extero_clip,
                 cond.extero.data());
  }
  size_t at = 0;
  for (size_t c = 0; channels == 1 && c < readings.depth_raw.size() && at < cond.extero.size();
       ++c) {
    // Same rule on the 1-channel path: the generator's list, never the tracker's subset.
    const CameraGeom g = GeomFromSpec(fm_->meta().cameras.empty() ? params_.depth_cameras[c]
                                                                 : fm_->meta().cameras[c]);
    for (double v : readings.depth_raw[c]) {
      if (at >= cond.extero.size()) break;
      cond.extero[at++] = NormalizeDepth(v, g.norm);
    }
  }
  if (!readings.height_scan.empty() && readings.depth_raw.empty()) {
    for (size_t i = 0; i < readings.height_scan.size() && i < cond.extero.size(); ++i) {
      cond.extero[i] = readings.height_scan[i];
    }
  }
  // The generator's own normalisation is folded into the graph, but the band clamp is not:
  // one non-finite entry would poison the whole conditioning. The band comes from the GRAPH
  // rather than the per-source table (see SanitizeExteroBand).
  //
  // SKIPPED for a widened row, which DepthXyzRows has already clamped in the one place
  // clamping belongs -- the depth image, before the unprojection. Applying it again here
  // would clamp METRES of position with a normalised-depth band, which is the bug this
  // ordering exists to prevent.
  if (channels == 1) {
    SanitizeExteroBand(cond.extero.data(), cond.extero.size(), fm_->meta().extero_clip,
                       cond.extero.data());
  }
  cond.future_cmd.assign(static_cast<size_t>(fm_->meta().f) * fm_->meta().cmd_dim, 0.0);
  for (int f = 0; f < fm_->meta().f; ++f) {
    cond.future_cmd[static_cast<size_t>(f) * 3 + 0] = readings.command.x();
    cond.future_cmd[static_cast<size_t>(f) * 3 + 1] = readings.command.y();
    cond.future_cmd[static_cast<size_t>(f) * 3 + 2] = readings.command.z();
  }
  last_cond_ = cond;  // for the replay harness; see PolicyRuntime::last_conditioning()
  fm_->SetConditioning(cond);}

TickOutcome FmReferenceSource::Step(const HardwareReadings& readings, const CoreState& state,
                                    double* window_out) {
  // The exteroception and future command the generator will be conditioned on. Sampled HERE,
  // i.e. at the request tick, which is what makes them the one thing that is genuinely `lead`
  // ticks old by the time the plan is adopted -- see gen_tracker_core.h.
  BuildConditioning(readings);
  last_outcome_ = core_->Step(state, window_out);
  return last_outcome_;
}

void FmReferenceSource::FillDiagnostics(TickDiagnostics* diag) const {
  const PlanScheduler& s = core_->scheduler();
  diag->outcome = last_outcome_;
  diag->cursor = s.cursor();
  diag->ctrl_step = s.ctrl_step();
  diag->late = s.late();
  diag->late_events = s.late_events();
  diag->offsets_clamped = s.offsets_clamped();
  diag->requests_skipped = worker_ ? worker_->requests_skipped() : 0;
  diag->plans_dropped = worker_ ? worker_->plans_dropped() : 0;
  // Read the pair CONSISTENTLY. The worker writes last_infer_ms_ and then increments
  // infer_count_, so a naive read of one then the other can catch an inference landing
  // between them and pair the OLD duration with the NEW count -- and the controller samples
  // fm_ms_ precisely when the count moves, so that plan's real duration would never be
  // recorded and the previous one would be recorded twice. Re-read the count and only publish
  // the increment once it brackets a stable read; otherwise report the previous count, and the
  // next tick (20 ms later, against a ~240 ms cadence) picks it up.
  const uint64_t before = fm_->infer_count();
  const double ms = fm_->last_infer_ms();
  const uint64_t after = fm_->infer_count();
  diag->fm_infer_ms = ms;
  diag->plans_produced = (before == after) ? after : before;
  diag->plan_id = s.plan_id();
  diag->anchor_pos = core_->anchor_pos();
  diag->anchor_quat = core_->anchor_quat();
  diag->plan_adopted = last_outcome_ == TickOutcome::kAdoptedOnTime ||
                       last_outcome_ == TickOutcome::kAdoptedLate;
  diag->lead_ticks = lead_ticks_;
}

// ---------------------------------------------------------------------------------------
// A baked clip
// ---------------------------------------------------------------------------------------

ClipReferenceSource::ClipReferenceSource(ClipReference clip, std::vector<int> offsets,
                                         int num_joints, int num_ee_frames,
                                         std::vector<ReferenceBlock> blocks, double duration_sec)
    : clip_(std::move(clip)),
      offsets_(std::move(offsets)),
      num_joints_(num_joints),
      num_ee_frames_(num_ee_frames),
      blocks_(std::move(blocks)) {
  // Last tick that PLAYS. Beyond it the reference holds its last played frame and the command
  // goes to zero -- see Step. Bounded by the clip's own length, so a duration longer than the
  // clip is a no-op rather than an error.
  stop_tick_ = clip_.real_steps() - 1 - clip_.start_step();
  if (duration_sec > 0.0) {
    stop_tick_ = std::min(stop_tick_, static_cast<int>(std::lround(duration_sec / clip_.dt())));
  }
  stop_tick_ = std::max(stop_tick_, 0);
  command_window_.assign(offsets_.size() * 3, 0.0);
}

void ClipReferenceSource::Reset() {
  tick_ = -1;
  exhausted_ = false;
}

TickOutcome ClipReferenceSource::Step(const HardwareReadings& readings, const CoreState& state,
                                      double* window_out) {
  (void)readings;
  ++tick_;
  const int played = std::min(tick_, stop_tick_);
  if (tick_ >= stop_tick_) exhausted_ = true;
  // ONE call, so the clip's re-placement and the window's anchor are solved from the same
  // robot pose on the same tick. That is what makes the served window a pure function of the
  // clip, and therefore what makes this path odometry-free -- see ClipReference::Serve.
  clip_.Serve(played, state.root_pos, state.root_quat, offsets_, num_joints_, num_ee_frames_,
              blocks_, window_out, &command_window_);
  if (exhausted_) {
    // Playback SATURATES on its last played frame rather than stopping, and the padding
    // repeats that frame's authored command -- so a case that ends mid-walk would otherwise
    // hold a walk command against a reference that has stopped moving. Measured on the python
    // side: a backward clip held -0.94 m/s for 16 s after its last frame and put the robot
    // down, which reads as a policy failure and is not one.
    std::fill(command_window_.begin(), command_window_.end(), 0.0);
  }
  return TickOutcome::kAdvanced;
}

void ClipReferenceSource::FillDiagnostics(TickDiagnostics* diag) const {
  diag->outcome = TickOutcome::kAdvanced;
  diag->ctrl_step = tick_ < 0 ? 0 : static_cast<uint64_t>(tick_);
  diag->cursor = -1;                  //: no plan, so no cursor into one
  diag->clip_mode = true;
  diag->clip_step = clip_.StepForTick(std::min(std::max(tick_, 0), stop_tick_));
  diag->anchor_pos = clip_.anchor_pos();
  diag->anchor_quat = clip_.anchor_quat();
  diag->lead_ticks = 0;
}

}  // namespace g1_deploy
