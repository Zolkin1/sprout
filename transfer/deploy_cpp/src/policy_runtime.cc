// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/policy_runtime.h"

#include <cmath>
#include <stdexcept>

#include "g1_deploy/depth_obs.h"
#include "g1_deploy/fm_worker.h"
#include "g1_deploy/quat_math.h"

namespace g1_deploy {
namespace {

/// Makes the FIRST collect block, so the first armed tick has a reference.
///
/// The request and the collect both happen inside one GenTrackerCore::Step, so the wait has to
/// live between them -- the caller cannot orchestrate it. Wrapping the sampler puts it exactly
/// there and leaves the core untouched.
///
/// Why it matters beyond "the first tick is nicer": the generator's conditioning is
/// autoregressive, so a first tick with no plan does not stay a first-tick problem. Its zero
/// window becomes the next history row and the divergence propagates. The threading test
/// measured this -- 22759 of 23232 served values differed from one absent first plan.
class FirstPlanBlocker : public FmSampler {
 public:
  FirstPlanBlocker(std::shared_ptr<FmWorker> worker, double timeout_s)
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

  /// Re-arm the blocking behaviour for the next arming.
  void Rearm() { first_done_ = false; }

 private:
  std::shared_ptr<FmWorker> worker_;
  double timeout_s_;
  bool first_done_ = false;
};

}  // namespace

PolicyRuntime::PolicyRuntime(const RuntimeConfig& cfg) : cfg_(cfg) {}
PolicyRuntime::~PolicyRuntime() = default;

void PolicyRuntime::Load() {
  tracker_ = std::make_unique<TrackerSession>(cfg_.tracker_onnx, cfg_.providers,
                                              cfg_.intra_threads);
  const PolicyParams& p = tracker_->params();
  if (!p.reference_window.has_value()) {
    throw std::runtime_error(
        "tracker.onnx observes no reference_window, so there is nothing for the generator to "
        "feed. This runtime builds the merged window layout only.");
  }
  const ReferenceWindowSpec& rw = *p.reference_window;

  tracker_->model().WarmUpAndCheckPlacement();
  RequireCuda(tracker_->model(), cfg_.providers);

  if (cfg_.fm_onnx.empty()) {
    throw std::runtime_error(
        "tracker-only (clip) playback is not implemented in the C++ runtime yet: it needs the "
        "HDF5 clip reference reader. Supply an fm.onnx.");
  }

  std::shared_ptr<NoiseSource> noise;
  if (cfg_.zero_noise) {
    noise = std::make_shared<ZeroNoise>();
  } else {
    noise = std::make_shared<GaussianNoise>(cfg_.noise_seed);
  }
  fm_ = std::make_shared<FmSession>(cfg_.fm_onnx, cfg_.providers, cfg_.intra_threads, noise);
  fm_->model().WarmUpAndCheckPlacement();
  RequireCuda(fm_->model(), cfg_.providers);

  // The generator's output IS the tracker's reference_window observation. Refuse a pair that
  // does not name the same blocks in the same order before anything runs.
  AssertWindowBlocksAgree(rw.blocks, fm_->meta().window_blocks);

  node_dim_ = fm_->meta().d;
  if (!p.fm_update_every.has_value()) {
    throw std::runtime_error(
        "tracker.onnx carries no fm_update_every but an fm.onnx was loaded. The re-plan cadence "
        "comes from that metadata and nowhere else; a guessed one is a silent tracking failure. "
        "Re-export with scripts/rsl_rl/export_parameters.py from a run that had the generator.");
  }
  const int k = *p.fm_update_every;

  // Lead time is configured in seconds and resolved against the artifact's dt, so the same
  // config means the same wall-clock lead if dt ever changes. Clamped to [1, K-1]: a request
  // issued before the previous plan was adopted would overwrite one still in flight.
  lead_ticks_ = static_cast<int>(std::ceil(cfg_.fm_lead_sec / p.dt));
  // The floor is 1 unless a caller explicitly asks for zero: a lead of 0 means the plan is
  // requested at the instant it is needed, which no real deployment wants. PlanScheduler
  // handles 0 correctly (ShouldRequest becomes ctrl_step % K == 0, the target is this step,
  // and BuildRequest projects nothing), which is exactly the synchronous behaviour the python
  // runtime has -- and the only configuration in which the two runtimes' PLANS can be
  // compared numerically at all.
  const int floor_ticks = cfg_.allow_zero_lead ? 0 : 1;
  lead_ticks_ = std::max(floor_ticks, std::min(lead_ticks_, k - 1));

  CoreConfig core_cfg;
  core_cfg.num_joints = fm_->meta().num_joints;
  core_cfg.num_ee_frames = static_cast<int>(rw.ee_frame_names.size());
  for (const auto& b : rw.blocks) core_cfg.blocks.push_back(BlockFromName(b.first));
  core_cfg.offsets = rw.offsets;
  core_cfg.history_length = fm_->meta().n;
  core_cfg.cmd_dim = fm_->meta().cmd_dim;
  core_cfg.schedule.replan_period = k;
  core_cfg.schedule.horizon = fm_->meta().f;
  core_cfg.schedule.max_offset = *std::max_element(rw.offsets.begin(), rw.offsets.end());
  core_cfg.schedule.lead_ticks = lead_ticks_;

  if (cfg_.threaded_fm) {
    worker_ = std::make_shared<FmWorker>(fm_, static_cast<size_t>(fm_->meta().f) * node_dim_);
    worker_->Start();
    blocker_ = std::make_shared<FirstPlanBlocker>(worker_, cfg_.first_plan_timeout_sec);
    sampler_ = blocker_;
  } else {
    // Inline: Submit runs the graph, so a plan is ready within the same Step and no wait is
    // needed. Same code path otherwise.
    sampler_ = fm_;
  }
  core_ = std::make_unique<GenTrackerCore>(core_cfg, sampler_);
  assembler_ = std::make_unique<ObsAssembler>(p.obs_terms, p.num_obs);

  window_.assign(static_cast<size_t>(rw.offsets.size()) * node_dim_, 0.0);
  obs_.assign(static_cast<size_t>(p.num_obs), 0.0f);
  joint_pos_isaac_.assign(p.joint_names_isaac.size(), 0.0);
  joint_vel_isaac_.assign(p.joint_names_isaac.size(), 0.0);
}

int PolicyRuntime::replan_period() const {
  return core_ ? core_->scheduler().config().replan_period : 1;
}

const std::vector<double>& PolicyRuntime::last_plan_window() const {
  static const std::vector<double> kEmpty;
  return core_ ? core_->predicted_window() : kEmpty;
}

const FmRequest& PolicyRuntime::last_plan_request() const {
  static const FmRequest kEmpty;
  return core_ ? core_->adopted_request() : kEmpty;
}

const FmMeta& PolicyRuntime::fm_meta() const {
  if (!fm_) throw std::runtime_error("no fm.onnx is loaded, so there is no generator metadata");
  return fm_->meta();
}

void PolicyRuntime::Reset() {
  if (core_) core_->Reset();
  if (assembler_) assembler_->Reset();
  // Re-arming means the next tick again has no plan, so the blocking first collect has to come
  // back with it.
  if (blocker_) static_cast<FirstPlanBlocker*>(blocker_.get())->Rearm();
}

void PolicyRuntime::BuildValues(const HardwareReadings& readings) {
  const PolicyParams& p = tracker_->params();

  // Joint reordering, resolved once. The robot's order is per limb and the policy's is
  // interleaved; 27 of 29 positions differ on the G1, so mixing them produces a large, smooth,
  // entirely fictitious error rather than an obvious failure.
  if (readings.joint_names != cached_joint_names_) {
    joint_permutation_.clear();
    for (const std::string& want : p.joint_names_isaac) {
      int found = -1;
      for (size_t i = 0; i < readings.joint_names.size(); ++i) {
        if (readings.joint_names[i] == want) {
          found = static_cast<int>(i);
          break;
        }
      }
      if (found < 0) {
        throw std::runtime_error("the policy expects joint '" + want +
                                 "' but the robot does not report it; the two disagree about "
                                 "the joint set");
      }
      joint_permutation_.push_back(found);
    }
    cached_joint_names_ = readings.joint_names;
  }
  for (size_t i = 0; i < joint_permutation_.size(); ++i) {
    const size_t src = static_cast<size_t>(joint_permutation_[i]);
    joint_pos_isaac_[i] = readings.joint_pos[src];
    joint_vel_isaac_[i] = readings.joint_vel[src];
  }

  values_.clear();
  values_["base_ang_vel"] = {readings.base_ang_vel_b.x(), readings.base_ang_vel_b.y(),
                             readings.base_ang_vel_b.z()};
  // Gravity in the base frame. quat_apply of the INVERSE, not the forward rotation: this is a
  // world vector expressed in the body, and the two differ by a sign pattern that no shape
  // check sees.
  const Vec3 g = QuatApply(QuatInv(readings.base_quat_xyzw), Vec3(0.0, 0.0, -1.0));
  values_["projected_gravity"] = {g.x(), g.y(), g.z()};

  // joint_pos is relative to the default pose; joint_vel is absolute.
  std::vector<double> jp(joint_pos_isaac_.size());
  for (size_t i = 0; i < jp.size(); ++i) {
    jp[i] = joint_pos_isaac_[i] - (p.default_joint_angles.empty()
                                       ? 0.0
                                       : p.default_joint_angles[i]);
  }
  values_["joint_pos"] = std::move(jp);
  values_["joint_vel"] = joint_vel_isaac_;

  // The previous tick's RAW network output, which is what the term means -- unless a replay
  // supplied it, in which case this tick stops depending on the last one. See
  // HardwareReadings::forced_prev_action.
  if (!readings.forced_prev_action.empty()) {
    values_["actions"] = readings.forced_prev_action;
  } else {
    std::vector<double> actions(tracker_->last_action().size());
    for (size_t i = 0; i < actions.size(); ++i) {
      actions[i] = static_cast<double>(tracker_->last_action()[i]);
    }
    values_["actions"] = std::move(actions);
  }

  values_["reference_window"] =
      readings.forced_window.empty() ? window_ : readings.forced_window;

  // The joystick-level command over the same offsets. Deliberately separate from the reference
  // window so it stays clean: the generator has no per-step command, so the current one is
  // held -- which is what the deployed joystick does anyway.
  const auto& offsets = tracker_->params().reference_window->offsets;
  std::vector<double> cmd_window;
  cmd_window.reserve(offsets.size() * 3);
  for (size_t i = 0; i < offsets.size(); ++i) {
    cmd_window.push_back(readings.command.x());
    cmd_window.push_back(readings.command.y());
    cmd_window.push_back(readings.command.z());
  }
  values_["command_window"] = std::move(cmd_window);

  // Exteroception. Required-ness comes from the ARTIFACTS, not from the caller: a policy that
  // reads terrain and is handed none would otherwise fail deep inside the observation assembly.
  if (!p.depth_cameras.empty()) {
    if (readings.depth_raw.size() != p.depth_cameras.size()) {
      throw std::runtime_error(
          "this policy reads " + std::to_string(p.depth_cameras.size()) +
          " depth camera(s) but " + std::to_string(readings.depth_raw.size()) +
          " reading(s) were supplied (metres, one per camera, in contract order)");
    }
    for (size_t c = 0; c < p.depth_cameras.size(); ++c) {
      const DepthCameraSpec& cam = p.depth_cameras[c];
      const std::vector<double>& raw = readings.depth_raw[c];
      if (static_cast<int>(raw.size()) != cam.pixels()) {
        throw std::runtime_error("camera '" + cam.obs_term + "' expects " +
                                 std::to_string(cam.pixels()) + " pixels, got " +
                                 std::to_string(raw.size()));
      }
      DepthNorm n;
      n.offset = cam.offset;
      n.scale = cam.scale;
      n.min_dist = cam.min_dist;
      n.max_dist = cam.max_dist;
      n.min_clip_val = cam.min_clip_val;
      n.max_clip_val = cam.max_clip_val;
      std::vector<double> normalised(raw.size());
      NormalizeDepthImage(raw.data(), raw.size(), n, normalised.data());
      values_[cam.obs_term] = std::move(normalised);
    }
  } else if (!readings.depth_raw.empty()) {
    throw std::runtime_error("depth readings supplied but this policy has no depth cameras");
  }

  if (p.needs_height_scan) {
    if (readings.height_scan.empty()) {
      throw std::runtime_error(
          "this policy reads a height scan; supply per-ray ground clearance in metres");
    }
    values_["height_scan"] = readings.height_scan;
  }
}

void PolicyRuntime::Step(const HardwareReadings& readings,
                         const std::vector<std::string>& joint_names_out, double* targets_out,
                         TickDiagnostics* diag) {
  if (!core_) throw std::runtime_error("PolicyRuntime::Load() has not been called");
  if (readings.joint_pos.size() != readings.joint_names.size() ||
      readings.joint_vel.size() != readings.joint_names.size()) {
    throw std::runtime_error("joint_pos / joint_vel / joint_names lengths disagree");
  }

  CoreState cs;
  cs.root_pos = readings.root_pos;
  cs.root_quat = readings.base_quat_xyzw;
  cs.command = {readings.command.x(), readings.command.y(), readings.command.z()};

  // The exteroception and future command the generator will be conditioned on. Sampled HERE,
  // i.e. at the request tick, which is what makes them the one thing that is genuinely `lead`
  // ticks old by the time the plan is adopted -- see gen_tracker_core.h.
  if (fm_) {
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
      std::vector<CameraGeom> geoms;
      std::vector<std::vector<double>> normalised;
      std::vector<const double*> images, positions, rotations;
      for (size_t c = 0; c < readings.depth_raw.size(); ++c) {
        const CameraGeom g = GeomFromSpec(tracker_->params().depth_cameras[c]);
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
      const CameraGeom g = GeomFromSpec(tracker_->params().depth_cameras[c]);
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
    last_cond_ = cond;  // for the replay harness; see last_conditioning()
    fm_->SetConditioning(cond);
  }

  const TickOutcome outcome = core_->Step(cs, window_.data());
  BuildValues(readings);
  assembler_->Assemble(values_, obs_.data());
  tracker_->GetAction(obs_.data(), joint_names_out, targets_out);

  if (diag != nullptr) {
    const PlanScheduler& s = core_->scheduler();
    diag->outcome = outcome;
    diag->obs_checksum = ObsChecksum(obs_.data(), obs_.size());
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
    // recorded and the previous one would be recorded twice. Re-read the count and only
    // publish the increment once it brackets a stable read; otherwise report the previous
    // count, and the next tick (20 ms later, against a ~240 ms cadence) picks it up.
    if (fm_) {
      const uint64_t before = fm_->infer_count();
      const double ms = fm_->last_infer_ms();
      const uint64_t after = fm_->infer_count();
      diag->fm_infer_ms = ms;
      diag->plans_produced = (before == after) ? after : before;
    } else {
      diag->fm_infer_ms = 0.0;
      diag->plans_produced = 0;
    }
    diag->trk_infer_ms = tracker_->last_infer_ms();
    diag->plan_id = s.plan_id();
    diag->anchor_pos = core_->anchor_pos();
    diag->anchor_quat = core_->anchor_quat();
    diag->plan_adopted =
        outcome == TickOutcome::kAdoptedOnTime || outcome == TickOutcome::kAdoptedLate;
    diag->lead_ticks = lead_ticks_;
  }
}

}  // namespace g1_deploy
