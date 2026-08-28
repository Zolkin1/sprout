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
  void Rearm() override { first_done_ = false; }

 private:
  std::shared_ptr<FmWorker> worker_;
  double timeout_s_;
  bool first_done_ = false;
};

}  // namespace

PolicyRuntime::PolicyRuntime(const RuntimeConfig& cfg) : cfg_(cfg) {}
PolicyRuntime::~PolicyRuntime() = default;

void PolicyRuntime::Load() {
  tracker_ = std::make_unique<TrackerSession>(cfg_.tracker_onnx, cfg_.tracker_providers,
                                              cfg_.tracker_intra_threads);
  const PolicyParams& p = tracker_->params();
  if (!p.reference_window.has_value()) {
    throw std::runtime_error(
        "tracker.onnx observes no reference_window, so there is nothing for the generator to "
        "feed. This runtime builds the merged window layout only.");
  }
  const ReferenceWindowSpec& rw = *p.reference_window;

  tracker_->model().WarmUpAndCheckPlacement();
  // Against the TRACKER's own request, so a deliberate CPU tracker is not refused while a
  // tracker that asked for CUDA and silently did not get it still is. RequireCuda is a no-op
  // when nothing in the list is an accelerator.
  RequireCuda(tracker_->model(), cfg_.tracker_providers);

  std::vector<ReferenceBlock> blocks;
  for (const auto& b : rw.blocks) blocks.push_back(BlockFromName(b.first));
  const int num_joints = static_cast<int>(p.joint_names_isaac.size());
  const int num_ee = static_cast<int>(rw.ee_frame_names.size());

  if (cfg_.fm_onnx.empty()) {
    LoadClipSource(rw, blocks, num_joints, num_ee);
  } else {
    LoadFmSource(rw, blocks, num_ee);
  }
  assembler_ = std::make_unique<ObsAssembler>(p.obs_terms, p.num_obs);
  window_.assign(static_cast<size_t>(rw.offsets.size()) * node_dim_, 0.0);
  obs_.assign(static_cast<size_t>(p.num_obs), 0.0f);
  joint_pos_isaac_.assign(p.joint_names_isaac.size(), 0.0);
  joint_vel_isaac_.assign(p.joint_names_isaac.size(), 0.0);
}

void PolicyRuntime::LoadClipSource(const ReferenceWindowSpec& rw,
                                   const std::vector<ReferenceBlock>& blocks, int num_joints,
                                   int num_ee) {
  const PolicyParams& p = tracker_->params();
  if (cfg_.clip_pack.empty() || cfg_.clip_case.empty()) {
    throw std::runtime_error(
        "no fm.onnx and no clip_pack: there would be no reference at all, and a tracker served "
        "zeros holds a pose that looks like a policy which has given up rather than like a "
        "configuration error. Supply an fm.onnx, or a clip_pack + clip_case (built with "
        "transfer/deploy/tools/dump_clip_pack.py).");
  }
  // The cadence is the ARTIFACT's, not a tuning knob. Under window_anchor="plan" the clip's
  // placement and the window's anchor must be solved from the robot on the same step, or up to
  // K-1 steps of the robot's own egomotion leak into base_pose -- a window matching neither
  // training nor an odometry-free serve, and identical in width to a correct one.
  const WindowAnchor anchor = AnchorFromConvention(rw.base_pose_convention);
  int every = cfg_.clip_ego_anchor_every;
  if (anchor == WindowAnchor::kPlan) {
    if (!p.fm_update_every.has_value()) {
      throw std::runtime_error(
          "tracker.onnx declares window_anchor='plan' but carries no fm_update_every, so the "
          "clip's re-anchor cadence -- which IS the latch period, and therefore part of the "
          "observation contract -- would have to be guessed. Re-export with "
          "scripts/rsl_rl/export_parameters.py from a run that had the generator.");
    }
    if (every != 0 && every != *p.fm_update_every) {
      throw std::runtime_error(
          "clip_ego_anchor_every=" + std::to_string(every) + " but this tracker latches its "
          "window anchor every " + std::to_string(*p.fm_update_every) + " steps. Under "
          "window_anchor='plan' the two must fire on the same step. Leave it 0 to derive it.");
    }
    every = *p.fm_update_every;
  }
  if (every < 1) every = 1;

  ClipReference::Config ccfg;
  ccfg.pack_dir = cfg_.clip_pack;
  ccfg.case_name = cfg_.clip_case;
  ccfg.joint_names = p.joint_names_isaac;
  ccfg.ee_frame_names = rw.ee_frame_names;
  // The base frame is the CLIP's own ref_frame, read from the pack -- the two libraries on
  // disk genuinely differ ("pelvis_link" vs "right_ankle_roll_link"), and restating it here
  // would be a second place for it to be wrong.
  ccfg.base_frame_name.clear();
  ccfg.start_time = cfg_.clip_start_time;
  ccfg.ego_anchor_every = every;
  // A tracking_mode="body" tracker observes body_pos_ref / body_vel_ref, which the clip path
  // does not serve. Detected from the OBS TERMS rather than from a metadata flag, because that
  // is what actually decides it -- ObsAssembler would otherwise fail much later, naming a term
  // rather than the configuration.
  for (const ObsTerm& term : p.obs_terms) {
    if (term.name == "body_pos_ref" || term.name == "body_vel_ref") {
      throw std::runtime_error(
          "this tracker observes '" + term.name + "' (tracking_mode=\"body\"), which the C++ "
          "clip path does not serve. Play it through the python controller "
          "(scripts/run_sim_clip.sh --python), or use a joint-tracking tracker.");
    }
  }
  ClipReference clip = ClipReference::Load(ccfg);
  if (std::abs(clip.dt() - p.dt) > 1e-6) {
    throw std::runtime_error("clip dt " + std::to_string(clip.dt()) + " != policy dt " +
                             std::to_string(p.dt) + ": the motion would play at the wrong rate.");
  }
  node_dim_ = WindowDim(num_joints, num_ee, blocks);
  auto src = std::make_unique<ClipReferenceSource>(std::move(clip), rw.offsets, num_joints,
                                                   num_ee, blocks, cfg_.clip_duration);
  clip_source_ = src.get();
  source_ = std::move(src);
}

void PolicyRuntime::LoadFmSource(const ReferenceWindowSpec& rw,
                                 const std::vector<ReferenceBlock>& blocks, int num_ee) {
  const PolicyParams& p = tracker_->params();
  std::shared_ptr<NoiseSource> noise;
  if (cfg_.zero_noise) {
    noise = std::make_shared<ZeroNoise>();
  } else {
    noise = std::make_shared<GaussianNoise>(cfg_.noise_seed);
  }
  auto fm_ = std::make_shared<FmSession>(cfg_.fm_onnx, cfg_.providers, cfg_.intra_threads,
                                         noise, cfg_.fm_cuda_graph);
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
  int lead_ticks = static_cast<int>(std::ceil(cfg_.fm_lead_sec / p.dt));
  // The floor is 1 unless a caller explicitly asks for zero: a lead of 0 means the plan is
  // requested at the instant it is needed, which no real deployment wants. PlanScheduler
  // handles 0 correctly (ShouldRequest becomes ctrl_step % K == 0, the target is this step,
  // and BuildRequest projects nothing), which is exactly the synchronous behaviour the python
  // runtime has -- and the only configuration in which the two runtimes' PLANS can be
  // compared numerically at all.
  const int floor_ticks = cfg_.allow_zero_lead ? 0 : 1;
  lead_ticks = std::max(floor_ticks, std::min(lead_ticks, k - 1));

  CoreConfig core_cfg;
  core_cfg.num_joints = fm_->meta().num_joints;
  core_cfg.num_ee_frames = num_ee;
  core_cfg.blocks = blocks;
  core_cfg.offsets = rw.offsets;
  core_cfg.history_length = fm_->meta().n;
  core_cfg.cmd_dim = fm_->meta().cmd_dim;
  core_cfg.schedule.replan_period = k;
  core_cfg.schedule.horizon = fm_->meta().f;
  core_cfg.schedule.max_offset = *std::max_element(rw.offsets.begin(), rw.offsets.end());
  core_cfg.schedule.lead_ticks = lead_ticks;
  // Which serve arm, from the artifact's own convention string. Throws on the odometry-
  // dependent arm and on anything unrecognised -- at LOAD, where it can be reported, rather
  // than by quietly serving a window whose base_pose means something else. The two arms have
  // identical widths, so nothing downstream could ever catch this.
  core_cfg.window_anchor = AnchorFromConvention(rw.base_pose_convention);
  // WHICH ROBOT each half was trained on. There is one physical robot, so a tracker and a
  // generator running together cannot have been trained on two -- the same argument as the
  // camera-contract check. Nothing else can catch this: all 29 joint names and their order
  // survive a payload change, a hand swap and a gear-ratio change, so the pair loads clean and
  // the only symptom is a robot that tracks slightly wrong.
  AssertRobotAssetsAgree(p.robot_asset, fm_->meta().robot_asset, cfg_.tracker_onnx,
                         cfg_.fm_onnx, cfg_.allow_robot_asset_mismatch);

  std::shared_ptr<FmWorker> worker;
  std::shared_ptr<FmSampler> blocker, sampler;
  if (cfg_.threaded_fm) {
    worker = std::make_shared<FmWorker>(fm_, static_cast<size_t>(fm_->meta().f) * node_dim_);
    worker->Start();
    blocker = std::make_shared<FirstPlanBlocker>(worker, cfg_.first_plan_timeout_sec);
    sampler = blocker;
  } else {
    // Inline: Submit runs the graph, so a plan is ready within the same Step and no wait is
    // needed. Same code path otherwise.
    sampler = fm_;
  }
  auto core = std::make_unique<GenTrackerCore>(core_cfg, sampler);
  auto src = std::make_unique<FmReferenceSource>(fm_, worker, sampler, std::move(core), p,
                                                 lead_ticks);
  src->set_blocker(blocker);
  fm_source_ = src.get();
  source_ = std::move(src);
}

// The generator-shaped accessors all answer for the CLIP arm too rather than assuming a plan
// exists: a logger runs the same code either way, and "there is no plan" is a fact about the
// run rather than an error to throw at a publisher. fm_meta is the exception -- there is no
// honest empty answer for a metadata block, so it throws.
int PolicyRuntime::replan_period() const {
  return fm_source_ ? fm_source_->replan_period() : 1;
}

int PolicyRuntime::lead_ticks() const { return fm_source_ ? fm_source_->lead_ticks() : 0; }

const std::vector<double>& PolicyRuntime::last_plan_window() const {
  static const std::vector<double> kEmpty;
  return fm_source_ ? fm_source_->plan_window() : kEmpty;
}

const FmRequest& PolicyRuntime::last_plan_request() const {
  static const FmRequest kEmpty;
  return fm_source_ ? fm_source_->plan_request() : kEmpty;
}

const FmConditioning& PolicyRuntime::last_conditioning() const {
  static const FmConditioning kEmpty;
  return fm_source_ ? fm_source_->conditioning() : kEmpty;
}

const FmMeta& PolicyRuntime::fm_meta() const {
  if (!fm_source_) {
    throw std::runtime_error("no fm.onnx is loaded (this is clip playback), so there is no "
                             "generator metadata");
  }
  return fm_source_->meta();
}

void PolicyRuntime::Reset() {
  if (source_) source_->Reset();
  if (assembler_) assembler_->Reset();
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
  // held -- which is what the deployed joystick does anyway. A CLIP authored one per step and
  // supplies them, and holding the current command instead would drive this term with a number
  // the motion never asked for.
  const auto& offsets = tracker_->params().reference_window->offsets;
  if (const std::vector<double>* served = source_->CommandWindow()) {
    if (served->size() != offsets.size() * 3) {
      throw std::runtime_error("the reference source served " + std::to_string(served->size()) +
                               " command values for " + std::to_string(offsets.size()) +
                               " offsets");
    }
    values_["command_window"] = *served;
  } else {
    std::vector<double> cmd_window;
    cmd_window.reserve(offsets.size() * 3);
    for (size_t i = 0; i < offsets.size(); ++i) {
      cmd_window.push_back(readings.command.x());
      cmd_window.push_back(readings.command.y());
      cmd_window.push_back(readings.command.z());
    }
    values_["command_window"] = std::move(cmd_window);
  }

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
  if (!source_) throw std::runtime_error("PolicyRuntime::Load() has not been called");
  if (readings.joint_pos.size() != readings.joint_names.size() ||
      readings.joint_vel.size() != readings.joint_names.size()) {
    throw std::runtime_error("joint_pos / joint_vel / joint_names lengths disagree");
  }

  CoreState cs;
  cs.root_pos = readings.root_pos;
  cs.root_quat = readings.base_quat_xyzw;
  cs.command = {readings.command.x(), readings.command.y(), readings.command.z()};


  source_->Step(readings, cs, window_.data());
  BuildValues(readings);
  assembler_->Assemble(values_, obs_.data());
  tracker_->GetAction(obs_.data(), joint_names_out, targets_out);

  if (diag != nullptr) {
    *diag = TickDiagnostics{};
    source_->FillDiagnostics(diag);
    diag->obs_checksum = ObsChecksum(obs_.data(), obs_.size());
    diag->trk_infer_ms = tracker_->last_infer_ms();
    // Offset 0 of the command block the observation was actually assembled from, whichever
    // source supplied it. Read BACK rather than re-derived, so it cannot disagree with what
    // the network saw.
    const std::vector<double>& cw = values_["command_window"];
    if (cw.size() >= 3) diag->command_effective = Vec3(cw[0], cw[1], cw[2]);
  }
}

}  // namespace g1_deploy
