// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// The whole policy, from robot readings to joint targets, in one object.
//
// Ties together what the earlier pieces do separately: GenTrackerCore (the receding-horizon
// schedule and the served reference window), ObsAssembler (the observation vector) and
// TrackerSession (the network and the action). Port of GenTrackerPolicy.create_obs +
// get_action, plus deploy/hardware.build_hardware_state.
//
// It exists so the ROS node does not have to wire those together. The node's job is ROS: read
// topics, latch them, call Step, publish, and enforce the safety rules. Anything that could be
// tested without ROS lives here, and is.

#include <map>
#include <memory>
#include <array>
#include <string>
#include <vector>

#include "g1_deploy/fm_session.h"
#include "g1_deploy/fm_worker.h"
#include "g1_deploy/gen_tracker_core.h"
#include "g1_deploy/obs_assembler.h"
#include "g1_deploy/tracker_session.h"
#include "g1_deploy/types.h"

namespace g1_deploy {

/// One tick of robot readings, in the ROBOT's joint order.
struct HardwareReadings {
  Vec3 root_pos = Vec3::Zero();
  QuatXyzw base_quat_xyzw{0.0, 0.0, 0.0, 1.0};
  //: Angular velocity in the ROOT LINK's frame -- what a gyro measures, and exactly the
  //: `base_ang_vel` obs term (trained as isaaclab's root_ang_vel_b). It tilts with the torso:
  //: not gravity-aligned and not heading-aligned, unlike the reference window's base_vel
  //: block, which is in the anchor's YAW-ALIGNED frame. Two different frames, deliberately.
  Vec3 base_ang_vel_b = Vec3::Zero();
  std::vector<double> joint_pos;   //: in joint_names order
  std::vector<double> joint_vel;
  std::vector<std::string> joint_names;
  Vec3 command = Vec3::Zero();     //: (vx, vy, wz), post-watchdog
  //: RAW depth in metres along the optical axis, one reading per camera, in CONTRACT order.
  //: Normalised here, so nothing upstream restates a camera number.
  std::vector<std::vector<double>> depth_raw;
  //: Per-ray ground clearance, for a height-scan policy.
  std::vector<double> height_scan;
  //: Each camera's pose in the ROOT'S YAW-ALIGNED frame, in CONTRACT order: position, then a
  //: ROW-major 3x3. Required once the generator conditions on more than depth, because the
  //: xyz channels are the image unprojected into that frame and this runtime cannot derive
  //: the pose itself -- it has no kinematics.
  //:
  //: Build it from FK to the camera mount plus the IMU's roll/pitch, and NEVER from odometry:
  //: the frame's yaw and the root position both cancel out of it, which is exactly why a
  //: drifting yaw estimate cannot corrupt the cloud. The nominal mount offset from the
  //: contract, not a calibration draw -- see DepthCameraSpec::offset_pos.
  std::vector<std::array<double, 3>> fm_camera_pos;
  std::vector<std::array<double, 9>> fm_camera_rot;

  // ---- teacher forcing, for the replay harness (tools/replay.cc) --------------------
  //
  // Both empty in normal operation, and the runtime then behaves exactly as before.
  //
  // A replay fed only the sensor readings is AUTOREGRESSIVE through two values, however
  // teacher-forced it looks: the served window is the generator's own output, and the
  // `actions` observation term is the tracker's own previous output. With either coming from
  // the runtime, a divergence at tick k makes every later tick a consequence rather than an
  // independent sample -- so a parity report can only name the first failure, not a rate.
  //
  // Supplying both from a recording makes each tick independent, which is what lets the
  // comparison report a distribution and localise a fault to (say) "only re-plan ticks".
  //: The reference window to observe, instead of the one this runtime would serve.
  std::vector<double> forced_window;
  //: The previous raw action for the `actions` term, instead of tracker_->last_action().
  std::vector<double> forced_prev_action;
};

/// How to load the pair.
struct RuntimeConfig {
  std::string tracker_onnx;
  std::string fm_onnx;                //: empty for tracker-only (clip) playback
  std::vector<Provider> providers{Provider::kCpu};
  int intra_threads = 0;
  bool zero_noise = false;
  uint64_t noise_seed = 0;
  //: How far before its cadence boundary a plan is requested, in SECONDS. Converted to ticks
  //: against the artifact's dt.
  double fm_lead_sec = 0.05;
  //: Run the generator on its own thread. False runs it inline, which is what the tests and
  //: the benchmark do; on this hardware the threaded path is what meets the control period.
  bool threaded_fm = true;
  //: Replay a captured CUDA graph for the GENERATOR instead of launching its ~3000 kernels
  //: individually. Measured 19.3 -> 14-15 ms per re-plan on the Thor, which is the difference
  //: between a plan that misses its 20 ms control period and one that makes it.
  //:
  //: Off by default because it is not universally available: it needs a CUDA provider AND an
  //: fm.onnx exported at a fixed batch. Both are checked at load and refused loudly rather
  //: than degraded silently. The tracker is left alone -- it is 17 nodes and 0.2 ms, so there
  //: is nothing to win and a second capture would only add a failure mode.
  bool fm_cuda_graph = false;
  //: Allow fm_lead_sec = 0 to mean "request the plan AT its boundary", the way the
  //: synchronous python runtime does.
  //:
  //: Off by default and deliberately not a tuning knob: the lead exists because the real
  //: runtime needs a whole window to do ~16 ms of work in, and a robot running with zero lead
  //: would be asking the generator for a plan it needs this instant. It is here so the replay
  //: harness can compare the two runtimes' PLANS at all -- with any lead >= 1 the C++
  //: conditions each plan on exteroception from earlier than python does, so the two are
  //: different policies and a numeric diff of their plans measures the lead rather than the
  //: code.
  bool allow_zero_lead = false;
  //: How long the FIRST tick after arming may block waiting for a plan.
  //:
  //: There is no PD warm-up to hide behind, so the alternative is a tick with no reference at
  //: all. This is the only blocking path in the runtime; every later tick is non-blocking by
  //: construction, because a plan is requested a whole lead window before it is needed.
  double first_plan_timeout_sec = 0.2;
};

/// Everything a logger wants that is not the command itself.
struct TickDiagnostics {
  TickOutcome outcome = TickOutcome::kNoPlan;
  uint64_t obs_checksum = 0;
  int cursor = 0;
  int64_t ctrl_step = -1;
  bool late = false;
  uint32_t late_events = 0;
  uint32_t offsets_clamped = 0;
  uint32_t requests_skipped = 0;
  uint32_t plans_dropped = 0;
  //: Wall time of the generator's last inference. CUMULATIVE in the sense that it does not
  //: change between plans -- sample it only when `plans_produced` moves, or the same number
  //: is counted K times over.
  double fm_infer_ms = 0.0;
  //: Wall time of the tracker's last inference. Fresh every tick, unlike fm_infer_ms.
  double trk_infer_ms = 0.0;
  //: Plans the generator has FINISHED, ever. Produced != adopted: the mailbox drops a plan
  //: superseded before the tick read it, and the scheduler refuses one that arrived too stale.
  uint64_t plans_produced = 0;
  //: Id of the live plan, from PlanScheduler. Changes only on adoption.
  uint64_t plan_id = 0;
  //: The measured base pose the LIVE plan is expressed relative to, captured at its adoption.
  //: Yaw-only in effect -- the window's base_pose block uses only the anchor's heading -- but
  //: carried in full so a viewer can lift the plan back to world without restating that.
  Vec3 anchor_pos = Vec3::Zero();
  QuatXyzw anchor_quat{0.0, 0.0, 0.0, 1.0};
  //: This tick adopted a new plan. The publish trigger for FmPlan.
  bool plan_adopted = false;
  //: The lead, in ticks. `lead_ticks * dt` is the generator's actual deadline: it is asked
  //: that far before the boundary its plan is for, and being slower than that is what `late`
  //: eventually reports.
  int lead_ticks = 0;
};

class PolicyRuntime {
 public:
  explicit PolicyRuntime(const RuntimeConfig& cfg);
  ~PolicyRuntime();

  /// Load both graphs, check the pair agrees, warm up, and verify GPU placement.
  ///
  /// Separate from the constructor because it is slow (~10 s for the ONNX pair) and can throw
  /// for contract reasons the caller wants to report rather than crash on.
  void Load();

  /// Clear the schedule, the histories and the observation windows. Call on arming.
  void Reset();

  /// One control step: readings in, joint position targets out in `joint_names_out` order.
  ///
  /// Throws std::runtime_error on a contract violation (a missing exteroception reading, an
  /// unknown joint, a term the tracker wants and nothing supplies).
  void Step(const HardwareReadings& readings, const std::vector<std::string>& joint_names_out,
            double* targets_out, TickDiagnostics* diag);

  const PolicyParams& params() const { return tracker_->params(); }
  int num_obs() const { return tracker_->params().num_obs; }
  int num_actions() const { return tracker_->params().num_actions; }
  double dt() const { return tracker_->params().dt; }
  /// The served reference window, re-anchored -- the most diagnostic quantity in the loop.
  const std::vector<double>& last_reference_window() const { return window_; }
  const std::vector<float>& last_obs() const { return assembler_->last_obs(); }
  /// The raw network output, in the POLICY's joint order.
  const std::vector<float>& last_action() const { return tracker_->last_action(); }
  int lead_ticks() const { return lead_ticks_; }
  /// K, the generator's re-plan period in control ticks. From tracker.onnx metadata only.
  int replan_period() const;

  /// The live plan (F x D) and the request it came from, for a logger. See GenTrackerCore.
  const std::vector<double>& last_plan_window() const;
  const FmRequest& last_plan_request() const;
  /// The exteroception and future command the last request was conditioned on.
  ///
  /// For the replay harness. The plan is a function of the history, THESE, and the noise, so
  /// when two runtimes' plans differ and their history and noise agree, this is where the
  /// difference is -- and it is pure arithmetic on both sides, so it should be exact.
  const FmConditioning& last_conditioning() const { return last_cond_; }

  /// The generator's own metadata: F, D, N, cmd_dim, extero_source. Throws with no fm.onnx.
  const FmMeta& fm_meta() const;

 private:
  void BuildValues(const HardwareReadings& readings);

  RuntimeConfig cfg_;
  std::unique_ptr<TrackerSession> tracker_;
  std::shared_ptr<FmSession> fm_;
  std::shared_ptr<FmWorker> worker_;
  std::shared_ptr<FmSampler> blocker_;   //: wraps the worker to block the first collect
  std::shared_ptr<FmSampler> sampler_;   //: what the core calls: blocker_, or fm_ inline
  std::unique_ptr<GenTrackerCore> core_;
  std::unique_ptr<ObsAssembler> assembler_;

  int lead_ticks_ = 1;
  int node_dim_ = 0;
  std::vector<double> window_;                 //: offsets x node_dim, re-anchored
  std::map<std::string, std::vector<double>> values_;
  std::vector<float> obs_;
  std::vector<double> joint_pos_isaac_;
  std::vector<double> joint_vel_isaac_;
  std::vector<int> joint_permutation_;         //: robot order -> policy order
  std::vector<std::string> cached_joint_names_;
  FmConditioning last_cond_;              //: what the last request was conditioned on
};

}  // namespace g1_deploy
