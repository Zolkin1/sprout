// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// The G1 low-level controller, in C++.
//
// Port of g1_control/g1_control/controller.py. The POLICY is not implemented here --
// g1_deploy::PolicyRuntime is, and everything numeric lives there where it can be tested
// against the python oracle with no ROS. This file is the ROS half: read topics, latch them,
// call the runtime once per tick, publish, and enforce the safety rules.
//
// ============================ why C++ at all ==============================================
// ROS 2 Humble pins the image to python 3.10, so rclpy is a cp310 C extension, and the only
// onnxruntime GPU build that runs on a Jetson Thor is cp312. Two C extensions built for
// different CPython ABIs cannot share a process, so the python controller can only reach the
// CPU here -- measured 471 ms per generator tick against a 20 ms control period. C++ has no
// such constraint.
//
// ============================ the safety rules are the point ==============================
// Everything below the control loop is there because it has caught something. Port the
// behaviour AND the reasons; a guard whose rationale was dropped is a guard someone
// simplifies away.

#include <obelisk_controller.h>
#include <obelisk_control_msgs/msg/pd_feed_forward.hpp>
#include <obelisk_estimator_msgs/msg/estimated_state.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <g1_msgs/msg/controller_debug.hpp>
#include <g1_msgs/msg/fm_plan.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "g1_deploy/camera_fk.h"
#include "g1_deploy/imu_hold.h"
#include "g1_deploy/pose_history.h"
#include "g1_deploy/policy_runtime.h"

namespace g1_control {

using ControlMsg = obelisk_control_msgs::msg::PDFeedForward;
using EstimatorMsg = obelisk_estimator_msgs::msg::EstimatedState;

/// Samples of one timing quantity over one report window.
///
/// Windowed, not a sliding ring: every report is over a disjoint interval, so two consecutive
/// lines describe two consecutive stretches of the run rather than overlapping ones. That
/// matters when chasing a transient -- a ring smears the bad five seconds into the next.
struct RollingStat {
  //: Hard cap on one window's samples. A report closes every few seconds at 50 Hz, so this is
  //: never reached in normal operation -- it only bounds the pathological case where the tick
  //: keeps sampling but never reaches the report (armed, but bailing out early every tick).
  static constexpr size_t kMaxSamples = 4096;

  std::vector<double> samples;

  void Add(double v) {
    if (samples.size() < kMaxSamples) samples.push_back(v);
  }
  void Clear() { samples.clear(); }
  bool empty() const { return samples.empty(); }
  size_t size() const { return samples.size(); }
  /// Nearest-rank percentile, `q` in [0, 1]. Sorts in place; call after the window closes.
  double Percentile(double q);
};

/// One estimator sample, latched as a UNIT.
///
/// Bundled deliberately. The estimator runs at 1 kHz against a 50 Hz tick, so a tick that read
/// the pose from message n and the joints from n+1 would mix two instants -- and
/// ControllerDebug.est_state_stamp, which exists to answer "which sample fed this tick", would
/// become a lie. The python got this for free from the GIL plus obelisk's single
/// mutually-exclusive callback group; C++ has to say it.
struct EstimatorSample {
  std::vector<double> q_base;      //: 7: position(3) + quaternion(4)
  std::vector<double> v_base;      //: 6: linear(3) + angular(3)
  std::vector<double> joint_pos;
  std::vector<double> joint_vel;
  std::vector<std::string> joint_names;
  rclcpp::Time stamp;
  bool valid = false;
};

class G1Controller : public obelisk::ObeliskController<ControlMsg, EstimatorMsg> {
 public:
  explicit G1Controller(const std::string& name);

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& prev) override;
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn on_cleanup(
      const rclcpp_lifecycle::State& prev) override;

 protected:
  /// The 50 Hz control tick. Publishes the command itself, per the obelisk contract.
  ControlMsg ComputeControl() override;

  /// Latch the state estimate. Runs at the estimator's rate, not the control rate.
  void UpdateXHat(const EstimatorMsg& msg) override;

 private:
  void DeclareParams();
  void LoadPolicy();
  /// Build the camera FK, if this generator conditions on more than depth. No-op otherwise.
  void SetupCameraFk();
  void SetupDepth();
  void DepthCallback(const sensor_msgs::msg::Image::SharedPtr msg, size_t index);

  void Arm();
  void Disarm(const std::string& why);
  /// Refresh the staleness latches, logging only the EDGES. Returns true if anything is stale.
  bool UpdateWatchdogs();
  /// The command after the watchdogs: zeros when the joystick has gone quiet.
  g1_deploy::Vec3 EffectiveCommand();
  /// Base orientation AND angular velocity from one estimator sample, with the
  /// degenerate-quaternion guard applied to both.
  ///
  /// Takes the sample the tick already copied rather than re-reading `est_`: the two must be
  /// the same sample or the guard is incoherent (it would test one message's quaternion and
  /// hold another message's gyro), and EstimatorSample exists precisely to be copied as a
  /// unit.
  bool BaseImu(const EstimatorSample& est, g1_deploy::QuatXyzw* quat,
               g1_deploy::Vec3* ang_vel);
  /// Append one estimator sample to the pose history. Call with `est_mutex_` held.
  void PushPoseSample(const EstimatorSample& est);
  /// Publish, unless the command is not finite and there is no finite fallback.
  bool PublishTargets(std::vector<double>* targets, ControlMsg* msg);
  void PublishDebug(const g1_deploy::TickDiagnostics& diag, bool have_obs);
  /// Publish the plan the tick just adopted, raw, with the conditioning it came from.
  void PublishFmPlan(const g1_deploy::TickDiagnostics& diag);
  std::string DiagnoseNonFinite() const;
  /// Fold this tick into the timing window, and emit the report line when it closes.
  void RecordTiming(const g1_deploy::TickDiagnostics& diag, double tick_ms);

  // ---- policy -------------------------------------------------------------------------
  std::unique_ptr<g1_deploy::PolicyRuntime> policy_;
  std::vector<std::string> hw_joint_names_;
  std::vector<double> targets_;
  std::vector<double> kp_hw_;
  std::vector<double> kd_hw_;
  std::vector<int> policy_to_hw_;   //: hardware index -> policy index, for the gains

  // ---- camera kinematics --------------------------------------------------------------
  //: Null unless the generator reads more than depth per pixel. See SetupCameraFk.
  std::unique_ptr<g1_deploy::CameraPoseFk> camera_fk_;
  //: Encoder angles in the POLICY's order, which is what CameraPoseFk was built against. The
  //: estimator speaks the robot's, and the two differ on 27 of the 29.
  std::vector<double> fk_joint_pos_;
  std::vector<std::array<double, 3>> fk_cam_pos_;
  std::vector<std::array<double, 9>> fk_cam_rot_;

  // ---- latched inputs -----------------------------------------------------------------
  std::mutex est_mutex_;
  EstimatorSample est_;            //: guarded by est_mutex_
  double est_time_ = -1.0;

  std::vector<std::vector<double>> depth_;        //: per camera, metres
  std::vector<double> depth_time_;
  //: The consumed frame's own CAPTURE stamp (header), seconds, or <0 if never published. Kept
  //: beside the receipt time rather than replacing it: the watchdog wants receipt (a skewed
  //: sender must not look alive), and the age-from-capture diagnostic wants this.
  std::vector<double> depth_stamp_;
  std::mutex depth_mutex_;

  g1_deploy::Vec3 vel_cmd_{0.0, 0.0, 0.0};
  double vel_cmd_time_ = -1.0;
  std::mutex cmd_mutex_;

  std::atomic<bool> armed_{false};

  // ---- guards -------------------------------------------------------------------------
  //: Ticks of consecutive non-finite commands before disarming. 10 at 50 Hz = 0.2 s.
  static constexpr int kNonFiniteDisarmTicks = 10;
  int nonfinite_streak_ = 0;
  std::vector<double> last_good_targets_;
  bool have_last_good_ = false;
  std::vector<double> hold_pose_;                 //: last-resort fallback, captured at arming
  //: Hold-last-valid for the base orientation AND gyro, for a robot whose estimator
  //: intermittently publishes an all-zero message. The logic and the reasoning live in
  //: g1_deploy::ImuHold, where they are tested without ROS; this node owns only the logging
  //: and the counters' lifetime. See imu_hold.h.
  g1_deploy::ImuHold imu_hold_;
  //: Whether the CURRENT tick ran on held values, for the debug tick.
  bool quat_held_ = false;

  // ---- pose history, for matching the camera pose to the image's capture time -------------
  //: Recent base poses, so the depth image can be unprojected with the pose it was CAPTURED
  //: with rather than the one from the tick that consumes it. The ring and its lookup live in
  //: g1_deploy::PoseHistory, where they are tested without ROS -- every failure mode there is
  //: a plausible cloud in the wrong place, so none of it is visible in the output.
  //:
  //: Guarded by `est_mutex_`, which the 1 kHz estimator callback already holds.
  std::unique_ptr<g1_deploy::PoseHistory> pose_hist_;
  double pose_history_sec_ = 0.2;
  bool depth_pose_match_ = true;
  //: Frames that used the LIVE pose because their stamp was unusable (absent, in the future,
  //: or older than the history -- all of which mean the camera box's clock is not the brain's).
  uint64_t depth_pose_fallbacks_ = 0;
  //: Lookups that ran off the end of the history and took its closest end instead.
  uint64_t depth_pose_clamped_ = 0;
  //: Per camera, the STAMP of the pose sample actually used, or <0 for the live-pose fallback.
  //: A stamp rather than an age, so the debug tick can age it against the same instant it ages
  //: the capture stamp -- the two are compared to each other, and computing them a millisecond
  //: apart put a 30 ms tail on a difference that is really sub-millisecond.
  std::vector<double> depth_pose_stamp_;
  //: The capture stamps the TICK consumed, snapshotted with the images. The live member can
  //: have moved on to a newer frame by the time the debug message is built, and reporting that
  //: one would describe a frame the pose was never matched to.
  std::vector<double> depth_stamp_used_;
  bool depth_stale_ = false;
  bool vel_cmd_stale_ = false;
  //: Set when a tick's inputs were non-finite, for DiagnoseNonFinite.
  //: Split, because the halves differ in consequence -- see DiagnoseNonFinite. The
  //: orientation is an observation; the position is not read by the policy on the plan arm.
  bool bad_base_quat_ = false;
  bool bad_base_pos_ = false;
  bool bad_v_base_ = false;
  bool bad_joints_ = false;
  bool bad_depth_ = false;

  // ---- params -------------------------------------------------------------------------
  std::string hf_repo_id_;
  std::string hf_policy_folder_;
  bool use_cuda_ = false;
  int onnx_threads_ = 2;
  std::vector<std::string> depth_topics_;
  double depth_timeout_ = 0.25;
  double vel_cmd_timeout_ = 1.0;
  bool verbose_ = false;
  bool debug_topics_ = true;
  double fm_lead_sec_ = 0.05;
  bool threaded_fm_ = true;
  //: A baked clip drives the tracker and no generator is loaded. Set in LoadPolicy from
  //: debug_clip_pack; the tick reads it only to decide what to publish, never what to compute.
  bool clip_mode_ = false;
  bool fm_cuda_graph_ = false;
  bool tracker_on_cpu_ = true;
  int tracker_threads_ = 2;
  int fm_late_disarm_ticks_ = 0;
  std::string urdf_path_;
  bool zero_noise_ = true;
  //: Arm a pair whose robot_asset fingerprints disagree. Off by default; see
  //: PolicyRuntime::Config::allow_robot_asset_mismatch for what it is and is not for.
  bool allow_robot_asset_mismatch_ = false;

  // ---- diagnostics --------------------------------------------------------------------
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> depth_subs_;
  rclcpp::Publisher<g1_msgs::msg::ControllerDebug>::SharedPtr debug_pub_;
  rclcpp::Publisher<g1_msgs::msg::FmPlan>::SharedPtr fm_plan_pub_;
  rclcpp::CallbackGroup::SharedPtr depth_cbg_;
  int64_t fm_late_streak_ = 0;
  //: Last plan id published, so exactly one FmPlan goes out per plan the robot actually used.
  uint64_t last_fm_plan_id_ = 0;

  // ---- timing -------------------------------------------------------------------------
  //: Whole tick: t0 to after the command is published. Budget: the control period.
  RollingStat tick_ms_;
  //: Wall time between consecutive ticks -- the ROS timer's own jitter, which nothing else
  //: measures. A control loop that is not actually running at 50 Hz makes every other number
  //: on this line, and every reference the generator serves, mean something different.
  RollingStat period_ms_;
  //: tracker.onnx, every tick. Part of the tick budget.
  RollingStat trk_ms_;
  //: fm.onnx, ONE sample per inference. Budget: lead_ticks * dt, because the generator runs
  //: off the control thread and is asked for its plan a whole lead window before the boundary.
  RollingStat fm_ms_;
  //: Last tick's entry time, for period_ms_. Zero-initialised means "no previous tick".
  std::chrono::steady_clock::time_point last_tick_at_{};
  bool have_last_tick_ = false;
  //: Snapshots of the CUMULATIVE counters at the last report, so the line prints per-window
  //: deltas. A cumulative count says a problem happened; a delta says it is happening.
  double last_report_at_ = -1.0;
  uint64_t last_plans_produced_ = 0;
  uint32_t last_late_events_ = 0;
  uint32_t last_plans_dropped_ = 0;
  uint32_t last_requests_skipped_ = 0;
  uint32_t last_offsets_clamped_ = 0;
  //: The tick's own latency, for ControllerDebug.latency_ms.
  double last_tick_ms_ = 0.0;
  double last_period_ms_ = 0.0;
};

}  // namespace g1_control
