// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_control/g1_controller.h"

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <obelisk_control_msgs/msg/execution_fsm.hpp>
#include <obelisk_control_msgs/msg/velocity_command.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>

namespace g1_control {
namespace {

using g1_deploy::QuatXyzw;
using g1_deploy::Vec3;

//: The execution FSM state in which low-level control is permitted. Anything else disarms.
constexpr uint8_t kExecFsmLowLevelControl = 3;

bool AllFinite(const std::vector<double>& v) {
  for (double x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t i = std::min(v.size() - 1,
                            static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5));
  return v[i];
}

}  // namespace

G1Controller::G1Controller(const std::string& name)
    : obelisk::ObeliskController<ControlMsg, EstimatorMsg>(name) {
  DeclareParams();

  // Extra obelisk-managed inputs, registered by key. The keys match what
  // g1_control/config/common/low_level.yml already declares (sub_vel_cmd_setting,
  // sub_fsm_state_setting), so the same config drives either controller.
  RegisterObkSubscription<obelisk_control_msgs::msg::VelocityCommand>(
      "sub_vel_cmd",
      [this](const obelisk_control_msgs::msg::VelocityCommand& msg) {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        vel_cmd_ = Vec3(msg.v_x, msg.v_y, msg.w_z);
        vel_cmd_time_ = this->now().seconds();
      });

  RegisterObkSubscription<obelisk_control_msgs::msg::ExecutionFSM>(
      "sub_fsm_state", [this](const obelisk_control_msgs::msg::ExecutionFSM& msg) {
        // Arming is driven ONLY by the FSM. There is deliberately no auto-arm here: that is a
        // parameter of fsm_state_echo, a node that exists only because MuJoCo has no execution
        // FSM, and tests/test_auto_arm_is_sim_only.py fails the build if this file grows one.
        const bool want = msg.cmd_exec_fsm_state == kExecFsmLowLevelControl;
        if (want && !armed_.load()) {
          Arm();
        } else if (!want && armed_.load()) {
          Disarm("execution FSM left LOW_LEVEL_CONTROL");
        }
      });
}

void G1Controller::DeclareParams() {
  // Every parameter declared here must also be READ somewhere, and every parameter a config
  // sets must be declared: rclpy/rclcpp silently ignore an override for an undeclared
  // parameter, which is how `log`, `log_decimation`, `config_name` and `fm_update_every` all
  // became dead knobs. tests/test_params_are_live.py fails the build on either mistake.
  this->declare_parameter<std::string>("hf_repo_id", "");
  this->declare_parameter<std::string>("hf_policy_folder", "");
  this->declare_parameter<bool>("use_cuda", false);
  this->declare_parameter<int>("onnx_threads", 2);
  this->declare_parameter<std::vector<std::string>>("depth_topics",
                                                  std::vector<std::string>{});
  this->declare_parameter<double>("depth_timeout", 0.25);
  this->declare_parameter<double>("vel_cmd_timeout", 1.0);
  this->declare_parameter<bool>("verbose", false);
  this->declare_parameter<bool>("debug_topics", true);
  // The generator's LEAD, not its cadence. The cadence comes from tracker.onnx metadata and
  // nowhere else -- a parameter for it would read as a live knob that can never take effect.
  this->declare_parameter<double>("fm_lead_sec", 0.05);
  this->declare_parameter<bool>("threaded_fm", true);
  this->declare_parameter<int>("fm_late_disarm_ticks", 0);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
G1Controller::on_configure(const rclcpp_lifecycle::State& prev) {
  ObeliskController<ControlMsg, EstimatorMsg>::on_configure(prev);

  hf_repo_id_ = this->get_parameter("hf_repo_id").as_string();
  hf_policy_folder_ = this->get_parameter("hf_policy_folder").as_string();
  use_cuda_ = this->get_parameter("use_cuda").as_bool();
  onnx_threads_ = static_cast<int>(this->get_parameter("onnx_threads").as_int());
  depth_topics_ = this->get_parameter("depth_topics").as_string_array();
  depth_timeout_ = this->get_parameter("depth_timeout").as_double();
  vel_cmd_timeout_ = this->get_parameter("vel_cmd_timeout").as_double();
  verbose_ = this->get_parameter("verbose").as_bool();
  debug_topics_ = this->get_parameter("debug_topics").as_bool();
  fm_lead_sec_ = this->get_parameter("fm_lead_sec").as_double();
  threaded_fm_ = this->get_parameter("threaded_fm").as_bool();
  fm_late_disarm_ticks_ = static_cast<int>(this->get_parameter("fm_late_disarm_ticks").as_int());

  LoadPolicy();
  SetupDepth();

  if (debug_topics_) {
    // Plain rclcpp publishers, not obelisk-managed: these are diagnostics, and RELIABLE with a
    // depth of 50 so a recorder does not silently lose ticks.
    rclcpp::QoS qos(50);
    qos.reliable().durability_volatile();
    debug_pub_ = this->create_publisher<g1_msgs::msg::ControllerDebug>(
        "/obelisk/g1/debug/tick", qos);
    fm_plan_pub_ = this->create_publisher<g1_msgs::msg::FmPlan>("/obelisk/g1/debug/fm_plan", qos);
  }
  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
G1Controller::on_cleanup(const rclcpp_lifecycle::State& prev) {
  armed_.store(false);
  // Drop the runtime before the node goes: it owns the generator thread and the ONNX sessions.
  policy_.reset();
  return ObeliskController<ControlMsg, EstimatorMsg>::on_cleanup(prev);
}

void G1Controller::LoadPolicy() {
  if (hf_policy_folder_.empty() || hf_policy_folder_ == "REPLACE_ME") {
    throw std::runtime_error(
        "hf_policy_folder is not set. This controller does NOT download artifacts -- it resolves "
        "them from the prefetched cache, so bring-up cannot hang on an HTTP call. Fetch them "
        "with scripts/prefetch_policy.py and point hf_policy_folder at the run.");
  }
  // Resolve the prefetched pair. Deliberately no download: an unbounded HTTP call inside
  // on_configure hangs BRING-UP, and the failure mode is a robot that will not arm for reasons
  // that look nothing like networking.
  std::string repo = hf_repo_id_;
  std::replace(repo.begin(), repo.end(), '/', '_');
  const std::filesystem::path base =
      std::filesystem::path(ament_index_cpp::get_package_share_directory("g1_control")) /
      "resource" / "hf_cache" / repo / hf_policy_folder_;

  g1_deploy::RuntimeConfig cfg;
  cfg.tracker_onnx = (base / "tracker.onnx").string();
  cfg.fm_onnx = (base / "fm.onnx").string();
  if (!std::filesystem::exists(cfg.tracker_onnx)) {
    throw std::runtime_error("no tracker.onnx at " + cfg.tracker_onnx +
                             "; run scripts/prefetch_policy.py");
  }
  if (!std::filesystem::exists(cfg.fm_onnx)) {
    throw std::runtime_error(
        "no fm.onnx at " + cfg.fm_onnx +
        ". Tracker-only (clip) playback is not implemented in the C++ runtime yet -- it needs "
        "the HDF5 clip reader. Use the generator pair.");
  }
  cfg.providers = use_cuda_ ? std::vector<g1_deploy::Provider>{g1_deploy::Provider::kCuda}
                            : std::vector<g1_deploy::Provider>{g1_deploy::Provider::kCpu};
  cfg.intra_threads = onnx_threads_;
  cfg.fm_lead_sec = fm_lead_sec_;
  cfg.threaded_fm = threaded_fm_;

  RCLCPP_INFO(this->get_logger(), "loading the policy pair from %s (cuda=%d, threads=%d)",
              base.c_str(), static_cast<int>(use_cuda_), onnx_threads_);
  policy_ = std::make_unique<g1_deploy::PolicyRuntime>(cfg);
  policy_->Load();   // throws on a CPU fallback when cuda was asked for; see RequireCuda

  const auto& p = policy_->params();
  targets_.assign(static_cast<size_t>(p.num_actions), 0.0);
  RCLCPP_INFO(this->get_logger(),
              "policy ready: num_obs=%d num_actions=%d dt=%.3f K=%d lead=%d ticks",
              p.num_obs, p.num_actions, p.dt, p.fm_update_every.value_or(1),
              policy_->lead_ticks());

  // The control timer's period must match the policy's dt, or every cadence number here is
  // about a different clock than the one the policy trained on.
  const auto timer = this->GetTimer("timer_ctrl");
  if (timer) {
    const double period = std::chrono::duration<double>(timer->time_until_trigger()).count();
    if (period > 0.0 && std::abs(period - p.dt) > 0.25 * p.dt) {
      RCLCPP_WARN(this->get_logger(),
                  "timer_ctrl period looks like %.4f s but the policy's dt is %.4f s; the "
                  "re-plan cadence is counted in TICKS, so a mismatch silently rescales it",
                  period, p.dt);
    }
  }
}

void G1Controller::SetupDepth() {
  const auto& cams = policy_->params().depth_cameras;
  if (cams.empty()) {
    if (!depth_topics_.empty()) {
      RCLCPP_WARN(this->get_logger(),
                  "depth_topics is set but this policy reads no depth cameras; ignoring");
    }
    return;
  }
  if (depth_topics_.size() != cams.size()) {
    throw std::runtime_error(
        "this policy reads " + std::to_string(cams.size()) + " depth camera(s) but " +
        std::to_string(depth_topics_.size()) +
        " topic(s) are configured. The order is the CONTRACT order from the artifact metadata, "
        "not an arbitrary list: swapping near and far is invisible in every shape.");
  }
  depth_.assign(cams.size(), {});
  depth_time_.assign(cams.size(), -1.0);

  // One mutually-exclusive group for both cameras, so a frame cannot preempt its sibling but
  // neither is serialised behind the 50 Hz tick.
  depth_cbg_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions opts;
  opts.callback_group = depth_cbg_;
  rclcpp::QoS qos = rclcpp::SensorDataQoS();

  for (size_t i = 0; i < depth_topics_.size(); ++i) {
    depth_subs_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
        depth_topics_[i], qos,
        [this, i](const sensor_msgs::msg::Image::SharedPtr msg) { DepthCallback(msg, i); },
        opts));
    RCLCPP_INFO(this->get_logger(), "depth camera %zu ('%s') <- %s", i,
                policy_->params().depth_cameras[i].obs_term.c_str(), depth_topics_[i].c_str());
  }
}

void G1Controller::DepthCallback(const sensor_msgs::msg::Image::SharedPtr msg, size_t index) {
  const auto& cam = policy_->params().depth_cameras[index];
  // VALIDATE before casting. The python has a TODO here and gets away with it because
  // np.frombuffer raises; in C++ a reinterpret_cast over a short buffer is an out-of-bounds
  // read. A wrong-sized frame is a different camera, not a resizable image.
  const size_t expected = static_cast<size_t>(cam.pixels()) * sizeof(float);
  if (msg->encoding != "32FC1" || msg->data.size() != expected ||
      static_cast<int>(msg->width) != cam.width || static_cast<int>(msg->height) != cam.height) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                         "dropping a frame on camera %zu: encoding '%s' %ux%u (%zu bytes), "
                         "expected 32FC1 %dx%d (%zu bytes)",
                         index, msg->encoding.c_str(), msg->width, msg->height,
                         msg->data.size(), cam.width, cam.height, expected);
    return;
  }
  const float* pixels = reinterpret_cast<const float*>(msg->data.data());
  std::lock_guard<std::mutex> lock(depth_mutex_);
  depth_[index].assign(pixels, pixels + cam.pixels());
  // Stamped on LOCAL RECEIPT, not from the header: the watchdog is asking "is this camera still
  // talking to me", and a sender with a skewed clock would answer the wrong question.
  depth_time_[index] = this->now().seconds();
}

void G1Controller::UpdateXHat(const EstimatorMsg& msg) {
  std::lock_guard<std::mutex> lock(est_mutex_);
  est_.q_base.assign(msg.q_base.begin(), msg.q_base.end());
  est_.v_base.assign(msg.v_base.begin(), msg.v_base.end());
  est_.joint_pos.assign(msg.q_joints.begin(), msg.q_joints.end());
  est_.joint_vel.assign(msg.v_joints.begin(), msg.v_joints.end());
  est_.joint_names.assign(msg.joint_names.begin(), msg.joint_names.end());
  est_.stamp = msg.header.stamp;
  est_.valid = true;
  est_time_ = this->now().seconds();
}

void G1Controller::Arm() {
  EstimatorSample snapshot;
  {
    std::lock_guard<std::mutex> lock(est_mutex_);
    snapshot = est_;
  }
  if (!snapshot.valid) {
    RCLCPP_ERROR(this->get_logger(), "refusing to arm: no state estimate yet");
    return;
  }
  if (snapshot.joint_names.empty()) {
    RCLCPP_ERROR(this->get_logger(), "refusing to arm: the estimate carries no joint names");
    return;
  }

  hw_joint_names_ = snapshot.joint_names;
  // Gains come from the metadata in the POLICY's joint order; the robot wants its own.
  const auto& p = policy_->params();
  policy_to_hw_.clear();
  for (const std::string& hw : hw_joint_names_) {
    int found = -1;
    for (size_t i = 0; i < p.joint_names_isaac.size(); ++i) {
      if (p.joint_names_isaac[i] == hw) {
        found = static_cast<int>(i);
        break;
      }
    }
    if (found < 0) {
      RCLCPP_ERROR(this->get_logger(),
                   "refusing to arm: the robot reports joint '%s' which the policy does not "
                   "produce; the two disagree about the joint set",
                   hw.c_str());
      return;
    }
    policy_to_hw_.push_back(found);
  }
  kp_hw_.resize(hw_joint_names_.size());
  kd_hw_.resize(hw_joint_names_.size());
  for (size_t i = 0; i < hw_joint_names_.size(); ++i) {
    kp_hw_[i] = p.kp[static_cast<size_t>(policy_to_hw_[i])];
    kd_hw_[i] = p.kd[static_cast<size_t>(policy_to_hw_[i])];
  }

  // The last-resort fallback, and ONLY that: the commanded target is always the policy's
  // output. There is no PD warm-up and no hold regime -- whatever feeds the tracker drives it
  // from the first tick, which is what the pair was trained for.
  hold_pose_ = snapshot.joint_pos;
  policy_->Reset();
  last_good_targets_.clear();
  have_last_good_ = false;
  nonfinite_streak_ = 0;
  fm_late_streak_ = 0;
  bad_quat_count_ = 0;
  quat_samples_ = 0;
  armed_.store(true);
  RCLCPP_INFO(this->get_logger(), "ARMED: %zu joints, driving from the first tick",
              hw_joint_names_.size());
}

void G1Controller::Disarm(const std::string& why) {
  armed_.store(false);
  RCLCPP_WARN(this->get_logger(), "DISARMED: %s", why.c_str());
}

bool G1Controller::UpdateWatchdogs() {
  const double now = this->now().seconds();
  bool stale = false;

  if (!policy_->params().depth_cameras.empty() && depth_timeout_ > 0.0) {
    bool any_stale = false;
    {
      std::lock_guard<std::mutex> lock(depth_mutex_);
      for (double t : depth_time_) {
        // A camera that has NEVER spoken is not "stale" -- it has not started. Treating it as
        // stale would latch a dropout at startup that never clears.
        if (t >= 0.0 && (now - t) > depth_timeout_) any_stale = true;
      }
    }
    if (any_stale && !depth_stale_) {
      RCLCPP_WARN(this->get_logger(),
                  "DEPTH DROPOUT: the policy is running on a frozen image.");
    } else if (!any_stale && depth_stale_) {
      RCLCPP_INFO(this->get_logger(), "depth recovered");
    }
    depth_stale_ = any_stale;
    stale = stale || any_stale;
  }

  bool cmd_stale = false;
  {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    cmd_stale = vel_cmd_time_ >= 0.0 && vel_cmd_timeout_ > 0.0 &&
                (now - vel_cmd_time_) > vel_cmd_timeout_;
  }
  if (cmd_stale && !vel_cmd_stale_) {
    RCLCPP_WARN(this->get_logger(), "VELOCITY COMMAND DROPOUT: commanding zero.");
  } else if (!cmd_stale && vel_cmd_stale_) {
    RCLCPP_INFO(this->get_logger(), "velocity command recovered");
  }
  vel_cmd_stale_ = cmd_stale;
  return stale || cmd_stale;
}

Vec3 G1Controller::EffectiveCommand() {
  if (vel_cmd_stale_) return Vec3(0.0, 0.0, 0.0);
  std::lock_guard<std::mutex> lock(cmd_mutex_);
  return vel_cmd_;
}

bool G1Controller::BaseQuat(QuatXyzw* out) {
  std::vector<double> q;
  {
    std::lock_guard<std::mutex> lock(est_mutex_);
    q = est_.q_base;
  }
  if (q.size() < 7) return false;
  // q_base is [pos(3), quat(4)]. The quaternion's order in this message is xyzw, matching the
  // rest of the stack.
  QuatXyzw quat(q[3], q[4], q[5], q[6]);
  ++quat_samples_;
  if (!std::isfinite(quat.norm()) || quat.norm() < 0.5) {
    // A degenerate quaternion is an estimator that has not converged, not a robot that has
    // folded in half. Reuse the last good one and say so -- throttled, with a running
    // percentage, because one bad sample is noise and a steady stream is a broken estimator.
    ++bad_quat_count_;
    if (bad_quat_count_ % 50 == 1) {
      RCLCPP_WARN(this->get_logger(),
                  "degenerate base quaternion (norm %.3f); reusing the last good one "
                  "(%ld of %ld samples, %.1f%%)",
                  quat.norm(), bad_quat_count_, quat_samples_,
                  100.0 * static_cast<double>(bad_quat_count_) /
                      static_cast<double>(quat_samples_));
    }
    *out = last_good_quat_;
    return true;
  }
  last_good_quat_ = quat / quat.norm();
  *out = last_good_quat_;
  return true;
}

std::string G1Controller::DiagnoseNonFinite() const {
  std::string which;
  auto add = [&which](const char* s) {
    if (!which.empty()) which += ", ";
    which += s;
  };
  if (bad_q_base_) add("q_base");
  if (bad_v_base_) add("v_base");
  if (bad_joints_) add("joint_pos/joint_vel");
  if (bad_depth_) add("depth (an all-invalid frame)");
  if (which.empty()) return "none (the policy itself diverged)";
  return which;
}

bool G1Controller::PublishTargets(std::vector<double>* targets, ControlMsg* msg) {
  if (AllFinite(*targets)) {
    last_good_targets_ = *targets;
    have_last_good_ = true;
    nonfinite_streak_ = 0;
  } else {
    ++nonfinite_streak_;
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "non-finite joint target (%d in a row); offending input: %s",
                          nonfinite_streak_, DiagnoseNonFinite().c_str());
    if (nonfinite_streak_ >= kNonFiniteDisarmTicks) {
      // 0.2 s of garbage is not a transient. Disarm and publish NOTHING, so the robot
      // interface holds its last commanded pose rather than being driven from noise.
      Disarm("sustained non-finite commands");
      return false;
    }
    if (!have_last_good_) {
      if (hold_pose_.size() != targets->size()) return false;
      *targets = hold_pose_;
    } else {
      *targets = last_good_targets_;
    }
  }

  msg->pos_target.assign(targets->begin(), targets->end());
  msg->vel_target.assign(targets->size(), 0.0);
  msg->feed_forward.assign(targets->size(), 0.0);
  msg->u_mujoco.clear();
  msg->u_mujoco.insert(msg->u_mujoco.end(), targets->begin(), targets->end());
  msg->u_mujoco.insert(msg->u_mujoco.end(), targets->size() * 2, 0.0);
  msg->joint_names = hw_joint_names_;
  msg->kp = kp_hw_;
  msg->kd = kd_hw_;
  msg->header.stamp = this->now();
  this->GetPublisher<ControlMsg>(this->ctrl_key_)->publish(*msg);
  return true;
}

ControlMsg G1Controller::ComputeControl() {
  ControlMsg msg;
  if (!armed_.load() || !policy_) {
    // Publish nothing while disarmed. The debug tick still goes out, so a bag can tell
    // "disarmed" from "the node died".
    PublishDebug({}, /*have_obs=*/false);
    return msg;
  }

  const auto t0 = std::chrono::steady_clock::now();
  UpdateWatchdogs();

  EstimatorSample est;
  {
    std::lock_guard<std::mutex> lock(est_mutex_);
    est = est_;   // ONE sample, copied as a unit; see EstimatorSample
  }
  if (!est.valid) {
    PublishDebug({}, false);
    return msg;
  }

  QuatXyzw quat;
  if (!BaseQuat(&quat)) {
    PublishDebug({}, false);
    return msg;
  }

  g1_deploy::HardwareReadings r;
  r.root_pos = Vec3(est.q_base[0], est.q_base[1], est.q_base[2]);
  r.base_quat_xyzw = quat;
  r.base_lin_vel_w = Vec3(est.v_base[0], est.v_base[1], est.v_base[2]);
  r.base_ang_vel_b = Vec3(est.v_base[3], est.v_base[4], est.v_base[5]);
  r.joint_pos = est.joint_pos;
  r.joint_vel = est.joint_vel;
  r.joint_names = est.joint_names;
  r.command = EffectiveCommand();

  bad_q_base_ = !AllFinite(est.q_base);
  bad_v_base_ = !AllFinite(est.v_base);
  bad_joints_ = !AllFinite(est.joint_pos) || !AllFinite(est.joint_vel);
  bad_depth_ = false;
  {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    for (const auto& img : depth_) {
      if (img.empty()) {
        // A camera that has not published yet cannot be run blind past: the policy reads it.
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "waiting for the first frame on every depth camera before running");
        PublishDebug({}, false);
        return msg;
      }
      r.depth_raw.push_back(img);
    }
  }

  g1_deploy::TickDiagnostics diag;
  try {
    policy_->Step(r, hw_joint_names_, targets_.data(), &diag);
  } catch (const std::exception& e) {
    // A contract violation is not survivable by continuing: the observation would be wrong in a
    // way the policy cannot see. Disarm loudly.
    Disarm(std::string("the policy refused this tick: ") + e.what());
    PublishDebug(diag, false);
    return msg;
  }

  if (diag.late && fm_late_disarm_ticks_ > 0) {
    if (++fm_late_streak_ >= fm_late_disarm_ticks_) {
      Disarm("the generator has been late for " + std::to_string(fm_late_streak_) + " ticks");
      PublishDebug(diag, true);
      return msg;
    }
  } else if (!diag.late) {
    fm_late_streak_ = 0;
  }

  std::vector<double> targets(targets_.begin(), targets_.end());
  PublishTargets(&targets, &msg);

  // Latency AFTER publishing: a diagnostic must never delay the command.
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
  latency_ms_.push_back(ms);
  if (latency_ms_.size() > 250) latency_ms_.pop_front();
  const double now = this->now().seconds();
  if (last_latency_report_ < 0.0) last_latency_report_ = now;
  if (now - last_latency_report_ >= 5.0) {
    const std::vector<double> v(latency_ms_.begin(), latency_ms_.end());
    const double budget = policy_->dt() * 1e3;
    const double p99 = Percentile(v, 0.99);
    if (p99 > budget) {
      RCLCPP_WARN(this->get_logger(),
                  "OVER BUDGET: tick p50 %.2f p95 %.2f p99 %.2f ms vs %.2f ms; fm %.2f ms",
                  Percentile(v, 0.5), Percentile(v, 0.95), p99, budget, diag.fm_infer_ms);
    } else if (verbose_) {
      RCLCPP_INFO(this->get_logger(),
                  "tick p50 %.2f p95 %.2f p99 %.2f ms vs %.2f ms; fm %.2f ms",
                  Percentile(v, 0.5), Percentile(v, 0.95), p99, budget, diag.fm_infer_ms);
    }
    last_latency_report_ = now;
  }

  PublishDebug(diag, true);
  return msg;
}

void G1Controller::PublishDebug(const g1_deploy::TickDiagnostics& diag, bool have_obs) {
  if (!debug_pub_) return;
  // Wrapped so a diagnostics bug can never stop the robot.
  try {
    g1_msgs::msg::ControllerDebug d;
    d.header.stamp = this->now();
    d.ctrl_step = static_cast<uint64_t>(std::max<int64_t>(diag.ctrl_step, 0));
    d.latency_ms = latency_ms_.empty() ? 0.0 : latency_ms_.back();
    d.armed = armed_.load();
    d.depth_stale = depth_stale_;
    d.vel_cmd_stale = vel_cmd_stale_;
    d.nonfinite_streak = static_cast<uint32_t>(std::max(nonfinite_streak_, 0));
    d.obs_checksum = have_obs ? diag.obs_checksum : 0;
    d.obs_size = have_obs && policy_ ? static_cast<uint32_t>(policy_->num_obs()) : 0;
    d.fm_cursor = diag.cursor;
    if (policy_ && have_obs) {
      d.joint_names_isaac = policy_->params().joint_names_isaac;
      const auto& a = policy_->last_action();
      d.action_isaac.assign(a.begin(), a.end());
      d.reference_window = policy_->last_reference_window();
    }
    const Vec3 cmd = vel_cmd_stale_ ? Vec3(0.0, 0.0, 0.0) : vel_cmd_;
    d.command_effective = {cmd.x(), cmd.y(), cmd.z()};
    debug_pub_->publish(d);
  } catch (const std::exception& e) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "debug publish failed: %s", e.what());
  }
}

}  // namespace g1_control
