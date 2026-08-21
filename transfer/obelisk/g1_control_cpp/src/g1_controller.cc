// Copyright 2026 Zachary Olkin. All rights reserved.

#include <builtin_interfaces/msg/time.hpp>

#include "g1_control/g1_controller.h"

#include <obelisk_control_msgs/msg/execution_fsm.hpp>
#include <obelisk_control_msgs/msg/velocity_command.hpp>

// For rcl_timer_get_period: rclcpp::TimerBase exposes no accessor for the period itself,
// only the rcl handle it wraps. See the note at the timer_ctrl check.
#include <rcl/timer.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <numeric>

namespace g1_control {
namespace {

using g1_deploy::QuatXyzw;
using g1_deploy::Vec3;

//: The execution FSM state in which low-level control is permitted. Anything else disarms.
constexpr uint8_t kExecFsmLowLevelControl = 3;

//: How often the timing line is emitted. Long enough that a healthy run is quiet, short
//: enough to see a transient while it is still on screen.
constexpr double kTimingReportSec = 5.0;

bool AllFinite(const std::vector<double>& v) {
  for (double x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

}  // namespace

double RollingStat::Percentile(double q) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const size_t i =
      std::min(samples.size() - 1,
               static_cast<size_t>(q * static_cast<double>(samples.size() - 1) + 0.5));
  return samples[i];
}

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
  //
  // The lead is the generator's deadline AND its exteroception staleness: the plan is
  // conditioned on the depth image and command as they were at REQUEST time, so each tick of
  // lead gives the generator another dt to answer and hands it an image dt older. Clamped to
  // [1, K-1]. A plan that misses its boundary does not stall the reference; see
  // plan_scheduler.h's lateness contract, and `fm_late_events` in a bag for how often.
  //
  // The config file is what actually sets this per deployment; this default only applies
  // when nothing does.
  this->declare_parameter<double>("fm_lead_sec", 0.02);
  this->declare_parameter<bool>("threaded_fm", true);
  // Generator-only, and OFF by default: it needs a CUDA provider and a fixed-batch
  // fm.onnx, and both are refused loudly at load rather than degraded silently.
  this->declare_parameter<bool>("fm_cuda_graph", false);
  this->declare_parameter<int>("fm_late_disarm_ticks", 0);
  // The kinematic model the camera poses are FK'd on. A parameter and not a constant because
  // the robot's models are on their way to being consolidated; when that lands this moves a
  // config line instead of a recompile. Relative paths resolve against the workspace root.
  this->declare_parameter<std::string>("urdf_path", "g1_model/urdf/g1_29j.urdf");
  // Integrate the generator's flow from x0 = 0 instead of x0 ~ N(0, I).
  //
  // TRUE for now, deliberately, and it is a debugging stance rather than the final one. The FM
  // is a stochastic sampler: zero noise gives ONE deterministic trajectory per conditioning,
  // which is not what it was trained as. What it buys is that this stack and the MuJoCo
  // sim2sim runner (whose own default is already zero noise) draw the same plan from the same
  // conditioning -- and they otherwise cannot, because the C++ NoiseSource is deliberately not
  // a bit-port of numpy's PCG64+ziggurat, so the same seed yields DIFFERENT samples on the two
  // sides. Comparing the two stacks while each samples its own x0 measures the draw.
  //
  // Set false to restore training-matched sampling.
  this->declare_parameter<bool>("zero_noise", true);
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
  fm_cuda_graph_ = this->get_parameter("fm_cuda_graph").as_bool();
  fm_late_disarm_ticks_ = static_cast<int>(this->get_parameter("fm_late_disarm_ticks").as_int());
  urdf_path_ = this->get_parameter("urdf_path").as_string();
  zero_noise_ = this->get_parameter("zero_noise").as_bool();

  LoadPolicy();
  SetupDepth();
  SetupCameraFk();

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

namespace {

/// The obelisk workspace root, from the environment the container exports.
///
/// From the SOURCE tree via ROBOT_RL_ROOT, never from an ament share directory: g1_control is
/// ament_python and does not install resource/ at all, so its share tree has never held the
/// artifacts. And never by counting ".." hops from the binary, which breaks the moment a file
/// moves.
std::filesystem::path Workspace() {
  const char* ws = std::getenv("ROBOT_RL_ROOT");
  if (ws != nullptr && *ws != '\0') return std::filesystem::path(ws);
  const char* repo_root = std::getenv("REPO_ROOT");
  if (repo_root != nullptr && *repo_root != '\0') {
    return std::filesystem::path(repo_root) / "transfer" / "obelisk";
  }
  throw std::runtime_error(
      "neither ROBOT_RL_ROOT nor REPO_ROOT is set, so the workspace cannot be located. Both "
      "are exported by docker/docker-compose.yml from docker/.env; running this node outside "
      "that container needs one of them pointing at transfer/obelisk.");
}

}  // namespace

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
  //
  // From the SOURCE tree, via the workspace root the container already exports -- NOT from the
  // ament share directory. g1_control is ament_python and does not install resource/ at all, so
  // get_package_share_directory points at a share tree that has never contained the artifacts.
  // controller.py resolves the same path from its own __file__; this uses the environment
  // instead, because counting ".." hops from a binary's location is the thing that breaks the
  // moment a file moves.
  const std::filesystem::path workspace = Workspace();

  std::string repo = hf_repo_id_;
  std::replace(repo.begin(), repo.end(), '/', '_');
  // "policies": the sub-directory of the HF repo every policy lives under, mirrored into the
  // cache because that is where hf_hub_download puts a miss (local_dir/<repo-relative path>).
  // hf_policy_folder_ is a BARE run name; the prefix is never part of the ROS param.
  //
  // Restated here because this is C++ and cannot import deploy.hardware.HF_POLICY_PREFIX,
  // which is the definition. Restating a constant across the language boundary is how
  // hf_repo_id already drifted once, so it is pinned rather than trusted:
  // transfer/obelisk/tests/test_policy_cache_layout.py reads BOTH and fails on a mismatch.
  const std::filesystem::path base = workspace / "g1_control" / "resource" / "hf_cache" /
                                     repo / "policies" / hf_policy_folder_;

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
  cfg.fm_cuda_graph = fm_cuda_graph_;
  cfg.zero_noise = zero_noise_;

  RCLCPP_INFO(this->get_logger(),
              "loading the policy pair from %s (cuda=%d, threads=%d, zero_noise=%d, "
              "fm_cuda_graph=%d)",
              base.c_str(), static_cast<int>(use_cuda_), onnx_threads_,
              static_cast<int>(zero_noise_), static_cast<int>(fm_cuda_graph_));
  policy_ = std::make_unique<g1_deploy::PolicyRuntime>(cfg);
  policy_->Load();   // throws on a CPU fallback when cuda was asked for; see RequireCuda

  const auto& p = policy_->params();
  targets_.assign(static_cast<size_t>(p.num_actions), 0.0);
  RCLCPP_INFO(this->get_logger(),
              "policy ready: num_obs=%d num_actions=%d dt=%.3f K=%d lead=%d ticks = %.0f ms "
              "(the generator's budget AND its exteroception staleness -- the same number)",
              p.num_obs, p.num_actions, p.dt, p.fm_update_every.value_or(1),
              policy_->lead_ticks(), policy_->lead_ticks() * p.dt * 1e3);

  // The control timer's period must match the policy's dt, or every cadence number here is
  // about a different clock than the one the policy trained on.
  //
  // rcl_timer_get_period, NOT TimerBase::time_until_trigger(). The latter is the time
  // REMAINING until the next fire, which is a different quantity and never the right one
  // here. Two ways it was wrong, and the first is the one that showed up:
  //
  //   * this runs in on_configure, where obelisk has CREATED the timer but not yet started
  //     it, and rclcpp maps a cancelled timer to nanoseconds::max(). So the check fired on
  //     every single bring-up reporting a period of 9223372036.8548 s (INT64_MAX ns) --
  //     a warning about the control rate, on a stack whose control rate is the thing under
  //     test, that was guaranteed to be noise.
  //   * on a RUNNING timer it would have been no better, only less obviously broken: the
  //     value is a random fraction of the period, so the check would pass or fail depending
  //     on when it happened to sample.
  const auto timer = this->GetTimer("timer_ctrl");
  if (timer) {
    int64_t period_ns = 0;
    if (rcl_timer_get_period(timer->get_timer_handle().get(), &period_ns) == RCL_RET_OK) {
      const double period = static_cast<double>(period_ns) * 1e-9;
      if (period > 0.0 && std::abs(period - p.dt) > 0.25 * p.dt) {
        RCLCPP_WARN(this->get_logger(),
                    "timer_ctrl period is %.4f s but the policy's dt is %.4f s; the "
                    "re-plan cadence is counted in TICKS, so a mismatch silently rescales it",
                    period, p.dt);
      }
    }
  }
}

void G1Controller::SetupCameraFk() {
  // Only a generator that reads MORE THAN DEPTH per pixel needs this. A 1-channel one gets a
  // normalised image and nothing else, and building the FK for it would make a kinematic model
  // a bring-up requirement for a policy that never consults one.
  const int channels = policy_->fm_meta().extero_channels;
  if (channels <= 1) {
    RCLCPP_INFO(this->get_logger(),
                "the generator reads depth alone (1 channel); no camera FK needed");
    return;
  }
  const auto& cams = policy_->params().depth_cameras;
  if (cams.empty()) {
    // A scan generator on a camera-less tracker: the xyz channels are the SCAN's, and it
    // carries its own metric coordinates. Nothing to FK.
    return;
  }

  std::filesystem::path model = urdf_path_;
  if (model.is_relative()) model = Workspace() / model;
  if (!std::filesystem::exists(model)) {
    throw std::runtime_error(
        "the generator conditions on " + std::to_string(channels) +
        " channels per pixel, so it needs camera kinematics, but urdf_path '" + model.string() +
        "' does not exist. Point the urdf_path parameter at the robot's model.");
  }

  // Built against the POLICY's joint order -- the artifact's own -- rather than the robot's,
  // which is not known until the first estimate arrives. That makes a bad model a BRING-UP
  // failure instead of a refusal to arm, and it is the order the parity fixtures pin.
  camera_fk_ = std::make_unique<g1_deploy::CameraPoseFk>(
      model.string(), policy_->params().joint_names_isaac, cams);
  fk_joint_pos_.assign(policy_->params().joint_names_isaac.size(), 0.0);
  RCLCPP_INFO(this->get_logger(),
              "camera FK ready: %zu camera(s) on %s (%d channels per pixel)",
              camera_fk_->num_cameras(), model.c_str(), channels);
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
  depth_stamp_.assign(cams.size(), -1.0);

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
  // The header stamp as well, for the age-from-CAPTURE diagnostic: receipt time measures only
  // this node's own contribution, and the interesting number is how old the pixels are.
  const double stamp = static_cast<double>(msg->header.stamp.sec) +
                       1e-9 * static_cast<double>(msg->header.stamp.nanosec);
  depth_stamp_[index] = stamp > 0.0 ? stamp : -1.0;
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
  // Counters only; the held values survive on purpose (see ImuHold::ResetCounters).
  imu_hold_.ResetCounters();
  quat_held_ = false;
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

bool G1Controller::BaseImu(const EstimatorSample& est, QuatXyzw* quat_out, Vec3* ang_vel_out) {
  if (est.q_base.size() < 7 || est.v_base.size() < 6) return false;
  // q_base is [pos(3), quat(4)]. The quaternion's order in this message is xyzw, matching the
  // rest of the stack. v_base is [linear(3), angular(3)]; only the angular half is a policy
  // observation (see the note where base_ang_vel_b is filled).
  const QuatXyzw quat(est.q_base[3], est.q_base[4], est.q_base[5], est.q_base[6]);
  const Vec3 gyro(est.v_base[3], est.v_base[4], est.v_base[5]);
  const g1_deploy::ImuHold::Result held = imu_hold_.Update(quat, gyro);
  quat_held_ = held.held;
  *quat_out = held.quat;
  *ang_vel_out = held.ang_vel;
  if (held.held) {
    // A degenerate quaternion is an estimator that has not converged, or -- on one robot with
    // a failing IMU -- an ALL-ZERO estimator message roughly 1 sample in 10. Either way it is
    // not a robot that has folded in half. Reuse the last good sample and say so, throttled,
    // with a running percentage, because one bad sample is noise and a steady stream is broken
    // hardware.
    //
    // The GYRO is held too, on this same test. That is the point of holding it here rather
    // than guarding it separately: an all-zero message carries a zero angular velocity, which
    // is a perfectly plausible reading (a robot standing still) that no test on the gyro alone
    // can reject. The quaternion's norm is the only available detector for the dropout, so it
    // has to gate everything the dropout corrupts. If instead only the quaternion were ever
    // bad, holding a good gyro for one sample costs ~1 ms of staleness at the estimator's
    // 1 kHz against a 50 Hz control tick -- far cheaper than telling a falling robot's policy
    // that it is not rotating.
    if (imu_hold_.bad_samples() % 50 == 1) {
      RCLCPP_WARN(this->get_logger(),
                  "degenerate base quaternion (norm %.3f); reusing the last good orientation "
                  "and gyro (%ld of %ld samples, %.1f%%)%s",
                  quat.norm(), imu_hold_.bad_samples(), imu_hold_.samples(),
                  100.0 * static_cast<double>(imu_hold_.bad_samples()) /
                      static_cast<double>(std::max<int64_t>(imu_hold_.samples(), 1)),
                  imu_hold_.any_good()
                      ? ""
                      : " -- AND NO GOOD SAMPLE HAS ARRIVED YET, so this is the identity "
                        "orientation and a zero gyro, not a measurement");
    }
  }
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
    // The tick clock stops with the loop. Without this the first tick after a re-arm measures
    // the whole DISARMED interval as one control period -- 60 s of standing in the FSM shows
    // up as a 60000 ms sample in period_ms_ and in ControllerDebug.tick_period_ms, which on
    // the health figure's linear axis flattens every real trace to the x-axis.
    have_last_tick_ = false;
    last_period_ms_ = 0.0;
    // Publish nothing while disarmed. The debug tick still goes out, so a bag can tell
    // "disarmed" from "the node died".
    PublishDebug({}, /*have_obs=*/false);
    return msg;
  }

  const auto t0 = std::chrono::steady_clock::now();
  // Sampled HERE, not in RecordTiming: several guards below return early, and those ticks
  // still consumed a control period. Measuring the gap at the top counts them, so a loop that
  // is running but bailing every tick shows as a healthy period rather than as no data.
  if (have_last_tick_) {
    last_period_ms_ = std::chrono::duration<double, std::milli>(t0 - last_tick_at_).count();
    period_ms_.Add(last_period_ms_);
  }
  last_tick_at_ = t0;
  have_last_tick_ = true;

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
  Vec3 base_ang_vel;
  if (!BaseImu(est, &quat, &base_ang_vel)) {
    PublishDebug({}, false);
    return msg;
  }

  g1_deploy::HardwareReadings r;
  r.root_pos = Vec3(est.q_base[0], est.q_base[1], est.q_base[2]);
  r.base_quat_xyzw = quat;
  // Only the ANGULAR half of v_base reaches the policy. base_ang_vel is an observation term
  // (root-link frame, straight off the gyro); base LINEAR velocity is not an observation of
  // any policy this runtime loads, and the generator conditions on its own previous output
  // rather than on measured velocity -- so the estimator's world-frame linear velocity is
  // read by nothing here. It is still validated for finiteness below, and still published for
  // the plotters. Do not reintroduce it as a field "for completeness": a reading nobody
  // consumes is one more thing that can look wrong.
  // From BaseImu, not straight off the message: it is held with the quaternion when the
  // estimator sample is degenerate.
  r.base_ang_vel_b = base_ang_vel;
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

  if (camera_fk_) {
    // Encoders and gravity, never odometry: the yaw and the root position both cancel out of
    // the yaw-aligned frame, so a drifting estimate cannot move the cloud. Recomputed every
    // tick because it is a function of the posture, and cheap (one FK pass).
    //
    // policy_to_hw_ was resolved at arming, against the joint names the estimate carried THEN.
    // The runtime re-resolves its own permutation whenever the names change; this one does not,
    // so a change is a disarm rather than a silently scrambled posture -- which would place the
    // cameras plausibly and wrongly.
    if (est.joint_names != hw_joint_names_) {
      Disarm("the estimator's joint set changed while armed, so the camera FK permutation is "
             "no longer valid");
      PublishDebug({}, false);
      return msg;
    }
    for (size_t i = 0; i < hw_joint_names_.size(); ++i) {
      fk_joint_pos_[static_cast<size_t>(policy_to_hw_[i])] = est.joint_pos[i];
    }
    // `quat` is ALREADY xyzw (QuatXyzw is a Vector4d in that order), so this is a copy and not
    // a re-ordering.
    camera_fk_->Compute(fk_joint_pos_, {quat[0], quat[1], quat[2], quat[3]}, &fk_cam_pos_,
                        &fk_cam_rot_);
    r.fm_camera_pos = fk_cam_pos_;
    r.fm_camera_rot = fk_cam_rot_;
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

  // Timing AFTER publishing: a diagnostic must never delay the command.
  RecordTiming(diag, std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count());

  // One message per plan the robot actually used. Gated on the id and not on `plan_adopted`
  // alone so a re-arm, which resets the schedule but deliberately not the id, cannot
  // re-publish a plan already on the bag under a second timestamp.
  if (diag.plan_adopted && diag.plan_id != last_fm_plan_id_) {
    last_fm_plan_id_ = diag.plan_id;
    PublishFmPlan(diag);
  }
  PublishDebug(diag, true);
  return msg;
}

void G1Controller::RecordTiming(const g1_deploy::TickDiagnostics& diag, double tick_ms) {
  last_tick_ms_ = tick_ms;
  tick_ms_.Add(tick_ms);
  trk_ms_.Add(diag.trk_infer_ms);
  // The generator runs at 1/K the control rate, so fm_infer_ms is the SAME number on the K-1
  // ticks between inferences. Sampling it every tick would weight one slow plan K times and
  // make the percentiles describe the cadence rather than the network.
  if (diag.plans_produced != last_plans_produced_) {
    fm_ms_.Add(diag.fm_infer_ms);
    last_plans_produced_ = diag.plans_produced;
  }

  const double now = this->now().seconds();
  if (last_report_at_ < 0.0) last_report_at_ = now;
  if (now - last_report_at_ < kTimingReportSec) return;
  const double elapsed = now - last_report_at_;
  last_report_at_ = now;

  const double dt_ms = policy_->dt() * 1e3;
  // The generator's real deadline. It is asked for a plan lead_ticks BEFORE the boundary that
  // plan is for and runs off the control thread, so it has the whole lead window -- not one
  // tick. Judging it against dt would report a healthy generator as chronically over budget.
  const double fm_budget_ms = std::max(1, diag.lead_ticks) * dt_ms;

  // These counters are cumulative SINCE ARMING, and PlanScheduler::Reset() -- which
  // PolicyRuntime::Reset() runs on every Arm() -- puts them back to zero. A plain subtraction
  // across a re-arm therefore underflows an unsigned type: 0 - 3 prints as 4294967293 and, via
  // `unhealthy`, fires the OVER-BUDGET warning on a perfectly healthy run. Treat a counter
  // that went BACKWARD as "restarted", and take its current value as the window's delta.
  auto window_delta = [](uint32_t now, uint32_t before) {
    return now >= before ? now - before : now;
  };
  const uint32_t late = window_delta(diag.late_events, last_late_events_);
  const uint32_t dropped = window_delta(diag.plans_dropped, last_plans_dropped_);
  const uint32_t skipped = window_delta(diag.requests_skipped, last_requests_skipped_);
  const uint32_t clamped = window_delta(diag.offsets_clamped, last_offsets_clamped_);
  last_late_events_ = diag.late_events;
  last_plans_dropped_ = diag.plans_dropped;
  last_requests_skipped_ = diag.requests_skipped;
  last_offsets_clamped_ = diag.offsets_clamped;

  // Plans the cadence CALLED for over this window, which is what "is the generator keeping
  // up" is measured against -- a shortfall is the generator falling behind its own schedule,
  // and it shows here before it shows as lateness. From elapsed WALL time rather than a tick
  // count, so a loop that is not running at 50 Hz does not also deflate its own target.
  const int k = policy_->replan_period();
  // Rounded, not truncated: the window is not a whole number of cadence periods, so 5.0 s at
  // K=12 and dt=0.02 is 20.83 -- and truncating printed "21 plans, 20 expected" on a run that
  // was keeping up exactly.
  const int expected =
      k > 0 ? static_cast<int>(std::lround(elapsed / (static_cast<double>(k) * policy_->dt())))
            : 0;

  const double tick_p99 = tick_ms_.Percentile(0.99);
  const double fm_p99 = fm_ms_.Percentile(0.99);
  const bool unhealthy = tick_p99 > dt_ms || fm_p99 > fm_budget_ms || late > 0 || dropped > 0 ||
                         skipped > 0 || clamped > 0;

  // Formatted once into a buffer and logged with a LITERAL "%s". Passing a `const char*`
  // format to RCLCPP_WARN/_INFO would be a non-literal format string, which trips
  // -Wformat-nonliteral / -Werror=format-security depending on the build flags -- and the two
  // branches must not drift apart, which two copies of an 18-argument call invite.
  char line[512];
  std::snprintf(
      line, sizeof(line),
      "timing (%.1fs): tick p50 %.2f p95 %.2f p99 %.2f / %.1f ms | period p50 %.2f p99 %.2f ms"
      " | trk p50 %.2f p99 %.2f ms | fm p50 %.2f p95 %.2f p99 %.2f / %.1f ms (%zu plans, %d"
      " expected) | late %u dropped %u skipped %u clamped %u",
      elapsed, tick_ms_.Percentile(0.5), tick_ms_.Percentile(0.95), tick_p99, dt_ms,
      period_ms_.Percentile(0.5), period_ms_.Percentile(0.99), trk_ms_.Percentile(0.5),
      trk_ms_.Percentile(0.99), fm_ms_.Percentile(0.5), fm_ms_.Percentile(0.95), fm_p99,
      fm_budget_ms, fm_ms_.size(), expected, late, dropped, skipped, clamped);

  // Deliberately NOT behind `verbose_`. One line per 5 s is the cheapest possible answer to
  // "is the loop keeping up", and needing to know is exactly when you have not set the flag.
  if (unhealthy) {
    RCLCPP_WARN(this->get_logger(), "%s", line);
  } else {
    RCLCPP_INFO(this->get_logger(), "%s", line);
  }

  tick_ms_.Clear();
  period_ms_.Clear();
  trk_ms_.Clear();
  fm_ms_.Clear();
}

void G1Controller::PublishFmPlan(const g1_deploy::TickDiagnostics& diag) {
  if (!fm_plan_pub_ || !policy_) return;
  // Wrapped like PublishDebug, and for the same reason: a diagnostics bug must not stop the
  // robot. This one runs once per re-plan (~4 Hz at K=12), not per tick.
  try {
    g1_msgs::msg::FmPlan m;
    m.header.stamp = this->now();
    m.plan_id = diag.plan_id;
    m.ctrl_step = static_cast<uint64_t>(std::max<int64_t>(diag.ctrl_step, 0));
    // The tick that CONDITIONED this plan, which is not the tick adopting it. Node j of the
    // plan is for step target + j, so target == ctrl_step - cursor, and the request went out
    // lead_ticks before that. Derived from the diagnostics rather than plumbed separately:
    // both terms come straight from PlanScheduler, so a second path could only disagree.
    m.lead_ticks = static_cast<uint32_t>(std::max(diag.lead_ticks, 0));
    m.request_ctrl_step = static_cast<uint64_t>(
        std::max<int64_t>(diag.ctrl_step - diag.cursor - diag.lead_ticks, 0));

    const g1_deploy::FmMeta& fm = policy_->fm_meta();
    m.trajectory = policy_->last_plan_window();
    m.num_nodes = static_cast<uint32_t>(fm.f);
    m.node_dim = static_cast<uint32_t>(fm.d);
    for (g1_deploy::ReferenceBlock b : fm.window_blocks) {
      m.window_blocks.push_back(g1_deploy::BlockName(b));
    }
    const auto& p = policy_->params();
    if (p.reference_window.has_value()) m.ee_frame_names = p.reference_window->ee_frame_names;

    m.anchor_pos = {diag.anchor_pos.x(), diag.anchor_pos.y(), diag.anchor_pos.z()};
    m.anchor_quat = {diag.anchor_quat.x(), diag.anchor_quat.y(), diag.anchor_quat.z(),
                     diag.anchor_quat.w()};

    const g1_deploy::FmRequest& req = policy_->last_plan_request();
    m.history_nodes = req.history_nodes;
    m.history_cmd = req.history_cmd;
    m.history_length = static_cast<uint32_t>(fm.n);
    m.history_null = req.history_null;

    m.extero_source = fm.extero_source;
    m.cmd_dim = static_cast<uint32_t>(fm.cmd_dim);
    // extero_tokens and cmd_tokens are deliberately EMPTY. PolicyRuntime::Step calls
    // SetConditioning on EVERY tick, not only on request ticks, so what the session holds now
    // is THIS tick's exteroception -- not the plan's. Publishing it would put a wrong depth
    // stream under a field whose stated purpose is catching a normalisation or camera-ordering
    // error, which is worse than publishing nothing. Filling them needs FmSession::Submit to
    // snapshot its conditioning alongside the plan; until it does, they stay empty.
    m.infer_ms = diag.fm_infer_ms;
    fm_plan_pub_->publish(m);
  } catch (const std::exception& e) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "fm_plan publish failed: %s", e.what());
  }
}

void G1Controller::PublishDebug(const g1_deploy::TickDiagnostics& diag, bool have_obs) {
  if (!debug_pub_) return;
  // Wrapped so a diagnostics bug can never stop the robot.
  try {
    g1_msgs::msg::ControllerDebug d;
    d.header.stamp = this->now();
    d.ctrl_step = static_cast<uint64_t>(std::max<int64_t>(diag.ctrl_step, 0));
    d.latency_ms = last_tick_ms_;
    d.fm_infer_ms = diag.fm_infer_ms;
    d.trk_infer_ms = diag.trk_infer_ms;
    d.tick_period_ms = last_period_ms_;
    d.fm_late = diag.late;
    d.fm_late_events = diag.late_events;
    d.fm_plans_dropped = diag.plans_dropped;
    d.fm_requests_skipped = diag.requests_skipped;
    d.fm_offsets_clamped = diag.offsets_clamped;
    d.armed = armed_.load();
    d.depth_stale = depth_stale_;
    d.vel_cmd_stale = vel_cmd_stale_;
    d.nonfinite_streak = static_cast<uint32_t>(std::max(nonfinite_streak_, 0));
    d.quat_samples = static_cast<uint64_t>(std::max<int64_t>(imu_hold_.samples(), 0));
    d.bad_quat_samples = static_cast<uint64_t>(std::max<int64_t>(imu_hold_.bad_samples(), 0));
    d.quat_held = quat_held_;
    d.obs_checksum = have_obs ? diag.obs_checksum : 0;
    d.obs_size = have_obs && policy_ ? static_cast<uint32_t>(policy_->num_obs()) : 0;
    d.fm_cursor = diag.cursor;

    // Input provenance. WHICH sample fed this tick, not its value -- the estimator runs at
    // 1 kHz against a 50 Hz tick, so without these no offline reconstruction can say which of
    // twenty candidate messages the observation was built from. Ages are on LOCAL receipt
    // time, matching the watchdogs and needing no clock sync with the camera host.
    const double now_s = this->now().seconds();
    {
      std::lock_guard<std::mutex> lock(est_mutex_);
      d.est_state_stamp = est_.stamp;
      d.est_state_age = est_time_ >= 0.0 ? now_s - est_time_ : -1.0;
    }
    {
      std::lock_guard<std::mutex> lock(depth_mutex_);
      d.depth_age.reserve(depth_time_.size());
      for (double t : depth_time_) d.depth_age.push_back(t >= 0.0 ? now_s - t : -1.0);
      d.depth_capture_age.reserve(depth_stamp_.size());
      d.depth_stamp.reserve(depth_stamp_.size());
      for (double t : depth_stamp_) {
        d.depth_capture_age.push_back(t >= 0.0 ? now_s - t : -1.0);
        builtin_interfaces::msg::Time stamp;
        if (t >= 0.0) {
          stamp.sec = static_cast<int32_t>(t);
          stamp.nanosec = static_cast<uint32_t>((t - static_cast<double>(stamp.sec)) * 1e9);
        }
        d.depth_stamp.push_back(stamp);
      }
    }
    Vec3 cmd{0.0, 0.0, 0.0};
    {
      // Under the lock, like every other read of these: the joystick callback runs on another
      // executor thread.
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      d.vel_cmd_age = vel_cmd_time_ >= 0.0 ? now_s - vel_cmd_time_ : -1.0;
      if (!vel_cmd_stale_) cmd = vel_cmd_;
    }
    d.command_effective = {cmd.x(), cmd.y(), cmd.z()};

    if (policy_) {
      const auto& p = policy_->params();
      d.joint_names_isaac = p.joint_names_isaac;
      // The window's layout, so a reader never restates the tracker metadata. Published every
      // tick even when nothing was served, because it describes the CONTRACT, not the sample.
      if (p.reference_window.has_value()) {
        for (const auto& b : p.reference_window->blocks) d.reference_blocks.push_back(b.first);
        d.reference_offsets.assign(p.reference_window->offsets.begin(),
                                   p.reference_window->offsets.end());
        int node_dim = 0;
        for (const auto& b : p.reference_window->blocks) node_dim += b.second;
        d.reference_node_dim = static_cast<uint32_t>(node_dim);
      }
      d.fm_plan_id = diag.plan_id;
      d.fm_anchor_pos = {diag.anchor_pos.x(), diag.anchor_pos.y(), diag.anchor_pos.z()};
      d.fm_anchor_quat = {diag.anchor_quat.x(), diag.anchor_quat.y(), diag.anchor_quat.z(),
                          diag.anchor_quat.w()};
    }
    if (policy_ && have_obs) {
      const auto& a = policy_->last_action();
      d.action_isaac.assign(a.begin(), a.end());
      d.reference_window = policy_->last_reference_window();
    }
    debug_pub_->publish(d);
  } catch (const std::exception& e) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "debug publish failed: %s", e.what());
  }
}

}  // namespace g1_control
