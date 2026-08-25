// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/clip_reference.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "g1_deploy/blob_pack.h"
#include "g1_deploy/quat_math.h"

namespace g1_deploy {
namespace {

/// Yaw angle (rad) about world z of an (x, y, z, w) quaternion.
///
/// Transcribed from clip_reference._yaw_angle rather than routed through YawQuat: the
/// re-anchor needs the ANGLE (it takes a difference of two of them), and going via the
/// quaternion and back would add a sin/asin round trip the python does not do.
double YawAngle(const QuatXyzw& q) {
  const double x = q.x(), y = q.y(), z = q.z(), w = q.w();
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

/// Pure-yaw quaternion of a yaw angle.
QuatXyzw YawQuatFromAngle(double yaw) {
  const double h = 0.5 * yaw;
  return QuatXyzw(0.0, 0.0, std::sin(h), std::cos(h));
}

/// Index of `name` in `names`.
///
/// \throws std::runtime_error Naming the channel and the case, because that is the whole
///   value of resolving by name: a pack built for another robot has to say WHICH channel it
///   is missing rather than silently reading a neighbouring column.
int Find(const std::vector<std::string>& names, const std::string& name,
         const std::string& what) {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    throw std::runtime_error("clip pack has no " + what + " channel '" + name +
                             "'. The pack and the tracker describe different robots; "
                             "rebuild it with transfer/deploy/tools/dump_clip_pack.py.");
  }
  return static_cast<int>(std::distance(names.begin(), it));
}

/// The [x, y, z] triple of a frame's positions in a name list.
std::vector<int> PosTriple(const std::vector<std::string>& names, const std::string& frame,
                           const std::string& what) {
  return {Find(names, frame + ":pos_x", what), Find(names, frame + ":pos_y", what),
          Find(names, frame + ":pos_z", what)};
}

/// Gather `idx` columns out of row `row` of a [T, W] buffer.
Vec3 Gather3(const std::vector<double>& buf, int width, int row, const std::vector<int>& idx) {
  const double* p = buf.data() + static_cast<size_t>(row) * width;
  return Vec3(p[idx[0]], p[idx[1]], p[idx[2]]);
}

}  // namespace

ClipReference ClipReference::Load(const Config& cfg) {
  if (!cfg.root_frame_bodies.empty()) {
    throw std::runtime_error(
        "this tracker is tracking_mode=\"body\": it observes body_pos_ref / body_vel_ref, "
        "which the C++ clip path does not serve. Play it through the python controller "
        "(scripts/run_sim_clip.sh --python) or export a joint-tracking tracker.");
  }
  const BlobPack pack = BlobPack::Load(cfg.pack_dir);
  const Case* found = nullptr;
  std::string available;
  for (const Case* c : pack.ForFunction("clip_reference")) {
    available += (available.empty() ? "" : ", ") + c->name;
    if (c->name == cfg.case_name) found = c;
  }
  if (found == nullptr) {
    throw std::runtime_error("clip pack " + cfg.pack_dir + " has no case '" + cfg.case_name +
                             "'; it has {" + available + "}.");
  }
  const Case& c = *found;

  ClipReference out;
  const Blob& y = c.In("y_des");
  const Blob& dy = c.In("dy_des");
  const Blob& tv = c.In("target_vel");
  out.y_ = y.data;
  out.dy_ = dy.data;
  out.tv_ = tv.data;
  out.num_steps_ = y.rows();
  out.pos_width_ = y.stride();
  out.vel_width_ = dy.stride();
  out.dt_ = c.Double("dt");

  const auto pos_it = c.string_lists.find("pos_output_names");
  const auto vel_it = c.string_lists.find("vel_output_names");
  if (pos_it == c.string_lists.end() || vel_it == c.string_lists.end()) {
    throw std::runtime_error("clip pack case '" + c.name +
                             "' carries no pos_output_names/vel_output_names, so its channels "
                             "cannot be resolved by name. Rebuild the pack.");
  }
  const std::vector<std::string>& pos_names = pos_it->second;
  const std::vector<std::string>& vel_names = vel_it->second;
  if (static_cast<int>(pos_names.size()) != out.pos_width_ ||
      static_cast<int>(vel_names.size()) != out.vel_width_) {
    throw std::runtime_error("clip pack case '" + c.name + "' declares " +
                             std::to_string(pos_names.size()) + "/" +
                             std::to_string(vel_names.size()) + " channel names but its arrays "
                             "are " + std::to_string(out.pos_width_) + "/" +
                             std::to_string(out.vel_width_) + " wide.");
  }

  const auto real_it = c.ints.find("real_steps");
  out.real_steps_ = real_it == c.ints.end() ? out.num_steps_ : real_it->second;
  out.real_steps_ = std::max(1, std::min(out.real_steps_, out.num_steps_));

  // The motion span, if the library declared one; otherwise the whole real clip is motion.
  const auto ms = c.ints.find("motion_start_step");
  const auto me = c.ints.find("motion_end_step");
  out.motion_start_ = ms == c.ints.end() ? 0 : std::max(0, ms->second);
  out.motion_end_ = me == c.ints.end() ? out.real_steps_
                                       : std::min(me->second, out.real_steps_);
  if (out.motion_end_ <= out.motion_start_) {   // a nonsense span is worse than none
    out.motion_start_ = 0;
    out.motion_end_ = out.real_steps_;
  }

  // The clip's own authored frame, unless the caller names one. The pack carries it because
  // the library does, and the two libraries on disk differ.
  std::string base = cfg.base_frame_name;
  if (base.empty()) {
    const auto it = c.strings.find("ref_frame");
    base = it == c.strings.end() ? "pelvis_link" : it->second;
  }
  const std::string& b = base;
  out.ch_.base_pos = PosTriple(pos_names, b, "position");
  out.ch_.base_ori = {Find(pos_names, b + ":ori_x", "position"),
                      Find(pos_names, b + ":ori_y", "position"),
                      Find(pos_names, b + ":ori_z", "position"),
                      Find(pos_names, b + ":ori_w", "position")};
  out.ch_.base_lin = PosTriple(vel_names, b, "velocity");
  out.ch_.base_ang = {Find(vel_names, b + ":ori_x", "velocity"),
                      Find(vel_names, b + ":ori_y", "velocity"),
                      Find(vel_names, b + ":ori_z", "velocity")};
  for (const std::string& n : cfg.joint_names) {
    out.ch_.jp.push_back(Find(pos_names, "joint:" + n, "position"));
    out.ch_.jv.push_back(Find(vel_names, "joint:" + n, "velocity"));
  }
  // EE POSITIONS ONLY. The window's ee_pos block is positions, and unlike the python -- which
  // also serves state_at() to a MuJoCo spawn -- nothing here needs the orientations or twists.
  for (const std::string& fr : cfg.ee_frame_names) {
    out.ch_.ee_pos.push_back(PosTriple(pos_names, fr, "position"));
  }

  if (cfg.ego_anchor_every < 1) {
    throw std::runtime_error("clip ego_anchor_every must be >= 1, got " +
                             std::to_string(cfg.ego_anchor_every));
  }
  out.ego_anchor_every_ = cfg.ego_anchor_every;
  const long start = std::lround(cfg.start_time / out.dt_);
  out.start_step_ = static_cast<int>(
      std::min<long>(std::max<long>(start, 0), out.real_steps_ - 1));
  out.SetPlacement(Vec3::Zero(), QuatXyzw(0.0, 0.0, 0.0, 1.0));
  return out;
}

void ClipReference::SetPlacement(const Vec3& anchor_pos, const QuatXyzw& anchor_quat_xyzw) {
  anchor_pos_ = anchor_pos;
  inv_yaw_ = QuatInv(YawQuat(QuatInv(anchor_quat_xyzw)));
}

void ClipReference::ReanchorEgo(int step, const Vec3& robot_pos_w,
                                const QuatXyzw& robot_quat_xyzw) {
  const int s = std::min(std::max(step, 0), num_steps_ - 1);
  const Vec3 base_pos_cf = Gather3(y_, pos_width_, s, ch_.base_pos);
  const double* p = y_.data() + static_cast<size_t>(s) * pos_width_;
  const QuatXyzw base_ori_cf(p[ch_.base_ori[0]], p[ch_.base_ori[1]], p[ch_.base_ori[2]],
                             p[ch_.base_ori[3]]);
  const double yaw = YawAngle(robot_quat_xyzw) - YawAngle(base_ori_cf);
  // R(yaw) @ base_pos_cf[:2], written out: the python does it this way so the placement is a
  // planar rotation rather than a quaternion apply, and the two differ in the last ulp.
  const double c = std::cos(yaw), sn = std::sin(yaw);
  const double rot_x = c * base_pos_cf.x() - sn * base_pos_cf.y();
  const double rot_y = sn * base_pos_cf.x() + c * base_pos_cf.y();
  SetPlacement(Vec3(robot_pos_w.x() - rot_x,
                    robot_pos_w.y() - rot_y,
                    // z is yaw-invariant, so lifted_z == world_pos.z + base_pos_cf.z: this
                    // snaps the current base HEIGHT onto the robot root too.
                    robot_pos_w.z() - base_pos_cf.z()),
               YawQuatFromAngle(yaw));
}

Vec3 ClipReference::LiftPos(const Vec3& p) const { return anchor_pos_ + QuatApply(inv_yaw_, p); }
QuatXyzw ClipReference::LiftQuat(const QuatXyzw& q) const { return QuatMul(inv_yaw_, q); }
Vec3 ClipReference::LiftVec(const Vec3& v) const { return QuatApply(inv_yaw_, v); }

WorldNode ClipReference::FrameState(int step) const {
  const int s = std::min(std::max(step, 0), real_steps_ - 1);
  const double* y = y_.data() + static_cast<size_t>(s) * pos_width_;
  const double* dy = dy_.data() + static_cast<size_t>(s) * vel_width_;

  WorldNode node;
  node.ref_pos_w = LiftPos(Gather3(y_, pos_width_, s, ch_.base_pos));
  node.ref_quat_w = LiftQuat(QuatXyzw(y[ch_.base_ori[0]], y[ch_.base_ori[1]],
                                      y[ch_.base_ori[2]], y[ch_.base_ori[3]]));
  node.ref_lin_w = LiftVec(Gather3(dy_, vel_width_, s, ch_.base_lin));
  node.ref_ang_w = LiftVec(Gather3(dy_, vel_width_, s, ch_.base_ang));
  node.joint_pos.reserve(ch_.jp.size());
  node.joint_vel.reserve(ch_.jv.size());
  for (int i : ch_.jp) node.joint_pos.push_back(y[i]);
  for (int i : ch_.jv) node.joint_vel.push_back(dy[i]);
  node.ee_pos_w.reserve(ch_.ee_pos.size());
  for (const auto& idx : ch_.ee_pos) node.ee_pos_w.push_back(LiftPos(Gather3(y_, pos_width_, s, idx)));
  return node;
}

Vec3 ClipReference::CommandAt(int step) const {
  const int s = std::min(std::max(step, 0), real_steps_ - 1);
  if (InHold(s)) return Vec3::Zero();
  const double* p = tv_.data() + static_cast<size_t>(s) * 3;
  return Vec3(p[0], p[1], p[2]);
}

void ClipReference::ReferenceAt(int step, const std::vector<int>& offsets,
                                std::vector<WorldNode>* out) const {
  out->clear();
  out->reserve(offsets.size());
  for (int o : offsets) out->push_back(FrameState(step + o));
}

void ClipReference::CommandWindow(int step, const std::vector<int>& offsets,
                                  std::vector<double>* out) const {
  out->clear();
  out->reserve(offsets.size() * 3);
  for (int o : offsets) {
    const Vec3 c = CommandAt(step + o);
    out->push_back(c.x());
    out->push_back(c.y());
    out->push_back(c.z());
  }
}

void ClipReference::Serve(int tick, const Vec3& robot_pos, const QuatXyzw& robot_quat,
                          const std::vector<int>& offsets, int num_joints, int num_ee_frames,
                          const std::vector<ReferenceBlock>& blocks, double* window_out,
                          std::vector<double>* command_out) {
  const int step = StepForTick(tick);
  // Both on the SAME tick, from the SAME pose. See the header for why they cannot be split.
  if (!latched_ || tick % ego_anchor_every_ == 0) {
    ReanchorEgo(step, robot_pos, robot_quat);
    latched_pos_ = robot_pos;
    latched_quat_ = robot_quat;
    latched_ = true;
  }
  ReferenceAt(step, offsets, &scratch_);
  WorldToWindow(latched_pos_, latched_quat_, scratch_, num_joints, num_ee_frames, blocks,
                window_out);
  if (command_out != nullptr) CommandWindow(step, offsets, command_out);
}

}  // namespace g1_deploy
