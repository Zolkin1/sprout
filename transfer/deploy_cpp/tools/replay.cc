// Copyright 2026 Zachary Olkin. All rights reserved.

// Run THIS runtime over a tick stream recorded by the python one, and write what it computed.
//
// The async C++ stack tracks worse than the synchronous python sim2sim on the same artifact
// pair. Two causes produce that -- the C++ computing something different, or the C++ being fed
// something different -- and nothing separated them. This removes the second: identical
// inputs, no threads (threaded_fm = false), no stochastic sampling (zero_noise), no MuJoCo,
// no ROS. What is left to differ is code.
//
// TWO MODES, and the difference is which rung of the ladder they answer:
//
//   --forced  Teacher-forced. The served window and the previous action come from the
//             RECORDING, so nothing this runtime computes feeds the next tick's input and
//             every tick is an INDEPENDENT test. This is the rung that should be exact, and
//             the only one whose divergences can be counted as a rate rather than a first
//             failure.
//   (default) Free-running. Only the sensor readings are fed; the window and the previous
//             action are this runtime's own. Autoregressive, so the first divergence is the
//             finding and everything after it is a consequence. NOTE this rung cannot be
//             bit-compared while the generator lead is >= 1 tick: the C++ conditions each
//             plan on exteroception from lead ticks earlier and the python does not, which is
//             a design difference, not a bug.
//
// Output is plain raw little-endian float64, one file per field, C order [ticks, width]. No
// manifest: the comparator reads the INPUT stream's manifest for ticks and widths, so a
// second format writer -- and a second thing to keep in step -- is avoided.
//
//   replay <stream_dir> --tracker t.onnx --fm fm.onnx [--forced] [--cuda] [--out dir]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fixture.h"
#include "g1_deploy/policy_runtime.h"

namespace {

using g1_deploy::test::Case;
using g1_deploy::test::Fixtures;

/// Write a [ticks, width] row-major buffer as raw float64.
///
/// \param path Destination.
/// \param rows The rows; every row must be `width` long.
/// \param width Values per row, checked so a short row cannot silently shift the file.
void WriteBlob(const std::string& path, const std::vector<std::vector<double>>& rows,
               size_t width) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  for (const auto& row : rows) {
    if (row.size() != width) {
      throw std::runtime_error("row width " + std::to_string(row.size()) + " != " +
                               std::to_string(width) + " writing " + path);
    }
    out.write(reinterpret_cast<const char*>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(double)));
  }
}

/// One argument's value, or the fallback.
std::string Arg(int argc, char** argv, const std::string& flag, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (flag == argv[i]) return argv[i + 1];
  }
  return fallback;
}

/// Whether a bare flag is present.
bool Flag(int argc, char** argv, const std::string& flag) {
  for (int i = 1; i < argc; ++i) {
    if (flag == argv[i]) return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: replay <stream_dir> --tracker t.onnx --fm fm.onnx "
                 "[--forced] [--zero_lead] [--cuda] [--out dir]\n";
    return 2;
  }
  const std::string stream_dir = argv[1];
  const std::string tracker = Arg(argc, argv, "--tracker", "");
  const std::string fm = Arg(argc, argv, "--fm", "");
  const bool forced = Flag(argc, argv, "--forced");
  const std::string out_dir = Arg(argc, argv, "--out", stream_dir);
  if (tracker.empty()) {
    std::cerr << "--tracker is required: the stream records which artifacts produced it, but "
                 "replaying against a DIFFERENT pair is a legitimate thing to ask for, so the "
                 "paths are stated here rather than taken from the recording.\n";
    return 2;
  }

  try {
    const Fixtures fixtures = Fixtures::Load(stream_dir);
    auto cases = fixtures.ForFunction("policy_runtime_step");
    if (cases.empty()) {
      std::cerr << "no policy_runtime_step case in " << stream_dir << "\n";
      return 2;
    }
    const Case& c = *cases.front();
    const int ticks = c.In("root_pos").rows();

    g1_deploy::RuntimeConfig cfg;
    cfg.tracker_onnx = tracker;
    cfg.fm_onnx = fm;
    // THE POINT of this tool: synchronous and deterministic. Threading is what the comparison
    // is trying to hold still, and a stochastic x0 would make two runs of THIS binary differ.
    cfg.threaded_fm = false;
    cfg.zero_noise = true;
    if (Flag(argc, argv, "--zero_lead")) {
      // Request each plan AT its boundary, as the python runtime does. Without this the two
      // condition their generators on different instants and their plans cannot be compared.
      cfg.allow_zero_lead = true;
      cfg.fm_lead_sec = 0.0;
    }
    cfg.providers = {Flag(argc, argv, "--cuda") ? g1_deploy::Provider::kCuda
                                                : g1_deploy::Provider::kCpu};
    g1_deploy::PolicyRuntime policy(cfg);
    policy.Load();
    policy.Reset();

    const auto& params = policy.params();
    const size_t nj = static_cast<size_t>(params.num_actions);
    const size_t n_cams = params.depth_cameras.size();
    const size_t px = n_cams ? static_cast<size_t>(c.In("depth_raw").stride()) / n_cams : 0;
    std::printf("[replay] %d ticks, %zu joints, %zu camera(s) x %zu px, mode=%s\n",
                ticks, nj, n_cams, px, forced ? "teacher-forced" : "free-running");

    std::vector<std::vector<double>> obs_rows, act_rows, win_rows, sum_rows, meta_rows;
    // What the generator was CONDITIONED on, per tick. Pure arithmetic on both sides, so a
    // difference here is a bug rather than a tolerance question -- and with history_null on,
    // the first plan is a function of nothing else.
    std::vector<std::vector<double>> extero_rows, futcmd_rows, histn_rows, histc_rows;
    std::vector<double> targets(nj, 0.0);

    for (int t = 0; t < ticks; ++t) {
      g1_deploy::HardwareReadings r;
      const double* p = c.In("root_pos").row(t);
      r.root_pos = g1_deploy::Vec3(p[0], p[1], p[2]);
      const double* q = c.In("base_quat_xyzw").row(t);
      r.base_quat_xyzw = g1_deploy::QuatXyzw(q[0], q[1], q[2], q[3]);
      const double* w = c.In("base_ang_vel_b").row(t);
      r.base_ang_vel_b = g1_deploy::Vec3(w[0], w[1], w[2]);
      const double* cmd = c.In("command").row(t);
      r.command = g1_deploy::Vec3(cmd[0], cmd[1], cmd[2]);

      // The stream's joints are in the POLICY's order, so naming them with the artifact's own
      // order makes the runtime's permutation the identity -- no reordering to get wrong here.
      r.joint_names = params.joint_names_isaac;
      r.joint_pos.assign(c.In("joint_pos").row(t), c.In("joint_pos").row(t) + nj);
      r.joint_vel.assign(c.In("joint_vel").row(t), c.In("joint_vel").row(t) + nj);

      if (n_cams) {
        const double* d = c.In("depth_raw").row(t);
        for (size_t cam = 0; cam < n_cams; ++cam) {
          r.depth_raw.emplace_back(d + cam * px, d + (cam + 1) * px);
        }
        const double* cp = c.In("fm_camera_pos").row(t);
        const double* cr = c.In("fm_camera_rot").row(t);
        for (size_t cam = 0; cam < n_cams; ++cam) {
          r.fm_camera_pos.push_back({cp[cam * 3], cp[cam * 3 + 1], cp[cam * 3 + 2]});
          std::array<double, 9> rot{};
          for (size_t i = 0; i < 9; ++i) rot[i] = cr[cam * 9 + i];
          r.fm_camera_rot.push_back(rot);
        }
      }
      if (c.In("height_scan").stride() > 0) {
        r.height_scan.assign(c.In("height_scan").row(t),
                             c.In("height_scan").row(t) + c.In("height_scan").stride());
      }
      if (forced) {
        const size_t win_w = static_cast<size_t>(c.In("prev_action").stride());
        r.forced_prev_action.assign(c.In("prev_action").row(t),
                                    c.In("prev_action").row(t) + win_w);
        const size_t ww = static_cast<size_t>(c.Out("window").stride());
        r.forced_window.assign(c.Out("window").row(t), c.Out("window").row(t) + ww);
      }

      g1_deploy::TickDiagnostics diag;
      policy.Step(r, params.joint_names_isaac, targets.data(), &diag);

      const auto& obs = policy.last_obs();
      obs_rows.emplace_back(obs.begin(), obs.end());
      const auto& act = policy.last_action();
      act_rows.emplace_back(act.begin(), act.end());
      win_rows.push_back(policy.last_reference_window());
      // Split, matching replay_stream.split_checksum: a 64-bit digest does not survive a
      // float64 whole (exact only to 2^53), and rounding it would fabricate mismatches.
      sum_rows.push_back({static_cast<double>(diag.obs_checksum >> 32),
                          static_cast<double>(diag.obs_checksum & 0xFFFFFFFFu)});
      const auto& cond = policy.last_conditioning();
      extero_rows.push_back(cond.extero);
      futcmd_rows.push_back(cond.future_cmd);
      // The history the LIVE plan was requested with -- this runtime's own served nodes.
      const auto& req = policy.last_plan_request();
      histn_rows.push_back(req.history_nodes);
      histc_rows.push_back(req.history_cmd);
      meta_rows.push_back({static_cast<double>(diag.ctrl_step),
                           diag.plan_adopted ? 1.0 : 0.0,
                           static_cast<double>(diag.plan_id),
                           diag.late ? 1.0 : 0.0,
                           // The mode, recorded so the comparator knows which of its
                           // comparisons are meaningful. In forced mode the window BELOW is
                           // this runtime's own plan while the observation used the recorded
                           // one, so the two plans are not comparable and saying so is the
                           // difference between a finding and a false red.
                           forced ? 1.0 : 0.0,
                           // The history_null the live plan was requested with. Compared
                           // against python's: if the two disagree the generator is reading
                           // its history tokens on one side and the learned null embedding on
                           // the other, which is a different function, not a rounding gap.
                           policy.last_plan_request().target_ctrl_step == 0 &&
                                   policy.last_plan_request().history_nodes.empty()
                               ? -1.0
                               : (policy.last_plan_request().history_null ? 1.0 : 0.0)});
    }

    WriteBlob(out_dir + "/cpp.obs.bin", obs_rows, obs_rows.front().size());
    WriteBlob(out_dir + "/cpp.action.bin", act_rows, act_rows.front().size());
    WriteBlob(out_dir + "/cpp.window.bin", win_rows, win_rows.front().size());
    WriteBlob(out_dir + "/cpp.checksum.bin", sum_rows, 2);
    WriteBlob(out_dir + "/cpp.meta.bin", meta_rows, 6);
    if (!extero_rows.front().empty()) {
      WriteBlob(out_dir + "/cpp.fm_extero.bin", extero_rows, extero_rows.front().size());
    }
    if (!futcmd_rows.front().empty()) {
      WriteBlob(out_dir + "/cpp.fm_future_cmd.bin", futcmd_rows, futcmd_rows.front().size());
    }
    // Widths can be zero on the very first ticks, before a request has been built; the
    // comparator skips a field the C++ did not write rather than comparing against nothing.
    if (!histn_rows.back().empty()) {
      WriteBlob(out_dir + "/cpp.fm_history_nodes.bin", histn_rows, histn_rows.back().size());
    }
    if (!histc_rows.back().empty()) {
      WriteBlob(out_dir + "/cpp.fm_history_cmd.bin", histc_rows, histc_rows.back().size());
    }
    std::printf("[replay] wrote %zu ticks of obs/action/window/checksum/meta to %s\n",
                obs_rows.size(), out_dir.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[replay] " << e.what() << "\n";
    return 1;
  }
}
