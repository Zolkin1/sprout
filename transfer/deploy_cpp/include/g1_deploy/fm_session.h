// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// fm.onnx behind the FmSampler seam: the generator that GenTrackerCore asks for plans.
//
// Port of transfer/deploy/fm_session.py. The graph folds in every normalization, so the
// inputs are RAW:
//
//   history      (1, N, D)                  the N window nodes this runtime last served
//   extero       (1, extero_tokens*height_dim)   depth or height-scan reading at "now"
//   cmd_tokens   (1, num_cmd_tokens, cmd_dim)    the future command, strided by cmd_steps
//   noise        (1, F, D)                  the sampler's starting point
//   history_null (1, 1)      OPTIONAL       swap the history for the learned null embedding
//   -> trajectory (1, F, D)
//
// The history needs no transform because it is the generator's OWN past output, already
// relative to the robot at the step each node was served. That is what took odometry off the
// generator's critical path -- an earlier version built these from measured world-frame root
// poses, which hardware only has via FAST-LIO.
//
// This implementation is SYNCHRONOUS: Submit runs the graph and TryCollect hands the result
// over. The threaded worker wraps this rather than replacing it, so the same code path is
// exercised by the single-threaded tests and on the robot.

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "g1_deploy/fm_meta.h"
#include "g1_deploy/gen_tracker_core.h"
#include "g1_deploy/ort_session.h"
#include "g1_deploy/window.h"

namespace g1_deploy {


/// The exteroception and command a plan is conditioned on, sampled at request time.
struct FmConditioning {
  std::vector<double> extero;      //: extero_tokens * height_dim, raw
  std::vector<double> future_cmd;  //: F * cmd_dim, the command profile over the window
};

/// Draws the sampler's starting point x0.
///
/// Deliberately NOT a port of numpy's PCG64 + ziggurat: matching that bit for bit is ~200
/// lines of surface for a draw that is supposed to be random. Parity with the python is
/// established instead with zero noise (exactly deterministic on both sides) and, in tests,
/// by replaying recorded draws.
class NoiseSource {
 public:
  virtual ~NoiseSource() = default;
  virtual void Fill(double* out, size_t count) = 0;
};

/// x0 = 0. Not the mean of the conditional distribution -- the flow ODE is nonlinear, so the
/// image of the source mode is not the target mean -- but fully deterministic per
/// conditioning, which removes sample-to-sample variability when attributing a re-plan
/// discontinuity, and it empirically tracks better in MuJoCo sim2sim.
class ZeroNoise : public NoiseSource {
 public:
  void Fill(double* out, size_t count) override;
};

/// x0 ~ N(0, I), what the rectified flow was trained to transport. The library default.
class GaussianNoise : public NoiseSource {
 public:
  explicit GaussianNoise(uint64_t seed) : rng_(seed) {}
  void Fill(double* out, size_t count) override;

 private:
  std::mt19937_64 rng_;
  std::normal_distribution<double> dist_{0.0, 1.0};
};

/// fm.onnx as an FmSampler.
class FmSession : public FmSampler {
 public:
  FmSession(const std::string& onnx_path, const std::vector<Provider>& providers,
            int intra_threads, std::shared_ptr<NoiseSource> noise);

  /// The conditioning for the NEXT Submit. Set by the caller each time it requests a plan;
  /// the history rides in the FmRequest itself.
  void SetConditioning(const FmConditioning& cond);

  void Submit(const FmRequest& req) override;
  bool TryCollect(double* pred_out, int64_t* target_ctrl_step) override;

  const FmMeta& meta() const { return meta_; }
  OrtModel& model() { return model_; }
  /// Wall time of the last inference, milliseconds.
  double last_infer_ms() const { return last_infer_ms_.load(std::memory_order_relaxed); }

  /// Inferences run, ever.
  ///
  /// Counted HERE rather than taken from FmWorker::plans_produced() so it is the same number
  /// in both modes: with `threaded_fm: false` there is no worker, and a timing report keyed
  /// on the worker's counter would silently never sample.
  uint64_t infer_count() const { return infer_count_.load(std::memory_order_relaxed); }

 private:
  OrtModel model_;
  FmMeta meta_;
  std::shared_ptr<NoiseSource> noise_;
  FmConditioning cond_;

  // Preallocated feeds: this runs on a control-rate path, so nothing allocates per call.
  std::vector<float> history_f_;
  std::vector<float> extero_f_;
  std::vector<float> cmd_tokens_f_;
  std::vector<float> noise_f_;
  std::vector<float> null_f_;
  std::vector<double> noise_d_;

  bool ready_ = false;
  int64_t ready_target_ = 0;
  std::vector<double> ready_pred_;
  // ATOMIC because it crosses a thread: FmWorker runs Submit (the writer) on its own thread
  // while the control tick reads this through PolicyRuntime::Step's diagnostics. A plain
  // double here was a data race -- benign in practice on this hardware, but the timing report
  // it now feeds makes it load-bearing.
  std::atomic<double> last_infer_ms_{0.0};
  std::atomic<uint64_t> infer_count_{0};
};

}  // namespace g1_deploy
