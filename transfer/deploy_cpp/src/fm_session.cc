// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/fm_session.h"


#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace g1_deploy {
namespace {

using OrderedJson = nlohmann::ordered_json;


}  // namespace

FmMeta ParseFmMeta(const std::string& json) {
  OrderedJson root;
  try {
    root = OrderedJson::parse(json);
  } catch (const std::exception& e) {
 count) {
  for (size_t i = 0; i < count; ++i) out[i] = dist_(rng_);
}

FmSession::FmSession(const std::string& onnx_path, const std::vector<Provider>& providers,
                     int intra_threads, std::shared_ptr<NoiseSource> noise)
    : model_(onnx_path, providers, intra_threads),
      meta_(ParseFmMeta(model_.Metadata("fm_meta"))),
      noise_(std::move(noise)) {
  if (!noise_) throw std::runtime_error("FmSession needs a NoiseSource");
  // has_history_null comes from the GRAPH, not the metadata: what matters is whether the
  // input exists to feed, and a graph exported before the embedding simply has no such input.
  meta_.has_history_null = model_.HasInput("history_null");
  if (!meta_.has_history_null) {
    // Said at LOAD, not on the first tick that needs it. Without this input the learned
    // null-history embedding cannot be reached at all, so the generator conditions on a
    // window that is partly the zero placeholder and is told nothing about it -- the same
    // failure the obelisk PD warm-up caused, where history rows of zeros flipped null_mask
    // off and denied the model the embedding meant to cover exactly those steps. It is
    // survivable, so this warns rather than refuses, but it must not be silent.
    std::fprintf(stderr,
                 "[WARN FmSession] %s has no 'history_null' input, so the learned "
                 "null-history embedding cannot be used. The first N steps after arming will "
                 "condition on a not-yet-full history window with no indication that it is "
                 "incomplete. Re-export with scripts/flow_matching/export_fm_onnx.py.\n",
                 onnx_path.c_str());
  }

  // The exteroception width, checked against the GRAPH rather than trusted from the metadata.
  // This is what makes ParseFmMeta accepting two key spellings safe rather than a guess: if an
  // export ever changes the actual contract instead of merely renaming its description, it is
  // caught here, at load, instead of as an opaque onnxruntime shape error on the first Run --
  // or, worse, not at all, if the widths happen to coincide.
  {
    const std::vector<int64_t> shape = model_.InputShape("extero");
    if (!shape.empty() && shape.back() > 0 && shape.back() != meta_.extero_dim()) {
      throw std::runtime_error(
          "fm_meta describes exteroception as " + std::to_string(meta_.extero_tokens) + " x " +
          std::to_string(meta_.height_dim) + " = " + std::to_string(meta_.extero_dim()) +
          " values, but the graph's 'extero' input is " + std::to_string(shape.back()) +
          " wide. The metadata does not describe this model.");
    }
  }

  history_f_.assign(static_cast<size_t>(meta_.n) * meta_.d, 0.0f);
  extero_f_.assign(static_cast<size_t>(meta_.extero_dim()), 0.0f);
  cmd_tokens_f_.assign(static_cast<size_t>(meta_.num_cmd_tokens) * meta_.cmd_dim, 0.0f);
  noise_f_.assign(static_cast<size_t>(meta_.f) * meta_.d, 0.0f);
  noise_d_.assign(noise_f_.size(), 0.0);
  null_f_.assign(1, 0.0f);
  ready_pred_.assign(static_cast<size_t>(meta_.f) * meta_.d, 0.0);

  cond_.extero.assign(static_cast<size_t>(meta_.extero_dim()), 0.0);
  cond_.future_cmd.assign(static_cast<size_t>(meta_.f) * meta_.cmd_dim, 0.0);
}

void FmSession::SetConditioning(const FmConditioning& cond) {
  if (cond.extero.size() != static_cast<size_t>(meta_.extero_dim())) {
    throw std::runtime_error("exteroception is " + std::to_string(cond.extero.size()) +
                             " wide but this generator expects " +
                             std::to_string(meta_.extero_dim()));
  }
  if (cond.future_cmd.size() != static_cast<size_t>(meta_.f) * meta_.cmd_dim) {
    throw std::runtime_error("the future command profile must be F * cmd_dim");
  }
  cond_ = cond;
}

void FmSession::Submit(const FmRequest& req) {
  if (req.history_nodes.size() != history_f_.size()) {
    throw std::runtime_error("history is " + std::to_string(req.history_nodes.size()) +
                             " values but this generator expects N*D = " +
                             std::to_string(history_f_.size()));
  }
  for (size_t i = 0; i < history_f_.size(); ++i) {
    history_f_[i] = static_cast<float>(req.history_nodes[i]);
  }
  for (size_t i = 0; i < extero_f_.size(); ++i) {
    extero_f_[i] = static_cast<float>(cond_.extero[i]);
  }
  // Stride the per-step command profile into tokens at cmd_steps.
  for (int t = 0; t < meta_.num_cmd_tokens; ++t) {
    const int step = meta_.cmd_steps[static_cast<size_t>(t)];
    for (int c = 0; c < meta_.cmd_dim; ++c) {
      cmd_tokens_f_[static_cast<size_t>(t) * meta_.cmd_dim + c] =
          static_cast<float>(cond_.future_cmd[static_cast<size_t>(step) * meta_.cmd_dim + c]);
    }
  }
  noise_->Fill(noise_d_.data(), noise_d_.size());
  for (size_t i = 0; i < noise_f_.size(); ++i) noise_f_[i] = static_cast<float>(noise_d_[i]);
  null_f_[0] = req.history_null ? 1.0f : 0.0f;

  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t hist_shape[3] = {1, meta_.n, meta_.d};
  const int64_t extero_shape[2] = {1, meta_.extero_dim()};
  const int64_t cmd_shape[3] = {1, meta_.num_cmd_tokens, meta_.cmd_dim};
  const int64_t noise_shape[3] = {1, meta_.f, meta_.d};
  const int64_t null_shape[2] = {1, 1};

  std::vector<const char*> names = {"history", "extero", "cmd_tokens", "noise"};
  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<float>(mem, history_f_.data(), history_f_.size(),
                                                   hist_shape, 3));
  inputs.push_back(Ort::Value::CreateTensor<float>(mem, extero_f_.data(), extero_f_.size(),
                                                   extero_shape, 2));
  inputs.push_back(Ort::Value::CreateTensor<float>(mem, cmd_tokens_f_.data(),
                                                   cmd_tokens_f_.size(), cmd_shape, 3));
  inputs.push_back(Ort::Value::CreateTensor<float>(mem, noise_f_.data(), noise_f_.size(),
                                                   noise_shape, 3));
  if (meta_.has_history_null) {
    names.push_back("history_null");
    inputs.push_back(Ort::Value::CreateTensor<float>(mem, null_f_.data(), null_f_.size(),
                                                     null_shape, 2));
  }

  const std::vector<const char*> outputs = {"trajectory"};
  std::vector<Ort::Value> result;
  const auto t0 = std::chrono::steady_clock::now();
  model_.Run(names, inputs, outputs, &result);
  last_infer_ms_.store(
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
      std::memory_order_relaxed);
  infer_count_.fetch_add(1, std::memory_order_relaxed);

  const float* traj = result[0].GetTensorData<float>();
  for (size_t i = 0; i < ready_pred_.size(); ++i) {
    ready_pred_[i] = static_cast<double>(traj[i]);
  }
  ready_ = true;
  ready_target_ = req.target_ctrl_step;
}

bool FmSession::TryCollect(double* pred_out, int64_t* target_ctrl_step) {
  if (!ready_) return false;
  std::copy(ready_pred_.begin(), ready_pred_.end(), pred_out);
  *target_ctrl_step = ready_target_;
  ready_ = false;
  return true;
}

}  // namespace g1_deploy
