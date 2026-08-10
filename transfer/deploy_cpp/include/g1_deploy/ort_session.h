// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Session management for the ONNX runtime: providers, metadata, and the CUDA guard.
//
// Port of the session half of transfer/deploy/tracker_session.py. The metadata PARSING lives
// in policy_params.h, deliberately separate, so it is testable without a model.
//
// ============================ the CUDA guard ==============================================
// onnxruntime falls back to the CPU SILENTLY. For this policy pair that is not a degraded
// mode, it is an unusable one -- the python measured 103.5 ms per generator tick on CPU
// against a 20 ms control period. The python guard compared the requested provider against
// what each session registered; the C++ API has no equivalent accessor, so this uses a
// stronger check instead: profile the warm-up (which the runtime needs anyway) and require a
// node to have actually been ASSIGNED to CUDA. That reports placement rather than capability,
// and EndProfiling turns profiling off, so the hot loop pays nothing.

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

namespace g1_deploy {

/// Execution providers, in the order they are offered to onnxruntime.
enum class Provider { kCpu, kCuda, kTensorrt };

/// Parse a provider name: "cpu", "cuda" or "trt"/"tensorrt".
Provider ProviderFromName(const std::string& name);
std::string ProviderName(Provider p);

/// One process-wide Ort::Env. Creating several is legal but wasteful, and each spins up its
/// own thread pool.
Ort::Env& GlobalOrtEnv();

/// A loaded model plus the bits every caller needs from it.
class OrtModel {
 public:
  /// Load `path`, offering `providers` in order with CPU appended as the final fallback --
  /// which is what makes a GPU request degrade rather than fail on a box without one.
  ///
  /// Args:
  ///   intra_threads: intra-op threads; 0 lets onnxruntime choose. Measured on this stack,
  ///     auto was ~3x slower than 4 on the generator, so it is worth setting explicitly and
  ///     re-measuring per box.
  OrtModel(const std::string& path, const std::vector<Provider>& providers, int intra_threads);
  ~OrtModel();

  /// A custom metadata entry, e.g. "policy_parameters" or "fm_meta". Throws if absent.
  std::string Metadata(const std::string& key) const;

  /// Run the model. Input and output names must match the graph.
  void Run(const std::vector<const char*>& input_names,
           const std::vector<Ort::Value>& inputs,
           const std::vector<const char*>& output_names,
           std::vector<Ort::Value>* outputs);

  /// Declared shape of an input, with dynamic dimensions resolved to 1.
  std::vector<int64_t> InputShape(const std::string& name) const;

  /// Whether the graph declares an input of this name. The generator's `history_null` is
  /// optional: a graph exported without it cannot use the learned null-history embedding.
  bool HasInput(const std::string& name) const;

  /// The graph's own input/output names. Read rather than assumed: a hardcoded name is a
  /// restatement of the contract that can disagree with the artifact, and it fails at load
  /// with "Invalid output name" instead of at export where the mistake was made.
  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const { return output_names_; }

  /// True if at least one node was assigned to CUDA during the warm-up.
  ///
  /// Only meaningful after WarmUpAndCheckPlacement.
  bool on_gpu() const { return on_gpu_; }

  /// Run the graph `iters` times on zeros with profiling on, then record whether any node
  /// landed on CUDA and turn profiling off.
  ///
  /// Warming up is not optional regardless: the first CUDA run builds the context and picks
  /// cuDNN algorithms, hundreds of milliseconds that must never land on a live control tick.
  void WarmUpAndCheckPlacement(int iters = 5);

  Ort::Session& session() { return *session_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  Ort::SessionOptions options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions alloc_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  bool on_gpu_ = false;
  bool profiling_enabled_ = false;
};

/// Refuse to run when CUDA was requested but the model is on the CPU.
///
/// Throws with the same content the python guard carried, because the causes are the same:
/// the no-GPU compose file, a container without the nvidia runtime, or a deliberate CPU run
/// that has not accounted for the deficit.
void RequireCuda(const OrtModel& model, const std::vector<Provider>& requested);

}  // namespace g1_deploy
