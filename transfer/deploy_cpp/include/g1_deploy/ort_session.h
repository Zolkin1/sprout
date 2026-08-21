// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Session management for the ONNX runtime: providers, metadata, and the CUDA guard.
//
// Port of the session half of transfer/deploy/tracker_session.py. The metadata PARSING lives
// in policy_params.h, deliberately separate, so it is testable without a model.
//
// ============================ CUDA Graph capture ==========================================
// The generator's graph is ~3000 kernels for a 62-node plan at batch 1, so onnxruntime is
// LAUNCH-bound rather than arithmetic-bound: wall time tracks the number of kernels, not the
// FLOPs. CUDA Graph capture records the whole launch sequence once and replays it as a single
// submission, which measured 19.3 -> 14-15 ms per re-plan on the Jetson Thor and 9.8 -> 8.3 ms
// on an RTX 4070.
//
// It has three hard requirements, and violating any of them is SILENT:
//
//   1. Every node on CUDA. onnxruntime refuses session creation outright otherwise ("all
//      compute graph nodes have not been partitioned to the CUDAExecutionProvider"), which is
//      why this is only reachable with an fm.onnx exported at a FIXED batch -- a symbolic
//      batch axis leaves ~2700 shape-arithmetic nodes on the host. See
//      scripts/flow_matching/export_fm_onnx.py::_export_graph.
//   2. Static shapes, for the same reason.
//   3. **Fixed device addresses.** A captured graph holds raw device pointers, so feeding new
//      data means copying into the SAME device buffers. This is the dangerous one, and it was
//      measured rather than assumed (transfer/deploy/tests/test_fm_cuda_graph.py):
//        - a plain Run() with host tensors returns an EMPTY output list from the second call on;
//        - binding host memory, or rebinding fresh host buffers, returns the FIRST call's
//          answer forever -- bit-identical to it, with no error;
//        - even mutating the same host buffer in place is not picked up, because the
//          host-to-device copy is itself inside the capture.
//      So RunPersistent owns device tensors allocated once and copies into them through
//      Ort::Env::CopyTensors (ORT >= 1.23), which needs no CUDA linkage of our own.
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
  ///   capture_cuda_graph: record the launch sequence once and replay it (see the note at the
  ///     top of this header). Requires a CUDA provider, a fixed-batch graph, and that every
  ///     call go through PreparePersistentIo + RunPersistent -- Run() is not capture-safe.
  OrtModel(const std::string& path, const std::vector<Provider>& providers, int intra_threads,
           bool capture_cuda_graph = false);
  ~OrtModel();

  /// A custom metadata entry, e.g. "policy_parameters" or "fm_meta". Throws if absent.
  std::string Metadata(const std::string& key) const;

  /// Run the model. Input and output names must match the graph.
  void Run(const std::vector<const char*>& input_names,
           const std::vector<Ort::Value>& inputs,
           const std::vector<const char*>& output_names,
           std::vector<Ort::Value>* outputs);

  /// Allocate device-resident tensors for `input_names` (shapes read off the graph) and for
  /// the single graph output, and bind them once.
  ///
  /// Capture-only, and throws without it: the addresses bound here are the ones the capture
  /// bakes in, so this path exists to keep them fixed. A session without capture uses Run().
  void PreparePersistentIo(const std::vector<std::string>& input_names);

  /// One inference over those buffers: host -> device, run, device -> host.
  ///
  /// `input_data[i]` supplies the input named `input_names[i]` from PreparePersistentIo, with
  /// exactly the element count that input's shape implies. The first call records the graph and
  /// the rest replay it; the result is bit-identical to what Run() gives on a plain session,
  /// which is what test/ort_sessions_test.cc checks.
  void RunPersistent(const std::vector<const float*>& input_data, float* output,
                     size_t output_count);

  /// Whether this session was created with capture enabled.
  bool cuda_graph() const { return capture_cuda_graph_; }

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
  bool capture_cuda_graph_ = false;

  // ---- persistent IO (PreparePersistentIo / RunPersistent) --------------------------------
  // Device tensors OWN their memory and must outlive every replay, hence members rather than
  // locals. The host staging tensors are views over host_*_ and are re-used, so the hot path
  // allocates nothing.
  std::unique_ptr<Ort::Allocator> device_alloc_;
  std::vector<std::string> bound_input_names_;
  std::vector<Ort::Value> device_inputs_;
  std::vector<Ort::Value> host_inputs_;
  std::vector<std::vector<float>> host_input_buf_;
  std::vector<size_t> input_counts_;
  // One-element vectors, not bare Values: Ort::Env::CopyTensors takes vectors by const
  // reference and Ort::Value is move-only, so wrapping a member in a temporary vector would
  // move the buffer out from under the binding.
  std::vector<Ort::Value> device_output_;
  std::vector<Ort::Value> host_output_;
  std::vector<float> host_output_buf_;
  std::unique_ptr<Ort::IoBinding> binding_;
};

/// Refuse to run when CUDA was requested but the model is on the CPU.
///
/// Throws with the same content the python guard carried, because the causes are the same:
/// the no-GPU compose file, a container without the nvidia runtime, or a deliberate CPU run
/// that has not accounted for the deficit.
void RequireCuda(const OrtModel& model, const std::vector<Provider>& requested);

}  // namespace g1_deploy
