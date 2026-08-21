// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/ort_session.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace g1_deploy {
namespace {

/// Zero-filled inputs matching the graph's declared signature, for the warm-up.
struct WarmupInput {
  std::vector<float> f32;
  std::vector<int64_t> i64;
  std::vector<char> boolean;
  Ort::Value value{nullptr};
};

/// OrtCUDAProviderOptionsV2, owned so it is released on every exit path.
///
/// The V2 API rather than the legacy OrtCUDAProviderOptions struct because that struct has no
/// field for `enable_cuda_graph`; it is reachable only as a string key/value.
struct CudaOptions {
  OrtCUDAProviderOptionsV2* p = nullptr;
  CudaOptions() { Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&p)); }
  ~CudaOptions() {
    if (p != nullptr) Ort::GetApi().ReleaseCUDAProviderOptions(p);
  }
  CudaOptions(const CudaOptions&) = delete;
  CudaOptions& operator=(const CudaOptions&) = delete;
};

/// Give the process-wide Env a CUDA data-transfer implementation, once.
///
/// Ort::Env::CopyTensors resolves its host<->device copy from the execution providers
/// registered on the ENV, which is a different registry from the one a SessionOptions appends
/// to. So a process can hold a perfectly good CUDA session and still be told "Data transfer
/// implementation between source and destination device was not found" -- an error about the
/// Env, raised on the first copy, that reads like a broken GPU.
///
/// The bare soname rather than a path: the provider .so sits beside libonnxruntime.so, which
/// the loader has already found by the time this runs, so the same lookup finds this. An
/// absolute /opt prefix would be a second statement of where onnxruntime is installed, and
/// the one the image is free to move.
void EnsureCudaDataTransfer() {
  static bool done = false;
  if (done) return;
  done = true;   // set FIRST: a second attempt would throw "already registered", not recover.
  try {
    GlobalOrtEnv().RegisterExecutionProviderLibrary("g1_cuda_ep",
                                                    std::string("libonnxruntime_providers_cuda.so"));
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("could not register the CUDA provider library on the onnxruntime Env, so "
                    "CUDA Graph capture has no way to copy its inputs to the device: ") +
        e.what());
  }
}

/// Element count implied by a shape.
size_t Elements(const std::vector<int64_t>& shape) {
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

}  // namespace

Provider ProviderFromName(const std::string& name) {
  if (name == "cpu") return Provider::kCpu;
  if (name == "cuda") return Provider::kCuda;
  if (name == "trt" || name == "tensorrt") return Provider::kTensorrt;
  throw std::runtime_error("unknown execution provider '" + name +
                           "'; expected cpu, cuda or trt");
}

std::string ProviderName(Provider p) {
  switch (p) {
    case Provider::kCpu:
      return "CPUExecutionProvider";
    case Provider::kCuda:
      return "CUDAExecutionProvider";
    case Provider::kTensorrt:
      return "TensorrtExecutionProvider";
  }
  throw std::runtime_error("unreachable: unknown provider");
}

Ort::Env& GlobalOrtEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "g1_deploy");
  return env;
}

OrtModel::OrtModel(const std::string& path, const std::vector<Provider>& providers,
                   int intra_threads, bool capture_cuda_graph)
    : path_(path), capture_cuda_graph_(capture_cuda_graph) {
  // Touch the Env FIRST. Registering the CUDA provider reaches for onnxruntime's default
  // logger, and that logger does not exist until an Ort::Env has been constructed -- so a
  // lazily-created Env means AppendExecutionProvider_CUDA throws
  // "Attempt to use DefaultLogger but none has been registered". It fails at session
  // creation with a message about logging, which points nowhere near the actual cause.
  (void)GlobalOrtEnv();

  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) options_.SetIntraOpNumThreads(intra_threads);

  // Offered in order; onnxruntime falls back through the list. CPU is implicit and last.
  bool cuda_offered = false;
  for (Provider p : providers) {
    switch (p) {
      case Provider::kCuda: {
        CudaOptions cuda;
        std::vector<const char*> keys{"device_id"};
        std::vector<const char*> vals{"0"};
        if (capture_cuda_graph_) {
          keys.push_back("enable_cuda_graph");
          vals.push_back("1");
        }
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(cuda.p, keys.data(),
                                                                  vals.data(), keys.size()));
        options_.AppendExecutionProvider_CUDA_V2(*cuda.p);
        cuda_offered = true;
        break;
      }
      case Provider::kTensorrt: {
        OrtTensorRTProviderOptions trt{};
        options_.AppendExecutionProvider_TensorRT(trt);
        break;
      }
      case Provider::kCpu:
        break;  // always available; nothing to append
    }
  }
  if (capture_cuda_graph_ && !cuda_offered) {
    throw std::runtime_error("CUDA Graph capture was requested for " + path +
                             " but no CUDA provider was offered. Capture is a CUDA feature; "
                             "there is nothing to capture on the CPU.");
  }

  // Profiling must be enabled BEFORE the session is created -- it cannot be turned on later.
  // WarmUpAndCheckPlacement then reads the trace to learn where nodes actually ran and calls
  // EndProfiling, which both writes the file and turns profiling OFF, so the hot loop pays
  // nothing. Without this the placement check has no trace to read, reports CPU, and
  // RequireCuda refuses to run a perfectly good GPU session.
  //
  // Skipped entirely under capture, where the trace is not needed: a capture-enabled session
  // that CREATES AT ALL has every node on CUDA, because onnxruntime refuses otherwise. So
  // placement is established by construction rather than measured, and the hot path never has
  // profiling switched on even briefly.
  if (!capture_cuda_graph_) {
    options_.EnableProfiling("/tmp/g1_ort_profile");
    profiling_enabled_ = true;
  }
  try {
    session_ = std::make_unique<Ort::Session>(GlobalOrtEnv(), path.c_str(), options_);
  } catch (const Ort::Exception& e) {
    if (!capture_cuda_graph_) throw;
    // The one failure mode worth translating: ORT says "all compute graph nodes have not been
    // partitioned to the CUDAExecutionProvider", which is true but says nothing about the
    // cause. For this pair the cause is essentially always a generator exported with a
    // symbolic batch axis, whose shape arithmetic lands on the host.
    throw std::runtime_error(
        std::string("CUDA Graph capture was refused for ") + path + ": " + e.what() +
        "\nThis almost always means the graph does not run entirely on the GPU. An fm.onnx "
        "exported with a SYMBOLIC batch axis leaves ~2700 shape-arithmetic nodes on the CPU, "
        "which rules capture out. Re-export it (scripts/flow_matching/export_fm_onnx.py, "
        "which now fixes the batch at 1 by default), or run with fm_cuda_graph:=false.");
  }
  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    input_names_.push_back(session_->GetInputNameAllocated(i, alloc_).get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    output_names_.push_back(session_->GetOutputNameAllocated(i, alloc_).get());
  }
}

std::string OrtModel::Metadata(const std::string& key) const {
  Ort::ModelMetadata meta = session_->GetModelMetadata();
  Ort::AllocatorWithDefaultOptions alloc;
  auto value = meta.LookupCustomMetadataMapAllocated(key.c_str(), alloc);
  if (!value) {
    throw std::runtime_error(path_ + " carries no '" + key +
                             "' metadata. Re-export it with scripts/rsl_rl/export_parameters.py.");
  }
  return std::string(value.get());
}

bool OrtModel::HasInput(const std::string& name) const {
  for (const std::string& n : input_names_) {
    if (n == name) return true;
  }
  return false;
}

std::vector<int64_t> OrtModel::InputShape(const std::string& name) const {
  for (size_t i = 0; i < input_names_.size(); ++i) {
    if (input_names_[i] != name) continue;
    // The TypeInfo must be a named local: GetTensorTypeAndShapeInfo() returns an UNOWNED
    // view into it, and binding it to a temporary reads freed memory -- which surfaces as a
    // nonsense element type or garbage dimensions rather than a crash.
    Ort::TypeInfo info = session_->GetInputTypeInfo(i);
    auto tensor = info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor.GetShape();
    for (int64_t& d : shape) {
      if (d < 0) d = 1;  // the batch axis is 1 in deploy; the rest are static
    }
    return shape;
  }
  throw std::runtime_error(path_ + " has no input named '" + name + "'");
}

void OrtModel::Run(const std::vector<const char*>& input_names,
                   const std::vector<Ort::Value>& inputs,
                   const std::vector<const char*>& output_names,
                   std::vector<Ort::Value>* outputs) {
  *outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                           inputs.size(), output_names.data(), output_names.size());
}

void OrtModel::PreparePersistentIo(const std::vector<std::string>& input_names) {
  if (binding_) throw std::runtime_error(path_ + ": PreparePersistentIo called twice");
  if (!capture_cuda_graph_) {
    throw std::runtime_error(path_ +
                             ": the persistent-IO path exists for CUDA Graph capture and is "
                             "only built for it. Without capture, use Run().");
  }
  if (output_names_.size() != 1) {
    throw std::runtime_error(path_ + " has " + std::to_string(output_names_.size()) +
                             " outputs; the persistent-IO path handles exactly one.");
  }
  EnsureCudaDataTransfer();
  bound_input_names_ = input_names;
  binding_ = std::make_unique<Ort::IoBinding>(*session_);
  auto cpu_mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // The allocator must outlive every tensor it hands out, so it is a member. OrtMemTypeDefault
  // on "Cuda" is device memory; the capture holds these exact addresses.
  Ort::MemoryInfo cuda_mem("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  device_alloc_ = std::make_unique<Ort::Allocator>(*session_, cuda_mem);

  host_input_buf_.resize(input_names.size());
  input_counts_.resize(input_names.size());
  for (size_t i = 0; i < input_names.size(); ++i) {
    // Shapes off the GRAPH, never restated here: the whole point of the artifact carrying its
    // own contract is that this code does not have a second opinion about it.
    const std::vector<int64_t> shape = InputShape(input_names[i]);
    const size_t n = Elements(shape);
    input_counts_[i] = n;
    host_input_buf_[i].assign(n, 0.0f);
    host_inputs_.push_back(Ort::Value::CreateTensor<float>(
        cpu_mem, host_input_buf_[i].data(), n, shape.data(), shape.size()));
    device_inputs_.push_back(Ort::Value::CreateTensor<float>(
        static_cast<OrtAllocator*>(*device_alloc_), shape.data(), shape.size()));
    binding_->BindInput(input_names[i].c_str(), device_inputs_.back());
  }

  Ort::TypeInfo out_info = session_->GetOutputTypeInfo(0);
  std::vector<int64_t> out_shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
  for (int64_t& d : out_shape) {
    if (d < 0) d = 1;
  }
  const size_t out_n = Elements(out_shape);
  host_output_buf_.assign(out_n, 0.0f);
  host_output_.push_back(Ort::Value::CreateTensor<float>(
      cpu_mem, host_output_buf_.data(), out_n, out_shape.data(), out_shape.size()));
  device_output_.push_back(Ort::Value::CreateTensor<float>(
      static_cast<OrtAllocator*>(*device_alloc_), out_shape.data(), out_shape.size()));
  binding_->BindOutput(output_names_[0].c_str(), device_output_[0]);
}

void OrtModel::RunPersistent(const std::vector<const float*>& input_data, float* output,
                             size_t output_count) {
  if (!binding_) throw std::runtime_error(path_ + ": call PreparePersistentIo first");
  if (input_data.size() != bound_input_names_.size()) {
    throw std::runtime_error(path_ + ": RunPersistent got " + std::to_string(input_data.size()) +
                             " inputs but " + std::to_string(bound_input_names_.size()) +
                             " were bound. The set and its order are fixed at bind time.");
  }
  if (output_count != host_output_buf_.size()) {
    throw std::runtime_error(path_ + ": output is " + std::to_string(host_output_buf_.size()) +
                            " values, not " + std::to_string(output_count));
  }
  for (size_t i = 0; i < input_data.size(); ++i) {
    std::copy(input_data[i], input_data[i] + input_counts_[i], host_input_buf_[i].begin());
  }
  // Host -> device INTO the bound buffers. Not a rebind: a rebind would hand the replay a
  // pointer it never captured, and it would then keep returning the first call's answer --
  // measured, silently, with no error (see the note at the top of ort_session.h).
  Ort::ThrowOnError(GlobalOrtEnv().CopyTensors(host_inputs_, device_inputs_, nullptr));
  session_->Run(Ort::RunOptions{nullptr}, *binding_);
  // CopyTensors on the default stream also serves as the synchronisation: the D2H cannot start
  // before the replay it is queued behind has finished.
  Ort::ThrowOnError(GlobalOrtEnv().CopyTensors(device_output_, host_output_, nullptr));
  std::copy(host_output_buf_.begin(), host_output_buf_.end(), output);
}

void OrtModel::WarmUpAndCheckPlacement(int iters) {
  if (capture_cuda_graph_) {
    // Under capture the warm-up MUST go through the binding: a plain Run() on a
    // capture-enabled session is exactly the unsupported path (it returns an empty output list
    // from the second call on), so warming up the old way would both mis-warm the session and
    // leave the first live tick to pay for the capture.
    if (!binding_) {
      throw std::runtime_error(path_ +
                               ": PreparePersistentIo must run before the warm-up when CUDA "
                               "Graph capture is on -- the capture records the bound addresses.");
    }
    std::vector<const float*> zeros;
    zeros.reserve(host_input_buf_.size());
    for (const std::vector<float>& buf : host_input_buf_) zeros.push_back(buf.data());
    std::vector<float> out(host_output_buf_.size(), 0.0f);
    for (int i = 0; i < iters; ++i) RunPersistent(zeros, out.data(), out.size());
    // Established by construction: a capture-enabled session does not get created at all
    // unless every node is on CUDA, so there is nothing left for the profile to tell us.
    on_gpu_ = true;
    return;
  }
  std::vector<WarmupInput> holders(session_->GetInputCount());
  std::vector<Ort::Value> values;
  std::vector<const char*> in_names;
  std::vector<const char*> out_names;
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  for (size_t i = 0; i < holders.size(); ++i) {
    Ort::TypeInfo info = session_->GetInputTypeInfo(i);
    auto tensor = info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = tensor.GetShape();
    for (int64_t& d : shape) {
      if (d < 0) d = 1;
    }
    size_t n = 1;
    for (int64_t d : shape) n *= static_cast<size_t>(d);

    WarmupInput& h = holders[i];
    switch (tensor.GetElementType()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        h.f32.assign(n, 0.0f);
        h.value = Ort::Value::CreateTensor<float>(mem, h.f32.data(), n, shape.data(),
                                                  shape.size());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        h.i64.assign(n, 0);
        h.value = Ort::Value::CreateTensor<int64_t>(mem, h.i64.data(), n, shape.data(),
                                                    shape.size());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        h.boolean.assign(n, 0);
        h.value = Ort::Value::CreateTensor(mem, h.boolean.data(), n, shape.data(), shape.size(),
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        break;
      default:
        throw std::runtime_error(path_ + " input '" + input_names_[i] +
                                 "' has an element type this runtime does not build");
    }
    in_names.push_back(input_names_[i].c_str());
    values.push_back(std::move(h.value));
  }
  for (const std::string& n : output_names_) out_names.push_back(n.c_str());

  for (int i = 0; i < iters; ++i) {
    session_->Run(Ort::RunOptions{nullptr}, in_names.data(), values.data(), values.size(),
                  out_names.data(), out_names.size());
  }

  // Placement is read from the profile: it reports where nodes ACTUALLY ran, where the
  // provider list only reports what the build supports.
  const std::string profile = session_->EndProfilingAllocated(alloc_).get();
  profiling_enabled_ = false;   // EndProfiling also turns it off
  std::ifstream trace(profile);
  std::string line;
  while (std::getline(trace, line)) {
    if (line.find("CUDAExecutionProvider") != std::string::npos) {
      on_gpu_ = true;
      break;
    }
  }
  trace.close();
  std::remove(profile.c_str());
}

OrtModel::~OrtModel() {
  // A session that was never warmed up still has profiling enabled, and leaving it on writes a
  // growing trace for the life of the process. End it and discard the file.
  if (profiling_enabled_ && session_) {
    try {
      const std::string path = session_->EndProfilingAllocated(alloc_).get();
      std::remove(path.c_str());
    } catch (...) {
      // A destructor must not throw, and a stale profile file is not worth crashing over.
    }
  }
}

void RequireCuda(const OrtModel& model, const std::vector<Provider>& requested) {
  bool want_cuda = false;
  for (Provider p : requested) {
    if (p == Provider::kCuda || p == Provider::kTensorrt) want_cuda = true;
  }
  if (!want_cuda || model.on_gpu()) return;
  throw std::runtime_error(
      "use_cuda is true but onnxruntime placed every node of " + model.path() +
      " on the CPU. This pair cannot run there: the generator measures ~103 ms per re-plan "
      "tick on CPU against a 20 ms control period, which is not a degraded mode but an "
      "unusable one. Check that the container was started with the nvidia runtime (the "
      "no-GPU compose file cannot provide a GPU), that nvidia-smi or the Jetson equivalent "
      "sees the device inside the container, and that the image's CUDA runtime matches the "
      "build. To run on the CPU deliberately, set use_cuda:=false and plan for the deficit.");
}

}  // namespace g1_deploy
