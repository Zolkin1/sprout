// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/ort_session.h"

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
                   int intra_threads)
    : path_(path) {
  // Touch the Env FIRST. Registering the CUDA provider reaches for onnxruntime's default
  // logger, and that logger does not exist until an Ort::Env has been constructed -- so a
  // lazily-created Env means AppendExecutionProvider_CUDA throws
  // "Attempt to use DefaultLogger but none has been registered". It fails at session
  // creation with a message about logging, which points nowhere near the actual cause.
  (void)GlobalOrtEnv();

  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (intra_threads > 0) options_.SetIntraOpNumThreads(intra_threads);

  // Offered in order; onnxruntime falls back through the list. CPU is implicit and last.
  for (Provider p : providers) {
    switch (p) {
      case Provider::kCuda: {
        OrtCUDAProviderOptions cuda{};
        options_.AppendExecutionProvider_CUDA(cuda);
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

  session_ = std::make_unique<Ort::Session>(GlobalOrtEnv(), path.c_str(), options_);
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

void OrtModel::WarmUpAndCheckPlacement(int iters) {
  // Profiling on only for the warm-up. EndProfilingAllocated writes the trace AND disables
  // further profiling, so the timed path afterwards is unaffected by having asked.
  Ort::SessionOptions probe_options;
  probe_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

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
