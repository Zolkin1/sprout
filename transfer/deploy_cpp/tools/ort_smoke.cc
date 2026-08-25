// Copyright 2026 Zachary Olkin. All rights reserved.

// Does the C++ onnxruntime actually run THIS policy pair on THIS GPU, fast enough?
//
// The gate for the whole C++ port. Everything else in transfer/deploy_cpp is arithmetic that
// can be checked against the python oracle offline; this is the one question that needs the
// real hardware, the real .onnx files and a real CUDA session, and it is worth answering
// before any more porting effort goes in.
//
// It deliberately does NOT hardcode the models' input names or shapes -- it reads them from
// the graph. A smoke test that restates the contract can pass against a model it has
// misunderstood.
//
// Two things are checked, in order of importance:
//
//   1. The session really placed work on the GPU. onnxruntime falls back to CPU SILENTLY, and
//      a CPU fallback here is not a slow robot, it is an unusable one (the python measured
//      103.5 ms per generator tick on CPU against a 20 ms control period). Rather than trust
//      the provider LIST -- which only reports what the wheel was built with -- this turns on
//      profiling for the warmup and greps the resulting trace for a node actually assigned to
//      CUDAExecutionProvider. Profiling is then off for the timed runs, so it costs nothing.
//   2. The latency distribution, reported as p50/p99, not a mean. The FM replan is synchronous
//      with the tick it fires on, so the tail is what has to fit the budget.
//
//   ort_smoke <tracker.onnx> [fm.onnx] [--iters N] [--provider cuda|cpu] [--budget-ms X]

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string tracker;
  std::string fm;
  int iters = 200;
  std::string provider = "cuda";
  double budget_ms = 20.0;
};

/// Storage for one input tensor, kept alive for the duration of a Run.
struct InputBuffer {
  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type;
  std::vector<float> f32;
  std::vector<int64_t> i64;
  std::vector<char> boolean;
  Ort::Value value{nullptr};
};

int64_t ElementCount(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::string ShapeString(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) os << (i ? "," : "") << shape[i];
  os << "]";
  return os.str();
}

/// Build zero-filled inputs matching the graph's declared signature.
///
/// A dynamic dimension (-1) is resolved to 1: every input of this pair is batched with a
/// leading env axis of 1 in deploy, and the rest of the dims are static.
std::vector<InputBuffer> MakeInputs(Ort::Session& session, Ort::AllocatorWithDefaultOptions& alloc,
                                    const Ort::MemoryInfo& mem) {
  std::vector<InputBuffer> inputs(session.GetInputCount());
  for (size_t i = 0; i < inputs.size(); ++i) {
    InputBuffer& in = inputs[i];
    in.name = session.GetInputNameAllocated(i, alloc).get();
    // The TypeInfo must be held in a named local: GetTensorTypeAndShapeInfo() returns an
    // UNOWNED view into it, so binding it to a temporary reads freed memory -- which
    // surfaces as a nonsense element-type id rather than a crash.
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto info = type_info.GetTensorTypeAndShapeInfo();
    in.shape = info.GetShape();
    for (int64_t& d : in.shape) {
      if (d < 0) d = 1;
    }
    in.type = info.GetElementType();
    const size_t n = static_cast<size_t>(ElementCount(in.shape));
    switch (in.type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        in.f32.assign(n, 0.0f);
        in.value = Ort::Value::CreateTensor<float>(mem, in.f32.data(), n, in.shape.data(),
                                                   in.shape.size());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        in.i64.assign(n, 0);
        in.value = Ort::Value::CreateTensor<int64_t>(mem, in.i64.data(), n, in.shape.data(),
                                                     in.shape.size());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        in.boolean.assign(n, 0);
        in.value = Ort::Value::CreateTensor(mem, in.boolean.data(), n, in.shape.data(),
                                            in.shape.size(),
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
        break;
      default:
        throw std::runtime_error("input '" + in.name + "' has an unsupported element type " +
                                 std::to_string(static_cast<int>(in.type)));
    }
    std::cout << "    in  " << in.name << " " << ShapeString(in.shape) << "\n";
  }
  return inputs;
}

double Percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t idx = std::min(v.size() - 1,
                              static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5));
  return v[idx];
}

/// Run one model: warm up (with profiling on, to prove placement), then time `iters` runs.
///
/// Returns false if a CUDA session was requested but no node landed on the GPU.
bool Benchmark(const std::string& label, const std::string& path, const Args& args) {
  std::cout << "\n=== " << label << ": " << path << " ===\n";
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort_smoke");
  Ort::SessionOptions so;
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  const bool want_cuda = (args.provider == "cuda");
  if (want_cuda) {
    OrtCUDAProviderOptions cuda_options{};
    so.AppendExecutionProvider_CUDA(cuda_options);
  }
  // Profiling is on ONLY for the warmup: EndProfiling both writes the trace and disables
  // further profiling, so the timed loop below pays nothing for this check.
  so.EnableProfiling("/tmp/ort_smoke_profile");

  Ort::Session session(env, path.c_str(), so);
  Ort::AllocatorWithDefaultOptions alloc;
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<InputBuffer> inputs = MakeInputs(session, alloc, mem);
  std::vector<const char*> in_names;
  std::vector<Ort::Value> in_values;
  for (auto& in : inputs) {
    in_names.push_back(in.name.c_str());
    in_values.push_back(std::move(in.value));
  }
  std::vector<std::string> out_name_store;
  std::vector<const char*> out_names;
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    out_name_store.push_back(session.GetOutputNameAllocated(i, alloc).get());
  }
  for (const auto& n : out_name_store) out_names.push_back(n.c_str());

  // Warmup. The first CUDA run builds the context and picks cuDNN algorithms -- hundreds of
  // milliseconds that must never land on a live control tick, which is why the real runtime
  // warms up before arming too.
  for (int i = 0; i < 5; ++i) {
    session.Run(Ort::RunOptions{nullptr}, in_names.data(), in_values.data(), in_names.size(),
                out_names.data(), out_names.size());
  }

  // EndProfilingAllocated, not EndProfiling: the C++ wrapper renamed it when it moved to the
  // allocator-owned string. It both writes the trace and turns profiling OFF, so the timed
  // loop below is unaffected by having asked this question.
  const std::string profile_path = session.EndProfilingAllocated(alloc).get();
  bool on_gpu = false;
  {
    std::ifstream prof(profile_path);
    std::string line;
    while (std::getline(prof, line)) {
      if (line.find("CUDAExecutionProvider") != std::string::npos) {
        on_gpu = true;
        break;
      }
    }
  }
  std::remove(profile_path.c_str());
  std::cout << "    placement: " << (on_gpu ? "CUDAExecutionProvider (GPU)" : "CPU only")
            << "\n";
  if (want_cuda && !on_gpu) {
    std::cout << "    FAIL: cuda was requested but every node ran on the CPU.\n";
    return false;
  }

  std::vector<double> ms;
  ms.reserve(static_cast<size_t>(args.iters));
  for (int i = 0; i < args.iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    session.Run(Ort::RunOptions{nullptr}, in_names.data(), in_values.data(), in_names.size(),
                out_names.data(), out_names.size());
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / static_cast<double>(ms.size());
  std::cout << "    iters " << args.iters << "  mean " << mean << " ms"
            << "  p50 " << Percentile(ms, 0.50) << "  p99 " << Percentile(ms, 0.99)
            << "  max " << *std::max_element(ms.begin(), ms.end()) << "\n";
  std::cout << "    budget " << args.budget_ms << " ms -> "
            << (Percentile(ms, 0.99) < args.budget_ms ? "FITS" : "OVER") << " at p99\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) {
      args.iters = std::stoi(argv[++i]);
    } else if (a == "--provider" && i + 1 < argc) {
      args.provider = argv[++i];
    } else if (a == "--budget-ms" && i + 1 < argc) {
      args.budget_ms = std::stod(argv[++i]);
    } else {
      positional.push_back(a);
    }
  }
  if (positional.empty()) {
    std::cerr << "usage: ort_smoke <tracker.onnx> [fm.onnx] [--iters N] [--provider cuda|cpu]"
                 " [--budget-ms X]\n";
    return 2;
  }
  args.tracker = positional[0];
  if (positional.size() > 1) args.fm = positional[1];

  try {
    std::cout << "onnxruntime " << Ort::GetVersionString() << ", provider " << args.provider
              << "\n";
    bool ok = Benchmark("tracker", args.tracker, args);
    if (!args.fm.empty()) ok = Benchmark("generator", args.fm, args) && ok;
    std::cout << "\n" << (ok ? "[smoke] PASS" : "[smoke] FAIL") << "\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "[smoke] ERROR: " << e.what() << "\n";
    return 1;
  }
}
