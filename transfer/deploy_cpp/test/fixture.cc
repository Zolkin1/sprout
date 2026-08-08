// Copyright 2026 Zachary Olkin. All rights reserved.

#include "fixture.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace g1_deploy::test {
namespace {

constexpr int kExpectedFormat = 1;

std::vector<double> ReadBlob(const std::filesystem::path& path, size_t count) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open fixture blob: " + path.string());
  const auto bytes = std::filesystem::file_size(path);
  if (bytes != count * sizeof(double)) {
    throw std::runtime_error("fixture blob " + path.string() + " is " + std::to_string(bytes) +
                             " bytes, but its declared shape needs " +
                             std::to_string(count * sizeof(double)) +
                             ". Re-run dump_parity_fixtures.py.");
  }
  std::vector<double> data(count);
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
  if (!in) throw std::runtime_error("short read on fixture blob: " + path.string());
  return data;
}

Blob ParseBlob(const nlohmann::json& entry, const std::filesystem::path& dir) {
  Blob blob;
  size_t count = 1;
  for (const auto& d : entry.at("shape")) {
    blob.shape.push_back(d.get<int>());
    count *= static_cast<size_t>(d.get<int>());
  }
  blob.data = ReadBlob(dir / entry.at("file").get<std::string>(), count);
  return blob;
}

}  // namespace

int Blob::stride() const {
  int n = 1;
  for (size_t i = 1; i < shape.size(); ++i) n *= shape[i];
  return n;
}

const Blob& Case::In(const std::string& key) const {
  auto it = inputs.find(key);
  if (it == inputs.end()) throw std::runtime_error(name + ": no input '" + key + "'");
  return it->second;
}

double Case::Double(const std::string& key) const {
  auto it = doubles.find(key);
  if (it == doubles.end()) throw std::runtime_error(name + ": no double attr '" + key + "'");
  return it->second;
}

const Blob& Case::Out(const std::string& key) const {
  auto it = outputs.find(key);
  if (it == outputs.end()) throw std::runtime_error(name + ": no output '" + key + "'");
  return it->second;
}

std::string Fixtures::DefaultDir() {
  if (const char* env = std::getenv("G1_PARITY_FIXTURES")) return env;
  return G1_DEPLOY_FIXTURE_DIR;  // baked in by CMake from the source tree layout
}

Fixtures Fixtures::Load(const std::string& dir) {
  const std::filesystem::path base(dir);
  std::ifstream in(base / "manifest.json");
  if (!in) {
    throw std::runtime_error(
        "no manifest.json under " + dir +
        ". Generate it with: python transfer/deploy/tools/dump_parity_fixtures.py");
  }
  nlohmann::json manifest;
  in >> manifest;

  if (manifest.at("format").get<int>() != kExpectedFormat) {
    throw std::runtime_error("fixture manifest format " +
                             std::to_string(manifest.at("format").get<int>()) + ", expected " +
                             std::to_string(kExpectedFormat) + ": regenerate the fixtures.");
  }
  // The blobs are raw doubles with no byte-order marker, so a big-endian producer would be
  // read as garbage rather than rejected. Both target arches are little-endian; check it.
  if (manifest.at("byteorder").get<std::string>() != "little") {
    throw std::runtime_error("fixtures were produced big-endian; the reader assumes little.");
  }

  Fixtures out;
  for (const auto& c : manifest.at("cases")) {
    Case parsed;
    parsed.name = c.at("case").get<std::string>();
    parsed.fn = c.at("fn").get<std::string>();
    parsed.compare = c.at("compare").get<std::string>();
    parsed.atol = c.at("atol").get<double>();
    for (const auto& b : c.at("inputs")) {
      parsed.inputs[b.at("name").get<std::string>()] = ParseBlob(b, base);
    }
    for (const auto& b : c.at("outputs")) {
      parsed.outputs[b.at("name").get<std::string>()] = ParseBlob(b, base);
    }
    for (const auto& [key, value] : c.at("attrs").items()) {
      if (value.is_number_integer()) {
        parsed.ints[key] = value.get<int>();
      } else if (value.is_number_float()) {
        parsed.doubles[key] = value.get<double>();
      } else if (value.is_string()) {
        parsed.strings[key] = value.get<std::string>();
      } else if (value.is_array()) {
        std::vector<std::string> items;
        for (const auto& v : value) items.push_back(v.get<std::string>());
        parsed.string_lists[key] = std::move(items);
      }
    }
    out.cases_.push_back(std::move(parsed));
  }
  return out;
}

std::vector<const Case*> Fixtures::ForFunction(const std::string& fn) const {
  std::vector<const Case*> out;
  for (const auto& c : cases_) {
    if (c.fn == fn) out.push_back(&c);
  }
  return out;
}

}  // namespace g1_deploy::test
