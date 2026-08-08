// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Reader for the parity fixtures written by transfer/deploy/tools/dump_parity_fixtures.py.
//
// The fixture set is the python oracle's answers plus, per case, the tolerance it must be
// matched to. The tolerance travels in the manifest deliberately: it is defined once, in
// python, next to the oracle, so the two languages cannot drift on what "agrees" means.
//
// Blobs are raw little-endian float64 in C order. The reader verifies the manifest's
// byteorder and every blob's length against its declared shape, so a truncated or
// half-regenerated fixture set fails loudly instead of comparing against garbage.

#include <map>
#include <string>
#include <vector>

namespace g1_deploy::test {

/// One array from the fixture set.
struct Blob {
  std::vector<int> shape;
  std::vector<double> data;

  /// Number of leading rows (shape[0]), i.e. how many independent cases the blob holds.
  int rows() const { return shape.empty() ? 0 : shape[0]; }

  /// Elements per row: the product of every dimension after the first.
  int stride() const;

  /// Pointer to row `i`.
  const double* row(int i) const { return data.data() + static_cast<size_t>(i) * stride(); }
};

/// One function's inputs, the oracle's outputs, and how to compare them.
struct Case {
  std::string name;
  std::string fn;
  std::string compare;  //: "elementwise" or "quat_rotation"
  double atol = 0.0;
  std::map<std::string, Blob> inputs;
  std::map<std::string, Blob> outputs;
  std::map<std::string, std::vector<std::string>> string_lists;  //: e.g. attrs.blocks
  std::map<std::string, int> ints;                               //: e.g. attrs.num_joints
  std::map<std::string, double> doubles;                         //: e.g. attrs.min_dist
  std::map<std::string, std::string> strings;                    //: e.g. attrs.source

  /// Double attr by name. Throws std::runtime_error if absent -- a defaulted camera
  /// contract would make the test pass against the port's own defaults rather than the
  /// fixture's, which is exactly the bug the non-zero-clips case exists to catch.
  double Double(const std::string& key) const;

  /// Input blob by name. Throws std::runtime_error if absent.
  const Blob& In(const std::string& key) const;

  /// Output blob by name. Throws std::runtime_error if absent.
  const Blob& Out(const std::string& key) const;
};

/// The whole fixture set.
class Fixtures {
 public:
  /// Load from a directory containing manifest.json and its blobs.
  ///
  /// Throws std::runtime_error on a missing/!=little-endian/format-mismatched manifest, a
  /// missing blob, or a blob whose byte length disagrees with its declared shape.
  static Fixtures Load(const std::string& dir);

  /// Every case whose `fn` matches, in manifest order. Empty is a caller error, not a pass:
  /// tests should assert non-empty so a renamed fixture cannot silently skip its check.
  std::vector<const Case*> ForFunction(const std::string& fn) const;

  const std::vector<Case>& cases() const { return cases_; }

  /// Default fixture directory, overridable with the G1_PARITY_FIXTURES env var so the
  /// test can run against a freshly dumped set without a rebuild.
  static std::string DefaultDir();

 private:
  std::vector<Case> cases_;
};

}  // namespace g1_deploy::test
