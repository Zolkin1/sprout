// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// THE reader for the manifest.json + raw-float64-blob format that transfer/deploy writes
// (transfer/deploy/fixture_io.py:FixtureWriter). One reader for one format, on this side of
// the language boundary, for the same reason there is one writer on the other: a second
// producer or consumer is how a format drifts.
//
// Three things are carried in it, and they are the same shape because they cross the same
// seam: the PARITY FIXTURES (dump_parity_fixtures.py -- the python oracle's answers plus,
// per case, the tolerance the C++ must match them to, defined once next to the oracle so the
// two languages cannot disagree on what "agrees" means); the per-tick REPLAY STREAM
// (replay_stream.py, played back by tools/replay.cc); and a CLIP PACK (dump_clip_pack.py --
// a debug motion library, so clip playback needs no HDF5 on the robot).
//
// Blobs are raw little-endian float64 in C order. The reader verifies the manifest's
// byteorder and every blob's length against its declared shape, so a truncated or
// half-regenerated set fails loudly instead of being compared against garbage.

#include <map>
#include <string>
#include <vector>

namespace g1_deploy {

/// One array from a pack.
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
  std::map<std::string, std::vector<int>> int_lists;              //: e.g. attrs.window_offsets
  std::map<std::string, std::vector<double>> double_lists;        //: a float array attr
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

/// One whole pack: a manifest plus its blobs.
class BlobPack {
 public:
  /// Load from a directory containing manifest.json and its blobs.
  ///
  /// Throws std::runtime_error on a missing/!=little-endian/format-mismatched manifest, a
  /// missing blob, or a blob whose byte length disagrees with its declared shape.
  static BlobPack Load(const std::string& dir);

  /// Every case whose `fn` matches, in manifest order. Empty is a caller error, not a pass:
  /// tests should assert non-empty so a renamed fixture cannot silently skip its check.
  std::vector<const Case*> ForFunction(const std::string& fn) const;

  const std::vector<Case>& cases() const { return cases_; }

 private:
  std::vector<Case> cases_;
};

/// Where the committed parity fixtures live, overridable with the G1_PARITY_FIXTURES env
/// var so a freshly dumped set can be checked without a rebuild.
///
/// Free rather than a member of BlobPack: the baked-in path is a property of THIS source
/// tree, not of the format, and a pack read on a robot has nothing to do with it.
///
/// \throws std::runtime_error If no directory was baked in at build time.
std::string DefaultFixtureDir();

}  // namespace g1_deploy
