// Copyright 2026 Zachary Olkin. All rights reserved.
#pragma once

// Order, clip, stack and scale the observation terms into the vector the policy sees.
//
// Port of GenTrackerPolicy._assemble / _push_history in transfer/deploy/gen_tracker.py, which
// in turn mirrors Isaac Lab's observation manager. The exported term table is the single
// authority: its ORDER is the concatenation order, its `shape` is the FLAT width (history
// already multiplied in), and its clip / scale / history_length are what training applied.
//
// ============================ the order is not negotiable ================================
// Per term, in Isaac Lab's order:
//
//   1. cast the raw float64 value to FLOAT32;
//   2. clip it, in float32;
//   3. push it through the term's history window (storing the clipped, UNSCALED sample);
//   4. scale, AFTER stacking -- and a per-element scale describes ONE sample, so it TILES
//      across the history.
//
// Doing the cast after the clip, or the scale before the stack, changes the float32 rounding
// of every element of that term. It still produces a correctly sized vector.
//
// ============================ the NaN hazard ==============================================
// np.clip PROPAGATES NaN, and the non-finite guard downstream depends on that: a NaN in the
// observation has to survive as far as _finite_targets, which refuses to publish and
// eventually disarms.
//
// Eigen's cwiseMin is (a < b) ? a : b, so cwiseMin(NaN, clip) returns CLIP. Using the obvious
// Eigen call here would silently convert a NaN into a legitimate-looking bound: the guard
// never fires, and the robot is commanded from a poisoned observation instead of holding pose.
// ClipPreservingNan below is written the long way for exactly that reason, and
// obs_assembler_test pins it.

#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "g1_deploy/policy_params.h"

namespace g1_deploy {

/// Clip to [-c, c] while letting a NaN through, matching np.clip.
///
/// Deliberately not std::clamp or an Eigen cwise op: std::clamp(NaN, lo, hi) is unspecified in
/// practice (it compares and returns a bound), and Eigen's min turns NaN into the bound.
inline float ClipPreservingNan(float v, float c) {
  if (std::isnan(v)) return v;
  return v < -c ? -c : (v > c ? c : v);
}

/// Fallback clips for artifacts exported before the terms carried their own `clip`.
///
/// Kept only for those; the metadata value wins whenever it is present. Anything clipped in
/// the env cfg but absent here was silently unclipped on those older exports.
double LegacyTermClip(const std::string& name, bool* found);

/// Builds the policy observation vector from per-term values.
///
/// Owns the per-term history windows, so one instance per policy and Reset() on arming.
class ObsAssembler {
 public:
  /// `terms` is the exported table, IN ORDER (see policy_params.h -- the order is the layout).
  explicit ObsAssembler(std::vector<ObsTerm> terms, int num_obs);

  /// Clear every history window.
  ///
  /// The next Assemble re-fills each window from its first sample, which is what Isaac Lab
  /// does on an env reset -- keeping a stale window would feed the policy motion from before
  /// the reset.
  void Reset();

  /// Assemble the vector. `values` maps term name -> that term's SINGLE-SAMPLE float64 value.
  ///
  /// Writes num_obs floats into `out`.
  ///
  /// Throws std::runtime_error if a term the tracker expects is missing, or if a supplied
  /// value's width disagrees with the metadata.
  void Assemble(const std::map<std::string, std::vector<double>>& values, float* out);

  /// The last assembled vector, for the debug topic and the checksum.
  const std::vector<float>& last_obs() const { return last_obs_; }

  int num_obs() const { return num_obs_; }

 private:
  /// Append to term `name`'s window and return it flattened oldest-first.
  ///
  /// Isaac Lab CircularBuffer semantics, which the policy trained against: the FIRST push
  /// fills EVERY slot with that sample (so step 0 is a static window, not a zero-padded one),
  /// later pushes roll, and the read is oldest-at-index-0 with the most recent last.
  const std::vector<float>& PushHistory(const std::string& name, const std::vector<float>& v,
                                        int history_length);

  std::vector<ObsTerm> terms_;
  int num_obs_;
  std::vector<float> last_obs_;
  //: term name -> the flattened window, plus the per-slot layout needed to roll it.
  struct History {
    std::vector<float> flat;  //: history_length * step_dim, oldest first
    int step_dim = 0;
    int length = 0;
  };
  std::map<std::string, History> history_;
  std::vector<float> scratch_;
};

/// blake2b-64 over the observation's float32 bytes, matching deploy/diagnostics.obs_checksum.
///
/// Recorded every tick so an offline reconstruction can PROVE it rebuilt what the network saw.
/// Note what it is not: a cross-language parity metric. One ulp anywhere flips the whole
/// digest, so a mismatch says "differs somewhere" and nothing about magnitude.
uint64_t ObsChecksum(const float* obs, size_t count);

}  // namespace g1_deploy
