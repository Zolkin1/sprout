// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/obs_assembler.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace g1_deploy {
namespace {

// ==========================================================================================
// BLAKE2b, vendored.
//
// deploy/diagnostics.obs_checksum is blake2b(bytes, digest_size=8). That is NOT a truncation
// of blake2b-512: the output length goes into the parameter block and changes every word of
// the state, so OpenSSL's EVP_blake2b512 cannot produce it. libb2 could, but this is the only
// hash the deploy runtime needs and a dependency in the image is a worse trade than ~90 lines
// of well-known reference code. Pinned against python's hashlib in obs_assembler_test.
// ==========================================================================================

constexpr uint64_t kIV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

constexpr uint8_t kSigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

inline uint64_t Rotr64(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

inline uint64_t LoadLe64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
  return v;
}

void Compress(uint64_t h[8], const uint8_t block[128], uint64_t t0, uint64_t t1, bool last) {
  uint64_t m[16];
  for (int i = 0; i < 16; ++i) m[i] = LoadLe64(block + 8 * i);

  uint64_t v[16];
  for (int i = 0; i < 8; ++i) v[i] = h[i];
  for (int i = 0; i < 8; ++i) v[8 + i] = kIV[i];
  v[12] ^= t0;
  v[13] ^= t1;
  if (last) v[14] = ~v[14];

  auto mix = [&](int r, int i, int a, int b, int c, int d) {
    v[a] = v[a] + v[b] + m[kSigma[r][2 * i]];
    v[d] = Rotr64(v[d] ^ v[a], 32);
    v[c] = v[c] + v[d];
    v[b] = Rotr64(v[b] ^ v[c], 24);
    v[a] = v[a] + v[b] + m[kSigma[r][2 * i + 1]];
    v[d] = Rotr64(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = Rotr64(v[b] ^ v[c], 63);
  };

  for (int r = 0; r < 12; ++r) {
    mix(r, 0, 0, 4, 8, 12);
    mix(r, 1, 1, 5, 9, 13);
    mix(r, 2, 2, 6, 10, 14);
    mix(r, 3, 3, 7, 11, 15);
    mix(r, 4, 0, 5, 10, 15);
    mix(r, 5, 1, 6, 11, 12);
    mix(r, 6, 2, 7, 8, 13);
    mix(r, 7, 3, 4, 9, 14);
  }
  for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[8 + i];
}

}  // namespace

uint64_t ObsChecksum(const float* obs, size_t count) {
  const uint8_t* data = reinterpret_cast<const uint8_t*>(obs);
  const size_t len = count * sizeof(float);

  uint64_t h[8];
  for (int i = 0; i < 8; ++i) h[i] = kIV[i];
  // Parameter block: digest_size=8, key_length=0, fanout=1, depth=1.
  h[0] ^= 0x01010000ULL ^ 8ULL;

  uint64_t t = 0;
  size_t off = 0;
  // Every block but the last is compressed with last=false. The final block is padded with
  // zeros, and the counter carries the REAL byte count, not the padded one.
  while (len - off > 128) {
    t += 128;
    Compress(h, data + off, t, 0, false);
    off += 128;
  }
  uint8_t final_block[128] = {0};
  const size_t rest = len - off;
  std::memcpy(final_block, data + off, rest);
  t += rest;
  Compress(h, final_block, t, 0, true);

  // The digest's bytes are h little-endian; python then reads them big-endian
  // (int.from_bytes(..., "big")), so the 64-bit result is h[0] byte-reversed.
  uint64_t out = 0;
  for (int i = 0; i < 8; ++i) out = (out << 8) | ((h[0] >> (8 * i)) & 0xFFULL);
  return out;
}

double LegacyTermClip(const std::string& name, bool* found) {
  *found = true;
  if (name == "actions") return 20.0;
  if (name == "height_scan") return 1.0;
  *found = false;
  return 0.0;
}

ObsAssembler::ObsAssembler(std::vector<ObsTerm> terms, int num_obs)
    : terms_(std::move(terms)), num_obs_(num_obs) {
  last_obs_.assign(static_cast<size_t>(num_obs), 0.0f);
}

void ObsAssembler::Reset() { history_.clear(); }

const std::vector<float>& ObsAssembler::PushHistory(const std::string& name,
                                                    const std::vector<float>& v,
                                                    int history_length) {
  History& hist = history_[name];
  const int step_dim = static_cast<int>(v.size());
  if (hist.length != history_length || hist.step_dim != step_dim) {
    // First push (or a shape change) fills EVERY slot with this sample, so step 0 is a static
    // window rather than a zero-padded one. That is what Isaac Lab's CircularBuffer does and
    // therefore what the policy trained against; zero-padding would hand it a discontinuity
    // at every reset that training never produced.
    hist.length = history_length;
    hist.step_dim = step_dim;
    hist.flat.resize(static_cast<size_t>(history_length) * step_dim);
    for (int s = 0; s < history_length; ++s) {
      std::copy(v.begin(), v.end(), hist.flat.begin() + static_cast<size_t>(s) * step_dim);
    }
    return hist.flat;
  }
  // Roll: drop the oldest, append the newest. Oldest at index 0, most recent last, which is
  // the order the manager flattens step-major -- [v_{t-(H-1)}, ..., v_t].
  std::memmove(hist.flat.data(), hist.flat.data() + step_dim,
               sizeof(float) * static_cast<size_t>(step_dim) * (history_length - 1));
  std::copy(v.begin(), v.end(),
            hist.flat.begin() + static_cast<size_t>(step_dim) * (history_length - 1));
  return hist.flat;
}

void ObsAssembler::Assemble(const std::map<std::string, std::vector<double>>& values, float* out) {
  int idx = 0;
  for (const ObsTerm& term : terms_) {
    auto it = values.find(term.name);
    if (it == values.end()) {
      if (term.name == "body_pos_ref" || term.name == "body_vel_ref") {
        throw std::runtime_error(
            "obs term '" + term.name + "': this tracker was trained with tracking_mode='body', "
            "so its reference must carry root-frame body targets. Clip playback supplies them; "
            "the generator path does NOT -- it predicts joint angles, and turning those into "
            "root-frame link poses needs an FK this runtime does not have.");
      }
      throw std::runtime_error("obs term '" + term.name +
                               "' was not supplied (a depth term means the caller passed no "
                               "image for that camera).");
    }

    const int step_dim = term.history_length > 0 ? term.width / term.history_length : term.width;
    if (static_cast<int>(it->second.size()) != step_dim) {
      throw std::runtime_error(
          "obs term '" + term.name + "': built dim " + std::to_string(it->second.size()) +
          " != metadata dim " + std::to_string(step_dim) +
          (term.history_length > 0 ? " (= " + std::to_string(term.width) + " / history_length " +
                                         std::to_string(term.history_length) + ")"
                                   : ""));
    }

    // 1. float32 FIRST, then clip in float32. Casting after the clip changes the rounding of
    //    every element.
    scratch_.resize(static_cast<size_t>(step_dim));
    for (int i = 0; i < step_dim; ++i) {
      scratch_[static_cast<size_t>(i)] = static_cast<float>(it->second[static_cast<size_t>(i)]);
    }

    // 2. Clip, NaN-preserving. See the header: this is the line that keeps the non-finite
    //    guard alive.
    double clip = 0.0;
    bool has_clip = false;
    if (term.clip.has_value()) {
      clip = *term.clip;
      has_clip = true;
    } else {
      clip = LegacyTermClip(term.name, &has_clip);
    }
    if (has_clip) {
      const float c = static_cast<float>(clip);
      for (float& v : scratch_) v = ClipPreservingNan(v, c);
    }

    // 3. History, storing the clipped but UNSCALED sample. 4. Scale after stacking, tiling a
    //    per-element scale across the window because it describes ONE sample.
    const std::vector<float>* vec = &scratch_;
    if (term.history_length > 0) vec = &PushHistory(term.name, scratch_, term.history_length);
    if (static_cast<int>(vec->size()) != term.width) {
      throw std::runtime_error("obs term '" + term.name + "': stacked width " +
                               std::to_string(vec->size()) + " != " + std::to_string(term.width));
    }
    const bool scalar_scale = term.scale.size() == 1;
    for (int i = 0; i < term.width; ++i) {
      const float s = scalar_scale
                          ? static_cast<float>(term.scale[0])
                          : static_cast<float>(term.scale[static_cast<size_t>(i % step_dim)]);
      out[idx + i] = (*vec)[static_cast<size_t>(i)] * s;
    }
    idx += term.width;
  }
  if (idx != num_obs_) {
    throw std::runtime_error("assembled " + std::to_string(idx) + " values but num_obs is " +
                             std::to_string(num_obs_));
  }
  std::copy(out, out + num_obs_, last_obs_.begin());
}

}  // namespace g1_deploy
