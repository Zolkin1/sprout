// Copyright 2026 Zachary Olkin. All rights reserved.

#include "g1_deploy/tracker_session.h"

#include <stdexcept>

#include "g1_deploy/window.h"

namespace g1_deploy {

TrackerSession::TrackerSession(const std::string& onnx_path,
                              const std::vector<Provider>& providers, int intra_threads)
    : model_(onnx_path, providers, intra_threads),
      params_(ParsePolicyParameters(model_.Metadata("policy_parameters"))) {
  // Read from the graph rather than hardcoded. The tracker takes one observation vector and
  // returns one action vector, so position is unambiguous, and taking the names from the
  // artifact means an export that renames them keeps working.
  if (model_.input_names().size() != 1 || model_.output_names().size() != 1) {
    throw std::runtime_error("tracker.onnx should have exactly one input and one output, got " +
                             std::to_string(model_.input_names().size()) + " and " +
                             std::to_string(model_.output_names().size()));
  }
  in_name_ = model_.input_names()[0];
  out_name_ = model_.output_names()[0];
  const auto in_shape = model_.InputShape(model_.input_names()[0]);
  int64_t flat = 1;
  for (int64_t d : in_shape) flat *= d;
  if (flat != params_.num_obs) {
    throw std::runtime_error("tracker.onnx takes " + std::to_string(flat) +
                             " observation values but its metadata declares num_obs=" +
                             std::to_string(params_.num_obs));
  }
  obs_buf_.assign(static_cast<size_t>(params_.num_obs), 0.0f);
  last_action_.assign(static_cast<size_t>(params_.num_actions), 0.0f);
}

void TrackerSession::Infer(const float* obs, float* action_out) {
  std::copy(obs, obs + params_.num_obs, obs_buf_.begin());
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t shape[2] = {1, params_.num_obs};
  std::vector<Ort::Value> inputs;
  inputs.push_back(
      Ort::Value::CreateTensor<float>(mem, obs_buf_.data(), obs_buf_.size(), shape, 2));

  const std::vector<const char*> in_names = {in_name_.c_str()};
  const std::vector<const char*> out_names = {out_name_.c_str()};
  std::vector<Ort::Value> outputs;
  model_.Run(in_names, inputs, out_names, &outputs);

  const float* action = outputs[0].GetTensorData<float>();
  std::copy(action, action + params_.num_actions, last_action_.begin());
  std::copy(last_action_.begin(), last_action_.end(), action_out);
}

void TrackerSession::GetAction(const float* obs, const std::vector<std::string>& joint_names_out,
                              double* targets_out) {
  if (params_.action_scale.size() != static_cast<size_t>(params_.num_actions) ||
      params_.default_joint_angles.size() != static_cast<size_t>(params_.num_actions)) {
    throw std::runtime_error(
        "tracker.onnx metadata carries no per-joint action_scale / default_joint_angles, so a "
        "raw network output cannot be turned into a joint target. Re-export with "
        "scripts/rsl_rl/export_parameters.py.");
  }
  Infer(obs, last_action_.data());

  targets_isaac_.resize(static_cast<size_t>(params_.num_actions));
  for (int i = 0; i < params_.num_actions; ++i) {
    targets_isaac_[static_cast<size_t>(i)] =
        static_cast<double>(last_action_[static_cast<size_t>(i)]) *
            params_.action_scale[static_cast<size_t>(i)] +
        params_.default_joint_angles[static_cast<size_t>(i)];
  }

  // Resolve the permutation once. Doing it per tick is a dict build per joint at 50 Hz, and
  // more importantly it hides the failure: an unknown joint should be a loud error at the
  // first tick, not a lookup that happens to miss.
  if (joint_names_out != cached_out_names_) {
    out_permutation_.clear();
    out_permutation_.reserve(joint_names_out.size());
    for (const std::string& want : joint_names_out) {
      int found = -1;
      for (size_t i = 0; i < params_.joint_names_isaac.size(); ++i) {
        if (params_.joint_names_isaac[i] == want) {
          found = static_cast<int>(i);
          break;
        }
      }
      if (found < 0) {
        throw std::runtime_error("joint '" + want +
                                 "' is not one the policy produces; the tracker and the robot "
                                 "disagree about the joint set");
      }
      out_permutation_.push_back(found);
    }
    cached_out_names_ = joint_names_out;
  }
  for (size_t i = 0; i < out_permutation_.size(); ++i) {
    targets_out[i] = targets_isaac_[static_cast<size_t>(out_permutation_[i])];
  }
}

void AssertWindowBlocksAgree(const std::vector<std::pair<std::string, int>>& tracker_blocks,
                             const std::vector<ReferenceBlock>& fm_blocks) {
  bool same = tracker_blocks.size() == fm_blocks.size();
  if (same) {
    for (size_t i = 0; i < fm_blocks.size(); ++i) {
      if (BlockName(fm_blocks[i]) != tracker_blocks[i].first) {
        same = false;
        break;
      }
    }
  }
  if (same) return;

  std::string fm_names;
  for (size_t i = 0; i < fm_blocks.size(); ++i) {
    fm_names += (i ? ", " : "") + BlockName(fm_blocks[i]);
  }
  std::string trk_names;
  for (size_t i = 0; i < tracker_blocks.size(); ++i) {
    trk_names += (i ? ", " : "") + tracker_blocks[i].first;
  }
  throw std::runtime_error(
      "fm.onnx predicts window blocks [" + fm_names + "] but tracker.onnx observes [" +
      trk_names +
      "]. The generator's output IS the tracker's reference_window observation -- there is no "
      "translation layer between them -- so the two have to be the same vector. Retrain the "
      "generator against this teacher (train.py resolves the blocks from the recording, so "
      "re-record with this teacher), or pair the tracker with a generator trained for it.");
}

}  // namespace g1_deploy
