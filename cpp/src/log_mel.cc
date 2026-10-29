#include "kws/log_mel.h"

#include <cmath>
#include <stdexcept>

namespace kws {

LogMelExtractor::LogMelExtractor(const Assets& assets, bool use_simd)
    : config_(assets.config),
      fft_(assets.config.n_fft, use_simd),
      window_(assets.window),
      inv_std_(assets.config.n_mels),
      mean_(assets.norm_mean),
      first_bin_(assets.config.n_mels),
      num_bins_(assets.config.n_mels),
      weight_offset_(assets.config.n_mels),
      fft_input_(assets.config.n_fft, 0.0f),
      power_(assets.config.num_fft_bins) {
  const int bins = config_.num_fft_bins;
  for (int m = 0; m < config_.n_mels; ++m) {
    inv_std_[m] = 1.0f / assets.norm_std[m];
    const float* row = &assets.mel_fbank[static_cast<size_t>(m) * bins];
    int first = 0;
    while (first < bins && row[first] == 0.0f) ++first;
    int last = bins - 1;
    while (last >= first && row[last] == 0.0f) --last;
    if (first > last) throw std::runtime_error("mel filter " + std::to_string(m) + " is empty");
    first_bin_[m] = first;
    num_bins_[m] = last - first + 1;
    weight_offset_[m] = static_cast<int>(weights_.size());
    weights_.insert(weights_.end(), row + first, row + last + 1);
  }
}

void LogMelExtractor::ComputeFrame(const int16_t* frame, float* out) {
  constexpr float kInt16Scale = 1.0f / 32768.0f;
  for (int i = 0; i < config_.frame_length; ++i) {
    fft_input_[i] = static_cast<float>(frame[i]) * kInt16Scale * window_[i];
  }
  // fft_input_[frame_length..n_fft) stays zero from construction.
  fft_.PowerSpectrum(fft_input_.data(), power_.data());

  for (int m = 0; m < config_.n_mels; ++m) {
    const float* w = &weights_[weight_offset_[m]];
    const float* p = &power_[first_bin_[m]];
    float mel = 0.0f;
    for (int k = 0; k < num_bins_[m]; ++k) mel += w[k] * p[k];
    out[m] = (std::log(mel + config_.log_offset) - mean_[m]) * inv_std_[m];
  }
}

void LogMelExtractor::ComputeClip(const int16_t* samples, int num_samples, float* out) {
  const int frames = config_.NumFrames(num_samples);
  for (int f = 0; f < frames; ++f) {
    ComputeFrame(samples + f * config_.hop_length, out + f * config_.n_mels);
  }
}

}  // namespace kws
