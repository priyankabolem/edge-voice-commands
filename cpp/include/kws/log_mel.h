#ifndef KWS_LOG_MEL_H_
#define KWS_LOG_MEL_H_

#include <cstdint>
#include <vector>

#include "kws/assets.h"
#include "kws/fft.h"

namespace kws {

// C++ port of kws/features.py:
//   int16 -> float / 32768 -> Hann window -> zero-pad to n_fft -> |rfft|^2
//   -> mel filterbank -> log(mel + offset) -> (x - mean) / std
// All buffers are allocated in the constructor; ComputeFrame/ComputeClip do not allocate.
class LogMelExtractor {
 public:
  explicit LogMelExtractor(const Assets& assets);

  const FeatureConfig& config() const { return config_; }

  // frame: frame_length samples. out: n_mels normalized log-mel values.
  void ComputeFrame(const int16_t* frame, float* out);

  // samples: num_samples values. out: NumFrames(num_samples) x n_mels, row-major.
  void ComputeClip(const int16_t* samples, int num_samples, float* out);

 private:
  FeatureConfig config_;
  RealFft fft_;
  std::vector<float> window_;
  std::vector<float> inv_std_;
  std::vector<float> mean_;
  // The filterbank is stored sparsely: filter m covers bins [first_bin_[m], first_bin_[m] +
  // weights length) with weights starting at weight_offset_[m] in weights_.
  std::vector<int> first_bin_;
  std::vector<int> num_bins_;
  std::vector<int> weight_offset_;
  std::vector<float> weights_;
  std::vector<float> fft_input_;
  std::vector<float> power_;
};

}  // namespace kws

#endif  // KWS_LOG_MEL_H_
