#ifndef KWS_ASSETS_H_
#define KWS_ASSETS_H_

#include <string>
#include <vector>

namespace kws {

struct FeatureConfig {
  int sample_rate = 0;
  int frame_length = 0;
  int hop_length = 0;
  int n_fft = 0;
  int n_mels = 0;
  int num_fft_bins = 0;
  float log_offset = 0.0f;

  int NumFrames(int num_samples) const {
    return num_samples < frame_length ? 0 : 1 + (num_samples - frame_length) / hop_length;
  }
};

// Feature-pipeline constants exported by kws/export_assets.py.
struct Assets {
  FeatureConfig config;
  std::vector<float> window;     // frame_length
  std::vector<float> mel_fbank;  // n_mels x num_fft_bins, row-major
  std::vector<float> norm_mean;  // n_mels
  std::vector<float> norm_std;   // n_mels
  std::vector<std::string> labels;

  // Loads and size-checks every file in `dir`; throws std::runtime_error on mismatch.
  static Assets Load(const std::string& dir);
};

}  // namespace kws

#endif  // KWS_ASSETS_H_
