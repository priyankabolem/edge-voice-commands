// Classifies a 1 s, 16 kHz, 16-bit mono WAV clip and prints the label and score.
//
// Usage: classify_wav <model.pte> <assets_dir> <clip.wav>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

#include "kws/assets.h"
#include "kws/kws_model.h"
#include "kws/log_mel.h"
#include "kws/wav_reader.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <model.pte> <assets_dir> <clip.wav>\n", argv[0]);
    return 2;
  }
  try {
    const kws::Assets assets = kws::Assets::Load(argv[2]);
    kws::LogMelExtractor extractor(assets);
    kws::KwsModel model(argv[1]);

    const int clip_samples = assets.config.sample_rate;
    std::vector<int16_t> audio = kws::ReadWavPcm16Mono(argv[3], assets.config.sample_rate);
    kws::PadOrTrim(&audio, clip_samples);
    if (extractor.config().NumFrames(clip_samples) * assets.config.n_mels != model.input_size()) {
      std::fprintf(stderr, "feature size does not match model input\n");
      return 1;
    }
    extractor.ComputeClip(audio.data(), clip_samples, model.input());

    const float* logits = model.Run();
    if (logits == nullptr) {
      std::fprintf(stderr, "inference failed\n");
      return 1;
    }
    const int n = model.num_classes();
    const int best = static_cast<int>(std::max_element(logits, logits + n) - logits);
    double denom = 0.0;
    for (int i = 0; i < n; ++i) denom += std::exp(logits[i] - logits[best]);
    std::printf("%s %.4f\n", assets.labels[best].c_str(), 1.0 / denom);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
