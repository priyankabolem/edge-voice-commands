// Computes normalized log-mel features for WAV files and writes them as raw float32.
// Each clip is padded/trimmed to 1 s, so the output is N x frames x mels, row-major.
//
// Usage: dump_features <assets_dir> <out.f32> <clip.wav>...

#include <cstdio>
#include <exception>
#include <fstream>
#include <vector>

#include "kws/assets.h"
#include "kws/log_mel.h"
#include "kws/wav_reader.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <assets_dir> <out.f32> <clip.wav>...\n", argv[0]);
    return 2;
  }
  try {
    const kws::Assets assets = kws::Assets::Load(argv[1]);
    kws::LogMelExtractor extractor(assets);
    const int clip_samples = assets.config.sample_rate;
    std::vector<float> features(static_cast<size_t>(extractor.config().NumFrames(clip_samples)) *
                                assets.config.n_mels);

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) throw std::runtime_error(std::string(argv[2]) + ": cannot open for writing");
    for (int i = 3; i < argc; ++i) {
      std::vector<int16_t> audio = kws::ReadWavPcm16Mono(argv[i], assets.config.sample_rate);
      kws::PadOrTrim(&audio, clip_samples);
      extractor.ComputeClip(audio.data(), clip_samples, features.data());
      out.write(reinterpret_cast<const char*>(features.data()),
                static_cast<std::streamsize>(features.size() * sizeof(float)));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
