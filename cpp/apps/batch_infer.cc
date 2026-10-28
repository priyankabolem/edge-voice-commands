// End-to-end inference over many 1 s clips stored as raw int16 (N x sample_rate).
// Writes N x num_classes float32 logits. Used for full-test-set parity checks,
// including synthetic _silence_ clips that have no WAV file.
//
// Usage: batch_infer <model.pte> <assets_dir> <clips.i16> <logits.f32>

#include <cstdio>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "kws/assets.h"
#include "kws/kws_model.h"
#include "kws/log_mel.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <model.pte> <assets_dir> <clips.i16> <logits.f32>\n", argv[0]);
    return 2;
  }
  try {
    const kws::Assets assets = kws::Assets::Load(argv[2]);
    kws::LogMelExtractor extractor(assets);
    kws::KwsModel model(argv[1]);
    const int clip_samples = assets.config.sample_rate;

    std::ifstream in(argv[3], std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error(std::string(argv[3]) + ": cannot open");
    const auto bytes = static_cast<size_t>(in.tellg());
    const size_t clip_bytes = clip_samples * sizeof(int16_t);
    if (bytes % clip_bytes != 0)
      throw std::runtime_error("input size is not a whole number of clips");
    in.seekg(0);

    std::ofstream out(argv[4], std::ios::binary);
    if (!out) throw std::runtime_error(std::string(argv[4]) + ": cannot open for writing");

    std::vector<int16_t> clip(clip_samples);
    const size_t num_clips = bytes / clip_bytes;
    for (size_t i = 0; i < num_clips; ++i) {
      in.read(reinterpret_cast<char*>(clip.data()), static_cast<std::streamsize>(clip_bytes));
      extractor.ComputeClip(clip.data(), clip_samples, model.input());
      const float* logits = model.Run();
      if (logits == nullptr)
        throw std::runtime_error("inference failed on clip " + std::to_string(i));
      out.write(reinterpret_cast<const char*>(logits),
                static_cast<std::streamsize>(model.num_classes() * sizeof(float)));
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
