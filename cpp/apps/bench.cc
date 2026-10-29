// Latency benchmark for the C++ pipeline. Times each stage separately with
// std::chrono::steady_clock, one measurement per iteration, after a warm-up:
//   feature_frame  one 25 ms frame -> 40 features (the per-hop streaming cost)
//   feature_clip   1 s clip -> 98 x 40 features
//   inference      one model call on a prepared input
//   clip_end_to_end  feature_clip + inference
// Prints min / mean / median / p95 / p99 / max in microseconds as JSON.
//
// Usage: bench <model.pte> <assets_dir> [--threads N] [--warmup N] [--iters N]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "kws/assets.h"
#include "kws/kws_model.h"
#include "kws/log_mel.h"
#include "kws/threads.h"
#include "kws/timing.h"

using kws::Measure;
using kws::PrintStats;
using kws::Stats;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model.pte> <assets_dir> [--threads N] [--warmup N] [--iters N]\n",
                 argv[0]);
    return 2;
  }
  int threads = 0;  // 0 keeps the runtime default.
  int warmup = 50;
  int iters = 1000;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--threads") == 0) {
      threads = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      warmup = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      iters = std::atoi(argv[i + 1]);
    } else {
      std::fprintf(stderr, "unknown option %s\n", argv[i]);
      return 2;
    }
  }

  try {
    const size_t default_threads = kws::RuntimeThreads();
    if (threads > 0 && !kws::SetRuntimeThreads(static_cast<uint32_t>(threads))) {
      std::fprintf(stderr, "failed to set %d threads\n", threads);
      return 1;
    }
    const kws::Assets assets = kws::Assets::Load(argv[2]);
    kws::LogMelExtractor extractor(assets);
    kws::KwsModel model(argv[1]);

    // Deterministic pseudo-noise clip; feature and model cost do not depend on content.
    const int clip_samples = assets.config.sample_rate;
    std::vector<int16_t> clip(clip_samples);
    uint32_t state = 12345;
    for (auto& s : clip) {
      state = state * 1664525u + 1013904223u;
      s = static_cast<int16_t>((state >> 16) - 32768);
    }
    std::vector<float> frame_out(assets.config.n_mels);
    std::vector<double> us(iters);
    const int num_frames = extractor.config().NumFrames(clip_samples);
    int frame = 0;

    const Stats feature_frame = Measure(
        [&] {
          extractor.ComputeFrame(clip.data() + frame * assets.config.hop_length, frame_out.data());
          frame = (frame + 1) % num_frames;
        },
        warmup, iters, &us);
    const Stats feature_clip =
        Measure([&] { extractor.ComputeClip(clip.data(), clip_samples, model.input()); }, warmup,
                iters, &us);
    const Stats inference = Measure([&] { model.Run(); }, warmup, iters, &us);
    const Stats end_to_end = Measure(
        [&] {
          extractor.ComputeClip(clip.data(), clip_samples, model.input());
          model.Run();
        },
        warmup, iters, &us);

    std::printf("{\n  \"threads_requested\": %d,\n  \"threads_default\": %zu,\n", threads,
                default_threads);
    std::printf("  \"threads_active\": %zu,\n  \"warmup\": %d,\n  \"iters\": %d,\n",
                kws::RuntimeThreads(), warmup, iters);
    std::printf("  \"latency_us\": {\n");
    PrintStats("feature_frame", feature_frame, false);
    PrintStats("feature_clip", feature_clip, false);
    PrintStats("inference", inference, false);
    PrintStats("clip_end_to_end", end_to_end, true);
    std::printf("  }\n}\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
