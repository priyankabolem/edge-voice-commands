// Real-time streaming simulation on a long WAV file.
//
// Audio arrives in hop-sized chunks (160 samples = 10 ms), as from a microphone.
// Each chunk completes exactly one new 25 ms frame, so only that frame's features are
// computed and pushed into a ring buffer holding the last 98 frames (~1 s). Every
// `infer_every` hops the model runs on the ring contents and a smoothed-posterior
// detector decides whether a keyword fired. Each hop's processing time is measured
// against the 10 ms budget. Prints detections and timing as JSON.
//
// Usage: stream <model.pte> <assets_dir> <audio.wav> [--threads N] [--infer-every N]
//        [--smooth N] [--threshold X] [--refractory-s X] [--dump-window HOP PATH]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kws/assets.h"
#include "kws/kws_model.h"
#include "kws/log_mel.h"
#include "kws/posterior.h"
#include "kws/ring_buffer.h"
#include "kws/threads.h"
#include "kws/wav_reader.h"

#ifdef KWS_HAVE_ALLOC_COUNTER
#include "alloc_counter.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  int threads = 1;
  int infer_every = 2;
  int smooth = 5;
  float threshold = 0.7f;
  float refractory_s = 1.0f;
  long dump_hop = -1;
  std::string dump_path;
};

Options ParseOptions(int argc, char** argv) {
  Options o;
  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> const char* {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--threads") {
      o.threads = std::atoi(next());
    } else if (arg == "--infer-every") {
      o.infer_every = std::atoi(next());
    } else if (arg == "--smooth") {
      o.smooth = std::atoi(next());
    } else if (arg == "--threshold") {
      o.threshold = std::strtof(next(), nullptr);
    } else if (arg == "--refractory-s") {
      o.refractory_s = std::strtof(next(), nullptr);
    } else if (arg == "--dump-window") {
      o.dump_hop = std::atol(next());
      o.dump_path = next();
    } else {
      throw std::invalid_argument("unknown option " + arg);
    }
  }
  return o;
}

double Percentile(const std::vector<double>& sorted, double p) {
  const size_t rank = static_cast<size_t>(p / 100.0 * (sorted.size() - 1) + 0.5);
  return sorted[std::min(rank, sorted.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <model.pte> <assets_dir> <audio.wav> [options]\n", argv[0]);
    return 2;
  }
  try {
    const Options opt = ParseOptions(argc, argv);
    if (!kws::SetRuntimeThreads(static_cast<uint32_t>(opt.threads))) {
      throw std::runtime_error("failed to set threads");
    }
    const kws::Assets assets = kws::Assets::Load(argv[2]);
    const kws::FeatureConfig& cfg = assets.config;
    kws::LogMelExtractor extractor(assets);
    kws::KwsModel model(argv[1]);
    const std::vector<int16_t> audio = kws::ReadWavPcm16Mono(argv[3], cfg.sample_rate);

    const int hop = cfg.hop_length;
    const int window_frames = model.input_size() / cfg.n_mels;
    // Frame f covers samples [f * hop, f * hop + frame_length). Keeping the last
    // ceil(frame_length / hop) hops of audio means the newest complete frame always
    // starts at the beginning of this history.
    const int history_hops = (cfg.frame_length + hop - 1) / hop;
    std::vector<int16_t> history(static_cast<size_t>(history_hops) * hop, 0);
    std::vector<float> frame_features(cfg.n_mels);
    kws::FeatureRingBuffer ring(window_frames, cfg.n_mels);

    std::vector<bool> is_keyword;
    for (const auto& label : assets.labels) is_keyword.push_back(label.empty() || label[0] != '_');
    const double hop_s = static_cast<double>(hop) / cfg.sample_rate;
    const int refractory_steps =
        static_cast<int>(opt.refractory_s / (hop_s * opt.infer_every) + 0.5);
    kws::PosteriorDetector detector(is_keyword, opt.smooth, opt.threshold, refractory_steps);

    const long num_hops = static_cast<long>(audio.size()) / hop;
    std::vector<double> hop_us(num_hops, 0.0);
    struct Event {
      double time_s;
      int label;
      float score;
    };
    // Preallocated for the most detections the refractory period allows.
    std::vector<Event> detections;
    detections.reserve(static_cast<size_t>(num_hops / std::max(1, refractory_steps)) + 2);
    long inferences = 0;

    // Warm-up inference at startup: the first run does one-time lazy setup inside the
    // runtime (it allocates and is slower), which should not land in the first hop.
    const auto warmup_start = Clock::now();
    if (model.Run() == nullptr) throw std::runtime_error("warm-up inference failed");
    const double warmup_us =
        std::chrono::duration<double, std::micro>(Clock::now() - warmup_start).count();

#ifdef KWS_HAVE_ALLOC_COUNTER
    AllocCounterStart();
#endif
    const auto stream_start = Clock::now();
    for (long h = 0; h < num_hops; ++h) {
      const int16_t* chunk = &audio[static_cast<size_t>(h) * hop];
      const auto t0 = Clock::now();

      // Shift the audio history left by one hop and append the new chunk.
      std::copy(history.begin() + hop, history.end(), history.begin());
      std::copy(chunk, chunk + hop, history.end() - hop);

      if (h + 1 >= history_hops) {
        extractor.ComputeFrame(history.data(), frame_features.data());
        ring.Push(frame_features.data());
      }
      if (ring.full() && h % opt.infer_every == 0) {
        ring.CopyOrdered(model.input());
        const float* logits = model.Run();
        if (logits == nullptr) throw std::runtime_error("inference failed");
        ++inferences;
        const auto det = detector.Update(logits);
        if (det.label >= 0) detections.push_back({(h + 1) * hop_s, det.label, det.score});
      }
      hop_us[h] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();

      if (h == opt.dump_hop) {
        std::vector<float> window(static_cast<size_t>(model.input_size()));
        ring.CopyOrdered(window.data());
        std::ofstream(opt.dump_path, std::ios::binary)
            .write(reinterpret_cast<const char*>(window.data()),
                   static_cast<std::streamsize>(window.size() * sizeof(float)));
      }
    }
    const double total_us =
        std::chrono::duration<double, std::micro>(Clock::now() - stream_start).count();
    long hot_loop_allocations = -1;  // -1: not measured on this platform.
#ifdef KWS_HAVE_ALLOC_COUNTER
    hot_loop_allocations = AllocCounterStop();
    if (opt.dump_hop >= 0) hot_loop_allocations = -1;  // The debug dump allocates.
#endif

    const double budget_us = hop_s * 1e6;
    const long overruns =
        std::count_if(hop_us.begin(), hop_us.end(), [&](double us) { return us > budget_us; });
    double processing_us = 0.0;
    for (double us : hop_us) processing_us += us;
    std::vector<double> sorted = hop_us;
    std::sort(sorted.begin(), sorted.end());
    const double audio_s = static_cast<double>(num_hops) * hop_s;

    std::printf("{\n  \"audio_seconds\": %.3f,\n  \"hops\": %ld,\n  \"inferences\": %ld,\n",
                audio_s, num_hops, inferences);
    std::printf(
        "  \"config\": {\"threads\": %d, \"infer_every_hops\": %d, \"smoothing_window\": %d, "
        "\"threshold\": %.4f, \"refractory_s\": %.3f, \"refractory_steps\": %d},\n",
        opt.threads, opt.infer_every, opt.smooth, opt.threshold, opt.refractory_s,
        refractory_steps);
    std::printf(
        "  \"hop_latency_us\": {\"min\": %.3f, \"mean\": %.3f, \"median\": %.3f, \"p95\": %.3f, "
        "\"p99\": %.3f, \"max\": %.3f},\n",
        sorted.front(), processing_us / num_hops, Percentile(sorted, 50), Percentile(sorted, 95),
        Percentile(sorted, 99), sorted.back());
    std::printf("  \"warmup_inference_us\": %.3f,\n", warmup_us);
    std::printf("  \"hop_budget_us\": %.1f,\n  \"budget_overruns\": %ld,\n", budget_us, overruns);
    std::printf("  \"real_time_factor\": %.6f,\n", processing_us / (audio_s * 1e6));
    std::printf("  \"wall_time_real_time_factor\": %.6f,\n", total_us / (audio_s * 1e6));
    std::printf("  \"hot_loop_allocations\": %ld,\n  \"detections\": [", hot_loop_allocations);
    for (size_t i = 0; i < detections.size(); ++i) {
      std::printf("%s\n    {\"time_s\": %.3f, \"label\": \"%s\", \"score\": %.4f}", i ? "," : "",
                  detections[i].time_s, assets.labels[detections[i].label].c_str(),
                  detections[i].score);
    }
    std::printf("\n  ]\n}\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
