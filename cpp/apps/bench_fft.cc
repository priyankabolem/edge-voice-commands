// Scalar vs NEON FFT: bit-exactness check and latency of the FFT, one feature frame
// and one 1 s clip of features, each measured with both implementations.
//
// Usage: bench_fft <assets_dir> [--warmup N] [--iters N] [--check-frames N]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "kws/assets.h"
#include "kws/fft.h"
#include "kws/log_mel.h"
#include "kws/timing.h"

namespace {

uint32_t g_state = 12345;

int16_t NextSample() {
  g_state = g_state * 1664525u + 1013904223u;
  return static_cast<int16_t>((g_state >> 16) - 32768);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <assets_dir> [--warmup N] [--iters N] [--check-frames N]\n",
                 argv[0]);
    return 2;
  }
  int warmup = 1000;
  int iters = 10000;
  int check_frames = 10000;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--warmup") == 0) {
      warmup = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      iters = std::atoi(argv[i + 1]);
    } else if (std::strcmp(argv[i], "--check-frames") == 0) {
      check_frames = std::atoi(argv[i + 1]);
    }
  }
  try {
    const kws::Assets assets = kws::Assets::Load(argv[1]);
    const kws::FeatureConfig& cfg = assets.config;
    if (!kws::RealFft::SimdAvailable()) {
      std::fprintf(stderr, "this build has no NEON path\n");
      return 1;
    }
    kws::RealFft fft_scalar(cfg.n_fft, false), fft_neon(cfg.n_fft, true);
    kws::LogMelExtractor mel_scalar(assets, false), mel_neon(assets, true);

    // Bit-exactness on random frames: power spectra and normalized features.
    std::vector<int16_t> clip(cfg.sample_rate);
    std::vector<float> in(cfg.n_fft, 0.0f), p_scalar(fft_scalar.num_bins()),
        p_neon(fft_neon.num_bins()), f_scalar(cfg.n_mels), f_neon(cfg.n_mels);
    long mismatched_power = 0, mismatched_features = 0;
    for (int f = 0; f < check_frames; ++f) {
      std::vector<int16_t> frame(cfg.frame_length);
      const int gain = 1 + f % 64;  // Cover quiet to loud frames.
      for (auto& s : frame) s = static_cast<int16_t>(NextSample() / gain);
      for (int i = 0; i < cfg.frame_length; ++i) in[i] = frame[i] / 32768.0f;
      fft_scalar.PowerSpectrum(in.data(), p_scalar.data());
      fft_neon.PowerSpectrum(in.data(), p_neon.data());
      mismatched_power += std::memcmp(p_scalar.data(), p_neon.data(), p_neon.size() * 4) != 0;
      mel_scalar.ComputeFrame(frame.data(), f_scalar.data());
      mel_neon.ComputeFrame(frame.data(), f_neon.data());
      mismatched_features += std::memcmp(f_scalar.data(), f_neon.data(), f_neon.size() * 4) != 0;
    }

    for (auto& s : clip) s = NextSample();
    std::vector<float> clip_out(static_cast<size_t>(cfg.NumFrames(cfg.sample_rate)) * cfg.n_mels);
    std::vector<double> us(iters);
    int frame = 0;
    const int num_frames = cfg.NumFrames(cfg.sample_rate);
    const auto next_frame = [&] {
      const int16_t* p = clip.data() + frame * cfg.hop_length;
      frame = (frame + 1) % num_frames;
      return p;
    };
    for (int i = 0; i < cfg.frame_length; ++i) in[i] = clip[i] / 32768.0f;

    const kws::Stats fft_s = kws::Measure(
        [&] { fft_scalar.PowerSpectrum(in.data(), p_scalar.data()); }, warmup, iters, &us);
    const kws::Stats fft_n =
        kws::Measure([&] { fft_neon.PowerSpectrum(in.data(), p_neon.data()); }, warmup, iters, &us);
    const kws::Stats frame_s = kws::Measure(
        [&] { mel_scalar.ComputeFrame(next_frame(), f_scalar.data()); }, warmup, iters, &us);
    const kws::Stats frame_n = kws::Measure(
        [&] { mel_neon.ComputeFrame(next_frame(), f_neon.data()); }, warmup, iters, &us);
    const int clip_iters = iters / 10;
    const kws::Stats clip_s =
        kws::Measure([&] { mel_scalar.ComputeClip(clip.data(), cfg.sample_rate, clip_out.data()); },
                     warmup / 10, clip_iters, &us);
    const kws::Stats clip_n =
        kws::Measure([&] { mel_neon.ComputeClip(clip.data(), cfg.sample_rate, clip_out.data()); },
                     warmup / 10, clip_iters, &us);

    std::printf("{\n  \"check_frames\": %d,\n  \"frames_with_power_mismatch\": %ld,\n",
                check_frames, mismatched_power);
    std::printf("  \"frames_with_feature_mismatch\": %ld,\n", mismatched_features);
    std::printf("  \"iters\": %d,\n  \"clip_iters\": %d,\n  \"latency_us\": {\n", iters,
                clip_iters);
    kws::PrintStats("fft_power_scalar", fft_s, false);
    kws::PrintStats("fft_power_neon", fft_n, false);
    kws::PrintStats("feature_frame_scalar", frame_s, false);
    kws::PrintStats("feature_frame_neon", frame_n, false);
    kws::PrintStats("feature_clip_scalar", clip_s, false);
    kws::PrintStats("feature_clip_neon", clip_n, true);
    std::printf("  }\n}\n");
    return (mismatched_power == 0 && mismatched_features == 0) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
