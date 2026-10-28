// Verifies that the per-frame, per-clip and per-inference paths do not touch the heap.
//
// Counts every malloc/calloc/realloc while the hot loop runs (see tools/alloc_counter).
// Calls from the ExecuTorch and XNNPACK libraries count too, as does C++ operator new
// (libc++ implements it with malloc). Prints a JSON summary; exits non-zero if anything
// was allocated.
//
// Usage: alloc_check <model.pte> <assets_dir> [iterations]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "alloc_counter.h"
#include "kws/assets.h"
#include "kws/kws_model.h"
#include "kws/log_mel.h"

namespace {

long CountAllocations(void (*fn)(void*), void* arg, int iterations) {
  AllocCounterStart();
  for (int i = 0; i < iterations; ++i) fn(arg);
  return AllocCounterStop();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.pte> <assets_dir> [iterations]\n", argv[0]);
    return 2;
  }
  const int iterations = argc > 3 ? std::atoi(argv[3]) : 1000;
  try {
    const kws::Assets assets = kws::Assets::Load(argv[2]);
    kws::LogMelExtractor extractor(assets);
    kws::KwsModel model(argv[1]);
    const int clip_samples = assets.config.sample_rate;
    std::vector<int16_t> clip(clip_samples);
    for (int i = 0; i < clip_samples; ++i)
      clip[i] = static_cast<int16_t>((i * 7919) % 20000 - 10000);
    std::vector<float> frame_out(assets.config.n_mels);

    struct Context {
      kws::LogMelExtractor* extractor;
      kws::KwsModel* model;
      std::vector<int16_t>* clip;
      std::vector<float>* frame_out;
    } ctx{&extractor, &model, &clip, &frame_out};

    // Self-test: strdup allocates inside libSystem, so it must be counted if
    // interposing covers other libraries. The volatile store keeps the optimizer
    // from removing the allocation.
    static void* volatile sink = nullptr;
    const long self_test = CountAllocations([](void*) { sink = strdup("self-test"); }, nullptr, 1);
    std::free(sink);
    if (self_test < 1) {
      std::fprintf(stderr, "malloc interposing is not active; cannot verify\n");
      return 1;
    }

    // Warm up once outside the counted region (first-run lazy initialization is allowed).
    extractor.ComputeClip(clip.data(), clip_samples, model.input());
    model.Run();

    const long frame = CountAllocations(
        [](void* p) {
          auto* c = static_cast<Context*>(p);
          c->extractor->ComputeFrame(c->clip->data(), c->frame_out->data());
        },
        &ctx, iterations);
    const long clip_features = CountAllocations(
        [](void* p) {
          auto* c = static_cast<Context*>(p);
          c->extractor->ComputeClip(c->clip->data(), static_cast<int>(c->clip->size()),
                                    c->model->input());
        },
        &ctx, iterations);
    const long inference =
        CountAllocations([](void* p) { static_cast<Context*>(p)->model->Run(); }, &ctx, iterations);

    std::printf(
        "{\"iterations\": %d, \"allocations\": {\"feature_frame\": %ld, \"feature_clip\": %ld, "
        "\"inference\": %ld}}\n",
        iterations, frame, clip_features, inference);
    return (frame == 0 && clip_features == 0 && inference == 0) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
