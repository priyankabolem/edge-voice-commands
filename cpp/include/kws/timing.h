#ifndef KWS_TIMING_H_
#define KWS_TIMING_H_

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

namespace kws {

// Latency summary in microseconds.
struct Stats {
  double min, mean, median, p95, p99, max;
};

// Nearest-rank percentile on sorted data.
inline double Percentile(const std::vector<double>& sorted, double p) {
  const size_t rank = static_cast<size_t>(p / 100.0 * (sorted.size() - 1) + 0.5);
  return sorted[std::min(rank, sorted.size() - 1)];
}

// Times `fn` once per iteration with steady_clock after `warmup` untimed calls.
// `us` is resized to `iters` before timing starts and is left sorted.
inline Stats Measure(const std::function<void()>& fn, int warmup, int iters,
                     std::vector<double>* us) {
  using Clock = std::chrono::steady_clock;
  us->resize(iters);
  for (int i = 0; i < warmup; ++i) fn();
  for (int i = 0; i < iters; ++i) {
    const auto start = Clock::now();
    fn();
    const auto end = Clock::now();
    (*us)[i] = std::chrono::duration<double, std::micro>(end - start).count();
  }
  std::sort(us->begin(), us->end());
  double sum = 0.0;
  for (int i = 0; i < iters; ++i) sum += (*us)[i];
  return {
      (*us)[0],        sum / iters, Percentile(*us, 50), Percentile(*us, 95), Percentile(*us, 99),
      (*us)[iters - 1]};
}

// Prints `"name": {...}` as one JSON object member.
inline void PrintStats(const char* name, const Stats& s, bool last) {
  std::printf(
      "    \"%s\": {\"min\": %.3f, \"mean\": %.3f, \"median\": %.3f, \"p95\": %.3f, "
      "\"p99\": %.3f, \"max\": %.3f}%s\n",
      name, s.min, s.mean, s.median, s.p95, s.p99, s.max, last ? "" : ",");
}

}  // namespace kws

#endif  // KWS_TIMING_H_
